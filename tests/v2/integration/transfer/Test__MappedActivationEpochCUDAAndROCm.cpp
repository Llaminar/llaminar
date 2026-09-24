/**
 * @file Test__MappedActivationEpochCUDAAndROCm.cpp
 * @brief Real-device proof for node-local CUDA/ROCm activation epochs.
 *
 * The test maps one POSIX shared allocation into real CUDA and ROCm contexts,
 * then queues alternating continuation/follower round trips without a host
 * bridge between stages. Every generation exercises both payload banks and
 * reverses backend roles on the next generation. A second fixture captures a
 * realistic 4 MiB prefill payload round trip through each backend's DMA engine
 * while keeping only timeline words mapped into device address space. Concurrent
 * side-stream pools reproduce hot-path queue pressure such as FusedQKV while
 * terminal events and PerfStats prove that the exact-stream protocol completed
 * without a per-stage host wait.
 */

#include <gtest/gtest.h>

#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "execution/moe/MoEOverlayActivationEpochABI.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "tensors/Tensors.h"
#include "transfer/TransferEngine.h"
#include "utils/PerfStatsCollector.h"

#include <sys/mman.h>

#include <array>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using namespace llaminar2;

namespace
{
    /** @brief Poll one real backend event without synchronizing its device. */
    bool awaitEvent(
        IBackend *backend,
        int ordinal,
        void *event,
        std::chrono::steady_clock::duration timeout)
    {
        if (!backend || !event)
            return false;
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        while (std::chrono::steady_clock::now() < deadline)
        {
            bool ready = false;
            if (!backend->queryEvent(event, ordinal, &ready))
                return false;
            if (ready)
                return true;
            std::this_thread::yield();
        }
        return false;
    }

    /** @brief One independently queued stream used to perturb GPU queue mapping. */
    struct SideStream
    {
        IBackend *backend = nullptr;
        int ordinal = -1;
        void *stream = nullptr;
        void *payload = nullptr;
        void *event = nullptr;
    };

    /**
     * @brief Own the complete shared-page and real-device lifecycle for a lane.
     *
     * Failure cleanup publishes the ABI abort sentinel before recording drain
     * events. This releases any already-submitted GPU wait without weakening
     * the successful path, where the host observes only terminal events after
     * every epoch has already been submitted.
     */
    class RealMappedActivationLane final
    {
    public:
        RealMappedActivationLane() = default;

        /** @brief Release blocked waits, drain streams, unregister, then unmap. */
        ~RealMappedActivationLane()
        {
            publishAbortFromHost();
            drainPrimary(cuda_, 0, cuda_stream_, cuda_terminal_);
            drainPrimary(rocm_, 0, rocm_stream_, rocm_terminal_);
            releaseSideStreams();
            cuda_graph_.reset();
            rocm_graph_.reset();

            if (cuda_ && cuda_terminal_)
                cuda_->destroyEvent(cuda_terminal_, 0);
            if (rocm_ && rocm_terminal_)
                rocm_->destroyEvent(rocm_terminal_, 0);
            if (cuda_ && cuda_stream_)
                cuda_->destroyStream(cuda_stream_, 0);
            if (rocm_ && rocm_stream_)
                rocm_->destroyStream(rocm_stream_, 0);

            // Registration must outlive every stream that may dereference an
            // alias, while the mmap lifetime must outlive unregistration.
            region_.reset();
            mapping_lifetime_.reset();
            control_ = nullptr;
        }

        RealMappedActivationLane(const RealMappedActivationLane &) = delete;
        RealMappedActivationLane &operator=(
            const RealMappedActivationLane &) = delete;

        /**
         * @brief Map/register the ABI and allocate all persistent stream state.
         * @return true when both real backends own exact aliases and streams.
         */
        bool initialize()
        {
            cuda_ = getCUDABackend();
            rocm_ = getROCmBackend();
            if (!cuda_ || !rocm_ || cuda_->deviceCount() < 1 ||
                rocm_->deviceCount() < 1)
            {
                error_ = "requires at least one CUDA and one ROCm device";
                return false;
            }

            void *const mapping = ::mmap(
                nullptr,
                kMappingBytes,
                PROT_READ | PROT_WRITE,
                MAP_SHARED | MAP_ANONYMOUS,
                -1,
                0);
            if (mapping == MAP_FAILED)
            {
                error_ = "mmap(MAP_SHARED|MAP_ANONYMOUS) failed";
                return false;
            }
            std::memset(mapping, 0, kMappingBytes);
            mapping_lifetime_ = std::shared_ptr<void>(
                mapping,
                [](void *address)
                {
                    if (address && address != MAP_FAILED)
                        (void)::munmap(address, kMappingBytes);
                });
            control_ = static_cast<MoEOverlayActivationEpochControl *>(mapping);

            try
            {
                const std::array devices{
                    DeviceId::cuda(0), DeviceId::rocm(0)};
                region_ = engine_.registerExternalMappedHostRegion(
                    mapping,
                    kMappingBytes,
                    devices,
                    mapping_lifetime_);
            }
            catch (const std::exception &exception)
            {
                error_ = exception.what();
                return false;
            }

            cuda_stream_ = cuda_->createStream(0);
            rocm_stream_ = rocm_->createStream(0);
            cuda_terminal_ = cuda_->createEvent(0);
            rocm_terminal_ = rocm_->createEvent(0);
            if (!cuda_stream_ || !rocm_stream_ || !cuda_terminal_ ||
                !rocm_terminal_)
            {
                error_ = "could not allocate exact primary streams/events";
                return false;
            }
            return initializeSideStreams(cuda_, 0) &&
                   initializeSideStreams(rocm_, 0);
        }

        /**
         * @brief Queue side-stream pressure followed by every device epoch.
         *
         * Odd generations make CUDA the continuation endpoint; even
         * generations make ROCm the continuation endpoint. The protocol itself
         * receives only planner-selected devices and exact streams.
         */
        bool enqueueEpochs(
            std::uint64_t generations,
            std::uint32_t stages_per_generation)
        {
            if (!region_ || generations == 0u || stages_per_generation == 0u)
            {
                error_ = "invalid epoch geometry or unbound mapping";
                return false;
            }
            if (!enqueueSideStreamPressure())
                return false;

            try
            {
                for (std::uint64_t generation = 1u;
                     generation <= generations;
                     ++generation)
                {
                    const bool cuda_continuation = (generation & 1u) != 0u;
                    const DeviceId continuation = cuda_continuation
                                                      ? DeviceId::cuda(0)
                                                      : DeviceId::rocm(0);
                    const DeviceId follower = cuda_continuation
                                                  ? DeviceId::rocm(0)
                                                  : DeviceId::cuda(0);
                    void *const continuation_stream = cuda_continuation
                                                          ? cuda_stream_
                                                          : rocm_stream_;
                    void *const follower_stream = cuda_continuation
                                                      ? rocm_stream_
                                                      : cuda_stream_;

                    for (std::uint32_t stage = 0u;
                         stage < stages_per_generation;
                         ++stage)
                    {
                        const std::uint32_t bank =
                            moeOverlayActivationBufferIndex(stage);
                        const std::uint64_t timeline =
                            moeOverlayActivationPackedTimelineValue(
                                generation,
                                moeOverlayActivationBufferVisit(stage));
                        if (timeline == 0u)
                        {
                            error_ = "timeline generation/visit overflow";
                            return false;
                        }

                        // The continuation publishes compact dispatch bytes,
                        // then its signal. The follower's stream cannot publish
                        // a return until that exact signal becomes visible.
                        engine_.enqueueMappedTimelinePublish64(
                            *region_,
                            dispatchSignalOffset(bank),
                            timeline,
                            continuation,
                            continuation_stream);
                        engine_.enqueueMappedTimelineWait64(
                            *region_,
                            dispatchSignalOffset(bank),
                            timeline,
                            follower,
                            follower_stream);
                        engine_.enqueueMappedTimelinePublish64(
                            *region_,
                            returnSignalOffset(bank),
                            timeline,
                            follower,
                            follower_stream);
                        engine_.enqueueMappedTimelineWait64(
                            *region_,
                            returnSignalOffset(bank),
                            timeline,
                            continuation,
                            continuation_stream);
                    }
                }
            }
            catch (const std::exception &exception)
            {
                error_ = exception.what();
                return false;
            }

            if (!cuda_->recordEvent(cuda_terminal_, 0, cuda_stream_) ||
                !rocm_->recordEvent(rocm_terminal_, 0, rocm_stream_))
            {
                error_ = "could not record terminal epoch events";
                return false;
            }
            terminals_recorded_ = true;
            return true;
        }

        /**
         * @brief Capture and launch one complete CUDA-to-ROCm timeline exchange.
         *
         * Each backend receives both a timeline operation and ordinary graph
         * dependency: CUDA publishes dispatch then waits for return, while
         * ROCm waits for dispatch then publishes return. The graph objects are
         * retained until terminal events complete, proving that TransferEngine
         * splices these edges into active native captures instead of executing
         * them during setup or submitting them separately at replay.
         */
        bool captureAndLaunchGraphEpoch()
        {
            if (!region_ || terminals_recorded_)
            {
                error_ = "graph epoch requires a fresh initialized lane";
                return false;
            }
            try
            {
                auto &cuda_context =
                    GPUDeviceContextPool::instance().getContext(
                        DeviceId::cuda(0));
                auto &rocm_context =
                    GPUDeviceContextPool::instance().getContext(
                        DeviceId::rocm(0));
                cuda_graph_ = cuda_context.createGraphCapture(cuda_stream_);
                rocm_graph_ = rocm_context.createGraphCapture(rocm_stream_);
            }
            catch (const std::exception &exception)
            {
                error_ = exception.what();
                return false;
            }
            if (!cuda_graph_ || !rocm_graph_)
            {
                error_ = "could not create exact-stream CUDA/ROCm captures";
                return false;
            }

            constexpr std::uint64_t timeline = 7u;
            try
            {
                if (!cuda_graph_->beginCapture())
                {
                    error_ = "CUDA timeline graph capture did not begin";
                    return false;
                }
                engine_.enqueueMappedTimelinePublish64(
                    *region_, dispatchSignalOffset(0u), timeline,
                    DeviceId::cuda(0), cuda_stream_);
                engine_.enqueueMappedTimelineWait64(
                    *region_, returnSignalOffset(0u), timeline,
                    DeviceId::cuda(0), cuda_stream_);
                if (!cuda_graph_->endCapture())
                {
                    error_ = "CUDA timeline graph capture did not end";
                    return false;
                }

                if (!rocm_graph_->beginCapture())
                {
                    error_ = "ROCm timeline graph capture did not begin";
                    return false;
                }
                engine_.enqueueMappedTimelineWait64(
                    *region_, dispatchSignalOffset(0u), timeline,
                    DeviceId::rocm(0), rocm_stream_);
                engine_.enqueueMappedTimelinePublish64(
                    *region_, returnSignalOffset(0u), timeline,
                    DeviceId::rocm(0), rocm_stream_);
                if (!rocm_graph_->endCapture())
                {
                    error_ = "ROCm timeline graph capture did not end";
                    return false;
                }
            }
            catch (const std::exception &exception)
            {
                error_ = exception.what();
                return false;
            }

            if (cuda_graph_->nodeCount() != 2u ||
                rocm_graph_->nodeCount() != 2u ||
                !cuda_graph_->instantiate() || !rocm_graph_->instantiate())
            {
                error_ = "active timeline captures did not produce two-node executables";
                return false;
            }
            if (!enqueueSideStreamPressure() ||
                !rocm_graph_->launch() || !cuda_graph_->launch() ||
                !cuda_->recordEvent(cuda_terminal_, 0, cuda_stream_) ||
                !rocm_->recordEvent(rocm_terminal_, 0, rocm_stream_))
            {
                error_ = "active timeline graph launch or terminal publication failed";
                return false;
            }
            terminals_recorded_ = true;
            return true;
        }

        /** @brief Observe both terminal events only after all epochs are queued. */
        bool awaitTerminals(std::chrono::steady_clock::duration timeout)
        {
            if (!terminals_recorded_)
                return false;
            const auto deadline = std::chrono::steady_clock::now() + timeout;
            bool cuda_ready = false;
            bool rocm_ready = false;
            while (std::chrono::steady_clock::now() < deadline)
            {
                if (!cuda_ready &&
                    !cuda_->queryEvent(cuda_terminal_, 0, &cuda_ready))
                {
                    error_ = "CUDA terminal query failed";
                    return false;
                }
                if (!rocm_ready &&
                    !rocm_->queryEvent(rocm_terminal_, 0, &rocm_ready))
                {
                    error_ = "ROCm terminal query failed";
                    return false;
                }
                if (cuda_ready && rocm_ready)
                    return awaitSideStreams(deadline);
                std::this_thread::yield();
            }
            error_ = "device-owned activation epochs timed out";
            return false;
        }

        /** @return Host-observed acquire value of one dispatch signal bank. */
        std::uint64_t dispatchSignal(std::uint32_t bank) const
        {
            return std::atomic_ref<std::uint64_t>(
                       control_->buffers[bank].dispatch_signal.value)
                .load(std::memory_order_acquire);
        }

        /** @return Host-observed acquire value of one return signal bank. */
        std::uint64_t returnSignal(std::uint32_t bank) const
        {
            return std::atomic_ref<std::uint64_t>(
                       control_->buffers[bank].return_signal.value)
                .load(std::memory_order_acquire);
        }

        /** @return Last detailed failure from setup or submission. */
        const std::string &error() const noexcept { return error_; }

    private:
        static constexpr size_t kMappingBytes = 4096u;
        static constexpr size_t kSidePayloadBytes = 256u * 1024u;
        static constexpr size_t kSideStreamsPerBackend = 4u;
        static constexpr int kSideRounds = 16;

        /** @return Byte offset of one bank's continuation-owned signal. */
        size_t dispatchSignalOffset(std::uint32_t bank) const
        {
            return static_cast<size_t>(
                reinterpret_cast<const std::byte *>(
                    &control_->buffers[bank].dispatch_signal.value) -
                reinterpret_cast<const std::byte *>(control_));
        }

        /** @return Byte offset of one bank's follower-owned return signal. */
        size_t returnSignalOffset(std::uint32_t bank) const
        {
            return static_cast<size_t>(
                reinterpret_cast<const std::byte *>(
                    &control_->buffers[bank].return_signal.value) -
                reinterpret_cast<const std::byte *>(control_));
        }

        /** @brief Allocate an FusedQKV-shaped persistent side-stream pool. */
        bool initializeSideStreams(IBackend *backend, int ordinal)
        {
            for (size_t index = 0u; index < kSideStreamsPerBackend; ++index)
            {
                SideStream side{
                    .backend = backend,
                    .ordinal = ordinal,
                    .stream = backend->createStream(ordinal),
                    .payload = backend->allocate(kSidePayloadBytes, ordinal),
                    .event = backend->createEvent(ordinal),
                };
                side_streams_.push_back(side);
                if (!side.stream || !side.payload || !side.event)
                {
                    error_ = "could not allocate concurrent side-stream pool";
                    return false;
                }
            }
            return true;
        }

        /** @brief Queue real work concurrently without waiting on the host. */
        bool enqueueSideStreamPressure()
        {
            for (size_t index = 0u; index < side_streams_.size(); ++index)
            {
                auto &side = side_streams_[index];
                for (int round = 0; round < kSideRounds; ++round)
                {
                    if (!side.backend->memset(
                            side.payload,
                            static_cast<int>((index + round) & 0xffu),
                            kSidePayloadBytes,
                            side.ordinal,
                            side.stream))
                    {
                        error_ = "side-stream work submission failed";
                        return false;
                    }
                }
                if (!side.backend->recordEvent(
                        side.event, side.ordinal, side.stream))
                {
                    error_ = "side-stream terminal event submission failed";
                    return false;
                }
            }
            return true;
        }

        /** @brief Await every side event within the shared terminal deadline. */
        bool awaitSideStreams(
            std::chrono::steady_clock::time_point deadline)
        {
            for (const auto &side : side_streams_)
            {
                const auto now = std::chrono::steady_clock::now();
                if (now >= deadline ||
                    !awaitEvent(
                        side.backend,
                        side.ordinal,
                        side.event,
                        deadline - now))
                {
                    error_ = "concurrent side-stream pool timed out";
                    return false;
                }
            }
            return true;
        }

        /** @brief Release all possible future GEQ waits during teardown. */
        void publishAbortFromHost() noexcept
        {
            if (!control_)
                return;
            for (std::uint32_t bank = 0u;
                 bank < kMoEOverlayActivationBufferCount;
                 ++bank)
            {
                std::atomic_ref<std::uint64_t>(
                    control_->buffers[bank].dispatch_signal.value)
                    .store(
                        kMoEOverlayActivationAbortTimeline,
                        std::memory_order_release);
                std::atomic_ref<std::uint64_t>(
                    control_->buffers[bank].return_signal.value)
                    .store(
                        kMoEOverlayActivationAbortTimeline,
                        std::memory_order_release);
            }
        }

        /** @brief Record and observe one cleanup event behind queued work. */
        static void drainPrimary(
            IBackend *backend,
            int ordinal,
            void *stream,
            void *event) noexcept
        {
            if (!backend || !stream || !event)
                return;
            if (backend->recordEvent(event, ordinal, stream))
            {
                (void)awaitEvent(
                    backend, ordinal, event, std::chrono::seconds(5));
            }
        }

        /** @brief Drain and destroy side resources before mapping teardown. */
        void releaseSideStreams() noexcept
        {
            for (auto &side : side_streams_)
            {
                if (side.backend && side.stream && side.event &&
                    side.backend->recordEvent(
                        side.event, side.ordinal, side.stream))
                {
                    (void)awaitEvent(
                        side.backend,
                        side.ordinal,
                        side.event,
                        std::chrono::seconds(5));
                }
                if (side.backend && side.event)
                    side.backend->destroyEvent(side.event, side.ordinal);
                if (side.backend && side.payload)
                    side.backend->free(side.payload, side.ordinal);
                if (side.backend && side.stream)
                    side.backend->destroyStream(side.stream, side.ordinal);
            }
            side_streams_.clear();
        }

        IBackend *cuda_ = nullptr; ///< Borrowed CUDA process authority.
        IBackend *rocm_ = nullptr; ///< Borrowed ROCm process authority.
        TransferEngine engine_{}; ///< Canonical registration/order authority.
        std::shared_ptr<void> mapping_lifetime_; ///< Owns the shared mmap.
        std::shared_ptr<MappedHostTransferRegion> region_; ///< GPU aliases.
        MoEOverlayActivationEpochControl *control_ = nullptr; ///< Host ABI view.
        void *cuda_stream_ = nullptr; ///< Exact CUDA epoch stream.
        void *rocm_stream_ = nullptr; ///< Exact ROCm epoch stream.
        void *cuda_terminal_ = nullptr; ///< Terminal CUDA observation event.
        void *rocm_terminal_ = nullptr; ///< Terminal ROCm observation event.
        std::unique_ptr<IGPUGraphCapture> cuda_graph_; ///< Complete CUDA timeline graph.
        std::unique_ptr<IGPUGraphCapture> rocm_graph_; ///< Complete ROCm timeline graph.
        std::vector<SideStream> side_streams_; ///< Persistent concurrent pool.
        bool terminals_recorded_ = false;
        std::string error_;
    };

    /**
     * @brief Captured heterogeneous round trip with mapped control and DMA payloads.
     *
     * Production prefill packets are several MiB, large enough that kernels which
     * dereference mapped host pages spend most of a layer on PCIe scalar loads and
     * stores. This fixture instead places the payload in persistent device tensors
     * and records four copy-engine operations around the same device-owned epoch:
     * source D2H, target H2D, target D2H, and source H2D. Only terminal events are
     * observed by the host after both complete endpoint graphs have been submitted.
     */
    class RealMappedBulkDmaRoundTrip final
    {
    public:
        RealMappedBulkDmaRoundTrip() = default;

        /** @brief Release queued waits before tearing down graphs and registrations. */
        ~RealMappedBulkDmaRoundTrip()
        {
            publishAbortFromHost();
            drain(source_backend_, source_device_, source_stream_, source_terminal_);
            drain(target_backend_, target_device_, target_stream_, target_terminal_);
            source_graph_.reset();
            target_graph_.reset();
            source_tensor_.reset();
            target_tensor_.reset();
            returned_tensor_.reset();
            destroyEndpoint(
                source_backend_, source_device_, source_stream_, source_terminal_);
            destroyEndpoint(
                target_backend_, target_device_, target_stream_, target_terminal_);
            region_.reset();
            mapping_lifetime_.reset();
        }

        RealMappedBulkDmaRoundTrip(const RealMappedBulkDmaRoundTrip &) = delete;
        RealMappedBulkDmaRoundTrip &operator=(
            const RealMappedBulkDmaRoundTrip &) = delete;

        /**
         * @brief Allocate stable mapping, endpoint tensors, streams, and events.
         * @param source Planner-selected continuation endpoint for this direction.
         * @param target Planner-selected follower endpoint for this direction.
         * @return true when exact cross-backend storage is ready for capture.
         */
        bool initialize(DeviceId source, DeviceId target)
        {
            source_device_ = source;
            target_device_ = target;
            source_backend_ = backendFor(source_device_);
            target_backend_ = backendFor(target_device_);
            if (!source_device_.is_gpu() || !target_device_.is_gpu() ||
                source_device_.type == target_device_.type || !source_backend_ ||
                !target_backend_ ||
                source_backend_->deviceCount() <= source_device_.gpu_ordinal() ||
                target_backend_->deviceCount() <= target_device_.gpu_ordinal())
            {
                error_ = "bulk DMA proof requires one available CUDA and one available ROCm endpoint";
                return false;
            }

            void *const mapping = ::mmap(
                nullptr,
                kMappingBytes,
                PROT_READ | PROT_WRITE,
                MAP_SHARED | MAP_ANONYMOUS,
                -1,
                0);
            if (mapping == MAP_FAILED)
            {
                error_ = "bulk DMA mmap(MAP_SHARED|MAP_ANONYMOUS) failed";
                return false;
            }
            std::memset(mapping, 0, kMappingBytes);
            mapping_lifetime_ = std::shared_ptr<void>(
                mapping,
                [](void *address)
                {
                    if (address && address != MAP_FAILED)
                        (void)::munmap(address, kMappingBytes);
                });

            try
            {
                const std::array devices{source_device_, target_device_};
                region_ = engine_.registerExternalMappedHostRegion(
                    mapping,
                    kMappingBytes,
                    devices,
                    mapping_lifetime_);
                source_stream_ = source_backend_->createStream(
                    source_device_.gpu_ordinal());
                target_stream_ = target_backend_->createStream(
                    target_device_.gpu_ordinal());
                source_terminal_ = source_backend_->createEvent(
                    source_device_.gpu_ordinal());
                target_terminal_ = target_backend_->createEvent(
                    target_device_.gpu_ordinal());
                if (!source_stream_ || !target_stream_ || !source_terminal_ ||
                    !target_terminal_)
                {
                    error_ = "could not allocate bulk DMA streams or terminal events";
                    return false;
                }

                source_tensor_ = std::make_shared<FP32Tensor>(
                    std::vector<std::size_t>{kPayloadFloatCount});
                target_tensor_ = std::make_shared<FP32Tensor>(
                    std::vector<std::size_t>{kPayloadFloatCount});
                returned_tensor_ = std::make_shared<FP32Tensor>(
                    std::vector<std::size_t>{kPayloadFloatCount});
                expected_.resize(kPayloadFloatCount);
                for (std::size_t index = 0u;
                     index < expected_.size();
                     ++index)
                {
                    const std::int32_t integer = static_cast<std::int32_t>(
                        (index * 1315423911u + 2654435761u) % 2003u);
                    expected_[index] =
                        static_cast<float>(integer - 1001) / 997.0f;
                }
                std::copy(
                    expected_.begin(),
                    expected_.end(),
                    source_tensor_->mutable_data());
                if (!source_tensor_->ensureOnDevice(
                        source_device_, source_stream_))
                {
                    error_ = "could not upload deterministic bulk DMA source";
                    return false;
                }
                TransferEngine::allocateDeviceStorage(
                    target_tensor_.get(), target_device_);
                TransferEngine::allocateDeviceStorage(
                    returned_tensor_.get(), source_device_);

                // Capture may import only an already completed external source.
                // Polling this setup event is outside request admission and does
                // not weaken the captured inference transaction below.
                if (!source_backend_->recordEvent(
                        source_terminal_,
                        source_device_.gpu_ordinal(),
                        source_stream_) ||
                    !awaitEvent(
                        source_backend_,
                        source_device_.gpu_ordinal(),
                        source_terminal_,
                        std::chrono::seconds(5)))
                {
                    error_ = "bulk DMA source upload did not complete";
                    return false;
                }
                TransferEngine::requireDeviceInput(
                    source_tensor_.get(), source_device_, source_stream_);
            }
            catch (const std::exception &exception)
            {
                error_ = exception.what();
                return false;
            }
            return true;
        }

        /**
         * @brief Capture complete source and target transactions exactly once.
         * @return true when both four-node graphs instantiate successfully.
         */
        bool capture()
        {
            if (!region_ || !source_tensor_ || !target_tensor_ ||
                !returned_tensor_)
            {
                error_ = "bulk DMA capture requires initialized persistent state";
                return false;
            }
            try
            {
                auto &source_context = GPUDeviceContextPool::instance().getContext(
                    source_device_);
                auto &target_context = GPUDeviceContextPool::instance().getContext(
                    target_device_);
                source_graph_ = source_context.createGraphCapture(source_stream_);
                target_graph_ = target_context.createGraphCapture(target_stream_);
                if (!source_graph_ || !target_graph_)
                {
                    error_ = "could not allocate bulk DMA graph owners";
                    return false;
                }

                std::vector<GraphCaptureDependencyLedger::StagePlan>
                    source_stages;
                source_stages.push_back({
                    .stage_identity = &source_export_stage_,
                    .stage_name = "bulk_dma_source_export",
                    .external_inputs = {
                        source_tensor_->transferStorageOwner()},
                });
                source_stages.push_back({
                    .stage_identity = &source_import_stage_,
                    .stage_name = "bulk_dma_source_import",
                    .outputs = {
                        returned_tensor_->transferStorageOwner()},
                });
                GraphCaptureDependencyLedger source_ledger(
                    source_device_,
                    source_stream_,
                    std::move(source_stages),
                    "mapped_bulk_dma_source");
                {
                    ScopedBackendGraphCapture source_capture(
                        source_context,
                        *source_graph_,
                        "mapped bulk DMA source",
                        &source_ledger);
                    if (!source_capture.begin())
                    {
                        error_ = "bulk DMA source graph capture did not begin";
                        return false;
                    }
                    {
                        ScopedGraphCaptureStage stage(&source_export_stage_);
                        engine_.enqueueDeviceToMappedHost(
                            source_tensor_.get(),
                            0u,
                            *region_,
                            kDispatchPayloadOffset,
                            kPayloadBytes,
                            source_device_,
                            source_stream_);
                        stage.complete();
                    }
                    engine_.enqueueMappedTimelinePublish64(
                        *region_,
                        kDispatchSignalOffset,
                        kTimeline,
                        source_device_,
                        source_stream_);
                    engine_.enqueueMappedTimelineWait64(
                        *region_,
                        kReturnSignalOffset,
                        kTimeline,
                        source_device_,
                        source_stream_);
                    {
                        ScopedGraphCaptureStage stage(&source_import_stage_);
                        engine_.enqueueMappedHostToDevice(
                            *region_,
                            kReturnPayloadOffset,
                            returned_tensor_.get(),
                            0u,
                            kPayloadBytes,
                            source_device_,
                            source_stream_);
                        stage.complete();
                    }
                    source_capture.finish();
                }

                std::vector<GraphCaptureDependencyLedger::StagePlan>
                    target_stages;
                target_stages.push_back({
                    .stage_identity = &target_import_stage_,
                    .stage_name = "bulk_dma_target_import",
                    .outputs = {target_tensor_->transferStorageOwner()},
                });
                target_stages.push_back({
                    .stage_identity = &target_export_stage_,
                    .stage_name = "bulk_dma_target_export",
                    .internal_inputs = {{
                        .tensor = target_tensor_->transferStorageOwner(),
                        .producer_stage_index = 0u,
                    }},
                });
                GraphCaptureDependencyLedger target_ledger(
                    target_device_,
                    target_stream_,
                    std::move(target_stages),
                    "mapped_bulk_dma_target");
                {
                    ScopedBackendGraphCapture target_capture(
                        target_context,
                        *target_graph_,
                        "mapped bulk DMA target",
                        &target_ledger);
                    if (!target_capture.begin())
                    {
                        error_ = "bulk DMA target graph capture did not begin";
                        return false;
                    }
                    engine_.enqueueMappedTimelineWait64(
                        *region_,
                        kDispatchSignalOffset,
                        kTimeline,
                        target_device_,
                        target_stream_);
                    {
                        ScopedGraphCaptureStage stage(&target_import_stage_);
                        engine_.enqueueMappedHostToDevice(
                            *region_,
                            kDispatchPayloadOffset,
                            target_tensor_.get(),
                            0u,
                            kPayloadBytes,
                            target_device_,
                            target_stream_);
                        stage.complete();
                    }
                    {
                        ScopedGraphCaptureStage stage(&target_export_stage_);
                        engine_.enqueueDeviceToMappedHost(
                            target_tensor_.get(),
                            0u,
                            *region_,
                            kReturnPayloadOffset,
                            kPayloadBytes,
                            target_device_,
                            target_stream_);
                        stage.complete();
                    }
                    engine_.enqueueMappedTimelinePublish64(
                        *region_,
                        kReturnSignalOffset,
                        kTimeline,
                        target_device_,
                        target_stream_);
                    target_capture.finish();
                }
            }
            catch (const std::exception &exception)
            {
                error_ = exception.what();
                return false;
            }

            if (source_graph_->nodeCount() != 4u ||
                target_graph_->nodeCount() != 4u ||
                !source_graph_->instantiate() ||
                !target_graph_->instantiate())
            {
                error_ = "bulk DMA transactions did not instantiate as exact four-node graphs";
                return false;
            }
            if (dispatchSignal() != 0u || returnSignal() != 0u)
            {
                error_ = "bulk DMA graph capture executed a timeline publication";
                return false;
            }
            return true;
        }

        /**
         * @brief Replay captured round trips and retain median wall latency.
         * @param warmups Completed samples omitted from the median.
         * @param measurements Positive number of retained samples.
         * @return true when every replay completes without a host stage bridge.
         */
        bool measure(std::size_t warmups, std::size_t measurements)
        {
            if (!source_graph_ || !target_graph_ || measurements == 0u)
            {
                error_ = "bulk DMA measurement requires instantiated graphs and samples";
                return false;
            }
            std::vector<double> samples_us;
            samples_us.reserve(measurements);
            for (std::size_t sample = 0u;
                 sample < warmups + measurements;
                 ++sample)
            {
                resetSignals();
                const auto begin = std::chrono::steady_clock::now();

                // Queue the waiter before the producer. There is no host callback,
                // payload copy, or stage-level event between these submissions.
                if (!target_graph_->launch() || !source_graph_->launch() ||
                    !source_backend_->recordEvent(
                        source_terminal_,
                        source_device_.gpu_ordinal(),
                        source_stream_) ||
                    !target_backend_->recordEvent(
                        target_terminal_,
                        target_device_.gpu_ordinal(),
                        target_stream_) ||
                    !awaitTerminals(std::chrono::seconds(5)))
                {
                    if (error_.empty())
                        error_ = "bulk DMA graph launch or terminal observation failed";
                    return false;
                }
                const auto elapsed = std::chrono::steady_clock::now() - begin;
                if (dispatchSignal() != kTimeline ||
                    returnSignal() != kTimeline)
                {
                    error_ = "bulk DMA replay completed without both timeline publications";
                    return false;
                }
                if (sample >= warmups)
                {
                    samples_us.push_back(
                        std::chrono::duration<double, std::micro>(elapsed)
                            .count());
                }
            }
            std::sort(samples_us.begin(), samples_us.end());
            median_microseconds_ = samples_us[samples_us.size() / 2u];
            return true;
        }

        /**
         * @brief Download the terminal tensor and compare every payload bit.
         * @return true only when the four copy operations preserved all bytes.
         */
        bool verifyIntegrity()
        {
            try
            {
                // The retained parent owns one terminal stream edge in
                // production. This focused harness publishes the same exact
                // stream here so the ordinary tensor download can join it.
                TransferEngine::publishDeviceWrite(
                    returned_tensor_.get(), source_device_, source_stream_);
                if (!returned_tensor_->ensureOnHost())
                {
                    error_ = "could not materialize returned bulk DMA payload";
                    return false;
                }
            }
            catch (const std::exception &exception)
            {
                error_ = exception.what();
                return false;
            }
            if (std::memcmp(
                    returned_tensor_->data(),
                    expected_.data(),
                    kPayloadBytes) != 0)
            {
                error_ = "bulk DMA round trip changed deterministic payload bytes";
                return false;
            }
            return true;
        }

        /** @return Median complete four-copy round-trip time in microseconds. */
        [[nodiscard]] double medianMicroseconds() const noexcept
        {
            return median_microseconds_;
        }

        /** @return Aggregate four-leg payload throughput in GiB/s. */
        [[nodiscard]] double aggregateGiBPerSecond() const noexcept
        {
            if (median_microseconds_ <= 0.0)
                return 0.0;
            constexpr double bytes_per_round_trip =
                static_cast<double>(kPayloadBytes) * 4.0;
            return bytes_per_round_trip /
                   (median_microseconds_ * 1.0e-6) /
                   static_cast<double>(std::uint64_t{1} << 30u);
        }

        /** @return Detailed setup, capture, replay, or parity failure. */
        [[nodiscard]] const std::string &error() const noexcept { return error_; }

    private:
        static constexpr std::size_t kPageBytes = 4096u;
        static constexpr std::size_t kPayloadBytes = 4u * 1024u * 1024u;
        static constexpr std::size_t kPayloadFloatCount =
            kPayloadBytes / sizeof(float);
        static constexpr std::size_t kDispatchSignalOffset = 0u;
        static constexpr std::size_t kReturnSignalOffset = 64u;
        static constexpr std::size_t kDispatchPayloadOffset = kPageBytes;
        static constexpr std::size_t kReturnPayloadOffset =
            kDispatchPayloadOffset + kPayloadBytes;
        static constexpr std::size_t kMappingBytes =
            kReturnPayloadOffset + kPayloadBytes;
        static constexpr std::uint64_t kTimeline = 17u;

        /** @return Process backend authority for one exact endpoint type. */
        static IBackend *backendFor(DeviceId device) noexcept
        {
            if (device.type == DeviceType::CUDA)
                return getCUDABackend();
            if (device.type == DeviceType::ROCm)
                return getROCmBackend();
            return nullptr;
        }

        /** @return Acquire observation of the source-owned publication word. */
        [[nodiscard]] std::uint64_t dispatchSignal() const noexcept
        {
            auto *const signal = static_cast<std::uint64_t *>(
                region_->mutableHostData(kDispatchSignalOffset));
            return std::atomic_ref<std::uint64_t>(*signal).load(
                std::memory_order_acquire);
        }

        /** @return Acquire observation of the target-owned publication word. */
        [[nodiscard]] std::uint64_t returnSignal() const noexcept
        {
            auto *const signal = static_cast<std::uint64_t *>(
                region_->mutableHostData(kReturnSignalOffset));
            return std::atomic_ref<std::uint64_t>(*signal).load(
                std::memory_order_acquire);
        }

        /** @brief Reset a completed test replay before reusing its fixed graph value. */
        void resetSignals() noexcept
        {
            auto *const dispatch = static_cast<std::uint64_t *>(
                region_->mutableHostData(kDispatchSignalOffset));
            auto *const returned = static_cast<std::uint64_t *>(
                region_->mutableHostData(kReturnSignalOffset));
            std::atomic_ref<std::uint64_t>(*dispatch).store(
                0u, std::memory_order_release);
            std::atomic_ref<std::uint64_t>(*returned).store(
                0u, std::memory_order_release);
        }

        /** @brief Release either graph if cleanup encounters an outstanding wait. */
        void publishAbortFromHost() noexcept
        {
            if (!region_)
                return;
            auto *const dispatch = static_cast<std::uint64_t *>(
                region_->mutableHostData(kDispatchSignalOffset));
            auto *const returned = static_cast<std::uint64_t *>(
                region_->mutableHostData(kReturnSignalOffset));
            std::atomic_ref<std::uint64_t>(*dispatch).store(
                std::numeric_limits<std::uint64_t>::max(),
                std::memory_order_release);
            std::atomic_ref<std::uint64_t>(*returned).store(
                std::numeric_limits<std::uint64_t>::max(),
                std::memory_order_release);
        }

        /** @brief Poll both terminal events under one shared deadline. */
        bool awaitTerminals(std::chrono::steady_clock::duration timeout)
        {
            const auto deadline = std::chrono::steady_clock::now() + timeout;
            bool source_ready = false;
            bool target_ready = false;
            while (std::chrono::steady_clock::now() < deadline)
            {
                if (!source_ready &&
                    !source_backend_->queryEvent(
                        source_terminal_,
                        source_device_.gpu_ordinal(),
                        &source_ready))
                {
                    error_ = "bulk DMA source terminal query failed";
                    return false;
                }
                if (!target_ready &&
                    !target_backend_->queryEvent(
                        target_terminal_,
                        target_device_.gpu_ordinal(),
                        &target_ready))
                {
                    error_ = "bulk DMA target terminal query failed";
                    return false;
                }
                if (source_ready && target_ready)
                    return true;
                std::this_thread::yield();
            }
            error_ = "captured bulk DMA round trip timed out";
            return false;
        }

        /** @brief Best-effort event drain used only by failure teardown. */
        static void drain(
            IBackend *backend,
            DeviceId device,
            void *stream,
            void *event) noexcept
        {
            if (!backend || !device.is_gpu() || !stream || !event)
                return;
            if (backend->recordEvent(event, device.gpu_ordinal(), stream))
            {
                (void)awaitEvent(
                    backend,
                    device.gpu_ordinal(),
                    event,
                    std::chrono::seconds(5));
            }
        }

        /** @brief Destroy one exact endpoint's event and stream after draining. */
        static void destroyEndpoint(
            IBackend *backend,
            DeviceId device,
            void *&stream,
            void *&event) noexcept
        {
            if (backend && device.is_gpu() && event)
                backend->destroyEvent(event, device.gpu_ordinal());
            if (backend && device.is_gpu() && stream)
                backend->destroyStream(stream, device.gpu_ordinal());
            event = nullptr;
            stream = nullptr;
        }

        DeviceId source_device_ = DeviceId::invalid(); ///< Current direction's authority.
        DeviceId target_device_ = DeviceId::invalid(); ///< Current direction's follower.
        IBackend *source_backend_ = nullptr; ///< Borrowed source backend.
        IBackend *target_backend_ = nullptr; ///< Borrowed target backend.
        TransferEngine engine_{}; ///< Canonical payload and epoch authority.
        std::shared_ptr<void> mapping_lifetime_; ///< Owns registered shared pages.
        std::shared_ptr<MappedHostTransferRegion> region_; ///< Both endpoint aliases.
        std::shared_ptr<FP32Tensor> source_tensor_; ///< Persistent source payload.
        std::shared_ptr<FP32Tensor> target_tensor_; ///< Persistent follower bounce.
        std::shared_ptr<FP32Tensor> returned_tensor_; ///< Persistent returned payload.
        std::vector<float> expected_; ///< Exact host parity oracle.
        void *source_stream_ = nullptr; ///< Exact source graph stream.
        void *target_stream_ = nullptr; ///< Exact follower graph stream.
        void *source_terminal_ = nullptr; ///< Source terminal observation only.
        void *target_terminal_ = nullptr; ///< Target terminal observation only.
        std::unique_ptr<IGPUGraphCapture> source_graph_; ///< Complete source graph.
        std::unique_ptr<IGPUGraphCapture> target_graph_; ///< Complete target graph.
        int source_export_stage_ = 0; ///< Stable capture-ledger identity.
        int source_import_stage_ = 0; ///< Stable capture-ledger identity.
        int target_import_stage_ = 0; ///< Stable capture-ledger identity.
        int target_export_stage_ = 0; ///< Stable capture-ledger identity.
        double median_microseconds_ = 0.0; ///< Retained economy sample.
        std::string error_; ///< Last actionable failure.
    };
} // namespace

/**
 * @test Real CUDA/ROCm devices exchange double-buffered epochs in either role.
 */
TEST(Test__MappedActivationEpochCUDAAndROCm,
     AlternatingRolesCompleteWithoutPerStageHostDispatchOrWait)
{
    IBackend *const cuda = getCUDABackend();
    IBackend *const rocm = getROCmBackend();
    ASSERT_NE(cuda, nullptr);
    ASSERT_NE(rocm, nullptr);
    if (cuda->deviceCount() < 1 || rocm->deviceCount() < 1)
    {
        GTEST_SKIP() << "Requires at least one CUDA and one ROCm device";
    }

    PerfStatsCollector::reset();
    RealMappedActivationLane lane;
    ASSERT_TRUE(lane.initialize()) << lane.error();

    constexpr std::uint64_t generations = 6u;
    constexpr std::uint32_t stages_per_generation = 32u;
    const auto submit_begin = std::chrono::steady_clock::now();
    ASSERT_TRUE(lane.enqueueEpochs(generations, stages_per_generation))
        << lane.error();
    const auto submit_elapsed =
        std::chrono::steady_clock::now() - submit_begin;

    // Every wait is device-queued. Submission must not wait for any stage to
    // complete, even while unrelated streams occupy backend queue resources.
    EXPECT_LT(submit_elapsed, std::chrono::seconds(2))
        << "epoch submission appears to contain an inline host bridge";
    ASSERT_TRUE(lane.awaitTerminals(std::chrono::seconds(10)))
        << lane.error();

    const std::uint64_t expected = moeOverlayActivationPackedTimelineValue(
        generations,
        moeOverlayActivationBufferVisit(stages_per_generation - 1u));
    ASSERT_NE(expected, 0u);
    for (std::uint32_t bank = 0u;
         bank < kMoEOverlayActivationBufferCount;
         ++bank)
    {
        EXPECT_EQ(lane.dispatchSignal(bank), expected);
        EXPECT_EQ(lane.returnSignal(bank), expected);
    }

    double regions = 0.0;
    double aliases = 0.0;
    double families = 0.0;
    double device_local_registrations = 0.0;
    double portable_registrations = 0.0;
    double waits = 0.0;
    double publications = 0.0;
    for (const auto &record : PerfStatsCollector::snapshot(
             {"moe_overlay_activation_epoch"}))
    {
        if (record.domain != "moe_overlay_activation_epoch")
            continue;
        if (record.name == "mapped_regions_registered")
        {
            regions += record.value;
            EXPECT_EQ(record.tags.at("mapping"), "typed_external_host_pages");
        }
        else if (record.name == "mapped_endpoint_aliases")
            aliases += record.value;
        else if (record.name == "mapped_backend_families")
            families += record.value;
        else if (record.name == "mapped_backend_registrations")
        {
            const auto &scope = record.tags.at("registration_scope");
            if (scope == "device_local")
                device_local_registrations += record.value;
            else if (scope == "backend_portable")
                portable_registrations += record.value;
            else
                ADD_FAILURE() << "Unknown mapped registration scope: " << scope;
        }
        else if (record.name == "device_timeline_waits_enqueued")
        {
            waits += record.value;
            EXPECT_EQ(record.tags.at("scope"), "node_local");
            EXPECT_EQ(record.tags.at("host_blocking"), "false");
        }
        else if (record.name == "device_timeline_publications_enqueued")
        {
            publications += record.value;
            EXPECT_EQ(record.tags.at("scope"), "node_local");
            EXPECT_EQ(record.tags.at("host_blocking"), "false");
        }
    }
    const double expected_direction_operations =
        static_cast<double>(generations * stages_per_generation * 2u);
    EXPECT_EQ(regions, 1.0);
    EXPECT_EQ(aliases, 2.0);
    EXPECT_EQ(families, 2.0);
    EXPECT_EQ(device_local_registrations, 2.0);
    EXPECT_EQ(portable_registrations, 0.0);
    EXPECT_EQ(waits, expected_direction_operations);
    EXPECT_EQ(publications, expected_direction_operations);
}

/**
 * @test Active CUDA/ROCm captures own mapped timeline edges in one executable.
 */
TEST(Test__MappedActivationEpochCUDAAndROCm,
     ActiveCapturesSpliceTimelineEdgesWithoutChildGraphs)
{
    IBackend *const cuda = getCUDABackend();
    IBackend *const rocm = getROCmBackend();
    ASSERT_NE(cuda, nullptr);
    ASSERT_NE(rocm, nullptr);
    if (cuda->deviceCount() < 1 || rocm->deviceCount() < 1)
    {
        GTEST_SKIP() << "Requires at least one CUDA and one ROCm device";
    }

    RealMappedActivationLane lane;
    ASSERT_TRUE(lane.initialize()) << lane.error();
    ASSERT_TRUE(lane.captureAndLaunchGraphEpoch()) << lane.error();
    ASSERT_TRUE(lane.awaitTerminals(std::chrono::seconds(10)))
        << lane.error();
    EXPECT_EQ(lane.dispatchSignal(0u), 7u);
    EXPECT_EQ(lane.returnSignal(0u), 7u);
    EXPECT_EQ(lane.dispatchSignal(1u), 0u);
    EXPECT_EQ(lane.returnSignal(1u), 0u);
}

/**
 * @test Late ROCm mappings remain graph-addressable at production cardinality.
 *
 * A production ExpertOverlay model owns independent logical ticket storage for
 * several prefill buckets, decode, and every retained MTP depth. The model-owned
 * arena must preserve every logical address while reducing native mapped-host
 * backing regions from linear ticket cardinality to logarithmic growth.
 * This regression retains production-scale slices, captures publications
 * against the final layer-sized suffix, and replays twice. It proves both late
 * slice stability and the structural mapped-backing bound that prevents ROCm
 * IOMMU workqueue/IH-ring exhaustion.
 */
TEST(Test__MappedActivationEpochCUDAAndROCm,
     ArenaBackedLateROCmMappingsRemainCapturedAtProductionCardinality)
{
    IBackend *const rocm = getROCmBackend();
    ASSERT_NE(rocm, nullptr);
    if (rocm->deviceCount() < 1)
    {
        GTEST_SKIP() << "Requires at least one ROCm device";
    }

    constexpr size_t kRetainedRegionCount = 640u;
    constexpr size_t kCapturedSuffixCount = 48u;
    constexpr std::uint64_t kPublishedTimeline = 11u;
    static_assert(kCapturedSuffixCount <= kRetainedRegionCount);

    TransferEngine engine;
    const DeviceId device = DeviceId::rocm(0);
    const std::array devices{device};
    auto arena = engine.createMappedHostArena(devices);
    ASSERT_NE(arena, nullptr);
    std::vector<std::shared_ptr<MappedHostTransferRegion>> regions;
    regions.reserve(kRetainedRegionCount);
    for (size_t index = 0u; index < kRetainedRegionCount; ++index)
    {
        SCOPED_TRACE(::testing::Message()
                     << "mapped_region_index=" << index);
        auto region = arena->allocate(
            2u * sizeof(std::uint64_t), /*alignment=*/64u);
        ASSERT_NE(region, nullptr);
        ASSERT_TRUE(region->isBound());
        auto *const words = static_cast<std::uint64_t *>(
            region->mutableHostData());
        words[0] = 0u;
        words[1] = 0u;
        regions.push_back(std::move(region));
    }

    const auto arena_snapshot = arena->snapshot();
    size_t logarithmic_registration_bound = 1u;
    for (size_t remaining = kRetainedRegionCount;
         remaining > 1u;
         remaining = (remaining + 1u) / 2u)
    {
        ++logarithmic_registration_bound;
    }
    EXPECT_EQ(arena_snapshot.slice_count, kRetainedRegionCount);
    EXPECT_LE(
        arena_snapshot.backing_region_count,
        logarithmic_registration_bound)
        << "mapped backing regions grew linearly with logical tickets";

    void *const stream = rocm->createStream(0);
    void *const terminal = rocm->createEvent(0);
    ASSERT_NE(stream, nullptr);
    ASSERT_NE(terminal, nullptr);

    auto &context = GPUDeviceContextPool::instance().getContext(device);
    auto graph = context.createGraphCapture(stream);
    ASSERT_NE(graph, nullptr);
    ASSERT_TRUE(graph->beginCapture());
    for (size_t index = kRetainedRegionCount - kCapturedSuffixCount;
         index < kRetainedRegionCount;
         ++index)
    {
        // These are the mappings most likely to expose a backend mapping
        // ceiling because they are created after every simulated prefill page.
        engine.enqueueMappedTimelinePublish64(
            *regions[index],
            0u,
            kPublishedTimeline,
            device,
            stream);
    }
    ASSERT_TRUE(graph->endCapture());
    ASSERT_EQ(graph->nodeCount(), kCapturedSuffixCount);
    ASSERT_TRUE(graph->instantiate());

    for (size_t replay = 0u; replay < 2u; ++replay)
    {
        for (size_t index = kRetainedRegionCount - kCapturedSuffixCount;
             index < kRetainedRegionCount;
             ++index)
        {
            *static_cast<std::uint64_t *>(regions[index]->mutableHostData()) = 0u;
        }
        ASSERT_TRUE(graph->launch());
        ASSERT_TRUE(rocm->recordEvent(terminal, 0, stream));
        ASSERT_TRUE(awaitEvent(
            rocm, 0, terminal, std::chrono::seconds(10)))
            << "late mapped publication graph did not complete on replay "
            << replay;
        for (size_t index = kRetainedRegionCount - kCapturedSuffixCount;
             index < kRetainedRegionCount;
             ++index)
        {
            EXPECT_EQ(
                *static_cast<std::uint64_t *>(
                    regions[index]->mutableHostData()),
                kPublishedTimeline)
                << "replay=" << replay << " mapped_region_index=" << index;
        }
    }

    graph.reset();
    rocm->destroyEvent(terminal, 0);
    rocm->destroyStream(stream, 0);
}

/**
 * @test Captured copy engines preserve and economically move prefill payloads.
 *
 * The direction sweep prevents the implementation from assigning implicit
 * continuation semantics to either vendor. Aggregate bandwidth counts all four
 * 4 MiB legs; the floor rejects a regression to direct mapped scalar traffic
 * while leaving headroom for different PCIe generations and NUMA placements.
 */
TEST(Test__MappedActivationEpochCUDAAndROCm,
     CapturedBulkDmaRoundTripIsExactAndEconomicInBothDirections)
{
    IBackend *const cuda = getCUDABackend();
    IBackend *const rocm = getROCmBackend();
    ASSERT_NE(cuda, nullptr);
    ASSERT_NE(rocm, nullptr);
    if (cuda->deviceCount() < 1 || rocm->deviceCount() < 1)
    {
        GTEST_SKIP() << "Requires at least one CUDA and one ROCm device";
    }

    struct Direction
    {
        DeviceId source;
        DeviceId target;
        const char *name;
    };
    const std::array directions{
        Direction{DeviceId::cuda(0), DeviceId::rocm(0), "cuda_to_rocm"},
        Direction{DeviceId::rocm(0), DeviceId::cuda(0), "rocm_to_cuda"},
    };
    for (const auto &direction : directions)
    {
        RealMappedBulkDmaRoundTrip lane;
        ASSERT_TRUE(lane.initialize(direction.source, direction.target))
            << direction.name << ": " << lane.error();
        ASSERT_TRUE(lane.capture())
            << direction.name << ": " << lane.error();
        ASSERT_TRUE(lane.measure(/*warmups=*/5u, /*measurements=*/25u))
            << direction.name << ": " << lane.error();
        ASSERT_TRUE(lane.verifyIntegrity())
            << direction.name << ": " << lane.error();

        std::cout << "[bulk-dma] direction=" << direction.name
                  << " payload_bytes=" << 4u * 1024u * 1024u
                  << " median_round_trip_us=" << lane.medianMicroseconds()
                  << " aggregate_gib_per_second="
                  << lane.aggregateGiBPerSecond() << '\n';
        EXPECT_GT(lane.aggregateGiBPerSecond(), 2.0)
            << direction.name
            << " fell below the economical captured-DMA floor";
    }
}
