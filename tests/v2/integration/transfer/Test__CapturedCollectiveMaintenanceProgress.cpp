/**
 * @file Test__CapturedCollectiveMaintenanceProgress.cpp
 * @brief Captured multi-GPU collectives remain live during CPU-rendezvous maintenance.
 *
 * Each participant owns one complete captured chain of collective, publication,
 * and CPU-acknowledgement edges. Concurrent native producers and mapped expert
 * copies use the production transfer pool. No model, numerical tolerance,
 * transport substitution, or inference-time host synchronization is involved.
 */

#include <gtest/gtest.h>

#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IBackend.h"
#include "backends/IGPUGraphCapture.h"
#include "backends/IWorkerGPUContext.h"
#include "collective/ILocalTPContext.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "execution/local_execution/orchestrators/TPWorkerPool.h"
#include "transfer/TransferEngine.h"
#include "../../utils/TestTensorFactory.h"

#include <atomic>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace llaminar2::test
{
namespace
{
    /** Immutable geometry; the layer count reaches beyond the observed layer-26 stall. */
    constexpr std::size_t kLayers = 32u;
    constexpr std::size_t kElements = 1024u;
    constexpr std::size_t kLanes = 32u;
    constexpr std::size_t kCopyBytes = 256u * 1024u;

    /** Whether independent consumers have acquired the graph's future terminal. */
    enum class OutputReaders { None, FutureConsumers };

    /** Match the three-way stream fork used by fused Q/K/V compute stages. */
    enum class ComputeTopology { SerialStream, ThreeWayFork };

    /** Diagnostic distinction between native and retained producer admission. */
    enum class ProducerAdmission { Native, RetainedGraph };

    /** Setup-owned resources; graphs retire before their embedded addresses. */
    struct Participant
    {
        DeviceId device;
        IWorkerGPUContext *context = nullptr;
        void *inference = nullptr;
        std::shared_ptr<MappedHostTransferRegion> timeline;
        std::vector<std::unique_ptr<FP32Tensor>> layers;
        std::shared_ptr<DeviceTransferBuffer> source;
        std::vector<std::shared_ptr<DeviceTransferBuffer>> produced;
        std::vector<PersistentTransferStagingSlice> staging;
        std::vector<PersistentTransferExecutionLane> execution;
        std::vector<void *> maintenance_done;
        std::vector<std::unique_ptr<IGPUGraphCapture>> maintenance_graphs;
        std::shared_ptr<MappedHostTransferRegion> maintenance_publications;
        std::array<std::size_t, kLanes> maintenance_submissions{};
        std::size_t maintenance_completions_during_inference = 0u;
        std::shared_ptr<MappedHostTransferRegion> observations;
        std::vector<void *> observers;
        void *terminal = nullptr;
        std::array<void *, 3> compute{};
        std::vector<void *> fork_ready;
        std::vector<std::array<void *, 3>> fork_done;
        std::array<std::shared_ptr<DeviceTransferBuffer>, 3> fork_outputs;
        std::unique_ptr<IGPUGraphCapture> graph;
    };

    /**
     * @brief Run symmetric participant actions and propagate their exact failures.
     * @param count Number of rank-local participants, not a hard-coded topology.
     * @param action Worker action, which must not leave a collective half-submitted.
     */
    void runParticipants(std::size_t count, const std::function<void(std::size_t)> &action)
    {
        TPWorkerPool workers(count);
        workers.dispatch([&](std::size_t index) {
            action(index);
            return true;
        });
        for (const auto &result : workers.collectAll(30'000))
        {
            if (result.exception) std::rethrow_exception(result.exception);
            if (!result.completed || !result.success)
                throw std::runtime_error("Captured collective participant did not finish its action");
        }
    }

    /**
     * @brief Exercise a real captured collective while the CPU acknowledges each layer.
     * @param first_device Backend selector; ordinals are discovered and validated below.
     * @param degree Required homogeneous TP degree.
     * @param readers Optional prefix-style consumers waiting for the terminal event.
     * @param compute_topology Participant-local computation stream topology.
     * @param producer_admission Diagnostic native versus graph scheduling control.
     *
     * Maintenance may be delayed by native scheduling, but it must not deadlock
     * the inference that lets that scheduling resume. Every graph reaches every
     * rendezvous, then all maintenance drains with byte-exact payloads. The CPU
     * acknowledgment guard releases all remaining waits before diagnostic joins,
     * including on assertion failure, so the test cannot leave a held GPU behind.
     */
    void proveCapturedCollectiveMaintenance(DeviceId first_device, std::size_t degree,
                                           OutputReaders readers = OutputReaders::None,
                                           ComputeTopology compute_topology = ComputeTopology::SerialStream,
                                           ProducerAdmission producer_admission = ProducerAdmission::Native)
    {
        auto *backend = getBackendFor(first_device);
        ASSERT_NE(backend, nullptr);
        if (backend->deviceCount() < static_cast<int>(degree))
            GTEST_SKIP() << "Requires " << degree << " devices for " << first_device.toString();

        TransferEngine transfers;
        std::vector<GlobalDeviceAddress> addresses;
        std::vector<Participant> participants(degree);
        for (std::size_t index = 0; index < degree; ++index)
        {
            auto &p = participants[index];
            p.device = first_device.is_cuda() ? DeviceId::cuda(index) : DeviceId::rocm(index);
            addresses.push_back(first_device.is_cuda()
                ? GlobalDeviceAddress::cuda(index) : GlobalDeviceAddress::rocm(index));
            p.context = &GPUDeviceContextPool::instance().getContext(p.device);
            const DeviceId mapped_devices[] = {p.device};
            p.timeline = transfers.allocateMappedHostRegion(128u, mapped_devices);
            std::memset(p.timeline->mutableHostData(), 0, 128u);
            p.maintenance_publications = transfers.allocateMappedHostRegion(kLanes * 64u, mapped_devices);
            std::memset(p.maintenance_publications->mutableHostData(), 0, kLanes * 64u);
            if (readers == OutputReaders::FutureConsumers)
            {
                p.observations = transfers.allocateMappedHostRegion(kLanes * 128u, mapped_devices);
                std::memset(p.observations->mutableHostData(), 0, kLanes * 128u);
            }
            p.source = transfers.allocateDeviceTransferBuffer(kCopyBytes, p.device);
            if (compute_topology == ComputeTopology::ThreeWayFork)
                for (auto &output : p.fork_outputs)
                    output = transfers.allocateDeviceTransferBuffer(kCopyBytes, p.device);
            p.staging = transfers.allocatePersistentTransferStagingSlices(kCopyBytes, kLanes, p.device);
            p.execution = transfers.allocatePersistentTransferExecutionLanes(
                kLanes, p.device, "captured_collective_maintenance");
            for (std::size_t lane = 0; lane < kLanes; ++lane)
                p.produced.push_back(transfers.allocateDeviceTransferBuffer(kCopyBytes, p.device));
            for (std::size_t layer = 0; layer < kLayers; ++layer)
            {
                p.layers.push_back(TestTensorFactory::createFP32({kElements}));
                TestTensorFactory::fillValue(p.layers.back().get(), static_cast<float>(index + 1u));
            }
            p.context->submitAndWait([&] {
                p.inference = p.context->getOrCreateAuxiliaryStream("captured_collective_main");
                if (!p.inference) throw std::runtime_error("Missing inference stream");
                for (std::size_t lane = 0; lane < kLanes; ++lane)
                {
                    p.maintenance_done.push_back(p.context->createEvent());
                    if (!p.maintenance_done.back())
                        throw std::runtime_error("Missing maintenance completion event");
                }
                if (compute_topology == ComputeTopology::ThreeWayFork)
                {
                    for (std::size_t fork = 0; fork < p.compute.size(); ++fork)
                    {
                        p.compute[fork] = p.context->getOrCreateAuxiliaryStream(
                            "captured_collective_compute_" + std::to_string(fork));
                        if (!p.compute[fork]) throw std::runtime_error("Missing compute stream");
                    }
                    for (std::size_t layer = 0; layer < kLayers; ++layer)
                    {
                        p.fork_ready.push_back(p.context->createEvent());
                        if (!p.fork_ready.back()) throw std::runtime_error("Missing compute fork event");
                        p.fork_done.emplace_back();
                        for (auto &event : p.fork_done.back())
                        {
                            event = p.context->createEvent();
                            if (!event) throw std::runtime_error("Missing compute join event");
                        }
                    }
                }
                if (readers == OutputReaders::FutureConsumers)
                {
                    p.terminal = p.context->createEvent();
                    if (!p.terminal) throw std::runtime_error("Missing terminal event");
                    for (std::size_t lane = 0; lane < kLanes; ++lane)
                    {
                        auto *observer = p.context->getOrCreateAuxiliaryStream(
                            "captured_collective_reader_" + std::to_string(lane));
                        if (!observer) throw std::runtime_error("Missing observation stream");
                        p.observers.push_back(observer);
                    }
                }
                for (auto &layer : p.layers)
                    if (!layer->ensureOnDevice(p.device, p.inference))
                        throw std::runtime_error("Collective payload setup failed");
                if (!backend->memset(p.source->mutableDeviceData(), 0x5a, kCopyBytes,
                                      p.device.gpu_ordinal(), p.inference))
                    throw std::runtime_error("Maintenance source setup failed");
                p.context->synchronizeStream(p.inference); // Cold fixture preparation only.
                if (producer_admission == ProducerAdmission::RetainedGraph)
                {
                    for (std::size_t lane = 0; lane < kLanes; ++lane)
                    {
                        auto graph = p.context->createGraphCapture(p.execution[lane].stream());
                        if (!graph) throw std::runtime_error("Missing maintenance capture owner");
                        ScopedBackendGraphCapture capture(*p.context, *graph,
                                                           "maintenance admission control");
                        if (!capture.begin() ||
                            !backend->copyDeviceVisibleRegionByKernelOnStream(
                                p.produced[lane]->mutableDeviceData(), p.source->deviceData(),
                                kCopyBytes, p.device.gpu_ordinal(), p.execution[lane].stream()) ||
                            !transfers.enqueueBackgroundStagingCopy(p.execution[lane],
                                MappedTransferDirection::DeviceToHost,
                                p.produced[lane]->mutableDeviceData(), kCopyBytes, 0u,
                                p.staging[lane], 0u, kCopyBytes))
                            throw std::runtime_error("Maintenance admission capture failed");
                        transfers.enqueueMappedTimelinePublish64(*p.maintenance_publications,
                            lane * 64u, 1u, p.device, p.execution[lane].stream());
                        capture.finish();
                        if (!graph->instantiate())
                            throw std::runtime_error("Maintenance admission instantiate failed");
                        p.maintenance_graphs.push_back(std::move(graph));
                    }
                }
                p.graph = p.context->createGraphCapture(p.inference);
                if (!p.graph) throw std::runtime_error("Missing native capture owner");
            });
        }
        auto collective = createLocalTPContext(addresses, {}, first_device.is_cuda()
            ? CollectiveBackendType::NCCL : CollectiveBackendType::RCCL);
        ASSERT_NE(collective, nullptr);

        // Warm only native communicator resources. Restore the exact input so
        // the later numerical assertion cannot pass using this eager result.
        runParticipants(degree, [&](std::size_t index) {
            auto &p = participants[index];
            p.context->submitAndWait([&] {
                if (!collective->allreduceOnStream(p.layers.front().get(),
                        "captured_maintenance_warmup", kElements, p.inference))
                    throw std::runtime_error("Collective resource warmup failed");
                p.context->synchronizeStream(p.inference);
                std::vector<float> initial(kElements, static_cast<float>(index + 1u));
                if (!backend->hostToDevice(p.layers.front()->gpu_data_ptr(), initial.data(),
                        initial.size() * sizeof(float), p.device.gpu_ordinal(), p.inference))
                    throw std::runtime_error("Collective input restoration failed");
                p.context->synchronizeStream(p.inference);
            });
        });
        runParticipants(degree, [&](std::size_t index) {
            auto &p = participants[index];
            p.context->submitAndWait([&] {
                ScopedBackendGraphCapture capture(*p.context, *p.graph,
                                                   "collective maintenance progress");
                if (!capture.begin()) throw std::runtime_error("Collective capture begin failed");
                for (std::size_t layer = 0; layer < kLayers; ++layer)
                {
                    if (compute_topology == ComputeTopology::ThreeWayFork)
                    {
                        // Every event is layer-owned: reusing a later record's
                        // identity must not accidentally close a graph cycle.
                        if (!p.context->recordEventChecked(p.fork_ready[layer], p.inference))
                            throw std::runtime_error("Compute fork publication failed");
                        for (std::size_t fork = 0; fork < p.compute.size(); ++fork)
                        {
                            if (!p.context->waitEventChecked(p.fork_ready[layer], p.compute[fork]) ||
                                !backend->copyDeviceVisibleRegionByKernelOnStream(
                                    p.fork_outputs[fork]->mutableDeviceData(), p.source->deviceData(),
                                    kCopyBytes, p.device.gpu_ordinal(), p.compute[fork]) ||
                                !p.context->recordEventChecked(p.fork_done[layer][fork], p.compute[fork]) ||
                                !p.context->waitEventChecked(p.fork_done[layer][fork], p.inference))
                                throw std::runtime_error("Compute branch publication failed");
                        }
                    }
                    if (!collective->allreduceOnStream(p.layers[layer].get(),
                            "captured_maintenance_layer_" + std::to_string(layer),
                            kElements, p.inference))
                        throw std::runtime_error("Captured collective submission failed");
                    transfers.enqueueMappedTimelinePublish64(*p.timeline, 0u,
                        layer + 1u, p.device, p.inference);
                    transfers.enqueueMappedTimelineWait64(*p.timeline, 64u,
                        layer + 1u, p.device, p.inference);
                }
                capture.finish();
                if (!p.graph->instantiate()) throw std::runtime_error("Collective graph instantiate failed");
            });
        });

        // This is a test-owned remote peer, not a host mirror of GPU model state.
        auto release = [&](void *) {
            for (auto &p : participants)
                std::atomic_ref<std::uint64_t>(*static_cast<std::uint64_t *>(
                    p.timeline->mutableHostData(64u))).store(kLayers, std::memory_order_release);
            runParticipants(degree, [&](std::size_t index) {
                auto &p = participants[index];
                p.context->submitAndWait([&] {
                    p.context->synchronizeStream(p.inference);
                    for (const auto &lane : p.execution)
                        p.context->synchronizeStream(lane.stream());
                    p.maintenance_graphs.clear();
                    for (void *event : p.maintenance_done) p.context->destroyEvent(event);
                    p.maintenance_done.clear();
                    for (void *observer : p.observers)
                        p.context->synchronizeStream(observer);
                    if (p.terminal)
                    {
                        p.context->destroyEvent(p.terminal);
                        p.terminal = nullptr;
                    }
                    // Release captured communicator references while their
                    // collective owner and every embedded allocation are live.
                    p.graph.reset();
                    for (void *event : p.fork_ready) p.context->destroyEvent(event);
                    p.fork_ready.clear();
                    for (const auto &events : p.fork_done)
                        for (void *event : events) p.context->destroyEvent(event);
                    p.fork_done.clear();
                });
            });
        };
        std::unique_ptr<void, decltype(release)> release_guard(&participants, release);
        runParticipants(degree, [&](std::size_t index) {
            auto &p = participants[index];
            p.context->submitAndWait([&] {
                if (!p.graph->launch()) throw std::runtime_error("Collective graph launch failed");
                if (p.terminal)
                {
                    if (!p.context->recordEventChecked(p.terminal, p.inference))
                        throw std::runtime_error("Terminal publication failed");
                    for (std::size_t lane = 0; lane < p.observers.size(); ++lane)
                    {
                        // A real kernel makes the wait visible to the runtime;
                        // these readers must never order earlier inference work.
                        if (!p.context->waitEventChecked(p.terminal, p.observers[lane]) ||
                            !backend->copyDeviceVisibleRegionByKernelOnStream(
                                p.observations->deviceAlias(p.device, lane * 128u),
                                p.source->deviceData(), 128u, p.device.gpu_ordinal(),
                                p.observers[lane]))
                            throw std::runtime_error("Future reader submission failed");
                    }
                }
            });
        });
        const auto submit_maintenance = [&](std::size_t index) {
            auto &p = participants[index];
            p.context->submitAndWait([&] {
                for (std::size_t lane = 0; lane < kLanes; ++lane)
                {
                    // An unused native event is ready. On later layers, only
                    // the exact completed receipt releases that lane's staging.
                    bool ready = false;
                    if (!p.context->queryEventChecked(p.maintenance_done[lane], ready))
                        throw std::runtime_error("Maintenance completion query failed");
                    if (!ready) continue;
                    if (p.maintenance_submissions[lane] != 0u)
                        ++p.maintenance_completions_during_inference;
                    const bool submitted = producer_admission == ProducerAdmission::RetainedGraph
                        ? p.maintenance_graphs[lane]->launch()
                        : backend->copyDeviceVisibleRegionByKernelOnStream(
                              p.produced[lane]->mutableDeviceData(), p.source->deviceData(),
                              kCopyBytes, p.device.gpu_ordinal(), p.execution[lane].stream()) &&
                          transfers.enqueueBackgroundStagingCopy(p.execution[lane],
                              MappedTransferDirection::DeviceToHost,
                              p.produced[lane]->mutableDeviceData(), kCopyBytes, 0u,
                              p.staging[lane], 0u, kCopyBytes);
                    if (submitted && producer_admission == ProducerAdmission::Native)
                        transfers.enqueueMappedTimelinePublish64(*p.maintenance_publications,
                            lane * 64u, p.maintenance_submissions[lane] + 1u,
                            p.device, p.execution[lane].stream());
                    if (!submitted ||
                        !p.context->recordEventChecked(p.maintenance_done[lane], p.execution[lane].stream()))
                        throw std::runtime_error("Concurrent maintenance submission failed");
                    ++p.maintenance_submissions[lane];
                }
            });
        };
        runParticipants(degree, submit_maintenance);
        for (std::size_t layer = 0; layer < kLayers; ++layer)
        {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            bool arrived = false;
            do
            {
                arrived = true;
                for (auto &p : participants)
                    arrived = (std::atomic_ref<std::uint64_t>(*static_cast<std::uint64_t *>(
                        p.timeline->mutableHostData())).load(std::memory_order_acquire) >= layer + 1u) && arrived;
                if (!arrived) std::this_thread::yield();
            } while (!arrived && std::chrono::steady_clock::now() < deadline);
            ASSERT_TRUE(arrived) << "Inference stopped at CPU rendezvous " << layer;
            // Keep maintenance active throughout the captured transaction,
            // rather than proving only a burst which finishes before layer 2.
            runParticipants(degree, submit_maintenance);
            for (auto &p : participants)
                std::atomic_ref<std::uint64_t>(*static_cast<std::uint64_t *>(
                    p.timeline->mutableHostData(64u))).store(layer + 1u, std::memory_order_release);
        }
        for (const auto &p : participants)
        {
            std::size_t published = 0u;
            for (std::size_t lane = 0; lane < kLanes; ++lane)
                published += std::atomic_ref<std::uint64_t>(*static_cast<std::uint64_t *>(
                    p.maintenance_publications->mutableHostData(lane * 64u)))
                    .load(std::memory_order_acquire) != 0u;
            std::cout << "[CAPTURED_COLLECTIVE_MAINTENANCE] device=" << p.device.toString()
                      << " completions_during_inference="
                      << p.maintenance_completions_during_inference
                      << " lanes_with_device_publication=" << published << '/' << kLanes << '\n';
        }
        release_guard.reset();
        const float expected = static_cast<float>(degree * (degree + 1u) / 2u);
        for (auto &p : participants)
        {
            p.context->submitAndWait([&] {
                std::vector<float> output(kElements);
                for (auto &layer : p.layers)
                {
                    if (!backend->deviceToHostOnStream(output.data(), layer->gpu_data_ptr(),
                            output.size() * sizeof(float), p.device.gpu_ordinal(), p.inference))
                        throw std::runtime_error("Collective result observation failed");
                    p.context->synchronizeStream(p.inference); // Terminal diagnostic readback only.
                    for (float value : output) EXPECT_EQ(value, expected);
                }
            });
            const std::vector<std::uint8_t> expected_bytes(kCopyBytes, 0x5a);
            for (const auto &slice : p.staging)
            {
                ASSERT_EQ(std::memcmp(slice.mutablePinnedData(), expected_bytes.data(),
                                      expected_bytes.size()), 0) << p.device.toString();
            }
            if (p.observations)
                ASSERT_EQ(std::memcmp(p.observations->mutableHostData(), expected_bytes.data(),
                                      kLanes * 128u), 0) << p.device.toString();
        }
    }
} // namespace

#ifdef HAVE_CUDA
/** CUDA's production collective receives the same CPU/maintenance interaction proof. */
TEST(CapturedCollectiveMaintenanceProgress, CUDA2)
{ proveCapturedCollectiveMaintenance(DeviceId::cuda(0), 2u); }

/** Prefix-style future readers must not close a cycle through CUDA collectives. */
TEST(CapturedCollectiveMaintenanceProgress, CUDA2FutureConsumers)
{ proveCapturedCollectiveMaintenance(DeviceId::cuda(0), 2u, OutputReaders::FutureConsumers); }

/** Independent stage stream pools remain live alongside collective maintenance. */
TEST(CapturedCollectiveMaintenanceProgress, CUDA2ConcurrentStages)
{ proveCapturedCollectiveMaintenance(DeviceId::cuda(0), 2u, OutputReaders::FutureConsumers,
                                    ComputeTopology::ThreeWayFork); }

/** Retained producer control isolates native queue admission from payload work. */
TEST(CapturedCollectiveMaintenanceProgress, CUDA2RetainedProducers)
{ proveCapturedCollectiveMaintenance(DeviceId::cuda(0), 2u, OutputReaders::FutureConsumers,
                                    ComputeTopology::ThreeWayFork, ProducerAdmission::RetainedGraph); }
#endif
#ifdef HAVE_ROCM
/** Two-device control matches the already-green 122B ROCm/CPU topology width. */
TEST(CapturedCollectiveMaintenanceProgress, ROCm2)
{ proveCapturedCollectiveMaintenance(DeviceId::rocm(0), 2u); }

/** Four-device sentinel matches the failing published-image topology width. */
TEST(CapturedCollectiveMaintenanceProgress, ROCm4)
{ proveCapturedCollectiveMaintenance(DeviceId::rocm(0), 4u); }

/** Prefix-style future readers must not close a cycle through RCCL collectives. */
TEST(CapturedCollectiveMaintenanceProgress, ROCm4FutureConsumers)
{ proveCapturedCollectiveMaintenance(DeviceId::rocm(0), 4u, OutputReaders::FutureConsumers); }

/** HIP's captured parallel branches retain progress with all native queues populated. */
TEST(CapturedCollectiveMaintenanceProgress, ROCm4ConcurrentStages)
{ proveCapturedCollectiveMaintenance(DeviceId::rocm(0), 4u, OutputReaders::FutureConsumers,
                                    ComputeTopology::ThreeWayFork); }
#endif
} // namespace llaminar2::test
