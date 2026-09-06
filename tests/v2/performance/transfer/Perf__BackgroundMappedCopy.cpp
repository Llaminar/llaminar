/**
 * @file Perf__BackgroundMappedCopy.cpp
 * @brief Production background-copy and retained CPU-ticket economy on GPUs.
 *
 * Persistent storage, function preparation and warmup precede event timing.
 * Every sample uses the same direction, byte extent and exact stream. The
 * progress path is the public background-transfer API, not a benchmark kernel.
 * This performance suite deliberately does not belong to parity preflight.
 */
#include <gtest/gtest.h>

#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IWorkerGPUContext.h"
#include "backends/IGPUGraphCapture.h"
#include "execution/moe/MoEOverlaySparseCollective.h"
#ifdef HAVE_CUDA
#include "kernels/cuda/moe/CUDAMoEKernel.h"
#endif
#ifdef HAVE_ROCM
#include "kernels/rocm/moe/ROCmMoEKernel.h"
#endif
#include "transfer/TransferEngine.h"
#include "transfer/MappedTransferProgressEpoch.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

namespace llaminar2
{
namespace
{
    /** Explicit public entrypoints, never a runtime transport retry policy. */
    enum class CopySample { DMA, BackgroundProgress, CPUStagingProgress };

    /**
     * @brief Measure one exact resumable-service geometry in isolation.
     * @param direction Fixed upload/download role selected by the test name.
     *
     * Four simultaneously published 4 MiB commands exercise all service CTAs.
     * CUDA's finite idle pass and captured intervals share this exact kernel,
     * cursor and receipt protocol. Kernel active time comes from GPU receipts;
     * wall time includes public publication/progression costs. These are separate
     * measurements, never substituted for each other. Run either test alone
     * under Nsight for an uncontaminated kernel resource/throughput record.
     */
    void measureBoundedService(MappedTransferDirection direction)
    {
        const auto device = DeviceId::cuda(0);
        auto *backend = getBackendFor(device);
        ASSERT_NE(backend, nullptr);
        if (!backend->deviceCount()) GTEST_SKIP();
        constexpr std::size_t bytes = 4u * 1024u * 1024u;
        constexpr std::size_t capacity = 4u;
        constexpr std::size_t warmups = 5u;
        constexpr std::size_t samples = 25u;
        auto &context = GPUDeviceContextPool::instance().getContext(device);
        TransferEngine engine;
        const DeviceId devices[] = {device};
        auto mapped = engine.allocateMappedHostTransferSlices(bytes, capacity, device);
        auto storage = engine.allocateDeviceTransferBuffer(bytes * capacity, device);
        auto observation = engine.allocateMappedHostRegion(bytes * capacity, devices);
        auto epoch = MappedTransferProgressEpoch::create({
            .device = device, .slot_capacity = capacity, .execution_lane_capacity = capacity,
            .execution_streams = engine.allocatePersistentTransferExecutionLanes(
                1u, device, "bounded_service_economy"),
            .maximum_bytes = bytes, .name = "bounded_service_economy"});
        std::vector<MappedTransferProgressSlot> commands;
        for (auto &region : mapped) commands.push_back(epoch->reserveSlot(direction, region));
        std::array<double, samples> wall_us{}, active_us{};
        void *setup = nullptr;
        void *ready = nullptr;
        context.submitAndWait([&] {
            setup = context.getOrCreateAuxiliaryStream("bounded_service_economy_setup");
            ready = context.createEvent();
            if (!setup || !ready) throw std::runtime_error("Service economy setup failed");
        });
        for (std::size_t sample = 0u; sample < warmups + samples; ++sample)
        {
            const auto value = static_cast<unsigned char>(sample + 1u);
            // A new payload each generation prevents stale receipts/data from
            // passing byte certification. Preparation is outside both timers.
            std::memset(observation->mutableHostData(), value, bytes * capacity);
            for (auto &region : mapped)
                std::memset(region->mutableHostData(),
                    direction == MappedTransferDirection::DeviceToHost ? 0 : value, bytes);
            if (direction == MappedTransferDirection::DeviceToHost)
                context.submitAndWait([&] {
                    engine.enqueueMappedHostToPersistentDeviceRegion(*observation, 0u,
                        storage->mutableDeviceData(), bytes * capacity, 0u,
                        bytes * capacity, device, setup);
                    if (!context.recordEventChecked(ready, setup) || !context.synchronizeEventChecked(ready))
                        throw std::runtime_error("Service economy source setup failed");
                });
            const auto started = std::chrono::steady_clock::now();
            for (std::size_t slot = 0u; slot < capacity; ++slot)
                if (direction == MappedTransferDirection::DeviceToHost)
                    commands[slot].publishDeviceToMappedHost(storage->deviceData(),
                        bytes * capacity, slot * bytes, bytes);
                else
                    commands[slot].publishMappedHostToDevice(storage->mutableDeviceData(),
                        bytes * capacity, slot * bytes, bytes);
            std::array<bool, capacity> complete{};
            std::uint64_t longest_active = 0u;
            std::size_t completed = 0u;
            const auto deadline = started + std::chrono::seconds(10);
            while (completed != capacity && std::chrono::steady_clock::now() < deadline)
            {
                if (!epoch->submitOutstandingProgress())
                    throw std::runtime_error("Service economy progress failed");
                for (std::size_t slot = 0u; slot < capacity; ++slot)
                    if (!complete[slot])
                    {
                        std::string error;
                        const auto progress = commands[slot].poll(&error);
                        if (progress == MappedTransferProgress::Failed) throw std::runtime_error(error);
                        if (progress == MappedTransferProgress::Ready)
                        {
                            const auto measured = commands[slot].completedDeviceNanoseconds();
                            if (!measured) throw std::runtime_error("Service completion lacks active timing");
                            longest_active = std::max(longest_active, *measured);
                            complete[slot] = true;
                            ++completed;
                        }
                    }
                if (completed != capacity) std::this_thread::yield();
            }
            if (completed != capacity) throw std::runtime_error("Service economy command did not complete");
            const auto elapsed = std::chrono::duration<double, std::micro>(
                std::chrono::steady_clock::now() - started).count();
            if (sample >= warmups)
            {
                wall_us[sample - warmups] = elapsed;
                active_us[sample - warmups] = static_cast<double>(longest_active) / 1000.0;
            }
            if (direction == MappedTransferDirection::HostToDevice)
                context.submitAndWait([&] {
                    engine.enqueuePersistentDeviceRegionToMappedHost(storage->deviceData(),
                        bytes * capacity, 0u, *observation, 0u, bytes * capacity, device, setup);
                    if (!context.recordEventChecked(ready, setup) || !context.synchronizeEventChecked(ready))
                        throw std::runtime_error("Service economy diagnostic readback failed");
                });
            for (std::size_t slot = 0u; slot < capacity; ++slot)
            {
                const auto *actual = static_cast<const unsigned char *>(
                    direction == MappedTransferDirection::DeviceToHost
                        ? mapped[slot]->mutableHostData()
                        : observation->mutableHostData(slot * bytes));
                EXPECT_TRUE(std::all_of(actual, actual + bytes,
                    [value](unsigned char byte) { return byte == value; }));
            }
        }
        std::sort(wall_us.begin(), wall_us.end());
        std::sort(active_us.begin(), active_us.end());
        std::printf("SERVICE_ECONOMY,%s,slots=%zu,bytes_per_slot=%zu,active_us=%.3f,wall_us=%.3f,wall_GBps=%.3f\n",
            direction == MappedTransferDirection::DeviceToHost ? "d2h" : "h2d", capacity, bytes,
            active_us[samples / 2u], wall_us[samples / 2u],
            bytes * capacity / (wall_us[samples / 2u] * 1000.0));
        context.submitAndWait([&] { context.destroyEvent(ready); });
    }

    /**
     * @brief Compare exact public transfer entrypoints with terminal byte checks.
     * @param device Exact backend/device used for every allocation and stream.
     * @param direction Immutable upload/download direction of this test.
     */
    void measureCopies(DeviceId device, MappedTransferDirection direction)
    {
        auto *backend = getBackendFor(device);
        ASSERT_NE(backend, nullptr);
        if (backend->deviceCount() <= device.gpu_ordinal()) GTEST_SKIP();
        TransferEngine engine;
        const auto lanes = engine.allocatePersistentTransferExecutionLanes(
            1u, device, "background_copy_economy");
        const auto &lane = lanes.front();
        auto &context = GPUDeviceContextPool::instance().getContext(device);
        const std::array devices{device};
        constexpr std::size_t capacity = 4u * 1024u * 1024u + 13u;
        auto mapped = engine.allocateMappedHostRegion(capacity, devices);
        auto storage = engine.allocateDeviceTransferBuffer(capacity, device);
        const auto staging = engine.allocatePersistentTransferStagingSlices(
            capacity, 1u, device);
        std::vector<unsigned char> expected(capacity);
        for (std::size_t i = 0; i < capacity; ++i)
            expected[i] = static_cast<unsigned char>((i * 43u + (i >> 3u)) & 255u);

        context.submitAndWait([&] {
            void *start = context.createEvent();
            void *end = context.createEvent();
            if (!start || !end) throw std::runtime_error("copy benchmark event setup failed");
            for (const std::size_t bytes : {std::size_t{4096}, std::size_t{196608}, capacity - 13u, capacity})
            for (const auto path : {CopySample::DMA, CopySample::BackgroundProgress,
                                   CopySample::CPUStagingProgress})
            {
                // Use identical bytes in every candidate. Diagnostic readback
                // occurs only after the terminal timing event has completed.
                std::memcpy(mapped->mutableHostData(), expected.data(), capacity);
                std::memcpy(staging.front().mutablePinnedData(), expected.data(), capacity);
                engine.enqueueMappedHostToPersistentDeviceRegion(*mapped, 0u,
                    storage->mutableDeviceData(), capacity, 0u, capacity, device, lane.stream());
                auto enqueue = [&] {
                    if (path == CopySample::CPUStagingProgress)
                    {
                        std::string error;
                        if (!engine.enqueueBackgroundStagingCopy(lane, direction,
                                storage->mutableDeviceData(), capacity, 0u,
                                staging.front(), 0u, bytes, &error))
                            throw std::runtime_error(error);
                    }
                    else if (path == CopySample::BackgroundProgress)
                        engine.enqueueBackgroundMappedCopy(lane, direction,
                            storage->mutableDeviceData(), capacity, 0u, *mapped, 0u, bytes);
                    else if (direction == MappedTransferDirection::DeviceToHost)
                        engine.enqueuePersistentDeviceRegionToMappedHost(storage->deviceData(),
                            capacity, 0u, *mapped, 0u, bytes, device, lane.stream());
                    else
                        engine.enqueueMappedHostToPersistentDeviceRegion(*mapped, 0u,
                            storage->mutableDeviceData(), capacity, 0u, bytes, device, lane.stream());
                };
                for (unsigned i = 0; i < 5u; ++i) enqueue();
                std::array<float, 5> timings{};
                constexpr unsigned repetitions = 100u;
                for (auto &timing : timings)
                {
                    if (!context.recordEventChecked(start, lane.stream()))
                        throw std::runtime_error("copy benchmark start recording failed");
                    for (unsigned i = 0; i < repetitions; ++i) enqueue();
                    if (!context.recordEventChecked(end, lane.stream()) ||
                        !context.synchronizeEventChecked(end))
                        throw std::runtime_error("copy benchmark terminal event failed");
                    timing = context.eventElapsedTime(start, end) / repetitions;
                    if (!(timing > 0.0f)) throw std::runtime_error("invalid copy event duration");
                }
                std::sort(timings.begin(), timings.end());
                const float median_ms = timings[timings.size() / 2u];
                std::printf("COPY_ECONOMY,%s,%s,%s,%zu,%.3f,%.3f\n",
                    device.toString().c_str(),
                    direction == MappedTransferDirection::DeviceToHost ? "d2h" : "h2d",
                    path == CopySample::DMA ? "dma" :
                        path == CopySample::BackgroundProgress ? "background_progress" :
                                                               "cpu_staging_progress", bytes,
                    median_ms * 1000.0, bytes / (median_ms * 1e6));
                if (direction == MappedTransferDirection::HostToDevice)
                {
                    engine.enqueuePersistentDeviceRegionToMappedHost(storage->deviceData(),
                        capacity, 0u, *mapped, 0u, bytes, device, lane.stream());
                    if (!context.recordEventChecked(end, lane.stream()) ||
                        !context.synchronizeEventChecked(end))
                        throw std::runtime_error("copy benchmark readback failed");
                }
                const void *actual = path == CopySample::CPUStagingProgress &&
                        direction == MappedTransferDirection::DeviceToHost
                    ? staging.front().mutablePinnedData() : mapped->mutableHostData();
                EXPECT_EQ(std::memcmp(actual, expected.data(), bytes), 0);
            }
            context.destroyEvent(end);
            context.destroyEvent(start);
        });
    }

    /**
     * @brief Time the complete retained CPU-ticket consumer with ready payloads.
     * @param device Exact continuation backend, independent of tier topology.
     *
     * Publication and setup are outside event timing. Every sample reuses the
     * production sequence/acknowledgement protocol and one retained graph; no
     * synthetic wait is included in the economy number. Functional held-ticket
     * progress and adversarial replay are covered by the integration fixture.
     */
    void measureCanonicalTickets(DeviceId device)
    {
        auto *backend = getBackendFor(device);
        ASSERT_NE(backend, nullptr);
        if (backend->deviceCount() <= device.gpu_ordinal()) GTEST_SKIP();
        auto &context = GPUDeviceContextPool::instance().getContext(device);
        context.submitAndWait([&] {
            void *stream = context.getOrCreateAuxiliaryStream("canonical_ticket_economy");
            void *start = context.createEvent();
            void *end = context.createEvent();
            ASSERT_NE(stream, nullptr);
            ASSERT_NE(start, nullptr);
            ASSERT_NE(end, nullptr);
            auto retire_events = [&](void *) {
                context.destroyEvent(end);
                context.destroyEvent(start);
            };
            std::unique_ptr<void, decltype(retire_events)> events(&context, retire_events);
            TransferEngine engine;
            const std::array devices{device};
            constexpr int width = 3072;
            for (const std::size_t routes : {8u, 512u, 4800u})
            {
                const std::size_t bytes = routes * width * sizeof(float);
                auto payload = engine.allocateMappedHostRegion(bytes, devices);
                auto output = engine.allocateDeviceTransferBuffer(bytes, device);
                auto metadata = engine.createMappedHostArena(devices);
                MoEOverlayCanonicalRouteReturnTicketStorage storage;
                storage.bindFixedCapacity(5, routes, width, device, 17u, payload, metadata);
                std::unique_ptr<IMoEKernel> kernel;
#ifdef HAVE_CUDA
                if (device.is_cuda()) kernel = std::make_unique<CUDAMoEKernel>(device.ordinal);
#endif
#ifdef HAVE_ROCM
                if (device.is_rocm()) kernel = std::make_unique<ROCmMoEKernel>(device.ordinal);
#endif
                ASSERT_NE(kernel, nullptr);
                MoEOverlayCanonicalRouteTicketConsumeLaunch launch{
                    .control = storage.controlDeviceAlias(),
                    .original_route_slots = storage.originalRouteSlotsDeviceAlias(),
                    .compact_route_slots = storage.compactRouteSlotsDeviceAlias(),
                    .compact_preweighted_contributions_fp32 = storage.contributionRowsDeviceAlias(),
                    .canonical_route_contributions_fp32 = static_cast<float *>(output->mutableDeviceData()),
                    .route_capacity = routes,
                    .d_model = width,
                };
                // Fill only while a CPU publication lease owns the mapped rows.
                auto initial = storage.arm(1u);
                ASSERT_TRUE(initial);
                for (std::size_t row = 0; row < routes; ++row)
                {
                    storage.originalRouteSlotsHost()[row] = static_cast<int>(row);
                    storage.compactRouteSlotsHost()[row] = static_cast<int>(row);
                }
                std::fill_n(storage.contributionRowsHost(), routes * width, 0.125f);
                ASSERT_TRUE(initial.publish(routes));
                ASSERT_TRUE(kernel->consumeMoEOverlayCanonicalRouteTicket({.stream = stream}, launch));
                ASSERT_TRUE(context.recordEventChecked(end, stream));
                ASSERT_TRUE(context.synchronizeEventChecked(end));
                auto graph = context.createGraphCapture(stream);
                ASSERT_TRUE(graph->beginCapture());
                ASSERT_TRUE(kernel->consumeMoEOverlayCanonicalRouteTicket({.stream = stream}, launch));
                ASSERT_TRUE(graph->endCapture());
                ASSERT_TRUE(graph->instantiate());
                auto drain_graph = [&](void *) {
                    // Failed timing assertions must not retire mapped bytes
                    // while an already accepted consumer still references them.
                    (void)storage.publishAbort();
                    (void)backend->synchronizeStream(stream, device.ordinal);
                };
                std::unique_ptr<void, decltype(drain_graph)> graph_guard(&storage, drain_graph);
                std::array<float, 50> timings{};
                for (std::size_t sample = 0; sample < timings.size() + 5u; ++sample)
                {
                    auto publication = storage.arm(sample + 2u);
                    ASSERT_TRUE(publication);
                    ASSERT_TRUE(publication.publish(routes));
                    ASSERT_TRUE(context.recordEventChecked(start, stream));
                    ASSERT_TRUE(graph->launchOnStream(stream));
                    ASSERT_TRUE(context.recordEventChecked(end, stream));
                    ASSERT_TRUE(context.synchronizeEventChecked(end));
                    ASSERT_TRUE(storage.publicationSucceededFor(sample + 2u));
                    if (sample >= 5u)
                        timings[sample - 5u] = context.eventElapsedTime(start, end);
                }
                std::sort(timings.begin(), timings.end());
                const float median_ms = timings[timings.size() / 2u];
                std::printf("TICKET_ECONOMY,%s,%zu,%.3f,%.3f\n",
                    device.toString().c_str(), routes, median_ms * 1000.0,
                    bytes / (median_ms * 1e6));
                std::vector<float> actual(routes * width);
                ASSERT_TRUE(backend->deviceToHost(actual.data(), output->deviceData(), bytes,
                    device.ordinal, stream));
                EXPECT_EQ(std::memcmp(actual.data(), payload->mutableHostData(), bytes), 0);
            }
        });
    }

#if defined(HAVE_CUDA)
    TEST(BoundedTransferServiceEconomy, CUDADownload4MiB)
    { measureBoundedService(MappedTransferDirection::DeviceToHost); }
    TEST(BoundedTransferServiceEconomy, CUDAUpload4MiB)
    { measureBoundedService(MappedTransferDirection::HostToDevice); }
    TEST(BackgroundMappedCopyEconomy, CUDADownload)
    { measureCopies(DeviceId::cuda(0), MappedTransferDirection::DeviceToHost); }
    TEST(BackgroundMappedCopyEconomy, CUDAUpload)
    { measureCopies(DeviceId::cuda(0), MappedTransferDirection::HostToDevice); }
    TEST(CanonicalRouteTicketEconomy, CUDA)
    { measureCanonicalTickets(DeviceId::cuda(0)); }
#endif
#if defined(HAVE_ROCM)
    TEST(BackgroundMappedCopyEconomy, ROCmDownload)
    { measureCopies(DeviceId::rocm(0), MappedTransferDirection::DeviceToHost); }
    TEST(BackgroundMappedCopyEconomy, ROCmUpload)
    { measureCopies(DeviceId::rocm(0), MappedTransferDirection::HostToDevice); }
    TEST(CanonicalRouteTicketEconomy, ROCm)
    { measureCanonicalTickets(DeviceId::rocm(0)); }
#endif
} // namespace
} // namespace llaminar2
