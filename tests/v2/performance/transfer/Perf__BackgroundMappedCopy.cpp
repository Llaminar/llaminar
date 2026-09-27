/**
 * @file Perf__BackgroundMappedCopy.cpp
 * @brief Production background-copy, retained ticket and sparse return economy.
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
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
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
     * This measures the finite-pass launch geometry only; a captured worker
     * has a different concurrency budget and is measured separately below.
     * Kernel active time comes from GPU receipts;
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
     * @brief Measure the actual captured copy worker, not its finite-pass sibling.
     * @param direction Exact upload or download direction for every sample.
     * @param capacity Physical inbox width, fixed for this profiler/test launch.
     *
     * A captured primary branch publishes entry and waits for this fixture to
     * release it. Commands arrive only after entry, so independently submitted
     * finite passes cannot accidentally supply the measured progress. All
     * copies and graph retirement use the production TransferEngine entrypoints.
     * The hold is a diagnostic liveness fixture, not simulated inference: the
     * reported bandwidth does not claim to measure interference with GEMMs.
     * Wall time ends at the last acquired receipt; retirement is timed separately.
     */
    void measureCapturedService(MappedTransferDirection direction, std::size_t capacity)
    {
        const auto device = DeviceId::cuda(0);
        auto *backend = getBackendFor(device);
        ASSERT_NE(backend, nullptr);
        if (!backend->deviceCount()) GTEST_SKIP();
        auto &context = GPUDeviceContextPool::instance().getContext(device);
        context.submitAndWait([&]
        {
            TransferEngine engine;
            const DeviceId devices[] = {device};
            constexpr std::size_t bytes = 4u * 1024u * 1024u;
            constexpr std::size_t warmups = 3u;
            constexpr std::size_t samples = 9u;
            {
                auto payload = engine.allocateMappedHostRegion(bytes * capacity, devices);
                auto storage = engine.allocateDeviceTransferBuffer(bytes * capacity, device);
                auto inbox = engine.allocateMappedHostRegion(capacity *
                    (sizeof(MappedTransferProgressCommand) + sizeof(MappedTransferProgressCompletion)), devices);
                auto controls = engine.allocateMappedHostRegion(2u * sizeof(std::uint64_t), devices);
                auto wake = engine.allocateMappedHostRegion(sizeof(std::uint64_t), devices);
                auto *commands = static_cast<MappedTransferProgressCommand *>(inbox->mutableHostData());
                auto *receipts = reinterpret_cast<MappedTransferProgressCompletion *>(commands + capacity);
                auto *control = static_cast<std::uint64_t *>(controls->mutableHostData());
                std::fill_n(commands, capacity, MappedTransferProgressCommand{});
                std::fill_n(receipts, capacity, MappedTransferProgressCompletion{});
                void *primary = context.getOrCreateAuxiliaryStream("captured_copy_economy_primary");
                void *worker = context.getOrCreateAuxiliaryStream("captured_copy_economy_worker",
                    GPUAuxiliaryStreamSchedulingClass::BackgroundMaintenance);
                void *fork = context.createEvent();
                void *worker_done = context.createEvent();
                void *terminal = context.createEvent();
                if (!primary || !worker || !fork || !worker_done || !terminal)
                    throw std::runtime_error("Captured copy economy setup failed");
                auto destroy_events = [&](void *)
                {
                    context.destroyEvent(terminal);
                    context.destroyEvent(worker_done);
                    context.destroyEvent(fork);
                };
                std::unique_ptr<void, decltype(destroy_events)> event_owner(&context, destroy_events);
                auto cursors = engine.allocateMappedTransferServiceCursors(capacity, device, primary);
                if (!context.recordEventChecked(terminal, primary) || !context.synchronizeEventChecked(terminal))
                    throw std::runtime_error("Captured copy cursor preparation failed");
                auto graph = context.createGraphCapture(primary);
                if (!graph || !graph->beginCapture())
                    throw std::runtime_error("Captured copy economy recording failed");
                engine.enqueueMappedTransferWakeState(*wake, MappedTransferWakeState::InferenceActive,
                    device, primary);
                if (!context.recordEventChecked(fork, primary) || !context.waitEventChecked(fork, worker))
                    throw std::runtime_error("Captured copy economy fork failed");
                engine.enqueueMappedTransferService(*inbox, *cursors, bytes, wake.get(),
                    MappedTransferServiceRun::CapturedInterval, worker);
                if (!context.recordEventChecked(worker_done, worker))
                    throw std::runtime_error("Captured copy worker terminal failed");
                engine.enqueueMappedTimelinePublish64(*controls, sizeof(std::uint64_t), 1u, device, primary);
                if (!backend->streamWaitTimelineSignal64(primary, controls->deviceAlias(device),
                        1u, device.gpu_ordinal()))
                    throw std::runtime_error("Captured copy economy hold failed");
                engine.enqueueMappedTransferWakeState(*wake, MappedTransferWakeState::InferenceComplete,
                    device, primary);
                if (!context.waitEventChecked(worker_done, primary) || !graph->endCapture() || !graph->instantiate())
                    throw std::runtime_error("Captured copy economy close/join failed");
                // Even a deadline/byte failure must first release the held graph.
                // Otherwise unwinding would free memory still referenced on GPU.
                auto release_graph = [&]() -> bool
                {
                    std::atomic_ref<std::uint64_t>(control[0]).store(1u, std::memory_order_release);
                    return context.recordEventChecked(terminal, primary) &&
                        context.synchronizeEventChecked(terminal);
                };
                auto unwind_graph = [&](void *) { (void)release_graph(); };
                std::unique_ptr<void, decltype(unwind_graph)> graph_owner(graph.get(), unwind_graph);
                std::array<double, samples> wall_us{}, active_sum_us{}, retire_us{};
                for (std::size_t sample = 0u; sample < samples + warmups; ++sample)
                {
                    // Distinguish both slots and generations: a wrong offset,
                    // duplicate slot, or stale receipt must not pass the check.
                    const auto expected = [sample](std::size_t slot)
                    { return static_cast<unsigned char>(sample + 1u + slot * 17u); };
                    for (std::size_t slot = 0u; slot < capacity; ++slot)
                        std::memset(payload->mutableHostData(slot * bytes),
                            direction == MappedTransferDirection::HostToDevice ? 0u : expected(slot), bytes);
                    engine.enqueueMappedHostToPersistentDeviceRegion(*payload, 0u,
                        storage->mutableDeviceData(), bytes * capacity, 0u, bytes * capacity, device, primary);
                    if (!context.recordEventChecked(terminal, primary) || !context.synchronizeEventChecked(terminal))
                        throw std::runtime_error("Captured copy sample setup failed");
                    for (std::size_t slot = 0u; slot < capacity; ++slot)
                        std::memset(payload->mutableHostData(slot * bytes),
                            direction == MappedTransferDirection::DeviceToHost ? 0u : expected(slot), bytes);
                    const std::uint64_t generation = sample + 1u;
                    for (std::size_t slot = 0u; slot < capacity; ++slot)
                    {
                        auto &command = commands[slot];
                        command.generation_magic = mappedTransferProgressGenerationMagic(generation);
                        command.generation_version = mappedTransferProgressGenerationVersion(generation);
                        command.source_address = reinterpret_cast<std::uintptr_t>(
                            direction == MappedTransferDirection::HostToDevice
                                ? payload->deviceAlias(device, slot * bytes) : storage->deviceData(slot * bytes));
                        command.destination_address = reinterpret_cast<std::uintptr_t>(
                            direction == MappedTransferDirection::DeviceToHost
                                ? payload->deviceAlias(device, slot * bytes) : storage->mutableDeviceData(slot * bytes));
                        command.bytes = bytes;
                        command.source_complement = ~command.source_address;
                        command.destination_complement = ~command.destination_address;
                        command.bytes_complement = ~command.bytes;
                    }
                    std::atomic_ref<std::uint64_t>(control[0]).store(0u, std::memory_order_release);
                    std::atomic_ref<std::uint64_t>(control[1]).store(0u, std::memory_order_release);
                    if (!graph->launchOnStream(primary))
                        throw std::runtime_error("Captured copy economy replay failed");
                    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
                    while (std::atomic_ref<std::uint64_t>(control[1]).load(std::memory_order_acquire) != 1u)
                    {
                        if (std::chrono::steady_clock::now() >= deadline)
                            throw std::runtime_error("Captured copy did not reach its admission point");
                        std::this_thread::yield();
                    }
                    const auto started = std::chrono::steady_clock::now();
                    for (std::size_t slot = 0u; slot < capacity; ++slot)
                        std::atomic_ref<std::uint64_t>(commands[slot].generation)
                            .store(generation, std::memory_order_release);
                    for (std::size_t slot = 0u; slot < capacity; ++slot)
                    {
                        while (std::atomic_ref<std::uint64_t>(receipts[slot].completed_generation)
                                   .load(std::memory_order_acquire) != generation)
                        {
                            if (std::chrono::steady_clock::now() >= deadline)
                                throw std::runtime_error("Captured copy did not finish before primary release");
                            std::this_thread::yield();
                        }
                    }
                    const auto completed = std::chrono::steady_clock::now();
                    if (!release_graph())
                        throw std::runtime_error("Captured copy graph retirement failed");
                    const auto retired = std::chrono::steady_clock::now();
                    std::uint64_t active_ns = 0u;
                    for (std::size_t slot = 0u; slot < capacity; ++slot)
                    {
                        EXPECT_EQ(receipts[slot].error, 0u);
                        EXPECT_EQ(receipts[slot].completed_bytes, bytes);
                        active_ns += receipts[slot].device_active_nanoseconds;
                    }
                    if (sample >= warmups)
                    {
                        wall_us[sample - warmups] = std::chrono::duration<double, std::micro>(completed - started).count();
                        active_sum_us[sample - warmups] = static_cast<double>(active_ns) / 1000.0;
                        retire_us[sample - warmups] = std::chrono::duration<double, std::micro>(retired - completed).count();
                    }
                    if (direction == MappedTransferDirection::HostToDevice)
                    {
                        engine.enqueuePersistentDeviceRegionToMappedHost(storage->deviceData(), bytes * capacity,
                            0u, *payload, 0u, bytes * capacity, device, primary);
                        if (!context.recordEventChecked(terminal, primary) || !context.synchronizeEventChecked(terminal))
                            throw std::runtime_error("Captured copy diagnostic readback failed");
                    }
                    for (std::size_t slot = 0u; slot < capacity; ++slot)
                    {
                        const auto *actual = static_cast<const unsigned char *>(
                            payload->mutableHostData(slot * bytes));
                        EXPECT_TRUE(std::all_of(actual, actual + bytes,
                            [value = expected(slot)](unsigned char byte) { return byte == value; }));
                    }
                }
                std::sort(wall_us.begin(), wall_us.end());
                std::sort(active_sum_us.begin(), active_sum_us.end());
                std::sort(retire_us.begin(), retire_us.end());
                std::printf("CAPTURED_SERVICE_ECONOMY,%s,slots=%zu,bytes_per_slot=%zu,active_sum_us=%.3f,wall_us=%.3f,wall_GBps=%.3f,retire_host_us=%.3f\n",
                    direction == MappedTransferDirection::DeviceToHost ? "d2h" : "h2d", capacity, bytes,
                    active_sum_us[samples / 2u], wall_us[samples / 2u],
                    bytes * capacity / (wall_us[samples / 2u] * 1000.0), retire_us[samples / 2u]);
            }
        });
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

    /** One captured capacity and its independently varying live CPU payload. */
    struct CanonicalTicketGeometry
    {
        std::size_t capacity; ///< Route bank embedded in the retained graph.
        std::size_t live_routes; ///< Actual routes published by the CPU.
        const char *name; ///< Exact geometry for isolated profiler launches.
    };

    /**
     * @brief Time the complete retained CPU-ticket consumer with ready payloads.
     * @param device Exact continuation backend, independent of tier topology.
     * @param geometry Captured capacity and actual, potentially sparse payload.
     *
     * Publication and setup are outside event timing. Every sample reuses the
     * production sequence/acknowledgement protocol and one retained graph; no
     * synthetic wait is included in the economy number. Functional held-ticket
     * progress and adversarial replay are covered by the integration fixture.
     */
    void measureCanonicalTickets(DeviceId device, CanonicalTicketGeometry geometry)
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
            // A production MTP graph retains its maximum-depth capacity even
            // when the controller publishes only a few active verifier rows.
            const std::size_t routes = geometry.capacity;
            const std::size_t live_routes = geometry.live_routes;
            ASSERT_LE(live_routes, routes);
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
                ASSERT_TRUE(backend->memset(output->mutableDeviceData(), 0, bytes,
                    device.ordinal, stream));
                ASSERT_TRUE(initial.publish(live_routes));
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
                    ASSERT_TRUE(publication.publish(live_routes));
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
                std::printf("TICKET_ECONOMY,%s,%zu,%zu,%.3f,%.3f\n",
                    device.toString().c_str(), routes, live_routes, median_ms * 1000.0,
                    (live_routes * width * sizeof(float)) / (median_ms * 1e6));
                std::vector<float> actual(routes * width);
                std::vector<float> expected(routes * width, 0.0f);
                std::fill_n(expected.data(), live_routes * width, 0.125f);
                ASSERT_TRUE(backend->deviceToHost(actual.data(), output->deviceData(), bytes,
                    device.ordinal, stream));
                EXPECT_EQ(std::memcmp(actual.data(), expected.data(), bytes), 0);
            }
        });
    }

    /**
     * @brief Time a captured follower return at one production route geometry.
     * @param device GPU publishing the mapped return, independent of vendor role.
     * @param geometry Retained route capacity and current sparse live prefix.
     *
     * This isolates an interior stage: setup supplies its already acquired
     * private grant, then the real pack entrypoint validates and publishes it.
     * Grant reset and payload upload precede the timing event. The full peer
     * admission/acquire/retirement protocol remains the integration fixture's
     * responsibility, not a claim made by this ready-payload microbenchmark.
     */
    void measureReturnPackets(DeviceId device, CanonicalTicketGeometry geometry)
    {
        auto *backend = getBackendFor(device);
        ASSERT_NE(backend, nullptr);
        if (backend->deviceCount() <= device.gpu_ordinal()) GTEST_SKIP();
        auto &context = GPUDeviceContextPool::instance().getContext(device);
        context.submitAndWait([&] {
            constexpr int width = 3072;
            constexpr int top_k = 8;
            const std::size_t routes = geometry.capacity;
            ASSERT_EQ(routes % top_k, 0u);
            ASSERT_LE(geometry.live_routes, routes);
            const auto rows = static_cast<int>(routes / top_k);
            const std::size_t bytes = routes * width * sizeof(float);
            TransferEngine engine;
            const std::array devices{device};
            void *stream = context.getOrCreateAuxiliaryStream("return_packet_economy");
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
            auto payload = engine.allocateMappedHostRegion(bytes, devices);
            auto source = engine.allocateDeviceTransferBuffer(bytes, device);
            auto control = engine.allocateMappedHostRegion(
                sizeof(MoEOverlayActivationEpochControl), devices);
            auto grant_staging = engine.allocateMappedHostRegion(
                sizeof(MoEOverlayActivationDeviceEpochGrant), devices);
            auto grant = engine.allocateDeviceTransferBuffer(
                sizeof(MoEOverlayActivationDeviceEpochGrant), device);
            // Independent CSR arrays have full captured extents even though
            // return packing reads only original and compact route identities.
            auto metadata = engine.allocateMappedHostRegion(7u * routes * sizeof(int), devices);
            auto *host_indices = static_cast<int *>(metadata->mutableHostData());
            auto *device_indices = static_cast<int *>(metadata->deviceAlias(device));
            std::fill_n(host_indices, 7u * routes, 0);
            for (std::size_t entry = 0; entry < geometry.live_routes; ++entry)
            {
                host_indices[entry] = static_cast<int>(routes - geometry.live_routes + entry);
                host_indices[routes + entry] = static_cast<int>(entry);
            }
            std::memset(control->mutableHostData(), 0, sizeof(MoEOverlayActivationEpochControl));
            std::vector<float> input(routes * width);
            for (std::size_t i = 0; i < input.size(); ++i)
                input[i] = static_cast<float>(static_cast<int>(i % 997u) - 498) * 0.125f;
            std::memcpy(payload->mutableHostData(), input.data(), bytes);
            engine.enqueueMappedHostToPersistentDeviceRegion(*payload, 0u,
                source->mutableDeviceData(), bytes, 0u, bytes, device, stream);
            ASSERT_TRUE(context.recordEventChecked(end, stream));
            ASSERT_TRUE(context.synchronizeEventChecked(end));
            std::fill_n(static_cast<float *>(payload->mutableHostData()), input.size(), 0.0f);
            *static_cast<MoEOverlayActivationDeviceEpochGrant *>(grant_staging->mutableHostData()) = {
                .digest = {.low = 17u, .high = 23u},
                .generation = 1u,
                .placement_epoch = 1u,
                .live_rows = (geometry.live_routes + top_k - 1u) / top_k,
                .live_entries = geometry.live_routes,
                .stage_count = 2u,
                .physical_rows = rows,
                .last_published_stage = 0,
                .last_consumed_stage = 1,
                .endpoint = static_cast<std::uint32_t>(MoEOverlayActivationEndpoint::Follower),
                .state = static_cast<std::uint32_t>(MoEOverlayActivationEndpointState::Active),
                .code = static_cast<std::uint32_t>(MoEOverlayActivationStatusCode::Success),
                .graph_role = MoEOverlayInferenceGraphRole::MainPrefill,
            };
            const MoEOverlayActivationReturnPackLaunch launch{
                .local_canonical_route_contributions_fp32 = static_cast<const float *>(source->deviceData()),
                .dispatch = {
                    .row_ids = device_indices + 2u * routes,
                    .entry_offsets = device_indices + 3u * routes,
                    .expert_ids = device_indices + 4u * routes,
                    .route_weights = reinterpret_cast<float *>(device_indices + 5u * routes),
                    .original_route_slots = device_indices,
                    .compact_route_slots = device_indices + routes,
                    .hidden_rows_fp32 = static_cast<float *>(payload->deviceAlias(device)),
                    .row_capacity = static_cast<std::size_t>(rows),
                    .entry_capacity = routes, .d_model = width, .top_k = top_k},
                .returned = {static_cast<float *>(payload->deviceAlias(device)), routes, width},
                .control = static_cast<MoEOverlayActivationEpochControl *>(control->deviceAlias(device)),
                .grant = static_cast<MoEOverlayActivationDeviceEpochGrant *>(grant->mutableDeviceData()),
                .physical_rows = rows, .stage_ordinal = 1u, .model_layer_index = 5,
            };
            std::unique_ptr<IMoEKernel> kernel;
#ifdef HAVE_CUDA
            if (device.is_cuda()) kernel = std::make_unique<CUDAMoEKernel>(device.ordinal);
#endif
#ifdef HAVE_ROCM
            if (device.is_rocm()) kernel = std::make_unique<ROCmMoEKernel>(device.ordinal);
#endif
            ASSERT_NE(kernel, nullptr);
            auto graph = context.createGraphCapture(stream);
            ASSERT_TRUE(graph->beginCapture());
            ASSERT_TRUE(kernel->packMoEOverlayActivationReturn({.stream = stream}, launch));
            ASSERT_TRUE(graph->endCapture());
            ASSERT_TRUE(graph->instantiate());
            auto drain_graph = [&](void *) { (void)backend->synchronizeStream(stream, device.ordinal); };
            std::unique_ptr<void, decltype(drain_graph)> graph_guard(&context, drain_graph);
            std::array<float, 50> timings{};
            for (std::size_t sample = 0; sample < timings.size() + 5u; ++sample)
            {
                engine.enqueueMappedHostToPersistentDeviceRegion(*grant_staging, 0u,
                    grant->mutableDeviceData(), sizeof(MoEOverlayActivationDeviceEpochGrant),
                    0u, sizeof(MoEOverlayActivationDeviceEpochGrant), device, stream);
                ASSERT_TRUE(context.recordEventChecked(start, stream));
                ASSERT_TRUE(graph->launchOnStream(stream));
                ASSERT_TRUE(context.recordEventChecked(end, stream));
                ASSERT_TRUE(context.synchronizeEventChecked(end));
                const auto &status = static_cast<const MoEOverlayActivationEpochControl *>(
                    control->mutableHostData())->follower_status;
                ASSERT_EQ(status.code, static_cast<std::uint32_t>(MoEOverlayActivationStatusCode::Success));
                ASSERT_EQ(status.last_published_stage, 1);
                if (sample >= 5u) timings[sample - 5u] = context.eventElapsedTime(start, end);
            }
            std::sort(timings.begin(), timings.end());
            std::printf("RETURN_PACK_ECONOMY,%s,%zu,%zu,%.3f,%.3f\n",
                device.toString().c_str(), routes, geometry.live_routes,
                timings[timings.size()/2u] * 1000.0f,
                geometry.live_routes * width * sizeof(float) / (timings[timings.size()/2u] * 1e6));
            std::vector<float> expected(input.size(), 0.0f);
            std::copy_n(input.begin(), geometry.live_routes * width,
                expected.begin() + (routes - geometry.live_routes) * width);
            EXPECT_EQ(std::memcmp(expected.data(), payload->mutableHostData(), bytes), 0);
        });
    }

    /**
     * @brief Measure captured dispatch validation and materialization together.
     * @param device Exact participant receiving an already-published packet.
     * @param geometry Captured route capacity and the live compact route prefix.
     *
     * The ready packet represents an interior stage of a previously admitted
     * epoch, just like the return benchmark. Full admission and cross-device
     * ordering are covered by the peer integration suite. Each sample resets
     * the private grant before the start event; payload reads, authentication,
     * sparse row expansion, and clearing inactive output remain inside timing.
     */
    void measureDispatchPackets(DeviceId device, CanonicalTicketGeometry geometry)
    {
        auto *backend = getBackendFor(device);
        ASSERT_NE(backend, nullptr);
        if (backend->deviceCount() <= device.gpu_ordinal()) GTEST_SKIP();
        auto &context = GPUDeviceContextPool::instance().getContext(device);
        context.submitAndWait([&] {
            constexpr int width = 3072;
            constexpr int top_k = 8;
            const auto routes = geometry.capacity;
            ASSERT_EQ(routes % top_k, 0u);
            ASSERT_EQ(geometry.live_routes % top_k, 0u);
            ASSERT_LE(geometry.live_routes, routes);
            const auto rows = static_cast<int>(routes / top_k);
            const auto live_rows = geometry.live_routes / top_k;
            const std::size_t hidden_elements = static_cast<std::size_t>(rows) * width;
            const std::size_t output_bytes = (hidden_elements + 2u * routes) * sizeof(float) + sizeof(int);
            TransferEngine engine;
            const std::array devices{device};
            void *stream = context.getOrCreateAuxiliaryStream("dispatch_packet_economy");
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
            /** One immutable admitted placement and its resettable endpoint grant. */
            struct State
            {
                MoEOverlayActivationDeviceEpochGrant grant{};
                DeviceMoEOverlayEpochTicket ticket{.epoch = 1u, .selector = 2u};
                DeviceMoEOverlayEpochStatus status{};
                std::int32_t owners[top_k]{};
                std::uint32_t epoch = 1u;
            };
            auto state_host = engine.allocateMappedHostRegion(sizeof(State), devices);
            auto state_device = engine.allocateDeviceTransferBuffer(sizeof(State), device);
            auto &state = *new (state_host->mutableHostData()) State{};
            state.grant = {
                .digest = {.low = 17u, .high = 23u},
                .generation = 1u, .placement_epoch = 1u, .stage_count = 2u,
                .physical_rows = rows, .last_published_stage = 0, .last_consumed_stage = 0,
                .endpoint = static_cast<std::uint32_t>(MoEOverlayActivationEndpoint::Follower),
                .state = static_cast<std::uint32_t>(MoEOverlayActivationEndpointState::Active),
                .code = static_cast<std::uint32_t>(MoEOverlayActivationStatusCode::Success),
                .graph_role = MoEOverlayInferenceGraphRole::MainPrefill};
            auto *device_state = static_cast<State *>(state_device->mutableDeviceData());
            auto control = engine.allocateMappedHostRegion(sizeof(MoEOverlayActivationEpochControl), devices);
            auto &host_control = *new (control->mutableHostData()) MoEOverlayActivationEpochControl{};
            const auto bank = moeOverlayActivationBufferIndex(1u);
            const auto timeline = moeOverlayActivationLeasedTimelineValue(moeOverlayActivationBufferVisit(1u));
            host_control.buffers[bank].dispatch_signal.value = timeline;
            host_control.buffers[bank].dispatch_descriptor = {
                .digest = state.grant.digest, .timeline = timeline, .placement_epoch = 1u,
                .live_rows = live_rows, .live_entries = geometry.live_routes,
                .payload_bytes = moeOverlayDispatchPayloadBytes(live_rows, geometry.live_routes, width),
                .stage_ordinal = 1u, .model_layer_index = 5};
            auto payload = engine.allocateMappedHostRegion(hidden_elements * sizeof(float), devices);
            auto *host_payload = static_cast<float *>(payload->mutableHostData());
            for (std::size_t i = 0; i < hidden_elements; ++i)
                host_payload[i] = static_cast<float>(static_cast<int>(i % 997u) - 498) * 0.125f;
            auto metadata = engine.allocateMappedHostRegion(7u * routes * sizeof(int), devices);
            auto *indices = static_cast<int *>(metadata->mutableHostData());
            auto *device_indices = static_cast<int *>(metadata->deviceAlias(device));
            std::fill_n(indices, 7u * routes, 0);
            for (std::size_t row = 0; row < live_rows; ++row)
            {
                // A sparse packet selects the last physical rows, then stores
                // them in a compact prefix at the receiving participant.
                indices[row] = static_cast<int>(rows - live_rows + row);
                indices[routes + row + 1u] = static_cast<int>((row + 1u) * top_k);
                for (int slot = 0; slot < top_k; ++slot)
                {
                    const auto entry = row * top_k + slot;
                    indices[2u * routes + entry] = slot;
                    reinterpret_cast<float *>(indices + 3u * routes)[entry] = 0.125f;
                    indices[4u * routes + entry] = indices[row] * top_k + slot;
                    indices[5u * routes + entry] = static_cast<int>(entry);
                }
            }
            auto output = engine.allocateDeviceTransferBuffer(output_bytes, device);
            auto readback = engine.allocateMappedHostRegion(output_bytes, devices);
            auto *hidden = static_cast<float *>(output->mutableDeviceData());
            const MoEOverlayActivationDispatchConsumeLaunch launch{
                .packet = {.row_ids = device_indices, .entry_offsets = device_indices + routes,
                    .expert_ids = device_indices + 2u * routes,
                    .route_weights = reinterpret_cast<float *>(device_indices + 3u * routes),
                    .original_route_slots = device_indices + 4u * routes,
                    .compact_route_slots = device_indices + 5u * routes,
                    .hidden_rows_fp32 = static_cast<float *>(payload->deviceAlias(device)),
                    .row_capacity = static_cast<std::size_t>(rows), .entry_capacity = routes,
                    .d_model = width, .top_k = top_k},
                .placement = {.banks = {{device_state->owners, &device_state->epoch},
                                       {device_state->owners, &device_state->epoch}},
                    .ticket = &device_state->ticket, .status = &device_state->status, .expert_count = top_k},
                .hidden_payload_layout = MoEOverlayActivationHiddenPayloadLayout::SharedPhysicalRows,
                .control = static_cast<MoEOverlayActivationEpochControl *>(control->deviceAlias(device)),
                .grant = &device_state->grant, .hidden_rows_fp32 = hidden,
                .routing_indices_fp32 = hidden + hidden_elements,
                .routing_weights_fp32 = hidden + hidden_elements + routes,
                .active_row_count_device = reinterpret_cast<int *>(hidden + hidden_elements + 2u * routes),
                .physical_rows = rows, .stage_ordinal = 1u, .model_layer_index = 5};
            std::unique_ptr<IMoEKernel> kernel;
#ifdef HAVE_CUDA
            if (device.is_cuda()) kernel = std::make_unique<CUDAMoEKernel>(device.ordinal);
#endif
#ifdef HAVE_ROCM
            if (device.is_rocm()) kernel = std::make_unique<ROCmMoEKernel>(device.ordinal);
#endif
            ASSERT_NE(kernel, nullptr);
            auto graph = context.createGraphCapture(stream);
            ASSERT_TRUE(graph->beginCapture());
            ASSERT_TRUE(kernel->consumeMoEOverlayActivationDispatch({.stream = stream}, launch));
            ASSERT_TRUE(graph->endCapture());
            ASSERT_TRUE(graph->instantiate());
            auto drain_graph = [&](void *) { (void)backend->synchronizeStream(stream, device.ordinal); };
            std::unique_ptr<void, decltype(drain_graph)> graph_guard(&context, drain_graph);
            std::array<float, 50> timings{};
            for (std::size_t sample = 0; sample < timings.size() + 5u; ++sample)
            {
                engine.enqueueMappedHostToPersistentDeviceRegion(*state_host, 0u,
                    state_device->mutableDeviceData(), sizeof(State), 0u, sizeof(State), device, stream);
                ASSERT_TRUE(context.recordEventChecked(start, stream));
                ASSERT_TRUE(graph->launchOnStream(stream));
                ASSERT_TRUE(context.recordEventChecked(end, stream));
                ASSERT_TRUE(context.synchronizeEventChecked(end));
                if (sample >= 5u) timings[sample - 5u] = context.eventElapsedTime(start, end);
            }
            std::sort(timings.begin(), timings.end());
            std::printf("DISPATCH_CONSUME_ECONOMY,%s,%zu,%zu,%.3f\n", device.toString().c_str(),
                routes, geometry.live_routes, timings[timings.size()/2u] * 1000.0f);
            engine.enqueuePersistentDeviceRegionToMappedHost(output->deviceData(), output_bytes,
                0u, *readback, 0u, output_bytes, device, stream);
            ASSERT_TRUE(context.recordEventChecked(end, stream));
            ASSERT_TRUE(context.synchronizeEventChecked(end));
            const auto *actual = static_cast<const float *>(readback->mutableHostData());
            EXPECT_EQ(*reinterpret_cast<const int *>(actual + hidden_elements + 2u * routes), live_rows);
            std::vector<float> expected(hidden_elements + 2u * routes, 0.0f);
            std::copy_n(host_payload + (rows - live_rows) * width, live_rows * width, expected.data());
            for (std::size_t entry = 0; entry < routes; ++entry)
            {
                expected[hidden_elements + entry] = entry < geometry.live_routes ? float(entry % top_k) : -1.0f;
                expected[hidden_elements + routes + entry] = entry < geometry.live_routes ? 0.125f : 0.0f;
            }
            EXPECT_EQ(std::memcmp(expected.data(), actual, expected.size() * sizeof(float)), 0);
        });
    }

    /**
     * @brief Time a topology-sized captured return join with all peers ready.
     * @param device Continuation GPU owning the canonical destination bank.
     * @param geometry Full route capacity and the live, possibly sparse prefix.
     * @param lanes Independent participant packets sharing the canonical wire bank.
     *
     * Production followers write disjoint original route slots into a shared
     * rank-pair return bank. Interleaved ownership exercises that exact memory
     * pattern. Setup supplies authenticated interior-stage grants; their reset
     * is outside event timing. The integration suite owns peer arrival/reuse
     * races, while this fixture isolates validation plus payload materialization.
     */
    void measureReturnBatch(DeviceId device, CanonicalTicketGeometry geometry, std::size_t lanes)
    {
        auto *backend = getBackendFor(device);
        ASSERT_NE(backend, nullptr);
        if (backend->deviceCount() <= device.gpu_ordinal()) GTEST_SKIP();
        auto &context = GPUDeviceContextPool::instance().getContext(device);
        context.submitAndWait([&] {
            constexpr int width = 3072;
            constexpr int top_k = 8;
            const auto routes = geometry.capacity;
            const auto rows = routes / top_k;
            const auto bytes = routes * width * sizeof(float);
            const auto descriptor_bytes = rows == 1u
                ? sizeof(MoEOverlayActivationSingleRowReturnConsumeLaunch)
                : sizeof(MoEOverlayActivationReturnConsumeLaunch);
            ASSERT_EQ(routes % top_k, 0u);
            ASSERT_LE(geometry.live_routes, routes);
            ASSERT_GT(lanes, 0u);
            TransferEngine engine;
            const std::array devices{device};
            void *stream = context.getOrCreateAuxiliaryStream("return_batch_economy");
            void *start = context.createEvent();
            void *end = context.createEvent();
            ASSERT_NE(stream, nullptr);
            ASSERT_NE(start, nullptr);
            ASSERT_NE(end, nullptr);
            auto retire_events = [&](void *) { context.destroyEvent(end); context.destroyEvent(start); };
            std::unique_ptr<void, decltype(retire_events)> events(&context, retire_events);
            auto payload = engine.allocateMappedHostRegion(bytes, devices);
            auto output = engine.allocateDeviceTransferBuffer(bytes, device);
            auto grants_host = engine.allocateMappedHostRegion(
                lanes * sizeof(MoEOverlayActivationDeviceEpochGrant), devices);
            auto grants = engine.allocateDeviceTransferBuffer(
                lanes * sizeof(MoEOverlayActivationDeviceEpochGrant), device);
            auto descriptions_host = engine.allocateMappedHostRegion(
                lanes * descriptor_bytes, devices);
            auto descriptions = engine.allocateDeviceTransferBuffer(
                lanes * descriptor_bytes, device);
            auto valid = engine.allocateDeviceTransferBuffer(lanes * sizeof(int), device);
            std::vector<std::shared_ptr<MappedHostTransferRegion>> controls, metadata;
            for (std::size_t lane = 0u; lane < lanes; ++lane)
            {
                controls.push_back(engine.allocateMappedHostRegion(sizeof(MoEOverlayActivationEpochControl), devices));
                metadata.push_back(engine.allocateMappedHostRegion(7u * routes * sizeof(int), devices));
                const auto entries = geometry.live_routes <= lane ? 0u
                    : (geometry.live_routes - lane + lanes - 1u) / lanes;
                auto *indices_host = static_cast<int *>(metadata.back()->mutableHostData());
                auto *indices_device = static_cast<int *>(metadata.back()->deviceAlias(device));
                std::fill_n(indices_host, 7u * routes, 0);
                for (std::size_t entry = 0; entry < entries; ++entry)
                {
                    indices_host[entry] = static_cast<int>(lane + entry * lanes);
                    indices_host[routes + entry] = static_cast<int>(entry);
                }
                const auto live_rows = (entries + top_k - 1u) / top_k;
                static_cast<MoEOverlayActivationDeviceEpochGrant *>(grants_host->mutableHostData())[lane] = {
                    .digest = {.low = 17u, .high = 23u}, .generation = 1u, .placement_epoch = 1u,
                    .live_rows = live_rows, .live_entries = entries, .stage_count = 2u,
                    .physical_rows = static_cast<int>(rows), .last_published_stage = 1, .last_consumed_stage = 0,
                    .endpoint = static_cast<std::uint32_t>(MoEOverlayActivationEndpoint::Continuation),
                    .state = static_cast<std::uint32_t>(MoEOverlayActivationEndpointState::Active),
                    .code = static_cast<std::uint32_t>(MoEOverlayActivationStatusCode::Success),
                    .graph_role = MoEOverlayInferenceGraphRole::MainPrefill,
                };
                auto *control = static_cast<MoEOverlayActivationEpochControl *>(controls.back()->mutableHostData());
                const auto bank = moeOverlayActivationBufferIndex(1u);
                const auto timeline = moeOverlayActivationLeasedTimelineValue(moeOverlayActivationBufferVisit(1u));
                control->buffers[bank].return_signal.value = timeline;
                control->buffers[bank].return_descriptor = {
                    .digest = {.low = 17u, .high = 23u}, .timeline = timeline, .placement_epoch = 1u,
                    .live_rows = live_rows, .live_entries = entries,
                    .payload_bytes = moeOverlayReturnPayloadBytes(entries, width),
                    .stage_ordinal = 1u, .model_layer_index = 5,
                };
                const MoEOverlayActivationReturnConsumeLaunch packet{
                    .dispatch = {
                        .row_ids = indices_device + 2u * routes, .entry_offsets = indices_device + 3u * routes,
                        .expert_ids = indices_device + 4u * routes,
                        .route_weights = reinterpret_cast<float *>(indices_device + 5u * routes),
                        .original_route_slots = indices_device, .compact_route_slots = indices_device + routes,
                        .hidden_rows_fp32 = static_cast<float *>(payload->deviceAlias(device)),
                        .row_capacity = rows, .entry_capacity = routes, .d_model = width, .top_k = top_k},
                    .returned = {static_cast<float *>(payload->deviceAlias(device)), routes, width},
                    .control = static_cast<MoEOverlayActivationEpochControl *>(controls.back()->deviceAlias(device)),
                    .grant = static_cast<MoEOverlayActivationDeviceEpochGrant *>(grants->mutableDeviceData()) + lane,
                    .canonical_route_contributions_fp32 = static_cast<float *>(output->mutableDeviceData()),
                    .physical_rows = static_cast<int>(rows), .stage_ordinal = 1u, .model_layer_index = 5,
                };
                if (rows == 1u)
                {
                    auto *device_control = static_cast<MoEOverlayActivationEpochControl *>(
                        controls.back()->deviceAlias(device));
                    static_cast<MoEOverlayActivationSingleRowReturnConsumeLaunch *>(
                        descriptions_host->mutableHostData())[lane] = {
                        .packet = packet,
                        .acquire = {&device_control->buffers[bank].return_signal.value, timeline},
                    };
                }
                else
                    static_cast<MoEOverlayActivationReturnConsumeLaunch *>(
                        descriptions_host->mutableHostData())[lane] = packet;
            }
            auto *source = static_cast<float *>(payload->mutableHostData());
            for (std::size_t element = 0; element < routes * width; ++element)
                source[element] = static_cast<float>(static_cast<int>(element % 997u) - 498) * 0.125f;
            ASSERT_TRUE(backend->memset(output->mutableDeviceData(), 0, bytes, device.ordinal, stream));
            engine.enqueueMappedHostToPersistentDeviceRegion(*descriptions_host, 0u,
                descriptions->mutableDeviceData(), lanes * descriptor_bytes,
                0u, lanes * descriptor_bytes, device, stream);
            ASSERT_TRUE(context.recordEventChecked(end, stream));
            ASSERT_TRUE(context.synchronizeEventChecked(end));
            std::unique_ptr<IMoEKernel> kernel;
#ifdef HAVE_CUDA
            if (device.is_cuda()) kernel = std::make_unique<CUDAMoEKernel>(device.ordinal);
#endif
#ifdef HAVE_ROCM
            if (device.is_rocm()) kernel = std::make_unique<ROCmMoEKernel>(device.ordinal);
#endif
            ASSERT_NE(kernel, nullptr);
            const MoEOverlayActivationMultiRowReturnBatchLaunch launch{
                .lanes = static_cast<const MoEOverlayActivationReturnConsumeLaunch *>(descriptions->deviceData()),
                .lane_valid = static_cast<int *>(valid->mutableDeviceData()),
                .canonical_route_contributions_fp32 = static_cast<float *>(output->mutableDeviceData()),
                .lane_count = static_cast<std::uint32_t>(lanes), .physical_rows = static_cast<int>(rows),
                .d_model = width, .top_k = top_k,
            };
            auto graph = context.createGraphCapture(stream);
            ASSERT_TRUE(graph->beginCapture());
            if (rows == 1u)
            {
                // The physical-row policy selects the real fused-wait draft
                // entrypoint, not a multi-row approximation of decode.
                const MoEOverlayActivationSingleRowReturnBatchLaunch single{
                    .lanes = static_cast<const MoEOverlayActivationSingleRowReturnConsumeLaunch *>(
                        descriptions->deviceData()),
                    .lane_valid = launch.lane_valid,
                    .canonical_route_contributions_fp32 = launch.canonical_route_contributions_fp32,
                    .lane_count = launch.lane_count, .d_model = width, .top_k = top_k,
                };
                ASSERT_TRUE(kernel->consumeSingleRowMoEOverlayActivationReturnBatch({.stream = stream}, single));
            }
            else
                ASSERT_TRUE(kernel->consumeMultiRowMoEOverlayActivationReturnBatch({.stream = stream}, launch));
            ASSERT_TRUE(graph->endCapture());
            ASSERT_TRUE(graph->instantiate());
            auto drain_graph = [&](void *) { (void)backend->synchronizeStream(stream, device.ordinal); };
            std::unique_ptr<void, decltype(drain_graph)> graph_guard(&context, drain_graph);
            std::array<float, 50> timings{};
            for (std::size_t sample = 0; sample < timings.size() + 5u; ++sample)
            {
                engine.enqueueMappedHostToPersistentDeviceRegion(*grants_host, 0u,
                    grants->mutableDeviceData(), lanes * sizeof(MoEOverlayActivationDeviceEpochGrant),
                    0u, lanes * sizeof(MoEOverlayActivationDeviceEpochGrant), device, stream);
                ASSERT_TRUE(context.recordEventChecked(start, stream));
                ASSERT_TRUE(graph->launchOnStream(stream));
                ASSERT_TRUE(context.recordEventChecked(end, stream));
                ASSERT_TRUE(context.synchronizeEventChecked(end));
                if (sample >= 5u) timings[sample - 5u] = context.eventElapsedTime(start, end);
            }
            std::sort(timings.begin(), timings.end());
            std::printf("RETURN_BATCH_ECONOMY,%s,%zu,%zu,%zu,%.3f\n", device.toString().c_str(),
                lanes, routes, geometry.live_routes, timings[timings.size()/2u] * 1000.0f);
            std::vector<float> actual(routes * width), expected(routes * width, 0.0f);
            std::copy_n(source, geometry.live_routes * width, expected.data());
            ASSERT_TRUE(backend->deviceToHost(actual.data(), output->deviceData(), bytes, device.ordinal, stream));
            EXPECT_EQ(std::memcmp(actual.data(), expected.data(), bytes), 0);
            for (const auto &control : controls)
                EXPECT_EQ(static_cast<const MoEOverlayActivationEpochControl *>(control->mutableHostData())->continuation_status.code,
                    static_cast<std::uint32_t>(MoEOverlayActivationStatusCode::Success));
        });
    }

    /** Captured multi-lane joins isolate the real shared-bank read geometry. */
    class MappedReturnBatchEconomy : public ::testing::TestWithParam<CanonicalTicketGeometry> {};
#if defined(HAVE_CUDA)
    TEST_P(MappedReturnBatchEconomy, CUDAFourLanes)
    { measureReturnBatch(DeviceId::cuda(0), GetParam(), 4u); }
#endif
#if defined(HAVE_ROCM)
    TEST_P(MappedReturnBatchEconomy, ROCmFourLanes)
    { measureReturnBatch(DeviceId::rocm(0), GetParam(), 4u); }
#endif
    INSTANTIATE_TEST_SUITE_P(ReadyPayload, MappedReturnBatchEconomy,
        ::testing::Values(
            CanonicalTicketGeometry{8u, 4u, "SparseDraft"},
            CanonicalTicketGeometry{8u, 8u, "FullDraft"},
            CanonicalTicketGeometry{128u, 32u, "SparseVerifier"},
            CanonicalTicketGeometry{128u, 128u, "FullVerifier"},
            CanonicalTicketGeometry{4096u, 512u, "SparsePrefill"},
            CanonicalTicketGeometry{4096u, 4096u, "FullPrefill"}),
        [](const ::testing::TestParamInfo<CanonicalTicketGeometry> &info) { return info.param.name; });

    /** One exact ready-dispatch shape per timing and counter collection. */
    class MappedDispatchConsumeEconomy : public ::testing::TestWithParam<CanonicalTicketGeometry> {};
#if defined(HAVE_CUDA)
    TEST_P(MappedDispatchConsumeEconomy, CUDA)
    { measureDispatchPackets(DeviceId::cuda(0), GetParam()); }
#endif
#if defined(HAVE_ROCM)
    TEST_P(MappedDispatchConsumeEconomy, ROCm)
    { measureDispatchPackets(DeviceId::rocm(0), GetParam()); }
#endif
    INSTANTIATE_TEST_SUITE_P(ReadyPayload, MappedDispatchConsumeEconomy,
        ::testing::Values(
            CanonicalTicketGeometry{128u, 32u, "SparseVerifier"},
            CanonicalTicketGeometry{128u, 128u, "FullVerifier"},
            CanonicalTicketGeometry{4096u, 512u, "SparsePrefill"},
            CanonicalTicketGeometry{4096u, 4096u, "FullPrefill"}),
        [](const ::testing::TestParamInfo<CanonicalTicketGeometry> &info) {
            return info.param.name;
        });

    /** One exact ready-return shape per timing and counter collection. */
    class MappedReturnPackEconomy : public ::testing::TestWithParam<CanonicalTicketGeometry> {};
#if defined(HAVE_CUDA)
    TEST_P(MappedReturnPackEconomy, CUDA)
    { measureReturnPackets(DeviceId::cuda(0), GetParam()); }
#endif
#if defined(HAVE_ROCM)
    TEST_P(MappedReturnPackEconomy, ROCm)
    { measureReturnPackets(DeviceId::rocm(0), GetParam()); }
#endif
    INSTANTIATE_TEST_SUITE_P(ReadyPayload, MappedReturnPackEconomy,
        ::testing::Values(
            CanonicalTicketGeometry{128u, 32u, "SparseVerifier"},
            CanonicalTicketGeometry{128u, 128u, "FullVerifier"},
            CanonicalTicketGeometry{4096u, 512u, "SparsePrefill"},
            CanonicalTicketGeometry{4096u, 4096u, "FullPrefill"}),
        [](const ::testing::TestParamInfo<CanonicalTicketGeometry> &info) {
            return info.param.name;
        });

#if defined(HAVE_CUDA)
    // Separate exact geometries keep profiler attribution unambiguous. Thirty
    // inboxes cover five migration slots, three projections and two banks.
    TEST(CapturedTransferServiceEconomy, CUDADownload4Slots4MiB)
    { measureCapturedService(MappedTransferDirection::DeviceToHost, 4u); }
    TEST(CapturedTransferServiceEconomy, CUDAUpload4Slots4MiB)
    { measureCapturedService(MappedTransferDirection::HostToDevice, 4u); }
    TEST(CapturedTransferServiceEconomy, CUDADownload30Slots4MiB)
    { measureCapturedService(MappedTransferDirection::DeviceToHost, 30u); }
    TEST(CapturedTransferServiceEconomy, CUDAUpload30Slots4MiB)
    { measureCapturedService(MappedTransferDirection::HostToDevice, 30u); }
    TEST(BoundedTransferServiceEconomy, CUDADownload4MiB)
    { measureBoundedService(MappedTransferDirection::DeviceToHost); }
    TEST(BoundedTransferServiceEconomy, CUDAUpload4MiB)
    { measureBoundedService(MappedTransferDirection::HostToDevice); }
    TEST(BackgroundMappedCopyEconomy, CUDADownload)
    { measureCopies(DeviceId::cuda(0), MappedTransferDirection::DeviceToHost); }
    TEST(BackgroundMappedCopyEconomy, CUDAUpload)
    { measureCopies(DeviceId::cuda(0), MappedTransferDirection::HostToDevice); }
#endif
#if defined(HAVE_ROCM)
    TEST(BackgroundMappedCopyEconomy, ROCmDownload)
    { measureCopies(DeviceId::rocm(0), MappedTransferDirection::DeviceToHost); }
    TEST(BackgroundMappedCopyEconomy, ROCmUpload)
    { measureCopies(DeviceId::rocm(0), MappedTransferDirection::HostToDevice); }
#endif

    /** Exact backend/shape cases keep counter runs independent of other shapes. */
    class CanonicalRouteTicketEconomy : public ::testing::TestWithParam<CanonicalTicketGeometry> {};
#if defined(HAVE_CUDA)
    TEST_P(CanonicalRouteTicketEconomy, CUDA)
    { measureCanonicalTickets(DeviceId::cuda(0), GetParam()); }
#endif
#if defined(HAVE_ROCM)
    TEST_P(CanonicalRouteTicketEconomy, ROCm)
    { measureCanonicalTickets(DeviceId::rocm(0), GetParam()); }
#endif
    INSTANTIATE_TEST_SUITE_P(ReadyPayload, CanonicalRouteTicketEconomy,
        ::testing::Values(
            CanonicalTicketGeometry{8u, 8u, "Decode"},
            CanonicalTicketGeometry{128u, 32u, "SparseVerifier"},
            CanonicalTicketGeometry{512u, 512u, "Prefill"},
            CanonicalTicketGeometry{4096u, 512u, "SparsePrefill"},
            CanonicalTicketGeometry{4800u, 4800u, "LargePrefill"}),
        [](const ::testing::TestParamInfo<CanonicalTicketGeometry> &info) {
            return info.param.name;
        });
} // namespace
} // namespace llaminar2
