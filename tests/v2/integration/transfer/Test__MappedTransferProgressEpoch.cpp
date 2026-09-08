/**
 * @file Test__MappedTransferProgressEpoch.cpp
 * @brief Real-device byte and lifecycle proofs for background expert copies.
 *
 * Each backend runs simultaneous D2H and H2D commands on permanent typed
 * slots. The tests prove reusable generation identity, byte exactness, and the
 * central lifecycle invariant: a production-shaped inference graph can finish
 * while maintenance completion remains unpublished, because no inference
 * stream waits for a complete transfer command. CUDA joins only its bounded
 * graph worker's retirement, preserving partial-copy cursors across intervals.
 * The converse is equally important: a peer-held inference graph must not
 * strand unrelated maintenance behind its not-yet-runnable DMA node.
 * Future observers of that graph are separate native streams, as in prefix
 * archival. They must not exhaust physical work queues needed by maintenance.
 */

#include <gtest/gtest.h>

#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IBackend.h"
#include "backends/IGPUGraphCapture.h"
#include "backends/IWorkerGPUContext.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "execution/local_execution/orchestrators/TPWorkerPool.h"
#include "execution/moe/ExpertTierWeightTransferLane.h"
#include "execution/moe/MoEOverlayGpuRemoteProjectionEndpoint.h"
#include "tensors/TensorKernels.h"
#include "transfer/MappedTransferProgressEpoch.h"
#include "transfer/TransferEngine.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace llaminar2
{
namespace
{
    constexpr std::size_t kTransferBytes =
        4u * 1024u * 1024u + 13u;

    /** @brief Fill a deterministic byte pattern with no repeated cache line. */
    std::vector<std::uint8_t> makePattern(
        std::size_t bytes,
        std::uint32_t salt)
    {
        std::vector<std::uint8_t> result(bytes);
        for (std::size_t index = 0u; index < result.size(); ++index)
        {
            result[index] = static_cast<std::uint8_t>(
                (index * 43u + (index >> 3u) + salt) & 0xffu);
        }
        return result;
    }

    /** @brief Poll both slots while submitting only bounded worker callbacks. */
    void drainBothSlots(
        const std::shared_ptr<MappedTransferProgressEpoch> &epoch,
        MappedTransferProgressSlot &device_to_host,
        MappedTransferProgressSlot &host_to_device)
    {
        MappedTransferProgress d2h = MappedTransferProgress::Pending;
        MappedTransferProgress h2d = MappedTransferProgress::Pending;
        std::string d2h_error;
        std::string h2d_error;
        const auto deadline = std::chrono::steady_clock::now() +
                              std::chrono::seconds(10);
        while ((d2h == MappedTransferProgress::Pending ||
                h2d == MappedTransferProgress::Pending) &&
               std::chrono::steady_clock::now() < deadline)
        {
            ASSERT_TRUE(epoch->submitOutstandingProgress());
            if (d2h == MappedTransferProgress::Pending)
                d2h = device_to_host.poll(&d2h_error);
            if (h2d == MappedTransferProgress::Pending)
                h2d = host_to_device.poll(&h2d_error);
            if (d2h == MappedTransferProgress::Pending ||
                h2d == MappedTransferProgress::Pending)
            {
                std::this_thread::yield();
            }
        }
        ASSERT_EQ(d2h, MappedTransferProgress::Ready) << d2h_error;
        ASSERT_EQ(h2d, MappedTransferProgress::Ready) << h2d_error;
    }

    /**
     * @brief Reuse two permanent slots through many device-address generations.
     *
     * Mapped storage is deliberately immutable slot identity. Alternating the
     * device allocations still forces every generation to publish genuinely
     * new device addresses and covers the rollover range that exposed the old
     * ROCm mapped-command copy-kernel hang.
     */
    void proveRepeatedGenerations(DeviceId device)
    {
        constexpr std::size_t kGenerations = 256u;
        IBackend *const backend = getBackendFor(device);
        ASSERT_NE(backend, nullptr);
        if (backend->deviceCount() <= device.gpu_ordinal())
            GTEST_SKIP() << device.toString() << " is unavailable";

        auto &context = GPUDeviceContextPool::instance().getContext(device);
        TransferEngine transfer_engine;
        const DeviceId mapped_devices[] = {device};
        auto mapped_source = transfer_engine.allocateMappedHostRegion(
            kTransferBytes, mapped_devices);
        auto mapped_destination = transfer_engine.allocateMappedHostRegion(
            kTransferBytes, mapped_devices);
        ASSERT_TRUE(mapped_source && mapped_destination);

        auto epoch = MappedTransferProgressEpoch::create({
            .device = device,
            .slot_capacity = 2u,
            .execution_lane_capacity = 2u,
            .execution_streams =
                TransferEngine::instance()
                    .allocatePersistentTransferExecutionLanes(
                        2u,
                        device,
                        "mapped_progress_repeated_generations"),
            .maximum_bytes = kTransferBytes,
            .name = "integration_repeated_generations_" + device.toString(),
            .perf_device = device.toString(),
        });
        auto device_to_host_slot = epoch->reserveSlot(
            MappedTransferDirection::DeviceToHost,
            mapped_destination,
            "stress_device_to_host");
        auto host_to_device_slot = epoch->reserveSlot(
            MappedTransferDirection::HostToDevice,
            mapped_source,
            "stress_host_to_device");

        std::array<std::shared_ptr<DeviceTransferBuffer>, 2u> device_sources;
        std::array<std::shared_ptr<DeviceTransferBuffer>, 2u>
            device_destinations;
        std::array<std::vector<std::uint8_t>, 2u> d2h_expected{
            makePattern(kTransferBytes, 0x31u),
            makePattern(kTransferBytes, 0x73u)};
        const auto h2d_expected = makePattern(kTransferBytes, 0xb5u);
        for (std::size_t lane = 0u; lane < 2u; ++lane)
        {
            device_sources[lane] =
                transfer_engine.allocateDeviceTransferBuffer(
                    kTransferBytes, device);
            device_destinations[lane] =
                transfer_engine.allocateDeviceTransferBuffer(
                    kTransferBytes, device);
            ASSERT_TRUE(device_sources[lane] && device_destinations[lane]);
        }

        void *setup_stream = nullptr;
        context.submitAndWait(
            [&]
            {
                setup_stream = context.getOrCreateAuxiliaryStream(
                    "mapped_transfer_progress_generation_setup:" +
                        device.toString(),
                    GPUAuxiliaryStreamSchedulingClass::Normal);
                if (!setup_stream)
                    throw std::runtime_error(
                        "Could not create generation setup stream");
                for (std::size_t lane = 0u; lane < 2u; ++lane)
                {
                    if (!backend->hostToDevice(
                            device_sources[lane]->mutableDeviceData(),
                            d2h_expected[lane].data(),
                            kTransferBytes,
                            device.gpu_ordinal(),
                            setup_stream))
                    {
                        throw std::runtime_error(
                            "Could not initialize generation source");
                    }
                }
                context.synchronizeStream(setup_stream);
            });
        std::memcpy(
            mapped_source->mutableHostData(),
            h2d_expected.data(),
            kTransferBytes);

        for (std::size_t generation = 1u;
             generation <= kGenerations;
             ++generation)
        {
            const std::size_t lane = generation & 1u;
            std::memset(
                mapped_destination->mutableHostData(), 0, kTransferBytes);
            ASSERT_EQ(
                device_to_host_slot.publishDeviceToMappedHost(
                    device_sources[lane]->deviceData(),
                    kTransferBytes,
                    0u,
                    kTransferBytes),
                generation);
            ASSERT_EQ(
                host_to_device_slot.publishMappedHostToDevice(
                    device_destinations[lane]->mutableDeviceData(),
                    kTransferBytes,
                    0u,
                    kTransferBytes),
                generation);

            drainBothSlots(
                epoch, device_to_host_slot, host_to_device_slot);
            ASSERT_EQ(
                std::memcmp(
                    mapped_destination->mutableHostData(),
                    d2h_expected[lane].data(),
                    kTransferBytes),
                0)
                << "generation=" << generation;
        }

        std::array<std::vector<std::uint8_t>, 2u> h2d_actual{
            std::vector<std::uint8_t>(kTransferBytes),
            std::vector<std::uint8_t>(kTransferBytes)};
        context.submitAndWait(
            [&]
            {
                for (std::size_t lane = 0u; lane < 2u; ++lane)
                {
                    if (!backend->deviceToHost(
                            h2d_actual[lane].data(),
                            device_destinations[lane]->deviceData(),
                            kTransferBytes,
                            device.gpu_ordinal(),
                            setup_stream))
                    {
                        throw std::runtime_error(
                            "Could not collect generation destination");
                    }
                }
                context.synchronizeStream(setup_stream);
            });
        EXPECT_EQ(h2d_actual[0], h2d_expected);
        EXPECT_EQ(h2d_actual[1], h2d_expected);

        const auto stats = epoch->stats();
        EXPECT_EQ(stats.commands_published, 2u * kGenerations);
        EXPECT_EQ(stats.commands_completed, 2u * kGenerations);
        EXPECT_EQ(stats.bytes_completed, 2u * kGenerations * kTransferBytes);
        EXPECT_EQ(stats.copy_submissions, 2u * kGenerations);
        EXPECT_EQ(stats.command_failures, 0u);
    }

    /**
     * @brief Prove inference completion has no maintenance lifecycle edge.
     *
     * The scheduler launches two large transfers, then deliberately performs no
     * maintenance progress callback while a separately captured inference-like
     * graph runs to its own terminal event. Transfer completion remains
     * unpublished until maintenance resumes, proving the inference graph cannot
     * have captured or joined the movement events.
     */
    void proveInferenceDoesNotJoinMaintenance(DeviceId device)
    {
        constexpr std::size_t kLargeBytes = 64u * 1024u * 1024u;
        IBackend *const backend = getBackendFor(device);
        ASSERT_NE(backend, nullptr);
        if (backend->deviceCount() <= device.gpu_ordinal())
            GTEST_SKIP() << device.toString() << " is unavailable";

        auto &context = GPUDeviceContextPool::instance().getContext(device);
        TransferEngine transfer_engine;
        const DeviceId mapped_devices[] = {device};
        auto mapped_source = transfer_engine.allocateMappedHostRegion(
            kLargeBytes, mapped_devices);
        auto mapped_destination = transfer_engine.allocateMappedHostRegion(
            kLargeBytes, mapped_devices);
        auto device_source = transfer_engine.allocateDeviceTransferBuffer(
            kLargeBytes, device);
        auto device_destination = transfer_engine.allocateDeviceTransferBuffer(
            kLargeBytes, device);
        auto witness = transfer_engine.allocateDeviceTransferBuffer(4096u, device);
        ASSERT_TRUE(
            mapped_source && mapped_destination && device_source &&
            device_destination && witness);

        const auto d2h_expected = makePattern(kLargeBytes, 0x5du);
        const auto h2d_expected = makePattern(kLargeBytes, 0xa7u);
        std::memcpy(
            mapped_source->mutableHostData(),
            h2d_expected.data(),
            kLargeBytes);
        std::memset(
            mapped_destination->mutableHostData(), 0, kLargeBytes);

        void *inference_stream = nullptr;
        void *inference_terminal = nullptr;
        std::unique_ptr<IGPUGraphCapture> inference_graph;
        context.submitAndWait(
            [&]
            {
                inference_stream = context.getOrCreateAuxiliaryStream(
                    "mapped_transfer_progress_inference_witness:" +
                        device.toString(),
                    GPUAuxiliaryStreamSchedulingClass::Normal);
                inference_terminal = context.createEvent();
                if (!inference_stream || !inference_terminal ||
                    !backend->hostToDevice(
                        device_source->mutableDeviceData(),
                        d2h_expected.data(),
                        kLargeBytes,
                        device.gpu_ordinal(),
                        inference_stream))
                {
                    throw std::runtime_error(
                        "Could not initialize no-join integration storage");
                }
                context.synchronizeStream(inference_stream);

                inference_graph = context.createGraphCapture(inference_stream);
                if (!inference_graph || !inference_graph->beginCapture())
                    throw std::runtime_error(
                        "Could not begin inference witness capture");
                bool capture_open = true;
                try
                {
                    if (!backend->memset(
                            witness->mutableDeviceData(),
                            0x3c,
                            witness->sizeBytes(),
                            device.gpu_ordinal(),
                            inference_stream) ||
                        !inference_graph->endCapture())
                    {
                        throw std::runtime_error(
                            "Could not capture inference witness node");
                    }
                    capture_open = false;
                    if (inference_graph->nodeCount() != 1u ||
                        !inference_graph->instantiate())
                    {
                        throw std::runtime_error(
                            "Inference witness unexpectedly contains maintenance nodes");
                    }
                }
                catch (...)
                {
                    if (capture_open)
                        (void)inference_graph->endCapture();
                    throw;
                }
            });

        auto epoch = MappedTransferProgressEpoch::create({
            .device = device,
            .slot_capacity = 2u,
            .execution_lane_capacity = 2u,
            .execution_streams =
                TransferEngine::instance()
                    .allocatePersistentTransferExecutionLanes(
                        2u,
                        device,
                        "mapped_progress_inference_no_join"),
            .maximum_bytes = kLargeBytes,
            .name = "integration_inference_no_join_" + device.toString(),
            .perf_device = device.toString(),
        });
        auto device_to_host_slot = epoch->reserveSlot(
            MappedTransferDirection::DeviceToHost,
            mapped_destination,
            "no_join_device_to_host");
        auto host_to_device_slot = epoch->reserveSlot(
            MappedTransferDirection::HostToDevice,
            mapped_source,
            "no_join_host_to_device");
        ASSERT_EQ(
            device_to_host_slot.publishDeviceToMappedHost(
                device_source->deviceData(), kLargeBytes, 0u, kLargeBytes),
            1u);
        ASSERT_EQ(
            host_to_device_slot.publishMappedHostToDevice(
                device_destination->mutableDeviceData(),
                kLargeBytes,
                0u,
                kLargeBytes),
            1u);
        ASSERT_TRUE(epoch->submitOutstandingProgress());

        context.submitAndWait(
            [&]
            {
                if (!inference_graph->launch() ||
                    !context.recordEventChecked(
                        inference_terminal, inference_stream))
                {
                    throw std::runtime_error(
                        "Could not launch inference witness graph");
                }
            });

        bool inference_ready = false;
        const auto deadline = std::chrono::steady_clock::now() +
                              std::chrono::seconds(5);
        while (!inference_ready &&
               std::chrono::steady_clock::now() < deadline)
        {
            context.submitAndWait(
                [&]
                {
                    if (!context.queryEventChecked(
                            inference_terminal, inference_ready))
                    {
                        throw std::runtime_error(
                            "Could not query inference witness event");
                    }
                });
            if (!inference_ready)
                std::this_thread::yield();
        }
        ASSERT_TRUE(inference_ready)
            << device.toString()
            << " inference graph did not finish independently of maintenance";

        /* No maintenance callback has queried the copy events since launch, so
         * host completion must remain unpublished even if a copy engine has
         * already moved the bytes. This is the absence of a lifecycle join. */
        EXPECT_EQ(
            device_to_host_slot.poll(), MappedTransferProgress::Pending);
        EXPECT_EQ(
            host_to_device_slot.poll(), MappedTransferProgress::Pending);

        drainBothSlots(epoch, device_to_host_slot, host_to_device_slot);
        EXPECT_EQ(
            std::memcmp(
                mapped_destination->mutableHostData(),
                d2h_expected.data(),
                kLargeBytes),
            0);

        std::vector<std::uint8_t> h2d_actual(kLargeBytes);
        context.submitAndWait(
            [&]
            {
                if (!backend->deviceToHost(
                        h2d_actual.data(),
                        device_destination->deviceData(),
                        kLargeBytes,
                        device.gpu_ordinal(),
                        inference_stream))
                {
                    throw std::runtime_error(
                        "Could not collect H2D no-join output");
                }
                context.synchronizeStream(inference_stream);
                inference_graph.reset();
                context.destroyEvent(inference_terminal);
                inference_terminal = nullptr;
            });
        EXPECT_EQ(h2d_actual, h2d_expected);

        const auto stats = epoch->stats();
        EXPECT_EQ(stats.commands_published, 2u);
        EXPECT_EQ(stats.commands_completed, 2u);
        EXPECT_EQ(stats.copy_submissions, 2u);
        EXPECT_EQ(stats.command_failures, 0u);
    }

    /**
     * @brief Reproduce the four-GPU command BOM without per-slot GPU resources.
     *
     * One GPU in an all-host-relayed four-GPU topology participates in three
     * incoming and three outgoing directed edges. Each edge has three expert
     * projections and two pipeline slots for every configured migration cycle.
     * The command directory must retain that complete topology identity, while
     * the physical stream/event pool remains bounded by the cycle-slot budget.
     */
    void proveTopologySizedDirectoryUsesBoundedExecutionLanes(DeviceId device)
    {
        constexpr std::size_t kMigrationCycleSlots = 49u;
        constexpr std::size_t kDirectedIncidentEdges = 6u;
        constexpr std::size_t kExpertProjections = 3u;
        constexpr std::size_t kPipelineSlots = 2u;
        constexpr std::size_t kTopologyCommandSlots =
            kMigrationCycleSlots * kDirectedIncidentEdges *
            kExpertProjections * kPipelineSlots;

        IBackend *const backend = getBackendFor(device);
        ASSERT_NE(backend, nullptr);
        if (backend->deviceCount() <= device.gpu_ordinal())
            GTEST_SKIP() << device.toString() << " is unavailable";

        auto epoch = MappedTransferProgressEpoch::create({
            .device = device,
            .slot_capacity = kTopologyCommandSlots,
            .execution_lane_capacity = kMigrationCycleSlots,
            .execution_streams =
                TransferEngine::instance()
                    .allocatePersistentTransferExecutionLanes(
                        4u,
                        device,
                        "mapped_progress_topology_sized_directory"),
            .maximum_bytes = 4096u,
            .name = "integration_topology_sized_directory_" +
                    device.toString(),
            .perf_device = device.toString(),
        });
        ASSERT_NE(epoch, nullptr);
        EXPECT_EQ(epoch->slotCapacity(), kTopologyCommandSlots);
        EXPECT_EQ(epoch->executionLaneCapacity(), kMigrationCycleSlots);
        EXPECT_EQ(epoch->executionStreamCapacity(), 4u);
        EXPECT_NE(epoch->executionStream(), nullptr);
        EXPECT_GT(epoch->slotCapacity(), epoch->executionLaneCapacity());
        EXPECT_EQ(epoch->stats().slots_reserved, 0u);
    }

    /**
     * @brief Prove queued permanent commands drain through one execution lane.
     *
     * Four distinct mapped destinations deliberately exceed the one-lane
     * physical pool. The first progress pass may submit exactly one copy; later
     * event-polled passes must preserve every command generation and byte while
     * recycling that same stream/event pair.
     */
    void proveBoundedExecutionLaneQueueIsByteExact(DeviceId device)
    {
        constexpr std::size_t kCommandCount = 4u;
        constexpr std::size_t kBytes = 64u * 1024u + 29u;

        IBackend *const backend = getBackendFor(device);
        ASSERT_NE(backend, nullptr);
        if (backend->deviceCount() <= device.gpu_ordinal())
            GTEST_SKIP() << device.toString() << " is unavailable";

        auto &context = GPUDeviceContextPool::instance().getContext(device);
        TransferEngine transfer_engine;
        auto device_source = transfer_engine.allocateDeviceTransferBuffer(
            kBytes, device);
        ASSERT_NE(device_source, nullptr);
        const auto expected = makePattern(kBytes, 0xd3u);

        void *setup_stream = nullptr;
        context.submitAndWait(
            [&]
            {
                setup_stream = context.getOrCreateAuxiliaryStream(
                    "mapped_transfer_progress_queue_setup:" +
                        device.toString(),
                    GPUAuxiliaryStreamSchedulingClass::Normal);
                if (!setup_stream ||
                    !backend->hostToDevice(
                        device_source->mutableDeviceData(),
                        expected.data(),
                        kBytes,
                        device.gpu_ordinal(),
                        setup_stream))
                {
                    throw std::runtime_error(
                        "Could not initialize bounded-lane queue source");
                }
                context.synchronizeStream(setup_stream);
            });

        auto epoch = MappedTransferProgressEpoch::create({
            .device = device,
            .slot_capacity = kCommandCount,
            .execution_lane_capacity = 1u,
            .execution_streams =
                TransferEngine::instance()
                    .allocatePersistentTransferExecutionLanes(
                        1u,
                        device,
                        "mapped_progress_bounded_lane_queue"),
            .maximum_bytes = kBytes,
            .name = "integration_bounded_lane_queue_" + device.toString(),
            .perf_device = device.toString(),
        });
        const DeviceId mapped_devices[] = {device};
        std::vector<std::shared_ptr<MappedHostTransferRegion>> destinations;
        std::vector<MappedTransferProgressSlot> slots;
        destinations.reserve(kCommandCount);
        slots.reserve(kCommandCount);
        for (std::size_t index = 0u; index < kCommandCount; ++index)
        {
            auto destination = transfer_engine.allocateMappedHostRegion(
                kBytes, mapped_devices);
            ASSERT_TRUE(destination && destination->isBound());
            std::memset(destination->mutableHostData(), 0, kBytes);
            slots.push_back(epoch->reserveSlot(
                MappedTransferDirection::DeviceToHost,
                destination,
                "queued_d2h_" + std::to_string(index)));
            destinations.push_back(std::move(destination));
        }

        for (auto &slot : slots)
        {
            EXPECT_EQ(
                slot.publishDeviceToMappedHost(
                    device_source->deviceData(), kBytes, 0u, kBytes),
                1u);
        }
        ASSERT_TRUE(epoch->submitOutstandingProgress());
        EXPECT_EQ(epoch->stats().copy_submissions, 1u)
            << "One execution lane must not submit several commands at once";

        std::vector<MappedTransferProgress> progress(
            kCommandCount, MappedTransferProgress::Pending);
        const auto deadline = std::chrono::steady_clock::now() +
                              std::chrono::seconds(10);
        while (std::any_of(
                   progress.begin(), progress.end(),
                   [](MappedTransferProgress value)
                   { return value == MappedTransferProgress::Pending; }) &&
               std::chrono::steady_clock::now() < deadline)
        {
            ASSERT_TRUE(epoch->submitOutstandingProgress());
            for (std::size_t index = 0u; index < kCommandCount; ++index)
            {
                if (progress[index] == MappedTransferProgress::Pending)
                    progress[index] = slots[index].poll();
            }
            std::this_thread::yield();
        }

        for (std::size_t index = 0u; index < kCommandCount; ++index)
        {
            EXPECT_EQ(progress[index], MappedTransferProgress::Ready)
                << "slot=" << index;
            EXPECT_EQ(
                std::memcmp(
                    destinations[index]->mutableHostData(),
                    expected.data(),
                    kBytes),
                0)
                << "slot=" << index;
        }
        const auto stats = epoch->stats();
        EXPECT_EQ(stats.commands_published, kCommandCount);
        EXPECT_EQ(stats.commands_completed, kCommandCount);
        EXPECT_EQ(stats.copy_submissions, kCommandCount);
        EXPECT_EQ(stats.command_failures, 0u);
    }

    /** CPU weight-copy participants sharing the same maintenance stream pool. */
    enum class CPUWeightEdgeProof { None, Local, Remote };

    /** Native consumer fan-out attached to the held inference publication. */
    enum class InferenceObservationTopology
    {
        ProducerOnly, ///< No external stream is waiting for the graph terminal.
        FutureConsumers, ///< Native observers await a not-yet-completed producer.
    };

    /** Resource creation and native recording need not use the same host worker. */
    enum class CaptureRecordingOwner { DeviceWorker, TopologyWorker };

    /**
     * @brief Prove maintenance progresses while an unrelated graph awaits a peer.
     * @param device Exact backend/device under test, independent of tier role.
     * @param direction Independent background-copy direction to prove.
     * @param cpu_edge Optional real CPU weight lane preceding relay work on each stream.
     * @param precision Floating descriptor precision for the local CPU edge.
     * @param observation Whether native observers await the held graph terminal.
     * @param recording_owner Dedicated context worker or production TP worker.
     *
     * The graph contains a real mapped timeline wait followed by a D2H node.
     * Its pending copy can occupy a physical CUDA copy connection even though
     * the maintenance streams have no dependency on this graph. A topology-sized
     * stream pool makes this deterministic rather than relying on one lucky
     * stream assignment. We always release the deliberate wait before checking
     * assertions or retiring slots, so a failing proof cannot strand GPU work.
     */
    void proveMaintenanceProgressAcrossBlockedInference(
        DeviceId device, MappedTransferDirection direction,
        CPUWeightEdgeProof cpu_edge = CPUWeightEdgeProof::None,
        TensorType precision = TensorType::FP32,
        InferenceObservationTopology observation =
            InferenceObservationTopology::ProducerOnly,
        CaptureRecordingOwner recording_owner = CaptureRecordingOwner::DeviceWorker)
    {
        SCOPED_TRACE("edge=" + std::to_string(static_cast<int>(cpu_edge)) +
                     " direction=" + std::to_string(static_cast<int>(direction)) +
                     " precision=" + std::to_string(static_cast<int>(precision)) +
                     " observation=" + std::to_string(static_cast<int>(observation)));
        constexpr std::size_t slots = 32u;
        constexpr std::size_t bytes = 4u * 1024u * 1024u + 13u;
        auto *backend = getBackendFor(device);
        ASSERT_NE(backend, nullptr);
        if (backend->deviceCount() <= device.gpu_ordinal())
            GTEST_SKIP() << device.toString() << " unavailable";
        auto &context = GPUDeviceContextPool::instance().getContext(device);
        TransferEngine transfers;
        const DeviceId devices[] = {device};
        auto control = transfers.allocateMappedHostRegion(4096u, devices);
        auto held_destination = transfers.allocateMappedHostRegion(bytes, devices);
        auto source = transfers.allocateDeviceTransferBuffer(bytes, device);
        const auto expected = makePattern(bytes, 0x37u);
        auto *word = static_cast<std::uint64_t *>(control->mutableHostData());
        std::atomic_ref<std::uint64_t>(*word).store(0u, std::memory_order_release);
        std::vector<std::shared_ptr<MappedHostTransferRegion>> destinations;
        std::vector<std::shared_ptr<DeviceTransferBuffer>> device_destinations;
        for (std::size_t i = 0; i < slots; ++i)
        {
            destinations.push_back(transfers.allocateMappedHostRegion(bytes, devices));
            std::memcpy(destinations.back()->mutableHostData(), expected.data(), bytes);
            if (direction == MappedTransferDirection::HostToDevice)
                device_destinations.push_back(transfers.allocateDeviceTransferBuffer(bytes, device));
            else
                std::memset(destinations.back()->mutableHostData(), 0, bytes);
        }
        const auto execution = transfers.allocatePersistentTransferExecutionLanes(
            slots, device, "progress_across_held_inference");
        // Every CUDA client shares the same captured service. Local and remote
        // CPU edges reserve independent read/write identities, not new queues.
        auto epoch = MappedTransferProgressEpoch::create({
            .device = device,
            .slot_capacity = slots * (cpu_edge == CPUWeightEdgeProof::None ? 1u : 3u),
            .execution_lane_capacity = slots,
            .execution_streams = execution,
            .maximum_bytes = bytes,
            .name = "progress_across_held_inference_" + device.toString(),
            .perf_device = device.toString(),
        });
        const auto client_progress = device.is_cuda()
            ? BackgroundTransferProgressBinding::graphService(epoch)
            : BackgroundTransferProgressBinding::nativeStream();
        // Production shares this pool across GPU relays and local/remote CPU
        // edges. A safe relay is insufficient if another participant inserts
        // a stranded native DMA copy ahead of it on that exact stream.
        constexpr std::size_t cpu_bytes = 4u * 1024u * 1024u;
        std::vector<PersistentTransferStagingSlice> cpu_staging;
        std::vector<std::unique_ptr<ExpertTierWeightTransferLane>> local_edges;
        std::vector<std::unique_ptr<MoEOverlayGpuRemoteProjectionLane>> remote_edges;
        std::vector<std::vector<std::uint8_t>> cpu_outputs;
        std::vector<std::shared_ptr<DeviceTransferBuffer>> cpu_device_outputs;
        if (cpu_edge != CPUWeightEdgeProof::None)
        {
            cpu_staging = transfers.allocatePersistentTransferStagingSlices(
                cpu_bytes, slots, device);
            for (std::size_t i = 0; i < slots; ++i)
            {
                cpu_outputs.emplace_back(cpu_bytes, 0u);
                cpu_device_outputs.push_back(
                    transfers.allocateDeviceTransferBuffer(cpu_bytes, device));
                if (cpu_edge == CPUWeightEdgeProof::Local)
                {
                    local_edges.push_back(std::make_unique<ExpertTierWeightTransferLane>(
                        ExpertTierWeightTransferLane::Config{
                            .device = device, .staging = cpu_staging[i],
                            .execution = execution[i],
                            .progress = client_progress,
                            .lane_name = "held_local_cpu_edge_" + std::to_string(i),
                            .collect_timing_measurements = true}));
                    std::string error;
                    ASSERT_TRUE(local_edges.back()->materialize(&error)) << error;
                }
                else
                {
                    remote_edges.push_back(std::make_unique<MoEOverlayGpuRemoteProjectionLane>(
                        MoEOverlayGpuRemoteProjectionLane::Config{
                            .device = device, .staging = cpu_staging[i],
                            .execution = execution[i],
                            .progress = client_progress,
                            .lane_name = "held_remote_cpu_edge_" + std::to_string(i)}));
                    std::string error;
                    ASSERT_TRUE(remote_edges.back()->materialize(&error)) << error;
                    ASSERT_TRUE(remote_edges.back()->tryAcquire(remote_edges.back().get()));
                    ASSERT_TRUE(remote_edges.back()->bindSourceReadiness(
                        remote_edges.back().get(),
                        ExpertTierSourceReadiness::publishedResidencyBank(1u)));
                }
            }
        }
        std::vector<MappedTransferProgressSlot> commands;
        for (std::size_t i = 0; i < slots; ++i)
            commands.push_back(epoch->reserveSlot(direction,
                destinations[i], "held_inference_independent_" + std::to_string(i)));
        void *stream = nullptr;
        void *producer_ready = nullptr;
        std::vector<void *> observation_streams;
        constexpr std::size_t kObservedConsumerCount = 32u;
        constexpr std::size_t kObservationBytes = 128u;
        auto observation_bytes =
            observation == InferenceObservationTopology::FutureConsumers
                ? transfers.allocateMappedHostRegion(
                      kObservedConsumerCount * kObservationBytes, devices)
                : std::shared_ptr<MappedHostTransferRegion>{};
        std::unique_ptr<IGPUGraphCapture> graph;
        const auto branch_factory = epoch->graphBranchFactory();
        auto progress_branch = branch_factory.valid() ? branch_factory.create() : nullptr;
        context.submitAndWait([&] {
            stream = context.getOrCreateAuxiliaryStream("held_inference_graph");
            if (observation == InferenceObservationTopology::FutureConsumers)
            {
                // These queues model independently admitted prefix/live-state
                // readers. Maintenance has no logical dependency on any of them.
                producer_ready = context.createEvent();
                if (!producer_ready)
                    throw std::runtime_error("held inference publication allocation failed");
                for (std::size_t i = 0u; i < kObservedConsumerCount; ++i)
                {
                    auto *consumer = context.getOrCreateAuxiliaryStream(
                        "held_inference_observer_" + std::to_string(i));
                    if (!consumer)
                        throw std::runtime_error("held inference observer allocation failed");
                    observation_streams.push_back(consumer);
                }
            }
            if (!stream || !backend->hostToDevice(source->mutableDeviceData(),
                    expected.data(), bytes, device.gpu_ordinal(), stream))
                throw std::runtime_error("held inference source setup failed");
            context.synchronizeStream(stream);
            graph = context.createGraphCapture(stream);
        });
        auto record = [&] {
            if (!graph)
                throw std::runtime_error("held inference graph allocation failed");
            // Use production's complete capture scope on the actual recording
            // thread. Resource ownership does not authorize moving half of a
            // native capture transaction onto a different host worker.
            ScopedBackendGraphCapture capture(context, *graph, "held inference progress proof");
            if (!capture.begin())
                throw std::runtime_error("held inference capture failed");
            if (!backend->streamWaitTimelineSignal64(stream, control->deviceAlias(device),
                    1u, device.gpu_ordinal()) ||
                !backend->deviceToHostOnStream(held_destination->mutableHostData(),
                    source->deviceData(), bytes, device.gpu_ordinal(), stream))
                throw std::runtime_error("held inference graph construction failed");
            capture.finish();
            if (progress_branch && !progress_branch->attach(*graph))
                throw std::runtime_error("held inference progress attachment failed");
            if (!graph->instantiate())
                throw std::runtime_error("held inference graph instantiation failed");
        };
        if (recording_owner == CaptureRecordingOwner::TopologyWorker)
        {
            TPWorkerPool workers(1u);
            workers.dispatch([&](std::size_t) {
                EXPECT_FALSE(context.ownsCurrentThread());
                record();
                return true;
            });
            for (const auto &result : workers.collectAll(30'000))
            {
                if (result.exception) std::rethrow_exception(result.exception);
                ASSERT_TRUE(result.completed && result.success);
            }
        }
        else
            context.submitAndWait(record);
        // Scope exit releases the CPU-owned signal even if a backend throws.
        auto release = [&](void *) {
            std::atomic_ref<std::uint64_t>(*word).store(1u, std::memory_order_release);
            context.submitAndWait([&] {
                // Always drain the deliberately held producer and all readers,
                // including when the progress assertion is about to fail.
                context.synchronizeStream(stream);
                for (void *consumer : observation_streams)
                    context.synchronizeStream(consumer);
                if (producer_ready)
                {
                    context.destroyEvent(producer_ready);
                    producer_ready = nullptr;
                }
            });
        };
        std::unique_ptr<void, decltype(release)> release_guard(word, release);
        context.submitAndWait([&] {
            if (!graph->launch()) throw std::runtime_error("held inference launch failed");
            if (producer_ready)
            {
                // Record exactly as a production forward-output publication:
                // after native graph submission, before its terminal is ready.
                if (!context.recordEventChecked(producer_ready, stream))
                    throw std::runtime_error("held inference publication failed");
                for (std::size_t index = 0u;
                     index < observation_streams.size(); ++index)
                {
                    void *consumer = observation_streams[index];
                    if (!context.waitEventChecked(producer_ready, consumer))
                        throw std::runtime_error("held inference observation join failed");
                    // A wait with no following operation can remain a runtime
                    // bookkeeping edge. Queue actual observer work, just as
                    // prefix archival queues gather kernels after its join.
                    if (!backend->copyDeviceVisibleRegionByKernelOnStream(
                            observation_bytes->deviceAlias(
                                device, index * kObservationBytes),
                            source->deviceData(), kObservationBytes,
                            device.gpu_ordinal(), consumer))
                        throw std::runtime_error("held inference observation copy failed");
                }
            }
        });
        bool submission_ok = true;
        context.submitAndWait([&] {
            for (std::size_t i = 0; i < cpu_staging.size(); ++i)
            {
                const bool download = direction == MappedTransferDirection::DeviceToHost;
                if (cpu_edge == CPUWeightEdgeProof::Local)
                {
                    const ContiguousFloatingPointWeightDescriptor descriptor{
                        .data = download ? source->deviceData()
                                         : cpu_device_outputs[i]->deviceData(),
                        .type = precision, .n = 1024,
                        .k = precision == TensorType::FP32 ? 1024 : 2048,
                        .bytes = cpu_bytes};
                    submission_ok = (download
                        ? local_edges[i]->startGpuToCpuContiguous(descriptor, cpu_outputs[i],
                            ExpertTierSourceReadiness::publishedResidencyBank(1u))
                        : local_edges[i]->startCpuToGpuContiguous(
                            std::span(expected.data(), cpu_bytes), descriptor)) && submission_ok;
                }
                else
                {
                    submission_ok = (download
                        ? remote_edges[i]->submitGpuBlobRead(remote_edges[i].get(),
                            static_cast<const std::uint8_t *>(source->deviceData()), cpu_bytes)
                        : remote_edges[i]->submitGpuBlobWrite(remote_edges[i].get(),
                            static_cast<std::uint8_t *>(cpu_device_outputs[i]->mutableDeviceData()),
                            std::span(expected.data(), cpu_bytes))) && submission_ok;
                }
            }
        });
        for (std::size_t i = 0u; i < commands.size(); ++i)
        {
            // A permanent command ID is not this client's local buffer index:
            // CPU/remote clients have already reserved IDs in the same epoch.
            auto &command = commands[i];
            if (direction == MappedTransferDirection::DeviceToHost)
                (void)command.publishDeviceToMappedHost(source->deviceData(), bytes, 0u, bytes);
            else
                (void)command.publishMappedHostToDevice(
                    device_destinations[i]->mutableDeviceData(), bytes, 0u, bytes);
        }

        std::array<bool, slots> completed{};
        std::array<bool, slots> cpu_completed{};
        std::size_t ready = 0u;
        std::size_t cpu_ready = 0u;
        auto poll = [&] {
            submission_ok = epoch->submitOutstandingProgress() && submission_ok;
            for (std::size_t i = 0; i < cpu_staging.size(); ++i)
                if (!cpu_completed[i])
                {
                    const bool complete = cpu_edge == CPUWeightEdgeProof::Local
                        ? local_edges[i]->poll() == ExpertTierWeightTransferProgress::Ready
                        : remote_edges[i]->poll(remote_edges[i].get()) == MoEOverlayGpuRemoteLaneProgress::Ready;
                    if (complete) { cpu_completed[i] = true; ++cpu_ready; }
                }
            for (std::size_t i = 0; i < slots; ++i)
                if (!completed[i] && commands[i].poll() == MappedTransferProgress::Ready) {
                    completed[i] = true;
                    ++ready;
                }
        };
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
        while ((ready != slots || cpu_ready != cpu_staging.size()) &&
               std::chrono::steady_clock::now() < deadline) {
            poll();
            std::this_thread::yield();
        }
        const auto before_release = ready;
        const auto cpu_before_release = cpu_ready;
        release_guard.reset();
        const auto drain_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while ((ready != slots || cpu_ready != cpu_staging.size()) &&
               std::chrono::steady_clock::now() < drain_deadline) {
            poll();
            std::this_thread::yield();
        }
        context.submitAndWait([&] {
            if (direction == MappedTransferDirection::HostToDevice)
            {
                for (std::size_t i = 0; i < slots; ++i)
                    if (!backend->deviceToHostOnStream(destinations[i]->mutableHostData(),
                            device_destinations[i]->deviceData(), bytes, device.gpu_ordinal(), stream))
                        throw std::runtime_error("held inference H2D verification failed");
                for (std::size_t i = 0; i < cpu_staging.size(); ++i)
                    if (!backend->deviceToHostOnStream(cpu_outputs[i].data(),
                            cpu_device_outputs[i]->deviceData(), cpu_bytes,
                            device.gpu_ordinal(), stream))
                        throw std::runtime_error("held inference CPU-edge H2D verification failed");
                context.synchronizeStream(stream); // Diagnostic host observation only.
            }
            graph.reset();
        });
        EXPECT_TRUE(submission_ok);
        EXPECT_EQ(before_release, slots)
            << "Independent maintenance was serialized behind a peer-held inference copy";
        ASSERT_EQ(ready, slots) << "Maintenance failed to drain after peer release";
        EXPECT_EQ(cpu_before_release, cpu_staging.size())
            << "CPU weight edge was serialized behind peer-held inference";
        ASSERT_EQ(cpu_ready, cpu_staging.size());
        for (std::size_t i = 0; i < cpu_staging.size(); ++i)
        {
            const void *actual = cpu_edge == CPUWeightEdgeProof::Remote &&
                    direction == MappedTransferDirection::DeviceToHost
                ? cpu_staging[i].mutablePinnedData() : cpu_outputs[i].data();
            EXPECT_EQ(std::memcmp(actual, expected.data(), cpu_bytes), 0);
            if (cpu_edge == CPUWeightEdgeProof::Remote)
                EXPECT_TRUE(remote_edges[i]->release(remote_edges[i].get()));
            else
            {
                const auto measurement = local_edges[i]->stats().last_measurement;
                EXPECT_EQ(measurement.bytes, cpu_bytes);
                EXPECT_GT(measurement.device_nanoseconds, 0u)
                    << "Economy must measure actual copy work on both backends";
            }
        }
        for (const auto &destination : destinations)
            EXPECT_EQ(std::memcmp(destination->mutableHostData(), expected.data(), bytes), 0);
        if (observation_bytes)
            for (std::size_t index = 0u; index < observation_streams.size(); ++index)
                EXPECT_EQ(std::memcmp(
                    static_cast<const std::uint8_t *>(observation_bytes->mutableHostData()) +
                        index * kObservationBytes,
                    expected.data(), kObservationBytes), 0)
                    << "Future observer did not execute after producer release: " << index;
    }

    /** @brief Cover local floating descriptors and format-agnostic remote wire blobs. */
    void proveCPUWeightEdgesShareProgressSafePool(DeviceId device)
    {
        for (const auto observation : {InferenceObservationTopology::ProducerOnly,
                                       InferenceObservationTopology::FutureConsumers})
        for (const auto direction : {MappedTransferDirection::DeviceToHost,
                                     MappedTransferDirection::HostToDevice})
        {
            for (const auto precision : {TensorType::FP16, TensorType::BF16, TensorType::FP32})
                proveMaintenanceProgressAcrossBlockedInference(
                    device, direction, CPUWeightEdgeProof::Local, precision, observation);
            proveMaintenanceProgressAcrossBlockedInference(
                device, direction, CPUWeightEdgeProof::Remote, TensorType::FP32, observation);
        }
    }

    /**
     * @brief A blocked producer gates only its own command, never unrelated work.
     *
     * The source's exact event is downstream of a captured graph held by a
     * CPU-owned test signal. One independent H2D command must finish before
     * release, while the dependent D2H destination remains untouched. This
     * exercises the admission edge through the same authority on both GPUs.
     */
    void proveExactProducerAdmission(DeviceId device)
    {
        auto *backend = getBackendFor(device);
        ASSERT_NE(backend, nullptr);
        if (backend->deviceCount() <= device.gpu_ordinal()) GTEST_SKIP();
        auto &context = GPUDeviceContextPool::instance().getContext(device);
        TransferEngine transfers;
        const DeviceId devices[] = {device};
        constexpr std::size_t bytes = 4099u;
        auto control = transfers.allocateMappedHostRegion(64u, devices);
        auto input = transfers.allocateMappedHostRegion(bytes, devices);
        auto output = transfers.allocateMappedHostTransferSlices(bytes, 1u, device).front();
        auto source = transfers.allocateDeviceTransferBuffer(bytes, device);
        auto destination = transfers.allocateDeviceTransferBuffer(bytes, device);
        const auto expected = makePattern(bytes, 0x39u);
        std::memcpy(input->mutableHostData(), expected.data(), bytes);
        std::memset(output->mutableHostData(), 0xcd, bytes);
        auto *signal = static_cast<std::uint64_t *>(control->mutableHostData());
        std::atomic_ref<std::uint64_t>(*signal).store(0u, std::memory_order_release);
        auto epoch = MappedTransferProgressEpoch::create({
            .device = device, .slot_capacity = 2u, .execution_lane_capacity = 2u,
            .execution_streams = transfers.allocatePersistentTransferExecutionLanes(
                2u, device, "producer_dependency_service"),
            .maximum_bytes = bytes + 13u, .name = "producer_dependency_service"});
        auto dependent = epoch->reserveSlot(MappedTransferDirection::DeviceToHost, output);
        auto independent = epoch->reserveSlot(MappedTransferDirection::HostToDevice, input);
        // A slot may be smaller than the epoch's maximum; its own bounds still
        // reject publication before any address becomes visible to the worker.
        EXPECT_THROW(dependent.publishDeviceToMappedHost(source->deviceData(),
            bytes + 1u, 0u, bytes + 1u), std::invalid_argument);
        auto factory = epoch->graphBranchFactory();
        auto branch = factory.valid() ? factory.create() : nullptr;
        std::unique_ptr<IGPUGraphCapture> graph;
        void *primary = nullptr;
        void *producer = nullptr;
        auto release = [&](void *) {
            std::atomic_ref<std::uint64_t>(*signal).store(1u, std::memory_order_release);
        };
        std::unique_ptr<void, decltype(release)> release_guard(signal, release);
        context.submitAndWait([&] {
            primary = context.getOrCreateAuxiliaryStream("exact_producer_dependency_graph");
            producer = context.createEvent();
            if (!primary || !producer || !backend->hostToDevice(source->mutableDeviceData(),
                    expected.data(), bytes, device.gpu_ordinal(), primary))
                throw std::runtime_error("Producer admission setup failed");
            context.synchronizeStream(primary); // Test oracle's initial source only.
            graph = context.createGraphCapture(primary);
            if (!graph) throw std::runtime_error("Producer admission graph allocation failed");
            ScopedBackendGraphCapture capture(context, *graph, "producer admission proof");
            if (!capture.begin() ||
                !backend->streamWaitTimelineSignal64(primary, control->deviceAlias(device),
                    1u, device.gpu_ordinal()))
                throw std::runtime_error("Producer admission graph construction failed");
            capture.finish();
            if (branch && !branch->attach(*graph))
                throw std::runtime_error("Producer admission graph decoration failed");
            if (!graph->instantiate() || !graph->launch())
                throw std::runtime_error("Producer admission graph launch failed");
            if (!context.recordEventChecked(producer, primary))
                throw std::runtime_error("Producer admission event publication failed");
        });
        dependent.publishDeviceToMappedHost(source->deviceData(), bytes, 0u, bytes,
            TransferProducerDependency::afterEvent(producer));
        independent.publishMappedHostToDevice(destination->mutableDeviceData(), bytes, 0u, bytes);
        auto independent_progress = MappedTransferProgress::Pending;
        bool submissions_ok = true;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
        while (independent_progress == MappedTransferProgress::Pending &&
               std::chrono::steady_clock::now() < deadline)
        {
            submissions_ok = epoch->submitOutstandingProgress() && submissions_ok;
            independent_progress = independent.poll();
            std::this_thread::yield();
        }
        EXPECT_EQ(independent_progress, MappedTransferProgress::Ready);
        EXPECT_EQ(dependent.poll(), MappedTransferProgress::Pending);
        EXPECT_TRUE(std::all_of(static_cast<const std::uint8_t *>(output->mutableHostData()),
            static_cast<const std::uint8_t *>(output->mutableHostData()) + bytes,
            [](std::uint8_t value) { return value == 0xcd; }));
        release_guard.reset();
        auto dependent_progress = MappedTransferProgress::Pending;
        const auto drain = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while ((dependent_progress == MappedTransferProgress::Pending ||
                independent_progress == MappedTransferProgress::Pending) &&
               std::chrono::steady_clock::now() < drain)
        {
            submissions_ok = epoch->submitOutstandingProgress() && submissions_ok;
            if (dependent_progress == MappedTransferProgress::Pending) dependent_progress = dependent.poll();
            if (independent_progress == MappedTransferProgress::Pending) independent_progress = independent.poll();
            std::this_thread::yield();
        }
        EXPECT_TRUE(submissions_ok);
        EXPECT_EQ(dependent_progress, MappedTransferProgress::Ready);
        EXPECT_EQ(std::memcmp(output->mutableHostData(), expected.data(), bytes), 0);
        context.submitAndWait([&] {
            context.synchronizeEvent(producer); // Retire the test producer's captured lifetime.
            graph.reset();
            context.destroyEvent(producer);
        });
    }

    /** Distinct proofs of the public maintenance path and shared kernel bytes. */
    enum class MappedCopyProofMechanism { Background, DeviceKernel };

    /**
     * @brief Sweep both byte-copy directions, vector tails and independent offsets.
     * @param device Exact GPU owning the prepared stream and device allocation.
     *
     * Whole-buffer canaries detect overruns. Every case resets the destination
     * to a different value, so a missing or partial copy cannot pass. This is
     * format-agnostic physical-byte coverage: quantized and floating experts use
     * the same primitive without interpretation or arithmetic.
     */
    void proveBackgroundCopyRegions(DeviceId device)
    {
        auto *backend = getBackendFor(device);
        ASSERT_NE(backend, nullptr);
        if (backend->deviceCount() <= device.gpu_ordinal()) GTEST_SKIP();
        TransferEngine engine;
        constexpr std::size_t capacity = 1024u;
        const DeviceId devices[] = {device};
        auto staging = engine.allocateMappedHostRegion(capacity, devices);
        auto observation = engine.allocateMappedHostRegion(capacity, devices);
        auto storage = engine.allocateDeviceTransferBuffer(capacity, device);
        const auto lanes = engine.allocatePersistentTransferExecutionLanes(
            1u, device, "bounded_byte_copy_sweep");
        const auto staging_slices = engine.allocatePersistentTransferStagingSlices(
            capacity, 2u, device);
        const auto &lane = lanes.front();
        const auto pattern = makePattern(capacity, 0x43u);
        auto &context = GPUDeviceContextPool::instance().getContext(device);
        context.submitAndWait([&] {
            void *terminal = context.createEvent();
            ASSERT_NE(terminal, nullptr);
            for (const auto direction : {MappedTransferDirection::DeviceToHost,
                                         MappedTransferDirection::HostToDevice})
            // Background submission has different native queue semantics on
            // each backend. Independently exercise the shared kernel primitive
            // too, including HIP's kernel-only payload path and odd byte tails.
            for (const auto mechanism : {MappedCopyProofMechanism::Background,
                                         MappedCopyProofMechanism::DeviceKernel})
            for (const std::size_t bytes : {1u, 15u, 16u, 17u, 129u, 511u})
            for (const auto offsets : {std::array<std::size_t, 2>{0u, 0u},
                                      {16u, 32u}, {1u, 0u}, {0u, 7u}, {3u, 15u}})
            {
                const auto device_offset = offsets[0];
                const auto mapped_offset = offsets[1];
                std::vector<std::uint8_t> expected(capacity, 0xcdu);
                if (direction == MappedTransferDirection::DeviceToHost)
                {
                    std::memcpy(observation->mutableHostData(), pattern.data(), capacity);
                    std::memset(staging->mutableHostData(), 0xcd, capacity);
                    std::memcpy(expected.data() + mapped_offset,
                                pattern.data() + device_offset, bytes);
                }
                else
                {
                    std::memset(observation->mutableHostData(), 0xcd, capacity);
                    std::memcpy(staging->mutableHostData(), pattern.data(), capacity);
                    std::memcpy(expected.data() + device_offset,
                                pattern.data() + mapped_offset, bytes);
                }
                engine.enqueueMappedHostToPersistentDeviceRegion(*observation, 0u,
                    storage->mutableDeviceData(), capacity, 0u, capacity, device, lane.stream());
                if (mechanism == MappedCopyProofMechanism::Background)
                    engine.enqueueBackgroundMappedCopy(lane, direction,
                        storage->mutableDeviceData(), capacity, device_offset,
                        *staging, mapped_offset, bytes);
                else
                {
                    // Test-only low-level boundary: verify the exact shared
                    // kernel in both directions without claiming it is HIP's
                    // production maintenance mechanism. Owners and preparation
                    // still come exclusively from TransferEngine.
                    void *device_bytes = storage->mutableDeviceData(device_offset);
                    void *mapped_bytes = staging->deviceAlias(device, mapped_offset);
                    const bool download = direction == MappedTransferDirection::DeviceToHost;
                    ASSERT_TRUE(backend->copyDeviceVisibleRegionByKernelOnStream(
                        download ? mapped_bytes : device_bytes,
                        download ? device_bytes : mapped_bytes,
                        bytes, device.gpu_ordinal(), lane.stream()));
                }
                if (direction == MappedTransferDirection::HostToDevice)
                    engine.enqueuePersistentDeviceRegionToMappedHost(storage->deviceData(),
                        capacity, 0u, *observation, 0u, capacity, device, lane.stream());
                ASSERT_TRUE(context.recordEventChecked(terminal, lane.stream()));
                ASSERT_TRUE(context.synchronizeEventChecked(terminal));
                const auto &actual = direction == MappedTransferDirection::DeviceToHost
                    ? staging : observation;
                EXPECT_EQ(std::memcmp(actual->mutableHostData(), expected.data(), capacity), 0)
                    << "bytes=" << bytes << " device_offset=" << device_offset
                    << " mapped_offset=" << mapped_offset;
            }
            context.destroyEvent(terminal);
        });
        EXPECT_THROW(engine.enqueueBackgroundMappedCopy(lane,
            MappedTransferDirection::DeviceToHost, storage->mutableDeviceData(),
            capacity, capacity - 1u, *staging, 0u, 2u), std::out_of_range);
        EXPECT_THROW(engine.enqueueBackgroundMappedCopy(lane,
            MappedTransferDirection::HostToDevice, storage->mutableDeviceData(),
            capacity, 0u, *staging, staging->sizeBytes() - 1u, 2u), std::out_of_range);
        EXPECT_THROW(engine.enqueueBackgroundMappedCopy(lane,
            static_cast<MappedTransferDirection>(9), storage->mutableDeviceData(),
            capacity, 0u, *staging, 0u, 1u), std::invalid_argument);
        std::string error;
        EXPECT_FALSE(engine.enqueueBackgroundStagingCopy(lane,
            MappedTransferDirection::HostToDevice, storage->mutableDeviceData(),
            capacity, 0u, staging_slices.front(), capacity - 1u, 2u, &error));
        EXPECT_NE(error.find("exclusive slice"), std::string::npos)
            << "An adjacent slice is not spare capacity for this transfer";
        EXPECT_FALSE(engine.enqueueBackgroundStagingCopy(lane,
            MappedTransferDirection::DeviceToHost, storage->mutableDeviceData(),
            capacity, capacity - 1u, staging_slices.front(), 0u, 2u, &error));
        EXPECT_NE(error.find("immutable region"), std::string::npos);
    }

    TEST(MappedTransferProgressEpochIntegration,
         InvalidExecutionLaneGeometryIsRejectedBeforeGPUSetup)
    {
#if defined(HAVE_CUDA)
        const DeviceId device = DeviceId::cuda(0);
#else
        const DeviceId device = DeviceId::rocm(0);
#endif
        EXPECT_THROW(
            (void)MappedTransferProgressEpoch::create({
                .device = device,
                .slot_capacity = 2u,
                .execution_lane_capacity = 0u,
                .maximum_bytes = 4096u,
                .name = "invalid_zero_execution_lanes",
            }),
            std::invalid_argument);
        EXPECT_THROW(
            (void)MappedTransferProgressEpoch::create({
                .device = device,
                .slot_capacity = 2u,
                .execution_lane_capacity = 3u,
                .maximum_bytes = 4096u,
                .name = "invalid_excess_execution_lanes",
            }),
            std::invalid_argument);
    }

#if defined(HAVE_CUDA)
    /** Production TP recording must not be mistaken for GPU resource management. */
    TEST(MappedTransferProgressEpochIntegration, CUDATopologyWorkerRecordsAndReplaysProgressBranch)
    {
        for (const auto direction : {MappedTransferDirection::DeviceToHost,
                                     MappedTransferDirection::HostToDevice})
            proveMaintenanceProgressAcrossBlockedInference(DeviceId::cuda(0), direction,
                CPUWeightEdgeProof::None, TensorType::FP32,
                InferenceObservationTopology::FutureConsumers, CaptureRecordingOwner::TopologyWorker);
    }

    /**
     * @brief Exercise the production service ABI across partial graph retirement.
     *
     * The diagnostic cursor download occurs only after the graph has joined.
     * Production never downloads or mirrors this execution state. Two retained
     * replays finish each command generation; a new generation changes every
     * source byte and poisons all destinations to expose skipped or stale work.
     */
    TEST(MappedTransferProgressEpochIntegration, CUDAGraphBoundedServiceResumesPartialCommands)
    {
        const auto device = DeviceId::cuda(0);
        auto *backend = getBackendFor(device);
        ASSERT_NE(backend, nullptr);
        if (backend->deviceCount() == 0) GTEST_SKIP() << "CUDA unavailable";
        auto &context = GPUDeviceContextPool::instance().getContext(device);
        TransferEngine transfers;
        const DeviceId devices[] = {device};
        constexpr size_t capacity = 4u;
        constexpr size_t bytes = 16u * 1024u * 1024u + 13u;
        auto inbox = transfers.allocateMappedHostRegion(capacity * 128u, devices);
        std::shared_ptr<DeviceTransferBuffer> cursors;
        auto interval = transfers.allocateDeviceTransferBuffer(sizeof(std::uint32_t), device);
        auto controls = transfers.allocateMappedHostRegion(4096u, devices);
        auto source = transfers.allocateDeviceTransferBuffer(bytes + 16u, device);
        auto *control = static_cast<std::uint64_t *>(controls->mutableHostData());
        auto *commands = static_cast<MappedTransferProgressCommand *>(inbox->mutableHostData());
        auto *receipts = reinterpret_cast<MappedTransferProgressCompletion *>(commands + capacity);
        std::vector<std::shared_ptr<MappedHostTransferRegion>> mapped;
        std::vector<std::shared_ptr<DeviceTransferBuffer>> destinations;
        for (size_t slot = 0u; slot < capacity; ++slot)
        {
            mapped.push_back(transfers.allocateMappedHostRegion(bytes + 16u, devices));
            destinations.push_back(transfers.allocateDeviceTransferBuffer(bytes + 16u, device));
        }
        void *primary = nullptr;
        void *auxiliary = nullptr;
        void *fork = nullptr;
        void *done = nullptr;
        std::unique_ptr<IGPUGraphCapture> graph;
        context.submitAndWait([&]
        {
            primary = context.getOrCreateAuxiliaryStream("partial_service_primary");
            auxiliary = context.getOrCreateAuxiliaryStream("partial_service_auxiliary",
                GPUAuxiliaryStreamSchedulingClass::BackgroundMaintenance);
            fork = context.createEvent(); done = context.createEvent();
            if (!primary || !auxiliary || !fork || !done)
                throw std::runtime_error("partial service setup failed");
            cursors = transfers.allocateMappedTransferServiceCursors(capacity, device, primary);
            context.synchronizeStream(primary); // Cold diagnostic fixture initialization.
            graph = context.createGraphCapture(primary);
            if (!graph || !graph->beginCapture())
                throw std::runtime_error("partial service capture failed");
            transfers.enqueueMappedTransferInterval(*interval, MappedTransferInterval::Open, primary);
            if (!context.recordEventChecked(fork, primary) || !context.waitEventChecked(fork, auxiliary))
                throw std::runtime_error("partial service fork failed");
            transfers.enqueueMappedTransferService(*inbox, *cursors, bytes, interval.get(),
                MappedTransferServiceRun::CapturedInterval, auxiliary);
            if (!context.recordEventChecked(done, auxiliary) ||
                !backend->streamPublishTimelineSignal64(primary, controls->deviceAlias(device, 8u),
                    1u, device.gpu_ordinal()) ||
                !backend->streamWaitTimelineSignal64(primary, controls->deviceAlias(device),
                    1u, device.gpu_ordinal()))
                throw std::runtime_error("partial service primary lifecycle failed");
            transfers.enqueueMappedTransferInterval(*interval, MappedTransferInterval::Closed, primary);
            if (!context.waitEventChecked(done, primary) || !graph->endCapture() || !graph->instantiate())
                throw std::runtime_error("partial service terminal capture failed");
        });
        auto release = [&](void *)
        {
            std::atomic_ref<std::uint64_t>(control[0]).store(1u, std::memory_order_release);
            context.submitAndWait([&] { context.synchronizeStream(primary); });
        };
        std::unique_ptr<void, decltype(release)> release_guard(control, release);
        std::uint64_t generation = 0u;
        for (const auto direction : {MappedTransferDirection::DeviceToHost, MappedTransferDirection::HostToDevice})
        for (const size_t offset : {0u, 1u})
        {
            ++generation;
            SCOPED_TRACE("generation=" + std::to_string(generation));
            const auto expected = makePattern(bytes + 16u, static_cast<std::uint32_t>(generation * 31u));
            std::vector<std::uint8_t> poison(bytes + 16u, 0xccu);
            context.submitAndWait([&]
            {
                if (!backend->hostToDevice(source->mutableDeviceData(), expected.data(), expected.size(),
                        device.gpu_ordinal(), primary))
                    throw std::runtime_error("partial service source initialization failed");
                for (size_t slot = 0u; slot < capacity; ++slot)
                {
                    const auto &initial = direction == MappedTransferDirection::HostToDevice ? expected : poison;
                    std::memcpy(mapped[slot]->mutableHostData(), initial.data(), initial.size());
                    if (!backend->hostToDevice(destinations[slot]->mutableDeviceData(), poison.data(), poison.size(),
                            device.gpu_ordinal(), primary))
                        throw std::runtime_error("partial service destination initialization failed");
                }
                context.synchronizeStream(primary);
            });
            for (size_t slot = 0u; slot < capacity; ++slot)
            {
                auto &command = commands[slot];
                command.generation_magic = mappedTransferProgressGenerationMagic(generation);
                command.generation_version = mappedTransferProgressGenerationVersion(generation);
                command.source_address = reinterpret_cast<std::uintptr_t>(
                    direction == MappedTransferDirection::HostToDevice
                        ? mapped[slot]->deviceAlias(device, offset) : source->deviceData(offset));
                command.destination_address = reinterpret_cast<std::uintptr_t>(
                    direction == MappedTransferDirection::DeviceToHost
                        ? mapped[slot]->deviceAlias(device, offset) : destinations[slot]->mutableDeviceData(offset));
                command.bytes = bytes;
                command.source_complement = ~command.source_address;
                command.destination_complement = ~command.destination_address;
                command.bytes_complement = ~command.bytes;
                std::atomic_ref<std::uint64_t>(command.generation).store(generation, std::memory_order_release);
            }
            for (unsigned replay = 0u; replay < 2u; ++replay)
            {
                std::atomic_ref<std::uint64_t>(control[0]).store(0u, std::memory_order_release);
                std::atomic_ref<std::uint64_t>(control[1]).store(0u, std::memory_order_release);
                context.submitAndWait([&] { if (!graph->launch()) throw std::runtime_error("partial service replay failed"); });
                const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
                if (replay == 0u)
                {
                    while (std::atomic_ref<std::uint64_t>(control[1]).load(std::memory_order_acquire) != 1u &&
                           std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
                    // Timing selects an adversarial close, not a performance
                    // assertion. The joined cursor below proves actual progress.
                    std::this_thread::sleep_for(std::chrono::microseconds(200));
                }
                else
                {
                    while (std::chrono::steady_clock::now() < deadline)
                    {
                        bool complete = true;
                        for (size_t slot = 0u; slot < capacity; ++slot)
                            complete &= std::atomic_ref<std::uint64_t>(receipts[slot].completed_generation)
                                .load(std::memory_order_acquire) == generation;
                        if (complete) break;
                        std::this_thread::yield();
                    }
                }
                release(control);
                std::array<MappedTransferServiceCursor, capacity> observed{};
                context.submitAndWait([&]
                {
                    if (!backend->deviceToHostOnStream(observed.data(), cursors->deviceData(), sizeof(observed),
                            device.gpu_ordinal(), primary))
                        throw std::runtime_error("partial service diagnostic observation failed");
                    context.synchronizeStream(primary);
                });
                size_t partial = 0u;
                for (const auto &cursor : observed)
                {
                    EXPECT_EQ(cursor.claimed, 0u) << "Graph returned with an unretired GPU claim";
                    partial += cursor.generation == generation && cursor.copied_bytes > 0u && cursor.copied_bytes < bytes;
                }
                if (replay == 0u) EXPECT_GT(partial, 0u) << "No actual partial-copy boundary was exercised";
                else for (size_t slot = 0u; slot < capacity; ++slot)
                {
                    EXPECT_EQ(receipts[slot].completed_generation, generation);
                    EXPECT_EQ(receipts[slot].completed_bytes, bytes);
                    EXPECT_EQ(receipts[slot].error, 0u);
                    EXPECT_GT(receipts[slot].device_active_nanoseconds, 0u);
                }
            }
            context.submitAndWait([&]
            {
                if (direction == MappedTransferDirection::HostToDevice)
                    for (size_t slot = 0u; slot < capacity; ++slot)
                        if (!backend->deviceToHostOnStream(mapped[slot]->mutableHostData(),
                                destinations[slot]->deviceData(), bytes + 16u, device.gpu_ordinal(), primary))
                            throw std::runtime_error("partial service final diagnostic copy failed");
                context.synchronizeStream(primary);
            });
            for (const auto &region : mapped)
            {
                const auto *actual = static_cast<const std::uint8_t *>(region->mutableHostData());
                EXPECT_EQ(std::memcmp(actual + offset, expected.data() + offset, bytes), 0)
                    << "Resumed copy lost or reused bytes";
                EXPECT_TRUE(std::all_of(actual, actual + offset,
                    [](std::uint8_t value) { return value == 0xccu; }));
                EXPECT_TRUE(std::all_of(actual + offset + bytes, actual + bytes + 16u,
                    [](std::uint8_t value) { return value == 0xccu; }))
                    << "Copy crossed its admitted destination extent";
            }
        }
        release_guard.reset();
        context.submitAndWait([&] { graph.reset(); context.destroyEvent(done); context.destroyEvent(fork); });
    }

    TEST(MappedTransferProgressEpochIntegration,
         CUDAMaintenanceProgressesAcrossFutureInferenceObservers)
    {
        for (const auto direction : {MappedTransferDirection::DeviceToHost,
                                     MappedTransferDirection::HostToDevice})
            proveMaintenanceProgressAcrossBlockedInference(
                DeviceId::cuda(0), direction, CPUWeightEdgeProof::None,
                TensorType::FP32, InferenceObservationTopology::FutureConsumers);
    }

    TEST(MappedTransferProgressEpochIntegration, CUDACPUWeightEdgesShareProgressSafePool)
    {
        proveCPUWeightEdgesShareProgressSafePool(DeviceId::cuda(0));
    }

    TEST(MappedTransferProgressEpochIntegration, CUDAProducerDependencyDoesNotBlockIndependentCommands)
    { proveExactProducerAdmission(DeviceId::cuda(0)); }

    TEST(MappedTransferProgressEpochIntegration, CUDABackgroundCopyBoundsAndTails)
    { proveBackgroundCopyRegions(DeviceId::cuda(0)); }

    TEST(MappedTransferProgressEpochIntegration,
         CUDAMaintenanceProgressesAcrossPeerHeldInference)
    {
        proveMaintenanceProgressAcrossBlockedInference(
            DeviceId::cuda(0), MappedTransferDirection::DeviceToHost);
        proveMaintenanceProgressAcrossBlockedInference(
            DeviceId::cuda(0), MappedTransferDirection::HostToDevice);
    }

    TEST(MappedTransferProgressEpochIntegration,
         CUDARepeatedGenerationsAreByteExact)
    {
        proveRepeatedGenerations(DeviceId::cuda(0));
    }

    TEST(MappedTransferProgressEpochIntegration,
         CUDAInferenceDoesNotJoinMaintenance)
    {
        proveInferenceDoesNotJoinMaintenance(DeviceId::cuda(0));
    }

    TEST(MappedTransferProgressEpochIntegration,
         CUDATopologySizedDirectoryHasBoundedExecutionLanePool)
    {
        proveTopologySizedDirectoryUsesBoundedExecutionLanes(
            DeviceId::cuda(0));
    }

    TEST(MappedTransferProgressEpochIntegration,
         CUDABoundedExecutionLanePoolQueuesWithoutLosingBytes)
    {
        proveBoundedExecutionLaneQueueIsByteExact(DeviceId::cuda(0));
    }
#endif

#if defined(HAVE_ROCM)
    /** HIP keeps native SDMA, but the same orchestration-thread capture contract applies. */
    TEST(MappedTransferProgressEpochIntegration, ROCmTopologyWorkerRecordsAndReplaysProgress)
    {
        for (const auto direction : {MappedTransferDirection::DeviceToHost,
                                     MappedTransferDirection::HostToDevice})
            proveMaintenanceProgressAcrossBlockedInference(DeviceId::rocm(0), direction,
                CPUWeightEdgeProof::None, TensorType::FP32,
                InferenceObservationTopology::FutureConsumers, CaptureRecordingOwner::TopologyWorker);
    }

    TEST(MappedTransferProgressEpochIntegration,
         ROCmMaintenanceProgressesAcrossFutureInferenceObservers)
    {
        for (const auto direction : {MappedTransferDirection::DeviceToHost,
                                     MappedTransferDirection::HostToDevice})
            proveMaintenanceProgressAcrossBlockedInference(
                DeviceId::rocm(0), direction, CPUWeightEdgeProof::None,
                TensorType::FP32, InferenceObservationTopology::FutureConsumers);
    }

    TEST(MappedTransferProgressEpochIntegration, ROCmCPUWeightEdgesShareProgressSafePool)
    {
        proveCPUWeightEdgesShareProgressSafePool(DeviceId::rocm(0));
    }

    TEST(MappedTransferProgressEpochIntegration, ROCmProducerDependencyDoesNotBlockIndependentCommands)
    { proveExactProducerAdmission(DeviceId::rocm(0)); }

    TEST(MappedTransferProgressEpochIntegration, ROCmBackgroundCopyBoundsAndTails)
    { proveBackgroundCopyRegions(DeviceId::rocm(0)); }

    TEST(MappedTransferProgressEpochIntegration,
         ROCmMaintenanceProgressesAcrossPeerHeldInference)
    {
        proveMaintenanceProgressAcrossBlockedInference(
            DeviceId::rocm(0), MappedTransferDirection::DeviceToHost);
        proveMaintenanceProgressAcrossBlockedInference(
            DeviceId::rocm(0), MappedTransferDirection::HostToDevice);
    }
    TEST(MappedTransferProgressEpochIntegration,
         ROCmRepeatedGenerationsAreByteExact)
    {
        proveRepeatedGenerations(DeviceId::rocm(0));
    }

    TEST(MappedTransferProgressEpochIntegration,
         ROCmInferenceDoesNotJoinMaintenance)
    {
        proveInferenceDoesNotJoinMaintenance(DeviceId::rocm(0));
    }

    TEST(MappedTransferProgressEpochIntegration,
         ROCmTopologySizedDirectoryHasBoundedExecutionLanePool)
    {
        proveTopologySizedDirectoryUsesBoundedExecutionLanes(
            DeviceId::rocm(0));
    }

    TEST(MappedTransferProgressEpochIntegration,
         ROCmBoundedExecutionLanePoolQueuesWithoutLosingBytes)
    {
        proveBoundedExecutionLaneQueueIsByteExact(DeviceId::rocm(0));
    }
#endif
} // namespace
} // namespace llaminar2
