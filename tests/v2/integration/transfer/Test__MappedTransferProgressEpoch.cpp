/**
 * @file Test__MappedTransferProgressEpoch.cpp
 * @brief Real-device byte and lifecycle proofs for background expert DMA.
 *
 * Each backend runs simultaneous D2H and H2D commands on permanent typed
 * slots. The tests prove reusable generation identity, byte exactness, and the
 * central lifecycle invariant: a production-shaped inference graph can finish
 * while maintenance completion remains unpublished, because no inference
 * stream captures, joins, or waits for the transfer scheduler.
 */

#include <gtest/gtest.h>

#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IBackend.h"
#include "backends/IGPUGraphCapture.h"
#include "backends/IWorkerGPUContext.h"
#include "transfer/MappedTransferProgressEpoch.h"
#include "transfer/TransferEngine.h"

#include <algorithm>
#include <array>
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
        EXPECT_EQ(stats.dma_submissions, 2u * kGenerations);
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

        /* No maintenance callback has queried the DMA events since launch, so
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
        EXPECT_EQ(stats.dma_submissions, 2u);
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
            .maximum_bytes = 4096u,
            .name = "integration_topology_sized_directory_" +
                    device.toString(),
            .perf_device = device.toString(),
        });
        ASSERT_NE(epoch, nullptr);
        EXPECT_EQ(epoch->slotCapacity(), kTopologyCommandSlots);
        EXPECT_EQ(epoch->executionLaneCapacity(), kMigrationCycleSlots);
        EXPECT_NE(epoch->executionStream(), nullptr);
        EXPECT_GT(epoch->slotCapacity(), epoch->executionLaneCapacity());
        EXPECT_EQ(epoch->stats().slots_reserved, 0u);
    }

    /**
     * @brief Prove queued permanent commands drain through one execution lane.
     *
     * Four distinct mapped destinations deliberately exceed the one-lane
     * physical pool. The first progress pass may submit exactly one DMA; later
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
        EXPECT_EQ(epoch->stats().dma_submissions, 1u)
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
        EXPECT_EQ(stats.dma_submissions, kCommandCount);
        EXPECT_EQ(stats.command_failures, 0u);
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
