/**
 * @file Test__GPURamPrefixArena.cpp
 * @brief Adversarial native transfer, eviction and archive reuse in a fixed RAM arena.
 *
 * An exact-stream timeline holds one producer behind an independent completed
 * producer. Admission must reclaim only the latter, without waiting, allocating
 * another pinned slab, or touching the held destination. The same backing also
 * survives metadata-only aliases, disk hydration and both owner retirement orders.
 * Harvest sheds its consumed lookup chain while the restore event continues
 * to protect every source range until the native DMA read has completed.
 */
#include <gtest/gtest.h>

#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IGPUGraphCapture.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "execution/prefix_cache/DiskPrefixStorageBackend.h"
#include "execution/prefix_cache/PrefixStateCache.h"
#include "execution/prefix_cache/RamPrefixStorageBackend.h"
#include "utils/PerfStatsCollector.h"
#include "../backends/MTPMainForwardReadRetirementProof.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <filesystem>
#include <future>
#include <limits>
#include <stdexcept>
#include <unistd.h>

using namespace llaminar2;

namespace
{
    /** @brief Fail a fixture operation without hiding a rejected native path. */
    void require(bool accepted, const char *boundary)
    {
        if (!accepted) throw std::runtime_error(boundary);
    }

    /** @brief Small aligned serialized payload, independent of tensor arithmetic. */
    PrefixPayloadLayout layoutFor(DeviceId device)
    {
        PrefixPayloadLayout layout;
        layout.device = device;
        layout.block_size = 64;
        layout.total_layers = layout.fa_layers = 1;
        layout.bytes_per_fa_layer_k = layout.bytes_per_fa_layer_v = 2048u;
        return layout;
    }

    /** @brief Distinct immutable identities prevent accidental same-key replacement. */
    PrefixCacheKey keyFor(int index)
    {
        return makePrefixCacheKey(0xa771, 0, index, index * 64, {index, index + 1});
    }

    /** @return Actual backend allocation calls, rather than cache-level intent. */
    double pinnedAllocations()
    {
        double count = 0.0;
        for (const auto &record : PerfStatsCollector::snapshot({"device_memory"}))
            if (record.name == "pinned_host_allocations") count += record.value;
        return count;
    }

    /** @brief Own all native infrastructure before adversarial retirement begins. */
    class Fixture final
    {
    public:
        static constexpr size_t kPayload = 4096u;
        static constexpr size_t kCapacity = 2u * kPayload;

        /** @brief Materialize one admitted tier and two retained producer graphs. */
        Fixture(DeviceId device, IWorkerGPUContext &gpu, size_t capacity = kCapacity) : device(device), backend(getBackendFor(device))
        {
            require(backend != nullptr, "prefix arena fixture has no backend");
            try
            {
                streams[0] = gpu.getOrCreateAuxiliaryStream("prefix_arena_held");
                streams[1] = gpu.getOrCreateAuxiliaryStream("prefix_arena_independent");
                control = gpu.getOrCreateAuxiliaryStream("prefix_arena_release");
                require(streams[0] && streams[1] && control, "prefix arena needs explicit streams");
                require(backend->supportsStreamTimelineSignal32(device.ordinal), "native timeline unavailable");
                signal = backend->allocateStreamTimelineSignal32(device.ordinal);
                terminal = backend->createEvent(device.ordinal);
                require(signal && terminal, "prefix arena fixture event/signal admission failed");
                require(backend->streamPublishTimelineSignal32(control, signal, 0u, device.ordinal),
                        "initialize native gate");
                require(backend->recordEvent(terminal, device.ordinal, control) &&
                        backend->waitForEvent(terminal, device.ordinal), "publish initial gate");
                for (size_t lane = 0; lane < 2u; ++lane)
                {
                    source[lane] = backend->allocate(kPayload, device.ordinal);
                    require(source[lane] != nullptr, "prefix arena fixture source allocation failed");
                    ready[lane] = std::shared_ptr<void>(backend->createEvent(device.ordinal),
                        [backend = backend, ordinal = device.ordinal](void *event) { backend->destroyEvent(event, ordinal); });
                    require(ready[lane] != nullptr, "prefix arena fixture readiness admission failed");
                    graphs[lane] = gpu.createGraphCapture(streams[lane]);
                    require(graphs[lane] != nullptr, "prefix arena fixture capture unavailable");
                    ScopedBackendGraphCapture capture(gpu, *graphs[lane], "prefix_arena_producer");
                    require(capture.begin(), "prefix arena producer capture failed");
                    require(backend->memset(source[lane], lane == 0 ? 0x5a : 0xb6,
                                           kPayload, device.ordinal, streams[lane]), "capture producer bytes");
                    capture.finish();
                    require(graphs[lane]->instantiate(), "instantiate retained producer");
                }
                PhysicalMemoryPlanBuilder plan;
                plan.add(PhysicalMemoryResource{.world_rank = 0, .device = DeviceId::cpu(),
                             .total_bytes = capacity, .admission_available_bytes = capacity},
                         PhysicalMemoryOwner::PrefixHostTier, capacity);
                authority = std::make_shared<PhysicalMemoryAuthority>(
                    std::make_shared<const PhysicalMemoryPlanAdmissionCertificate>(plan.build()), 0);
                std::string error;
                initial_allocations = pinnedAllocations();
                ram = RamPrefixStorageBackend::create(device, capacity, authority, &error);
                if (!ram) throw std::runtime_error(error);
            }
            catch (...)
            {
                retire();
                throw;
            }
        }

        /** @brief Release held work before any assertion-failure teardown can free bytes. */
        ~Fixture() { retire(); }

        /** @brief Enqueue the retained producer and its real archive DMA behind an optional gate. */
        void produce(PrefixBlockHandle &handle, size_t lane, uint32_t generation = 0u)
        {
            require(handle.payload_readiness->prepare(ready[lane], device, streams[lane]),
                    "prepare exact archive edge");
            if (generation)
                require(hold(lane, generation), "enqueue held producer frontier");
            require(graphs[lane]->launch(), "replay retained producer");
            require(backend->deviceToHostOnStream(handle.kv_payload, source[lane], kPayload,
                                                  device.ordinal, streams[lane]), "enqueue archive DMA");
            require(backend->recordEvent(ready[lane].get(), device.ordinal, streams[lane]) &&
                    handle.payload_readiness->publishRecorded(), "publish archive readiness");
        }

        /** @brief Hold an exact fixture stream and retain the generation needed for safe early-exit cleanup. */
        bool hold(size_t lane, uint32_t generation)
        {
            const bool submitted = backend->streamWaitTimelineSignal32(
                streams.at(lane), signal, generation, device.ordinal);
            if (submitted) highest_gate_generation = std::max(highest_gate_generation, generation);
            return submitted;
        }

        /** @brief Publish one monotonically increasing gate generation on its independent stream. */
        void release(uint32_t generation)
        {
            require(backend->streamPublishTimelineSignal32(control, signal, generation, device.ordinal),
                    "release held archive producer");
        }

        /** @return Actual live backing charged by the canonical authority. */
        size_t claimed() const
        {
            return authority->claimedBytes(DeviceId::cpu(), PhysicalMemoryOwner::PrefixHostTier,
                                           PhysicalMemoryMaterializationKind::NewAllocation);
        }

        /** @brief Prove exactly one native pinned allocation across every replay/retirement. */
        void expectPersistentBacking() const
        {
            EXPECT_TRUE(PerfStatsCollector::isDomainEnabled("device_memory"));
            EXPECT_EQ(pinnedAllocations(), initial_allocations + 1.0);
        }

        DeviceId device;
        IBackend *backend;
        std::array<void *, 2> streams{};
        void *control = nullptr;
        void *signal = nullptr;
        void *terminal = nullptr;
        std::array<void *, 2> source{};
        std::array<std::shared_ptr<void>, 2> ready;
        std::array<std::unique_ptr<IGPUGraphCapture>, 2> graphs;
        std::shared_ptr<PhysicalMemoryAuthority> authority;
        std::shared_ptr<RamPrefixStorageBackend> ram;
        double initial_allocations = 0.0;
        uint32_t highest_gate_generation = 0u;

    private:
        /** @brief Drain only fixture streams, then retire their bound native resources. */
        void retire() noexcept
        {
            if (signal && control &&
                !backend->streamPublishTimelineSignal32(control, signal,
                    highest_gate_generation, device.ordinal)) std::terminate();
            if (terminal)
                for (void *stream : {streams[0], streams[1], control})
                    if (stream && (!backend->recordEvent(terminal, device.ordinal, stream) ||
                                   !backend->waitForEvent(terminal, device.ordinal))) std::terminate();
            for (size_t lane = 0; lane < 2u; ++lane)
            {
                graphs[lane].reset();
                ready[lane].reset();
                if (source[lane]) backend->free(source[lane], device.ordinal);
                source[lane] = nullptr;
            }
            if (signal) backend->freeStreamTimelineSignal32(signal, device.ordinal);
            if (terminal) backend->destroyEvent(terminal, device.ordinal);
            signal = terminal = nullptr;
        }
    };

    /** @brief Assert a whole archived extent without a tensor-format-dependent oracle. */
    void expectBytes(const void *data, size_t bytes, uint8_t expected)
    {
        const auto *begin = static_cast<const uint8_t *>(data);
        EXPECT_TRUE(std::all_of(begin, begin + bytes, [&](uint8_t value) { return value == expected; }));
    }

    /** @brief Hold one DMA while reclaiming an independent completion, twenty times. */
    void verifyOutOfOrderRetirement(DeviceId device, IWorkerGPUContext &gpu)
    {
        Fixture fixture(device, gpu);
        for (int iteration = 0; iteration < 20; ++iteration)
        {
            SCOPED_TRACE(iteration);
            auto held = fixture.ram->allocate(keyFor(iteration * 3), layoutFor(device));
            auto completed = fixture.ram->allocate(keyFor(iteration * 3 + 1), layoutFor(device));
            ASSERT_TRUE(held.valid() && completed.valid());
            void *const held_address = held.kv_payload;
            void *const completed_address = completed.kv_payload;
            std::memset(held_address, 0x71, Fixture::kPayload);
            fixture.produce(held, 0u, iteration + 1u);
            fixture.produce(completed, 1u);
            auto held_readiness = held.payload_readiness;
            ASSERT_TRUE(fixture.ram->release(held));
            held = {};
            bool ready = true;
            ASSERT_TRUE(held_readiness->queryComplete(&ready));
            EXPECT_FALSE(ready);
            // The captured fast producer must progress while the other stream
            // is held. Only its event is joined at this test observation edge.
            ASSERT_TRUE(completed.waitForPayloadOnHost());
            expectBytes(completed_address, Fixture::kPayload, 0xb6);
            ASSERT_TRUE(fixture.ram->release(completed));
            completed = {};
            EXPECT_EQ(fixture.ram->availableAllocationBytes(), Fixture::kPayload);
            auto replacement = fixture.ram->allocate(keyFor(iteration * 3 + 2), layoutFor(device));
            ASSERT_TRUE(replacement.valid());
            EXPECT_EQ(replacement.kv_payload, completed_address);
            expectBytes(replacement.kv_payload, Fixture::kPayload, 0u);
            std::memset(replacement.kv_payload, 0x3c, Fixture::kPayload);
            EXPECT_FALSE(fixture.ram->canStore(Fixture::kPayload));
            expectBytes(held_address, Fixture::kPayload, 0x71);
            fixture.release(iteration + 1u);
            ASSERT_TRUE(held_readiness->waitOnHost());
            expectBytes(held_address, Fixture::kPayload, 0x5a);
            expectBytes(replacement.kv_payload, Fixture::kPayload, 0x3c);
            ASSERT_TRUE(fixture.ram->release(replacement));
            replacement = {};
            EXPECT_EQ(fixture.ram->availableAllocationBytes(), Fixture::kCapacity);
            EXPECT_EQ(fixture.claimed(), Fixture::kCapacity);
            fixture.expectPersistentBacking();
        }
    }

    /** @brief Last owner teardown joins its pending event after logical/backend retirement. */
    void verifyPendingTeardown(DeviceId device, IWorkerGPUContext &gpu)
    {
        Fixture fixture(device, gpu);
        auto held = fixture.ram->allocate(keyFor(81), layoutFor(device));
        ASSERT_TRUE(held.valid());
        fixture.produce(held, 0u, 1u);
        ASSERT_TRUE(fixture.ram->release(held));
        fixture.ram.reset();
        EXPECT_EQ(fixture.claimed(), Fixture::kCapacity);
        // The model-lifetime arena has one last request alias. Its teardown
        // may wait; ordinary eviction and admission above never may.
        auto teardown = std::async(std::launch::async, [handle = std::move(held)]() mutable { handle = {}; });
        EXPECT_EQ(fixture.claimed(), Fixture::kCapacity);
        fixture.release(1u);
        teardown.get();
        EXPECT_EQ(fixture.claimed(), 0u);
        fixture.expectPersistentBacking();
    }

    /** @brief Assertion-failure cleanup releases the real gate generation on both native backends. */
    void verifyHeldEarlyExitCleanup(DeviceId device, IWorkerGPUContext &gpu)
    {
        for (uint32_t generation = 1u; generation <= 20u; ++generation)
        {
            Fixture fixture(device, gpu);
            auto held = fixture.ram->allocate(keyFor(8200 + generation), layoutFor(device));
            ASSERT_TRUE(held.valid());
            fixture.produce(held, 0u, generation);
            bool ready = true;
            ASSERT_TRUE(fixture.backend->queryEvent(fixture.ready[0].get(), device.ordinal, &ready));
            EXPECT_FALSE(ready);
            ASSERT_TRUE(fixture.ram->release(held));
            // Intentionally leave the producer held. Fixture teardown must
            // publish this exact generation, not a wraparound sentinel that
            // CUDA's signed stream comparison can interpret as still pending.
        }
    }

    /** @brief A completed producer does not release a range borrowed by pending restore DMA. */
    void verifyRestoreAliases(DeviceId device, IWorkerGPUContext &gpu)
    {
        Fixture fixture(device, gpu);
        for (int iteration = 0; iteration < 20; ++iteration)
        {
            SCOPED_TRACE(iteration);
            auto archive = fixture.ram->allocate(keyFor(300 + iteration * 2), layoutFor(device));
            ASSERT_TRUE(archive.valid());
            fixture.produce(archive, 1u);
            ASSERT_TRUE(archive.waitForPayloadOnHost());
            auto request_alias = archive;
            auto restore_alias = archive;
            ASSERT_TRUE(fixture.hold(0u, iteration + 1u));
            ASSERT_TRUE(fixture.backend->hostToDeviceOnStream(fixture.source[0], archive.kv_payload,
                Fixture::kPayload, device.ordinal, fixture.streams[0]));
            ASSERT_TRUE(fixture.backend->recordEvent(fixture.ready[0].get(), device.ordinal, fixture.streams[0]));
            ASSERT_TRUE(fixture.ram->release(archive));
            archive = {};
            auto scratch = fixture.ram->allocate(keyFor(301 + iteration * 2), layoutFor(device));
            ASSERT_TRUE(scratch.valid());
            request_alias = {};
            EXPECT_FALSE(fixture.ram->canStore(Fixture::kPayload));
            fixture.release(iteration + 1u);
            ASSERT_TRUE(fixture.backend->waitForEvent(fixture.ready[0].get(), device.ordinal));
            ASSERT_TRUE(fixture.backend->deviceToHostOnStream(scratch.kv_payload, fixture.source[0],
                Fixture::kPayload, device.ordinal, fixture.streams[0]));
            ASSERT_TRUE(fixture.backend->recordEvent(fixture.ready[0].get(), device.ordinal, fixture.streams[0]));
            ASSERT_TRUE(fixture.backend->waitForEvent(fixture.ready[0].get(), device.ordinal));
            expectBytes(scratch.kv_payload, Fixture::kPayload, 0xb6);
            restore_alias = {};
            EXPECT_EQ(fixture.ram->availableAllocationBytes(), Fixture::kPayload);
            ASSERT_TRUE(fixture.ram->release(scratch));
            scratch = {};
            EXPECT_EQ(fixture.ram->availableAllocationBytes(), Fixture::kCapacity);
            fixture.expectPersistentBacking();
        }
    }

    /** @brief Round-trip tensor and odd-sized MoE metadata archives through the same GPU arena. */
    void verifyDiskHydration(DeviceId device, IWorkerGPUContext &gpu)
    {
        Fixture fixture(device, gpu);
        const auto path = std::filesystem::temp_directory_path() /
            ("llaminar-prefix-arena-" + std::to_string(getpid()) + "-" + device.toString());
        std::filesystem::create_directories(path);
        auto cleanup = std::shared_ptr<void>(reinterpret_cast<void *>(1),
            [path](void *) { std::filesystem::remove_all(path); });
        auto disk = std::make_shared<DiskPrefixStorageBackend>(
            path / (std::string(64u, '7') + ".kvcache"),
            4u * Fixture::kCapacity, std::string(64u, '7'));
        ASSERT_TRUE(disk->ready()) << disk->initializationError();
        for (int iteration = 0; iteration < 20; ++iteration)
        {
            SCOPED_TRACE(iteration);
            auto handle = fixture.ram->allocate(keyFor(iteration + 100), layoutFor(device));
            ASSERT_TRUE(handle.valid());
            auto metadata = std::make_shared<std::vector<uint8_t>>(13u + iteration, iteration + 3);
            ASSERT_TRUE(fixture.ram->attachModelRuntimeState(&handle, metadata));
            fixture.produce(handle, 0u, iteration + 1u);
            std::string error;
            const auto key = handle.key;
            const auto layout = handle.layout;
            auto publication = disk->scheduleWrite(handle);
            ASSERT_TRUE(publication.valid());
            auto runtime_alias = handle.model_runtime_state_storage;
            ASSERT_TRUE(fixture.ram->release(handle));
            handle = {};
            EXPECT_EQ(publication.publication(), nullptr);
            EXPECT_FALSE(fixture.ram->canStore(Fixture::kCapacity));
            fixture.release(iteration + 1u);
            auto receipt = publication.waitForPublication();
            ASSERT_NE(receipt, nullptr);
            ASSERT_TRUE(std::holds_alternative<PrefixArchiveWritePublication>(*receipt));
            auto verified = disk->beginHydration(key, layout, &error);
            ASSERT_TRUE(verified.has_value()) << error;
            PrefixBlockHandle hydrated;
            ASSERT_TRUE(disk->hydrateSelected(*verified, *fixture.ram, &hydrated, &error)) << error;
            expectBytes(hydrated.kv_payload, Fixture::kPayload, 0x5a);
            ASSERT_NE(hydrated.model_runtime_state_storage, nullptr);
            EXPECT_TRUE(std::ranges::equal(hydrated.model_runtime_state_storage->bytes(), *metadata));
            EXPECT_TRUE(std::ranges::equal(runtime_alias->bytes(), *metadata));
            ASSERT_TRUE(fixture.ram->release(hydrated));
            hydrated = {};
            runtime_alias.reset();
            EXPECT_EQ(fixture.ram->availableAllocationBytes(), Fixture::kCapacity);
            EXPECT_EQ(fixture.claimed(), Fixture::kCapacity);
            fixture.expectPersistentBacking();
        }
    }

    /** @brief Harvest ownership and native restore completion independently gate arena reuse. */
    void verifyHarvestAdmission(DeviceId device, IWorkerGPUContext &gpu)
    {
        Fixture fixture(device, gpu);
        PrefixStateCache cache(Fixture::kCapacity, fixture.ram);
        for (int iteration = 0; iteration < 20; ++iteration)
        {
            PrefixLookupResult admission;
            admission.supported = admission.cache_enabled = true;
            admission.block_size = 64;
            for (int block = 0; block < 2; ++block)
            {
                auto archive = fixture.ram->allocate(keyFor(1000 + iteration * 3 + block), layoutFor(device));
                ASSERT_TRUE(archive.valid());
                fixture.produce(archive, 1u);
                ASSERT_TRUE(archive.waitForPayloadOnHost());
                ASSERT_TRUE(cache.insert(archive));
                admission.blocks.push_back(std::move(archive));
            }
            admission.cached_tokens = admission.blocks.back().key.token_start + admission.blocks.back().key.token_count;
            const auto *terminal_address = admission.blocks.back().kv_payload;
            auto restore_sources = admission.blocks;
            ASSERT_TRUE(fixture.hold(0u, iteration + 1u));
            for (const auto &archive : restore_sources)
                ASSERT_TRUE(fixture.backend->hostToDeviceOnStream(fixture.source[0], archive.kv_payload,
                    Fixture::kPayload, device.ordinal, fixture.streams[0]));
            ASSERT_TRUE(fixture.backend->recordEvent(fixture.ready[0].get(), device.ordinal, fixture.streams[0]));
            admission = admission.forHarvest(admission.cached_tokens);
            ASSERT_EQ(admission.blocks.size(), 1u);
            EXPECT_EQ(cache.prepareCapacity(Fixture::kPayload), PrefixRamInsertPreparation::Busy);
            EXPECT_FALSE(fixture.ram->canStore(Fixture::kPayload));
            fixture.release(iteration + 1u);
            ASSERT_TRUE(fixture.backend->waitForEvent(fixture.ready[0].get(), device.ordinal));
            restore_sources.clear();
            EXPECT_EQ(fixture.ram->availableAllocationBytes(), Fixture::kPayload);
            EXPECT_NO_THROW(cache.completeInsertPreparation(keyFor(1002 + iteration * 3), Fixture::kPayload));
            auto next = fixture.ram->allocate(keyFor(1002 + iteration * 3), layoutFor(device));
            ASSERT_TRUE(next.valid());
            std::memset(next.kv_payload, 0x3c, Fixture::kPayload);
            expectBytes(terminal_address, Fixture::kPayload, 0xb6);
            ASSERT_TRUE(fixture.ram->release(next));
            next = {};
            admission = {};
            EXPECT_EQ(fixture.ram->availableAllocationBytes(), Fixture::kCapacity);
            fixture.expectPersistentBacking();
        }
    }

    /** @brief Completed restore reads release storage before the live-state handoff is consumed. */
    void verifyUnconsumedRestoreAdmission(DeviceId device, IWorkerGPUContext &gpu)
    {
        using Peer = DeviceGraphOrchestratorLiveStateTestAccess;
        // On assertion failure the fixture drains its exact streams before
        // the participant releases any still-retained archive read sources.
        std::unique_ptr<DeviceGraphOrchestrator> runner;
        Fixture fixture(device, gpu);
        runner = std::make_unique<DeviceGraphOrchestrator>(
            std::make_shared<QwenStandardGraph>(GraphConfig{}, nullptr), nullptr);
        PrefixStateCache cache(Fixture::kCapacity, fixture.ram);
        for (int iteration = 0; iteration < 20; ++iteration)
        {
            SCOPED_TRACE(iteration);
            PrefixLookupResult admission;
            admission.supported = admission.cache_enabled = true;
            admission.block_size = 64;
            for (int block = 0; block < 2; ++block)
            {
                auto archive = fixture.ram->allocate(keyFor(2000 + iteration * 3 + block), layoutFor(device));
                ASSERT_TRUE(archive.valid());
                fixture.produce(archive, 1u);
                ASSERT_TRUE(archive.waitForPayloadOnHost());
                ASSERT_TRUE(cache.insert(archive));
                admission.blocks.push_back(std::move(archive));
            }
            admission.cached_tokens = admission.blocks.back().key.token_start + admission.blocks.back().key.token_count;
            const auto *terminal_address = admission.blocks.back().kv_payload;
            ASSERT_TRUE(fixture.hold(0u, iteration + 1u));
            for (const auto &archive : admission.blocks)
                ASSERT_TRUE(fixture.backend->hostToDeviceOnStream(fixture.source[0], archive.kv_payload,
                    Fixture::kPayload, device.ordinal, fixture.streams[0]));
            ASSERT_TRUE(Peer::publishPrefixRestore(*runner, device, fixture.streams[0], admission.blocks));
            void *event = Peer::prefixRestoreEvent(*runner);
            ASSERT_NE(event, nullptr);
            admission = admission.forHarvest(admission.cached_tokens);
            Peer::retirePrefixRestoreSources(*runner);
            EXPECT_EQ(cache.prepareCapacity(Fixture::kPayload), PrefixRamInsertPreparation::Busy);
            EXPECT_FALSE(fixture.ram->canStore(Fixture::kPayload));
            EXPECT_TRUE(Peer::prefixRestoreHandoffRetained(*runner, event));
            fixture.release(iteration + 1u);
            ASSERT_TRUE(fixture.backend->waitForEvent(event, device.ordinal));
            Peer::retirePrefixRestoreSources(*runner);
            EXPECT_TRUE(Peer::prefixRestoreHandoffRetained(*runner, event));
            EXPECT_EQ(fixture.ram->availableAllocationBytes(), Fixture::kPayload);
            // This is the admission boundary that failed during OpenCode's
            // suffix prefill, before decode consumed the restore handoff.
            EXPECT_EQ(cache.prepareCapacity(Fixture::kPayload), PrefixRamInsertPreparation::Prepared);
            expectBytes(terminal_address, Fixture::kPayload, 0xb6);
            Peer::consumePrefixRestore(*runner);
            Peer::retirePrefixRestoreSources(*runner);
            admission = {};
            EXPECT_EQ(fixture.ram->availableAllocationBytes(), Fixture::kCapacity);
            fixture.expectPersistentBacking();
        }
        runner.reset();
    }

    /**
     * @brief Required RAM admission itself retires completed restore sources.
     *
     * A harvest can begin before restore completes and finish after several
     * archives or durable writes. Polling only at harvest entry leaves the
     * source chain charged at its later required publication. Exercise both
     * consumed and unconsumed ordering handoffs without a manual retirement
     * between native completion and the real cache admission boundary.
     */
    void verifyRestoreCapacityRetirement(DeviceId device, IWorkerGPUContext &gpu)
    {
        using Peer = DeviceGraphOrchestratorLiveStateTestAccess;
        std::unique_ptr<DeviceGraphOrchestrator> runner;
        Fixture fixture(device, gpu);
        runner = std::make_unique<DeviceGraphOrchestrator>(
            std::make_shared<QwenStandardGraph>(GraphConfig{}, nullptr), nullptr);
        PrefixStateCache cache(Fixture::kCapacity, fixture.ram, nullptr, nullptr, runner.get());
        for (const bool consumed : {false, true})
            for (int iteration = 0; iteration < 20; ++iteration)
            {
                SCOPED_TRACE(consumed);
                SCOPED_TRACE(iteration);
                const auto generation = static_cast<uint32_t>(iteration + (consumed ? 21 : 1));
                PrefixLookupResult admission;
                admission.supported = admission.cache_enabled = true;
                admission.block_size = 64;
                for (int block = 0; block < 2; ++block)
                {
                    auto archive = fixture.ram->allocate(keyFor(3000 + generation * 3 + block), layoutFor(device));
                    ASSERT_TRUE(archive.valid());
                    fixture.produce(archive, 1u);
                    ASSERT_TRUE(archive.waitForPayloadOnHost());
                    ASSERT_TRUE(cache.insert(archive));
                    admission.blocks.push_back(std::move(archive));
                }
                admission.cached_tokens = admission.blocks.back().key.token_start + admission.blocks.back().key.token_count;
                const auto *terminal_address = admission.blocks.back().kv_payload;
                ASSERT_TRUE(fixture.hold(0u, generation));
                for (const auto &archive : admission.blocks)
                    ASSERT_TRUE(fixture.backend->hostToDeviceOnStream(fixture.source[0], archive.kv_payload,
                        Fixture::kPayload, device.ordinal, fixture.streams[0]));
                ASSERT_TRUE(Peer::publishPrefixRestore(*runner, device, fixture.streams[0], admission.blocks));
                void *event = Peer::prefixRestoreEvent(*runner);
                ASSERT_NE(event, nullptr);
                admission = admission.forHarvest(admission.cached_tokens);
                if (consumed)
                    Peer::consumePrefixRestore(*runner);
                EXPECT_EQ(cache.prepareCapacity(Fixture::kPayload), PrefixRamInsertPreparation::Busy);
                EXPECT_FALSE(fixture.ram->canStore(Fixture::kPayload));
                fixture.release(generation);
                ASSERT_TRUE(fixture.backend->waitForEvent(event, device.ordinal));

                const auto next_key = keyFor(3000 + generation * 3 + 2);
                EXPECT_NO_THROW(cache.completeInsertPreparation(next_key, Fixture::kPayload));
                EXPECT_EQ(fixture.ram->availableAllocationBytes(), Fixture::kPayload);
                if (!consumed)
                    EXPECT_TRUE(Peer::prefixRestoreHandoffRetained(*runner, event));
                expectBytes(terminal_address, Fixture::kPayload, 0xb6);

                // Keep a failing implementation's teardown safe as well. This
                // explicit fixture cleanup occurs only after the assertions.
                Peer::retirePrefixRestoreSources(*runner);
                Peer::consumePrefixRestore(*runner);
                admission = {};
                EXPECT_EQ(fixture.ram->availableAllocationBytes(), Fixture::kCapacity);
                fixture.expectPersistentBacking();
            }
        runner.reset();
    }

    /**
     * @brief Rich nonterminal checkpoints release unused sections while real row reads stay pending.
     *
     * Three checkpoints fill the tier. Their historical recurrent images are
     * not restore consumers. Required publication must fit its independently
     * owned sections into the resulting holes without waiting for the held GPU
     * read, overwriting its KV/MTP sources, or materializing another pinned slab.
     */
    void verifyRichRestoreSections(DeviceId device, IWorkerGPUContext &gpu)
    {
        using Peer = DeviceGraphOrchestratorLiveStateTestAccess;
        for (const bool mtp : {false, true})
        {
            auto layout = layoutFor(device);
            layout.includes_hybrid_state = true;
            layout.hybrid_state_bytes = 2u * Fixture::kPayload;
            layout.includes_mtp_state = mtp;
            layout.mtp_kv_bytes = mtp ? Fixture::kPayload : 0u;
            layout.mtp_layers = mtp ? 1 : 0;
            layout.bytes_per_mtp_layer_k = layout.bytes_per_mtp_layer_v = Fixture::kPayload / 2u;
            const auto allocation = PrefixPayloadAllocationPlan::archive(layout);
            const size_t capacity = 3u * allocation.totalBytes() + Fixture::kPayload;
            std::unique_ptr<DeviceGraphOrchestrator> runner;
            Fixture fixture(device, gpu, capacity);
            runner = std::make_unique<DeviceGraphOrchestrator>(
                std::make_shared<QwenStandardGraph>(GraphConfig{}, nullptr), nullptr);
            PrefixStateCache cache(capacity, fixture.ram, nullptr, nullptr, runner.get());
            for (const bool consumed : {false, true})
                for (int iteration = 0; iteration < 20; ++iteration)
                {
                    SCOPED_TRACE(mtp);
                    SCOPED_TRACE(consumed);
                    SCOPED_TRACE(iteration);
                    const auto generation = static_cast<uint32_t>(1 + iteration + (consumed ? 20 : 0));
                    std::vector<PrefixBlockHandle> archives;
                    std::array<const void *, 3> kv_sources{};
                    for (size_t block = 0; block < 3u; ++block)
                    {
                        auto archive = fixture.ram->allocate(keyFor(6000 + generation * 4 + block), layout);
                        ASSERT_TRUE(archive.valid());
                        std::memset(archive.hybrid_payload, 0x8b, layout.hybrid_state_bytes);
                        if (mtp) std::memset(archive.mtp_payload, 0x9c, Fixture::kPayload);
                        fixture.produce(archive, 1u);
                        ASSERT_TRUE(archive.waitForPayloadOnHost());
                        ASSERT_TRUE(cache.insert(archive));
                        kv_sources[block] = archive.kv_payload;
                        archives.push_back(std::move(archive));
                    }
                    ASSERT_TRUE(fixture.hold(0u, generation));
                    std::vector<PrefixPayloadReadLease> reads;
                    for (size_t block = 0; block < archives.size(); ++block)
                    {
                        const auto &archive = archives[block];
                        ASSERT_TRUE(fixture.backend->hostToDeviceOnStream(fixture.source[0], archive.kv_payload,
                            Fixture::kPayload, device.ordinal, fixture.streams[0]));
                        if (mtp)
                            ASSERT_TRUE(fixture.backend->hostToDeviceOnStream(fixture.source[0], archive.mtp_payload,
                                Fixture::kPayload, device.ordinal, fixture.streams[0]));
                        const bool terminal = block + 1u == archives.size();
                        if (terminal)
                            for (size_t offset = 0; offset < layout.hybrid_state_bytes; offset += Fixture::kPayload)
                                ASSERT_TRUE(fixture.backend->hostToDeviceOnStream(fixture.source[0],
                                    static_cast<const uint8_t *>(archive.hybrid_payload) + offset,
                                    Fixture::kPayload, device.ordinal, fixture.streams[0]));
                        reads.push_back(terminal ? PrefixPayloadReadLease::wholeArchive(archive)
                                                 : PrefixPayloadReadLease::sequenceRows(archive));
                    }
                    ASSERT_TRUE(Peer::publishPrefixReadSources(*runner, device, fixture.streams[0], std::move(reads)));
                    void *const event = Peer::prefixRestoreEvent(*runner);
                    archives.clear();
                    if (consumed) Peer::consumePrefixRestore(*runner);
                    bool complete = true;
                    ASSERT_TRUE(fixture.backend->queryEvent(event, device.ordinal, &complete));
                    EXPECT_FALSE(complete);
                    const auto next_key = keyFor(6003 + generation * 4);
                    EXPECT_NO_THROW(cache.completeInsertPreparation(next_key, allocation));
                    ASSERT_TRUE(fixture.ram->canStore(allocation));
                    auto next = fixture.ram->allocate(next_key, layout);
                    ASSERT_TRUE(next.valid());
                    std::memset(next.hybrid_payload, 0x3c, layout.hybrid_state_bytes);
                    for (const auto *source : kv_sources) expectBytes(source, Fixture::kPayload, 0xb6);
                    ASSERT_TRUE(fixture.backend->queryEvent(event, device.ordinal, &complete));
                    EXPECT_FALSE(complete);
                    fixture.release(generation);
                    ASSERT_TRUE(fixture.backend->waitForEvent(event, device.ordinal));
                    ASSERT_TRUE(fixture.backend->deviceToHostOnStream(next.kv_payload, fixture.source[0],
                        Fixture::kPayload, device.ordinal, fixture.streams[0]));
                    ASSERT_TRUE(fixture.backend->recordEvent(fixture.terminal, device.ordinal, fixture.streams[0]));
                    ASSERT_TRUE(fixture.backend->waitForEvent(fixture.terminal, device.ordinal));
                    expectBytes(next.kv_payload, Fixture::kPayload, 0x8b);
                    expectBytes(next.hybrid_payload, layout.hybrid_state_bytes, 0x3c);
                    Peer::retirePrefixRestoreSources(*runner);
                    Peer::consumePrefixRestore(*runner);
                    cache.clear();
                    ASSERT_TRUE(fixture.ram->release(next));
                    next = {};
                    EXPECT_EQ(fixture.ram->availableAllocationBytes(), capacity);
                    EXPECT_EQ(fixture.claimed(), capacity);
                    fixture.expectPersistentBacking();
                }
            runner.reset();
        }
    }

    /** @brief Execute the identical adversarial native lifecycle on each backend. */
    void run(DeviceId device)
    {
        auto &gpu = GPUDeviceContextPool::instance().getContext(device);
        gpu.submitAndWait([&] {
            ASSERT_NO_FATAL_FAILURE(verifyOutOfOrderRetirement(device, gpu));
            ASSERT_NO_FATAL_FAILURE(verifyPendingTeardown(device, gpu));
            ASSERT_NO_FATAL_FAILURE(verifyHeldEarlyExitCleanup(device, gpu));
            ASSERT_NO_FATAL_FAILURE(verifyRestoreAliases(device, gpu));
            ASSERT_NO_FATAL_FAILURE(verifyDiskHydration(device, gpu));
        });
    }
} // namespace

#ifdef HAVE_CUDA
TEST(GPURamPrefixArena, CUDA) { run(DeviceId::cuda(0)); }
TEST(PrefixRestoredChainOwnership, CUDA)
{
    auto &gpu = GPUDeviceContextPool::instance().getContext(DeviceId::cuda(0));
    gpu.submitAndWait([&] { verifyHarvestAdmission(DeviceId::cuda(0), gpu); });
}
TEST(PrefixUnconsumedRestoreOwnership, CUDA)
{
    auto &gpu = GPUDeviceContextPool::instance().getContext(DeviceId::cuda(0));
    gpu.submitAndWait([&] { verifyUnconsumedRestoreAdmission(DeviceId::cuda(0), gpu); });
}
TEST(PrefixRichRestoreSectionOwnership, CUDA)
{
    auto &gpu = GPUDeviceContextPool::instance().getContext(DeviceId::cuda(0));
    gpu.submitAndWait([&] { verifyRichRestoreSections(DeviceId::cuda(0), gpu); });
}
TEST(PrefixRestoreCapacityRetirement, CUDA)
{
    auto &gpu = GPUDeviceContextPool::instance().getContext(DeviceId::cuda(0));
    gpu.submitAndWait([&] { verifyRestoreCapacityRetirement(DeviceId::cuda(0), gpu); });
}
#endif
#ifdef HAVE_ROCM
TEST(GPURamPrefixArena, ROCm) { run(DeviceId::rocm(0)); }
TEST(PrefixRestoredChainOwnership, ROCm)
{
    auto &gpu = GPUDeviceContextPool::instance().getContext(DeviceId::rocm(0));
    gpu.submitAndWait([&] { verifyHarvestAdmission(DeviceId::rocm(0), gpu); });
}
TEST(PrefixUnconsumedRestoreOwnership, ROCm)
{
    auto &gpu = GPUDeviceContextPool::instance().getContext(DeviceId::rocm(0));
    gpu.submitAndWait([&] { verifyUnconsumedRestoreAdmission(DeviceId::rocm(0), gpu); });
}
TEST(PrefixRichRestoreSectionOwnership, ROCm)
{
    auto &gpu = GPUDeviceContextPool::instance().getContext(DeviceId::rocm(0));
    gpu.submitAndWait([&] { verifyRichRestoreSections(DeviceId::rocm(0), gpu); });
}
TEST(PrefixRestoreCapacityRetirement, ROCm)
{
    auto &gpu = GPUDeviceContextPool::instance().getContext(DeviceId::rocm(0));
    gpu.submitAndWait([&] { verifyRestoreCapacityRetirement(DeviceId::rocm(0), gpu); });
}
#endif
