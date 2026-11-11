/**
 * @file Test__MappedActivationPacketCUDAAndROCm.cpp
 * @brief Original-owned canonical CPU ticket, sparse route and dense publication proofs.
 *
 * Each native body selects actual rank-local GPUs from the original MPI/NUMA
 * inventory and admits its complete serial geometry roster once. The numerical
 * oracles preserve exact sparse routing order, changing ownership, retained
 * capture reuse, dirty capacity tails, and independent background-copy progress
 * behind a held CPU ticket. True mixed node-local routes require both device
 * families to belong to the same original rank; cross-rank packet transport is
 * tested by the separate captured-transfer fixture. Device-free source-map
 * controls live in their original protocol Unit owner without native startup.
 */

#include <gtest/gtest.h>

#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IGPUGraphCapture.h"
#include "execution/moe/MoEOverlaySparseCollective.h"
#include "execution/moe/MoEOverlayNodeLocalRouteExchange.h"
#include "OriginalMappedComponentResources.h"
#include "kernels/cuda/moe/CUDAMoEKernel.h"
#include "kernels/rocm/moe/ROCmMoEKernel.h"
#include "transfer/TransferEngine.h"

#include <algorithm>
#include <bit>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <exception>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using namespace llaminar2;
using namespace llaminar2::test;

namespace
{
    /** @brief Poll one terminal event without synchronizing its complete device. */
    bool awaitEvent(
        IBackend *backend,
        void *event,
        std::chrono::steady_clock::duration timeout,
        int device_ordinal = 0)
    {
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        while (backend && event && std::chrono::steady_clock::now() < deadline)
        {
            bool ready = false;
            if (!backend->queryEvent(event, device_ordinal, &ready))
                return false;
            if (ready)
                return true;
            std::this_thread::yield();
        }
        return false;
    }

    /** @return Backend selected solely from one planner-provided device id. */
    IBackend *backendFor(DeviceId device)
    {
        if (device.is_cuda())
            return getCUDABackend();
        if (device.is_rocm())
            return getROCmBackend();
        return nullptr;
    }

    /** @return Production MoE kernel selected solely from endpoint device type. */
    std::unique_ptr<IMoEKernel> kernelFor(DeviceId device)
    {
        if (device.is_cuda())
            return std::make_unique<CUDAMoEKernel>(device.ordinal);
        if (device.is_rocm())
            return std::make_unique<ROCmMoEKernel>(device.ordinal);
        return nullptr;
    }

    /** @brief One raw allocation tracked with its exact backend owner. */
    struct DeviceAllocation
    {
        IBackend *backend = nullptr;
        void *address = nullptr;
        int ordinal = 0;
        std::optional<PhysicalMemoryAllocationLease> claim;
    };

    /**
     * @brief Own exact streams and buffers for one sparse-route exchange proof.
     *
     * The destructor first publishes the fabric's terminal sentinel and then
     * drains the two exact streams. This makes every assertion path safe even
     * when one endpoint kernel is resident in a device-side epoch wait.
     */
    class SparseRouteExchangeResources final
    {
    public:
        using Graph=RetainedParticipantGraphExecutableFixtureAccess;
        SparseRouteExchangeResources(OriginalMappedComponentSource& original,OriginalMappedComponentProgram program,
            DeviceId root,DeviceId producer)
            : source(original),program_(original.requireProgram(program)),root_device(root),producer_device(producer),
              root_backend(backendFor(root)),producer_backend(backendFor(producer)) {
            source.index(root);source.index(producer);
            const auto banks=program_.deviceBanks();
            const auto count=banks[0].size()+banks[1].size();
            allocations.reserve(count);transfer_buffers.reserve(count);transfer_claims_.reserve(count);
            mapped_claims_.reserve(2);mapped_regions_.reserve(2);arenas_.reserve(1);graphs_.reserve(2);
            root_stream=source.stream(root);
            producer_stream=source.stream(producer,producer==root);
            root_terminal=source.event(root,0);producer_terminal=source.event(producer,producer==root?1:0);
        }
        ~SparseRouteExchangeResources() {
            if(exchange)exchange->abortForShutdown();
            if(root_backend && root_stream && !root_backend->synchronizeStream(root_stream,root_device.ordinal))std::terminate();
            if(producer_backend && producer_stream && producer_stream!=root_stream &&
               !producer_backend->synchronizeStream(producer_stream,producer_device.ordinal))std::terminate();
            for(auto& graph:graphs_)if(graph.submissions && !Graph::observeTerminal(graph.owner))std::terminate();
            graphs_.clear();exchange.reset();
            for(auto iterator=allocations.rbegin();iterator!=allocations.rend();++iterator) {
                if(iterator->address)iterator->backend->free(iterator->address,iterator->ordinal);
                iterator->claim.reset();
            }
            transfer_buffers.clear();transfer_claims_.clear();
            arenas_.clear();mapped_regions_.clear();mapped_claims_.clear();
        }

        SparseRouteExchangeResources(
            const SparseRouteExchangeResources &) = delete;
        SparseRouteExchangeResources &operator=(
            const SparseRouteExchangeResources &) = delete;

        /** @brief Allocate one buffer on its exact endpoint and retain ownership. */
        void *allocate(IBackend *backend,DeviceId device,std::size_t bytes) {
            consumeBank(device,bytes);
            auto claim=source.memory()->claimNewAllocation(device,PhysicalMemoryOwner::ExecutionWorkspace,bytes);
            void* address=backend?backend->allocate(bytes,device.ordinal):nullptr;
            allocations.push_back({backend,address,device.ordinal,std::move(claim)});return address;
        }
        /** @brief Retain the exact original claim until TransferEngine storage is destroyed. */
        std::shared_ptr<DeviceTransferBuffer> allocateTransferBuffer(DeviceId device,std::size_t bytes) {
            consumeBank(device,bytes);
            auto claim=source.memory()->claimNewAllocation(device,PhysicalMemoryOwner::ExecutionWorkspace,bytes);
            auto buffer=TransferEngine::instance().allocateDeviceTransferBuffer(bytes,device);
            transfer_buffers.push_back(buffer);transfer_claims_.push_back(std::move(claim));return buffer;
        }
        /** @brief Original page-rounded mapping claim precedes the sole native allocator. */
        std::shared_ptr<MappedHostTransferRegion> mapped(std::size_t bytes,DeviceId device) {
            source.index(device);
            auto claim=source.memory()->claimNewAllocation(DeviceId::cpu(),PhysicalMemoryOwner::ActivationTransportStaging,
                TransferEngine::mappedHostRegionAllocationBytes(bytes));
            auto region=TransferEngine::instance().allocateMappedHostRegion(bytes,std::array{device});
            mapped_regions_.push_back(region);mapped_claims_.push_back(std::move(claim));return region;
        }
        /** @brief Keep every captured metadata backing alive until its original graph retires. */
        std::shared_ptr<MappedHostTransferArena> arena(std::span<const DeviceId> devices,
            std::span<const MappedHostTransferArena::AllocationRequest> requests) {
            auto arena=TransferEngine::instance().createMappedHostArena(devices,source.memory(),requests);
            arenas_.push_back(arena);return arena;
        }
        /** @brief Require exact declared native bank order before the first captured/native request. */
        void requireDeclarationConsumed()const {
            const auto expected=program_.deviceBanks();
            if(consumed_[0]!=expected[0].size() || consumed_[1]!=expected[1].size())
                throw std::logic_error("Original component did not materialize its exact declared banks");
        }
        /** @return Real admitted General definition created off-worker and retained through retirement. */
        std::shared_ptr<RetainedParticipantGraphExecutable> graph(DeviceId device,void* stream) {
            auto owner=Graph::original(source.permit(device),stream);graphs_.push_back({owner,0});return owner;
        }
        /** @brief Submit through the original owner; native identity advances once per accepted launch. */
        bool submit(const std::shared_ptr<RetainedParticipantGraphExecutable>& graph) {
            auto found=std::find_if(graphs_.begin(),graphs_.end(),[&](const auto& entry){return entry.owner==graph;});
            if(found==graphs_.end())throw std::logic_error("Component submitted a foreign native definition");
            return Graph::submit(graph,++found->submissions);
        }
        OriginalMappedComponentSource& source;
    private:
        /** @brief Immutable declared bank consumption; the PMA remains the only live byte ledger. */
        void consumeBank(DeviceId device,std::size_t bytes) {
            const auto role=device==root_device?0u:device==producer_device?1u:2u;
            const auto banks=program_.deviceBanks();
            if(role>1 || consumed_[role]>=banks[role].size() || banks[role][consumed_[role]]!=bytes)
                throw std::logic_error("Component allocation changed its original declared bank geometry");
            ++consumed_[role];
        }
        struct GraphOwner {std::shared_ptr<RetainedParticipantGraphExecutable> owner;std::uint64_t submissions;};
        OriginalMappedComponentProgram program_;
        std::array<std::size_t,2> consumed_{};
        std::vector<GraphOwner> graphs_;
        std::vector<PhysicalMemoryAllocationLease> transfer_claims_,mapped_claims_;
        std::vector<std::shared_ptr<MappedHostTransferRegion>> mapped_regions_;
        std::vector<std::shared_ptr<MappedHostTransferArena>> arenas_;
    public:

        DeviceId root_device;
        DeviceId producer_device;
        IBackend *root_backend = nullptr;
        IBackend *producer_backend = nullptr;
        void *root_stream = nullptr;
        void *producer_stream = nullptr;
        void *root_terminal = nullptr;
        void *producer_terminal = nullptr;
        std::unique_ptr<MoEOverlayNodeLocalRouteExchange> exchange;
        std::vector<std::shared_ptr<DeviceTransferBuffer>> transfer_buffers;
        std::vector<DeviceAllocation> allocations;
    };

    /**
     * @brief Own disjoint maintenance copies that must progress behind a ticket.
     *
     * Streams and kernel preparation belong to setup, before graph launch. Each
     * stream writes its own destination and publishes its own terminal event;
     * observing one stream cannot accidentally certify the other queues. The
     * caller must release the held inference graph before destroying this scope.
     */
    class CanonicalTicketMaintenanceProof final
    {
    public:
        /** @brief Prepare thirty-two independent exact-stream copy witnesses. */
        explicit CanonicalTicketMaintenanceProof(SparseRouteExchangeResources& resources,DeviceId device)
            :device_(device),backend_(backendFor(device)),
             destination_(resources.allocateTransferBuffer(device,kLanes*kBytes)),
             lanes_(TransferEngine::instance().allocatePersistentTransferExecutionLanes(kLanes,device,
                 "canonical_ticket_maintenance_progress",resources.source.permit(device))) {
            echo_=resources.mapped(kLanes*kBytes,device);
            for(std::size_t lane=0;lane<kLanes;++lane)events_[lane]=resources.source.event(device,2+lane);
            const auto& original=resources.source.permit(device);
            original.worker()->submitAndWait([&] {
                for(std::size_t lane=0;lane<kLanes;++lane) {
                    const auto name="persistent_transfer_execution:canonical_ticket_maintenance_progress:"+device.toString()+":lane:"+std::to_string(lane);
                    bool created=true;
                    if(original.worker()->getOrCreateAdmittedAuxiliaryStream(name,GPUAuxiliaryStreamSchedulingClass::BackgroundMaintenance,original,&created)!=lanes_[lane].stream() || created)
                        throw std::logic_error("Background lane did not retain its original admitted stream identity");
                    for(const auto changed:{GPUAuxiliaryStreamSchedulingClass::Normal,GPUAuxiliaryStreamSchedulingClass::LatencyCritical})
                        EXPECT_THROW(original.worker()->getOrCreateAdmittedAuxiliaryStream(name,changed,original),std::logic_error);
                }
            });
        }
        ~CanonicalTicketMaintenanceProof() {
            for(std::size_t lane=0;lane<kLanes;++lane) {
                if(!backend_->synchronizeStream(lanes_[lane].stream(),device_.ordinal)) {
                    std::fprintf(stderr,"Original canonical ticket maintenance retirement failed: device=%s lane=%zu\n",
                        device_.toString().c_str(),lane);
                    std::terminate();
                }
            }
        }

        CanonicalTicketMaintenanceProof(const CanonicalTicketMaintenanceProof &) = delete;
        CanonicalTicketMaintenanceProof &operator=(const CanonicalTicketMaintenanceProof &) = delete;

        /** @brief Queue one native copy and terminal event on every prepared lane. */
        void submit(const MappedHostTransferRegion &source)
        {
            ASSERT_NE(destination_, nullptr);
            for (std::size_t lane = 0; lane < kLanes; ++lane)
            {
                ASSERT_NE(events_[lane], nullptr);
                TransferEngine::instance().enqueueBackgroundMappedCopy(
                    lanes_[lane], MappedTransferDirection::HostToDevice,
                    destination_->mutableDeviceData(), kLanes * kBytes, lane * kBytes,
                    source, 0, kBytes);
                TransferEngine::instance().enqueueBackgroundMappedCopy(
                    lanes_[lane], MappedTransferDirection::DeviceToHost,
                    destination_->mutableDeviceData(), kLanes * kBytes, lane * kBytes,
                    *echo_, lane * kBytes, kBytes);
                ASSERT_TRUE(backend_->recordEvent(
                    events_[lane], device_.ordinal, lanes_[lane].stream()));
            }
        }

        /**
         * @brief Count exact native completions under one shared bounded deadline.
         * @return Number of completed lanes, never inferred from another event.
         */
        std::size_t awaitAll()
        {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
            std::array<bool, kLanes> ready{};
            std::size_t completed = 0;
            do
            {
                for (std::size_t lane = 0; lane < kLanes; ++lane)
                {
                    if (ready[lane])
                        continue;
                    bool native_ready = false;
                    if (!backend_->queryEvent(events_[lane], device_.ordinal, &native_ready))
                        return completed;
                    if (native_ready)
                    {
                        ready[lane] = true;
                        ++completed;
                    }
                }
                if (completed == kLanes)
                    return completed;
                std::this_thread::yield();
            } while (std::chrono::steady_clock::now() < deadline);
            return completed;
        }

        /** @brief Verify every destination after the consumer and copies retire. */
        void verify(const void *source, void *read_stream)
        {
            ASSERT_EQ(awaitAll(), kLanes);
            std::array<std::byte, kLanes * kBytes> actual{};
            ASSERT_TRUE(backend_->deviceToHost(
                actual.data(), destination_->deviceData(), actual.size(),
                device_.ordinal, read_stream));
            for (std::size_t lane = 0; lane < kLanes; ++lane)
            {
                EXPECT_EQ(std::memcmp(actual.data() + lane * kBytes, source, kBytes), 0)
                    << "maintenance lane=" << lane;
                EXPECT_EQ(std::memcmp(echo_->mutableHostData(lane * kBytes), source, kBytes), 0)
                    << "maintenance return lane=" << lane;
            }
        }

        static constexpr std::size_t kLanes = 32;
        static constexpr std::size_t kBytes = 4096;

    private:
        DeviceId device_;
        IBackend *backend_;
        std::shared_ptr<DeviceTransferBuffer> destination_;
        std::shared_ptr<MappedHostTransferRegion> echo_;
        std::vector<PersistentTransferExecutionLane> lanes_;
        std::array<void *, kLanes> events_{};
    };

    /** Live extents varied independently of the captured route-bank capacity. */
    enum class TicketRoutePattern { Full, Varying };

    /**
     * @brief Reuse one CPU-published canonical ticket for twenty GPU replays.
     *
     * Each replay publishes a different route permutation and payload. The CPU
     * must be unable to arm the next replay while the current publication is
     * live; after the exact-stream materialization and acknowledgement event,
     * the same storage must become reusable without a stream/device sync. This
     * proves both mapped bytes and the producer/consumer sequence protocol on
     * the selected continuation backend.
     * @param continuation_device Native captured consumer backend and ordinal.
     * @param route_capacity Original route slots, covering decode and prefill.
     * @param d_model Hidden width, including partial-warp tails.
     * @param pattern Alternate empty, sparse and full publications
     *        without rebinding the captured capacity or clearing old device rows.
     */
    void runCanonicalRouteTicketReuseProof(
        OriginalMappedComponentSource& original,DeviceId continuation_device,
        std::size_t route_capacity,
        std::int32_t d_model = 3072,
        TicketRoutePattern pattern = TicketRoutePattern::Full)
    {
        SCOPED_TRACE(continuation_device.toString());
        SCOPED_TRACE("capacity=" + std::to_string(route_capacity) + " width=" + std::to_string(d_model));
        /* Exercise both decode and prefill-sized grids at the real hidden
         * width. A publication-first eager launch cannot prove that a retained
         * waiting consumer leaves resources available to maintenance. */
        constexpr std::uint64_t workspace_generation = 17u;
        constexpr std::uint64_t first_residency_epoch = 101u;
        constexpr std::size_t replay_count = 20u;
        const std::size_t element_count =
            route_capacity * static_cast<std::size_t>(d_model);
        const std::size_t payload_bytes = element_count * sizeof(float);

        SparseRouteExchangeResources resources(original,
            {OriginalMappedComponentProgram::Kind::Ticket,route_capacity,std::size_t(d_model),20,true},continuation_device,continuation_device);
        ASSERT_NE(resources.root_backend, nullptr);
        ASSERT_GT(
            resources.root_backend->deviceCount(),
            continuation_device.ordinal);
        ASSERT_NE(resources.root_stream, nullptr);
        ASSERT_NE(resources.root_terminal, nullptr);
        ASSERT_NE(resources.producer_stream, nullptr);
        ASSERT_NE(resources.producer_terminal, nullptr);
        constexpr std::size_t progress_bytes = 4096u;
        void *const progress_device = resources.allocate(
            resources.root_backend, continuation_device, progress_bytes);
        ASSERT_NE(progress_device, nullptr);

        void *const output_device = resources.allocate(
            resources.root_backend, continuation_device, payload_bytes);
        ASSERT_NE(output_device, nullptr);
        ASSERT_TRUE(resources.root_backend->memset(
            output_device,
            0,
            payload_bytes,
            continuation_device.ordinal,
            resources.root_stream));

        const std::array<DeviceId, 1> endpoints{continuation_device};
        auto metadata_arena =
            resources.arena(endpoints,
                std::array{MappedHostTransferArena::AllocationRequest{
                    MoEOverlayCanonicalRouteReturnTicketStorage::metadataLayout(route_capacity).bytes,64}});
        auto contribution_region =
            resources.mapped(payload_bytes,continuation_device);
        ASSERT_NE(contribution_region, nullptr);
        ASSERT_TRUE(contribution_region->isBound());
        CanonicalTicketMaintenanceProof maintenance(resources,continuation_device);

        MoEOverlayCanonicalRouteReturnTicketStorage storage;
        storage.bindFixedCapacity(
            /*layer_idx=*/5,
            route_capacity,
            d_model,
            continuation_device,
            workspace_generation,
            contribution_region,
            metadata_arena);
        ASSERT_TRUE(storage.hasValidBoundIdentity());
        metadata_arena->requireDeclarationConsumed();resources.requireDeclarationConsumed();

        std::unique_ptr<IMoEKernel> kernel;
        if (continuation_device.is_cuda())
        {
            kernel = std::make_unique<CUDAMoEKernel>(
                continuation_device.ordinal);
        }
        else if (continuation_device.is_rocm())
        {
            kernel = std::make_unique<ROCmMoEKernel>(
                continuation_device.ordinal);
        }
        ASSERT_NE(kernel, nullptr);

        MoEOverlayCanonicalRouteTicketConsumeLaunch ticket{
            .control = storage.controlDeviceAlias(),
            .original_route_slots =
                storage.originalRouteSlotsDeviceAlias(),
            .compact_route_slots =
                storage.compactRouteSlotsDeviceAlias(),
            .compact_preweighted_contributions_fp32 =
                storage.contributionRowsDeviceAlias(),
            .canonical_route_contributions_fp32 =
                static_cast<float *>(output_device),
            .route_capacity = route_capacity,
            .d_model = d_model,
        };
        ASSERT_TRUE(ticket.valid());

        {
            auto abandoned = storage.arm(first_residency_epoch - 1u);
            ASSERT_TRUE(abandoned);
        }
        EXPECT_FALSE(storage.payloadReady());
        EXPECT_TRUE(storage.arm(first_residency_epoch - 1u))
            << "dropping an unpublished lease must restore quiescent ownership";

        /*
         * Warm the consumer and verify its abort acknowledgement before graph
         * capture. Aborts release the sequence without manufacturing route
         * rows; subsequent retained success replays reuse the same storage.
         */
        ASSERT_TRUE(storage.publishAbort());
        EXPECT_FALSE(storage.payloadReady());
        ASSERT_TRUE(kernel->consumeMoEOverlayCanonicalRouteTicket(
            MoEKernelLaunchContext{.stream = resources.root_stream},
            ticket));
        ASSERT_TRUE(resources.root_backend->recordEvent(
            resources.root_terminal,
            continuation_device.ordinal,
            resources.root_stream));
        ASSERT_TRUE(awaitEvent(
            resources.root_backend,
            resources.root_terminal,
            std::chrono::seconds(5),
            continuation_device.ordinal));
        EXPECT_FALSE(storage.payloadReady());

        auto &worker = GPUDeviceContextPool::instance().getContext(
            continuation_device);
        auto consumer_owner=resources.graph(continuation_device,resources.root_stream);
        auto* consumer=&RetainedParticipantGraphExecutableFixtureAccess::capture(consumer_owner);
        worker.submitAndWait([&] {
            ScopedBackendGraphCapture recording(worker,*consumer,"original canonical ticket consumer");
            ASSERT_TRUE(recording.begin());
            ASSERT_TRUE(kernel->consumeMoEOverlayCanonicalRouteTicket(
                MoEKernelLaunchContext{.stream = resources.root_stream}, ticket));
            recording.finish();
            ASSERT_TRUE(consumer->prepareRuntimeContextStorage(original.memory()));
            ASSERT_TRUE(consumer->instantiate());
        });
        ASSERT_TRUE(consumer && consumer->hasExecutable());
        bool consumer_pending = false;
        auto drain_consumer = [&](void *) {
            if (!consumer_pending)
                return;
            // An assertion must not strand a retained wait or release mapped
            // payload storage before the accepted graph has stopped reading.
            (void)storage.publishAbort();
            if(!resources.root_backend->synchronizeStream(resources.root_stream,continuation_device.ordinal)) {
                std::fputs("Original canonical ticket consumer retirement failed before backing release\n",stderr);
                std::terminate();
            }
        };
        std::unique_ptr<void, decltype(drain_consumer)> consumer_guard(
            &storage, drain_consumer);
        std::vector<float> actual(element_count, 0.0f);
        std::vector<float> expected(element_count, 0.0f);
        for (std::size_t replay = 0u; replay < replay_count; ++replay)
        {
            const std::uint64_t residency_epoch =
                first_residency_epoch + replay;
            auto publication = storage.arm(residency_epoch);
            ASSERT_TRUE(publication)
                << "replay=" << replay;
            EXPECT_FALSE(storage.arm(residency_epoch + 1u))
                << "a producer cannot arm one ticket twice";

            const std::array live_extents{
                std::size_t{0}, std::size_t{1}, std::min(std::size_t{3}, route_capacity),
                route_capacity, route_capacity / 2u, route_capacity - 1u};
            const std::size_t live_routes = pattern == TicketRoutePattern::Varying
                ? live_extents[replay % live_extents.size()] : route_capacity;
            // Unpublished slots must retain their prior bytes. Keeping the old
            // bank catches capacity-sized reads/writes on empty or sparse replay.
            auto *const original_slots = storage.originalRouteSlotsHost();
            auto *const compact_slots = storage.compactRouteSlotsHost();
            float *const contribution_rows = storage.contributionRowsHost();
            ASSERT_NE(original_slots, nullptr);
            ASSERT_NE(compact_slots, nullptr);
            ASSERT_NE(contribution_rows, nullptr);
            for (std::size_t entry = 0u; entry < route_capacity; ++entry)
            {
                const std::size_t original_slot =
                    (entry + replay) % route_capacity;
                const std::size_t compact_slot =
                    (entry + 2u * replay + 1u) % route_capacity;
                original_slots[entry] =
                    static_cast<std::int32_t>(original_slot);
                compact_slots[entry] =
                    static_cast<std::int32_t>(compact_slot);
                for (std::size_t column = 0u;
                     column < static_cast<std::size_t>(d_model);
                     ++column)
                {
                    // Copy semantics must preserve arbitrary bits, including
                    // signed zero, subnormals and NaN payloads; no FP fold is
                    // permitted in this materialization kernel. Every replay
                    // changes even unpublished source rows to expose overreads.
                    const auto bits = static_cast<std::uint32_t>(
                        0x9e3779b9u * (replay + 1u) + 0x85ebca6bu * entry + column);
                    const float value = std::bit_cast<float>(bits);
                    contribution_rows[
                        compact_slot * static_cast<std::size_t>(d_model) +
                        column] = value;
                    if (entry < live_routes)
                        expected[
                            original_slot * static_cast<std::size_t>(d_model) +
                            column] = value;
                }
            }

            ASSERT_TRUE(resources.submit(consumer_owner));
            consumer_pending = true;
            ASSERT_TRUE(resources.root_backend->recordEvent(
                resources.root_terminal,
                continuation_device.ordinal,
                resources.root_stream));
            // The CPU has deliberately not released its ticket yet. A
            // separate exact stream must still complete unrelated transfer
            // work; observing that event cannot satisfy the waiting ticket.
            EXPECT_TRUE(resources.root_backend->hostToDeviceOnStream(
                progress_device, contribution_rows, progress_bytes,
                continuation_device.ordinal, resources.producer_stream));
            EXPECT_TRUE(resources.root_backend->recordEvent(
                resources.producer_terminal, continuation_device.ordinal,
                resources.producer_stream));
            EXPECT_TRUE(awaitEvent(
                resources.root_backend, resources.producer_terminal,
                std::chrono::seconds(1), continuation_device.ordinal))
                << "Pending captured CPU ticket blocked independent DMA";
            maintenance.submit(*contribution_region);
            // CUDA maintenance uses bounded SM copies, unlike the DMA witness
            // above. Both must progress while the real captured consumer waits.
            const std::size_t maintenance_ready = maintenance.awaitAll();
            bool consumer_ready = true;
            EXPECT_TRUE(resources.root_backend->queryEvent(
                resources.root_terminal, continuation_device.ordinal,
                &consumer_ready));
            EXPECT_FALSE(consumer_ready)
                << "Consumer completed without its CPU publication";
            EXPECT_FALSE(storage.arm(residency_epoch + 1u))
                << "unpublished payload is still owned by its CPU producer";

            ASSERT_TRUE(publication.publish(live_routes))
                << "replay=" << replay;
            // A live captured consumer may acknowledge before publish returns.
            // The durable success receipt, not transient readiness, is proof.
            ASSERT_TRUE(storage.publicationSucceededFor(residency_epoch));
            EXPECT_FALSE(publication.publish(live_routes))
                << "publication requires one fresh arm transition";

            ASSERT_TRUE(awaitEvent(
                resources.root_backend,
                resources.root_terminal,
                std::chrono::seconds(5),
                continuation_device.ordinal));
            ASSERT_TRUE(RetainedParticipantGraphExecutableFixtureAccess::observeTerminal(consumer_owner));
            consumer_pending = false;
            maintenance.verify(contribution_rows, resources.producer_stream);
            ASSERT_EQ(maintenance_ready, CanonicalTicketMaintenanceProof::kLanes)
                << "Pending captured CPU ticket blocked native background copies";
            EXPECT_FALSE(storage.payloadReady())
                << "GPU did not acknowledge replay=" << replay;
            EXPECT_TRUE(storage.publicationSucceededFor(residency_epoch))
                << "the CPU protocol terminal must accept an exact publication "
                   "even when the concurrent GPU parent acknowledged it first; "
                   "replay="
                << replay;
            EXPECT_FALSE(
                storage.publicationSucceededFor(residency_epoch + 1u));

            ASSERT_TRUE(resources.root_backend->deviceToHost(
                actual.data(),
                output_device,
                payload_bytes,
                continuation_device.ordinal,
                resources.root_stream));
            EXPECT_EQ(std::memcmp(actual.data(), expected.data(), payload_bytes), 0)
                << "Captured route permutation changed payload bytes; replay=" << replay;
        }
    }

    /**
     * @brief Exercise five device-owned epochs, including placement invariance.
     *
     * The final two epochs retain identical per-route bytes while changing the
     * owning participant. Their cancellation-heavy values distinguish the
     * canonical original-slot fold from the obsolete participant-subtotal
     * arithmetic that made Dynamic placement numerically observable.
     *
     * @param root_device Continuation-root GPU that folds canonical slots.
     * @param producer_device Non-root GPU that owns a sparse route subset.
     * @param rows Live physical-row count for the retained route envelope.
     * @param d_model Hidden width of every canonical route contribution.
     * @param epochs Number of monotonic publications to retain and fold.
     * @param replay_through_retained_graphs Capture both endpoint transactions.
     * @param sparse_owner_stride Zero selects the dense ownership pattern;
     *        otherwise select one peer slot per stride and start with an empty
     *        peer epoch to exercise blocks which must do no payload work.
     */
    void runSparseCanonicalRouteExchange(
        OriginalMappedComponentSource& original,DeviceId root_device,
        DeviceId producer_device,
        std::uint32_t rows = 64u,
        std::uint32_t d_model = 512u,
        std::uint32_t epochs = 5u,
        bool replay_through_retained_graphs = false,
        std::uint32_t sparse_owner_stride = 0u)
    {
        constexpr std::uint32_t top_k = 8u;
        constexpr std::int32_t root_participant = 11;
        constexpr std::int32_t producer_participant = 37;
        const std::uint32_t route_slots = rows * top_k;

        ASSERT_GT(rows, 0u);
        ASSERT_GT(d_model, 0u);
        ASSERT_GE(epochs, 2u);
        const std::uint32_t invariant_epoch_begin = epochs - 2u;

        SparseRouteExchangeResources resources(original,
            {OriginalMappedComponentProgram::Kind::Sparse,rows,d_model,epochs,replay_through_retained_graphs},root_device,producer_device);
        ASSERT_NE(resources.root_backend, nullptr);
        ASSERT_NE(resources.producer_backend, nullptr);
        ASSERT_GT(resources.root_backend->deviceCount(), root_device.ordinal);
        ASSERT_GT(
            resources.producer_backend->deviceCount(),
            producer_device.ordinal);

        ASSERT_NE(resources.root_stream, nullptr);
        ASSERT_NE(resources.producer_stream, nullptr);
        ASSERT_NE(resources.root_terminal, nullptr);
        ASSERT_NE(resources.producer_terminal, nullptr);

        resources.exchange =
            std::make_unique<MoEOverlayNodeLocalRouteExchange>(
                MoEOverlayNodeLocalRouteExchange::Config{
                    .devices = {root_device, producer_device},
                    .root_device = root_device,
                    .identity = "original_component_routes",
                },MoEOverlayNodeLocalRouteExchange::declareStorage(
                    {{root_device,producer_device},root_device,"original_component_routes"},rows,top_k,d_model),original.memory(),original.inventory());
        resources.exchange->materialize(
            {
                {.participant_id = root_participant, .device = root_device},
                {.participant_id = producer_participant,
                 .device = producer_device},
            },
            root_participant,
            rows,
            top_k,
            d_model);

        const std::size_t route_elements =
            static_cast<std::size_t>(route_slots) * d_model;
        std::vector<float> expected(
            static_cast<std::size_t>(rows) * d_model, 0.0f);
        /*
         * Change both the final assignment ledger and both candidate payloads
         * every epoch. This models a promoted expert becoming executable on a
         * new participant and proves that transport follows the live route
         * publication rather than a setup-time owner map. Distinct bytes on
         * both endpoints also expose a stale ledger or premature lane reuse.
         */
        std::vector<std::vector<std::int32_t>> route_participant_epochs(
            epochs);
        std::vector<std::vector<float>> root_route_epochs(epochs);
        std::vector<std::vector<float>> producer_route_epochs(epochs);
        std::vector<std::vector<float>> expected_epochs(epochs);
        for (std::uint32_t epoch = 0u; epoch < epochs; ++epoch)
        {
            route_participant_epochs[epoch].resize(route_slots);
            root_route_epochs[epoch].resize(route_elements);
            producer_route_epochs[epoch].resize(route_elements);
            expected_epochs[epoch].assign(expected.size(), 0.0f);
            for (std::uint32_t slot = 0u; slot < route_slots; ++slot)
            {
                const bool producer_owns = sparse_owner_stride == 0u
                    ? ((slot + epoch) % 3u) != 0u
                    : epoch != 0u && ((slot + epoch) % sparse_owner_stride) == 0u;
                route_participant_epochs[epoch][slot] =
                    producer_owns
                        ? producer_participant
                        : root_participant;
                for (std::uint32_t column = 0u; column < d_model; ++column)
                {
                    const std::size_t route_index =
                        static_cast<std::size_t>(slot) * d_model + column;
                    const std::uint32_t route = slot % top_k;
                    const float base =
                        epoch < invariant_epoch_begin
                            ? static_cast<float>((slot % 13u) + 1u) +
                                  static_cast<float>(column % 8u) * 0.0625f +
                                  static_cast<float>(epoch) * 0.5f
                            : route == 0u
                                  ? 1.0e20f
                                  : route == 2u ? -1.0e20f : 1.0f;
                    root_route_epochs[epoch][route_index] = base;
                    producer_route_epochs[epoch][route_index] =
                        epoch < invariant_epoch_begin
                            ? base + 32.0f
                            : base;
                    expected_epochs[epoch][
                        static_cast<std::size_t>(slot / top_k) * d_model +
                        column] += producer_owns
                                       ? producer_route_epochs[epoch][route_index]
                                       : root_route_epochs[epoch][route_index];
                }
            }
        }
        ASSERT_EQ(
            std::memcmp(
                expected_epochs[invariant_epoch_begin].data(),
                expected_epochs[invariant_epoch_begin + 1u].data(),
                expected_epochs[invariant_epoch_begin].size() *
                    sizeof(float)),
            0)
            << "placement-invariance oracle changed canonical route order";
        for (std::uint32_t epoch = invariant_epoch_begin;
             epoch < epochs;
             ++epoch)
        {
            bool exposes_participant_grouping = false;
            for (std::uint32_t row = 0u;
                 row < rows && !exposes_participant_grouping;
                 ++row)
            {
                for (std::uint32_t column = 0u;
                     column < d_model;
                     ++column)
                {
                    float root_subtotal = 0.0f;
                    float producer_subtotal = 0.0f;
                    for (std::uint32_t route = 0u;
                         route < top_k;
                         ++route)
                    {
                        const std::uint32_t slot = row * top_k + route;
                        const float value = root_route_epochs[epoch][
                            static_cast<std::size_t>(slot) * d_model +
                            column];
                        if (route_participant_epochs[epoch][slot] ==
                            root_participant)
                        {
                            root_subtotal += value;
                        }
                        else
                        {
                            producer_subtotal += value;
                        }
                    }
                    const float obsolete_grouped =
                        root_subtotal + producer_subtotal;
                    const float canonical = expected_epochs[epoch][
                        static_cast<std::size_t>(row) * d_model + column];
                    exposes_participant_grouping =
                        obsolete_grouped != canonical;
                    if (exposes_participant_grouping)
                        break;
                }
            }
            ASSERT_TRUE(exposes_participant_grouping)
                << "adversarial epoch no longer distinguishes participant grouping from canonical route order";
        }

        const auto &route_participants = route_participant_epochs.front();
        const auto &root_routes = root_route_epochs.front();
        const auto &producer_routes = producer_route_epochs.front();

        const std::size_t route_bytes = root_routes.size() * sizeof(float);
        const std::size_t participant_bytes =
            route_participants.size() * sizeof(std::int32_t);
        const std::size_t output_bytes = expected.size() * sizeof(float);
        void *const root_routes_device = resources.allocate(
            resources.root_backend, root_device, route_bytes);
        void *const producer_routes_device = resources.allocate(
            resources.producer_backend, producer_device, route_bytes);
        void *const root_participants_device = resources.allocate(
            resources.root_backend, root_device, participant_bytes);
        void *const producer_participants_device = resources.allocate(
            resources.producer_backend, producer_device, participant_bytes);
        void *const output_device = resources.allocate(
            resources.root_backend, root_device, output_bytes);
        void *const output_epochs_device = resources.allocate(
            resources.root_backend,
            root_device,
            output_bytes * epochs);
        void *const validation_device = resources.allocate(
            resources.root_backend, root_device, sizeof(std::int32_t));
        const auto root_peer_bindings =
            resources.exchange->rootPeerBindings(root_device);
        ASSERT_EQ(root_peer_bindings.size(), 1u);
        void *const peer_bindings_device = resources.allocate(
            resources.root_backend,
            root_device,
            root_peer_bindings.size() *
                sizeof(MoENodeLocalRoutePeerDeviceBinding));
        resources.requireDeclarationConsumed();
        ASSERT_TRUE(std::all_of(
            resources.allocations.begin(), resources.allocations.end(),
            [](const DeviceAllocation &allocation)
            {
                return allocation.address != nullptr;
            }));

        ASSERT_TRUE(resources.root_backend->hostToDevice(
            root_routes_device,
            root_routes.data(),
            route_bytes,
            root_device.ordinal,
            resources.root_stream));
        ASSERT_TRUE(resources.producer_backend->hostToDevice(
            producer_routes_device,
            producer_routes.data(),
            route_bytes,
            producer_device.ordinal,
            resources.producer_stream));
        ASSERT_TRUE(resources.root_backend->hostToDevice(
            root_participants_device,
            route_participants.data(),
            participant_bytes,
            root_device.ordinal,
            resources.root_stream));
        ASSERT_TRUE(resources.producer_backend->hostToDevice(
            producer_participants_device,
            route_participants.data(),
            participant_bytes,
            producer_device.ordinal,
            resources.producer_stream));
        ASSERT_TRUE(resources.root_backend->hostToDevice(
            peer_bindings_device,
            root_peer_bindings.data(),
            root_peer_bindings.size() *
                sizeof(MoENodeLocalRoutePeerDeviceBinding),
            root_device.ordinal,
            resources.root_stream));

        auto root_kernel = kernelFor(root_device);
        auto producer_kernel = kernelFor(producer_device);
        ASSERT_NE(root_kernel, nullptr);
        ASSERT_NE(producer_kernel, nullptr);
        const MoEKernelLaunchContext root_launch{
            .stream = resources.root_stream,
        };
        const MoEKernelLaunchContext producer_launch{
            .stream = resources.producer_stream,
        };
        const MoENodeLocalRouteConsumeLaunch consume{
            .peers = static_cast<const MoENodeLocalRoutePeerDeviceBinding *>(
                peer_bindings_device),
            .peer_count = 1u,
            .root_canonical_route_contributions =
                static_cast<const float *>(root_routes_device),
            .domain_assignment = {
                .participant_ids =
                    static_cast<const std::int32_t *>(
                        root_participants_device),
                .capacity = route_slots,
            },
            .external_route_source =
                MoEExternalCanonicalRouteSource::DeferredDenseMerge,
            .dense_output = static_cast<float *>(output_device),
            .validation_status =
                static_cast<std::int32_t *>(validation_device),
            .root_participant = root_participant,
            .physical_rows = rows,
            .top_k = top_k,
            .d_model = d_model,
        };
        const MoENodeLocalRoutePublishLaunch publish{
            .lane = resources.exchange->producerBinding(producer_device),
            .canonical_route_contributions =
                static_cast<const float *>(producer_routes_device),
            .domain_assignment = {
                .participant_ids =
                    static_cast<const std::int32_t *>(
                        producer_participants_device),
                .capacity = route_slots,
            },
            .live_route_slots = route_slots,
        };
        ASSERT_TRUE(consume.valid());
        ASSERT_TRUE(publish.valid());

        /*
         * The immediate-stream proof below validates the lane protocol, but
         * production Ornith inference embeds the same producer and consumer
         * sequence in independent retained graphs.  Capture both endpoint
         * transactions before any epoch is published: a capture must record
         * the device-owned wait kernels rather than accidentally observing a
         * host-completed payload.  The persistent buffers and immutable launch
         * descriptors are deliberately shared with the immediate path so this
         * is a transport-lifecycle proof, not a synthetic alternate path.
         */
        std::shared_ptr<RetainedParticipantGraphExecutable> root_owner,producer_owner;
        IGPUGraphCapture* root_capture=nullptr;IGPUGraphCapture* producer_capture=nullptr;
        if (replay_through_retained_graphs)
        {
            auto &root_worker = GPUDeviceContextPool::instance().getContext(
                root_device);
            auto &producer_worker =
                GPUDeviceContextPool::instance().getContext(producer_device);
            root_owner=resources.graph(root_device,resources.root_stream);
            producer_owner=resources.graph(producer_device,resources.producer_stream);
            root_capture=&RetainedParticipantGraphExecutableFixtureAccess::capture(root_owner);
            producer_capture=&RetainedParticipantGraphExecutableFixtureAccess::capture(producer_owner);
            root_worker.submitAndWait([&] {
                ScopedBackendGraphCapture recording(root_worker,*root_capture,"original canonical route root");
                ASSERT_TRUE(recording.begin());
                ASSERT_TRUE(root_kernel->acquireNodeLocalCanonicalRoutes(
                    root_launch, consume));
                ASSERT_TRUE(root_kernel->stageNodeLocalCanonicalRoutes(
                    root_launch, consume));
                ASSERT_TRUE(root_kernel->foldNodeLocalCanonicalRoutes(
                    root_launch, consume));
                recording.finish();ASSERT_TRUE(root_capture->prepareRuntimeContextStorage(original.memory()));
                ASSERT_TRUE(root_capture->instantiate());
            });
            producer_worker.submitAndWait([&] {
                ScopedBackendGraphCapture recording(producer_worker,*producer_capture,"original canonical route producer");
                ASSERT_TRUE(recording.begin());
                ASSERT_TRUE(producer_kernel->publishNodeLocalCanonicalRoutes(
                    producer_launch, publish));
                recording.finish();ASSERT_TRUE(producer_capture->prepareRuntimeContextStorage(original.memory()));
                ASSERT_TRUE(producer_capture->instantiate());
            });
            ASSERT_NE(root_capture, nullptr);
            ASSERT_NE(producer_capture, nullptr);
            ASSERT_TRUE(root_capture->hasExecutable());
            ASSERT_TRUE(producer_capture->hasExecutable());
        }

        const auto submission_start = std::chrono::steady_clock::now();
        for (std::uint32_t epoch = 0u; epoch < epochs; ++epoch)
        {
            ASSERT_TRUE(resources.root_backend->hostToDeviceOnStream(
                root_routes_device,
                root_route_epochs[epoch].data(),
                route_bytes,
                root_device.ordinal,
                resources.root_stream));
            ASSERT_TRUE(resources.root_backend->hostToDeviceOnStream(
                root_participants_device,
                route_participant_epochs[epoch].data(),
                participant_bytes,
                root_device.ordinal,
                resources.root_stream));
            // Submit the waiter first to prove the root graph does not rely on
            // host ordering. Both endpoint streams then advance through their
            // monotonic SPSC epochs without an intervening host observation.
            if (replay_through_retained_graphs)
            {
                ASSERT_TRUE(resources.submit(root_owner));
            }
            else
            {
                ASSERT_TRUE(root_kernel->acquireNodeLocalCanonicalRoutes(
                    root_launch, consume));
                ASSERT_TRUE(root_kernel->stageNodeLocalCanonicalRoutes(
                    root_launch, consume));
                ASSERT_TRUE(root_kernel->foldNodeLocalCanonicalRoutes(
                    root_launch, consume));
            }
            ASSERT_TRUE(resources.root_backend->deviceCopyAsync(
                static_cast<std::byte *>(output_epochs_device) +
                    static_cast<std::size_t>(epoch) * output_bytes,
                output_device,
                output_bytes,
                root_device.ordinal,
                resources.root_stream));
            ASSERT_TRUE(resources.producer_backend->hostToDeviceOnStream(
                producer_routes_device,
                producer_route_epochs[epoch].data(),
                route_bytes,
                producer_device.ordinal,
                resources.producer_stream));
            ASSERT_TRUE(resources.producer_backend->hostToDeviceOnStream(
                producer_participants_device,
                route_participant_epochs[epoch].data(),
                participant_bytes,
                producer_device.ordinal,
                resources.producer_stream));
            if (replay_through_retained_graphs)
            {
                ASSERT_TRUE(
                    resources.submit(producer_owner));
            }
            else
            {
                ASSERT_TRUE(producer_kernel->publishNodeLocalCanonicalRoutes(
                    producer_launch, publish));
            }
        }
        EXPECT_LT(
            std::chrono::steady_clock::now() - submission_start,
            std::chrono::seconds(2))
            << "route exchange submission blocked on device-owned progress";
        ASSERT_TRUE(resources.root_backend->recordEvent(
            resources.root_terminal,
            root_device.ordinal,
            resources.root_stream));
        ASSERT_TRUE(resources.producer_backend->recordEvent(
            resources.producer_terminal,
            producer_device.ordinal,
            resources.producer_stream));
        ASSERT_TRUE(awaitEvent(
            resources.root_backend,
            resources.root_terminal,
            std::chrono::seconds(30),
            root_device.ordinal));
        ASSERT_TRUE(awaitEvent(
            resources.producer_backend,
            resources.producer_terminal,
            std::chrono::seconds(30),
            producer_device.ordinal));

        std::vector<float> actual(
            expected.size() * static_cast<std::size_t>(epochs), 0.0f);
        std::int32_t validation = -1;
        ASSERT_TRUE(resources.root_backend->deviceToHost(
            actual.data(),
            output_epochs_device,
            output_bytes * epochs,
            root_device.ordinal,
            resources.root_stream));
        ASSERT_TRUE(resources.root_backend->deviceToHost(
            &validation,
            validation_device,
            sizeof(validation),
            root_device.ordinal,
            resources.root_stream));
        EXPECT_EQ(
            validation,
            static_cast<std::int32_t>(
                MoENodeLocalRouteExchangeCode::Success));
        ASSERT_EQ(
            actual.size(),
            expected.size() * static_cast<std::size_t>(epochs));
        for (std::uint32_t epoch = 0u; epoch < epochs; ++epoch)
        {
            EXPECT_EQ(
                std::memcmp(
                    actual.data() + static_cast<std::size_t>(epoch) * expected.size(),
                    expected_epochs[epoch].data(), output_bytes),
                0)
                << "epoch=" << epoch << " canonical route bytes changed";
            for (std::size_t index = 0u; index < expected.size(); ++index)
            {
                EXPECT_FLOAT_EQ(
                    actual[static_cast<std::size_t>(epoch) *
                               expected.size() +
                           index],
                    expected_epochs[epoch][index])
                    << "epoch=" << epoch << " index=" << index;
            }
        }
        EXPECT_EQ(
            std::memcmp(
                actual.data() +
                    static_cast<std::size_t>(invariant_epoch_begin) *
                        expected.size(),
                actual.data() +
                    static_cast<std::size_t>(invariant_epoch_begin + 1u) *
                        expected.size(),
                output_bytes),
            0)
            << "changing route ownership changed canonical FP32 output bytes";
    }

    /**
     * @brief Exercise three root-to-peer dense epochs through captured-copy APIs.
     *
     * The peer waiter is submitted before the root on every epoch. Immediate
     * submission proves that progress remains entirely device owned; exact
     * equality after the terminal events proves that TransferEngine's D2H/H2D
     * nodes are ordered by the monotonic channel rather than host timing.
     */
    void runDenseContinuationPublication(
        OriginalMappedComponentSource& original,DeviceId root_device,
        DeviceId peer_device)
    {
        constexpr std::uint32_t rows = 64u;
        constexpr std::uint32_t top_k = 8u;
        constexpr std::uint32_t d_model = 512u;
        constexpr std::int32_t root_participant = 11;
        constexpr std::int32_t peer_participant = 37;
        constexpr std::uint32_t epochs = 3u;
        constexpr std::size_t elements =
            static_cast<std::size_t>(rows) * d_model;
        constexpr std::size_t bytes = elements * sizeof(float);

        SparseRouteExchangeResources resources(original,
            {OriginalMappedComponentProgram::Kind::Dense,rows,d_model,epochs,false},root_device,peer_device);
        ASSERT_NE(resources.root_backend, nullptr);
        ASSERT_NE(resources.producer_backend, nullptr);
        ASSERT_GT(resources.root_backend->deviceCount(), root_device.ordinal);
        ASSERT_GT(
            resources.producer_backend->deviceCount(), peer_device.ordinal);

        ASSERT_NE(resources.root_stream, nullptr);
        ASSERT_NE(resources.producer_stream, nullptr);
        ASSERT_NE(resources.root_terminal, nullptr);
        ASSERT_NE(resources.producer_terminal, nullptr);

        resources.exchange =
            std::make_unique<MoEOverlayNodeLocalRouteExchange>(
                MoEOverlayNodeLocalRouteExchange::Config{
                    .devices = {root_device, peer_device},
                    .root_device = root_device,
                    .identity = "original_component_routes",
                },MoEOverlayNodeLocalRouteExchange::declareStorage(
                    {{root_device,peer_device},root_device,"original_component_routes"},rows,top_k,d_model),original.memory(),original.inventory());
        resources.exchange->materialize(
            {
                {.participant_id = root_participant, .device = root_device},
                {.participant_id = peer_participant, .device = peer_device},
            },
            root_participant,
            rows,
            top_k,
            d_model);

        auto root_payload =
            resources.allocateTransferBuffer(root_device, bytes);
        auto peer_payload =
            resources.allocateTransferBuffer(peer_device, bytes);
        ASSERT_NE(root_payload, nullptr);
        ASSERT_NE(peer_payload, nullptr);
        ASSERT_TRUE(root_payload->isBound());
        ASSERT_TRUE(peer_payload->isBound());resources.requireDeclarationConsumed();

        std::vector<float> expected(elements);
        for (std::size_t index = 0u; index < elements; ++index)
        {
            expected[index] =
                static_cast<float>((index % 257u) + 1u) * 0.03125f;
        }
        ASSERT_TRUE(resources.root_backend->hostToDevice(
            root_payload->mutableDeviceData(),
            expected.data(),
            bytes,
            root_device.ordinal,
            resources.root_stream));

        auto root_kernel = kernelFor(root_device);
        auto peer_kernel = kernelFor(peer_device);
        ASSERT_NE(root_kernel, nullptr);
        ASSERT_NE(peer_kernel, nullptr);
        const MoEKernelLaunchContext root_launch{
            .stream = resources.root_stream,
        };
        const MoEKernelLaunchContext peer_launch{
            .stream = resources.producer_stream,
        };
        const MoENodeLocalDensePublicationLaunch root_publication{
            .binding =
                resources.exchange->densePublicationBinding(root_device),
            .element_count = elements,
        };
        const MoENodeLocalDensePublicationLaunch peer_publication{
            .binding =
                resources.exchange->densePublicationBinding(peer_device),
            .element_count = elements,
        };
        ASSERT_TRUE(root_publication.valid());
        ASSERT_TRUE(root_publication.binding.isRoot());
        ASSERT_TRUE(peer_publication.valid());
        ASSERT_FALSE(peer_publication.binding.isRoot());

        const auto region = resources.exchange->mappedRegion();
        const std::size_t payload_offset =
            resources.exchange->densePublicationPayloadOffset();
        ASSERT_NE(region, nullptr);
        const auto submission_start = std::chrono::steady_clock::now();
        for (std::uint32_t epoch = 0u; epoch < epochs; ++epoch)
        {
            ASSERT_TRUE(peer_kernel->beginNodeLocalDensePublicationConsume(
                peer_launch, peer_publication));
            TransferEngine::instance().enqueueMappedHostToDevice(
                *region,
                payload_offset,
                *peer_payload,
                0u,
                bytes,
                peer_device,
                resources.producer_stream);
            ASSERT_TRUE(peer_kernel->finishNodeLocalDensePublicationConsume(
                peer_launch, peer_publication));

            ASSERT_TRUE(root_kernel->beginNodeLocalDensePublication(
                root_launch, root_publication));
            TransferEngine::instance().enqueueDeviceToMappedHost(
                *root_payload,
                0u,
                *region,
                payload_offset,
                bytes,
                root_device,
                resources.root_stream);
            ASSERT_TRUE(root_kernel->finishNodeLocalDensePublication(
                root_launch, root_publication));
        }
        EXPECT_LT(
            std::chrono::steady_clock::now() - submission_start,
            std::chrono::seconds(2))
            << "dense publication submission blocked on peer progress";

        ASSERT_TRUE(resources.root_backend->recordEvent(
            resources.root_terminal,
            root_device.ordinal,
            resources.root_stream));
        ASSERT_TRUE(resources.producer_backend->recordEvent(
            resources.producer_terminal,
            peer_device.ordinal,
            resources.producer_stream));
        ASSERT_TRUE(awaitEvent(
            resources.root_backend,
            resources.root_terminal,
            std::chrono::seconds(30),
            root_device.ordinal));
        ASSERT_TRUE(awaitEvent(
            resources.producer_backend,
            resources.producer_terminal,
            std::chrono::seconds(30),
            peer_device.ordinal));

        std::vector<float> actual(elements, 0.0f);
        ASSERT_TRUE(resources.producer_backend->deviceToHost(
            actual.data(),
            peer_payload->deviceData(),
            bytes,
            peer_device.ordinal,
            resources.producer_stream));
        ASSERT_EQ(actual.size(), expected.size());
        for (std::size_t index = 0u; index < actual.size(); ++index)
            EXPECT_FLOAT_EQ(actual[index], expected[index]) << "index=" << index;
    }

    /** @brief One actual original source covers the full existing ticket geometry/policy roster. */
    void originalCanonicalTicketReplay(DeviceType family) {
        originalRankLocalReplay(family,1,[&](const auto& devices) {
            std::vector<OriginalMappedComponentProgram> programs;
            for(const auto [slots,width]:std::array<std::pair<std::size_t,std::size_t>,10>{{
                {8,3072},{512,3072},{4800,3072},{19,67},{128,3072},{19,67},{19,257},{19,513},{640,2560},{4480,2560}}})
                programs.push_back({OriginalMappedComponentProgram::Kind::Ticket,slots,width,20,true});
            OriginalMappedComponentSource original(devices,programs);
            for(std::size_t profile=0;profile<programs.size();++profile)
                runCanonicalRouteTicketReuseProof(original,devices[0],programs[profile].rows,programs[profile].width,
                    profile<4?TicketRoutePattern::Full:TicketRoutePattern::Varying);
        });
    }
    /** @brief Original sparse row/stride record; runtime ownership still comes from real discovery. */
    struct OriginalSparseProfile {std::uint32_t rows=64,width=512,epochs=5;bool retained=false;std::uint32_t stride=0;};
    /** @brief Admit one real local pair once and preserve each original sparse numerical program. */
    void originalSparseReplay(DeviceType first,DeviceType second,std::vector<OriginalSparseProfile> profiles) {
        const std::vector<NativeReplayFamily> families=first==second?std::vector<NativeReplayFamily>{{first,2}}:
            std::vector<NativeReplayFamily>{{first,1},{second,1}};
        originalRankLocalReplay(families,[&](const auto& devices) {
            std::vector<OriginalMappedComponentProgram> programs;
            for(const auto& profile:profiles)programs.push_back({OriginalMappedComponentProgram::Kind::Sparse,
                profile.rows,profile.width,profile.epochs,profile.retained});
            OriginalMappedComponentSource original(devices,programs);
            for(const auto& profile:profiles)runSparseCanonicalRouteExchange(original,devices[0],devices[1],
                profile.rows,profile.width,profile.epochs,profile.retained,profile.stride);
        });
    }
    /** @brief Dense publication stays rank-local on its two observed native owners. */
    void originalDenseReplay(DeviceType first,DeviceType second) {
        const std::vector<NativeReplayFamily> families=first==second?std::vector<NativeReplayFamily>{{first,2}}:
            std::vector<NativeReplayFamily>{{first,1},{second,1}};
        originalRankLocalReplay(families,[&](const auto& devices) {
            OriginalMappedComponentSource original(devices,{{OriginalMappedComponentProgram::Kind::Dense,64,512,3,false}});
            runDenseContinuationPublication(original,devices[0],devices[1]);
        });
    }
} // namespace


TEST(Test__MappedActivationPacketCUDAAndROCm,
     CPUCanonicalRouteTicketReusesSafelyOnCUDA)
{
    originalCanonicalTicketReplay(DeviceType::CUDA);
}

TEST(Test__MappedActivationPacketCUDAAndROCm,
     CPUCanonicalRouteTicketReusesSafelyOnROCm)
{
    originalCanonicalTicketReplay(DeviceType::ROCm);
}

TEST(Test__MappedActivationPacketCUDAAndROCm,
     SparseCanonicalRoutesAcrossTwoCUDADevicesAreExactAndAsync)
{
    originalSparseReplay(DeviceType::CUDA,DeviceType::CUDA,{{}});
}

/**
 * @brief Prove the mapped lane at Ornith's actual 39-row Q4_K prefill geometry.
 *
 * The numerical model defect begins on a 39-token prefill with a 2,048-wide
 * hidden state.  The generic 64-by-512 proof is deliberately retained because
 * it covers a distinct lane shape, while this focused geometry closes the
 * capacity/page-layout gap without pretending that synthetic payload values
 * are a substitute for the real-weight HF checkpoint test.
 */
TEST(Test__MappedActivationPacketCUDAAndROCm,
     SparseCanonicalRoutesAcrossTwoCUDADevicesMatchOrnithQ4KPrefillGeometry)
{
    originalSparseReplay(DeviceType::CUDA,DeviceType::CUDA,{{39,2048}});
}

/**
 * @brief Retain one complete 40-layer mapped route epoch sequence on CUDA.
 *
 * A graph-recording warmup and its first retained replay can advance the same
 * lane many times before the parity harness inspects layer zero.  This test
 * therefore crosses more than Ornith's 40 main layers at the exact prefill
 * geometry, ensuring that slot tags, acknowledgements, and root scratch reuse
 * do not merely work for the five-epoch unit-sized probe.
 */
TEST(Test__MappedActivationPacketCUDAAndROCm,
     SparseCanonicalRoutesAcrossTwoCUDADevicesSurviveOrnithModelEpochCount)
{
    originalSparseReplay(DeviceType::CUDA,DeviceType::CUDA,{{39,2048,48}});
}

/**
 * @brief Prove Ornith-sized sparse route exchange under the retained CUDA graph.
 *
 * The production LocalTP ExpertOverlay path captures a distinct producer graph
 * on CUDA:1 and root graph on CUDA:0.  This regression submits the root wait
 * before every peer replay, changes route ownership and payload bytes for all
 * 48 main-model epochs, and checks every completed FP32 fold byte-for-byte.
 * It therefore catches a capture-only mapped-alias or system-fence regression
 * that an immediate stream submission cannot expose.
 */
TEST(Test__MappedActivationPacketCUDAAndROCm,
     SparseCanonicalRoutesAcrossTwoCUDADevicesRemainExactUnderRetainedGraphsAtOrnithGeometry)
{
    originalSparseReplay(DeviceType::CUDA,DeviceType::CUDA,{{39,2048,48,true}});
}

TEST(Test__MappedActivationPacketCUDAAndROCm,
     SparseCanonicalRoutesAcrossTwoROCmDevicesAreExactAndAsync)
{
    originalSparseReplay(DeviceType::ROCm,DeviceType::ROCm,{{}});
}

/**
 * @brief Prove root-local staging/cache reuse at the 122B verifier envelope.
 *
 * The same retained graphs consume twenty different payloads and owner maps.
 * An aligned model width exercises vector reads; an extra column exercises
 * unaligned rows/tails. Cancellation-heavy final epochs lock the FP32 fold
 * order independently of which participant owns each original router slot.
 */
TEST(Test__MappedActivationPacketCUDAAndROCm,
     CUDASparseCanonicalRoutesRetainVerifierBytesAcrossTwentyEpochs)
{
    originalSparseReplay(DeviceType::CUDA,DeviceType::CUDA,{{16,3072,20,true,17},{16,3073,20,true,17}});
}

/** @brief Symmetric HIP proof of staged route lifetime, tails and fold order. */
TEST(Test__MappedActivationPacketCUDAAndROCm,
     ROCmSparseCanonicalRoutesRetainVerifierBytesAcrossTwentyEpochs)
{
    originalSparseReplay(DeviceType::ROCm,DeviceType::ROCm,{{16,3072,20,true,17},{16,3073,20,true,17}});
}

TEST(Test__MappedActivationPacketCUDAAndROCm,
     SparseCanonicalRoutesAcrossCUDAAndROCmAreExactAndAsync)
{
    originalSparseReplay(DeviceType::CUDA,DeviceType::ROCm,{{}});
}

/**
 * @brief Preserve full-prefill route bytes through independently captured endpoints.
 *
 * Adjacent bucket/tail geometries exercise uneven rows, empty peer payloads,
 * changed ownership, and cancellation-sensitive canonical route order.
 * Each retained graph is reused for twenty device-owned publication epochs.
 */
TEST(Test__MappedActivationPacketCUDAAndROCm,
     CUDASparseCanonicalRoutesRetainPrefillBytesAcrossTwentyEpochs)
{
    originalSparseReplay(DeviceType::CUDA,DeviceType::CUDA,{{63,2049,20,true,17},{64,2049,20,true,17},{65,2049,20,true,17},{512,2049,20,true,17}});
}

/** @brief Symmetric retained-HIP proof of full-prefill publication and reuse. */
TEST(Test__MappedActivationPacketCUDAAndROCm,
     ROCmSparseCanonicalRoutesRetainPrefillBytesAcrossTwentyEpochs)
{
    originalSparseReplay(DeviceType::ROCm,DeviceType::ROCm,{{63,2049,20,true,17},{64,2049,20,true,17},{65,2049,20,true,17},{512,2049,20,true,17}});
}

TEST(Test__MappedActivationPacketCUDAAndROCm,
     DenseContinuationPublicationAcrossTwoCUDADevicesIsExactAndAsync)
{
    originalDenseReplay(DeviceType::CUDA,DeviceType::CUDA);
}

TEST(Test__MappedActivationPacketCUDAAndROCm,
     DenseContinuationPublicationAcrossTwoROCmDevicesIsExactAndAsync)
{
    originalDenseReplay(DeviceType::ROCm,DeviceType::ROCm);
}

TEST(Test__MappedActivationPacketCUDAAndROCm,
     DenseContinuationPublicationAcrossCUDAAndROCmIsExactAndAsync)
{
    originalDenseReplay(DeviceType::CUDA,DeviceType::ROCm);
}
