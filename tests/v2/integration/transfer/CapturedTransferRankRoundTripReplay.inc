/**
 * @file CapturedTransferRankRoundTripReplay.inc
 * @brief Original node-local MPI endpoints retain bidirectional GPU graphs.
 *
 * The original rank session owns shared pages. Each process prepares only its
 * NUMA-owned device and admits its scratch, snapshots, captured graph family
 * and optional retained workspace. Terminal observations verify live extents,
 * poison guards and retirement without exchanging process-local pointers.
 */
#include "collective/CollectiveTimeoutPolicy.h"
#include <optional>

namespace
{
/**
 * @brief Preserve local round-trip oracles on distinct original ranks.
 * @param first Original source/result GPU family.
 * @param second Original receive/reply GPU family.
 * @param workspace_peer Retain the bank after its workspace namespace retires.
 * @details Known replay inputs are uploaded at explicit test request admission.
 * Counts then move through the production GPU protocol. Full, small and empty
 * admissions reuse the same retained graphs. Only terminal evidence uses MPI.
 */
void rankCapturedRoundTripReplay(DeviceType first, DeviceType second, bool workspace_peer)
{
    const auto mpi = MPIContextFactory::global();
    require(mpi && mpi->world_size() >= 2 && mpi->clusterInventory(),
        "Rank round trip requires original multi-rank MPI discovery");
    const auto endpoints = rankReplayEndpoints(*mpi->clusterInventory(), first, second);
    const ChannelWorkload workload = workspace_peer
        ? ChannelWorkload{.capacity = 257, .bytes = {4, 17, 257}, .offsets = {1, 16},
            .replays = 8, .patterns = 2, .peer_storage = PeerStorage::RetainedWorkspace}
        : ChannelWorkload{};
    const bool source_rank = mpi->rank() == endpoints[0].rank;
    const bool active = source_rank || mpi->rank() == endpoints[1].rank;
    const auto local = active ? endpoints[source_rank ? 0 : 1].device : DeviceId::invalid();
    const auto buffer_bytes = workload.capacity + 64;
    const auto snapshot_bytes = buffer_bytes + sizeof(std::uint64_t);
    const auto mapped_snapshot_bytes = TransferEngine::mappedHostRegionAllocationBytes(snapshot_bytes);
    const auto mapped_input_bytes = TransferEngine::mappedHostRegionAllocationBytes(buffer_bytes);
    // Original source is lexical first and retires after every payload owner.
    std::unique_ptr<test::PreparedNativeLookupFixture> original;
    auto session = CapturedTransferRankSession::create(mpi);
    const std::array plans{
        CapturedTransferRankPlan::fromInventory(*mpi->clusterInventory(), {session->originalGeneration(), 711},
            endpoints[0], endpoints[1], workload.capacity),
        CapturedTransferRankPlan::fromInventory(*mpi->clusterInventory(), {session->originalGeneration(), 712},
            endpoints[1], endpoints[0], workload.capacity)};
    std::shared_ptr<PhysicalMemoryAuthority> memory;
    IWorkerGPUContext *worker = nullptr;
    IBackend *backend = nullptr;
    void *stream = nullptr;
    exchangePlanningSamples(mpi, [&](int count) {
        return std::vector<std::vector<std::uint8_t>>(count, plans[0].serialize());
    }, [&](auto wire) {
        require(std::vector<std::uint8_t>(wire.begin(), wire.end()) == plans[0].serialize(),
            "Rank round-trip declarations disagree");
        if (active)
        {
            const std::array devices{local};
            original = test::PreparedNativeLookupFixture::prepare(devices, mpi->rank());
            PhysicalMemoryPlanBuilder builder;
            for (const auto &plan : plans)
                plan.contributeMemory(*plan.localRole(mpi->rank()), original->resource(local),
                    plan.hostRank() == mpi->rank() ? std::optional{original->resource(DeviceId::cpu())} : std::nullopt,
                    builder);
            builder.add(original->resource(DeviceId::cpu()), PhysicalMemoryOwner::ActivationTransportStaging,
                mapped_snapshot_bytes + (source_rank ? 2 * mapped_input_bytes : 0));
            builder.add(original->resource(local), PhysicalMemoryOwner::ActivationTransportStaging,
                sizeof(std::uint64_t) + (source_rank ? 3 : workspace_peer ? 0 : 1) * buffer_bytes);
            if (!source_rank && workspace_peer)
                builder.add(original->resource(local), PhysicalMemoryOwner::ExecutionWorkspace, buffer_bytes);
            builder.add(original->resource(local), PhysicalMemoryOwner::NativeGraphExecutable,
                2 * GPUGraphMemoryContract::reservationBytesPerExecutable(local,
                    GPUGraphExecutableClass::BoundedFlatHelper));
            original->admit(builder.build());
            memory = original->memory();
            worker = original->permit(local).worker().get();
            backend = getBackendFor(local);
            require(worker && backend, "Rank round-trip source lost its original worker/backend");
            worker->submitAndWait([&] {
                stream = worker->getOrCreateAuxiliaryStream("rank_captured_round_trip");
            });
        }
        return std::vector<std::uint8_t>{1};
    }, [](auto receipts) {
        for (const auto &receipt : receipts)
            require(receipt.bytes.size() == 1 && receipt.bytes[0] == 1, "Rank round-trip setup omitted a participant");
    });
    auto &transfer = TransferEngine::instance();
    // Print the backend exception before fatal retirement protects live owners.
    const auto submit = [&](auto operation) {
        try { worker->submitAndWait(std::move(operation)); }
        catch (const std::exception &error)
        {
            std::cerr << "Original rank " << mpi->rank() << " round-trip worker failed: " << error.what() << '\n';
            throw;
        }
    };
    std::array<std::shared_ptr<CapturedTransferChannel>, 2> channels;
    // Fatal cleanup precedes every channel, graph and physical buffer release.
    // Actual native failure cannot unwind shared pages while a peer is active.
    const auto abort_failure = [mpi] {
        std::cerr << "Original rank captured round-trip failed before terminal retirement\n";
        MPI_Abort(mpi->communicator(), 2);
        std::terminate();
    };
    NativeReplayFailureDrain fatal{abort_failure};
    for (size_t direction = 0; direction < plans.size(); ++direction)
        channels[direction] = transfer.createRankCapturedTransferChannel(plans[direction], session, memory, stream);
    {
        PhysicalMemoryAllocationLease payload_claim, snapshot_claim;
        std::shared_ptr<DeviceTransferBuffer> count, result, peer;
        std::array<std::shared_ptr<DeviceTransferBuffer>, 2> sources;
        std::shared_ptr<const WorkspaceBufferLease> peer_region;
        std::shared_ptr<MappedHostTransferRegion> snapshot;
        std::array<std::shared_ptr<MappedHostTransferRegion>, 2> inputs;
        std::optional<Terminal> terminal;
        NativeReplayFailureDrain payload_fatal{abort_failure};
        if (active)
        {
            payload_claim = memory->claimNewAllocation(local, PhysicalMemoryOwner::ActivationTransportStaging,
                sizeof(std::uint64_t) + (source_rank ? 3 : workspace_peer ? 0 : 1) * buffer_bytes);
            snapshot_claim = memory->claimNewAllocation(DeviceId::cpu(), PhysicalMemoryOwner::ActivationTransportStaging,
                mapped_snapshot_bytes + (source_rank ? 2 * mapped_input_bytes : 0));
            count = transfer.allocateDeviceTransferBuffer(sizeof(std::uint64_t), local);
            const std::array devices{local};
            snapshot = transfer.allocateMappedHostRegion(snapshot_bytes, devices);
            if (source_rank)
            {
                result = transfer.allocateDeviceTransferBuffer(buffer_bytes, local);
                for (size_t pattern = 0; pattern < 2; ++pattern)
                {
                    sources[pattern] = transfer.allocateDeviceTransferBuffer(buffer_bytes, local);
                    inputs[pattern] = transfer.allocateMappedHostRegion(buffer_bytes, devices);
                    auto *data = static_cast<unsigned char *>(inputs[pattern]->mutableHostData());
                    for (size_t index = 0; index < buffer_bytes; ++index)
                        data[index] = patternByte(index, pattern);
                    submit([&] {
                        transfer.enqueueMappedHostToDevice(*inputs[pattern], 0, *sources[pattern], 0,
                            buffer_bytes, local, stream);
                    });
                }
            }
            else if (workspace_peer)
            {
                DeviceWorkspaceManager workspace(local, buffer_bytes, memory);
                WorkspaceRequirements requirements;
                requirements.buffers.push_back({"rank_pipeline_metadata", buffer_bytes, 1, true});
                require(workspace.allocate(requirements), "Rank round-trip workspace allocation failed");
                peer_region = workspace.retainBuffer("rank_pipeline_metadata", 0, buffer_bytes);
                workspace.release();
                require(memory->claimedBytes(local, PhysicalMemoryOwner::ExecutionWorkspace,
                    PhysicalMemoryMaterializationKind::NewAllocation) == buffer_bytes,
                    "Rank round-trip namespace retirement released captured workspace bytes");
                EXPECT_THROW((void)transfer.bindCapturedTransfer(channels[0], CapturedTransferEndpoint::Producer,
                    {11, 1}, peer_region), std::invalid_argument);
                EXPECT_THROW((void)transfer.bindCapturedTransfer(channels[0], CapturedTransferEndpoint::Consumer,
                    {11, 1}, peer_region, buffer_bytes), std::out_of_range);
            }
            else
                peer = transfer.allocateDeviceTransferBuffer(buffer_bytes, local);
            terminal.emplace(backend, local, backend->createEvent(local.ordinal));
            require(terminal->event != nullptr, "Rank round-trip terminal event is absent");
        }
        std::uint64_t invocation_index = 0;
        for (const auto bytes : workload.bytes)
            for (const auto offset : workload.offsets)
            {
                PhysicalMemoryOwnerReservation graph_claim;
                std::array<std::unique_ptr<IGPUGraphCapture>, 2> graphs;
                NativeReplayFailureDrain graph_fatal{abort_failure};
                if (active)
                {
                    graph_claim = memory->reserveNewAllocations(local, PhysicalMemoryOwner::NativeGraphExecutable,
                        2 * GPUGraphMemoryContract::reservationBytesPerExecutable(local,
                            GPUGraphExecutableClass::BoundedFlatHelper));
                    for (size_t pattern = 0; pattern < 2; ++pattern)
                    {
                        auto outward = source_rank
                            ? transfer.bindCapturedTransfer(channels[0], CapturedTransferEndpoint::Producer,
                                {11, bytes}, sources[pattern], offset)
                            : workspace_peer
                                ? transfer.bindCapturedTransfer(channels[0], CapturedTransferEndpoint::Consumer,
                                    {11, bytes}, peer_region, offset)
                                : transfer.bindCapturedTransfer(channels[0], CapturedTransferEndpoint::Consumer,
                                    {11, bytes}, peer, offset);
                        auto inward = source_rank
                            ? transfer.bindCapturedTransfer(channels[1], CapturedTransferEndpoint::Consumer,
                                {12, bytes}, result, offset)
                            : workspace_peer
                                ? transfer.bindCapturedTransfer(channels[1], CapturedTransferEndpoint::Producer,
                                    {12, bytes}, peer_region, offset)
                                : transfer.bindCapturedTransfer(channels[1], CapturedTransferEndpoint::Producer,
                                    {12, bytes}, peer, offset);
                        outward = transfer.bindDeviceCountedTransfer(std::move(outward), count, 0);
                        inward = transfer.bindDeviceCountedTransfer(std::move(inward), count, 0);
                        const std::array identity{outward.captureIdentity(), inward.captureIdentity()};
                        // Native graph factories belong to the original setup
                        // caller. Recording uses the exact original GPU worker.
                        graphs[pattern] = worker->createGraphCapture(stream);
                        require(bool(graphs[pattern]), "Rank round-trip graph factory is absent");
                        submit([&] {
                            ScopedBackendGraphCapture capture(*worker, *graphs[pattern], "original rank captured round trip");
                            require(capture.begin(), "Rank round-trip capture begin failed");
                            void *destination = source_rank ? result->mutableDeviceData()
                                : workspace_peer ? peer_region->data() : peer->mutableDeviceData();
                            require(backend->memset(destination, 0xa5, buffer_bytes, local.ordinal, stream),
                                "Rank round-trip captured guard initialization failed");
                            transfer.enqueueCapturedTransfer(outward, stream);
                            transfer.enqueueCapturedTransfer(inward, stream);
                            capture.finish();
                            counted_channel_test::verifyCapturedChannelGraph(*graphs[pattern], bytes, 2);
                            std::string error;
                            require(graphs[pattern]->validateExecutableMemoryClass(
                                GPUGraphExecutableClass::BoundedFlatHelper, &error), error.c_str());
                            require(graphs[pattern]->prepareRuntimeContextStorage(memory) && graphs[pattern]->instantiate(),
                                "Rank round-trip graph admission/instantiation failed");
                        });
                        require(identity == std::array{outward.captureIdentity(), inward.captureIdentity()},
                            "Rank round-trip recording changed original capture identity");
                    }
                }
                // Full -> partial -> empty -> full admissions reuse these exact
                // definitions. The other replays retain full useful payloads
                // and immutable alternating source patterns.
                for (size_t replay = 0; replay < workload.replays; ++replay)
                {
                    const size_t pattern = replay % 2;
                    const std::uint64_t live_bytes = replay % 8 == 2 ? std::min<size_t>(bytes, 3)
                        : replay % 8 == 3 ? 0 : bytes;
                    const CapturedTransferInvocation invocation{session->originalGeneration(), ++invocation_index};
                    if (active)
                    {
                        for (const auto &channel : channels)
                            transfer.beginCapturedTransferInvocation(channel, invocation);
                        submit([&] {
                            if (source_rank)
                            {
                                std::memcpy(static_cast<std::byte *>(snapshot->mutableHostData()) + buffer_bytes,
                                    &live_bytes, sizeof(live_bytes));
                                transfer.enqueueMappedHostToDevice(*snapshot, buffer_bytes, *count, 0,
                                    sizeof(live_bytes), local, stream);
                            }
                            require(graphs[pattern]->launch(), "Rank round-trip retained launch failed");
                            if (source_rank)
                                transfer.enqueueDeviceToMappedHost(*result, 0, *snapshot, 0, buffer_bytes, local, stream);
                            else if (workspace_peer)
                                transfer.enqueuePersistentDeviceRegionToMappedHost(peer_region->data(), buffer_bytes, 0,
                                    *snapshot, 0, buffer_bytes, local, stream);
                            else
                                transfer.enqueueDeviceToMappedHost(*peer, 0, *snapshot, 0, buffer_bytes, local, stream);
                            transfer.enqueueDeviceToMappedHost(*count, 0, *snapshot, buffer_bytes, sizeof(live_bytes), local, stream);
                            require(backend->recordEvent(terminal->event, local.ordinal, stream),
                                "Rank round-trip original terminal publication failed");
                        });
                        terminal->await(std::chrono::duration_cast<std::chrono::seconds>(
                            std::chrono::milliseconds(collective_timeout_policy::kDefaultCollectiveTimeoutMs)));
                    }
                    mpi->barrier(); // Exact test terminal observation follows local completion.
                    if (active)
                    {
                        const auto *actual = static_cast<const unsigned char *>(snapshot->mutableHostData());
                        for (size_t index = 0; index < buffer_bytes; ++index)
                        {
                            const auto expected = index >= offset && index - offset < live_bytes
                                ? patternByte(index, pattern) : static_cast<unsigned char>(0xa5);
                            require(actual[index] == expected, "Rank round-trip payload, pattern or inactive guard mismatch");
                        }
                        std::uint64_t actual_bytes = 0;
                        std::memcpy(&actual_bytes, actual + buffer_bytes, sizeof(actual_bytes));
                        require(actual_bytes == live_bytes, "Rank round-trip device-owned receive count is stale");
                        for (const auto &channel : channels)
                        {
                            const auto &control = CapturedTransferChannelReplayProbe::terminalControl(*channel);
                            require(control.producer.epoch == invocation_index && control.consumer.epoch == invocation_index &&
                                control.producer.message.bytes == live_bytes && control.consumer.message.bytes == live_bytes,
                                "Rank round-trip communicated inactive capacity or lost an empty metadata receipt");
                            require(transfer.finishCapturedTransferInvocation(channel, invocation) ==
                                CapturedTransferInvocationRetirement::Completed,
                                "Rank round-trip completed an incompatible original admission");
                        }
                    }
                    // Both original ranks finish reading shared terminal
                    // controls before either can publish the next epoch.
                    mpi->barrier();
                }
                if (active)
                    submit([&] { graphs = {}; });
                graph_fatal.completed = true;
            }
        payload_fatal.completed = true;
    }
    mpi->barrier();
    channels = {};
    session->retireAfterOriginalGraphRelease();
    require(session->retired(), "Rank round-trip original shared mappings survived retirement");
    if (original)
    {
        original->requirePayloadRetired();
        original->retire();
        original->requireExactRetired();
    }
    fatal.completed = true;
}
} // namespace

#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
/** @test Alternating original rank captures preserve bidirectional live bytes and guards. */
TEST(CapturedTransferRankDevices, CUDA_ROCm_RetainedRoundTrip)
{ rankCapturedRoundTripReplay(DeviceType::CUDA, DeviceType::ROCm, false); }
/** @test Reverse original rank captures retain the complete round-trip oracle. */
TEST(CapturedTransferRankDevices, ROCm_CUDA_RetainedRoundTrip)
{ rankCapturedRoundTripReplay(DeviceType::ROCm, DeviceType::CUDA, false); }
/** @test Remote original peers retain workspace bytes after namespace destruction. */
TEST(CapturedTransferRankDevices, CUDA_ROCm_RetainedWorkspaceRoundTrip)
{ rankCapturedRoundTripReplay(DeviceType::CUDA, DeviceType::ROCm, true); }
/** @test Reverse workspace ownership retires after captured borrowers. */
TEST(CapturedTransferRankDevices, ROCm_CUDA_RetainedWorkspaceRoundTrip)
{ rankCapturedRoundTripReplay(DeviceType::ROCm, DeviceType::CUDA, true); }
#endif
