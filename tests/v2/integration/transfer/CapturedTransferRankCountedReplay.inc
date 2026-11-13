/**
 * @file CapturedTransferRankCountedReplay.inc
 * @brief Original-rank GPU-authored source slices and multichannel packet replay.
 *
 * These traces reuse the original local counted-kernel arithmetic, capacity
 * matrix, source Tensor/descriptor publication ledger, packet geometry checks
 * and poison oracle. Source preparation spans the entire trace family, while
 * each mutually exclusive case releases its exact PMA-owned banks and graphs.
 */
namespace
{
/** @brief The existing local counted fixture's independently qualified traces. */
enum class RankCountedTrace { Counted, SourceSlices, FusedBoundaries, BatchedEndpoints, Batch16KiBBoundaries };

/** @brief One immutable capture geometry, separate from GPU-authored live state. */
struct RankCountedCase
{
    std::vector<size_t> capacities; ///< One exact maximum per original independent endpoint.
    bool sliced = false; ///< The source GPU authors a descriptor instead of an extent scalar.
    bool tensor_source = false; ///< Source and descriptor are distinct original Tensor roots.
    bool batched = false; ///< Use the production argument-packet enqueue API.
    std::vector<counted_channel_test::CapturedChannelBatchPacketProof> packets; ///< Exact mixed-packet oracle.
};

/** @brief Preserve original local trace declarations.
 * @param trace Existing counted, sliced or argument-packet family.
 * @return The unchanged capacities, source representations and packet variants.
 */
std::vector<RankCountedCase> rankCountedCases(RankCountedTrace trace)
{
    using Packet = counted_channel_test::CapturedChannelBatchPacketProof;
    std::vector<RankCountedCase> cases;
    switch (trace)
    {
    case RankCountedTrace::Counted:
        cases.push_back({{1024 * 1024 + 3}});
        break;
    case RankCountedTrace::SourceSlices:
        for (const bool batched : {false, true})
            for (const size_t capacity : {size_t{257}, size_t{16385}})
                for (const bool tensor_source : {false, true})
                    cases.push_back({{capacity}, true, tensor_source, batched});
        break;
    case RankCountedTrace::FusedBoundaries:
        for (const size_t capacity : {size_t{257}, size_t{4099},
                size_t{kCapturedTransferSingleBlockBytes}, size_t{kCapturedTransferSingleBlockBytes + 1}})
            cases.push_back({{capacity}});
        break;
    case RankCountedTrace::BatchedEndpoints:
        cases.push_back({std::vector<size_t>(3, 4099), false, false, true});
        cases.push_back({std::vector<size_t>(4, kCapturedTransferSingleBlockBytes + 1), false, false, true});
        cases.push_back({std::vector<size_t>(3, 1024 * 1024 + 3), false, false, true});
        cases.push_back({{4099, kCapturedTransferSingleBlockBytes + 1, 1024 * 1024 + 3, 257},
            false, false, true, std::vector<Packet>{
                {3, CapturedTransferKernelPlan::ParallelCopy, 32}, {1, CapturedTransferKernelPlan::SingleBlock, 1}}});
        break;
    case RankCountedTrace::Batch16KiBBoundaries:
        for (const size_t capacity : {size_t{8191}, size_t{8192}, size_t{8193}, size_t{10240},
                size_t{16383}, size_t{16384}, size_t{16385}})
            cases.push_back({std::vector<size_t>(capacity == 16384 ? 4 : 3, capacity), false, false, true});
        cases.push_back({{8193, 10240, 16385, 16384}, false, false, true, std::vector<Packet>{
                {3, CapturedTransferKernelPlan::ParallelCopy, 32}, {1, CapturedTransferKernelPlan::SingleBlock, 1}}});
        break;
    }
    return cases;
}

/** @brief Geometry shared by the exact admission envelope and one case materialization. */
struct RankCountedStorage
{
    size_t bytes = 0; ///< Payload and complete outer guard extent.
    size_t count_bytes = 0; ///< Three independent UINT64 controls per endpoint.
    size_t tensor_bytes = 0; ///< Original source Tensor extent rounded only to its INT32 element.
    std::vector<size_t> origins; ///< Prefix sum of original per-endpoint bank extents.
    WorkspaceRequirements workspace; ///< Exact participant-local namespace layout.
};

/**
 * @brief Derive source/consumer storage directly from original operation declarations.
 * @param geometry Exact case maxima and original source representation.
 * @param source Whether this original rank owns the source/result stage.
 */
RankCountedStorage rankCountedStorage(const RankCountedCase &geometry, bool source)
{
    RankCountedStorage result;
    for (const auto capacity : geometry.capacities)
    {
        result.origins.push_back(result.bytes);
        result.bytes += capacity + (geometry.sliced ? 128 : 64);
    }
    result.count_bytes = geometry.capacities.size() * 3 * sizeof(std::uint64_t);
    result.tensor_bytes = (result.bytes + sizeof(std::int32_t) - 1) / sizeof(std::int32_t) * sizeof(std::int32_t);
    if (!source || !geometry.tensor_source)
        result.workspace.buffers.push_back({"payload", result.bytes, 16, true});
    if (source)
        result.workspace.buffers.push_back({"returned", result.bytes, 16, true});
    result.workspace.buffers.push_back({"counts", result.count_bytes, 8, true});
    return result;
}

/** @brief Dispatch the existing GPU-authored writer without publishing host live counts. */
void rankPrepareCounted(DeviceId device, void *payload, std::uint64_t *count,
    const std::uint64_t *sequence, size_t capacity, void *stream)
{
#ifdef HAVE_CUDA
    if (device.is_cuda())
    { require(counted_channel_test::prepareCUDA(payload, count, sequence, capacity, stream), "Original CUDA count writer failed"); return; }
#endif
#ifdef HAVE_ROCM
    if (device.is_rocm())
    { require(counted_channel_test::prepareROCm(payload, count, sequence, capacity, stream), "Original ROCm count writer failed"); return; }
#endif
    throw std::invalid_argument("Counted rank trace has no native writer");
}

/** @brief Dispatch the existing captured source-descriptor writer on its original GPU. */
void rankPrepareSlice(DeviceId device, void *payload, CapturedTransferSourceSlice *slice,
    const std::uint64_t *sequence, size_t capacity, size_t span, std::uint64_t origin, void *stream)
{
#ifdef HAVE_CUDA
    if (device.is_cuda())
    { require(counted_channel_test::prepareSlicedCUDA(payload, slice, sequence, capacity, span, origin, stream), "Original CUDA slice writer failed"); return; }
#endif
#ifdef HAVE_ROCM
    if (device.is_rocm())
    { require(counted_channel_test::prepareSlicedROCm(payload, slice, sequence, capacity, span, origin, stream), "Original ROCm slice writer failed"); return; }
#endif
    throw std::invalid_argument("Sliced rank trace has no native writer");
}

/** @brief Advance the existing sequence only after its original round trip completes. */
void rankAdvanceCounted(DeviceId device, std::uint64_t *sequence, void *stream)
{
#ifdef HAVE_CUDA
    if (device.is_cuda())
    { require(counted_channel_test::advanceCUDA(sequence, stream), "Original CUDA sequence advance failed"); return; }
#endif
#ifdef HAVE_ROCM
    if (device.is_rocm())
    { require(counted_channel_test::advanceROCm(sequence, stream), "Original ROCm sequence advance failed"); return; }
#endif
    throw std::invalid_argument("Counted rank trace has no native sequence controller");
}

/**
 * @brief Replay the original counted/sliced/packet oracle through actual MPI-owned GPUs.
 * @param first Original producer/result family.
 * @param second Original receiver/reply family.
 * @param trace Exact existing local trace family.
 */
void rankCapturedCountedReplay(DeviceType first, DeviceType second, RankCountedTrace trace)
{
    const auto mpi = MPIContextFactory::global();
    require(mpi && mpi->world_size() >= 2 && mpi->clusterInventory(), "Counted rank trace requires original discovery");
    const auto endpoints = rankReplayEndpoints(*mpi->clusterInventory(), first, second);
    const bool source = mpi->rank() == endpoints[0].rank;
    const bool active = source || mpi->rank() == endpoints[1].rank;
    const auto local = active ? endpoints[source ? 0 : 1].device : DeviceId::invalid();
    const auto cases = rankCountedCases(trace);
    std::unique_ptr<test::PreparedNativeLookupFixture> original;
    auto session = CapturedTransferRankSession::create(mpi);
    std::vector<std::vector<std::array<CapturedTransferRankPlan, 2>>> plans;
    std::uint64_t boundary = 801;
    for (const auto &geometry : cases)
    {
        std::vector<std::array<CapturedTransferRankPlan, 2>> lanes;
        for (const auto capacity : geometry.capacities)
        {
            lanes.push_back({
                CapturedTransferRankPlan::fromInventory(*mpi->clusterInventory(), {session->originalGeneration(), boundary++},
                    endpoints[0], endpoints[1], capacity),
                CapturedTransferRankPlan::fromInventory(*mpi->clusterInventory(), {session->originalGeneration(), boundary++},
                    endpoints[1], endpoints[0], capacity)});
        }
        plans.push_back(std::move(lanes));
    }
    std::shared_ptr<PhysicalMemoryAuthority> memory;
    IWorkerGPUContext *worker = nullptr;
    IBackend *backend = nullptr;
    void *stream = nullptr;
    const CPUTensorStorageDeclaration descriptor(1, 4, BufferTensorType::INT32,
        PhysicalMemoryOwner::ActivationTransportStaging, DeviceId::cpu());
    exchangePlanningSamples(mpi, [&](int count) {
        return std::vector<std::vector<std::uint8_t>>(count, plans.front().front()[0].serialize());
    }, [&](auto wire) {
        require(std::vector<std::uint8_t>(wire.begin(), wire.end()) == plans.front().front()[0].serialize(),
            "Counted rank trace changed original declarations");
        if (active)
        {
            const std::array devices{local};
            original = test::PreparedNativeLookupFixture::prepare(devices, mpi->rank());
            PhysicalMemoryPlanBuilder builder;
            std::vector<PhysicalMemoryPlan> alternatives;
            for (size_t index = 0; index < cases.size(); ++index)
            {
                for (const auto &lane : plans[index])
                    for (const auto &plan : lane)
                        plan.contributeMemory(*plan.localRole(mpi->rank()), original->resource(local),
                            plan.hostRank() == mpi->rank() ? std::optional{original->resource(DeviceId::cpu())} : std::nullopt,
                            builder);
                const auto shape = rankCountedStorage(cases[index], source);
                PhysicalMemoryPlanBuilder payload;
                payload.add(original->resource(local), PhysicalMemoryOwner::ExecutionWorkspace,
                    shape.workspace.total_bytes_with_alignment());
                payload.add(original->resource(local), PhysicalMemoryOwner::NativeGraphExecutable,
                    GPUGraphMemoryContract::reservationBytesPerExecutable(local,
                        GPUGraphExecutableClass::BoundedFlatHelper));
                payload.add(original->resource(DeviceId::cpu()), PhysicalMemoryOwner::ActivationTransportStaging,
                    TransferEngine::mappedHostRegionAllocationBytes(shape.bytes + shape.count_bytes));
                if (source && cases[index].sliced)
                {
                    payload.add(original->resource(local), PhysicalMemoryOwner::ActivationTransportStaging, sizeof(CapturedTransferSourceSlice));
                    payload.add(original->resource(DeviceId::cpu()), descriptor.owner(), descriptor.physicalBytes());
                }
                if (source && cases[index].tensor_source)
                {
                    const CPUTensorStorageDeclaration source_root(1, shape.tensor_bytes / sizeof(std::int32_t),
                        BufferTensorType::INT32, PhysicalMemoryOwner::ActivationTransportStaging, DeviceId::cpu());
                    payload.add(original->resource(local), PhysicalMemoryOwner::ActivationTransportStaging, shape.tensor_bytes);
                    payload.add(original->resource(DeviceId::cpu()), source_root.owner(), source_root.physicalBytes());
                }
                alternatives.push_back(payload.build());
            }
            builder.addMutuallyExclusive(alternatives);
            original->admit(builder.build());
            memory = original->memory();
            worker = original->permit(local).worker().get();
            backend = getBackendFor(local);
            require(worker && backend, "Counted rank trace omitted an original worker/backend");
            worker->submitAndWait([&] {
                stream = worker->getOrCreateAuxiliaryStream("rank_counted_trace");
#ifdef HAVE_CUDA
                if (local.is_cuda()) require(counted_channel_test::prepareKernelsCUDA(), "Prepare original CUDA count/slice kernels");
#endif
#ifdef HAVE_ROCM
                if (local.is_rocm()) require(counted_channel_test::prepareKernelsROCm(), "Prepare original ROCm count/slice kernels");
#endif
            });
        }
        return std::vector<std::uint8_t>{1};
    }, [](auto receipts) {
        for (const auto &receipt : receipts)
            require(receipt.bytes.size() == 1 && receipt.bytes[0] == 1, "Counted rank source preparation omitted a peer");
    });
    auto &transfer = TransferEngine::instance();
    std::vector<std::vector<std::array<std::shared_ptr<CapturedTransferChannel>, 2>>> channels;
    const auto abort_failure = [mpi] {
        std::cerr << "Original counted/sliced rank trace failed before native retirement\n";
        MPI_Abort(mpi->communicator(), 2);
        std::terminate();
    };
    NativeReplayFailureDrain channel_retirement{abort_failure};
    for (const auto &group : plans)
    {
        std::vector<std::array<std::shared_ptr<CapturedTransferChannel>, 2>> lanes;
        for (const auto &lane : group)
            lanes.push_back({transfer.createRankCapturedTransferChannel(lane[0], session, memory, stream),
                transfer.createRankCapturedTransferChannel(lane[1], session, memory, stream)});
        channels.push_back(std::move(lanes));
    }
    const auto submit = [&](auto operation) {
        try { worker->submitAndWait(std::move(operation)); }
        catch (const std::exception &error)
        {
            std::cerr << "Original counted rank " << mpi->rank() << " failed: " << error.what() << '\n';
            throw;
        }
    };
    for (size_t case_index = 0; case_index < cases.size(); ++case_index)
    {
        const auto &geometry = cases[case_index];
        const auto shape = rankCountedStorage(geometry, source);
        const auto lanes = geometry.capacities.size();
        std::shared_ptr<const WorkspaceBufferLease> payload, returned, counts;
        PhysicalMemoryAllocationLease snapshot_claim, descriptor_claim, source_claim;
        std::shared_ptr<TensorBase> slice_root, source_root;
        std::shared_ptr<MappedHostTransferRegion> snapshot;
        std::optional<Terminal> terminal;
        NativeReplayFailureDrain payload_retirement{abort_failure};
        if (active)
        {
            DeviceWorkspaceManager workspace(local, shape.workspace.total_bytes_with_alignment(), memory);
            require(workspace.allocate(shape.workspace), "Counted original workspace allocation failed");
            if (!source || !geometry.tensor_source) payload = workspace.retainBuffer("payload", 0, shape.bytes);
            if (source) returned = workspace.retainBuffer("returned", 0, shape.bytes);
            counts = workspace.retainBuffer("counts", 0, shape.count_bytes);
            workspace.release();
            if (source && geometry.sliced)
            {
                descriptor_claim = memory->claimNewAllocation(local, PhysicalMemoryOwner::ActivationTransportStaging,
                    sizeof(CapturedTransferSourceSlice));
                slice_root = descriptor.materialize(memory);
                submit([&] { TransferEngine::prepareDeviceOutput(slice_root.get(), local, stream); });
            }
            if (source && geometry.tensor_source)
            {
                const CPUTensorStorageDeclaration declaration(1, shape.tensor_bytes / sizeof(std::int32_t),
                    BufferTensorType::INT32, PhysicalMemoryOwner::ActivationTransportStaging, DeviceId::cpu());
                source_claim = memory->claimNewAllocation(local, PhysicalMemoryOwner::ActivationTransportStaging,
                    shape.tensor_bytes);
                source_root = declaration.materialize(memory);
                require(source_root.get() != slice_root.get(), "Counted source reused its descriptor Tensor");
                submit([&] { TransferEngine::prepareDeviceOutput(source_root.get(), local, stream); });
            }
            snapshot_claim = memory->claimNewAllocation(DeviceId::cpu(), PhysicalMemoryOwner::ActivationTransportStaging,
                TransferEngine::mappedHostRegionAllocationBytes(shape.bytes + shape.count_bytes));
            const std::array devices{local};
            snapshot = transfer.allocateMappedHostRegion(shape.bytes + shape.count_bytes, devices);
            terminal.emplace(backend, local, backend->createEvent(local.ordinal));
            require(terminal->event != nullptr, "Counted original terminal event is absent");
        }
        std::uint64_t invocation_index = 0;
        for (const size_t offset : {size_t{1}, size_t{16}})
        {
            std::vector<CapturedTransferBinding> first_phase, second_phase;
            PhysicalMemoryOwnerReservation graph_claim;
            std::unique_ptr<IGPUGraphCapture> graph;
            NativeReplayFailureDrain graph_retirement{abort_failure};
            if (active)
            {
                for (size_t lane = 0; lane < lanes; ++lane)
                {
                    const auto capacity = geometry.capacities[lane];
                    const auto start = shape.origins[lane] + offset;
                    const auto count_base = lane * 3 * sizeof(std::uint64_t);
                    auto outward = source
                        ? source_root
                            ? transfer.bindCapturedTransfer(channels[case_index][lane][0], CapturedTransferEndpoint::Producer,
                                {101 + lane, capacity}, source_root, start)
                            : transfer.bindCapturedTransfer(channels[case_index][lane][0], CapturedTransferEndpoint::Producer,
                                {101 + lane, capacity}, payload, start)
                        : transfer.bindCapturedTransfer(channels[case_index][lane][0], CapturedTransferEndpoint::Consumer,
                            {101 + lane, capacity}, payload, start);
                    if (source && geometry.sliced)
                        outward = transfer.bindDeviceSlicedTransfer(std::move(outward), slice_root, 0, capacity + 64,
                            geometry.tensor_source ? 4000 : 0);
                    else
                        outward = transfer.bindDeviceCountedTransfer(std::move(outward), counts, count_base);
                    auto inward = transfer.bindDeviceCountedTransfer(source
                        ? transfer.bindCapturedTransfer(channels[case_index][lane][1], CapturedTransferEndpoint::Consumer,
                            {201 + lane, capacity}, returned, start)
                        : transfer.bindCapturedTransfer(channels[case_index][lane][1], CapturedTransferEndpoint::Producer,
                            {201 + lane, capacity}, payload, start), counts,
                        count_base + (source ? sizeof(std::uint64_t) : 0));
                    first_phase.push_back(std::move(outward));
                    second_phase.push_back(std::move(inward));
                }
                // Offset is graph identity. Both alignments preserve their
                // original device sequence and all 128 live-input transitions.
                submit([&] {
                    require(backend->memset(counts->data(), 0, shape.count_bytes, local.ordinal, stream),
                        "Counted original sequence reset failed");
                    if (source)
                        for (size_t lane = 0; lane < lanes; ++lane)
                            for (size_t prior = 0; prior < lane; ++prior)
                                rankAdvanceCounted(local,
                                    static_cast<std::uint64_t *>(counts->data((lane * 3 + 2) * sizeof(std::uint64_t))), stream);
                    require(backend->recordEvent(terminal->event, local.ordinal, stream), "Counted original reset event failed");
                });
                terminal->await(std::chrono::seconds(30));
                graph_claim = memory->reserveNewAllocations(local, PhysicalMemoryOwner::NativeGraphExecutable,
                    GPUGraphMemoryContract::reservationBytesPerExecutable(local, GPUGraphExecutableClass::BoundedFlatHelper));
                graph = worker->createGraphCapture(stream);
                require(bool(graph), "Counted original capture factory is absent");
                submit([&] {
                    std::array<int, 2> stage_identity{};
                    std::optional<GraphCaptureDependencyLedger> ledger;
                    if (source && geometry.sliced)
                    {
                        std::vector<const TensorBase *> outputs{slice_root.get()};
                        std::vector<GraphCaptureDependencyLedger::InternalInput> inputs{{slice_root.get(), 0}};
                        if (source_root) { outputs.push_back(source_root.get()); inputs.push_back({source_root.get(), 0}); }
                        ledger.emplace(local, stream, std::vector<GraphCaptureDependencyLedger::StagePlan>{
                            {.stage_identity = &stage_identity[0], .stage_name = "original rank slice writer", .outputs = std::move(outputs)},
                            {.stage_identity = &stage_identity[1], .stage_name = "original directed slice transfer",
                                .internal_inputs = std::move(inputs)}}, "original rank source slice");
                    }
                    ScopedBackendGraphCapture capture(*worker, *graph, "original rank counted/sliced packet", ledger ? &*ledger : nullptr);
                    require(capture.begin(), "Counted original capture begin failed");
                    require(backend->memset(source ? returned->data() : payload->data(), 0xa5, shape.bytes, local.ordinal, stream),
                        "Counted original guard reset failed");
                    if (source)
                    {
                        for (size_t lane = 0; lane < lanes; ++lane)
                        {
                            auto *sequence = static_cast<std::uint64_t *>(counts->data((lane * 3 + 2) * sizeof(std::uint64_t)));
                            void *address = source_root ? static_cast<std::byte *>(source_root->gpu_data_ptr()) + offset
                                : payload->data(shape.origins[lane] + offset);
                            if (geometry.sliced)
                            {
                                ScopedGraphCaptureStage writer(&stage_identity[0]);
                                rankPrepareSlice(local, address, static_cast<CapturedTransferSourceSlice *>(slice_root->gpu_data_ptr()),
                                    sequence, geometry.capacities[lane], geometry.capacities[lane] + 64,
                                    geometry.tensor_source ? 4000 : 0, stream);
                                TransferEngine::publishDeviceWrite(slice_root.get(), local, stream);
                                if (source_root) TransferEngine::publishDeviceWrite(source_root.get(), local, stream);
                                writer.complete();
                            }
                            else
                                rankPrepareCounted(local, address,
                                    static_cast<std::uint64_t *>(counts->data(lane * 3 * sizeof(std::uint64_t))),
                                    sequence, geometry.capacities[lane], stream);
                        }
                    }
                    ScopedGraphCaptureStage transfer_stage(&stage_identity[1]);
                    if (geometry.batched)
                    {
                        transfer.enqueueCapturedTransferBatch(first_phase, stream);
                        transfer.enqueueCapturedTransferBatch(second_phase, stream);
                    }
                    else
                    {
                        require(lanes == 1, "Scalar counted trace unexpectedly contains a packet");
                        transfer.enqueueCapturedTransfer(first_phase.front(), stream);
                        transfer.enqueueCapturedTransfer(second_phase.front(), stream);
                    }
                    if (source)
                        for (size_t lane = 0; lane < lanes; ++lane)
                            rankAdvanceCounted(local,
                                static_cast<std::uint64_t *>(counts->data((lane * 3 + 2) * sizeof(std::uint64_t))), stream);
                    transfer_stage.complete();
                    capture.finish();
                    if (!geometry.batched)
                        counted_channel_test::verifyCapturedChannelGraph(*graph, geometry.capacities.front(), 2);
                    else if (geometry.packets.empty())
                        counted_channel_test::verifyCapturedChannelBatchGraph(*graph, geometry.capacities.front(), lanes, 2);
                    else
                        counted_channel_test::verifyCapturedChannelBatchGraph(*graph, geometry.packets, 2, local.is_cuda() ? 32 : 64);
                    std::string error;
                    require(graph->validateExecutableMemoryClass(GPUGraphExecutableClass::BoundedFlatHelper, &error), error.c_str());
                    require(graph->prepareRuntimeContextStorage(memory) && graph->instantiate(),
                        "Counted original graph admission/instantiation failed");
                });
            }
            // Epochs belong to the channel, while reset restarts only the GPU
            // input sequence. Each offset is a fresh original invocation series.
            const size_t replays = geometry.batched && !geometry.sliced ? 16 : 128;
            for (size_t replay = 0; replay < replays; ++replay)
            {
                const CapturedTransferInvocation invocation{session->originalGeneration(), ++invocation_index};
                if (active)
                {
                    if (source && geometry.sliced && replay == 64)
                    {
                        submit([&] {
                            require(backend->memset(counts->data(2 * sizeof(std::uint64_t)), 0,
                                sizeof(std::uint64_t), local.ordinal, stream), "Original source request reset failed");
                            require(backend->recordEvent(terminal->event, local.ordinal, stream), "Original source request reset event failed");
                        });
                        terminal->await(std::chrono::seconds(30));
                    }
                    for (size_t lane = 0; lane < lanes; ++lane)
                        for (size_t direction = 0; direction < 2; ++direction)
                        {
                            const auto &channel = channels[case_index][lane][direction];
                            transfer.beginCapturedTransferInvocation(channel, invocation);
                            // Only this actual shared backing's declared host
                            // owner restores diagnostic poison before its own
                            // producer can launch. No remote host writes it.
                            if (plans[case_index][lane][direction].hostRank() == mpi->rank())
                            {
                                auto wire = CapturedTransferChannelReplayProbe::payload(*channel);
                                std::fill(wire.begin(), wire.end(), std::byte{0xa5});
                            }
                        }
                    if ((replay % 2 == 0 ? !source : source) && replay % 7 == 0)
                        std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    submit([&] {
                        require(graph->launch(), "Counted original retained graph launch failed");
                        transfer.enqueuePersistentDeviceRegionToMappedHost(source ? returned->data() : payload->data(),
                            shape.bytes, 0, *snapshot, 0, shape.bytes, local, stream);
                        transfer.enqueuePersistentDeviceRegionToMappedHost(counts->data(), shape.count_bytes, 0,
                            *snapshot, shape.bytes, shape.count_bytes, local, stream);
                        require(backend->recordEvent(terminal->event, local.ordinal, stream), "Counted original terminal publication failed");
                    });
                    terminal->await(std::chrono::seconds(30));
                }
                mpi->barrier();
                if (active)
                {
                    const auto *actual = static_cast<const std::uint8_t *>(snapshot->mutableHostData());
                    for (size_t lane = 0; lane < lanes; ++lane)
                    {
                        const auto sequence = (geometry.sliced ? replay % 64 : replay) + lane;
                        const auto capacity = geometry.capacities[lane];
                        const auto live = counted_channel_test::extent(sequence, capacity);
                        const auto source_offset = geometry.sliced ? counted_channel_test::sliceOffset(sequence, capacity, capacity + 64) : 0;
                        const auto count_offset = shape.bytes + (lane * 3 + (source ? 1 : 0)) * sizeof(std::uint64_t);
                        std::uint64_t observed_count = 0;
                        std::memcpy(&observed_count, actual + count_offset, sizeof(observed_count));
                        require(observed_count == live, "Counted original terminal extent differs from device trace");
                        if (source)
                        {
                            std::uint64_t authored_count = 0, next_sequence = 0;
                            std::memcpy(&authored_count, actual + shape.bytes + lane * 3 * sizeof(std::uint64_t),
                                sizeof(authored_count));
                            std::memcpy(&next_sequence, actual + shape.bytes + (lane * 3 + 2) * sizeof(std::uint64_t),
                                sizeof(next_sequence));
                            if (!geometry.sliced)
                                require(authored_count == live, "Counted source changed its authored useful extent");
                            require(next_sequence == sequence + 1, "Counted source advanced its original sequence incorrectly");
                        }
                        const auto stride = capacity + (geometry.sliced ? 128 : 64);
                        for (size_t byte = 0; byte < stride; ++byte)
                        {
                            const auto expected = byte >= offset && byte - offset < live
                                ? counted_channel_test::byte(sequence, source_offset + byte - offset) : std::uint8_t{0xa5};
                            require(actual[shape.origins[lane] + byte] == expected,
                                "Counted original source origin, payload or independent endpoint guard differs");
                        }
                        for (const auto &channel : channels[case_index][lane])
                        {
                            const auto &control = CapturedTransferChannelReplayProbe::terminalControl(*channel);
                            require(control.producer.epoch == invocation_index && control.consumer.epoch == invocation_index &&
                                control.producer.message.bytes == live && control.consumer.message.bytes == live,
                                "Counted original metadata extent/epoch differs from its device-owned trace");
                            const auto wire = CapturedTransferChannelReplayProbe::payload(*channel);
                            for (size_t byte = 0; byte < wire.size(); ++byte)
                                require(wire[byte] == (byte < live
                                    ? static_cast<std::byte>(counted_channel_test::byte(sequence, source_offset + byte)) : std::byte{0xa5}),
                                    "Counted original channel copied inactive capacity");
                            require(transfer.finishCapturedTransferInvocation(channel, invocation) == CapturedTransferInvocationRetirement::Completed,
                                "Counted original invocation lost exact retirement");
                        }
                    }
                }
                // All peers finish observing shared controls/wire poison
                // before the next admitted request can publish another epoch.
                mpi->barrier();
            }
            if (active) submit([&] { graph.reset(); });
            graph_retirement.completed = true;
        }
        payload_retirement.completed = true;
    }
    mpi->barrier();
    channels.clear();
    session->retireAfterOriginalGraphRelease();
    if (original)
    {
        original->requirePayloadRetired();
        original->retire();
        original->requireExactRetired();
    }
    channel_retirement.completed = true;
}
} // namespace

#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
#define LLAMINAR_RANK_COUNTED_TESTS(first, second) \
    TEST(CapturedTransferRankCounted, first##_##second) \
    { rankCapturedCountedReplay(DeviceType::first, DeviceType::second, RankCountedTrace::Counted); } \
    TEST(CapturedTransferRankCounted, first##_##second##_SourceSlices) \
    { rankCapturedCountedReplay(DeviceType::first, DeviceType::second, RankCountedTrace::SourceSlices); } \
    TEST(CapturedTransferRankCounted, first##_##second##_FusedBoundaries) \
    { rankCapturedCountedReplay(DeviceType::first, DeviceType::second, RankCountedTrace::FusedBoundaries); } \
    TEST(CapturedTransferRankCounted, first##_##second##_BatchedEndpoints) \
    { rankCapturedCountedReplay(DeviceType::first, DeviceType::second, RankCountedTrace::BatchedEndpoints); } \
    TEST(CapturedTransferRankCounted, first##_##second##_Batch16KiBBoundaries) \
    { rankCapturedCountedReplay(DeviceType::first, DeviceType::second, RankCountedTrace::Batch16KiBBoundaries); }
LLAMINAR_RANK_COUNTED_TESTS(CUDA, ROCm)
LLAMINAR_RANK_COUNTED_TESTS(ROCm, CUDA)
#undef LLAMINAR_RANK_COUNTED_TESTS
#endif
