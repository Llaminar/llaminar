/**
 * @file Test__ImmutableAdaptiveLookupStage.cpp
 * @brief Captured participant-local adaptive stage and retained source-boundary
 * proof.
 *
 * This fixture uses the actual component, PMA, protocol2 worker, original
 * roots, canonical controller and DeviceGraphExecutor frontier. The synthetic
 * query belongs to this component fixture; it does not certify Rank/DGO request
 * or prefix/history provenance, planner reachability, or a public serving mode.
 */
#include "execution/compute_stages/stages/qwen4exp/ImmutableLookupStage.h"
#include "execution/local_execution/graph/ComputeGraph.h"
#include "execution/local_execution/graph/DeviceGraphExecutor.h"
#include "fixtures/qwen4exp/AdaptiveLookupNativeComponent.h"
#include "kernels/qwen4exp/QwenAdaptiveTerminalCounters.h"
#include <gtest/gtest.h>

namespace llaminar2::test::adaptive_native
{
namespace
{
/** @brief Retire captured native work before stage/root/source owners on all
 * exits. */
struct StageRetirement final
{
    Component &component;
    std::shared_ptr<qwen4exp::ImmutableLookupExecutionBoundary> boundary;
    std::unique_ptr<IGPUGraphCapture> graph, terminal, warm;

    /** @brief Fence only fixture terminal/exception paths while the real source
     * worker remains live. */
    ~StageRetirement() noexcept
    {
        try
        {
            component.join();
            graph.reset();
            terminal.reset();
            warm.reset();
            if (boundary->hasActiveInterval())
            {
                boundary->recordTerminalClose(component.stream);
                component.join();
                boundary->awaitTerminal();
            }
        }
        catch (...)
        {
            // An uncertain source response may not outlive its mapped backing.
            // close() itself can enqueue before an error check throws. Fence
            // that second frontier before cancelling its mapped source worker.
            if (!component.backend->synchronizeStream(component.stream, component.device.ordinal))
                std::terminate();
            component.service->cancel();
            try
            {
                if (boundary->hasActiveInterval())
                    boundary->awaitTerminal();
            }
            catch (...)
            {
            }
        }
    }
};

/**
 * @brief Publish diagnostic inputs without replacing the device-owned counter row.
 * @param c Actual fixture component. @param active Original live token rows.
 * @param ids Exact original active IDs, preserving duplicate order.
 * @note The canonical controller is reset once per request, never downloaded
 * and re-uploaded around a frame. Full poison extents are fixture diagnostics.
 */
void workloadWithoutController(Component &c, int active, std::span<const std::int64_t> ids)
{
    require(active >= 0 && active <= c.rows && ids.size() == static_cast<std::size_t>(active) * 16,
            "Adaptive stage diagnostic geometry mismatch");
    std::vector<std::int64_t> all(c.maximumIds, -1);
    std::copy(ids.begin(), ids.end(), all.begin());
    c.upload(c.ids(), std::span<const std::int64_t>(all));
    c.upload(c.live(), std::span<const int>(&active, 1));
    std::vector<float> output(c.maximumIds * c.source.columns, -12345.f);
    c.upload(c.output(), std::span<const float>(output));
    std::vector<std::byte> native(c.maximumIds * c.geometry->rowBytes(), std::byte{0xa5});
    c.upload(static_cast<std::byte *>(c.frame->gpu_data_ptr()) + c.geometry->frame().native_response,
             std::span<const std::byte>(native));
}

/** @brief Exercise one original shared stage/boundary across dirty retained
 * replay and reset.
 * @param device Exact native backend selected by its explicit integration lane.
 * @param coldCapture Record the real stage before any request admission, then
 * acquire actual admitted bytes before its first retained replay.
 * @param retentionProbe Check that only the actual terminal worker join permits
 * retention observations, and that closure/Preserve retain complete warmed metadata.
 */
void stageLifecycle(DeviceId device, bool coldCapture = false, bool retentionProbe = false)
{
    auto *backend = getBackendFor(device);
    if (!backend || backend->deviceCount() <= device.ordinal)
    {
        GTEST_SKIP() << "Requested backend unavailable";
    }
    if (!hasCPUBackend())
        initCPUBackend(-1);
    auto &worker = GPUDeviceContextPool::instance().getContext(device);
    worker.submitAndWait(
        [&]
        {
            for (const std::uint32_t slots : {0u, 64u})
                for (const int capacity : {1, 17})
                {
                    const auto native = iq4Source(97, 160);
                    Source source(native, 97, 160, GGUFTensorType::IQ4_NL);
                    // Both captured M1 and M17 use the same larger model frame.
                    // The physical query, never that allocation capacity, owns
                    // merge/scan launch bounds and the active output extent.
                    Component c(source, device, worker, 17, slots,
                                capacity == 1 ? AdaptiveFixtureGraphFamily::LookupTerminalAndWarm
                                              : AdaptiveFixtureGraphFamily::LookupAndTerminal);
                    ASSERT_TRUE(c.graphLease);
                    EXPECT_EQ(c.graphLease->bytes(), (capacity == 1 ? 3u : 2u) *
                                                       GPUGraphMemoryContract::reservationBytesPerExecutable(device));
                    // The shared primitive helper creates tiny diagnostic
                    // host-backed roots. Retire those physical allocations and
                    // only then their claims, before using the actual serving
                    // materializer. Its aliasing handles retain claims through
                    // the last root/stage borrow without a test adoption API.
                    c.join();
                    c.cache.reset();
                    c.frame.reset();
                    c.protocol.reset();
                    c.cacheLease.reset();
                    c.frameLease.reset();
                    c.hostRootLease.reset();
                    auto rootOwner = AdaptiveImmutableLookupRootStorage::create(c.geometry->nativePlan(),
                                                                                c.memory, c.stream);
                    const auto originalRoots = rootOwner->roots();
                    c.cache = std::dynamic_pointer_cast<INT8Tensor>(originalRoots.cache);
                    c.frame = std::dynamic_pointer_cast<INT8Tensor>(originalRoots.frame);
                    c.protocol = std::dynamic_pointer_cast<INT32Tensor>(originalRoots.protocol);
                    require(c.frame && c.protocol && (bool(c.cache) == bool(slots)),
                            "Adaptive serving root materializer returned foreign original types");
                    auto boundary = qwen4exp::ImmutableLookupExecutionBoundary::createAdaptive(
                        c.service, c.geometry->nativePlan(), rootOwner, c.stream);
                    c.join();
                    auto context = IDeviceContext::create(device, 1);
                    // Model setup binds these exact source-owned roots, never
                    // duplicate graph-arena allocations. Identical setup reuse
                    // is inert; conflicting declarations fail before mutation.
                    BufferArena foreignArena(ArenaConfig{.factory = &c.factory});
                    require(foreignArena.registerExternalBuffer(BufferId::QFN_NGRAM_ADAPTIVE_FRAME,
                                                                c.protocol.get()),
                            "Adaptive conflicting root fixture failed");
                    EXPECT_THROW(boundary->bindAdaptiveArena(foreignArena), std::invalid_argument);
                    EXPECT_FALSE(foreignArena.isRegistered(BufferId::QFN_NGRAM_ADAPTIVE_CACHE));
                    EXPECT_FALSE(foreignArena.isRegistered(BufferId::QFN_NGRAM_LOOKUP_STATE));
                    boundary->bindAdaptiveArena(c.arena);
                    boundary->bindAdaptiveArena(c.arena);
                    // Typed primitive bindings retain the actual original
                    // TensorBase owners, not BufferArena's ITensor projection.
                    const auto ids = c.arena.getSharedTensor(BufferId::QFN_NGRAM_ROW_IDS);
                    const auto output = c.arena.getSharedTensor(BufferId::QFN_PLE_EMBEDDING);
                    const auto liveOwner = c.arena.getSharedTensor(BufferId::PREFILL_CHUNK_GEOMETRY);
                    require(ids && output && liveOwner &&
                                c.arena.getTensor(BufferId::QFN_NGRAM_ROW_IDS) == ids.get() &&
                                c.arena.getTensor(BufferId::QFN_PLE_EMBEDDING) == output.get() &&
                                c.arena.getTensor(BufferId::PREFILL_CHUNK_GEOMETRY) == liveOwner.get(),
                            "Adaptive stage fixture lost its original arena tensor owners");
                    ImmutableLookupStage::AdaptiveGPUStorage binding{
                        boundary,
                        *c.control,
                        c.cache ? std::optional{Qwen4ExpArenaBinding<TensorType::INT8>(
                                      *c.cache, BufferId::QFN_NGRAM_ADAPTIVE_CACHE)}
                                : std::nullopt,
                        {*c.frame, BufferId::QFN_NGRAM_ADAPTIVE_FRAME},
                        {*c.protocol, BufferId::QFN_NGRAM_LOOKUP_STATE},
                        DeviceRowRange::deviceCounted(capacity, c.live()),
                        Qwen4ExpIntBinding(*liveOwner, BufferId::PREFILL_CHUNK_GEOMETRY)};
                    const ImmutableLookupStage::Params params{
                        .device_id = device,
                        .rows = {capacity, Qwen4ExpNativeIntInput{device, c.live(), 1}},
                        .ids = {*ids, BufferId::QFN_NGRAM_ROW_IDS},
                        .output = {*output, BufferId::QFN_PLE_EMBEDDING},
                        .format = TensorType::IQ4_NL,
                        .rowElements = 160,
                        .rowBytes = 90,
                        .storage = binding};
                    EXPECT_EQ(binding.count->tensor, liveOwner.get());
                    EXPECT_EQ(params.ids.tensor, ids.get());
                    EXPECT_EQ(params.output.tensor, output.get());
                    auto wrong = params;
                    auto wrongBinding = binding;
                    wrongBinding.originalRows =
                        DeviceRowRange::deviceCounted(capacity + 1, c.live()).slice(0, capacity);
                    wrong.storage = wrongBinding;
                    EXPECT_THROW(ImmutableLookupStage rejected(wrong), std::invalid_argument);
                    wrongBinding = binding;
                    wrongBinding.frame.buffer = BufferId::MTP_GENERATION_CONTROL;
                    wrong.storage = wrongBinding;
                    EXPECT_THROW(ImmutableLookupStage rejected(wrong), std::invalid_argument);
                    wrongBinding = binding;
                    wrongBinding.state.byteOffset = sizeof(std::int32_t);
                    wrong.storage = wrongBinding;
                    EXPECT_THROW(ImmutableLookupStage rejected(wrong), std::invalid_argument);
                    wrongBinding = binding;
                    wrongBinding.count.reset();
                    wrong.storage = wrongBinding;
                    EXPECT_THROW(ImmutableLookupStage rejected(wrong), std::invalid_argument);
                    wrongBinding = binding;
                    wrongBinding.originalRows =
                        DeviceRowRange::deviceCounted(9, c.live()).repeatEachRow(2).slice(0, capacity);
                    wrong.storage = wrongBinding;
                    EXPECT_THROW(ImmutableLookupStage rejected(wrong), std::invalid_argument);
                    auto stageOwner = std::make_unique<ImmutableLookupStage>(params);
                    auto *stage = stageOwner.get();
                    const auto declarationIdentity = stage->graphCaptureVariantSignature();
                    ComputeGraph graph;
                    graph.addNode("adaptive_native_lookup", std::move(stageOwner), device);
                    DeviceGraphExecutor executor;
                    executor.setArena(&c.arena);
                    StageRetirement retirement{c, boundary, {}, {}, {}};
                    const auto retainedSource = boundary->retainedSourceOwnerObservation();
                    const auto retainedGPU = boundary->rootStorage()->retainedOwnerObservation();
                    const auto retainedChannel = c.sourceBinding->captureIdentity();
                    std::vector<std::byte> retainedCache;
                    std::uint64_t readyGeneration = 0;
                    auto before = c.service->stats();
                    std::vector<int> freshController(sampling_math::kDeviceGenerationControlCount);
                    freshController[sampling_math::kDeviceGenerationControlOk] = 1;
                    c.upload(c.control->row(), std::span<const int>(freshController));
                    if (!coldCapture)
                        boundary->beginAdmission(1, c.stream);
                    else
                    {
                        EXPECT_FALSE(boundary->hasActiveInterval());
                        EXPECT_EQ(c.service->stats().requests, 0u);
                        EXPECT_THROW(TransferEngine::requireDeviceInput(c.protocol.get(), device, c.stream),
                                     std::runtime_error);
                        EXPECT_THROW(boundary->adaptiveKernels().gather(*boundary->adaptiveStorage(),
                                     *c.query, *c.control, c.stream), std::logic_error);
                    }
                    require(stage->prepareGraphLaunch(context.get(), c.stream),
                            "Adaptive stage preparation failed");
                    const auto authority = coldCapture
                        ? GraphCaptureDependencyLedger::ExternalInputAuthority::BindDeclaredAddressesOnly
                        : GraphCaptureDependencyLedger::ExternalInputAuthority::RequireReadyBytes;
                    require(executor.prepareGraphStorageForCapture(graph, context.get(), c.stream,
                                                                   "adaptive_original_root_frontier", authority),
                            "Adaptive original root frontier failed");
                    std::unique_ptr<GraphCaptureDependencyLedger> dependency;
                    if (coldCapture)
                    {
                        const std::array<std::string, 1> order{"adaptive_native_lookup"};
                        dependency = executor.planGraphCaptureDependencies(
                            graph, order, device, c.stream, "adaptive_cold_original_root_stage", {}, authority);
                        require(bool(dependency), "Adaptive cold dependency plan missing");
                        // Genuine native capture without a matching setup-only
                        // protocol declaration may not borrow this permission.
                        // Every rejection capture is discarded uninstantiated;
                        // even a failed assertion cannot execute cold kernels.
                        const auto rejectRecording = [&](bool installLedger, bool originalProtocol,
                                                         auto inputAuthority, DeviceId ledgerDevice,
                                                         void *ledgerStream)
                        {
                            GraphCaptureDependencyLedger::StagePlan plan;
                            plan.stage_identity = stage;
                            plan.stage_name = "adaptive_cold_rejection";
                            plan.external_inputs = {originalProtocol
                                ? static_cast<const TensorBase *>(c.protocol.get())
                                : static_cast<const TensorBase *>(c.frame.get())};
                            GraphCaptureDependencyLedger ledger(ledgerDevice, ledgerStream, {std::move(plan)},
                                                                "adaptive_cold_rejection", inputAuthority);
                            auto rejected = worker.createGraphCapture(c.stream);
                            require(bool(rejected), "Adaptive rejection capture owner missing");
                            ScopedBackendGraphCapture recording(worker, *rejected,
                                "adaptive cold exact-ledger rejection", installLedger ? &ledger : nullptr);
                            require(recording.begin(), "Adaptive rejection capture failed");
                            {
                                ScopedGraphCaptureStage current(stage);
                                EXPECT_THROW(boundary->adaptiveKernels().gather(*boundary->adaptiveStorage(),
                                             *c.query, *c.control, c.stream), std::logic_error);
                            }
                            recording.finish();
                        };
                        rejectRecording(false, true, authority, device, c.stream);
                        rejectRecording(true, false, authority, device, c.stream);
                        rejectRecording(true, true,
                            GraphCaptureDependencyLedger::ExternalInputAuthority::RequireReadyBytes, device, c.stream);
                        // No native operation touches this different ordinal:
                        // the active typed transaction rejects it before enqueue.
                        auto otherDevice = device;
                        ++otherDevice.ordinal;
                        rejectRecording(true, true, authority, otherDevice, c.stream);
                        /** @brief Retain a genuine unused stream for the exact-stream rejection. */
                        struct UnusedStreamRetirement
                        {
                            IWorkerGPUContext &worker;
                            void *stream;
                            /** @brief No work is ever submitted to this wrong-ledger stream. */
                            ~UnusedStreamRetirement() { if (stream) worker.destroyStream(stream); }
                        } unused{worker, worker.createStream()};
                        require(unused.stream && unused.stream != c.stream,
                                "Adaptive foreign-ledger stream owner missing");
                        rejectRecording(true, true, authority, device, unused.stream);
                        EXPECT_FALSE(boundary->hasActiveInterval());
                        EXPECT_EQ(c.service->stats().requests, 0u);
                    }
                    EXPECT_EQ(stage->graphCaptureVariantSignature(), declarationIdentity);
                    const auto identity = boundary->captureIdentity();
                    retirement.graph = worker.createGraphCapture(c.stream);
                    retirement.terminal = worker.createGraphCapture(c.stream);
                    require(retirement.graph && retirement.terminal, "Adaptive stage capture owner missing");
                    {
                        ScopedBackendGraphCapture capture(worker, *retirement.graph,
                                                          "adaptive original stage transaction", dependency.get());
                        require(capture.begin(), "Adaptive stage capture failed");
                        require(executor.recordSetupGraphMaterialization(graph, context.get()),
                                "Adaptive stage recording failed");
                        capture.finish();
                    }
                    if (coldCapture)
                    {
                        EXPECT_FALSE(boundary->hasActiveInterval());
                        EXPECT_EQ(c.service->stats().requests, 0u);
                        EXPECT_EQ(stage->graphCaptureVariantSignature(), declarationIdentity);
                        // Recording published no global protocol bytes and
                        // activated no source worker. Transaction zero MUST
                        // still acquire actual admit's exact producer event.
                        EXPECT_FALSE(executor.prepareGraphStorageForCapture(
                            graph, context.get(), c.stream, "adaptive_cold_requires_real_admission"));
                        boundary->beginAdmission(1, c.stream);
                        require(executor.prepareGraphStorageForCapture(
                            graph, context.get(), c.stream, "adaptive_cold_admitted_replay"),
                            "Adaptive first replay lacks its real admission frontier");
                    }
                    {
                        ScopedBackendGraphCapture capture(worker, *retirement.terminal,
                                                          "adaptive stage interval terminal");
                        require(capture.begin(), "Adaptive terminal capture failed");
                        boundary->recordTerminalClose(
                            c.stream, c.control->healthy(),
                            c.control->row() + sampling_math::kDeviceGenerationControlRequestComplete);
                        capture.finish();
                    }
                    require(retirement.graph->instantiate() && retirement.terminal->instantiate(),
                            "Adaptive stage instantiation failed");
                    // Inspect the actual captured kernels: the M1 graph may
                    // not inherit M17 merge/scan launches from the shared root.
                    std::vector<GPUGraphKernelNodeInfo> nodes;
                    std::string nodeError;
                    require(retirement.graph->inspectKernelNodes(nodes, &nodeError), nodeError.c_str());
                    EXPECT_EQ(nodes.size(), capacity == 1 ? 22u : 27u);
                    unsigned decodeNodes = 0, mergeNodes = 0;
                    for (const auto &node : nodes)
                    {
                        EXPECT_TRUE(node.valid());
                        EXPECT_EQ(node.local_memory_bytes_per_thread, 0u);
                        // The inspector publishes the actual runtime symbol;
                        // CUDA/HIP may return its original mangled spelling.
                        // These own flat kernels have unique source identifiers
                        // within the authenticated adaptive lookup namespace.
                        EXPECT_TRUE(node.name_resolved);
                        EXPECT_NE(node.name.find("adaptive_lookup"), std::string::npos);
                        if (node.name.find("merge") != std::string::npos)
                            ++mergeNodes;
                        if (node.name.find("decode") != std::string::npos)
                        {
                            ++decodeNodes;
                            EXPECT_EQ(node.grid_x, (capacity * 16 * 160 + 255) / 256);
                        }
                    }
                    EXPECT_EQ(decodeNodes, 1u);
                    EXPECT_EQ(mergeNodes, capacity == 1 ? 0u : 1u);
                    // Execute the large original query first on these same
                    // roots, then retain the small graph through full→0→1→0.
                    // This is real captured component work, not a substituted
                    // query producer or a host reconstruction of GPU state.
                    if (capacity == 1)
                    {
                        retirement.warm = worker.createGraphCapture(c.stream);
                        require(bool(retirement.warm), "Adaptive shared-frame warm capture owner missing");
                        {
                            ScopedBackendGraphCapture capture(worker, *retirement.warm,
                                                              "adaptive shared frame large query");
                            require(capture.begin(), "Adaptive shared-frame warm capture failed");
                            boundary->adaptiveKernels().gather(*boundary->adaptiveStorage(), *c.query,
                                                               *c.control, c.stream);
                            capture.finish();
                        }
                        require(retirement.warm->instantiate(),
                                "Adaptive shared-frame warm instantiate failed");
                        workloadWithoutController(c, 17, std::vector<std::int64_t>(17 * 16, 7));
                        require(retirement.warm->launch(), "Adaptive shared-frame warm launch failed");
                        c.join();
                    }
                    for (int request = 0; request < 2; ++request)
                    {
                        if (request)
                        {
                            before = c.service->stats();
                            c.upload(c.control->row(), std::span<const int>(freshController));
                            boundary->beginAdmission(2, c.stream);
                        }
                        if (retentionProbe)
                        {
                            EXPECT_TRUE(boundary->hasActiveInterval());
                            // Even after a native frame finishes, this source
                            // is not reusable retention credit until its real
                            // terminal close and original worker join complete.
                            EXPECT_THROW(boundary->retainedSourceOwnerObservation(), std::logic_error);
                            if (slots && request)
                            {
                                const auto preserved = c.read(
                                    static_cast<const std::byte *>(c.cache->gpu_data_ptr()),
                                    c.geometry->cache().bytes);
                                EXPECT_EQ(preserved, retainedCache)
                                    << "Preserve admission changed warmed cache bytes or slot generations";
                            }
                        }
                        for (const int live : {capacity, 0, 1, 0, capacity})
                        {
                            std::vector<std::int64_t> selected(static_cast<std::size_t>(live) * 16, 7);
                            workloadWithoutController(c, live, selected);
                            require(retirement.graph->launch(), "Adaptive stage retained replay failed");
                            const auto actual = c.read(c.output(), c.maximumIds * 160);
                            std::vector<float> expected(actual.size(), -12345.f);
                            std::vector<std::byte> rows(selected.size() * 90);
                            for (std::size_t i = 0; i < selected.size(); ++i)
                                std::memcpy(rows.data() + i * 90, source.native.data() + 7 * 90, 90);
                            qwen4exp::decodeCPUImmutableLookupRows(TensorType::IQ4_NL, 160, 90, rows.data(),
                                                                   selected.size(), expected.data());
                            EXPECT_EQ(
                                std::memcmp(actual.data(), expected.data(), expected.size() * sizeof(float)),
                                0);
                            EXPECT_EQ(stage->graphCaptureVariantSignature(), declarationIdentity);
                            EXPECT_EQ(boundary->captureIdentity(), identity);
                        }
                        if (retentionProbe)
                        {
                            EXPECT_THROW(boundary->retainedSourceOwnerObservation(), std::logic_error);
                            const auto frame = c.frameState();
                            EXPECT_EQ(frame.phase, policy::Phase::Idle);
                            EXPECT_EQ(frame.status, policy::Status::Success);
                            EXPECT_EQ(frame.held_pins, 0u);
                            const auto protocol = c.protocolState();
                            EXPECT_EQ(protocol.issuedSequence, protocol.consumedSequence);
                            EXPECT_EQ(protocol.admissionGeneration, static_cast<std::uint64_t>(request + 1));
                            if (slots)
                            {
                                const auto *cache = static_cast<const std::byte *>(c.cache->gpu_data_ptr());
                                const auto directory = c.read(
                                    reinterpret_cast<const policy::Slot32 *>(cache + c.geometry->cache().slots),
                                    slots);
                                unsigned ready = 0;
                                for (std::size_t index = 0; index < directory.size(); ++index)
                                    if (policy::Operations<policy::Slot32>::state(directory[index]) ==
                                        policy::SlotState::Ready)
                                    {
                                        ++ready;
                                        const auto &slot = directory[index];
                                        EXPECT_EQ(slot.row, 7u);
                                        EXPECT_FALSE(policy::Operations<policy::Slot32>::pinned(slot));
                                        EXPECT_GT(slot.generation, 0u);
                                        if (request)
                                        {
                                            EXPECT_EQ(slot.generation, readyGeneration);
                                        }
                                        else
                                            readyGeneration = slot.generation;
                                        const auto native = c.read(
                                            cache + c.geometry->cache().payload + index * c.geometry->slotStride(),
                                            c.geometry->rowBytes());
                                        EXPECT_EQ(std::memcmp(native.data(), source.native.data() + 7 * 90, 90), 0);
                                    }
                                EXPECT_EQ(ready, 1u);
                                retainedCache = c.read(cache, c.geometry->cache().bytes);
                            }
                        }
                        EXPECT_THROW(
                            boundary->recordTerminalClose(
                                c.stream, c.control->error(),
                                c.control->row() + sampling_math::kDeviceGenerationControlRequestComplete),
                            std::invalid_argument);
                        // The captured HIP-ticket-compatible close must leave a healthy
                        // incomplete request active. Reading here is diagnostic only.
                        require(retirement.terminal->launch(), "Adaptive incomplete close replay failed");
                        const auto closed =
                            c.read(&c.sourceBinding->deviceView().base.acknowledgement->closedGeneration, 1);
                        EXPECT_EQ(closed.front(), static_cast<std::uint64_t>(request));
                        const int complete = 1;
                        c.upload(c.control->row() + sampling_math::kDeviceGenerationControlRequestComplete,
                                 std::span<const int>(&complete, 1));
                        require(retirement.terminal->launch(), "Adaptive stage terminal replay failed");
                        c.join();
                        boundary->awaitTerminal();
                        if (retentionProbe)
                        {
                            EXPECT_FALSE(boundary->hasActiveInterval());
                            const auto actualSource = boundary->retainedSourceOwnerObservation();
                            EXPECT_EQ(actualSource.authority().get(), retainedSource.authority().get());
                            EXPECT_EQ(actualSource.allocator(), retainedSource.allocator());
                            EXPECT_EQ(actualSource.hostPlacement(), retainedSource.hostPlacement());
                            EXPECT_EQ(actualSource.tableStorageBytes(), retainedSource.tableStorageBytes());
                            EXPECT_EQ(actualSource.lookupStagingBytes(), retainedSource.lookupStagingBytes());
                            const auto actualGPU = boundary->rootStorage()->retainedOwnerObservation();
                            EXPECT_EQ(actualGPU.authority().get(), retainedGPU.authority().get());
                            EXPECT_EQ(actualGPU.allocator(), retainedGPU.allocator());
                            EXPECT_EQ(actualGPU.tableStorageBytes(), retainedGPU.tableStorageBytes());
                            EXPECT_EQ(actualGPU.lookupStagingBytes(), retainedGPU.lookupStagingBytes());
                            EXPECT_EQ(c.sourceBinding->captureIdentity(), retainedChannel);
                            EXPECT_EQ(boundary->captureIdentity(), identity);
                            if (slots)
                            {
                                EXPECT_EQ(c.read(static_cast<const std::byte *>(c.cache->gpu_data_ptr()),
                                                 c.geometry->cache().bytes), retainedCache);
                            }
                        }
                        // The same request row accumulated every real frame;
                        // no frame reset or inferred generation delta is used.
                        const auto words =
                            c.read(c.control->row(), sampling_math::kDeviceGenerationControlCount);
                        qwen4exp::QwenAdaptiveCounterTotals gpu;
                        ASSERT_TRUE(qwen4exp::readQwenAdaptiveCounters(words.data(), words.size(), gpu));
                        const bool warmLarge = capacity == 1 && request == 0;
                        const std::uint64_t calls = 5 + warmLarge;
                        const std::uint64_t misses = slots ? (request == 0 ? 1 : 0) : (3 + warmLarge);
                        const std::uint64_t hits = slots ? (3 + warmLarge - misses) : 0;
                        const std::uint64_t liveIds = (2 * capacity + 1) * 16 + (warmLarge ? 17 * 16 : 0);
                        EXPECT_EQ(gpu.calls, calls);
                        EXPECT_EQ(gpu.completed_live_ids, liveIds);
                        EXPECT_EQ(gpu.hit_rows, hits);
                        EXPECT_EQ(gpu.miss_rows, misses);
                        EXPECT_EQ(gpu.source_rows, misses);
                        EXPECT_EQ(gpu.source_bytes, misses * 90);
                        EXPECT_EQ(gpu.promotions, slots && request == 0 ? 1u : 0u);
                        EXPECT_EQ(gpu.evictions, 0u);
                        EXPECT_EQ(gpu.failed_frames, 0u);
                        EXPECT_TRUE(qwen4exp::validQwenAdaptiveCounters(gpu, 90, c.maximumIds));
                        const auto observed = boundary->terminalObservation();
                        EXPECT_EQ(observed.transport, RetainedStorageTransport::AdaptiveSubordinate);
                        EXPECT_EQ(observed.counters.failures, 0u);
                        if (request && slots)
                        {
                            EXPECT_EQ(observed.counters.requests, 0u);
                            EXPECT_EQ(observed.counters.ids, 0u);
                            EXPECT_EQ(observed.counters.nativeBytes, 0u);
                            EXPECT_EQ(c.service->stats().requests, before.requests);
                        }
                        if (retentionProbe)
                        {
                            ASSERT_FALSE(::testing::Test::HasFailure())
                                << "Retire the failed source observation before any Preserve replay";
                        }
                    }
                    EXPECT_THROW(boundary->channel(), std::logic_error);
                    EXPECT_THROW(boundary->kernels(), std::logic_error);
                    require(::ftruncate(source.fd, static_cast<off_t>(source.native.size() - 1)) == 0,
                            "Adaptive source truncation fixture failed");
                    EXPECT_THROW(boundary->beginAdmission(3, c.stream), std::exception);
                    EXPECT_FALSE(boundary->hasActiveInterval());
                    EXPECT_EQ(stage->graphCaptureVariantSignature(), declarationIdentity);
                }
        });
}
/**
 * @brief Abandon successful asynchronous setup before any graph/frontier or worker interval.
 * @param device Exact explicit integration backend.
 * @note No fixture stream join occurs between initialize and boundary teardown.
 * The boundary's own setup event must retire metadata before its source/root
 * references can release; Component's later teardown is not that authority.
 */
void abandonSetup(DeviceId device)
{
    auto *backend = getBackendFor(device);
    if (!backend || backend->deviceCount() <= device.ordinal)
    {
        GTEST_SKIP() << "Requested backend unavailable";
    }
    if (!hasCPUBackend())
        initCPUBackend(-1);
    auto &worker = GPUDeviceContextPool::instance().getContext(device);
    worker.submitAndWait(
        [&]
        {
            for (const std::uint32_t slots : {0u, 64u})
                for (const bool admit : {false, true})
                    for (int iteration = 0; iteration < 20; ++iteration)
                    {
                        Source source(iq4Source(97, 160), 97, 160, GGUFTensorType::IQ4_NL);
                        Component c(source, device, worker, 17, slots, AdaptiveFixtureGraphFamily::None);
                        EXPECT_FALSE(c.graphLease);
                        c.join();
                        c.cache.reset();
                        c.frame.reset();
                        c.protocol.reset();
                        c.cacheLease.reset();
                        c.frameLease.reset();
                        c.hostRootLease.reset();
                        auto roots = AdaptiveImmutableLookupRootStorage::create(c.geometry->nativePlan(),
                                                                                c.memory, c.stream);
                        auto boundary = qwen4exp::ImmutableLookupExecutionBoundary::createAdaptive(
                            c.service, c.geometry->nativePlan(), roots, c.stream);
                        EXPECT_FALSE(boundary->hasActiveInterval());
                        EXPECT_EQ(c.service->stats().requests, 0u);
                        if (admit)
                        {
                            boundary->beginAdmission(1, c.stream);
                            EXPECT_TRUE(boundary->hasActiveInterval());
                        }
                        // No capture, pre-launch frontier or synthetic source
                        // gather may mask either early-teardown path. This does
                        // not inspect live worker counters after admission.
                        boundary.reset();
                        roots.reset();
                        if (admit)
                        {
                            // Its last source owner joins the abandoned worker.
                            c.sourceBinding.reset();
                            c.service.reset();
                            c.provider.reset();
                        }
                        else
                        {
                            EXPECT_EQ(c.service->stats().requests, 0u);
                        }
                    }
        });
}
/**
 * @brief Retire ordinary mapped protocol writes if admission is abandoned before a graph.
 * @param device Exact explicit integration backend.
 * @note This is the symmetric existing RAM/mmap lane. Every source/channel and
 * tiny host/device protocol byte has its actual canonical PMA owner. There is
 * no stream join between successful admission and boundary destruction.
 */
void abandonOrdinaryAdmission(DeviceId device)
{
    auto *backend = getBackendFor(device);
    if (!backend || backend->deviceCount() <= device.ordinal)
    {
        GTEST_SKIP() << "Requested backend unavailable";
    }
    if (!hasCPUBackend())
        initCPUBackend(-1);
    auto &worker = GPUDeviceContextPool::instance().getContext(device);
    worker.submitAndWait(
        [&]
        {
            const auto resource = [](DeviceId id)
            {
                return PhysicalMemoryResource{.world_rank = 0,
                                              .device = id,
                                              .total_bytes = std::size_t{1} << 28,
                                              .admission_available_bytes = std::size_t{1} << 28};
            };
            for (const auto mode : {ImmutableRowStorage::RAM, ImmutableRowStorage::MemoryMapped})
                for (int iteration = 0; iteration < 20; ++iteration)
                {
                    Source source(iq4Source(97, 160), 97, 160, GGUFTensorType::IQ4_NL);
                    const auto policy = ImmutableRowTableStoragePolicy::forStorage(mode, 4096, 2);
                    const auto layout = ImmutableLookupLayout::create(16, 90);
                    PhysicalMemoryPlanBuilder plan;
                    plan.add(resource(DeviceId::cpu()), PhysicalMemoryOwner::ImmutableTableStorage,
                             ImmutableRowTable::physicalBytes(source.descriptor(), policy));
                    plan.add(resource(DeviceId::cpu()), PhysicalMemoryOwner::ImmutableLookupStaging,
                             ImmutableLookupService::hostBytes(layout));
                    plan.add(resource(DeviceId::cpu()), PhysicalMemoryOwner::ActivationArena, 64);
                    plan.add(resource(device), PhysicalMemoryOwner::ActivationArena, 64);
                    auto memory = std::make_shared<PhysicalMemoryAuthority>(
                        std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()), 0);
                    auto hostClaim =
                        memory->claimNewAllocation(DeviceId::cpu(), PhysicalMemoryOwner::ActivationArena, 64);
                    auto deviceClaim =
                        memory->claimNewAllocation(device, PhysicalMemoryOwner::ActivationArena, 64);
                    auto protocol = std::make_shared<INT32Tensor>(std::vector<std::size_t>{16});
                    const auto stream = ExplicitGPUStream(worker.defaultStream()).get();
                    TransferEngine::prepareDeviceInput(protocol.get(), device, stream);
                    TransferEngine::requireDeviceInput(protocol.get(), device, stream);
                    auto table = HostImmutableRowProvider::fromWholeTable(std::make_unique<ImmutableRowTable>(
                        source.descriptor(), policy, DeviceId::cpu(), memory,
                        ImmutableRowTableHostPlacement::taskPolicy()));
                    auto boundary = qwen4exp::ImmutableLookupExecutionBoundary::create(table, device,
                                                                                       protocol, 1, memory);
                    // Null/default admission fails before any producer changes.
                    EXPECT_THROW(boundary->beginAdmission(1, nullptr), std::logic_error);
                    EXPECT_FALSE(boundary->hasActiveInterval());
                    boundary->beginAdmission(1, stream);
                    EXPECT_TRUE(boundary->hasActiveInterval());
                    boundary.reset();
                    protocol.reset();
                    table.reset();
                }
        });
}
} // namespace
TEST(ImmutableAdaptiveLookupStage, CUDAOrdinaryAdmissionTeardownBeforeFirstGraph)
{
    abandonOrdinaryAdmission(DeviceId::cuda(0));
}
TEST(ImmutableAdaptiveLookupStage, ROCmOrdinaryAdmissionTeardownBeforeFirstGraph)
{
    abandonOrdinaryAdmission(DeviceId::rocm(0));
}
TEST(ImmutableAdaptiveLookupStage, CUDASetupTeardownBeforeFirstGraph)
{
    abandonSetup(DeviceId::cuda(0));
}
TEST(ImmutableAdaptiveLookupStage, ROCmSetupTeardownBeforeFirstGraph)
{
    abandonSetup(DeviceId::rocm(0));
}
TEST(ImmutableAdaptiveLookupStage, CUDARetainedOriginalRoots)
{
    stageLifecycle(DeviceId::cuda(0));
}
TEST(ImmutableAdaptiveLookupStage, ROCmRetainedOriginalRoots)
{
    stageLifecycle(DeviceId::rocm(0));
}

#ifdef HAVE_CUDA
/** @test Actual CUDA retained-source credit requires the original terminal join and preserves warmed rows. */
TEST(ImmutableAdaptiveLookupStage, CUDARetainedSourceAfterTerminal)
{
    stageLifecycle(DeviceId::cuda(0), false, true);
}
#endif
#ifdef HAVE_ROCM
/** @test Actual HIP retained-source credit requires the original terminal join and preserves warmed rows. */
TEST(ImmutableAdaptiveLookupStage, ROCmRetainedSourceAfterTerminal)
{
    stageLifecycle(DeviceId::rocm(0), false, true);
}
#endif

#ifdef HAVE_CUDA
TEST(ImmutableAdaptiveLookupStage, CUDAColdCaptureBeforeAdmission)
{
    stageLifecycle(DeviceId::cuda(0), true);
}
#endif
#ifdef HAVE_ROCM
TEST(ImmutableAdaptiveLookupStage, ROCmColdCaptureBeforeAdmission)
{
    stageLifecycle(DeviceId::rocm(0), true);
}
#endif
} // namespace llaminar2::test::adaptive_native
