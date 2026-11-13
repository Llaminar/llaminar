/**
 * @file NativePipelineByteReplay.inc
 * @brief Actual captured native pipeline edges, scalar geometry, publication and exact byte receipts.
 *
 * The original NUMA-owned cohort/PMA is retained throughout 20 empty, partial,
 * large-to-small and reset replays. No model, fake rank or host-selected payload
 * count is introduced: the recorded original INT32 row tensor authors extents.
 * Recording and payload observation run on the original workers. Whole-request
 * terminal aggregation and retained-definition retirement belong to the host
 * fixture owner, which joins recording jobs before releasing any definition.
 * Shared-child parents record their marker and evidence copies under a local
 * dependency ledger. A storage-only receiver publishes rows from actual native
 * completion, independent of its fixture's poisoned local count. Only the
 * launched parent publishes completion events.
 */
#include "execution/local_execution/orchestrators/PipelineNativeByteChannel.h"
#include "collective/OriginalNativeGraphCaptureGroup.h"
#include "tensors/TensorSlice.h"
namespace llaminar2::test {
/** @brief Passive original-frame access; every copy below follows the graph's actual terminal stream. */
struct PipelineNativeByteChannelReplayProbe final {
    static std::pair<DeviceId, const void *> frame(const PipelineNativeByteChannel &owner, int consumer, int participant)
    { return owner.terminalObservation(consumer, participant); }
};
}
namespace {
/** @brief Actual FP32 storage with passive event/frontier observations, never a publication override. */
class NativePipelinePublicationTensor final : public FP32Tensor {
public:
    /** @brief Adopt the exact PMA-priced original CPU tensor root. */
    NativePipelinePublicationTensor(std::size_t rows, std::size_t columns,
        const std::shared_ptr<PhysicalMemoryAuthority> &memory)
        : FP32Tensor({rows, columns}, CPUStorageRoot<float>(rows * columns,
            PhysicalMemoryOwner::ExecutionWorkspace, DeviceId::cpu()).allocate(memory)) {}
    /** @return Actual device completion event protecting this tensor's current bytes. */
    void *publicationEvent() const {
        std::lock_guard lock(coherence_mutex_);
        return completionEventProtectsDeviceValue_() ? device_completion_event_ : nullptr;
    }
    /** @return Whether actual publication records this device and has no imported consumer frontier yet. */
    bool freshPublicationOn(DeviceId device) const {
        std::lock_guard lock(coherence_mutex_);
        return completionEventProtectsDeviceValue_() && device_completion_event_ &&
            event_device_ == std::optional{device} && authoritative_device_ == std::optional{device} &&
            !last_joined_completion_event_ && !last_joined_consumer_stream_;
    }
    /** @return Whether public input acquisition imported that exact original event on this consumer stream. */
    bool acquiredOn(void *event, void *stream) const {
        std::lock_guard lock(coherence_mutex_);
        return event && last_joined_completion_event_ == event && last_joined_consumer_stream_ == stream;
    }
};
/** @brief Retain one actual tensor identity while replacing its mapped physical root during a cold negative control. */
class NativePipelineMappedCount final : public INT32Tensor {
public:
    /** @brief Allocate the exact original CPU tensor storage. */
    explicit NativePipelineMappedCount(const std::shared_ptr<PhysicalMemoryAuthority> &memory)
        : INT32Tensor({1}, CPUStorageRoot<std::int32_t>(1, PhysicalMemoryOwner::ExecutionWorkspace, DeviceId::cpu()).allocate(memory)) {}
    /** @brief Replace this tensor's actual mapped allocation on its original worker. */
    bool remap(DeviceId device)
    {
        freeMappedMemory();
        return initMappedMemory(sizeof(std::int32_t), device);
    }
};
/** @brief Select the exact recorded definition family exercised by one original process. */
enum class NativePipelineByteDefinition { Direct, SharedChild, ProducerRows };
/** @brief Replay the production native binding on an exact original native cohort. */
void nativePipelineByteReplay(DeviceType family, int degree,
    NativePipelineByteDefinition mode = NativePipelineByteDefinition::Direct)
{
    using Graph = test::RetainedParticipantGraphExecutableFixtureAccess;
    const bool shared = mode == NativePipelineByteDefinition::SharedChild;
    const bool producer_rows = mode == NativePipelineByteDefinition::ProducerRows;
    originalRankLocalReplay(family, degree, [&](const auto &devices) {
        const int source = 1; // Genuine nonzero, non-last source when the actual cohort permits it.
        constexpr std::size_t row_bytes = 7 * sizeof(float), capacity_rows = 31;
        constexpr std::size_t capacity = capacity_rows * row_bytes, storage_bytes = capacity + 2 * row_bytes;
        constexpr std::size_t rows_bytes = 4 * sizeof(std::int32_t);
        const NativeCountedAllGatherLayout frame_layout(1);
        const auto frame_bytes = frame_layout.bytes();
        const auto evidence_offset = storage_bytes + rows_bytes + (devices.size() - 1) * frame_bytes;
        const auto snapshot_stride = evidence_offset + (shared ? sizeof(std::int32_t) : 0);
        const auto snapshot_bytes = devices.size() * snapshot_stride;
        const auto owner_rank = MPIContextFactory::global()->rank();
        auto original = test::PreparedNativeLookupFixture::prepare(devices, owner_rank);
        std::vector<GlobalDeviceAddress> addresses;
        std::vector<int> consumers;
        for (std::size_t i = 0; i < devices.size(); ++i) {
            addresses.push_back(GlobalDeviceAddress::fromLocalDeviceId(devices[i]));
            if (static_cast<int>(i) != source) consumers.push_back(static_cast<int>(i));
        }
        auto collective = std::make_unique<LocalTPContext>(addresses, std::vector<float>{},
            family == DeviceType::CUDA ? CollectiveBackendType::NCCL : CollectiveBackendType::RCCL);
        require(bool(collective), "Native pipeline replay has no original native context");
        const auto native_backend = family == DeviceType::CUDA ? CollectiveBackendType::NCCL : CollectiveBackendType::RCCL;
        const auto native_boundary_bytes = CollectiveMemoryEstimator::nativePipelineBoundaryBytes(native_backend);
        const CPUTensorStorageDeclaration payload_declaration((capacity_rows + 2), 7, BufferTensorType::FP32,
            PhysicalMemoryOwner::ExecutionWorkspace, DeviceId::cpu());
        const CPUTensorStorageDeclaration rows_declaration(1, 4, BufferTensorType::INT32,
            PhysicalMemoryOwner::ExecutionWorkspace, DeviceId::cpu());
        const CPUTensorStorageDeclaration evidence_declaration(1, 1, BufferTensorType::INT32,
            PhysicalMemoryOwner::ExecutionWorkspace, DeviceId::cpu());
        PhysicalMemoryPlanBuilder plan;
        plan.add(original->resource(DeviceId::cpu()), PhysicalMemoryOwner::ExecutionWorkspace,
            devices.size() * (payload_declaration.physicalBytes() + rows_declaration.physicalBytes()) +
            (shared ? (devices.size() + 1) * evidence_declaration.physicalBytes() : 0));
        plan.add(original->resource(DeviceId::cpu()), PhysicalMemoryOwner::ActivationTransportStaging,
            TransferEngine::mappedHostRegionAllocationBytes(snapshot_bytes));
        if (producer_rows) {
            plan.add(original->resource(DeviceId::cpu()), PhysicalMemoryOwner::ExecutionWorkspace,
                CPUStorageRoot<std::int32_t>(1, PhysicalMemoryOwner::ExecutionWorkspace, DeviceId::cpu()).physicalBytes());
            plan.add(original->resource(DeviceId::cpu()), PhysicalMemoryOwner::ActivationTransportStaging,
                2 * TransferEngine::mappedHostRegionAllocationBytes(sizeof(std::int32_t)));
        }
        for (std::size_t i = 0; i < devices.size(); ++i) {
            const std::size_t parents = static_cast<int>(i) == source ? 2 : 1;
            plan.add(original->resource(devices[i]), PhysicalMemoryOwner::ExecutionWorkspace,
                storage_bytes + rows_bytes + (shared ? parents * sizeof(std::int32_t) : 0));
            plan.add(original->resource(devices[i]), PhysicalMemoryOwner::ActivationTransportStaging,
                frame_bytes * (static_cast<int>(i) == source ? consumers.size() : 1));
            plan.add(original->resource(devices[i]), PhysicalMemoryOwner::LocalCollective, native_boundary_bytes);
            // One original graph-only recording and precisely the replay parents.
            // The shared child never owns an instantiated executable.
            const auto graphs = RetainedParticipantGraphExecutable::generalBOM(original->resource(DeviceId::cpu()),
                original->resource(devices[i]), shared ? parents + 1 : 1);
            for (const auto &line : graphs.resources()) plan.add(line);
        }
        original->admit(plan.build());
        auto memory = original->memory();
        require(collective->reserveGraphCaptureBoundaryResources(memory), "Native pipeline original boundary BOM failed");
        {
            std::vector<std::shared_ptr<IWorkerGPUContext>> workers;
            std::vector<void *> streams;
            std::vector<std::shared_ptr<TensorBase>> payloads, rows;
            std::vector<std::shared_ptr<RetainedParticipantGraphExecutable>> graphs, children;
            std::shared_ptr<RetainedParticipantGraphExecutable> second_source;
            std::vector<std::array<std::shared_ptr<TensorBase>, 2>> evidence(devices.size());
            for (const auto device : devices) {
                workers.push_back(original->permit(device).worker()); streams.push_back(workers.back()->defaultStream());
                payloads.push_back(std::make_shared<NativePipelinePublicationTensor>(capacity_rows + 2, 7, memory));
                rows.push_back(rows_declaration.materialize(memory));
                const auto index = workers.size() - 1;
                if (shared) {
                    evidence[index][0] = evidence_declaration.materialize(memory);
                    if (static_cast<int>(index) == source) evidence[index][1] = evidence_declaration.materialize(memory);
                }
                workers.back()->submitAndWait([&, index] {
                    auto *count = static_cast<std::int32_t *>(rows[index]->raw_mutable_data());
                    count[0] = 0x1357; count[1] = capacity_rows; count[2] = 0x2468; count[3] = 0x369;
                    TransferEngine::publishHostWrite(rows[index].get());
                    TransferEngine::prepareDeviceInput(rows[index].get(), devices[index], streams[index]);
                    std::fill_n(static_cast<std::uint8_t *>(payloads[index]->raw_mutable_data()), storage_bytes, 0xa5);
                    TransferEngine::publishHostWrite(payloads[index].get());
                    TransferEngine::prepareDeviceInput(payloads[index].get(), devices[index], streams[index]);
                    for (const auto &output : evidence[index]) if (output)
                        TransferEngine::prepareDeviceOutput(output.get(), devices[index], streams[index]);
                });
                graphs.push_back(test::RetainedParticipantGraphExecutableFixtureAccess::original(original->permit(device), streams.back()));
            }
            auto channel = PipelineNativeByteChannel::create(*collective, source, consumers, capacity, memory, streams);
            std::vector<PipelineNativeByteBinding> bindings;
            for (std::size_t i = 0; i < devices.size(); ++i) {
                const CapturedTransferMessage message{0x414354000000001fULL, capacity};
                if (producer_rows && static_cast<int>(i) != source)
                    bindings.emplace_back(channel, i, message, payloads[i], row_bytes,
                        PipelineNativeReceivedRows{rows[i], 3 * sizeof(std::int32_t), row_bytes});
                else
                    bindings.emplace_back(channel, i, message, payloads[i], row_bytes,
                        PipelineNativeRowCount{rows[i], sizeof(std::int32_t), row_bytes});
            }
            if (producer_rows) {
                EXPECT_THROW((PipelineNativeByteBinding(channel, source, {0x414354000000001fULL, capacity},
                    payloads[source], row_bytes, PipelineNativeReceivedRows{rows[source], 3 * sizeof(std::int32_t), row_bytes})),
                    std::invalid_argument);
                const int consumer = consumers.front();
                auto alias = std::make_shared<TensorSlice>(rows[consumer], SliceMetadata{});
                EXPECT_THROW((PipelineNativeByteBinding(channel, consumer, {0x51464e0000000001ULL, 4},
                    rows[consumer], 0, PipelineNativeReceivedRows{alias, 0, 4})), std::invalid_argument);
                EXPECT_NO_THROW((PipelineNativeByteBinding(channel, consumer, {0x51464e0000000001ULL, 4},
                    rows[consumer], 0, PipelineNativeReceivedRows{alias, 8, 4})));
                const auto mapped_bytes = TransferEngine::mappedHostRegionAllocationBytes(sizeof(std::int32_t));
                auto first_claim = memory->claimNewAllocation(DeviceId::cpu(), PhysicalMemoryOwner::ActivationTransportStaging, mapped_bytes);
                auto second_claim = memory->claimNewAllocation(DeviceId::cpu(), PhysicalMemoryOwner::ActivationTransportStaging, mapped_bytes);
                auto mapped = std::make_shared<NativePipelineMappedCount>(memory);
                const auto device = devices[consumer];
                workers[consumer]->submitAndWait([&] {
                    require(mapped->remap(device), "Original mapped count preparation failed");
                    const CapturedTransferMessage message{0x51464e0000000002ULL, capacity};
                    const PipelineNativeReceivedRows count{mapped, 0, row_bytes};
                    const PipelineNativeByteBinding first(channel, consumer, message, payloads[consumer], row_bytes, count);
                    const auto old_identity = first.captureIdentity();
                    const auto *old_address = mapped->gpu_data_ptr();
                    require(mapped->remap(device), "Original mapped count replacement failed");
                    require(mapped->gpu_data_ptr() != old_address, "Old capture lease did not retain its physical mapped range");
                    const PipelineNativeByteBinding replacement(channel, consumer, message, payloads[consumer], row_bytes, count);
                    EXPECT_NE(replacement.captureIdentity(), old_identity);
                    EXPECT_THROW(first.enqueue(streams[consumer]), std::logic_error);
                });
            }
            for (const int consumer : consumers) {
                TransferEngine::publishHostWrite(payloads[consumer].get());
                require(!payloads[consumer]->deviceValid() &&
                    !static_cast<NativePipelinePublicationTensor &>(*payloads[consumer]).publicationEvent(),
                    "Native pipeline output must require a fresh original publication");
            }
            auto snapshot_claim = memory->claimNewAllocation(DeviceId::cpu(), PhysicalMemoryOwner::ActivationTransportStaging,
                TransferEngine::mappedHostRegionAllocationBytes(snapshot_bytes));
            auto snapshot = TransferEngine::instance().allocateMappedHostRegion(snapshot_bytes, devices);
            const auto await = [&] {
                const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
                for (std::size_t i = 0; i < devices.size(); ++i) for (;;) {
                    GPUStreamExecutionState state = GPUStreamExecutionState::Pending;
                    workers[i]->submitAndWait([&] { state = workers[i]->queryStreamExecutionState(streams[i], "native pipeline replay terminal"); });
                    if (state == GPUStreamExecutionState::Complete) break;
                    require(std::chrono::steady_clock::now() < deadline, "Native pipeline replay original stream timed out");
                    std::this_thread::yield();
                }
            };
            NativeReplayFailureDrain drain;
            drain.abort_and_join = [&] {
                require(collective->signalNativeCompletionAbort(), "Native pipeline replay abort failed");
                await();
                if (second_source) require(Graph::observeTerminal(second_source), "Second source terminal proof failed");
                for (const auto &graph : graphs)
                    require(Graph::observeTerminal(graph), "Native pipeline failed graph has no original terminal proof");
                second_source.reset(); graphs.clear(); children.clear();
            };
            // Join cold producers before any participant enters the native
            // recording group. The consumer payload is a recorded-only output.
            for (std::size_t i = 0; i < devices.size(); ++i) workers[i]->submitAndWait([&, i] {
                TransferEngine::requireDeviceInput(rows[i].get(), devices[i], streams[i]);
                if (static_cast<int>(i) == source)
                    TransferEngine::requireDeviceInput(payloads[i].get(), devices[i], streams[i]);
            });
            std::vector<std::unique_ptr<GraphCaptureDependencyLedger>> ledgers;
            std::vector<OriginalNativeGraphRecording> recordings;
            for (std::size_t i = 0; i < devices.size(); ++i) {
                const bool sender = static_cast<int>(i) == source;
                const bool storage_only = !sender && producer_rows;
                ledgers.push_back(std::make_unique<GraphCaptureDependencyLedger>(devices[i], streams[i],
                    std::vector<GraphCaptureDependencyLedger::StagePlan>{{
                        .stage_identity = &bindings[i], .stage_name = "native pipeline complete edge",
                        .external_inputs = sender ? std::vector<const TensorBase *>{payloads[i].get(), rows[i].get()}
                                                  : storage_only ? std::vector<const TensorBase *>{}
                                                  : std::vector<const TensorBase *>{rows[i].get()},
                        .outputs = sender ? std::vector<const TensorBase *>{}
                            : storage_only ? std::vector<const TensorBase *>{payloads[i].get(), rows[i].get()}
                                           : std::vector<const TensorBase *>{payloads[i].get()},
                    }}, "native pipeline exact output publication"));
                auto &ledger = *ledgers.back();
                recordings.push_back({workers[i], devices[i], &Graph::capture(graphs[i]), &ledger, [&, i, sender, storage_only] {
                    ScopedGraphCaptureStage stage(&bindings[i]);
                    if (!sender) {
                        bool rejected = false;
                        try { TransferEngine::requireDeviceInput(payloads[i].get(), devices[i], streams[i]); }
                        catch (const std::logic_error &) { rejected = true; }
                        require(rejected, "Native pipeline output was readable before its producer was recorded");
                    }
                    bindings[i].enqueue(streams[i]);
                    if (!sender) {
                        auto &output = static_cast<NativePipelinePublicationTensor &>(*payloads[i]);
                        require(!payloads[i]->deviceValid() && !output.publicationEvent(),
                            "Native pipeline recording published unexecuted output bytes");
                        require(ledgers[i]->classifyInput(payloads[i].get(), devices[i], streams[i]) ==
                            GraphCaptureDependencyLedger::InputDisposition::InternalRecorded,
                            "Native pipeline producer did not publish its exact recorded output edge");
                        TransferEngine::requireDeviceInput(payloads[i].get(), devices[i], streams[i]);
                        require(!output.publicationEvent(), "Recorded input acquisition invented a global output event");
                        if (storage_only)
                            require(ledgers[i]->classifyInput(rows[i].get(), devices[i], streams[i]) ==
                                GraphCaptureDependencyLedger::InputDisposition::InternalRecorded,
                                "Storage-only row count lost its exact recorded receipt producer");
                    }
                    stage.complete();
                }});
            }
            const auto inspect = [&](const IGPUGraphCapture &definition, std::size_t i, bool child) {
                std::vector<GPUGraphKernelNodeInfo> kernels; std::string detail;
                require(definition.inspectKernelNodes(kernels, &detail), "Native pipeline edge kernel inspection failed");
                std::size_t preparation = 0, validation = 0;
                for (const auto &node : kernels) {
                    const bool prepare = node.name.find("prepareNativePipelineByteFrame") != std::string::npos;
                    const bool validate = node.name.find("validateNativePipelineByteFrame") != std::string::npos;
                    if (!prepare && !validate) continue;
                    require(node.valid() && node.name_resolved && node.grid_x == 1 && node.grid_y == 1 && node.grid_z == 1 &&
                        node.block_x == 1 && node.block_y == 1 && node.block_z == 1 && node.static_shared_memory_bytes == 0 &&
                        node.dynamic_shared_memory_bytes == 0 && node.local_memory_bytes_per_thread == 0,
                        "Native pipeline scalar launch dimensions/resources changed");
                    require(child == (node.nesting_depth > 0), "Native pipeline scalar escaped its original child definition");
                    preparation += prepare; validation += validate;
                }
                const auto expected = static_cast<int>(i) == source ? consumers.size() : 1;
                require(preparation == expected && validation == expected,
                    "Native pipeline edge omitted/duplicated preparation or receipt validation");
            };
            try {
                OriginalNativeGraphCaptureGroup::record(recordings, collective->borrowOriginalNativeContext(),
                    "native pipeline original byte edge");
                for (std::size_t i = 0; i < devices.size(); ++i) workers[i]->submitAndWait([&, i] {
                    auto &definition = Graph::capture(graphs[i]);
                    inspect(definition, i, false);
                    if (shared) Graph::sealSharedChild(graphs[i]);
                    else require(definition.prepareRuntimeContextStorage(memory) && definition.instantiate(),
                        "Native pipeline edge original runtime storage failed");
                });
                if (shared) {
                    children = std::move(graphs);
                    const auto make_parent = [&](std::size_t i, std::size_t variant) {
                        auto parent = Graph::original(original->permit(devices[i]), streams[i]);
                        const bool sender = static_cast<int>(i) == source;
                        const auto device = devices[i];
                        void *const stream = streams[i];
                        auto *const payload = payloads[i].get();
                        auto *const row_count = rows[i].get();
                        auto *const exit = evidence[i][variant].get();
                        const std::array<char, 3> stage_ids{};
                        std::vector<GraphCaptureDependencyLedger::StagePlan> stages;
                        if (sender)
                            stages.push_back({.stage_identity = &stage_ids[0], .stage_name = "source variant marker",
                                .external_inputs = {row_count, payload}, .outputs = {payload}});
                        stages.push_back({.stage_identity = &stage_ids[1], .stage_name = "original native wire child",
                            .external_inputs = {row_count},
                            .internal_inputs = sender ? std::vector<GraphCaptureDependencyLedger::InternalInput>{{payload, 0}}
                                                      : std::vector<GraphCaptureDependencyLedger::InternalInput>{},
                            .outputs = sender ? std::vector<const TensorBase *>{} : std::vector<const TensorBase *>{payload}});
                        // The source marker and the receiving child are each
                        // their respective parent's first payload producer.
                        stages.push_back({.stage_identity = &stage_ids[2], .stage_name = "native child exit copy",
                            .internal_inputs = {{payload, 0}}, .outputs = {exit}});
                        GraphCaptureDependencyLedger parent_ledger(device, stream, std::move(stages),
                            "native pipeline shared child parent");
                        workers[i]->submitAndWait([&] {
                            Graph::bindSharedChild(parent, children[i]);
                            auto &definition = Graph::capture(parent);
                            auto *backend = getBackendFor(device);
                            auto *first_word = static_cast<std::byte *>(payload->gpu_data_ptr()) + row_bytes;
                            ScopedBackendGraphCapture recording(*workers[i], definition,
                                "native pipeline shared child parent", &parent_ledger);
                            require(recording.begin(), "Native shared child parent capture failed");
                            if (sender) {
                                ScopedGraphCaptureStage stage(&stage_ids[0], stream);
                                // The two real source parents have distinct input
                                // addresses and output roots. Their common wire
                                // definition must wait for the selected marker. The
                                // partial copy preserves the prior payload tail.
                                TransferEngine::requireDeviceInput(row_count, device, stream);
                                TransferEngine::requireDeviceInput(payload, device, stream);
                                TransferEngine::requireDeviceOutput(payload, device, stream);
                                const auto *marker = static_cast<const std::int32_t *>(row_count->gpu_data_ptr()) + variant * 2;
                                require(backend->deviceCopyAsync(first_word, marker, sizeof(*marker), device.ordinal, stream),
                                    "Native source variant marker producer failed");
                                TransferEngine::publishDeviceWrite(payload, device, stream);
                                stage.complete();
                            }
                            {
                                ScopedGraphCaptureStage stage(&stage_ids[1], stream);
                                TransferEngine::requireDeviceInput(row_count, device, stream);
                                if (sender) TransferEngine::requireDeviceInput(payload, device, stream);
                                else TransferEngine::requireDeviceOutput(payload, device, stream);
                                Graph::recordSharedChild(parent);
                                if (!sender) TransferEngine::publishDeviceWrite(payload, device, stream);
                                stage.complete();
                            }
                            {
                                ScopedGraphCaptureStage stage(&stage_ids[2], stream);
                                TransferEngine::requireDeviceInput(payload, device, stream);
                                TransferEngine::requireDeviceOutput(exit, device, stream);
                                require(backend->deviceCopyAsync(exit->gpu_data_ptr(), first_word,
                                    sizeof(std::int32_t), device.ordinal, stream), "Native child exit consumer failed");
                                TransferEngine::publishDeviceWrite(exit, device, stream);
                                require(!exit->deviceValid(), "Native child exit copy published before graph launch");
                                stage.complete();
                            }
                            recording.finish();
                            inspect(definition, i, true);
                            require(definition.prepareRuntimeContextStorage(memory) && definition.instantiate(),
                                "Native shared child parent instantiation failed");
                        });
                        return parent;
                    };
                    for (std::size_t i = 0; i < devices.size(); ++i) graphs.push_back(make_parent(i, 0));
                    second_source = make_parent(source, 1);
                    EXPECT_THROW(Graph::retireSharedDefinition(children[source]), std::logic_error);
                    for (const auto consumer : consumers)
                        EXPECT_THROW(Graph::retireSharedDefinition(children[consumer]), std::logic_error);
                }
            } catch (const std::exception &error) {
                std::cerr << "Native pipeline original recording failed: " << error.what() << '\n';
                throw;
            }
            std::uint64_t total_live_bytes = 0;
            std::array<std::uint64_t, 2> source_submissions{};
            for (std::uint64_t replay = 1; replay <= 20; ++replay) {
                const auto live_rows = std::array<std::int32_t, 8>{31, 0, 1, 7, 0, 3, 30, 2}[replay % 8];
                const auto live = std::uint64_t(live_rows) * row_bytes; total_live_bytes += live;
                const std::size_t variant = shared ? replay % 2 : 0;
                const std::int32_t marker = variant ? 0x2468 : 0x1357;
                std::array<std::uint8_t, sizeof(marker)> marker_bytes{};
                std::memcpy(marker_bytes.data(), &marker, sizeof(marker));
                for (std::size_t i = 0; i < devices.size(); ++i) workers[i]->submitAndWait([&, i] {
                    auto *count = static_cast<std::int32_t *>(rows[i]->raw_mutable_data());
                    count[1] = producer_rows && static_cast<int>(i) != source ? -11 : live_rows;
                    count[3] = 0x369;
                    TransferEngine::publishHostWrite(rows[i].get()); TransferEngine::prepareDeviceInput(rows[i].get(), devices[i], streams[i]);
                    if (static_cast<int>(i) == source) {
                        auto *data = static_cast<std::uint8_t *>(payloads[i]->raw_mutable_data());
                        for (std::size_t byte = 0; byte < storage_bytes; ++byte) data[byte] =
                            byte >= row_bytes && byte < row_bytes + capacity ? std::uint8_t((byte + replay * 17) % 251) : 0xa5;
                        TransferEngine::publishHostWrite(payloads[i].get()); TransferEngine::prepareDeviceInput(payloads[i].get(), devices[i], streams[i]);
                    } else {
                        require(getBackendFor(devices[i])->memset(payloads[i]->gpu_data_ptr(), 0xa5, storage_bytes, devices[i].ordinal, streams[i]), "Native pipeline poison failed");
                        TransferEngine::publishDeviceWrite(payloads[i].get(), devices[i], streams[i]);
                    }
                });
                std::vector<std::future<void>> pending;
                pending.reserve(devices.size());
                std::mutex launch_error_mutex;
                std::exception_ptr launch_error;
                const auto fail_launch = [&](std::exception_ptr error) noexcept {
                    { std::lock_guard lock(launch_error_mutex); if (!launch_error) launch_error = std::move(error); }
                    try { if (!collective->signalNativeCompletionAbort()) std::terminate(); } catch (...) { std::terminate(); }
                };
                const auto source_ordinal = shared ? ++source_submissions[variant] : replay;
                try {
                    for (std::size_t i = 0; i < devices.size(); ++i) {
                        pending.push_back(workers[i]->submitAsync([&, i] {
                            try {
                                const bool sender = static_cast<int>(i) == source;
                                const auto &parent = sender && variant ? second_source : graphs[i];
                                require(Graph::submit(parent, sender ? source_ordinal : replay),
                                    "Native pipeline actual edge replay failed");
                                if (shared) {
                                    TransferEngine::publishDeviceWrite(evidence[i][sender ? variant : 0].get(), devices[i], streams[i]);
                                    if (sender) TransferEngine::publishDeviceWrite(payloads[i].get(), devices[i], streams[i]);
                                }
                                if (!sender) {
                                    // The launched parent, rather than its recorded child,
                                    // is the original output event publisher.
                                    TransferEngine::publishDeviceWrite(payloads[i].get(), devices[i], streams[i]);
                                    if (producer_rows)
                                        TransferEngine::publishDeviceWrite(rows[i].get(), devices[i], streams[i]);
                                    auto &output = static_cast<NativePipelinePublicationTensor &>(*payloads[i]);
                                    void *const event = output.publicationEvent();
                                    require(output.freshPublicationOn(devices[i]) && payloads[i]->deviceValid() && event,
                                        "Native pipeline launch boundary omitted its exact-device publication");
                                    TransferEngine::requireDeviceInput(payloads[i].get(), devices[i], streams[i]);
                                    require(output.acquiredOn(event, streams[i]),
                                        "Native pipeline output acquisition did not import its exact event/stream");
                                }
                            } catch (...) { fail_launch(std::current_exception()); throw; }
                        }));
                    }
                } catch (...) { fail_launch(std::current_exception()); }
                joinOriginalNativeWorkerSubmissions(pending, fail_launch,
                    std::chrono::steady_clock::now() + std::chrono::milliseconds(collective_timeout_policy::kDefaultCollectiveTimeoutMs));
                if (launch_error) std::rethrow_exception(launch_error);
                await();
                for (const auto &graph : graphs)
                    require(Graph::observeTerminal(graph), "Native pipeline original graph is not terminal");
                if (second_source) require(Graph::observeTerminal(second_source), "Native alternate source is not terminal");
                for (std::size_t i = 0; i < devices.size(); ++i) workers[i]->submitAndWait([&, i] {
                    auto *destination = static_cast<std::byte *>(snapshot->mutableHostData()) + i * snapshot_stride;
                    auto *backend = getBackendFor(devices[i]);
                    require(backend->deviceToHostOnStream(destination, payloads[i]->gpu_data_ptr(), storage_bytes, devices[i].ordinal, streams[i]), "Native pipeline payload observation failed");
                    require(backend->deviceToHostOnStream(destination + storage_bytes, rows[i]->gpu_data_ptr(), rows_bytes, devices[i].ordinal, streams[i]), "Native pipeline original row observation failed");
                    if (shared) {
                        const auto &output = evidence[i][static_cast<int>(i) == source ? variant : 0];
                        TransferEngine::requireDeviceInput(output.get(), devices[i], streams[i]);
                        require(backend->deviceToHostOnStream(destination + evidence_offset, output->gpu_data_ptr(),
                            sizeof(marker), devices[i].ordinal, streams[i]), "Native pipeline child exit evidence observation failed");
                    }
                    const auto actual_frames = static_cast<int>(i) == source ? consumers : std::vector<int>{static_cast<int>(i)};
                    for (std::size_t edge = 0; edge < actual_frames.size(); ++edge) {
                        const auto observation = test::PipelineNativeByteChannelReplayProbe::frame(*channel, actual_frames[edge], i);
                        require(observation.first == devices[i] && backend->deviceToHostOnStream(
                            destination + storage_bytes + rows_bytes + edge * frame_bytes,
                            observation.second, frame_bytes, devices[i].ordinal, streams[i]), "Native pipeline native receipt observation failed");
                    }
                });
                await();
                for (std::size_t i = 0; i < devices.size(); ++i) {
                    const auto *observed = static_cast<const std::uint8_t *>(snapshot->mutableHostData()) + i * snapshot_stride;
                    for (std::size_t byte = 0; byte < storage_bytes; ++byte) {
                        const bool actual_data = byte >= row_bytes && byte - row_bytes < (static_cast<int>(i) == source ? capacity : live);
                        auto expected = std::uint8_t{0xa5};
                        if (actual_data) {
                            expected = std::uint8_t((byte + replay * 17) % 251);
                            if (shared && byte - row_bytes < marker_bytes.size()) expected = marker_bytes[byte - row_bytes];
                        }
                        ASSERT_EQ(observed[byte], expected) << " participant=" << i << " replay=" << replay << " byte=" << byte;
                    }
                    if (shared) {
                        std::array<std::uint8_t, sizeof(marker)> exit_bytes{};
                        std::memcpy(exit_bytes.data(), observed + evidence_offset, exit_bytes.size());
                        auto expected = marker_bytes;
                        if (static_cast<int>(i) != source && !live) expected.fill(0xa5);
                        EXPECT_EQ(exit_bytes, expected);
                    }
                    std::array<std::int32_t, 4> original_rows{}; std::memcpy(original_rows.data(), observed + storage_bytes, rows_bytes);
                    const bool storage_only = producer_rows && static_cast<int>(i) != source;
                    EXPECT_EQ(original_rows, (std::array<std::int32_t, 4>{0x1357,
                        storage_only ? -11 : live_rows, 0x2468, storage_only ? live_rows : 0x369}));
                    for (std::size_t edge = 0; edge < (static_cast<int>(i) == source ? consumers.size() : 1); ++edge) {
                        NativeCountedAllGatherFrame frame{}; ncclLlaminarByteCompletion receipt{}; std::uint64_t extent = 0, traffic = 0;
                        const auto *native = observed + storage_bytes + rows_bytes + edge * frame_bytes;
                        std::memcpy(&frame, native, sizeof(frame)); std::memcpy(&extent, native + frame_layout.extents(), sizeof(extent));
                        std::memcpy(&receipt, native + frame_layout.receipts(), sizeof(receipt)); std::memcpy(&traffic, native + frame_layout.counters(), sizeof(traffic));
                        EXPECT_EQ(frame.sequence, replay); EXPECT_EQ(frame.first_error, 0); EXPECT_EQ(extent, live);
                        const auto authority = collective->nativeCompletionAuthority(i).binding();
                        EXPECT_EQ(receipt.nativeAbortFlag, authority.abortFlag); EXPECT_EQ(receipt.nativeCommunicator, authority.deviceCommunicator);
                        EXPECT_EQ(receipt.phase, ncclLlaminarCompletionPhase::Completed); EXPECT_EQ(receipt.bytes, live);
                        EXPECT_EQ(receipt.sequence, replay); EXPECT_EQ(receipt.table, 0x414354000000001fULL); EXPECT_EQ(receipt.admission, capacity);
                        EXPECT_EQ(receipt.reserved, 0u); EXPECT_NE(receipt.message, 0u);
                        EXPECT_EQ(traffic, static_cast<int>(i) == source ? total_live_bytes : 0);
                    }
                }
            }
            // The host owner has authenticated all terminal submissions above;
            // each retained definition performs its own exact-worker teardown.
            if (shared) {
                EXPECT_EQ(source_submissions, (std::array<std::uint64_t, 2>{10, 10}));
                Graph::retireSharedDefinition(graphs[source]);
                EXPECT_THROW(Graph::retireSharedDefinition(children[source]), std::logic_error);
            }
            second_source.reset(); graphs.clear();
            for (const auto &child : children) Graph::retireSharedDefinition(child);
            children.clear();
            drain.completed = true;
            // Replacement exercises the actual payload/capture lease bundle;
            // owners must outlive every copied and moved capture borrow.
            auto retained = std::make_optional(bindings[consumers.front()]);
            *retained = bindings[source];
            *retained = std::move(bindings[consumers.front()]);
            *retained = *retained;
            *retained = std::move(*retained);
            bindings.clear(); channel.reset();
            collective.reset();
            // A direct-created binding has the same revocable original-context
            // authority as installed ports. Rejection precedes backend/event work.
            EXPECT_THROW(retained->enqueue(streams[consumers.front()]), std::logic_error);
            retained.reset(); payloads.clear(); rows.clear();
        }
        collective.reset(); original->requirePayloadRetired(); original->retire(); original->requireExactRetired();
    });
}
}
#ifdef HAVE_CUDA
TEST(NativePipelineBytes, CUDA2_RecordedReplay) { nativePipelineByteReplay(DeviceType::CUDA, 2); }
TEST(NativePipelineBytes, CUDA2_SharedChildReplay)
{ nativePipelineByteReplay(DeviceType::CUDA, 2, NativePipelineByteDefinition::SharedChild); }
TEST(NativePipelineBytes, CUDA2_ProducerRowsReplay)
{ nativePipelineByteReplay(DeviceType::CUDA, 2, NativePipelineByteDefinition::ProducerRows); }
#endif
#ifdef HAVE_ROCM
TEST(NativePipelineBytes, ROCm4_RecordedReplay) { nativePipelineByteReplay(DeviceType::ROCm, 4); }
TEST(NativePipelineBytes, ROCm2_SharedChildReplay)
{ nativePipelineByteReplay(DeviceType::ROCm, 2, NativePipelineByteDefinition::SharedChild); }
TEST(NativePipelineBytes, ROCm4_ProducerRowsReplay)
{ nativePipelineByteReplay(DeviceType::ROCm, 4, NativePipelineByteDefinition::ProducerRows); }
#endif
