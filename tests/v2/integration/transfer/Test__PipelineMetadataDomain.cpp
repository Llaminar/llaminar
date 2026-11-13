/**
 * @file Test__PipelineMetadataDomain.cpp
 * @brief Original two-rank CUDA TP2/ROCm TP2 captured metadata round trip.
 *
 * Each original rank owns only its NUMA-local native TP2 population, PMA and
 * retained graphs. Leaders alone attach rank-channel cursors; native NCCL/RCCL
 * fanout stays within its original rank. Every original metadata schema keeps
 * tensor and retained-workspace guards, independent member poison and twenty
 * recorded-input replays. Exact final wire fields and device epochs prove
 * compact communication; terminal observation precedes shared-session retirement.
 */
#include "execution/local_execution/orchestrators/PipelineMetadataExchange.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "fixtures/backends/PreparedNativeLookupFixture.h"
#include "fixtures/backends/RetainedParticipantGraphExecutableFixtureAccess.h"
#include "planning/CollectiveMemoryEstimator.h"
#include "planning/PlanningPublication.h"
#include "planning/RankHardwareOwnership.h"
#include "memory/CPUTensorStorageDeclaration.h"
#include "collective/ILocalTPContext.h"
#include "CapturedTransferChannelReplayProbe.h"
#include "utils/MPIContext.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <barrier>
#include <chrono>
#include <cstring>
#include <functional>
#include <cstdint>
#include <exception>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>
#include <set>
#include <thread>

using namespace llaminar2;

namespace
{
/** @brief Reject incomplete original setup, execution or terminal evidence. */
void require(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }

/** @brief Complete one original all-rank phase before advancing its lifecycle.
 * @param mpi Original communicator, including both native domain owners.
 * @param action Local work whose failure must be observed by the peer rank.
 */
template<class Action>
void consensus(const std::shared_ptr<IMPIContext> &mpi, Action action)
{
    exchangePlanningSamples(mpi, [](int ranks) {
        return std::vector<std::vector<uint8_t>>(ranks, std::vector<uint8_t>{1});
    }, [&](auto request) {
        require(request.size() == 1 && request[0] == 1, "Pipeline metadata phase declaration disagrees");
        action();
        return std::vector<uint8_t>{1};
    }, [](auto receipts) {
        for (const auto &receipt : receipts)
            require(receipt.bytes.size() == 1 && receipt.bytes[0] == 1,
                "Pipeline metadata phase omitted an original rank");
    });
}

/** @brief Exact two-member original ownership, independent of ordinal/NUMA arithmetic. */
struct Domain
{
    int rank;
    std::array<DeviceId, 2> devices;
};

/**
 * @brief Select CUDA on original rank0 and ROCm on original rank1 from observation.
 * @param inventory Canonical original MPI discovery publication.
 * @return Two native domains; larger local ordinal is deliberately the leader.
 * @throws std::exception For absent/foreign/duplicate physical ownership.
 */
std::array<Domain, 2> originalDomains(const ClusterInventory &inventory)
{
    require(inventory.ranks.size() == 2, "Pipeline metadata requires exactly two original ranks");
    std::array<Domain, 2> domains;
    std::set<std::string> uuids;
    std::optional<int> node;
    for (int owner = 0; owner < 2; ++owner)
    {
        const auto rank = std::find_if(inventory.ranks.begin(), inventory.ranks.end(),
            [owner](const auto &candidate) { return candidate.rank == owner; });
        require(rank != inventory.ranks.end() && rank->node_id >= 0,
            "Pipeline metadata original rank/node observation is absent");
        require(!node || *node == rank->node_id, "Pipeline metadata rank channels require one physical node");
        node = rank->node_id;
        const auto family = owner == 0 ? DeviceType::CUDA : DeviceType::ROCm;
        std::vector<const DeviceInfo *> owned;
        for (const auto &gpu : rank->gpus)
            if (gpu.type == family && rankOwnsGPUDevice(*rank, gpu)) owned.push_back(&gpu);
        std::sort(owned.begin(), owned.end(), [](const auto *a, const auto *b) {
            return a->local_device_id < b->local_device_id;
        });
        require(owned.size() >= 2, "Pipeline metadata requires original NUMA-local CUDA TP2 on rank0 and ROCm TP2 on rank1");
        domains[owner].rank = owner;
        for (size_t member = 0; member < 2; ++member)
        {
            const auto &gpu = *owned[1 - member];
            require(!gpu.uuid.empty() && uuids.insert(gpu.uuid).second,
                "Pipeline metadata domains must have four distinct physical GPU UUIDs");
            domains[owner].devices[member] = {family, gpu.local_device_id};
            requireRankLocalGPUOwnership(*rank, domains[owner].devices[member], "Pipeline metadata original native domain");
        }
        require(domains[owner].devices[0].ordinal > domains[owner].devices[1].ordinal,
            "Pipeline metadata leader coordinates were not reversed");
    }
    return domains;
}

/** @brief Cancel native work and prove physical completion before payload owners unwind. */
struct FailureDrain
{
    std::function<void()> drain;
    std::shared_ptr<IMPIContext> mpi;
    bool completed = false;
    /** @brief Failed cancellation/retirement is fatal before any shared backing can disappear. */
    ~FailureDrain() noexcept
    {
        if (completed) return;
        try { drain(); }
        catch (...) { MPI_Abort(mpi->communicator(), 2); std::terminate(); }
    }
};

/** @brief One original rank source and native communicator for all seven schemas.
 * General graph opaque reservations survive individual graph destruction, so
 * admission covers seven definitions per device. Tensor, channel and workspace
 * payloads are sequential and reuse one schema's admitted peak capacity.
 */
struct OriginalPopulation
{
    std::unique_ptr<test::PreparedNativeLookupFixture> original;
    std::unique_ptr<ILocalTPContext> collective;
};

/**
 * @brief Exercise both original native broadcasts and both explicit rank channels.
 * @param cuda_first Select which original domain publishes the initial metadata.
 * @param layout Complete immutable metadata schema, including physical verifier rows.
 * @param population Original source and communicator retained across every schema.
 * Every schema retains one captured graph across twenty independently poisoned
 * recorded inputs, preserving exact bank extents and workspace ownership.
 */
void exerciseLayout(bool cuda_first, PipelineMetadataLayout layout, OriginalPopulation &population)
{
    using Kind = PipelineMetadataLayout::Kind;
    constexpr size_t capacity = 16, payload_bytes = capacity * sizeof(int32_t);
    constexpr size_t storage_words = 128, storage_bytes = storage_words * sizeof(int32_t);
    constexpr size_t workspace_words = 32, workspace_bytes = workspace_words * sizeof(int32_t);
    constexpr size_t snapshot_bytes = 2 * (storage_bytes + workspace_bytes);
    constexpr uint64_t forward_key = 101, backward_key = 102;
    const auto mpi = MPIContextFactory::global();
    require(mpi && mpi->world_size() == 2 && mpi->clusterInventory(),
        "Pipeline metadata requires original MPI2 inventory publication");
    // Lazy topology construction contains original MPI collectives. Finish it
    // on every rank before either native vendor enters its owner-only setup.
    consensus(mpi, [&] { require(mpi->topology() != nullptr, "Pipeline metadata topology is unavailable"); });
    const auto inventory = mpi->clusterInventory();
    const auto domains = originalDomains(*inventory);
    const int source_rank = cuda_first ? 0 : 1;
    const int destination_rank = 1 - source_rank;
    const bool source = mpi->rank() == source_rank;
    const auto devices = domains[mpi->rank()].devices;
    auto session = CapturedTransferRankSession::create(mpi);
    const std::array plans{
        CapturedTransferRankPlan::fromInventory(*inventory, {session->originalGeneration(), forward_key},
            {source_rank, domains[source_rank].devices[0]}, {destination_rank, domains[destination_rank].devices[0]}, payload_bytes),
        CapturedTransferRankPlan::fromInventory(*inventory, {session->originalGeneration(), backward_key},
            {destination_rank, domains[destination_rank].devices[0]}, {source_rank, domains[source_rank].devices[0]}, payload_bytes,
            CapturedTransferEndpoint::Consumer)};
    const CPUTensorStorageDeclaration hidden_declaration(1, storage_words, BufferTensorType::INT32,
        PhysicalMemoryOwner::ExecutionWorkspace, DeviceId::cpu());
    auto &original = population.original;
    auto &collective = population.collective;
    std::shared_ptr<PhysicalMemoryAuthority> memory;
    // The canonical exchange supplies all-rank failure consensus for source/PMA
    // preparation before either process may attach a shared native channel.
    try {
        exchangePlanningSamples(mpi, [&](int count) {
            return std::vector<std::vector<uint8_t>>(count, plans[0].serialize());
        }, [&](auto wire) {
            require(std::vector<uint8_t>(wire.begin(), wire.end()) == plans[0].serialize(),
                "Pipeline metadata original rank declarations disagree");
            if (original)
            {
                memory = original->memory();
                return std::vector<uint8_t>{1};
            }
            original = test::PreparedNativeLookupFixture::prepare(devices, mpi->rank());
            std::vector<GlobalDeviceAddress> addresses;
            for (const auto device : devices) addresses.push_back(GlobalDeviceAddress::fromLocalDeviceId(device));
            const auto native_backend = devices[0].is_cuda() ? CollectiveBackendType::NCCL : CollectiveBackendType::RCCL;
            collective = createLocalTPContext(addresses, {}, native_backend);
            require(bool(collective), "Pipeline metadata original native TP context is absent");
            const auto native = CollectiveMemoryEstimator::localTP(1, capacity, native_backend);
            PhysicalMemoryPlanBuilder builder;
            for (const auto &plan : plans)
                plan.contributeMemory(*plan.localRole(mpi->rank()), original->resource(devices[0]),
                    plan.hostRank() == mpi->rank() ? std::optional{original->resource(DeviceId::cpu())} : std::nullopt, builder);
            builder.add(original->resource(DeviceId::cpu()), PhysicalMemoryOwner::ActivationTransportStaging,
                TransferEngine::mappedHostRegionAllocationBytes(snapshot_bytes));
            builder.add(original->resource(DeviceId::cpu()), PhysicalMemoryOwner::ExecutionWorkspace,
                2 * hidden_declaration.physicalBytes());
            for (const auto device : devices)
            {
                builder.add(original->resource(device), PhysicalMemoryOwner::ExecutionWorkspace, storage_bytes + workspace_bytes);
                builder.add(original->resource(device), PhysicalMemoryOwner::LocalCollective, native.perDeviceBytes());
                const auto graph_bom = RetainedParticipantGraphExecutable::generalBOM(
                    original->resource(DeviceId::cpu()), original->resource(device), 7);
                for (const auto &row : graph_bom.resources()) builder.add(row);
            }
            original->admit(builder.build()); memory = original->memory();
            require(collective->reserveCollectiveResources(native.backend_payload_capacity_bytes, native.fp16_scratch_elements, memory),
                "Pipeline metadata original native resources were not admitted");
            return std::vector<uint8_t>{1};
        }, [](auto receipts) {
            for (const auto &receipt : receipts)
                require(receipt.bytes.size() == 1 && receipt.bytes[0] == 1, "Pipeline metadata preparation omitted an original rank");
        });
    }
    catch (...)
    {
        // No channel exists yet. Drop native payload before original source
        // retirement, including when a peer rejected its own preparation.
        collective.reset(); original.reset();
        throw;
    }
    auto &transfer = TransferEngine::instance();
    std::array<std::shared_ptr<IWorkerGPUContext>, 2> workers{original->permit(devices[0]).worker(), original->permit(devices[1]).worker()};
    std::array<void *, 2> streams{workers[0]->defaultStream(), workers[1]->defaultStream()};
    std::array<IBackend *, 2> backends{getBackendFor(devices[0]), getBackendFor(devices[1])};
    const auto await = [&] {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        for (size_t member = 0; member < 2; ++member)
            for (;;)
            {
                GPUStreamExecutionState state = GPUStreamExecutionState::Pending;
                workers[member]->submitAndWait([&, member] {
                    state = workers[member]->queryStreamExecutionState(streams[member], "original rank TP metadata terminal");
                });
                if (state == GPUStreamExecutionState::Complete) break;
                require(std::chrono::steady_clock::now() < deadline, "Pipeline metadata original terminal exceeded protocol deadline");
                std::this_thread::yield();
            }
    };
    std::array<std::shared_ptr<CapturedTransferChannel>, 2> channels;
    try { consensus(mpi, [&] {
        for (size_t direction = 0; direction < 2; ++direction)
        {
            channels[direction] = transfer.createRankCapturedTransferChannel(plans[direction], session, memory, streams[0]);
            require(bool(channels[direction]), "Pipeline metadata lost its original rank channel");
            const auto own_role = *plans[direction].localRole(mpi->rank());
            const auto peer_role = own_role == CapturedTransferEndpoint::Producer
                ? CapturedTransferEndpoint::Consumer : CapturedTransferEndpoint::Producer;
            require(channels[direction]->endpointDevice(own_role) == devices[0], "Pipeline metadata changed its original leader");
            require(!CapturedTransferChannelReplayProbe::terminalCursor(*channels[direction], peer_role),
                "Pipeline metadata created a foreign-rank GPU cursor");
        }
    }); }
    catch (...)
    {
        await();
        channels = {};
        session->retireAfterOriginalGraphRelease();
        workers = {};
        throw;
    }
    bool premature_retirement_rejected = false;
    try { session->retireAfterOriginalGraphRelease(); }
    catch (const std::exception &) { premature_retirement_rejected = true; }
    require(premature_retirement_rejected && !session->retired(), "Pipeline metadata retired shared backing with live borrowers");
    std::exception_ptr failure;
    try
    {
        std::array<std::shared_ptr<TensorBase>, 2> hidden;
        std::array<std::shared_ptr<const WorkspaceBufferLease>, 2> workspaces;
        const auto incoming = channels[source ? 1 : 0];
        const auto outgoing = channels[source ? 0 : 1];
        std::array<std::unique_ptr<PipelineMetadataExchange>, 2> receives;
        std::unique_ptr<PipelineMetadataExchange> send;
        PhysicalMemoryAllocationLease snapshot_claim;
        std::shared_ptr<MappedHostTransferRegion> snapshot;
        std::array<std::shared_ptr<RetainedParticipantGraphExecutable>, 2> owners;
        try { consensus(mpi, [&] {
            for (size_t member = 0; member < 2; ++member)
            {
                hidden[member] = hidden_declaration.materialize(memory);
                workers[member]->submitAndWait([&, member] {
                    auto *values = static_cast<int32_t *>(hidden[member]->raw_mutable_data());
                    std::fill_n(values, storage_words, -91 - int(member));
                    TransferEngine::publishHostWrite(hidden[member].get());
                    TransferEngine::prepareDeviceInput(hidden[member].get(), devices[member], streams[member]);
                    DeviceWorkspaceManager manager(devices[member], workspace_bytes, memory);
                    WorkspaceRequirements requirements;
                    requirements.buffers.push_back({"accepted_state_slot_indices", workspace_bytes, sizeof(int32_t), true});
                    require(manager.allocate(requirements), "Pipeline metadata workspace allocation failed");
                    workspaces[member] = manager.retainBuffer("accepted_state_slot_indices", 0, workspace_bytes);
                    manager.release(); // The bank retains the original PMA allocation through graph retirement.
                });
            }
            for (size_t member = 0; member < 2; ++member)
            {
                std::vector<PipelineMetadataBank> banks;
                for (size_t field = 0; field < layout.fieldCount(); ++field)
                    if (layout.kind() == Kind::CommittedState && field == 0)
                        banks.emplace_back(workspaces[member], sizeof(int32_t), 1);
                    else
                        banks.emplace_back(hidden[member], BufferId::MTP_LOGICAL_SEQUENCE_STATE,
                            (field * 32 + 1) * sizeof(int32_t), layout.elements(field));
                receives[member] = std::make_unique<PipelineMetadataExchange>(devices[member], layout,
                    PipelineMetadataExchange::CapturedDomain{.context = collective.get(), .participant = int(member),
                        .inbound = incoming}, banks);
                receives[member]->setGPUStream(streams[member]);
                if (member == 0)
                {
                    send = std::make_unique<PipelineMetadataExchange>(devices[0], layout,
                        PipelineMetadataExchange::CapturedDomain{.context = collective.get(), .outbound = outgoing}, std::move(banks));
                    send->setGPUStream(streams[0]);
                }
            }
            snapshot_claim = memory->claimNewAllocation(DeviceId::cpu(), PhysicalMemoryOwner::ActivationTransportStaging,
                TransferEngine::mappedHostRegionAllocationBytes(snapshot_bytes));
            snapshot = transfer.allocateMappedHostRegion(snapshot_bytes, devices);
            for (size_t member = 0; member < 2; ++member)
                owners[member] = test::RetainedParticipantGraphExecutableFixtureAccess::original(original->permit(devices[member]), streams[member]);
        }); }
        catch (...)
        {
            // Admission uploads must finish before partially built payload
            // owners unwind, including when only the peer rejected setup.
            await();
            throw;
        }
        CapturedTransferInvocation active{};
        std::array<bool, 2> admitted{};
        FailureDrain drain{[&] {
            for (size_t direction = 0; direction < 2; ++direction)
                if (admitted[direction]) require(transfer.cancelCapturedTransferInvocation(channels[direction], active),
                    "Pipeline metadata lost its original cancellation admission");
            require(collective->signalNativeCompletionAbort(), "Pipeline metadata original native cancellation failed");
            await();
            for (size_t direction = 0; direction < 2; ++direction)
                if (admitted[direction])
                {
                    require(transfer.finishCapturedTransferInvocation(channels[direction], active) == CapturedTransferInvocationRetirement::Cancelled,
                        "Pipeline metadata original cancelled invocation failed retirement");
                    admitted[direction] = false;
                }
            active = {};
        }, mpi};
        std::barrier begun(2), recorded(2), ended(2);
        std::array<bool, 2> began{}, failed{};
        std::array<std::exception_ptr, 2> errors{};
        // Capture access is a construction capability; preserve only its const
        // observations once instantiation seals the retained owner.
        std::array<const IGPUGraphCapture *, 2> frozen_definitions{};
        const auto capture = [&](size_t member) {
            IGPUGraphCapture *definition = nullptr;
            std::unique_ptr<GraphCaptureDependencyLedger> ledger;
            std::unique_ptr<ScopedBackendGraphCapture> scope;
            try
            {
                definition = &test::RetainedParticipantGraphExecutableFixtureAccess::capture(owners[member]);
                frozen_definitions[member] = definition;
                // External upload events belong to the uncaptured admission
                // frontier. Join them on their exact consumer before recording
                // either native domain, as the production executor does.
                // Followers also retain poisoned guards from their admission
                // upload, so every local bank joins its original upload event.
                TransferEngine::requireDeviceInput(hidden[member].get(), devices[member], streams[member]);
                using StagePlan = GraphCaptureDependencyLedger::StagePlan;
                std::vector<StagePlan> stages;
                if (source && member == 0)
                    stages.push_back({.stage_identity = send.get(), .stage_name = send->name(),
                        .external_inputs = {hidden[member].get()}});
                stages.push_back({.stage_identity = receives[member].get(), .stage_name = receives[member]->name(),
                    .outputs = {hidden[member].get()}});
                if (!source && member == 0)
                    stages.push_back({.stage_identity = send.get(), .stage_name = send->name(),
                        .internal_inputs = {{hidden[member].get(), 0}}});
                ledger = std::make_unique<GraphCaptureDependencyLedger>(devices[member], streams[member], std::move(stages),
                    "original rank TP2 metadata round trip");
                scope = std::make_unique<ScopedBackendGraphCapture>(*workers[member], *definition,
                    "original rank TP2 metadata round trip", ledger.get());
                require(scope->begin(), "Pipeline metadata original capture begin failed"); began[member] = true;
            }
            catch (...) { errors[member] = std::current_exception(); }
            begun.arrive_and_wait();
            if (began[0] && began[1]) try
            {
                auto execution = IDeviceContext::create(devices[member], 1);
                require(bool(execution), "Pipeline metadata original execution context is absent");
                const auto record = [&](PipelineMetadataExchange &stage) {
                    ScopedGraphCaptureStage stage_scope(&stage);
                    require(stage.execute(execution.get()), "Pipeline metadata original stage capture failed");
                    stage_scope.complete();
                };
                if (source && member == 0) record(*send);
                record(*receives[member]);
                if (!source && member == 0) record(*send);
            }
            catch (...) { errors[member] = std::current_exception(); collective->signalNativeCompletionAbort(); }
            recorded.arrive_and_wait();
            if (scope) try { scope->finish(); }
                catch (...) { if (!errors[member]) errors[member] = std::current_exception(); collective->signalNativeCompletionAbort(); }
            failed[member] = bool(errors[member]);
            ended.arrive_and_wait();
            if (!failed[0] && !failed[1]) try
            {
                require(definition->prepareRuntimeContextStorage(memory) && definition->instantiate(),
                    "Pipeline metadata original retained graph instantiation failed");
            }
            catch (...) { errors[member] = std::current_exception(); collective->signalNativeCompletionAbort(); }
        };
        auto first = workers[0]->submitAsync([&] { capture(0); });
        auto second = workers[1]->submitAsync([&] { capture(1); });
        first.get(); second.get();
        // All ranks agree capture success before any retained graph is launched.
        exchangePlanningSamples(mpi, [&](int count) {
            return std::vector<std::vector<uint8_t>>(count, std::vector<uint8_t>{1});
        }, [&](auto) {
            for (const auto &error : errors) if (error) std::rethrow_exception(error);
            return std::vector<uint8_t>{1};
        }, [](auto receipts) {
            for (const auto &receipt : receipts) require(receipt.bytes.size() == 1 && receipt.bytes[0] == 1, "Pipeline metadata capture omitted a rank");
        });
        const std::array frozen_nodes{frozen_definitions[0]->nodeCount(), frozen_definitions[1]->nodeCount()};
        require(frozen_nodes[0] > 0 && frozen_nodes[1] > 0, "Pipeline metadata original graphs are empty");
        uint64_t ordinal = 0;
        for (int replay = 0; replay < 20; ++replay)
            {
                SCOPED_TRACE(layout.name());
                SCOPED_TRACE(replay);
                ++ordinal;
                // Original all-rank failure consensus encloses the complete
                // request, so a local launch/guard failure cannot make a peer
                // enter a different retirement collective.
                exchangePlanningSamples(mpi, [&](int count) {
                    return std::vector<std::vector<uint8_t>>(count, std::vector<uint8_t>{1});
                }, [&](auto) {
                    try
                    {
                        for (size_t member = 0; member < 2; ++member)
                        {
                            // Explicit test admission restores recorded input, never a
                            // host shadow read from live GPU state to choose a branch.
                            auto *values = static_cast<int32_t *>(hidden[member]->raw_mutable_data());
                            for (size_t word = 0; word < storage_words; ++word)
                                values[word] = source && member == 0 ? int32_t(1000 * ordinal + word) : -91 - int(member);
                            TransferEngine::publishHostWrite(hidden[member].get());
                            workers[member]->submitAndWait([&, member] {
                                TransferEngine::prepareDeviceInput(hidden[member].get(), devices[member], streams[member]);
                                auto *target = static_cast<std::byte *>(snapshot->mutableHostData()) + member * (storage_bytes + workspace_bytes);
                                auto *workspace_input = reinterpret_cast<int32_t *>(target + storage_bytes);
                                for (size_t word = 0; word < workspace_words; ++word)
                                    workspace_input[word] = source && member == 0 ? int32_t(1000 * ordinal + storage_words + word) : -91 - int(member);
                                require(backends[member]->hostToDevice(workspaces[member]->data(), workspace_input, workspace_bytes,
                                    devices[member].ordinal, streams[member]), "Pipeline metadata workspace admission upload failed");
                            });
                        }
                        active = {session->originalGeneration(), ordinal};
                        for (size_t direction = 0; direction < 2; ++direction)
                        {
                            transfer.beginCapturedTransferInvocation(channels[direction], active);
                            admitted[direction] = true;
                        }
                        for (const auto member : ordinal % 2 ? std::array<size_t, 2>{1, 0} : std::array<size_t, 2>{0, 1})
                            workers[member]->submitAndWait([&, member] {
                                require(frozen_definitions[member]->executionStream() == streams[member] &&
                                    frozen_definitions[member]->nodeCount() == frozen_nodes[member],
                                    "Pipeline metadata request reset changed the retained graph definition");
                                require(test::RetainedParticipantGraphExecutableFixtureAccess::submit(owners[member], ordinal),
                                    "Pipeline metadata original retained replay submission failed");
                            });
                        await();
                        for (size_t member = 0; member < 2; ++member)
                        {
                            require(test::RetainedParticipantGraphExecutableFixtureAccess::observeTerminal(owners[member]),
                                "Pipeline metadata retained graph lacks original terminal proof");
                            workers[member]->submitAndWait([&, member] {
                                auto *target = static_cast<std::byte *>(snapshot->mutableHostData()) + member * (storage_bytes + workspace_bytes);
                                require(backends[member]->deviceToHostOnStream(target, hidden[member]->gpu_data_ptr(), storage_bytes,
                                    devices[member].ordinal, streams[member]), "Pipeline metadata terminal payload readback failed");
                                require(backends[member]->deviceToHostOnStream(target + storage_bytes, workspaces[member]->data(), workspace_bytes,
                                    devices[member].ordinal, streams[member]), "Pipeline metadata terminal retained workspace readback failed");
                            });
                        }
                        await();
                        for (size_t member = 0; member < 2; ++member)
                        {
                            const auto *target = static_cast<const std::byte *>(snapshot->mutableHostData()) + member * (storage_bytes + workspace_bytes);
                            const auto *actual = reinterpret_cast<const int32_t *>(target);
                            for (size_t word = 0; word < storage_words + workspace_words; ++word)
                            {
                                bool transferred = false;
                                for (size_t field = 0; field < layout.fieldCount(); ++field)
                                {
                                    const size_t start = layout.kind() == Kind::CommittedState && field == 0
                                        ? storage_words + 1 : field * 32 + 1;
                                    transferred |= word >= start && word - start < layout.elements(field);
                                }
                                require(actual[word] == (transferred || (source && member == 0)
                                    ? int32_t(1000 * ordinal + word) : -91 - int(member)),
                                    "Pipeline metadata changed a live bank or copied unused capacity/guard");
                            }
                        }
                        for (size_t direction = 0; direction < 2; ++direction)
                        {
                            const auto &channel = channels[direction];
                            const auto &control = CapturedTransferChannelReplayProbe::terminalControl(*channel);
                            const auto final = layout.message(layout.fieldCount() - 1);
                            require(control.producer.message.bytes == final.bytes && control.consumer.message.bytes == final.bytes &&
                                control.producer.message.key == final.key && control.consumer.message.key == final.key &&
                                control.producer.epoch == ordinal * layout.fieldCount() &&
                                control.consumer.epoch == ordinal * layout.fieldCount(),
                                "Pipeline metadata communicated capacity, stale bank identity or incomplete field epochs");
                            require(transfer.finishCapturedTransferInvocation(channel, active) == CapturedTransferInvocationRetirement::Completed,
                                "Pipeline metadata original request retirement failed");
                            admitted[direction] = false;
                        }
                        active = {};
                        return std::vector<uint8_t>{1};
                    }
                    catch (...)
                    {
                        // Cancel while every original graph/stage borrow is
                        // still alive, then propagate through shared consensus.
                        drain.drain();
                        drain.completed = true;
                        throw;
                    }
                }, [](auto receipts) {
                    for (const auto &receipt : receipts)
                        require(receipt.bytes.size() == 1 && receipt.bytes[0] == 1, "Pipeline metadata terminal evidence omitted a rank");
                });
            }
        ::testing::Test::RecordProperty("pipeline_metadata_rank_domains", "originalMPI2;CUDA_TP2_rank0;ROCm_TP2_rank1;reversed_leaders;20_retained_replays_per_schema;retained_workspace_guard");
        drain.completed = true;
    }
    catch (...) { failure = std::current_exception(); }
    // FailureDrain cancelled in-flight invocations/native operations and proved
    // the exact local streams retired before graph/stage/channel destruction.
    // Both original ranks retain the backing until the same closed boundary.
    mpi->barrier();
    channels = {};
    session->retireAfterOriginalGraphRelease();
    require(session->retired(), "Pipeline metadata shared original rank session did not retire");
    workers = {};
    if (failure) std::rethrow_exception(failure);
}
/** @brief Preserve every original condition, verifier, committed-state and token schema.
 * @param cuda_first Select the existing original metadata source domain. */
void exerciseDomains(bool cuda_first)
{
    using Layout = PipelineMetadataLayout;
    using Kind = Layout::Kind;
    const auto mpi = MPIContextFactory::global();
    OriginalPopulation population;
    std::exception_ptr failure;
    try
    {
        for (const auto layout : {Layout(Kind::Condition, 1), Layout(Kind::Verifier, 2),
                 Layout(Kind::Verifier, 3), Layout(Kind::Verifier, 4), Layout(Kind::Verifier, 16),
                 Layout(Kind::CommittedState, 1), Layout(Kind::NextToken, 1)})
            exerciseLayout(cuda_first, layout, population);
    }
    catch (...) { failure = std::current_exception(); }
    // Every schema retires its rank session and native borrowers before the
    // one original communicator/source closes. Process-retained General tokens
    // stay with that same source; no later schema adopts a fresh authority.
    consensus(mpi, [&] {
        population.collective.reset();
        if (population.original)
        {
            population.original->requirePayloadRetired();
            population.original->retire();
            population.original->requireExactRetired();
        }
    });
    if (failure) std::rethrow_exception(failure);
}

}

/** @test Real CUDA TP2 output reaches both original ROCm TP2 members and returns. */
TEST(PipelineMetadataDomains, CUDA2_ROCm2_CapturedBanks) { exerciseDomains(true); }
/** @test Reverse vendor order preserves both native domains and original rank ownership. */
TEST(PipelineMetadataDomains, ROCm2_CUDA2_CapturedBanks) { exerciseDomains(false); }
