/**
 * @file Test__PipelineActivationDomain.cpp
 * @brief Original two-rank CUDA TP2/ROCm TP2 captured activation round trip.
 *
 * Each original MPI rank creates only its NUMA-owned native TP2 population.
 * Only the local leaders attach rank-channel cursors; NCCL/RCCL disseminate
 * completed live prefixes inside their own domain. Retained graphs consume
 * recorded INT32 request inputs, with no host count readback or tensor relay.
 * Independent member poison, row-input guards, maximum-to-small/empty replays,
 * reversed leader coordinates and exact shared-header extents prove economy
 * alongside bytes. Terminal observations precede graph/channel/source retirement.
 */
#include "execution/local_execution/orchestrators/PipelineActivationExchange.h"
#include "execution/local_execution/orchestrators/PipelineActivationFanout.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "fixtures/backends/PreparedNativeLookupFixture.h"
#include "fixtures/backends/RetainedParticipantGraphExecutableFixtureAccess.h"
#include "planning/CollectiveMemoryEstimator.h"
#include "planning/PipelineTransferMemory.h"
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
    require(inventory.ranks.size() == 2, "Pipeline activation requires exactly two original ranks");
    std::array<Domain, 2> domains;
    std::set<std::string> uuids;
    std::optional<int> node;
    for (int owner = 0; owner < 2; ++owner)
    {
        const auto rank = std::find_if(inventory.ranks.begin(), inventory.ranks.end(),
            [owner](const auto &candidate) { return candidate.rank == owner; });
        require(rank != inventory.ranks.end() && rank->node_id >= 0,
            "Pipeline activation original rank/node observation is absent");
        require(!node || *node == rank->node_id, "Pipeline activation rank channels require one physical node");
        node = rank->node_id;
        const auto family = owner == 0 ? DeviceType::CUDA : DeviceType::ROCm;
        std::vector<const DeviceInfo *> owned;
        for (const auto &gpu : rank->gpus)
            if (gpu.type == family && rankOwnsGPUDevice(*rank, gpu)) owned.push_back(&gpu);
        std::sort(owned.begin(), owned.end(), [](const auto *a, const auto *b) {
            return a->local_device_id < b->local_device_id;
        });
        require(owned.size() >= 2, "Pipeline activation requires original NUMA-local CUDA TP2 on rank0 and ROCm TP2 on rank1");
        domains[owner].rank = owner;
        for (size_t member = 0; member < 2; ++member)
        {
            const auto &gpu = *owned[1 - member];
            require(!gpu.uuid.empty() && uuids.insert(gpu.uuid).second,
                "Pipeline activation domains must have four distinct physical GPU UUIDs");
            domains[owner].devices[member] = {family, gpu.local_device_id};
            requireRankLocalGPUOwnership(*rank, domains[owner].devices[member], "Pipeline activation original native domain");
        }
        require(domains[owner].devices[0].ordinal > domains[owner].devices[1].ordinal,
            "Pipeline activation leader coordinates were not reversed");
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

/**
 * @brief Exercise both original native broadcasts and both explicit rank channels.
 * @param cuda_first Select which existing domain publishes the initial activation.
 * Twenty retained cycles cover the original1/17/129/257 rows plus empty rows;
 * each maximum-to-small transition dirties all unused storage independently.
 */
void exerciseDomains(bool cuda_first)
{
    constexpr size_t capacity = 257, row_bytes = sizeof(float), payload_bytes = capacity * row_bytes;
    constexpr size_t storage_words = capacity + 2, storage_bytes = storage_words * sizeof(float);
    constexpr size_t count_bytes = 4 * sizeof(int32_t), snapshot_bytes = 2 * (storage_bytes + count_bytes);
    constexpr uint64_t forward_key = 101, backward_key = 102;
    const auto mpi = MPIContextFactory::global();
    require(mpi && mpi->world_size() == 2 && mpi->clusterInventory(),
        "Pipeline activation requires original MPI2 inventory publication");
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
    const CPUTensorStorageDeclaration hidden_declaration(1, storage_words, BufferTensorType::FP32,
        PhysicalMemoryOwner::ExecutionWorkspace, DeviceId::cpu());
    const CPUTensorStorageDeclaration rows_declaration(1, 4, BufferTensorType::INT32,
        PhysicalMemoryOwner::ExecutionWorkspace, DeviceId::cpu());
    std::unique_ptr<test::PreparedNativeLookupFixture> original;
    std::unique_ptr<ILocalTPContext> collective;
    std::shared_ptr<PhysicalMemoryAuthority> memory;
    // The canonical exchange supplies all-rank failure consensus for source/PMA
    // preparation before either process may attach a shared native channel.
    exchangePlanningSamples(mpi, [&](int count) {
        return std::vector<std::vector<uint8_t>>(count, plans[0].serialize());
    }, [&](auto wire) {
        require(std::vector<uint8_t>(wire.begin(), wire.end()) == plans[0].serialize(),
            "Pipeline activation original rank declarations disagree");
        original = test::PreparedNativeLookupFixture::prepare(devices, mpi->rank());
        std::vector<GlobalDeviceAddress> addresses;
        for (const auto device : devices) addresses.push_back(GlobalDeviceAddress::fromLocalDeviceId(device));
        const auto native_backend = devices[0].is_cuda() ? CollectiveBackendType::NCCL : CollectiveBackendType::RCCL;
        collective = createLocalTPContext(addresses, {}, native_backend);
        require(bool(collective), "Pipeline activation original native TP context is absent");
        const auto native = CollectiveMemoryEstimator::localTP(1, capacity, native_backend);
        PhysicalMemoryPlanBuilder builder;
        for (const auto &plan : plans)
            plan.contributeMemory(*plan.localRole(mpi->rank()), original->resource(devices[0]),
                plan.hostRank() == mpi->rank() ? std::optional{original->resource(DeviceId::cpu())} : std::nullopt, builder);
        builder.add(original->resource(DeviceId::cpu()), PhysicalMemoryOwner::ActivationTransportStaging,
            2 * PipelineTransferMemory::fanoutHostBytes() + TransferEngine::mappedHostRegionAllocationBytes(snapshot_bytes));
        builder.add(original->resource(DeviceId::cpu()), PhysicalMemoryOwner::ExecutionWorkspace,
            2 * (hidden_declaration.physicalBytes() + rows_declaration.physicalBytes()));
        for (const auto device : devices)
        {
            builder.add(original->resource(device), PhysicalMemoryOwner::ActivationTransportStaging,
                PipelineTransferMemory::fanoutDeviceBytes());
            builder.add(original->resource(device), PhysicalMemoryOwner::ExecutionWorkspace, storage_bytes + count_bytes);
            builder.add(original->resource(device), PhysicalMemoryOwner::LocalCollective, native.perDeviceBytes());
            const auto graph_bom = RetainedParticipantGraphExecutable::generalBOM(
                original->resource(DeviceId::cpu()), original->resource(device), 1);
            for (const auto &row : graph_bom.resources()) builder.add(row);
        }
        original->admit(builder.build()); memory = original->memory();
        require(collective->reserveCollectiveResources(native.backend_payload_capacity_bytes, native.fp16_scratch_elements, memory),
            "Pipeline activation original native resources were not admitted");
        return std::vector<uint8_t>{1};
    }, [](auto receipts) {
        for (const auto &receipt : receipts)
            require(receipt.bytes.size() == 1 && receipt.bytes[0] == 1, "Pipeline activation preparation omitted an original rank");
    });
    auto &transfer = TransferEngine::instance();
    std::array<std::shared_ptr<IWorkerGPUContext>, 2> workers{original->permit(devices[0]).worker(), original->permit(devices[1]).worker()};
    std::array<void *, 2> streams{workers[0]->defaultStream(), workers[1]->defaultStream()};
    std::array<IBackend *, 2> backends{getBackendFor(devices[0]), getBackendFor(devices[1])};
    std::array<std::shared_ptr<CapturedTransferChannel>, 2> channels;
    for (size_t direction = 0; direction < 2; ++direction)
    {
        channels[direction] = transfer.createRankCapturedTransferChannel(plans[direction], session, memory, streams[0]);
        require(bool(channels[direction]), "Pipeline activation lost its original rank channel");
        const auto own_role = *plans[direction].localRole(mpi->rank());
        const auto peer_role = own_role == CapturedTransferEndpoint::Producer
            ? CapturedTransferEndpoint::Consumer : CapturedTransferEndpoint::Producer;
        require(channels[direction]->endpointDevice(own_role) == devices[0], "Pipeline activation changed its original leader");
        require(!CapturedTransferChannelReplayProbe::terminalCursor(*channels[direction], peer_role),
            "Pipeline activation created a foreign-rank GPU cursor");
    }
    bool premature_retirement_rejected = false;
    try { session->retireAfterOriginalGraphRelease(); }
    catch (const std::exception &) { premature_retirement_rejected = true; }
    require(premature_retirement_rejected && !session->retired(), "Pipeline activation retired shared backing with live borrowers");
    std::exception_ptr failure;
    try
    {
        std::array<std::shared_ptr<TensorBase>, 2> hidden, rows;
        for (size_t member = 0; member < 2; ++member)
        {
            hidden[member] = hidden_declaration.materialize(memory);
            rows[member] = rows_declaration.materialize(memory);
            workers[member]->submitAndWait([&, member] {
                auto *values = static_cast<float *>(hidden[member]->raw_mutable_data());
                std::fill_n(values, storage_words, -91.0f);
                auto *counts = static_cast<int32_t *>(rows[member]->raw_mutable_data());
                counts[0] = 0x1357; counts[1] = capacity; counts[2] = 0x2468; counts[3] = 0x369;
                TransferEngine::publishHostWrite(hidden[member].get());
                TransferEngine::publishHostWrite(rows[member].get());
                TransferEngine::prepareDeviceInput(hidden[member].get(), devices[member], streams[member]);
                TransferEngine::prepareDeviceInput(rows[member].get(), devices[member], streams[member]);
            });
        }
        const auto incoming = channels[source ? 1 : 0];
        const auto outgoing = channels[source ? 0 : 1];
        std::array<std::shared_ptr<PipelineActivationFanout>, 2> fanout;
        std::array<std::unique_ptr<PipelineActivationExchange>, 2> receives;
        for (size_t member = 0; member < 2; ++member)
        {
            fanout[member] = PipelineActivationFanout::create(devices[member], *collective, int(member), incoming, memory, streams[member]);
            receives[member] = std::make_unique<PipelineActivationExchange>(devices[member], hidden[member],
                PipelineActivationExchange::CapturedDomain{.context = collective.get(), .participant = int(member),
                    .channel = incoming, .message = {source ? backward_key : forward_key, payload_bytes},
                    .endpoint = CapturedTransferEndpoint::Consumer,
                    .rows = PipelineActivationExchange::DeviceRows{rows[member], sizeof(int32_t), row_bytes}, .fanout = fanout[member]});
            receives[member]->setGPUStream(streams[member]);
        }
        auto send = std::make_unique<PipelineActivationExchange>(devices[0], hidden[0],
            PipelineActivationExchange::CapturedDomain{.context = collective.get(), .channel = outgoing,
                .message = {source ? forward_key : backward_key, payload_bytes},
                .rows = PipelineActivationExchange::DeviceRows{rows[0], sizeof(int32_t), row_bytes}});
        send->setGPUStream(streams[0]);
        auto snapshot_claim = memory->claimNewAllocation(DeviceId::cpu(), PhysicalMemoryOwner::ActivationTransportStaging,
            TransferEngine::mappedHostRegionAllocationBytes(snapshot_bytes));
        auto snapshot = transfer.allocateMappedHostRegion(snapshot_bytes, devices);
        std::array<std::shared_ptr<RetainedParticipantGraphExecutable>, 2> owners;
        for (size_t member = 0; member < 2; ++member)
            owners[member] = test::RetainedParticipantGraphExecutableFixtureAccess::original(original->permit(devices[member]), streams[member]);
        const auto await = [&] {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
            for (size_t member = 0; member < 2; ++member)
                for (;;)
                {
                    GPUStreamExecutionState state = GPUStreamExecutionState::Pending;
                    workers[member]->submitAndWait([&, member] {
                        state = workers[member]->queryStreamExecutionState(streams[member], "original rank TP activation terminal");
                    });
                    if (state == GPUStreamExecutionState::Complete) break;
                    require(std::chrono::steady_clock::now() < deadline, "Pipeline activation original terminal exceeded protocol deadline");
                    std::this_thread::yield();
                }
        };
        CapturedTransferInvocation active{};
        std::array<bool, 2> admitted{};
        FailureDrain drain{[&] {
            for (size_t direction = 0; direction < 2; ++direction)
                if (admitted[direction]) require(transfer.cancelCapturedTransferInvocation(channels[direction], active),
                    "Pipeline activation lost its original cancellation admission");
            require(collective->signalNativeCompletionAbort(), "Pipeline activation original native cancellation failed");
            await();
            for (size_t direction = 0; direction < 2; ++direction)
                if (admitted[direction])
                {
                    require(transfer.finishCapturedTransferInvocation(channels[direction], active) == CapturedTransferInvocationRetirement::Cancelled,
                        "Pipeline activation original cancelled invocation failed retirement");
                    admitted[direction] = false;
                }
            active = {};
        }, mpi};
        std::barrier begun(2), recorded(2), ended(2);
        std::array<bool, 2> began{}, failed{};
        std::array<std::exception_ptr, 2> errors{};
        // Retain the recording facade while construction is legal. Its const
        // observations stay available after sealing; requesting a new recording
        // capability after instantiation correctly fails the lifecycle guard.
        std::array<const IGPUGraphCapture *, 2> frozen_definitions{};
        const auto capture = [&](size_t member) {
            auto &definition = test::RetainedParticipantGraphExecutableFixtureAccess::capture(owners[member]);
            frozen_definitions[member] = &definition;
            std::unique_ptr<GraphCaptureDependencyLedger> ledger;
            std::unique_ptr<ScopedBackendGraphCapture> scope;
            try
            {
                // External upload events belong to the uncaptured admission
                // frontier. Join them on their exact consumer before recording
                // either native domain, as the production executor does.
                TransferEngine::requireDeviceInput(rows[member].get(), devices[member], streams[member]);
                if (source && member == 0)
                    TransferEngine::requireDeviceInput(hidden[member].get(), devices[member], streams[member]);
                using StagePlan = GraphCaptureDependencyLedger::StagePlan;
                std::vector<StagePlan> stages;
                if (source && member == 0)
                    stages.push_back({.stage_identity = send.get(), .stage_name = send->name(),
                        .external_inputs = {hidden[member].get(), rows[member].get()}});
                stages.push_back({.stage_identity = receives[member].get(), .stage_name = receives[member]->name(),
                    .external_inputs = {rows[member].get()}, .outputs = {hidden[member].get()}});
                if (!source && member == 0)
                    stages.push_back({.stage_identity = send.get(), .stage_name = send->name(),
                        .external_inputs = {rows[member].get()}, .internal_inputs = {{hidden[member].get(), 0}}});
                ledger = std::make_unique<GraphCaptureDependencyLedger>(devices[member], streams[member], std::move(stages),
                    "original rank TP2 activation round trip");
                scope = std::make_unique<ScopedBackendGraphCapture>(*workers[member], definition,
                    "original rank TP2 activation round trip", ledger.get());
                require(scope->begin(), "Pipeline activation original capture begin failed"); began[member] = true;
            }
            catch (...) { errors[member] = std::current_exception(); }
            begun.arrive_and_wait();
            if (began[0] && began[1]) try
            {
                auto execution = IDeviceContext::create(devices[member], 1);
                require(bool(execution), "Pipeline activation original execution context is absent");
                const auto record = [&](PipelineActivationExchange &stage) {
                    ScopedGraphCaptureStage stage_scope(&stage);
                    require(stage.execute(execution.get()), "Pipeline activation original stage capture failed");
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
                require(definition.prepareRuntimeContextStorage(memory) && definition.instantiate(),
                    "Pipeline activation original retained graph instantiation failed");
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
            for (const auto &receipt : receipts) require(receipt.bytes.size() == 1 && receipt.bytes[0] == 1, "Pipeline activation capture omitted a rank");
        });
        const std::array frozen_nodes{frozen_definitions[0]->nodeCount(), frozen_definitions[1]->nodeCount()};
        require(frozen_nodes[0] > 0 && frozen_nodes[1] > 0, "Pipeline activation original graphs are empty");
        uint64_t ordinal = 0;
        for (int replay = 0; replay < 20; ++replay)
            for (const int32_t live : {257, 17, 129, 1, 0})
            {
                SCOPED_TRACE(replay);
                SCOPED_TRACE(live);
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
                            auto *values = static_cast<float *>(hidden[member]->raw_mutable_data());
                            for (size_t word = 0; word < storage_words; ++word)
                                values[word] = source && member == 0 ? float(1000 * ordinal + word) : float(-91 - int(member));
                            require(rows[member]->hostValid(), "Pipeline activation read-only INT32 input was falsely published as output");
                            auto *counts = static_cast<int32_t *>(rows[member]->raw_mutable_data());
                            counts[1] = live;
                            TransferEngine::publishHostWrite(hidden[member].get());
                            TransferEngine::publishHostWrite(rows[member].get());
                            workers[member]->submitAndWait([&, member] {
                                TransferEngine::prepareDeviceInput(hidden[member].get(), devices[member], streams[member]);
                                TransferEngine::prepareDeviceInput(rows[member].get(), devices[member], streams[member]);
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
                                require(frozen_definitions[member]->nodeCount() == frozen_nodes[member] &&
                                    frozen_definitions[member]->executionStream() == streams[member],
                                    "Pipeline activation request reset changed the retained graph definition");
                                require(test::RetainedParticipantGraphExecutableFixtureAccess::submit(owners[member], ordinal),
                                    "Pipeline activation original retained replay submission failed");
                            });
                        await();
                        for (size_t member = 0; member < 2; ++member)
                        {
                            require(test::RetainedParticipantGraphExecutableFixtureAccess::observeTerminal(owners[member]),
                                "Pipeline activation retained graph lacks original terminal proof");
                            workers[member]->submitAndWait([&, member] {
                                auto *target = static_cast<std::byte *>(snapshot->mutableHostData()) + member * (storage_bytes + count_bytes);
                                require(backends[member]->deviceToHostOnStream(target, hidden[member]->gpu_data_ptr(), storage_bytes,
                                    devices[member].ordinal, streams[member]), "Pipeline activation terminal payload readback failed");
                                require(backends[member]->deviceToHostOnStream(target + storage_bytes, rows[member]->gpu_data_ptr(), count_bytes,
                                    devices[member].ordinal, streams[member]), "Pipeline activation terminal INT32 readback failed");
                            });
                        }
                        await();
                        for (size_t member = 0; member < 2; ++member)
                        {
                            const auto *target = static_cast<const std::byte *>(snapshot->mutableHostData()) + member * (storage_bytes + count_bytes);
                            const auto *actual = reinterpret_cast<const float *>(target);
                            for (size_t word = 0; word < storage_words; ++word)
                                require(actual[word] == (word < size_t(live) || (source && member == 0)
                                    ? float(1000 * ordinal + word) : float(-91 - int(member))),
                                    "Pipeline activation changed a live value or copied unused capacity/guard");
                            std::array<int32_t, 4> counts{};
                            std::memcpy(counts.data(), target + storage_bytes, count_bytes);
                            require(counts == std::array<int32_t, 4>{0x1357, live, 0x2468, 0x369},
                                "Pipeline activation changed recorded INT32 rows or adjacent guards");
                        }
                        for (size_t direction = 0; direction < 2; ++direction)
                        {
                            const auto &channel = channels[direction];
                            const auto &control = CapturedTransferChannelReplayProbe::terminalControl(*channel);
                            require(control.producer.message.bytes == size_t(live) * row_bytes &&
                                control.consumer.message.bytes == size_t(live) * row_bytes &&
                                control.producer.epoch == ordinal && control.consumer.epoch == ordinal,
                                "Pipeline activation communicated capacity, stale extent or incomplete epochs");
                            require(transfer.finishCapturedTransferInvocation(channel, active) == CapturedTransferInvocationRetirement::Completed,
                                "Pipeline activation original request retirement failed");
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
                        require(receipt.bytes.size() == 1 && receipt.bytes[0] == 1, "Pipeline activation terminal evidence omitted a rank");
                });
            }
        ::testing::Test::RecordProperty("pipeline_activation_rank_domains", "originalMPI2;CUDA_TP2_rank0;ROCm_TP2_rank1;reversed_leaders;100_retained_replays;device_INT32_live_prefix");
        drain.completed = true;
    }
    catch (...) { failure = std::current_exception(); }
    // FailureDrain cancelled in-flight invocations/native operations and proved
    // the exact local streams retired before graph/stage/channel destruction.
    // Both original ranks retain the backing until the same closed boundary.
    mpi->barrier();
    channels = {};
    session->retireAfterOriginalGraphRelease();
    require(session->retired(), "Pipeline activation shared original rank session did not retire");
    collective.reset(); workers = {};
    original->requirePayloadRetired(); original->retire(); original->requireExactRetired();
    if (failure) std::rethrow_exception(failure);
}
}

/** @test Real CUDA TP2 output reaches both original ROCm TP2 members and returns. */
TEST(PipelineActivationDomains, CUDA2_ROCm2_CapturedRoundTrip) { exerciseDomains(true); }
/** @test Reverse vendor order preserves both native domains and original rank ownership. */
TEST(PipelineActivationDomains, ROCm2_CUDA2_CapturedRoundTrip) { exerciseDomains(false); }
