/**
 * @file Test__PipelineMetadataDomain.cpp
 * @brief Exact captured MTP metadata across real CUDA and ROCm TP domains.
 *
 * Four participant-local graphs exchange condition, verifier and committed-state
 * banks through the production collective. Only domain leaders touch the wire;
 * every receiving TP member participates in NCCL/RCCL. Guards catch oversized
 * copies, request-like alternating payloads expose stale epochs, and retained
 * workspace metadata survives namespace retirement before capture begins.
 */
#include "execution/local_execution/orchestrators/PipelineMetadataExchange.h"
#include "execution/local_execution/orchestrators/TPWorkerPool.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "collective/LocalTPContext.h"
#include "planning/CollectiveMemoryEstimator.h"
#include "planning/PlanningObservedResource.h"
#include "tensors/Tensors.h"
#include "utils/MPIContext.h"
#include <gtest/gtest.h>
#include <array>
#include <chrono>
#include <thread>

using namespace llaminar2;
namespace
{
/** @brief Fail before submitting a dependent captured operation. */
void require(bool ok, const char *reason) { if (!ok) throw std::runtime_error(reason); }

/** @brief Join setup workers, never synchronize an inference stream between fields. */
void join(TPWorkerPool &workers)
{
    for (const auto &result : workers.collectAll(30000))
    {
        require(result.completed, "Metadata graph worker did not join");
        if (result.exception) std::rethrow_exception(result.exception);
        require(result.success, "Metadata graph worker failed");
    }
}

/** @brief One explicitly joined terminal event per participant. */
struct Completion
{
    IBackend *backend;
    DeviceId device;
    void *event;
    /** @brief Release only after terminal work has been observed. */
    ~Completion() { if (event) backend->destroyEvent(event, device.ordinal); }
    /** @brief Bound the final observation and surface asynchronous native errors. */
    void await() const
    {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(35);
        bool ready = false;
        while (!ready)
        {
            require(backend->queryEvent(event, device.ordinal, &ready), "Metadata terminal query failed");
            require(ready || std::chrono::steady_clock::now() < deadline, "Metadata terminal deadline expired");
            if (!ready) std::this_thread::yield();
        }
    }
};

/** @brief Exercise one terminal/follower vendor order without host metadata relay. */
void exercise(bool cuda_first)
{
    using Layout = PipelineMetadataLayout;
    using Kind = Layout::Kind;
    constexpr size_t words = 128, tensor_bytes = words * sizeof(int32_t);
    constexpr size_t workspace_words = 32, workspace_bytes = workspace_words * sizeof(int32_t);
    constexpr size_t io_bytes = tensor_bytes + workspace_bytes, capacity = 16 * sizeof(int32_t);
    const auto mpi = MPIContextFactory::global();
    const auto &rank = mpi->clusterInventory()->ranks.at(mpi->rank());
    const std::array devices = cuda_first
        ? std::array{DeviceId::cuda(1), DeviceId::cuda(0), DeviceId::rocm(1), DeviceId::rocm(0)}
        : std::array{DeviceId::rocm(1), DeviceId::rocm(0), DeviceId::cuda(1), DeviceId::cuda(0)};
    const auto channel_memory = CapturedTransferChannel::memoryFor(capacity);
    const auto mapped_bytes = TransferEngine::mappedHostRegionAllocationBytes(io_bytes);
    PhysicalMemoryPlanBuilder builder;
    builder.add(planningObservedResource(rank, DeviceId::cpu()), PhysicalMemoryOwner::ActivationTransportStaging,
        2 * channel_memory.mapped_host_bytes + devices.size() * 2 * mapped_bytes);
    for (size_t i = 0; i < devices.size(); ++i)
    {
        ASSERT_NE(getBackendFor(devices[i]), nullptr);
        ASSERT_GT(getBackendFor(devices[i])->deviceCount(), devices[i].ordinal);
        const auto resource = planningObservedResource(rank, devices[i]);
        builder.add(resource, PhysicalMemoryOwner::ActivationTransportStaging,
            tensor_bytes + (i % 2 == 0 ? 2 * channel_memory.cursor_bytes_per_device : 0));
        builder.add(resource, PhysicalMemoryOwner::ExecutionWorkspace, workspace_bytes);
        builder.add(resource, PhysicalMemoryOwner::LocalCollective,
            CollectiveMemoryEstimator::nativePipelineBoundaryBytes(devices[i].is_cuda()
                ? CollectiveBackendType::NCCL : CollectiveBackendType::RCCL));
        builder.add(resource, PhysicalMemoryOwner::NativeGraphExecutable,
            GPUGraphMemoryContract::reservationBytesPerExecutable(devices[i]));
    }
    auto authority = std::make_shared<PhysicalMemoryAuthority>(
        std::make_shared<const PhysicalMemoryPlanAdmissionCertificate>(builder.build()), rank.rank);
    auto &transfer = TransferEngine::instance();
    {
        LocalTPContext first({GlobalDeviceAddress::fromLocalDeviceId(devices[0]),
            GlobalDeviceAddress::fromLocalDeviceId(devices[1])}, {}, CollectiveBackendType::AUTO);
        LocalTPContext second({GlobalDeviceAddress::fromLocalDeviceId(devices[2]),
            GlobalDeviceAddress::fromLocalDeviceId(devices[3])}, {}, CollectiveBackendType::AUTO);
        ASSERT_TRUE(first.reserveGraphCaptureBoundaryResources(authority));
        ASSERT_TRUE(second.reserveGraphCaptureBoundaryResources(authority));
        TPWorkerPool workers(devices.size());
        workers.setFailureCallback([&] { first.requestAbort(); second.requestAbort(); });
        std::array<IWorkerGPUContext *, 4> contexts;
        std::array<void *, 4> streams;
        for (size_t i = 0; i < devices.size(); ++i)
        {
            contexts[i] = &GPUDeviceContextPool::instance().getContext(devices[i]);
            contexts[i]->submitAndWait([&, i] {
                streams[i] = contexts[i]->getOrCreateAuxiliaryStream("pipeline_metadata_fixture");
            });
        }
        auto forward = transfer.createCapturedTransferChannel(*authority, DeviceId::cpu(),
            devices[0], streams[0], devices[2], streams[2], capacity);
        auto backward = transfer.createCapturedTransferChannel(*authority, DeviceId::cpu(),
            devices[2], streams[2], devices[0], streams[0], capacity);
        auto host_claim = authority->claimNewAllocation(DeviceId::cpu(), PhysicalMemoryOwner::ActivationTransportStaging,
            devices.size() * 2 * mapped_bytes);
        std::array<PhysicalMemoryAllocationLease, 4> tensor_claims;
        std::array<std::shared_ptr<INT32Tensor>, 4> tensors;
        std::array<std::shared_ptr<const WorkspaceBufferLease>, 4> workspaces;
        std::array<std::shared_ptr<MappedHostTransferRegion>, 4> inputs, outputs;
        std::array<std::unique_ptr<Completion>, 4> completions;
        for (size_t i = 0; i < devices.size(); ++i)
        {
            const auto device = devices[i];
            tensor_claims[i] = authority->claimNewAllocation(device, PhysicalMemoryOwner::ActivationTransportStaging, tensor_bytes);
            tensors[i] = std::make_shared<INT32Tensor>(std::vector<size_t>{words});
            TransferEngine::prepareDeviceInput(tensors[i].get(), device, streams[i]);
            DeviceWorkspaceManager manager(device, workspace_bytes, authority);
            WorkspaceRequirements requirements;
            requirements.buffers.push_back({"accepted_state_slot_indices", workspace_bytes, sizeof(int32_t), true});
            ASSERT_TRUE(manager.allocate(requirements));
            workspaces[i] = manager.retainBuffer("accepted_state_slot_indices", 0, workspace_bytes);
            manager.release(); // Capture may retain only the original allocation, not a live manager.
            inputs[i] = transfer.allocateMappedHostRegion(io_bytes, std::array{device});
            outputs[i] = transfer.allocateMappedHostRegion(io_bytes, std::array{device});
            auto *backend = getBackendFor(device);
            completions[i] = std::make_unique<Completion>(backend, device, backend->createEvent(device.ordinal));
            ASSERT_NE(completions[i]->event, nullptr);
        }
        for (const auto layout : {Layout(Kind::Condition, 1), Layout(Kind::Verifier, 2),
                 Layout(Kind::Verifier, 3), Layout(Kind::Verifier, 4), Layout(Kind::Verifier, 16),
                 Layout(Kind::CommittedState, 1), Layout(Kind::NextToken, 1)})
        {
            SCOPED_TRACE(layout.name());
            std::array<std::unique_ptr<PipelineMetadataExchange>, 4> receives;
            std::array<std::unique_ptr<PipelineMetadataExchange>, 2> sends;
            for (size_t i = 0; i < devices.size(); ++i)
            {
                std::vector<PipelineMetadataBank> banks;
                for (size_t field = 0; field < layout.fieldCount(); ++field)
                    if (layout.kind() == Kind::CommittedState && field == 0)
                        banks.emplace_back(workspaces[i], sizeof(int32_t), 1);
                    else
                        banks.emplace_back(tensors[i], BufferId::MTP_LOGICAL_SEQUENCE_STATE,
                            (field * 32 + 1) * sizeof(int32_t), layout.elements(field));
                auto &group = i < 2 ? first : second;
                receives[i] = std::make_unique<PipelineMetadataExchange>(devices[i], layout,
                    PipelineMetadataExchange::CapturedDomain{.context = &group, .participant = int(i % 2),
                        .inbound = i < 2 ? backward : forward}, banks);
                if (i % 2 == 0)
                    sends[i / 2] = std::make_unique<PipelineMetadataExchange>(devices[i], layout,
                        PipelineMetadataExchange::CapturedDomain{.context = &group,
                            .outbound = i < 2 ? forward : backward}, std::move(banks));
            }
            std::array<PhysicalMemoryOwnerReservation, 4> graph_claims;
            std::array<std::unique_ptr<IGPUGraphCapture>, 4> graphs;
            for (size_t i = 0; i < devices.size(); ++i)
                graph_claims[i] = authority->reserveNewAllocations(devices[i], PhysicalMemoryOwner::NativeGraphExecutable,
                    GPUGraphMemoryContract::reservationBytesPerExecutable(devices[i]));
            workers.dispatch([&](size_t i) {
                contexts[i]->submitAndWait([&, i] {
                    auto &group = i < 2 ? first : second;
                    auto execution = IDeviceContext::create(devices[i], 1);
                    require(bool(execution), "Metadata device context missing");
                    receives[i]->setGPUStream(streams[i]);
                    if (i % 2 == 0) sends[i / 2]->setGPUStream(streams[i]);
                    using StagePlan = GraphCaptureDependencyLedger::StagePlan;
                    std::vector<StagePlan> stages;
                    if (i == 0)
                    {
                        TransferEngine::requireDeviceInput(tensors[i].get(), devices[i], streams[i]);
                        stages.push_back({.stage_identity = sends[0].get(), .stage_name = "terminal metadata send",
                            .external_inputs = {tensors[i].get()}});
                    }
                    stages.push_back({.stage_identity = receives[i].get(), .stage_name = "local metadata receive",
                        .outputs = {tensors[i].get()}});
                    if (i == 2)
                        stages.push_back({.stage_identity = sends[1].get(), .stage_name = "metadata reply",
                            .internal_inputs = {{tensors[i].get(), 0}}});
                    GraphCaptureDependencyLedger ledger(devices[i], streams[i], std::move(stages), "pipeline metadata round trip");
                    const auto record = [&](PipelineMetadataExchange &stage) {
                        ScopedGraphCaptureStage scope(&stage);
                        require(stage.execute(execution.get()), "Metadata stage capture failed");
                        scope.complete();
                    };
                    require(group.graphCaptureBoundaryOnStream("metadata_begin", int(i % 2), streams[i], 30000),
                        "Metadata capture begin rendezvous failed");
                    graphs[i] = contexts[i]->createGraphCapture(streams[i]);
                    ScopedBackendGraphCapture capture(*contexts[i], *graphs[i], "pipeline metadata", &ledger);
                    require(capture.begin(), "Metadata capture begin failed");
                    if (i == 0) record(*sends[0]);
                    record(*receives[i]);
                    if (i == 2) record(*sends[1]);
                    capture.finish();
                    require(group.graphCaptureBoundaryOnStream("metadata_end", int(i % 2), streams[i], 30000),
                        "Metadata capture end rendezvous failed");
                    require(graphs[i]->instantiate(), "Metadata graph instantiate failed");
                });
                return true;
            });
            join(workers);
            for (int replay = 0; replay < 8; ++replay)
            {
                SCOPED_TRACE(replay);
                for (size_t i = 0; i < devices.size(); ++i)
                {
                    auto *values = static_cast<int32_t *>(inputs[i]->mutableHostData());
                    for (size_t word = 0; word < io_bytes / sizeof(int32_t); ++word)
                        values[word] = i == 0 ? int32_t(1000 * replay + word) : -91;
                    auto *backend = getBackendFor(devices[i]);
                    require(backend->hostToDevice(tensors[i]->gpu_data_ptr(), values, tensor_bytes,
                        devices[i].ordinal, streams[i]), "Metadata tensor upload failed");
                    require(backend->hostToDevice(workspaces[i]->data(), values + words, workspace_bytes,
                        devices[i].ordinal, streams[i]), "Metadata workspace upload failed");
                }
                for (size_t slot = 0; slot < devices.size(); ++slot)
                {
                    const size_t i = replay % 2 ? devices.size() - 1 - slot : slot;
                    contexts[i]->submitAndWait([&, i] {
                        require(graphs[i]->launch(), "Metadata captured replay failed");
                        TransferEngine::publishDeviceWrite(tensors[i].get(), devices[i], streams[i]);
                        auto *backend = getBackendFor(devices[i]);
                        auto *output = static_cast<int32_t *>(outputs[i]->mutableHostData());
                        require(backend->deviceToHostOnStream(output, tensors[i]->gpu_data_ptr(), tensor_bytes,
                            devices[i].ordinal, streams[i]), "Metadata tensor terminal read failed");
                        require(backend->deviceToHostOnStream(output + words, workspaces[i]->data(), workspace_bytes,
                            devices[i].ordinal, streams[i]), "Metadata workspace terminal read failed");
                        require(backend->recordEvent(completions[i]->event, devices[i].ordinal, streams[i]),
                            "Metadata terminal event failed");
                    });
                }
                for (const auto &completion : completions) completion->await();
                for (size_t i = 0; i < devices.size(); ++i)
                    for (size_t word = 0; word < io_bytes / sizeof(int32_t); ++word)
                    {
                        bool transferred = false;
                        for (size_t field = 0; field < layout.fieldCount(); ++field)
                        {
                            const size_t start = layout.kind() == Kind::CommittedState && field == 0 ? words + 1 : field * 32 + 1;
                            transferred |= word >= start && word - start < layout.elements(field);
                        }
                        const auto expected = i == 0 || transferred ? int32_t(1000 * replay + word) : -91;
                        ASSERT_EQ(static_cast<const int32_t *>(outputs[i]->mutableHostData())[word], expected)
                            << "device=" << devices[i].toString() << " word=" << word;
                    }
            }
        }
    }
    for (const auto device : devices)
    {
        EXPECT_EQ(authority->claimedBytes(device, PhysicalMemoryOwner::ActivationTransportStaging,
            PhysicalMemoryMaterializationKind::NewAllocation), 0u);
        EXPECT_EQ(authority->claimedBytes(device, PhysicalMemoryOwner::ExecutionWorkspace,
            PhysicalMemoryMaterializationKind::NewAllocation), 0u);
        EXPECT_EQ(authority->claimedBytes(device, PhysicalMemoryOwner::LocalCollective,
            PhysicalMemoryMaterializationKind::NewAllocation), 0u);
        EXPECT_EQ(authority->reservedBytes(device, PhysicalMemoryOwner::NativeGraphExecutable), 0u);
    }
    EXPECT_EQ(authority->claimedBytes(DeviceId::cpu(), PhysicalMemoryOwner::ActivationTransportStaging,
        PhysicalMemoryMaterializationKind::NewAllocation), 0u);
}
}

/** @test CUDA terminal metadata reaches every ROCm TP member and returns exactly. */
TEST(PipelineMetadataDomains, CUDA2_ROCm2_CapturedBanks) { exercise(true); }
/** @test ROCm terminal metadata follows the identical ownership/capture contract. */
TEST(PipelineMetadataDomains, ROCm2_CUDA2_CapturedBanks) { exercise(false); }
