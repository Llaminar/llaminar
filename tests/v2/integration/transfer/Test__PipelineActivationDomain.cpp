/**
 * @file Test__PipelineActivationDomain.cpp
 * @brief Captured activation proof across two real native TP domains.
 *
 * Four independent GPU graphs keep NCCL/RCCL inside each two-device domain.
 * Only leaders cross the vendor boundary through TransferEngine. A round trip
 * reaches every destination member and returns through the opposite domain,
 * exercising native broadcasts on both sides without a host activation relay.
 * Exact physical rows, guards, repeated replay and reversed leader ordinals
 * distinguish the protocol from an accidental full-buffer or leader-only copy.
 */
#include "execution/local_execution/orchestrators/PipelineActivationExchange.h"
#include "execution/local_execution/orchestrators/TPWorkerPool.h"
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
/** @brief Fail before a dependent stage can be recorded/submitted. */
void require(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }

/** @brief Join setup submission threads, never synchronize a live inference stream. */
void join(TPWorkerPool &workers)
{
    for (const auto &result : workers.collectAll(30000))
    {
        require(result.completed, "Pipeline domain fixture worker did not complete");
        if (result.exception) std::rethrow_exception(result.exception);
        require(result.success, "Pipeline domain fixture worker failed");
    }
}

/** @brief Terminal event owner used only after every device has been submitted. */
struct Completion
{
    IBackend *backend;
    DeviceId device;
    void *event;
    /** @brief Destroy the terminal handle after its final observed completion. */
    ~Completion() { if (event) backend->destroyEvent(event, device.ordinal); }
    /** @brief Reject native errors or missing progress within the protocol deadline. */
    void await() const
    {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(35);
        bool ready = false;
        while (!ready)
        {
            require(backend->queryEvent(event, device.ordinal, &ready), "Pipeline domain terminal query failed");
            require(ready || std::chrono::steady_clock::now() < deadline, "Pipeline domain terminal deadline expired");
            if (!ready) std::this_thread::yield();
        }
    }
};

/** @brief Prove two complete native groups in either vendor order, with no host relay. */
void exerciseDomains(bool cuda_first)
{
    constexpr size_t capacity = 257, buffer_bytes = capacity * sizeof(float);
    const auto mpi = MPIContextFactory::global();
    const auto &rank = mpi->clusterInventory()->ranks.at(mpi->rank());
    // Leaders are deliberately ordinal one, not zero or sorted inventory order.
    const std::array devices = cuda_first
        ? std::array{DeviceId::cuda(1), DeviceId::cuda(0), DeviceId::rocm(1), DeviceId::rocm(0)}
        : std::array{DeviceId::rocm(1), DeviceId::rocm(0), DeviceId::cuda(1), DeviceId::cuda(0)};
    for (const auto device : devices)
    {
        ASSERT_NE(getBackendFor(device), nullptr);
        ASSERT_GT(getBackendFor(device)->deviceCount(), device.ordinal);
    }
    const auto channel_memory = CapturedTransferChannel::memoryFor(buffer_bytes);
    const auto mapped_bytes = TransferEngine::mappedHostRegionAllocationBytes(buffer_bytes);
    PhysicalMemoryPlanBuilder builder;
    builder.add(planningObservedResource(rank, DeviceId::cpu()), PhysicalMemoryOwner::ActivationTransportStaging,
        2 * channel_memory.mapped_host_bytes + devices.size() * 2 * mapped_bytes);
    for (size_t member = 0; member < devices.size(); ++member)
    {
        const auto resource = planningObservedResource(rank, devices[member]);
        builder.add(resource, PhysicalMemoryOwner::ActivationTransportStaging,
            buffer_bytes + (member % 2 == 0 ? 2 * channel_memory.cursor_bytes_per_device : 0));
        builder.add(resource, PhysicalMemoryOwner::LocalCollective,
            CollectiveMemoryEstimator::nativePipelineBoundaryBytes(devices[member].is_cuda()
                ? CollectiveBackendType::NCCL : CollectiveBackendType::RCCL));
        builder.add(resource, PhysicalMemoryOwner::NativeGraphExecutable,
            GPUGraphMemoryContract::reservationBytesPerExecutable(devices[member]));
    }
    auto authority = std::make_shared<PhysicalMemoryAuthority>(
        std::make_shared<const PhysicalMemoryPlanAdmissionCertificate>(builder.build()), rank.rank);
    auto &transfer = TransferEngine::instance();
    {
        LocalTPContext source({GlobalDeviceAddress::fromLocalDeviceId(devices[0]),
            GlobalDeviceAddress::fromLocalDeviceId(devices[1])}, {}, CollectiveBackendType::AUTO);
        LocalTPContext destination({GlobalDeviceAddress::fromLocalDeviceId(devices[2]),
            GlobalDeviceAddress::fromLocalDeviceId(devices[3])}, {}, CollectiveBackendType::AUTO);
        ASSERT_TRUE(source.reserveGraphCaptureBoundaryResources(authority));
        ASSERT_TRUE(destination.reserveGraphCaptureBoundaryResources(authority));
        TPWorkerPool workers(devices.size());
        workers.setFailureCallback([&] { source.requestAbort(); destination.requestAbort(); });
        std::array<IWorkerGPUContext *, 4> contexts;
        std::array<void *, 4> streams;
        for (size_t i = 0; i < devices.size(); ++i)
        {
            contexts[i] = &GPUDeviceContextPool::instance().getContext(devices[i]);
            contexts[i]->submitAndWait([&, i] { streams[i] = contexts[i]->getOrCreateAuxiliaryStream("pipeline_domain_fixture"); });
            ASSERT_NE(streams[i], nullptr);
        }
        auto forward = transfer.createCapturedTransferChannel(*authority, DeviceId::cpu(),
            devices[0], streams[0], devices[2], streams[2], buffer_bytes);
        auto backward = transfer.createCapturedTransferChannel(*authority, DeviceId::cpu(),
            devices[2], streams[2], devices[0], streams[0], buffer_bytes);
        auto host_claim = authority->claimNewAllocation(DeviceId::cpu(), PhysicalMemoryOwner::ActivationTransportStaging,
            devices.size() * 2 * mapped_bytes);
        std::array<PhysicalMemoryAllocationLease, 4> bank_claims;
        std::array<std::shared_ptr<FP32Tensor>, 4> banks;
        std::array<std::shared_ptr<MappedHostTransferRegion>, 4> inputs, outputs;
        std::array<std::unique_ptr<Completion>, 4> complete;
        for (size_t i = 0; i < devices.size(); ++i)
        {
            const auto device = devices[i];
            bank_claims[i] = authority->claimNewAllocation(device, PhysicalMemoryOwner::ActivationTransportStaging, buffer_bytes);
            banks[i] = std::make_shared<FP32Tensor>(std::vector<size_t>{capacity});
            TransferEngine::prepareDeviceInput(banks[i].get(), device, streams[i]);
            inputs[i] = transfer.allocateMappedHostRegion(buffer_bytes, std::array{device});
            outputs[i] = transfer.allocateMappedHostRegion(buffer_bytes, std::array{device});
            auto *backend = getBackendFor(device);
            complete[i] = std::make_unique<Completion>();
            complete[i]->backend = backend;
            complete[i]->device = device;
            complete[i]->event = backend->createEvent(device.ordinal);
            ASSERT_NE(complete[i]->event, nullptr);
        }
        for (const size_t elements : {1u, 17u, 129u, 257u})
        {
            SCOPED_TRACE(elements);
            std::array<std::unique_ptr<PipelineActivationExchange>, 4> receives;
            std::array<std::unique_ptr<PipelineActivationExchange>, 2> sends;
            for (size_t i = 0; i < devices.size(); ++i)
                receives[i] = std::make_unique<PipelineActivationExchange>(devices[i], banks[i],
                    PipelineActivationExchange::CapturedDomain{.context = i < 2 ? &source : &destination,
                        .participant = static_cast<int>(i % 2), .channel = i < 2 ? backward : forward,
                        .message = {i < 2 ? 102u : 101u, elements * sizeof(float)},
                        .endpoint = CapturedTransferEndpoint::Consumer});
            for (size_t domain = 0; domain < 2; ++domain)
                sends[domain] = std::make_unique<PipelineActivationExchange>(devices[domain * 2], banks[domain * 2],
                    PipelineActivationExchange::CapturedDomain{.context = domain == 0 ? &source : &destination,
                        .channel = domain == 0 ? forward : backward,
                        .message = {domain == 0 ? 101u : 102u, elements * sizeof(float)}});
            std::array<PhysicalMemoryOwnerReservation, 4> graph_claims;
            std::array<std::unique_ptr<IGPUGraphCapture>, 4> graphs;
            for (size_t i = 0; i < devices.size(); ++i)
                graph_claims[i] = authority->reserveNewAllocations(devices[i], PhysicalMemoryOwner::NativeGraphExecutable,
                    GPUGraphMemoryContract::reservationBytesPerExecutable(devices[i]));
            workers.dispatch([&](size_t i) {
                contexts[i]->submitAndWait([&, i] {
                    auto &group = i < 2 ? source : destination;
                    const auto device = devices[i];
                    auto execution = IDeviceContext::create(device, 1);
                    require(bool(execution), "Pipeline domain context missing");
                    receives[i]->setGPUStream(streams[i]);
                    if (i % 2 == 0) sends[i / 2]->setGPUStream(streams[i]);
                    // Use the production dependency ledger: recording receive
                    // does not make its bytes externally ready. The following
                    // send consumes an internal graph producer on this stream.
                    using StagePlan = GraphCaptureDependencyLedger::StagePlan;
                    std::vector<StagePlan> stages;
                    if (i < 2)
                    {
                        stages.push_back({.stage_identity = &source, .stage_name = "source native activation",
                            .external_inputs = i == 0 ? std::vector<const TensorBase *>{banks[i].get()}
                                                      : std::vector<const TensorBase *>{},
                            .outputs = {banks[i].get()}});
                        if (i == 0)
                        {
                            TransferEngine::requireDeviceInput(banks[i].get(), device, streams[i]);
                            stages.push_back({.stage_identity = sends[0].get(), .stage_name = sends[0]->name(),
                                .internal_inputs = {{banks[i].get(), 0}}});
                        }
                    }
                    stages.push_back({.stage_identity = receives[i].get(), .stage_name = receives[i]->name(),
                        .outputs = {banks[i].get()}});
                    if (i == 2)
                        stages.push_back({.stage_identity = sends[1].get(), .stage_name = sends[1]->name(),
                            .internal_inputs = {{banks[i].get(), 0}}});
                    GraphCaptureDependencyLedger ledger(device, streams[i], std::move(stages), "pipeline domain round trip");
                    const auto record_stage = [&](PipelineActivationExchange &stage) {
                        ScopedGraphCaptureStage scope(&stage);
                        require(stage.execute(execution.get()), "Pipeline activation stage capture failed");
                        scope.complete();
                    };
                    const auto boundary = "pipeline_domain_" + std::to_string(elements);
                    require(group.graphCaptureBoundaryOnStream(boundary + "_begin", int(i % 2), streams[i], 30000),
                        "Pipeline domain capture begin rendezvous failed");
                    graphs[i] = contexts[i]->createGraphCapture(streams[i]);
                    require(bool(graphs[i]), "Pipeline domain capture unavailable");
                    {
                        ScopedBackendGraphCapture capture(*contexts[i], *graphs[i], "pipeline domain round trip", &ledger);
                        require(capture.begin(), "Pipeline domain capture failed");
                        if (i < 2)
                        {
                            // Stand in for a TP model's replicated final output.
                            // Only the domain leader subsequently reads it for send.
                            {
                                ScopedGraphCaptureStage scope(&source);
                                if (i == 0) TransferEngine::requireDeviceInput(banks[i].get(), device, streams[i]);
                                else TransferEngine::requireDeviceOutput(banks[i].get(), device, streams[i]);
                                require(source.broadcastRawOnStream(banks[i]->gpu_data_ptr(), banks[i]->gpu_data_ptr(), elements,
                                    CollectiveDataType::FLOAT32, 0, int(i), streams[i], "source replicated activation"),
                                    "Source native broadcast failed");
                                TransferEngine::publishDeviceWrite(banks[i].get(), device, streams[i]);
                                scope.complete();
                            }
                            if (i == 0) record_stage(*sends[0]);
                            record_stage(*receives[i]);
                        }
                        else
                        {
                            record_stage(*receives[i]);
                            if (i == 2) record_stage(*sends[1]);
                        }
                        capture.finish();
                    }
                    require(group.graphCaptureBoundaryOnStream(boundary + "_end", int(i % 2), streams[i], 30000),
                        "Pipeline domain capture end rendezvous failed");
                    require(graphs[i]->instantiate(), "Pipeline domain graph instantiation failed");
                });
                return true;
            });
            join(workers);
            for (int replay = 0; replay < 20; ++replay)
            {
                SCOPED_TRACE(replay);
                for (size_t i = 0; i < devices.size(); ++i)
                {
                    auto *values = static_cast<float *>(inputs[i]->mutableHostData());
                    for (size_t word = 0; word < capacity; ++word)
                        values[word] = i == 0 ? float(1000 * replay + word) : -91.0f;
                    require(getBackendFor(devices[i])->hostToDevice(banks[i]->gpu_data_ptr(), values, buffer_bytes,
                        devices[i].ordinal, streams[i]), "Pipeline fixture input upload failed");
                }
                // Alternating order makes both a receive-first and a send-first
                // schedule wait on real device epochs. No intermediate host join.
                for (size_t slot = 0; slot < devices.size(); ++slot)
                {
                    const size_t i = replay % 2 ? devices.size() - 1 - slot : slot;
                    contexts[i]->submitAndWait([&, i] {
                        require(graphs[i]->launch(), "Pipeline domain replay failed");
                        TransferEngine::publishDeviceWrite(banks[i].get(), devices[i], streams[i]);
                        auto *backend = getBackendFor(devices[i]);
                        require(backend->deviceToHostOnStream(outputs[i]->mutableHostData(), banks[i]->gpu_data_ptr(), buffer_bytes,
                            devices[i].ordinal, streams[i]), "Pipeline terminal readback failed");
                        require(backend->recordEvent(complete[i]->event, devices[i].ordinal, streams[i]), "Pipeline terminal event failed");
                    });
                }
                for (const auto &terminal : complete) terminal->await();
                for (size_t i = 0; i < devices.size(); ++i)
                {
                    const auto *actual = static_cast<const float *>(outputs[i]->mutableHostData());
                    for (size_t word = 0; word < capacity; ++word)
                        ASSERT_EQ(actual[word], i == 0 || word < elements ? float(1000 * replay + word) : -91.0f)
                            << "device=" << devices[i].toString() << " word=" << word;
                }
            }
        }
    }
    for (const auto device : {DeviceId::cpu(), devices[0], devices[1], devices[2], devices[3]})
        EXPECT_EQ(authority->claimedBytes(device, PhysicalMemoryOwner::ActivationTransportStaging,
            PhysicalMemoryMaterializationKind::NewAllocation), 0u);
    for (const auto device : devices)
    {
        EXPECT_EQ(authority->claimedBytes(device, PhysicalMemoryOwner::LocalCollective,
            PhysicalMemoryMaterializationKind::NewAllocation), 0u);
        EXPECT_EQ(authority->reservedBytes(device, PhysicalMemoryOwner::NativeGraphExecutable), 0u);
    }
}
}

/** @test CUDA TP output reaches both ROCm TP members and returns in retained graphs. */
TEST(PipelineActivationDomains, CUDA2_ROCm2_CapturedRoundTrip) { exerciseDomains(true); }
/** @test Reverse vendor ordering does not invent a CUDA-only continuation assumption. */
TEST(PipelineActivationDomains, ROCm2_CUDA2_CapturedRoundTrip) { exerciseDomains(false); }
