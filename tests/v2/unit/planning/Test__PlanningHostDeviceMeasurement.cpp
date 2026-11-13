/**
 * @file Test__PlanningHostDeviceMeasurement.cpp
 * @brief Device-free ownership, exact-BOM and complete-mechanism evidence regressions.
 *
 * Synthetic times are codec/math witnesses only. No backend, mapped allocation,
 * kernel or MPI operation is initialized by these tests.
 */
#include "planning/PlanningHostDeviceMeasurement.h"
#include "planning/PlanningExecutionMeasurement.h"
#include "planning/PlanningGPUClockBuffer.h"
#include "backends/GPUGraphMemoryContract.h"
#include "transfer/TransferEngine.h"
#include "../../mocks/MockWorkerGPUContext.h"
#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>
#include <unistd.h>

using namespace llaminar2;

namespace
{
    /** @return Distinct original GPU owner and first-touch host on sparse NUMA IDs. */
    ClusterInventory inventory()
    {
        ClusterInventory cluster;
        cluster.world_size = 5;
        for (int index = 0; index < cluster.world_size; ++index)
        {
            RankInventory rank;
            rank.rank = index;
            rank.node_id = 2;
            rank.cpu.numa_node = index == 3 ? 11 : 7;
            rank.cpu.memory_bytes = rank.cpu.free_memory_bytes = 1ull << 30;
            rank.cpu_cores = 2;
            if (index == 3)
                for (auto type : {DeviceType::CUDA, DeviceType::ROCm})
                    rank.gpus.push_back({.type = type, .local_device_id = 5, .memory_bytes = 1ull << 30,
                        .free_memory_bytes = 1ull << 30, .uuid = "physical-gpu", .numa_node = 11});
            cluster.ranks.push_back(std::move(rank));
        }
        return cluster;
    }
    /** @brief Strict retained readback double: one launch is the complete byte proof. */
    class ByteProofGraph final : public IGPUGraphCapture
    {
    public:
        void *stream=nullptr;
        bool executable=true,fail_launch=false;
        mutable int launches=0;
        bool beginCapture() override {ADD_FAILURE();return false;}
        bool endCapture() override {ADD_FAILURE();return false;}
        bool instantiate() override {ADD_FAILURE();return false;}
        bool launch() override {ADD_FAILURE();return false;}
        bool launchOnStream(void *actual) const override {
            EXPECT_EQ(actual,stream);++launches;return !fail_launch;
        }
        void *executionStream() const noexcept override {return stream;}
        GraphUpdateResult tryUpdate() override {ADD_FAILURE();return GraphUpdateResult::Failed;}
        bool supportsExecutableUpdate() const noexcept override {return false;}
        bool hasExecutable() const override {return executable;}
        size_t residentMemoryBytes() const noexcept override {return 0;}
        size_t nodeCount() const override {return 1;}
        void reset() override {ADD_FAILURE();}
        /** @return Device-free identity; this mock never loads a native backend. */
        const char *backendName() const override {return "ByteProofMock";}
    };
    /** @brief Events only order byte inspection; timestamp and blocking APIs are forbidden. */
    class ByteProofWorker final : public llaminar2::testing::MockWorkerGPUContext
    {
    public:
        ByteProofWorker():MockWorkerGPUContext(5){}
        int token=0,creates=0,destroys=0,records=0,queries=0;
        bool fail_create=false,fail_record=false,fail_query=false;
        void *createEvent(GPUEventPurpose purpose) override {
            EXPECT_TRUE(ownsCurrentThread());EXPECT_EQ(purpose,GPUEventPurpose::Ordering);
            ++creates;return fail_create?nullptr:&token;
        }
        void destroyEvent(void *event) override {
            EXPECT_TRUE(ownsCurrentThread());EXPECT_EQ(event,&token);++destroys;
        }
        bool recordEventChecked(void *event,void *stream) override {
            EXPECT_TRUE(ownsCurrentThread());EXPECT_EQ(event,&token);EXPECT_EQ(stream,defaultStream());
            ++records;return !fail_record;
        }
        bool queryEventChecked(void *event,bool &ready) override {
            EXPECT_TRUE(ownsCurrentThread());EXPECT_EQ(event,&token);++queries;ready=true;return !fail_query;
        }
        float eventElapsedTime(void *,void *) override {ADD_FAILURE();return 0;}
        void synchronizeEvent(void *) override {ADD_FAILURE();}
        void synchronizeStream(void *) override {ADD_FAILURE();}
        void synchronize() override {ADD_FAILURE();}
    };
    /** @return Completed synthetic coordinates with distinct mechanisms and directions. */
    PlanningHostDeviceObservations observation(const PlanningHostDeviceRequest &request)
    {
        PlanningHostDeviceObservations result{request, {}};
        for (auto mechanism : {PlanningHostTransferMechanism::DMA, PlanningHostTransferMechanism::MappedKernel})
            for (auto direction : {MappedTransferDirection::HostToDevice, MappedTransferDirection::DeviceToHost})
                for (size_t bytes : {size_t{1}, request.bulkBytes()})
                    result.phases.push_back({mechanism, direction, bytes, 1,
                        PlanningServiceObservation(PlanningWorkUnit::Bytes, bytes * 3.0, .01, "synthetic receipt")});
        return result;
    }
}

TEST(PlanningHostDeviceMeasurement, PageGeometryAndPMAUseTheAllocationAuthority)
{
    const auto page = size_t(sysconf(_SC_PAGESIZE));
    EXPECT_EQ(TransferEngine::mappedHostRegionAllocationBytes(1), page);
    EXPECT_EQ(TransferEngine::mappedHostRegionAllocationBytes(page), page);
    EXPECT_EQ(TransferEngine::mappedHostRegionAllocationBytes(page + 1), 2 * page);
    EXPECT_THROW(TransferEngine::mappedHostRegionAllocationBytes(0), std::invalid_argument);
    EXPECT_THROW(TransferEngine::mappedHostRegionAllocationBytes(SIZE_MAX), std::overflow_error);
    const auto cluster = inventory();
    for (auto type : {DeviceType::CUDA, DeviceType::ROCm})
    {
        const DeviceId device(type, 5);
        const auto request = PlanningHostDeviceRequest::fromInventory(cluster, {3, device}, {4}, page + 1);
        EXPECT_EQ(request.hostNumaNode(), 7);
        EXPECT_EQ(request.physicalNode(), 2);
        const PhysicalMemoryResource host{4, DeviceId::cpu(), 1ull << 30, 1ull << 30};
        const PhysicalMemoryResource gpu{3, device, 1ull << 30, 1ull << 30};
        const PhysicalMemoryResource executor_host{3, DeviceId::cpu(), 1ull << 30, 1ull << 30};
        PhysicalMemoryPlanBuilder builder;
        PlanningHostDeviceMeasurement::contributeHostMemory(request, host, builder);
        PlanningHostDeviceMeasurement::contributeExecutionMemory(request, executor_host, gpu, builder);
        const auto plan = builder.build();
        EXPECT_EQ(plan.find({4, DeviceId::cpu()})->bytes(PhysicalMemoryOwner::ActivationTransportStaging), 4 * page);
        EXPECT_EQ(plan.find({3, device})->bytes(PhysicalMemoryOwner::ActivationTransportStaging), page + 1);
        EXPECT_EQ(plan.find({3, device})->bytes(PhysicalMemoryOwner::NativeGraphExecutable),
            (PlanningExecutionMeasurement::kPreparedGraphCount+1) * GPUGraphMemoryContract::reservationBytesPerExecutable(device));
        EXPECT_EQ(plan.find({3,DeviceId::cpu()})->bytes(PhysicalMemoryOwner::ExecutionWorkspace),PlanningNativeGPUClockBuffer::hostAllocationBytes(2));
        EXPECT_EQ(plan.find({4,DeviceId::cpu()})->bytes(PhysicalMemoryOwner::ExecutionWorkspace),0u);
        EXPECT_EQ(plan.find({3,device})->bytes(PhysicalMemoryOwner::ExecutionWorkspace),PlanningNativeGPUClockBuffer::deviceAllocationBytes(2));
        EXPECT_THROW(PlanningHostDeviceMeasurement::contributeExecutionMemory(request,host,gpu,builder),std::invalid_argument);
        auto wrong = gpu;
        wrong.world_rank = 4;
        EXPECT_THROW(PlanningHostDeviceMeasurement::contributeExecutionMemory(request, executor_host, wrong, builder), std::invalid_argument);
        EXPECT_THROW(PlanningHostDeviceMeasurement::collect(request, nullptr, nullptr), std::runtime_error);
        EXPECT_EQ(request.gpuRank(), 3);
        EXPECT_EQ(request.hostRank(), 4);
        auto wrong_host = host;
        wrong_host.world_rank = 3;
        EXPECT_THROW(PlanningHostDeviceMeasurement::contributeHostMemory(request, wrong_host, builder), std::invalid_argument);
        const auto local = PlanningHostDeviceRequest::fromInventory(cluster, {3, device}, {3}, page + 1);
        EXPECT_EQ(local.hostNumaNode(), 11);
        EXPECT_EQ(local.gpuRank(), local.hostRank());
        EXPECT_NE(local.serialize(), request.serialize());
    }
}

TEST(PlanningHostDeviceMeasurement, MissingIdentityAndInvalidGeometryFailWithoutDeviceWork)
{
    for (int defect = 0; defect < 14; ++defect)
    {
        SCOPED_TRACE(defect);
        auto cluster = inventory();
        if (defect == 0) cluster.ranks[3].rank = -1;
        if (defect == 1) cluster.ranks[3].node_id = -1;
        if (defect == 2) cluster.ranks[4].cpu.numa_node = -2;
        if (defect == 3) cluster.ranks[4].cpu.memory_bytes = 0;
        if (defect == 4) cluster.ranks[4].cpu_cores = 0;
        if (defect == 5) cluster.ranks[3].gpus[0].uuid.clear();
        if (defect == 6) cluster.ranks[3].gpus.push_back(cluster.ranks[3].gpus[0]);
        if (defect == 7) cluster.ranks[3].gpus.erase(cluster.ranks[3].gpus.begin());
        if (defect == 9) cluster.ranks[3].cpu.numa_node = 7;
        if (defect == 10) cluster.ranks[4].node_id = 5;
        if (defect == 11) cluster.ranks[3].cpu.numa_node = -1;
        if (defect == 12) cluster.ranks.pop_back();
        if (defect == 13) cluster.ranks[4].cpu.numa_node = -1;
        const size_t bytes = defect == 8 ? SIZE_MAX : 1051;
        EXPECT_THROW(PlanningHostDeviceRequest::fromInventory(cluster, {3, DeviceId::cuda(5)}, {4}, bytes), std::invalid_argument);
    }
    const auto cluster = inventory();
    EXPECT_THROW(PlanningHostDeviceRequest::fromInventory(cluster, {3, DeviceId::cpu()}, {4}, 1051), std::invalid_argument);
    EXPECT_THROW(PlanningHostDeviceRequest::fromInventory(cluster, {3, DeviceId::cuda(5)}, {4}, 0), std::invalid_argument);
    EXPECT_THROW(PlanningHostDeviceRequest::fromInventory(cluster, {3, DeviceId::cuda(5)}, {4}, 1), std::invalid_argument);
    EXPECT_THROW(PlanningHostDeviceRequest::fromInventory(cluster, {5, DeviceId::cuda(5)}, {4}, 1051), std::invalid_argument);
    EXPECT_THROW(PlanningHostDeviceRequest::fromInventory(cluster, {3, DeviceId::cuda(5)}, {-1}, 1051), std::invalid_argument);
}

TEST(PlanningHostDeviceMeasurement, AllEightCoordinatesMustRetainGraphAndByteEvidence)
{
    const auto request = PlanningHostDeviceRequest::fromInventory(inventory(), {3, DeviceId::rocm(5)}, {4}, 1051);
    const auto valid = observation(request);
    EXPECT_NO_THROW(PlanningHostDeviceMeasurement::validate(valid));
    for (int defect = 0; defect < 8; ++defect)
    {
        auto broken = valid;
        if (defect == 0) broken.phases.pop_back();
        if (defect == 1) broken.phases[1] = broken.phases[0];
        if (defect == 2) broken.phases[0].mechanism = static_cast<PlanningHostTransferMechanism>(9);
        if (defect == 3) broken.phases[0].direction = static_cast<MappedTransferDirection>(99);
        if (defect == 4) broken.phases[0].payload_bytes = 2;
        if (defect == 5) broken.phases[0].graph_nodes = 0;
        if (defect == 6) broken.phases[0].service = {PlanningWorkUnit::ArithmeticOperations, 3, .01, "foreign units"};
        if (defect == 7) broken.phases[0].service = {PlanningWorkUnit::Bytes, 4, .01, "foreign count"};
        EXPECT_THROW(PlanningHostDeviceMeasurement::validate(broken), std::invalid_argument);
    }
}

/** @brief Every original inbound coordinate has exactly one untimed proof, without changing service work. */
TEST(PlanningHostDeviceMeasurement, ByteProofReplaysOncePerOriginalInboundCoordinate)
{
    for(auto type:{DeviceType::CUDA,DeviceType::ROCm}) {
        const auto request=PlanningHostDeviceRequest::fromInventory(inventory(),{3,DeviceId(type,5)},{4},65536);
        const auto measured=observation(request);
        PlanningHostDeviceMeasurement::validate(measured);
        ASSERT_EQ(measured.phases.size(),8u);
        ByteProofWorker worker;
        ByteProofGraph graph;
        graph.stream=worker.defaultStream();
        int proofs=0;
        for(const auto &phase:measured.phases) {
            EXPECT_DOUBLE_EQ(phase.service.completedWork(),3.0*phase.payload_bytes);
            if(phase.direction==MappedTransferDirection::HostToDevice) {
                worker.submitAndWait([&]{PlanningHostDeviceMeasurement::replayByteProof(worker,graph);});
                ++proofs;
            }
        }
        EXPECT_EQ(proofs,4);EXPECT_EQ(graph.launches,proofs);
        EXPECT_EQ(worker.creates,proofs);EXPECT_EQ(worker.destroys,proofs);
        EXPECT_EQ(worker.records,proofs);EXPECT_EQ(worker.queries,proofs);
    }
}

/** @brief Invalid original state and asynchronous errors fail without timing or replacement submission. */
TEST(PlanningHostDeviceMeasurement, ByteProofRejectsInvalidOwnerStreamGraphAndEvents)
{
    for(int defect=0;defect<8;++defect) {
        SCOPED_TRACE(defect);
        ByteProofWorker worker;
        ByteProofGraph graph;
        graph.stream=worker.defaultStream();
        int foreign_stream=0;
        if(defect==1)graph.stream=nullptr;
        if(defect==2)graph.stream=&foreign_stream;
        if(defect==3)graph.executable=false;
        worker.fail_create=defect==4;graph.fail_launch=defect==5;
        worker.fail_record=defect==6;worker.fail_query=defect==7;
        const auto operation=[&]{PlanningHostDeviceMeasurement::replayByteProof(worker,graph);};
        if(defect==0)EXPECT_THROW(operation(),std::invalid_argument);
        else worker.submitAndWait([&]{
            if(defect<=3)EXPECT_THROW(operation(),std::invalid_argument);
            else EXPECT_THROW(operation(),std::runtime_error);
        });
        EXPECT_EQ(graph.launches,defect>=5?1:0);
        EXPECT_EQ(worker.destroys,defect>=5?1:0);
    }
}
