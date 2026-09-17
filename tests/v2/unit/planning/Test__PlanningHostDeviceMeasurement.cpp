/**
 * @file Test__PlanningHostDeviceMeasurement.cpp
 * @brief Device-free ownership, exact-BOM and complete-mechanism evidence regressions.
 *
 * Synthetic times are codec/math witnesses only. No backend, mapped allocation,
 * kernel or MPI operation is initialized by these tests.
 */
#include "planning/PlanningHostDeviceMeasurement.h"
#include "backends/GPUGraphMemoryContract.h"
#include "transfer/TransferEngine.h"
#include <gtest/gtest.h>
#include <limits>
#include <unistd.h>

using namespace llaminar2;

namespace
{
    /** @return Sparse GPU ordinal with deliberately different host/GPU NUMA IDs. */
    RankInventory inventory()
    {
        RankInventory rank;
        rank.rank = 3;
        rank.node_id = 2;
        rank.cpu.numa_node = 7;
        rank.cpu.memory_bytes = rank.cpu.free_memory_bytes = 1ull << 30;
        rank.cpu_cores = 2;
        for (auto type : {DeviceType::CUDA, DeviceType::ROCm})
            rank.gpus.push_back({.type = type, .local_device_id = 5, .memory_bytes = 1ull << 30,
                .free_memory_bytes = 1ull << 30, .uuid = "physical-gpu", .numa_node = 11});
        return rank;
    }
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
    const auto rank = inventory();
    for (auto type : {DeviceType::CUDA, DeviceType::ROCm})
    {
        const DeviceId device(type, 5);
        const auto request = PlanningHostDeviceRequest::fromInventory(rank, device, page + 1);
        EXPECT_EQ(request.hostNumaNode(), 7);
        EXPECT_EQ(request.physicalNode(), 2);
        const PhysicalMemoryResource host{3, DeviceId::cpu(), 1ull << 30, 1ull << 30};
        const PhysicalMemoryResource gpu{3, device, 1ull << 30, 1ull << 30};
        PhysicalMemoryPlanBuilder builder;
        PlanningHostDeviceMeasurement::contributeMemory(request, host, gpu, builder);
        const auto plan = builder.build();
        EXPECT_EQ(plan.find({3, DeviceId::cpu()})->bytes(PhysicalMemoryOwner::ActivationTransportStaging), 4 * page);
        EXPECT_EQ(plan.find({3, device})->bytes(PhysicalMemoryOwner::ActivationTransportStaging), page + 1);
        EXPECT_EQ(plan.find({3, device})->bytes(PhysicalMemoryOwner::NativeGraphExecutable),
            2 * GPUGraphMemoryContract::reservationBytesPerExecutable(device));
        auto wrong = gpu;
        wrong.world_rank = 4;
        EXPECT_THROW(PlanningHostDeviceMeasurement::contributeMemory(request, host, wrong, builder), std::invalid_argument);
        EXPECT_THROW(PlanningHostDeviceMeasurement::measure(request, nullptr), std::invalid_argument);
    }
}

TEST(PlanningHostDeviceMeasurement, MissingIdentityAndInvalidGeometryFailWithoutDeviceWork)
{
    for (int defect = 0; defect < 9; ++defect)
    {
        auto rank = inventory();
        if (defect == 0) rank.rank = -1;
        if (defect == 1) rank.node_id = -1;
        if (defect == 2) rank.cpu.numa_node = -2;
        if (defect == 3) rank.cpu.memory_bytes = 0;
        if (defect == 4) rank.cpu_cores = 0;
        if (defect == 5) rank.gpus[0].uuid.clear();
        if (defect == 6) rank.gpus.push_back(rank.gpus[0]);
        if (defect == 7) rank.gpus.erase(rank.gpus.begin());
        const size_t bytes = defect == 8 ? SIZE_MAX : 1051;
        EXPECT_THROW(PlanningHostDeviceRequest::fromInventory(rank, DeviceId::cuda(5), bytes), std::invalid_argument);
    }
    const auto rank = inventory();
    EXPECT_THROW(PlanningHostDeviceRequest::fromInventory(rank, DeviceId::cpu(), 1051), std::invalid_argument);
    EXPECT_THROW(PlanningHostDeviceRequest::fromInventory(rank, DeviceId::cuda(5), 0), std::invalid_argument);
    EXPECT_THROW(PlanningHostDeviceRequest::fromInventory(rank, DeviceId::cuda(5), 1), std::invalid_argument);
}

TEST(PlanningHostDeviceMeasurement, AllEightCoordinatesMustRetainGraphAndByteEvidence)
{
    const auto request = PlanningHostDeviceRequest::fromInventory(inventory(), DeviceId::rocm(5), 1051);
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
