/**
 * @file Test__PlanningMemoryBandwidthMeasurement.cpp
 * @brief Device-free cache-domain, exact-BOM and malformed-observation regressions.
 *
 * These synthetic inventories only prove geometry and ownership arithmetic.
 * Actual streaming execution and native capture belong to the preflight tests.
 */
#include "planning/PlanningMemoryBandwidthMeasurement.h"
#include "backends/GPUGraphMemoryContract.h"
#include "../../utils/CPUExecutionTestGeometry.h"
#include <gtest/gtest.h>
#include <limits>

using namespace llaminar2;

namespace
{
    /** @return Two nonadjacent CPU sockets and sparse, symmetric GPU ordinals. */
    RankInventory inventory()
    {
        RankInventory rank;
        rank.rank = 3;
        rank.cpu.numa_node = -1;
        rank.cpu.last_level_cache_bytes = 96u << 20;
        rank.cpu_cores = 8;
        rank.cpu_worker_threads = 8;
        rank.cpu_execution = test::kSyntheticCPUExecutionGeometry;
        for (int index : {2, 7})
        {
            CPUSocketInfo socket;
            socket.socket_id = index;
            socket.numa_node = index;
            rank.cpu_socket_info.push_back(socket);
        }
        for (auto backend : {DeviceType::CUDA, DeviceType::ROCm})
            rank.gpus.push_back({.type = backend, .local_device_id = 4,
                .last_level_cache_bytes = 6u << 20});
        return rank;
    }
}

TEST(PlanningMemoryBandwidthMeasurement, EveryStreamExceedsObservedCacheDomains)
{
    auto rank = inventory();
    auto request = PlanningMemoryBandwidthRequest::fromInventory(rank, DeviceId::cpu());
    EXPECT_EQ(request.cacheBytes(), rank.cpu.last_level_cache_bytes +
        8 * rank.cpu_execution.cache.private_l2_bytes);
    EXPECT_GE(request.streamBytes(), 4 * request.cacheBytes());
    EXPECT_EQ(request.streamBytes() % 4096, 0u);
    EXPECT_EQ(request.usefulBytes(), 3 * request.streamBytes());
    EXPECT_EQ(request.workers(), 8);
    rank.cpu.numa_node = 7;
    rank.cpu.last_level_cache_bytes = 64u << 20;
    request = PlanningMemoryBandwidthRequest::fromInventory(rank, DeviceId::cpu());
    EXPECT_EQ(request.cacheBytes(), rank.cpu.last_level_cache_bytes +
        8 * rank.cpu_execution.cache.private_l2_bytes);
    for (auto backend : {DeviceType::CUDA, DeviceType::ROCm})
    {
        request = PlanningMemoryBandwidthRequest::fromInventory(rank, {backend, 4});
        EXPECT_EQ(request.cacheBytes(), 6u << 20);
        EXPECT_EQ(request.streamBytes(), 24u << 20);
        EXPECT_EQ(request.workers(), 0);
    }
}

TEST(PlanningMemoryBandwidthMeasurement, MissingAndUnrepresentableEvidenceCannotBecomeCacheResidentFallback)
{
    for (int defect = 0; defect != 8; ++defect)
    {
        auto rank = inventory();
        if (defect == 0) rank.cpu_cores = 0;
        if (defect == 1) rank.cpu_execution.cache.shared_l3_bytes = 0;
        if (defect == 2) rank.cpu_socket_info.clear();
        if (defect == 3) rank.cpu.numa_node = 19;
        if (defect == 4) rank.rank = -1;
        if (defect == 5) rank.cpu.last_level_cache_bytes = 0;
        if (defect == 6) rank.cpu_worker_threads = 0;
        if (defect == 7) rank.cpu_worker_threads = -1;
        EXPECT_THROW(PlanningMemoryBandwidthRequest::fromInventory(rank, DeviceId::cpu()), std::invalid_argument);
    }
    auto rank = inventory();
    EXPECT_THROW(PlanningMemoryBandwidthRequest::fromInventory(rank, DeviceId(DeviceType::CPU, 7)), std::invalid_argument);
    EXPECT_THROW(PlanningMemoryBandwidthRequest::fromInventory(rank, DeviceId::cuda(0)), std::invalid_argument);
    rank.gpus[0].last_level_cache_bytes = 0;
    EXPECT_THROW(PlanningMemoryBandwidthRequest::fromInventory(rank, DeviceId::cuda(4)), std::invalid_argument);
    rank.gpus[0].last_level_cache_bytes = std::numeric_limits<size_t>::max();
    EXPECT_THROW(PlanningMemoryBandwidthRequest::fromInventory(rank, DeviceId::cuda(4)), std::overflow_error);
    rank.cpu_execution.cache.private_l2_bytes = std::numeric_limits<uint64_t>::max();
    EXPECT_THROW(PlanningMemoryBandwidthRequest::fromInventory(rank, DeviceId::cpu()), std::overflow_error);
}

TEST(PlanningMemoryBandwidthMeasurement, ExplicitWorkshareDoesNotBecomePhysicalCoreCount)
{
    auto rank = inventory();
    rank.cpu_worker_threads = 1;
    const auto request = PlanningMemoryBandwidthRequest::fromInventory(rank, DeviceId::cpu());
    EXPECT_EQ(request.workers(), 1);
    EXPECT_EQ(rank.cpu_cores, 8);
    EXPECT_EQ(request.cacheBytes(), rank.cpu.last_level_cache_bytes + rank.cpu_execution.cache.private_l2_bytes);
    // Physical topology still matters for cache-domain coverage, but it cannot
    // silently enlarge the observed team or its thread-private storage.
    rank.cpu_cores = 64;
    const auto same_team = PlanningMemoryBandwidthRequest::fromInventory(rank, DeviceId::cpu());
    EXPECT_EQ(same_team.workers(), request.workers());
    EXPECT_EQ(same_team.streamBytes(), request.streamBytes());
}

TEST(PlanningMemoryBandwidthMeasurement, AdmissionOwnsExactOverlappingHostAndDevicePayloads)
{
    const auto rank = inventory();
    const PhysicalMemoryResource host{3, DeviceId::cpu(), 2ull << 30, 2ull << 30};
    for (auto device : {DeviceId::cpu(), DeviceId::cuda(4), DeviceId::rocm(4)})
    {
        const auto request = PlanningMemoryBandwidthRequest::fromInventory(rank, device);
        const PhysicalMemoryResource execution{3, device, 2ull << 30, 2ull << 30};
        PhysicalMemoryPlanBuilder builder;
        PlanningMemoryBandwidthMeasurement::contributeMemory(request, host, execution, builder);
        const auto plan = builder.build();
        EXPECT_EQ(plan.find({3, DeviceId::cpu()})->bytes(PhysicalMemoryOwner::ExecutionWorkspace), request.usefulBytes());
        EXPECT_EQ(plan.find({3, device})->bytes(PhysicalMemoryOwner::ExecutionWorkspace), request.usefulBytes());
        EXPECT_EQ(plan.find({3, device})->bytes(PhysicalMemoryOwner::NativeGraphExecutable),
            device.is_gpu() ? GPUGraphMemoryContract::reservationBytesPerExecutable(device) : 0);
        auto wrong = execution;
        wrong.world_rank = 2;
        EXPECT_THROW(PlanningMemoryBandwidthMeasurement::contributeMemory(request, host, wrong, builder), std::invalid_argument);
        EXPECT_THROW(PlanningMemoryBandwidthMeasurement::measure(request, nullptr), std::invalid_argument);
    }
}
