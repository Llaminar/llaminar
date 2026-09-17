/**
 * @file Test__PlanningLocalTPMeasurement.cpp
 * @brief Device-free native-group, payload, physical ownership and timing arithmetic gates.
 *
 * Synthetic resources prove admission uses the ordinary collective BOM without
 * querying a GPU. Real NCCL/RCCL capture and replay belong to integration tests.
 */
#include "planning/PlanningLocalTPMeasurement.h"
#include "backends/GPUGraphMemoryContract.h"
#include <gtest/gtest.h>
#include <climits>
#include <limits>

using namespace llaminar2;

namespace
{
    /** @return Device-free capacity fixture, not another allocation ledger. */
    PhysicalMemoryResource resource(DeviceId device, int rank = 3)
    {
        return {.world_rank = rank, .device = device, .total_bytes = 1ull << 30,
            .admission_available_bytes = 1ull << 30};
    }
}

TEST(PlanningLocalTPMeasurement, RejectsUnrepresentableAndNonNativeGroupsWithoutADevice)
{
    for (const auto devices : std::vector<std::vector<DeviceId>>{
        {}, {DeviceId::cuda(0)}, {DeviceId::cpu(), DeviceId::cpu()},
        {DeviceId::cuda(0), DeviceId::rocm(1)}, {DeviceId::cuda(2), DeviceId::cuda(2)},
        {DeviceId::invalid(), DeviceId::cuda(1)}, {DeviceId::cuda(-1), DeviceId::cuda(1)}})
        EXPECT_THROW(PlanningLocalTPRequest(devices, 256, 32, PlanningAllreducePrecision::FP32), std::invalid_argument);
    const std::vector devices{DeviceId::rocm(7), DeviceId::rocm(2)};
    EXPECT_THROW(PlanningLocalTPRequest(devices, 0, 32, PlanningAllreducePrecision::FP32), std::invalid_argument);
    EXPECT_THROW(PlanningLocalTPRequest(devices, 256, -1, PlanningAllreducePrecision::FP32), std::invalid_argument);
    EXPECT_THROW(PlanningLocalTPRequest(devices, 256, 32, PlanningAllreducePrecision(42)), std::invalid_argument);
    EXPECT_THROW(PlanningLocalTPRequest(devices, INT_MAX, INT_MAX, PlanningAllreducePrecision::FP16), std::overflow_error);
}

TEST(PlanningLocalTPMeasurement, ExactPayloadsAndCanonicalCollectiveBOMOnBothVendors)
{
    for (const auto backend : {DeviceType::CUDA, DeviceType::ROCm})
        for (const auto precision : {PlanningAllreducePrecision::FP32, PlanningAllreducePrecision::FP16})
        {
            const std::vector devices{DeviceId(backend, 7), DeviceId(backend, 2)};
            const PlanningLocalTPRequest request(devices, 257, 31, precision);
            EXPECT_EQ(request.devices(), devices);
            EXPECT_EQ(request.backend(), backend == DeviceType::CUDA ? CollectiveBackendType::NCCL : CollectiveBackendType::RCCL);
            const size_t bytes = precision == PlanningAllreducePrecision::FP32 ? 4 : 2;
            EXPECT_EQ(request.payloadBytes(1), 257 * bytes);
            EXPECT_EQ(request.payloadBytes(31), 31 * 257 * bytes);
            EXPECT_EQ(request.activationBytes(), 31 * 257 * sizeof(float));
            EXPECT_THROW(request.payloadBytes(30), std::invalid_argument);
            PhysicalMemoryPlanBuilder builder;
            const std::vector resources{resource(devices[0]), resource(devices[1])};
            PlanningLocalTPMeasurement::contributeMemory(request, resource(DeviceId::cpu()), resources, builder);
            PhysicalMemoryAuthority memory(std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(builder.build()), 3);
            EXPECT_EQ(memory.plannedBytes(DeviceId::cpu(), PhysicalMemoryOwner::ExecutionWorkspace), 2 * request.activationBytes());
            const auto collective = CollectiveMemoryEstimator::localTP(31, 257, request.backend());
            for (const auto device : devices)
            {
                EXPECT_EQ(memory.plannedBytes(device, PhysicalMemoryOwner::ExecutionWorkspace), request.activationBytes());
                EXPECT_EQ(memory.plannedBytes(device, PhysicalMemoryOwner::LocalCollective), collective.perDeviceBytes());
                EXPECT_EQ(memory.plannedBytes(device, PhysicalMemoryOwner::NativeGraphExecutable),
                    2 * GPUGraphMemoryContract::reservationBytesPerExecutable(device));
            }
            EXPECT_THROW(PlanningLocalTPMeasurement::measure(request, {}), std::invalid_argument);
            for (const auto wrong : std::vector<std::vector<PhysicalMemoryResource>>{
                {resources[0]}, {resources[1], resources[0]}, {resources[0], resource(devices[1], 4)}})
            {
                PhysicalMemoryPlanBuilder rejected;
                EXPECT_THROW(PlanningLocalTPMeasurement::contributeMemory(request, resource(DeviceId::cpu()), wrong, rejected),
                    std::invalid_argument);
            }
        }
}

TEST(PlanningLocalTPMeasurement, CoupledEndpointLatencyUsesSlowestNotSumOrAverage)
{
    PlanningLocalTPPhaseObservation phase{1, 1024,
        {{DeviceId::cuda(7), 3, .000010}, {DeviceId::cuda(2), 5, .000040}}};
    EXPECT_DOUBLE_EQ(phase.secondsPerCollective(), .000040);
    for (double invalid : {0.0, -1.0, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()})
    {
        auto rejected = phase;
        rejected.endpoints[1].seconds_per_collective = invalid;
        EXPECT_THROW(rejected.secondsPerCollective(), std::invalid_argument);
    }
    auto rejected = phase;
    rejected.endpoints[1].graph_nodes = 0;
    EXPECT_THROW(rejected.secondsPerCollective(), std::invalid_argument);
    rejected = phase;
    rejected.endpoints[1].device = rejected.endpoints[0].device;
    EXPECT_THROW(rejected.secondsPerCollective(), std::invalid_argument);
    rejected = phase;
    rejected.endpoints[1].device = DeviceId::rocm(2);
    EXPECT_THROW(rejected.secondsPerCollective(), std::invalid_argument);
    rejected = phase;
    rejected.endpoints.pop_back();
    EXPECT_THROW(rejected.secondsPerCollective(), std::invalid_argument);
}
