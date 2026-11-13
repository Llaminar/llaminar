/**
 * @file Test__PlanningLocalTPMeasurement.cpp
 * @brief Device-free native-group, payload, physical ownership and timing arithmetic gates.
 *
 * Synthetic resources prove admission uses the ordinary collective BOM without
 * querying a GPU. Real NCCL/RCCL capture and replay belong to integration tests.
 */
#include "planning/PlanningLocalTPMeasurement.h"
#include "planning/PlanningGPUClockBuffer.h"
#include "backends/GPUGraphMemoryContract.h"
#include "planning/PlanningExecutionMeasurement.h"
#include "collective/NativeCountedAllGatherFrame.h"
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
            const size_t slots = 2 * request.pairPayloadAnchors().size() * PlanningLocalTPPairSampling::kSamples;
            EXPECT_EQ(memory.plannedBytes(DeviceId::cpu(), PhysicalMemoryOwner::ExecutionWorkspace),
                2 * (request.storageBytes() + request.gatherStorageBytes() + PlanningNativeGPUClockBuffer::hostAllocationBytes(slots)));
            const auto collective = CollectiveMemoryEstimator::localTP(31, 257, request.backend());
            for (const auto device : devices)
            {
                EXPECT_EQ(memory.plannedBytes(device, PhysicalMemoryOwner::ExecutionWorkspace),
                    request.storageBytes() + request.gatherStorageBytes() + PlanningNativeGPUClockBuffer::deviceAllocationBytes(slots));
                EXPECT_EQ(memory.plannedBytes(device, PhysicalMemoryOwner::LocalCollective), collective.perDeviceBytes());
                EXPECT_EQ(memory.plannedBytes(device,PhysicalMemoryOwner::ActivationTransportStaging),NativeCountedAllGatherLayout(devices.size()).bytes());
                EXPECT_EQ(memory.plannedBytes(device, PhysicalMemoryOwner::NativeGraphExecutable),
                    PlanningExecutionMeasurement::kPreparedGraphCount * GPUGraphMemoryContract::reservationBytesPerExecutable(device));
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

/** @test Captured anchors retain independent live destinations and untouched guard bytes. */
TEST(PlanningLocalTPMeasurement, DirectedPairPayloadRegionsAreDisjointAndPMAAdmitted)
{
    for (const auto backend : {DeviceType::CUDA, DeviceType::ROCm})
        for (const auto precision : {PlanningAllreducePrecision::FP32, PlanningAllreducePrecision::FP16})
            for (const int width : {1, 3, 4, 257})
                for (const int rows : {1, 2, 31, 128})
                {
                    const PlanningLocalTPRequest request({DeviceId(backend, 7), DeviceId(backend, 2)},
                        width, rows, precision);
                    const auto regions = request.pairPayloadRegions();
                    const auto anchors = request.pairPayloadAnchors();
                    ASSERT_EQ(regions.size(), anchors.size());
                    size_t end = 0;
                    for (size_t index = 0; index < regions.size(); ++index)
                    {
                        const auto &region = regions[index];
                        EXPECT_GE(region.byte_offset, end);
                        EXPECT_EQ(region.byte_offset % sizeof(float), 0u);
                        EXPECT_EQ(region.live_bytes, anchors[index]);
                        EXPECT_GT(region.guard_bytes, 0u);
                        end = region.byte_offset + region.live_bytes + region.guard_bytes;
                        EXPECT_LE(end, request.storageBytes());
                    }
                    EXPECT_GE(request.storageBytes(), request.activationBytes());
                    EXPECT_EQ(request.storageBytes() % sizeof(float), 0u);
                    PhysicalMemoryPlanBuilder builder;
                    const std::vector resources{resource(request.devices()[0]), resource(request.devices()[1])};
                    PlanningLocalTPMeasurement::contributeMemory(request, resource(DeviceId::cpu()), resources, builder,PlanningLocalTPMeasurementMode::PhysicalPeersOnly);
                    PhysicalMemoryAuthority memory(std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(builder.build()), 3);
                    const size_t slots = 2 * anchors.size() * PlanningLocalTPPairSampling::kSamples;
                    EXPECT_EQ(memory.plannedBytes(DeviceId::cpu(), PhysicalMemoryOwner::ExecutionWorkspace),
                        request.devices().size() *
                            (request.storageBytes() + PlanningNativeGPUClockBuffer::hostAllocationBytes(slots)));
                    for (const auto device : request.devices())
                        EXPECT_EQ(memory.plannedBytes(device, PhysicalMemoryOwner::ExecutionWorkspace),
                            request.storageBytes() + PlanningNativeGPUClockBuffer::deviceAllocationBytes(slots));
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

/** @test Both backend orders, odd groups and repeated validation preserve every directed byte anchor. */
TEST(PlanningLocalTPMeasurement, DirectedPairMatrixCoverageAndReplayAreExact)
{
    for (const auto backend : {DeviceType::CUDA, DeviceType::ROCm})
        for (const auto precision : {PlanningAllreducePrecision::FP32, PlanningAllreducePrecision::FP16})
        {
            const PlanningLocalTPRequest request({DeviceId(backend, 7), DeviceId(backend, 2), DeviceId(backend, 4)},
                257, 31, precision);
            EXPECT_EQ(request.pairPayloadAnchors(), (std::vector<size_t>{16, 1028, 31868}));
            PlanningLocalTPPairSampling matrix;
            matrix.payload_anchors = request.pairPayloadAnchors();
            matrix.direct_peer_access = {0,1,1, 1,0,1, 1,1,0};
            for (size_t source = 0; source < 3; ++source)
                for (size_t destination = 0; destination < 3; ++destination)
                    if (source != destination)
                        for (const auto bytes : matrix.payload_anchors)
                            matrix.directed_pairs.push_back({source, destination, bytes, {.00001, .00002, .00003}, 1});
            matrix.capture_count = matrix.directed_pairs.size() / matrix.payload_anchors.size();
            matrix.measured_launches = matrix.capture_count;
            for (int replay = 0; replay < 3; ++replay) EXPECT_NO_THROW(matrix.validate(request));
            auto rejected = matrix;
            rejected.directed_pairs.pop_back();
            EXPECT_THROW(rejected.validate(request), std::invalid_argument);
            rejected = matrix;
            rejected.directed_pairs.back() = rejected.directed_pairs.front();
            EXPECT_THROW(rejected.validate(request), std::invalid_argument);
            rejected = matrix;
            rejected.directed_pairs.front().seconds_per_transfer[1] = 0;
            EXPECT_THROW(rejected.validate(request), std::invalid_argument);
            rejected = matrix;
            rejected.directed_pairs.front().source_index = 3;
            EXPECT_THROW(rejected.validate(request), std::invalid_argument);
            rejected = matrix;
            rejected.directed_pairs.front().payload_bytes = request.activationBytes() + 1;
            EXPECT_THROW(rejected.validate(request), std::invalid_argument);
            rejected = matrix;
            --rejected.measured_launches;
            EXPECT_THROW(rejected.validate(request), std::invalid_argument);
        }
    const PlanningLocalTPRequest tiny({DeviceId::cuda(9), DeviceId::cuda(3)}, 1, 1,
        PlanningAllreducePrecision::FP16);
    EXPECT_EQ(tiny.pairPayloadAnchors(), (std::vector<size_t>{4}));
}

/** @test Whole-group gather latency is coupled at the slowest endpoint and never divided by participant count. */
TEST(PlanningLocalTPMeasurement, NativeGatherLatencyKeepsWholeGroupAndRejectsIncompleteEvidence)
{
    for(const size_t count:{2u,3u,5u,9u}){
        PlanningLocalTPGatherPhaseObservation phase{PlanningNativeGatherProtocol::CountedAllgather,0,count,{}};
        for(size_t i=0;i<count;++i)phase.endpoints.push_back({DeviceId::rocm(int(i)),count,.00001*(i+1)});
        EXPECT_DOUBLE_EQ(phase.secondsPerCollective(),.00001*count);
        auto malformed=phase;malformed.participant_count=0;
        EXPECT_THROW(malformed.secondsPerCollective(),std::invalid_argument);
        malformed=phase;malformed.endpoints[0].seconds_per_collective=0;
        EXPECT_THROW(malformed.secondsPerCollective(),std::invalid_argument);
        malformed=phase;malformed.endpoints[0].graph_nodes=0;
        EXPECT_THROW(malformed.secondsPerCollective(),std::invalid_argument);
    }
}
