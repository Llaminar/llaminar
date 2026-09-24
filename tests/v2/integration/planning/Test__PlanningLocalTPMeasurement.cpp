/**
 * @file Test__PlanningLocalTPMeasurement.cpp
 * @brief Real NCCL/RCCL retained-graph sampling, resource rejection and retirement proofs.
 *
 * This functional gate has no throughput threshold. Both native transport
 * precisions, odd row/column tails, reversed group order and repeated group
 * construction must produce complete positive event observations. An exhausted
 * endpoint must fail without stranding its peer or retaining physical owners.
 */
#include "planning/PlanningLocalTPMeasurement.h"
#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>

using namespace llaminar2;

namespace
{
    /** @return Small explicit test ceiling; production contributor owns every demand. */
    PhysicalMemoryResource resource(DeviceId device)
    {
        return {.world_rank = 0, .device = device, .total_bytes = 256ull << 20,
            .admission_available_bytes = 256ull << 20};
    }

    /** @return Ordinary canonical authority with one collective group admitted. */
    std::shared_ptr<PhysicalMemoryAuthority> memoryFor(const PlanningLocalTPRequest &request)
    {
        PhysicalMemoryPlanBuilder builder;
        std::vector<PhysicalMemoryResource> resources;
        for (const auto device : request.devices()) resources.push_back(resource(device));
        PlanningLocalTPMeasurement::contributeMemory(request, resource(DeviceId::cpu()), resources, builder);
        return std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(builder.build()), 0);
    }

    /** @brief No sampled payload, graph reservation or collective scratch escapes the transaction. */
    void expectRetired(const PlanningLocalTPRequest &request, const PhysicalMemoryAuthority &memory)
    {
        auto devices = request.devices();
        devices.push_back(DeviceId::cpu());
        for (const auto device : devices)
            for (const auto owner : {PhysicalMemoryOwner::ExecutionWorkspace,
                PhysicalMemoryOwner::LocalCollective, PhysicalMemoryOwner::NativeGraphExecutable})
            {
                EXPECT_EQ(memory.claimedBytes(device, owner, PhysicalMemoryMaterializationKind::NewAllocation), 0u);
                EXPECT_EQ(memory.reservedBytes(device, owner), 0u);
            }
    }

    /** @brief Exercise exact backend/group evidence without inferring a link type from its speed. */
    void prove(DeviceType backend)
    {
        if (!hasCPUBackend()) initCPUBackend(-1);
        auto &pool = GPUDeviceContextPool::instance();
        const int count = backend == DeviceType::CUDA ? pool.nvidiaDeviceCount() : pool.amdDeviceCount();
        ASSERT_GE(count, 2) << "Native collective planning proof requires two GPUs of the requested backend";
        std::vector<DeviceId> devices;
        for (int index = 0; index < count; ++index) devices.emplace_back(backend, index);
        // Every available member participates; no hard-coded CUDA/ROCm degree.
        for (const auto precision : {PlanningAllreducePrecision::FP32, PlanningAllreducePrecision::FP16})
        {
            const PlanningLocalTPRequest request(devices, 257, 31, precision);
            auto memory = memoryFor(request);
            for (int repeat = 0; repeat < 2; ++repeat)
            {
                const auto result = PlanningLocalTPMeasurement::measure(request, memory);
                ASSERT_EQ(result.phases.size(), 2u);
                EXPECT_EQ(result.request.devices(), devices);
                EXPECT_EQ(result.request.precision(), precision);
                for (size_t phase_index = 0; phase_index < result.phases.size(); ++phase_index)
                {
                    const auto &phase = result.phases[phase_index];
                    EXPECT_EQ(phase.rows, phase_index ? 31 : 1);
                    EXPECT_EQ(phase.payload_bytes, request.payloadBytes(phase.rows));
                    ASSERT_EQ(phase.endpoints.size(), devices.size());
                    EXPECT_GT(phase.secondsPerCollective(), 0.0);
                    for (size_t index = 0; index < devices.size(); ++index)
                    {
                        EXPECT_EQ(phase.endpoints[index].device, devices[index]);
                        EXPECT_GT(phase.endpoints[index].graph_nodes, 0u);
                        EXPECT_TRUE(std::isfinite(phase.endpoints[index].seconds_per_collective));
                    }
                    ::testing::Test::RecordProperty((precision == PlanningAllreducePrecision::FP32 ? "fp32_" : "fp16_") +
                        std::string(phase_index ? "prefill_us" : "decode_us"), std::to_string(phase.secondsPerCollective() * 1e6));
                }
                expectRetired(request, *memory);
            }
            std::reverse(devices.begin(), devices.end());
        }
        const PlanningLocalTPRequest request(devices, 256, 32, PlanningAllreducePrecision::FP32);
        auto memory = memoryFor(request);
        // Exhaust only the final endpoint: earlier endpoint claims must unwind,
        // not leave another worker waiting in capture or leak its reservation.
        for (const auto owner : {PhysicalMemoryOwner::ExecutionWorkspace, PhysicalMemoryOwner::NativeGraphExecutable,
            PhysicalMemoryOwner::LocalCollective})
        {
            {
                auto occupied = memory->reserveNewAllocations(devices.back(), owner,
                    memory->plannedBytes(devices.back(), owner));
                EXPECT_THROW(PlanningLocalTPMeasurement::measure(request, memory), std::exception);
            }
            expectRetired(request, *memory);
        }
        // A previous failed transaction cannot poison the next group's lifecycle.
        EXPECT_NO_THROW(PlanningLocalTPMeasurement::measure(request, memory));
        expectRetired(request, *memory);
    }
}

#ifdef HAVE_CUDA
TEST(PlanningExecutionMeasurementIntegration, CUDA_NativeLocalTP)
{
    ensureNvidiaFactoryRegistered();
    prove(DeviceType::CUDA);
}
#endif
#ifdef HAVE_ROCM
TEST(PlanningExecutionMeasurementIntegration, ROCm_NativeLocalTP)
{
    ensureAMDFactoryRegistered();
    prove(DeviceType::ROCm);
}
#endif
