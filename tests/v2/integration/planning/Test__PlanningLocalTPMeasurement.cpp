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
#include "backends/GPUGraphMemoryContract.h"
#include "transfer/TransferEngine.h"
#ifdef HAVE_CUDA
#include "backends/cuda/CUDABackend.h"
#endif
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>

using namespace llaminar2;

namespace
{
    /** @return Small explicit test ceiling; production contributor owns every demand. */
    PhysicalMemoryResource resource(DeviceId device)
    {
        if (device.is_gpu())
        {
            auto *backend = getBackendFor(device);
            if (!backend) throw std::runtime_error("Native LocalTP proof lost its physical backend");
            return {.world_rank = 0, .device = device,
                .total_bytes = backend->deviceMemoryTotal(device.ordinal),
                .admission_available_bytes = backend->deviceMemoryFree(device.ordinal)};
        }
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

/**
 * @test Cold production-size native kernels retain context bytes after both graphs retire.
 *
 * The original small odd-tail sample selected a lower-stack native algorithm
 * and did not expose the second growth event. Start this case in its own cold
 * process, use the real 2048-wide/64-row service geometry, and repeat with the
 * exact same admitted graph units. The context's canonical leases survive the
 * setup authority; repeating the family must not grow them again. Only the
 * public exclusive model-retirement transaction may release them, and the next
 * context generation must admit its own new leases before the same sample runs.
 */
namespace
{
/**
 * @brief Prove cold admission, graph retirement and successor ownership at an exact collective geometry.
 * @param columns Actual projection width communicated by each sample.
 * @param rows Exact admitted prefill/collective capacity of the sample family.
 */
void proveContextStorageLifetime(int columns, int rows)
{
    ensureNvidiaFactoryRegistered();
    if (!hasCPUBackend()) initCPUBackend(-1);
    auto &pool = GPUDeviceContextPool::instance();
    ASSERT_GE(pool.nvidiaDeviceCount(), 2);
    std::vector<DeviceId> devices;
    for (int index = 0; index < pool.nvidiaDeviceCount(); ++index) devices.push_back(DeviceId::cuda(index));
    auto *backend = dynamic_cast<CUDABackend *>(getBackendFor(devices.front()));
    ASSERT_NE(backend, nullptr);
    for (const auto device : devices)
        ASSERT_EQ(backend->nativeExecutionContextStorageBytes(device.ordinal), 0u)
            << "Run this cold-context proof as its independently registered preflight case";
    const PlanningLocalTPRequest request(devices, columns, rows, PlanningAllreducePrecision::FP32);
    std::vector<std::size_t> retained;
    std::weak_ptr<PhysicalMemoryAuthority> setup_lifetime;
    {
        auto memory = memoryFor(request);
        setup_lifetime = memory;
        const auto result = PlanningLocalTPMeasurement::measure(request, memory);
        ASSERT_EQ(result.phases.size(), 2u);
        expectRetired(request, *memory);
        for (const auto device : devices)
        {
            const auto bytes = backend->nativeExecutionContextStorageBytes(device.ordinal);
            EXPECT_GT(bytes, 0u);
            retained.push_back(bytes);
            EXPECT_EQ(memory->plannedBytes(device, PhysicalMemoryOwner::NativeGraphExecutable),
                2 * GPUGraphMemoryContract::reservationBytesPerExecutable(device));
        }
    }
    EXPECT_TRUE(setup_lifetime.expired());
    for (std::size_t index = 0; index < devices.size(); ++index)
        EXPECT_EQ(backend->nativeExecutionContextStorageBytes(devices[index].ordinal), retained[index]);
    auto replay_memory = memoryFor(request);
    EXPECT_NO_THROW(PlanningLocalTPMeasurement::measure(request, replay_memory));
    expectRetired(request, *replay_memory);
    for (std::size_t index = 0; index < devices.size(); ++index)
        EXPECT_EQ(backend->nativeExecutionContextStorageBytes(devices[index].ordinal), retained[index]);

    // Destroying graphs was deliberately insufficient. End the complete
    // native-context generation through the same public transaction as model
    // unload, rather than calling cudaDeviceReset behind the lifecycle owner.
    TransferEngine transfers;
    constexpr std::size_t model_workspace_bytes = 4096u;
    std::vector<PhysicalMemoryAllocationLease> model_claims;
    std::vector<std::shared_ptr<DeviceTransferBuffer>> model_buffers;
    for (const auto device : devices)
    {
        model_claims.push_back(replay_memory->claimNewAllocation(device,
            PhysicalMemoryOwner::ExecutionWorkspace, model_workspace_bytes));
        model_buffers.push_back(transfers.allocateDeviceTransferBuffer(model_workspace_bytes, device));
    }
    // A retirement ticket requires a real positive model BOM. These small
    // PMA-owned buffers stand for its final live workspace; release backing
    // before releasing its ledger claims, and both before completing unload.
    std::vector<ExclusiveModelRetirementTicket> tickets;
    for (const auto device : devices)
        tickets.push_back(transfers.beginExclusiveModelRetirement(
            ModelDeviceMemoryRetention{.device = device, .reusable_workspace_bytes = model_workspace_bytes}));
    model_buffers.clear();
    model_claims.clear();
    const auto receipts = transfers.completeExclusiveModelRetirements(std::move(tickets));
    ASSERT_EQ(receipts.size(), devices.size());
    for (std::size_t index = 0; index < devices.size(); ++index)
    {
        EXPECT_EQ(receipts[index].device, devices[index]);
        EXPECT_TRUE(receipts[index].runtime_reset_invoked);
        EXPECT_EQ(receipts[index].runtime_post_reset_state, DeviceRuntimePostResetState::Quiescent);
        EXPECT_EQ(backend->nativeExecutionContextStorageBytes(devices[index].ordinal), 0u);
    }

    // Old certificates cannot lend a stale context lease to the successor.
    // Its first captured native kernels must create fresh canonical ownership.
    auto successor_memory = memoryFor(request);
    EXPECT_NO_THROW(PlanningLocalTPMeasurement::measure(request, successor_memory));
    expectRetired(request, *successor_memory);
    for (const auto device : devices)
        EXPECT_GT(backend->nativeExecutionContextStorageBytes(device.ordinal), 0u);
}
}

TEST(PlanningExecutionMeasurementIntegration, CUDA_NativeLocalTPContextStorageLifetime)
{
    proveContextStorageLifetime(2048, 64);
}

/** @test The failing large prefill sample owns exact context bytes in both native generations. */
TEST(PlanningExecutionMeasurementIntegration, CUDA_NativeLocalTPContextStoragePrefillBucket)
{
    proveContextStorageLifetime(5120, 384);
}

#endif
#ifdef HAVE_ROCM
TEST(PlanningExecutionMeasurementIntegration, ROCm_NativeLocalTP)
{
    ensureAMDFactoryRegistered();
    prove(DeviceType::ROCm);
}
#endif
