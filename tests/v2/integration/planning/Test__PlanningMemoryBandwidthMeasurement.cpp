/**
 * @file Test__PlanningMemoryBandwidthMeasurement.cpp
 * @brief Real cache-exceeding CPU/CUDA/ROCm observation and retirement proofs.
 *
 * The canonical runtime inventory sizes the streams and the canonical PMA
 * admits them. Assertions are functional: useful bytes, real captured nodes,
 * finite completed service, and empty claims on retirement. Measured GB/s is
 * diagnostic output, never a performance threshold in production preflight.
 */
#include "planning/PlanningMemoryBandwidthMeasurement.h"
#include "planning/PlanningHostDeviceMeasurement.h"
#include "planning/PlanningObservedResource.h"
#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IWorkerGPUContext.h"
#include "transfer/TransferEngine.h"
#include "utils/MPIContext.h"
#include <gtest/gtest.h>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <omp.h>

using namespace llaminar2;

namespace
{
    /** @brief Prove both copy mechanisms, both directions, failure retirement and exact public bounds. */
    void proveHostDevice(DeviceId device)
    {
        const auto mpi = MPIContextFactory::global();
        const auto inventory = mpi->clusterInventory();
        ASSERT_NE(inventory, nullptr);
        const auto &rank = inventory->ranks.at(mpi->rank());
        // Odd payload covers vector tails while the collector separately uses
        // the model's bounded activation payload. No throughput gate is imposed.
        const auto request = PlanningHostDeviceRequest::fromInventory(rank, device, 4099);
        PhysicalMemoryPlanBuilder builder;
        PlanningHostDeviceMeasurement::contributeMemory(request, planningObservedResource(rank, DeviceId::cpu()),
            planningObservedResource(rank, device), builder);
        auto memory = std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(builder.build()), rank.rank);
        const auto run = [&] {
            const auto observed = PlanningHostDeviceMeasurement::measure(request, memory);
            EXPECT_EQ(observed.request, request);
            EXPECT_NO_THROW(PlanningHostDeviceMeasurement::validate(observed));
            for (const auto &phase : observed.phases)
            {
                EXPECT_GT(phase.graph_nodes, 0u);
                EXPECT_GT(phase.service.unitsPerSecond(), 0);
                EXPECT_EQ(phase.service.completedWork(), 3.0 * phase.payload_bytes);
            }
        };
        run();
        for (auto allocator : {DeviceId::cpu(), device})
        {
            EXPECT_EQ(memory->claimedBytes(allocator, PhysicalMemoryOwner::ActivationTransportStaging,
                PhysicalMemoryMaterializationKind::NewAllocation), 0u);
            EXPECT_EQ(memory->reservedBytes(allocator, PhysicalMemoryOwner::NativeGraphExecutable), 0u);
        }
        {
            auto occupied = memory->reserveNewAllocations(device, PhysicalMemoryOwner::ActivationTransportStaging,
                memory->plannedBytes(device, PhysicalMemoryOwner::ActivationTransportStaging));
            EXPECT_THROW(run(), std::exception);
        }
        EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(), PhysicalMemoryOwner::ActivationTransportStaging,
            PhysicalMemoryMaterializationKind::NewAllocation), 0u);
        auto &worker = GPUDeviceContextPool::instance().getContext(device);
        worker.submitAndWait([&] {
            EXPECT_THROW(PlanningHostDeviceMeasurement::measure(request, memory), std::invalid_argument);
        });
        run(); // A pre-allocation failure must not leave a poisoned/reused native binding.
        {
            auto &transfer = TransferEngine::instance();
            const auto host_bytes = TransferEngine::mappedHostRegionAllocationBytes(request.bulkBytes());
            auto host_claim = memory->claimNewAllocation(DeviceId::cpu(), PhysicalMemoryOwner::ActivationTransportStaging, host_bytes);
            auto gpu_claim = memory->claimNewAllocation(device, PhysicalMemoryOwner::ActivationTransportStaging, request.bulkBytes());
            const std::array devices{device};
            auto mapped = transfer.allocateMappedHostRegion(request.bulkBytes(), devices);
            const auto lanes = transfer.allocatePersistentTransferExecutionLanes(1, device, "planning_host_device");
            worker.submitAndWait([&] {
                auto scratch = transfer.allocateDeviceTransferBuffer(request.bulkBytes(), device);
                const auto &lane = lanes.front();
                EXPECT_THROW(transfer.enqueueMappedKernelCopy({}, MappedTransferDirection::HostToDevice,
                    *scratch, 0, *mapped, 0, 1), std::invalid_argument);
                EXPECT_THROW(transfer.enqueueMappedKernelCopy(lane, static_cast<MappedTransferDirection>(99),
                    *scratch, 0, *mapped, 0, 1), std::invalid_argument);
                EXPECT_THROW(transfer.enqueueMappedKernelCopy(lane, MappedTransferDirection::HostToDevice,
                    *scratch, 0, *mapped, 0, 0), std::invalid_argument);
                EXPECT_THROW(transfer.enqueueMappedKernelCopy(lane, MappedTransferDirection::HostToDevice,
                    *scratch, request.bulkBytes(), *mapped, 0, 1), std::out_of_range);
                EXPECT_THROW(transfer.enqueueMappedKernelCopy(lane, MappedTransferDirection::DeviceToHost,
                    *scratch, 0, *mapped, mapped->sizeBytes(), 1), std::out_of_range);
                EXPECT_THROW(transfer.enqueueMappedKernelCopy(lane, MappedTransferDirection::DeviceToHost,
                    *scratch, std::numeric_limits<size_t>::max(), *mapped, 0, 2), std::out_of_range);
            });
        }
        EXPECT_EQ(memory->claimedBytes(device, PhysicalMemoryOwner::ActivationTransportStaging,
            PhysicalMemoryMaterializationKind::NewAllocation), 0u);
        EXPECT_EQ(memory->reservedBytes(device, PhysicalMemoryOwner::NativeGraphExecutable), 0u);
    }

    /** @brief Use real inventory and owning workers without changing launch affinity or ISA. */
    void prove(DeviceId device)
    {
        const auto mpi = MPIContextFactory::global();
        const auto inventory = mpi->clusterInventory();
        ASSERT_NE(inventory, nullptr);
        const auto &rank = inventory->ranks.at(mpi->rank());
        if (!hasCPUBackend()) initCPUBackend(-1);
        const auto request = PlanningMemoryBandwidthRequest::fromInventory(rank, device);
        const PhysicalMemoryResource host{rank.rank, DeviceId::cpu(), rank.cpu.memory_bytes, rank.cpu.free_memory_bytes};
        auto execution = host;
        if (device.is_gpu())
        {
            const auto found = std::find_if(rank.gpus.begin(), rank.gpus.end(), [&](const auto &gpu) {
                return gpu.type == device.type && gpu.local_device_id == device.ordinal;
            });
            ASSERT_NE(found, rank.gpus.end());
            execution = {rank.rank, device, found->memory_bytes, found->free_memory_bytes};
        }
        PhysicalMemoryPlanBuilder builder;
        PlanningMemoryBandwidthMeasurement::contributeMemory(request, host, execution, builder);
        const auto memory = std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(builder.build()), rank.rank);
        if (device.is_cpu())
        {
            // The observer may not label a smaller workshare as the inventory's
            // full CPU endpoint. Check rejection before any allocation/capture.
            const int workers = omp_get_max_threads();
            omp_set_num_threads(workers == 1 ? 2 : 1);
            EXPECT_THROW(PlanningMemoryBandwidthMeasurement::measure(request, memory), std::invalid_argument);
            omp_set_num_threads(workers);
        }
        const auto run = [&] {
            const auto result = PlanningMemoryBandwidthMeasurement::measure(request, memory);
            EXPECT_EQ(result.graph_nodes != 0, device.is_gpu());
            EXPECT_EQ(result.service.unit(), PlanningWorkUnit::Bytes);
            EXPECT_EQ(result.service.completedWork(), double(request.usefulBytes()) * PlanningExecutionMeasurement::kTimedInvocations);
            EXPECT_TRUE(std::isfinite(result.service.unitsPerSecond()));
            EXPECT_GT(result.service.unitsPerSecond(), 0);
            EXPECT_GE(request.streamBytes(), 4 * request.cacheBytes());
            std::cout << "STREAMING_BANDWIDTH device=" << device.toString()
                << " workers=" << request.workers() << " cache_bytes=" << request.cacheBytes()
                << " stream_bytes=" << request.streamBytes()
                << " useful_GB_per_s=" << result.service.unitsPerSecond() / 1e9 << '\n';
        };
        const auto submit = [&] {
            if (device.is_gpu()) GPUDeviceContextPool::instance().getContext(device).submitAndWait(run);
            else run();
        };
        submit();
        for (auto allocator : {DeviceId::cpu(), device})
        {
            EXPECT_EQ(memory->claimedBytes(allocator, PhysicalMemoryOwner::ExecutionWorkspace,
                PhysicalMemoryMaterializationKind::NewAllocation), 0u);
            EXPECT_EQ(memory->reservedBytes(allocator, PhysicalMemoryOwner::NativeGraphExecutable), 0u);
        }
        // Failed physical admission must precede payload creation. A retry is a
        // fresh test transaction, not a production fallback or changed geometry.
        {
            auto occupied = memory->reserveNewAllocations(device, PhysicalMemoryOwner::ExecutionWorkspace,
                memory->plannedBytes(device, PhysicalMemoryOwner::ExecutionWorkspace));
            EXPECT_THROW(submit(), std::exception);
        }
        submit();
        EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(), PhysicalMemoryOwner::ExecutionWorkspace,
            PhysicalMemoryMaterializationKind::NewAllocation), 0u);
        EXPECT_EQ(memory->claimedBytes(device, PhysicalMemoryOwner::ExecutionWorkspace,
            PhysicalMemoryMaterializationKind::NewAllocation), 0u);
        EXPECT_EQ(memory->reservedBytes(device, PhysicalMemoryOwner::NativeGraphExecutable), 0u);
    }
}

TEST(PlanningExecutionMeasurementIntegration, CPU_StreamingBandwidth) { prove(DeviceId::cpu()); }
#ifdef HAVE_CUDA
TEST(PlanningExecutionMeasurementIntegration, CUDA_StreamingBandwidth) { prove(DeviceId::cuda(0)); }
TEST(PlanningExecutionMeasurementIntegration, CUDA_HostDeviceService) { proveHostDevice(DeviceId::cuda(0)); }
#endif
#ifdef HAVE_ROCM
TEST(PlanningExecutionMeasurementIntegration, ROCm_StreamingBandwidth) { prove(DeviceId::rocm(0)); }
TEST(PlanningExecutionMeasurementIntegration, ROCm_HostDeviceService) { proveHostDevice(DeviceId::rocm(0)); }
#endif
