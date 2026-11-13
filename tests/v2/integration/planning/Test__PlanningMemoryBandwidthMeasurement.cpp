/**
 * @file Test__PlanningMemoryBandwidthMeasurement.cpp
 * @brief Real cache-exceeding CPU/CUDA/ROCm observation and retirement proofs.
 *
 * The canonical runtime inventory sizes the streams and the canonical PMA
 * admits them. Assertions are functional: useful bytes, real captured nodes,
 * finite completed service, and empty claims on retirement. Measured GB/s is
 * diagnostic output, never a performance threshold in production preflight.
 */
#include "../../fixtures/transfer/ComponentTransferExecutionPoolFixtureAccess.h"
#include "planning/PlanningMemoryBandwidthMeasurement.h"
#include "planning/PlanningHostDeviceMeasurement.h"
#include "planning/PlanningExecutionMeasurement.h"
#include "planning/PlanningGPUClockBuffer.h"
#include "backends/GPUGraphMemoryContract.h"
#include "planning/PlanningObservedResource.h"
#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IWorkerGPUContext.h"
#include "transfer/TransferEngine.h"
#include "utils/MPIContext.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <omp.h>
#include "planning/RankHardwareOwnership.h"

using namespace llaminar2;

namespace
{
    /** @brief Prove both copy mechanisms, both directions, failure retirement and exact public bounds. */
    void proveHostDevice(DeviceId device, bool require_captured_clock=false)
    {
        const auto mpi = MPIContextFactory::global();
        const auto inventory = mpi->clusterInventory();
        ASSERT_NE(inventory, nullptr);
        const auto &rank = inventory->ranks.at(mpi->rank());
        const auto local = std::find_if(rank.gpus.begin(), rank.gpus.end(), [&](const auto &gpu) {
            return gpu.type == device.type && !gpu.uuid.empty() && rankOwnsGPUDevice(rank, gpu);
        });
        ASSERT_NE(local, rank.gpus.end()) << "Host-device measurement requires its original NUMA-owned GPU";
        device = {local->type, local->local_device_id};
        if (!hasCPUBackend()) initCPUBackend(rank.cpu.numa_node);
        // Odd payload covers vector tails while the collector separately uses
        // the model's bounded activation payload. No throughput gate is imposed.
        const auto request = PlanningHostDeviceRequest::fromInventory(*inventory, {rank.rank, device}, {rank.rank}, 4099);
        PhysicalMemoryPlanBuilder builder;
        PlanningHostDeviceMeasurement::contributeHostMemory(request, planningObservedResource(rank, DeviceId::cpu()), builder);
        PlanningHostDeviceMeasurement::contributeExecutionMemory(request, planningObservedResource(rank, DeviceId::cpu()), planningObservedResource(rank, device), builder);
        auto memory = std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(builder.build()), rank.rank);
        if(require_captured_clock){
            EXPECT_EQ(memory->plannedBytes(DeviceId::cpu(),PhysicalMemoryOwner::ExecutionWorkspace),PlanningNativeGPUClockBuffer::hostAllocationBytes(2));
            EXPECT_EQ(memory->plannedBytes(device,PhysicalMemoryOwner::ExecutionWorkspace),PlanningNativeGPUClockBuffer::deviceAllocationBytes(2));
            EXPECT_EQ(memory->plannedBytes(device,PhysicalMemoryOwner::NativeGraphExecutable),
                (PlanningExecutionMeasurement::kPreparedGraphCount+1)*GPUGraphMemoryContract::reservationBytesPerExecutable(device));
            ::testing::Test::RecordProperty("host_device_original_gpu_uuid",local->uuid);
            ::testing::Test::RecordProperty("host_device_executor_rank",rank.rank);
            ::testing::Test::RecordProperty("host_device_first_touch_rank",request.hostRank());
            ::testing::Test::RecordProperty("host_device_first_touch_numa",request.hostNumaNode());
        }
        const auto run = [&] {
            const auto observed = PlanningHostDeviceMeasurement::collect(request, mpi, memory);
            ASSERT_TRUE(observed.has_value());
            EXPECT_EQ(observed->request, request);
            EXPECT_NO_THROW(PlanningHostDeviceMeasurement::validate(*observed));
            for (const auto &phase : observed->phases)
            {
                EXPECT_GT(phase.graph_nodes, 0u);
                EXPECT_GT(phase.service.unitsPerSecond(), 0);
                EXPECT_EQ(phase.service.completedWork(), 3.0 * phase.payload_bytes);
                if(require_captured_clock){
                    EXPECT_NE(phase.service.provenance().find("sum of captured fixed-device-clock intervals"),std::string::npos);
                    EXPECT_NE(phase.service.provenance().find("fresh host publication before every invocation"),std::string::npos);
                }
            }
            for(const auto allocator:{DeviceId::cpu(),device})
                for(const auto owner:{PhysicalMemoryOwner::ActivationTransportStaging,
                    PhysicalMemoryOwner::ExecutionWorkspace,PhysicalMemoryOwner::NativeGraphExecutable}){
                    EXPECT_EQ(memory->claimedBytes(allocator,owner,PhysicalMemoryMaterializationKind::NewAllocation),0u);
                    EXPECT_EQ(memory->reservedBytes(allocator,owner),0u);
                }
        };
        // Serial calibration reuses the original worker stream across complete
        // captures and byte-proof retirement, without an auxiliary pool lifetime.
        auto &calibration_worker = GPUDeviceContextPool::instance().getContext(device);
        void *original_stream = nullptr;
        calibration_worker.submitAndWait([&] { original_stream = calibration_worker.defaultStream(); });
        ASSERT_NE(original_stream, nullptr);
        for (int replay = 0; replay < 2; ++replay)
        {
            run();
            calibration_worker.submitAndWait([&] {
                EXPECT_EQ(calibration_worker.defaultStream(), original_stream);
            });
        }
        for (auto allocator : {DeviceId::cpu(), device})
        {
            EXPECT_EQ(memory->claimedBytes(allocator, PhysicalMemoryOwner::ActivationTransportStaging,
                PhysicalMemoryMaterializationKind::NewAllocation), 0u);
            EXPECT_EQ(memory->reservedBytes(allocator, PhysicalMemoryOwner::NativeGraphExecutable), 0u);
        }
        for (const auto allocator : {DeviceId::cpu(), device})
        {
            auto occupied = memory->reserveNewAllocations(allocator, PhysicalMemoryOwner::ActivationTransportStaging,
                memory->plannedBytes(allocator, PhysicalMemoryOwner::ActivationTransportStaging));
            EXPECT_THROW(run(), std::exception);
        }
        EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(), PhysicalMemoryOwner::ActivationTransportStaging,
            PhysicalMemoryMaterializationKind::NewAllocation), 0u);
        auto &worker = GPUDeviceContextPool::instance().getContext(device);
        worker.submitAndWait([&] {
            try
            {
                (void)PlanningHostDeviceMeasurement::collect(request, mpi, memory);
                FAIL() << "GPU worker cannot replace the original setup caller";
            }
            catch (const std::exception &error)
            {
                EXPECT_NE(std::string(error.what()).find("rank setup caller"), std::string::npos);
            }
        });
        run(); // A pre-allocation failure must not leave a poisoned/reused native binding.
        {
            auto &transfer = TransferEngine::instance();
            const auto host_bytes = TransferEngine::mappedHostRegionAllocationBytes(request.bulkBytes());
            auto host_claim = memory->claimNewAllocation(DeviceId::cpu(), PhysicalMemoryOwner::ActivationTransportStaging, host_bytes);
            auto gpu_claim = memory->claimNewAllocation(device, PhysicalMemoryOwner::ActivationTransportStaging, request.bulkBytes());
            const std::array devices{device};
            auto mapped = transfer.allocateMappedHostRegion(request.bulkBytes(), devices);
            const auto lanes = ::llaminar2::test::ComponentTransferExecutionPoolFixtureAccess::allocate(transfer,1, device, "planning_host_device");
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

#ifdef HAVE_CUDA
/** @test Captured DMA/mapped service retains late byte publication and repeated original-worker retirement. */
TEST(PlanningExecutionMeasurementIntegration, CUDA_HostDeviceCapturedClockReplay)
{proveHostDevice(DeviceId::cuda(0),true);}
#endif
#ifdef HAVE_ROCM
/** @test HIP host-device calibration uses compact device clocks rather than enqueue-spanning events. */
TEST(PlanningExecutionMeasurementIntegration, ROCm_HostDeviceCapturedClockReplay)
{proveHostDevice(DeviceId::rocm(0),true);}
#endif
