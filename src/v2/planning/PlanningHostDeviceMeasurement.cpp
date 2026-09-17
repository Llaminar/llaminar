/**
 * @file PlanningHostDeviceMeasurement.cpp
 * @brief Exact-mechanism GPU/host sampling with rank-owned first touch and event-owned completion.
 *
 * PMA claims outlive every buffer. Mapped host storage is allocated on the rank
 * setup thread; GPU-worker affinity cannot silently relocate the sample pages.
 * Capture, host publication and byte verification are outside all timed native
 * intervals. No weight, KV, sampler or live inference state is constructed.
 */
#include "PlanningHostDeviceMeasurement.h"
#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/GPUGraphMemoryContract.h"
#include "backends/IWorkerGPUContext.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "transfer/TransferEngine.h"
#include <algorithm>
#include <array>
#include <climits>
#include <cstring>
#include <set>
#include <stdexcept>
#include <tuple>

namespace llaminar2
{
    namespace
    {
        /** @brief Fail at the missing evidence/operation rather than fabricate a bandwidth. */
        void require(bool condition, const char *detail)
        {
            if (!condition) throw std::invalid_argument(std::string("Planning GPU/host service: ") + detail);
        }
        /** @brief Fill reproducible nonconstant bytes; a new salt publishes new input before each H2D replay. */
        void fill(void *storage, size_t bytes, uint32_t salt)
        {
            auto *target = static_cast<uint8_t *>(storage);
            for (size_t index = 0; index < bytes; ++index)
            {
                uint32_t value = static_cast<uint32_t>(index) ^ (salt * 0x9e3779b9u);
                value ^= value >> 16;
                value *= 0x7feb352du;
                value ^= value >> 15;
                target[index] = static_cast<uint8_t>(value);
            }
        }
    }

    PlanningHostDeviceRequest PlanningHostDeviceRequest::fromInventory(const RankInventory &rank,
        DeviceId device, size_t bulk_bytes)
    {
        require(rank.rank >= 0 && rank.node_id >= 0 && rank.cpu.numa_node >= -1 && rank.cpu.memory_bytes > 0 &&
            rank.cpu_cores > 0, "missing rank host identity");
        require((device.is_cuda() || device.is_rocm()) && device.ordinal >= 0 && bulk_bytes > 1 && bulk_bytes <= INT_MAX,
            "invalid GPU or bounded bulk geometry");
        const DeviceInfo *found = nullptr;
        for (const auto &gpu : rank.gpus)
            if (gpu.type == device.type && gpu.local_device_id == device.ordinal)
            {
                require(!found && !gpu.uuid.empty(), "ambiguous GPU physical identity");
                found = &gpu;
            }
        require(found, "GPU is not visible on this rank");
        return {rank.rank, rank.node_id, rank.cpu.numa_node, device, found->uuid, bulk_bytes};
    }

    void PlanningHostDeviceMeasurement::contributeMemory(const PlanningHostDeviceRequest &request,
        PhysicalMemoryResource host, PhysicalMemoryResource execution, PhysicalMemoryPlanBuilder &builder)
    {
        require(host.world_rank == request.rank() && execution.world_rank == request.rank() &&
            host.device == DeviceId::cpu() && execution.device == request.device(), "BOM resource identity mismatch");
        const auto host_bytes = TransferEngine::mappedHostRegionAllocationBytes(request.bulkBytes());
        builder.add(host, PhysicalMemoryOwner::ActivationTransportStaging, 2 * host_bytes);
        builder.add(execution, PhysicalMemoryOwner::ActivationTransportStaging, request.bulkBytes());
        // One readback graph can coexist with the current measured graph. Each
        // new coordinate retires the latter before another is instantiated.
        builder.add(execution, PhysicalMemoryOwner::NativeGraphExecutable,
            2 * GPUGraphMemoryContract::reservationBytesPerExecutable(request.device()));
    }

    PlanningHostDeviceObservations PlanningHostDeviceMeasurement::measure(const PlanningHostDeviceRequest &request,
        const std::shared_ptr<PhysicalMemoryAuthority> &memory)
    {
        require(bool(memory), "missing physical admission");
        const auto device = request.device();
        auto &worker = GPUDeviceContextPool::instance().getContext(device);
        require(!worker.ownsCurrentThread(), "host pages must be first-touched on the rank setup caller");
        auto &transfer = TransferEngine::instance();
        const size_t host_bytes = TransferEngine::mappedHostRegionAllocationBytes(request.bulkBytes());
        auto host_claim = memory->claimNewAllocation(DeviceId::cpu(), PhysicalMemoryOwner::ActivationTransportStaging, 2 * host_bytes);
        auto device_claim = memory->claimNewAllocation(device, PhysicalMemoryOwner::ActivationTransportStaging, request.bulkBytes());
        const std::array devices{device};
        auto input = transfer.allocateMappedHostRegion(request.bulkBytes(), devices);
        auto output = transfer.allocateMappedHostRegion(request.bulkBytes(), devices);
        require(input && output && input->sizeBytes() == host_bytes && output->sizeBytes() == host_bytes,
            "mapped allocation extent differs from canonical geometry");
        // The typed pool prepares module functions before any capture. No
        // first-use loader synchronization can contaminate a timed operation.
        const auto lanes = transfer.allocatePersistentTransferExecutionLanes(1, device, "planning_host_device");
        PlanningHostDeviceObservations result{request, {}};
        worker.submitAndWait([&] {
            auto buffer = transfer.allocateDeviceTransferBuffer(request.bulkBytes(), device);
            require(bool(buffer), "device staging allocation failed");
            auto *backend = getBackendFor(device);
            require(backend != nullptr, "missing exact GPU backend");
            const auto &lane = lanes.front();
            const auto graph_bytes = GPUGraphMemoryContract::reservationBytesPerExecutable(device);
            for (size_t payload : {size_t{1}, request.bulkBytes()})
            {
                auto readback_claim = memory->reserveNewAllocations(device, PhysicalMemoryOwner::NativeGraphExecutable, graph_bytes);
                auto readback = worker.createGraphCapture(lane.stream());
                require(bool(readback), "readback graph creation failed");
                {
                    ScopedBackendGraphCapture capture(worker, *readback, "planning host-device byte verification");
                    require(capture.begin(), "readback capture failed");
                    transfer.enqueueDeviceToMappedHost(*buffer, 0, *output, 0, payload, device, lane.stream());
                    capture.finish();
                }
                require(readback->nodeCount() && readback->instantiate() &&
                    GPUGraphMemoryContract::acceptsFamilyObservation(device, readback->residentMemoryBytes(), graph_bytes),
                    "readback graph incomplete or outside admission");
                for (auto mechanism : {PlanningHostTransferMechanism::DMA, PlanningHostTransferMechanism::MappedKernel})
                    for (auto direction : {MappedTransferDirection::HostToDevice, MappedTransferDirection::DeviceToHost})
                    {
                        uint32_t salt = 7;
                        fill(input->mutableHostData(), payload, salt);
                        // Initial upload precedes warmup on the exact same
                        // stream; it is outside every timed interval.
                        if (direction == MappedTransferDirection::DeviceToHost)
                            transfer.enqueueMappedHostToDevice(*input, 0, *buffer, 0, payload, device, lane.stream());
                        auto claim = memory->reserveNewAllocations(device, PhysicalMemoryOwner::NativeGraphExecutable, graph_bytes);
                        auto graph = worker.createGraphCapture(lane.stream());
                        require(bool(graph), "copy graph creation failed");
                        {
                            ScopedBackendGraphCapture capture(worker, *graph, "planning host-device primitive");
                            require(capture.begin(), "copy capture failed");
                            const bool inbound = direction == MappedTransferDirection::HostToDevice;
                            const auto &mapped = inbound ? *input : *output;
                            if (mechanism == PlanningHostTransferMechanism::MappedKernel)
                                transfer.enqueueMappedKernelCopy(lane, direction, *buffer, 0, mapped, 0, payload);
                            else if (inbound)
                                transfer.enqueueMappedHostToDevice(mapped, 0, *buffer, 0, payload, device, lane.stream());
                            else transfer.enqueueDeviceToMappedHost(*buffer, 0, mapped, 0, payload, device, lane.stream());
                            capture.finish();
                        }
                        require(graph->nodeCount() && graph->instantiate() &&
                            GPUGraphMemoryContract::acceptsFamilyObservation(device, graph->residentMemoryBytes(), graph_bytes),
                            "copy graph incomplete or outside admission");
                        const PlanningMeasurementWork work(PlanningWorkUnit::Bytes, payload,
                            "GPU-host byte primitive; device=" + device.toString() +
                            "; first-touch-rank=" + std::to_string(request.rank()) +
                            "; host-numa=" + std::to_string(request.hostNumaNode()) +
                            "; mechanism=" + std::to_string(int(mechanism)) + "; direction=" + std::to_string(int(direction)) +
                            "; not-complete-activation-packet-service");
                        auto observation = PlanningExecutionMeasurement::gpuPublished(work, *backend, device, *graph, [&] {
                            if (direction == MappedTransferDirection::HostToDevice)
                                fill(input->mutableHostData(), payload, ++salt);
                            else std::memset(output->mutableHostData(), 0, payload);
                        });
                        if (direction == MappedTransferDirection::HostToDevice)
                        {
                            // Separate retained diagnostic readback, after the
                            // measurement event. Its time is deliberately not
                            // part of either direction's reported service.
                            (void)PlanningExecutionMeasurement::gpu(
                                PlanningMeasurementWork(PlanningWorkUnit::Bytes, payload, "untimed sample byte proof"),
                                *backend, device, *readback);
                        }
                        require(std::memcmp(input->mutableHostData(), output->mutableHostData(), payload) == 0,
                            "captured copy did not preserve exact source bytes");
                        result.phases.push_back({mechanism, direction, payload, graph->nodeCount(), std::move(observation)});
                    }
            }
        });
        validate(result);
        return result;
    }

    void PlanningHostDeviceMeasurement::validate(const PlanningHostDeviceObservations &observed)
    {
        require(observed.phases.size() == 8, "incomplete host-device phase family");
        std::set<std::tuple<PlanningHostTransferMechanism, MappedTransferDirection, size_t>> seen;
        for (const auto &phase : observed.phases)
        {
            require((phase.mechanism == PlanningHostTransferMechanism::DMA || phase.mechanism == PlanningHostTransferMechanism::MappedKernel) &&
                (phase.direction == MappedTransferDirection::HostToDevice || phase.direction == MappedTransferDirection::DeviceToHost) &&
                (phase.payload_bytes == 1 || phase.payload_bytes == observed.request.bulkBytes()) && phase.graph_nodes > 0,
                "invalid host-device mechanism/direction/geometry or empty graph");
            require(phase.service.unit() == PlanningWorkUnit::Bytes &&
                phase.service.completedWork() == double(phase.payload_bytes) * PlanningExecutionMeasurement::kTimedInvocations,
                "host-device receipt contains foreign work units or count");
            require(seen.emplace(phase.mechanism, phase.direction, phase.payload_bytes).second, "duplicate host-device phase");
        }
    }
}
