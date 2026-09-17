/**
 * @file PlanningMemoryBandwidthMeasurement.cpp
 * @brief Cache-exceeding production stream measurement with bounded, PMA-owned setup.
 *
 * Initialization first-touches host pages with the same CPU workshare used by
 * inference. Nonconstant inputs prevent compressible fill patterns from
 * exaggerating GPU bandwidth. Exact native events delimit GPU replay; source
 * uploads, graph construction and warmup are outside the service observation.
 */
#include "planning/PlanningMemoryBandwidthMeasurement.h"
#include "planning/RankHardwareOwnership.h"
#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/GPUGraphMemoryContract.h"
#include "backends/IWorkerGPUContext.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "kernels/KernelFactory.h"
#include "tensors/TensorClasses.h"
#include "transfer/TransferEngine.h"
#include "utils/CPUFeatures.h"
#include <algorithm>
#include <bit>
#include <climits>
#include <limits>
#include <omp.h>
#include <set>
#include <stdexcept>

namespace llaminar2
{
    namespace
    {
        /** @brief Treat incomplete preparation/execution as an error, never zero bandwidth. */
        void require(bool condition, const char *detail)
        {
            if (!condition) throw std::runtime_error(std::string("Planning memory bandwidth: ") + detail);
        }
        /** @return Checked cache arithmetic; these are geometry inputs, not free-byte accounting. */
        size_t multiply(size_t a, size_t b)
        {
            if (b && a > std::numeric_limits<size_t>::max() / b)
                throw std::overflow_error("Planning streaming geometry overflow");
            return a * b;
        }
        /** @return Reproducible noncompressible positive FP32 mantissas, without a random-state owner. */
        float value(uint32_t index)
        {
            index ^= index >> 16;
            index *= 0x7feb352du;
            index ^= index >> 15;
            index *= 0x846ca68bu;
            index ^= index >> 16;
            return std::bit_cast<float>(0x3f800000u | (index & 0x007fffffu));
        }
    }

    PlanningMemoryBandwidthRequest PlanningMemoryBandwidthRequest::fromInventory(const RankInventory &rank, DeviceId device)
    {
        if (rank.rank < 0 || (!device.is_cpu() && !device.is_gpu()))
            throw std::invalid_argument("Streaming measurement requires a rank-bound CPU/CUDA/ROCm endpoint");
        size_t cache = 0;
        int workers = 0;
        CPUExecutionGeometry cpu;
        if (device.is_cpu())
        {
            if (device != DeviceId::cpu() || rank.cpu_cores <= 0 || !rank.cpu_execution.isValid())
                throw std::invalid_argument("Streaming CPU measurement requires exact published worker/cache geometry");
            std::set<int> sockets;
            for (const auto &socket : rank.cpu_socket_info)
                if (rankOwnsCPUNode(rank, socket.numa_node)) sockets.insert(socket.socket_id);
            if (sockets.empty())
                throw std::invalid_argument("Streaming CPU measurement has no owned cache domains");
            cpu = rank.cpu_execution;
            workers = rank.cpuWorkerThreads();
            if (!rank.cpu.last_level_cache_bytes)
                throw std::invalid_argument("Streaming CPU measurement requires observed aggregate cache coverage");
            const size_t shared = rank.cpu.last_level_cache_bytes;
            const size_t private_bytes = multiply(cpu.cache.private_l2_bytes, static_cast<size_t>(workers));
            if (shared > std::numeric_limits<size_t>::max() - private_bytes)
                throw std::overflow_error("Planning CPU cache coverage overflow");
            cache = shared + private_bytes;
        }
        else
        {
            const auto found = std::find_if(rank.gpus.begin(), rank.gpus.end(), [&](const auto &gpu) {
                return gpu.type == device.type && gpu.local_device_id == device.ordinal;
            });
            if (found == rank.gpus.end() || found->last_level_cache_bytes == 0)
                throw std::invalid_argument("Streaming GPU measurement requires an observed endpoint and cache size");
            cache = found->last_level_cache_bytes;
        }
        // Four full cache capacities per independent stream leaves no complete
        // input resident between passes. Align upward for ordinary tensor/SIMD
        // access; do not shrink this demand to fit memory and call it streaming.
        const size_t raw = multiply(cache, 4);
        if (raw > std::numeric_limits<size_t>::max() - 4095)
            throw std::overflow_error("Planning stream alignment overflow");
        const size_t stream = (raw + 4095) / 4096 * 4096;
        (void)multiply(stream, 3);
        if (stream / sizeof(float) > INT_MAX)
            throw std::overflow_error("Planning stream exceeds production residual-add geometry");
        return {device, rank.rank, cache, stream, workers, cpu};
    }

    void PlanningMemoryBandwidthMeasurement::contributeMemory(const PlanningMemoryBandwidthRequest &request,
        PhysicalMemoryResource host, PhysicalMemoryResource execution, PhysicalMemoryPlanBuilder &builder)
    {
        if (host.world_rank != request.rank() || execution.world_rank != request.rank() ||
            host.device != DeviceId::cpu() || execution.device != request.device())
            throw std::invalid_argument("Streaming measurement BOM does not match its exact rank/device");
        builder.add(host, PhysicalMemoryOwner::ExecutionWorkspace, request.usefulBytes());
        if (request.device().is_gpu())
        {
            builder.add(execution, PhysicalMemoryOwner::ExecutionWorkspace, request.usefulBytes());
            builder.add(execution, PhysicalMemoryOwner::NativeGraphExecutable,
                GPUGraphMemoryContract::reservationBytesPerExecutable(request.device()));
        }
    }

    PlanningMemoryBandwidthObservation PlanningMemoryBandwidthMeasurement::measure(
        const PlanningMemoryBandwidthRequest &request, const std::shared_ptr<PhysicalMemoryAuthority> &memory)
    {
        const auto device = request.device();
        if (!memory) throw std::invalid_argument("Streaming measurement requires canonical physical admission");
        if (device.is_cpu())
        {
            if (omp_in_parallel() || omp_get_dynamic() || omp_get_max_threads() != request.workers() ||
                omp_get_thread_limit() < request.workers() || CPUExecutionGeometry::local() != request.cpuGeometry())
                throw std::invalid_argument("Streaming CPU observer differs from its published workshare/cache geometry");
        }
        else if (!GPUDeviceContextPool::instance().getContext(device).ownsCurrentThread())
            throw std::invalid_argument("Streaming GPU measurement requires its ordinary owning worker");
        auto host_claim = memory->claimNewAllocation(DeviceId::cpu(), PhysicalMemoryOwner::ExecutionWorkspace, request.usefulBytes());
        std::optional<PhysicalMemoryAllocationLease> gpu_claim;
        if (device.is_gpu()) gpu_claim.emplace(memory->claimNewAllocation(device, PhysicalMemoryOwner::ExecutionWorkspace, request.usefulBytes()));
        const size_t count = request.streamBytes() / sizeof(float);
        // Ordinary tensor construction zeroes storage on the calling thread.
        // Adopt untouched, page-owned mappings instead so the workshare below
        // really determines physical NUMA placement, even with allocator reuse.
        auto input_pages = AlignedVector<float>::pageMappedUninitialized(count);
        auto residual_pages = AlignedVector<float>::pageMappedUninitialized(count);
        auto output_pages = AlignedVector<float>::pageMappedUninitialized(count);
        float *a = input_pages.data(), *b = residual_pages.data(), *c = output_pages.data();
        // GPU workers have their own OpenMP context. Host first-touch setup is
        // intentionally untimed there; CPU measurement retains its real team.
#pragma omp parallel for schedule(static) if(device.is_cpu())
        for (size_t index = 0; index < count; ++index)
        {
            a[index] = value(static_cast<uint32_t>(index));
            b[index] = value(static_cast<uint32_t>(index) ^ 0x9e3779b9u);
            c[index] = 0;
        }
        FP32Tensor input({1, count}, std::move(input_pages));
        FP32Tensor residual({1, count}, std::move(residual_pages));
        FP32Tensor output({1, count}, std::move(output_pages));
        auto kernel = llaminar::v2::kernels::KernelFactory::createResidualAdd(&input, device.type, device.ordinal);
        require(bool(kernel), "production residual kernel is missing");
        const auto execute = [&] { return kernel->apply_tensor(&input, &residual, &output, count, nullptr, device.toKernelDeviceIndex()); };
        std::string identity = "streaming FP32 residual add; device=" + device.toString() +
            "; observed_cache_bytes=" + std::to_string(request.cacheBytes()) +
            "; bytes_per_stream=" + std::to_string(request.streamBytes()) +
            "; traffic=two-reads-one-write; useful-bytes-not-bus-counter";
        if (device.is_cpu()) identity += "; workers=" + std::to_string(request.workers()) + "; ISA=" + isaLevelName(activeISALevel());
        const PlanningMeasurementWork work(PlanningWorkUnit::Bytes, request.usefulBytes(), std::move(identity));
        if (device.is_cpu()) return {request, 0, PlanningExecutionMeasurement::cpu(work, execute)};

        auto &worker = GPUDeviceContextPool::instance().getContext(device);
        const ExplicitGPUStream stream(worker.defaultStream());
        kernel->bindGPUStream(stream);
        TransferEngine::prepareDeviceInput(&input, device, stream.get());
        TransferEngine::prepareDeviceInput(&residual, device, stream.get());
        TransferEngine::prepareDeviceOutput(&output, device, stream.get());
        TransferEngine::requireDeviceInput(&input, device, stream.get());
        TransferEngine::requireDeviceInput(&residual, device, stream.get());
        const size_t graph_bytes = GPUGraphMemoryContract::reservationBytesPerExecutable(device);
        auto graph_claim = memory->reserveNewAllocations(device, PhysicalMemoryOwner::NativeGraphExecutable, graph_bytes);
        auto graph = worker.createGraphCapture(stream.get());
        require(bool(graph), "native graph creation failed");
        ScopedBackendGraphCapture capture(worker, *graph, "planning streaming memory bandwidth");
        require(capture.begin(), "native graph begin failed");
        require(execute(), "captured production residual failed");
        capture.finish();
        require(graph->nodeCount() && graph->instantiate(), "nonempty native graph instantiation failed");
        require(GPUGraphMemoryContract::acceptsFamilyObservation(device, graph->residentMemoryBytes(), graph_bytes),
            "native graph exceeded physical admission");
        auto *backend = getBackendFor(device);
        require(backend != nullptr, "prepared GPU backend is missing");
        return {request, graph->nodeCount(), PlanningExecutionMeasurement::gpu(work, *backend, device, *graph)};
    }
}
