/**
 * @file PlanningHostDeviceMeasurement.cpp
 * @brief Exact-mechanism GPU/host sampling with rank-owned first touch and event-owned completion.
 *
 * The host role first-touches shared RAM on its setup thread; only the original
 * GPU owner imports those pages. All-rank agreement precedes registration and
 * follows native graph retirement. PMA host claims outlive both mappings.
 * Capture, host publication and byte verification are outside all timed native
 * intervals. No weight, KV, sampler or live inference state is constructed.
 */
#include "PlanningHostDeviceMeasurement.h"
#include "PlanningPublication.h"
#include "planning/RankHardwareOwnership.h"
#include "execution/moe/MoEOverlaySharedBackingStartup.h"
#include "interfaces/IMPIContext.h"
#include "interfaces/IMPITopology.h"
#include "utils/MPISessionOwner.h"
#include <nlohmann/json.hpp>
#include <sys/mman.h>
#include <unistd.h>
#include <cerrno>
#include <limits>
#include <system_error>
#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/GPUGraphMemoryContract.h"
#include "backends/IWorkerGPUContext.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "transfer/TransferEngine.h"
#include <algorithm>
#include <functional>
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

    PlanningHostDeviceRequest PlanningHostDeviceRequest::fromInventory(const ClusterInventory &inventory,
        PlanningHostDeviceGPUExecutor gpu, PlanningHostDeviceFirstTouch host, size_t bulk_bytes)
    {
        require(inventory.world_size > 0 && inventory.ranks.size() == size_t(inventory.world_size) &&
            gpu.rank >= 0 && gpu.rank < inventory.world_size && host.rank >= 0 && host.rank < inventory.world_size,
            "missing executor/first-touch rank");
        const auto &executor = inventory.getRank(gpu.rank);
        const auto &allocator = inventory.getRank(host.rank);
        require(executor.rank == gpu.rank && allocator.rank == host.rank &&
            inventory.connectionBetweenRanks(gpu.rank, host.rank).locality() != RankConnectionLocality::CrossNode,
            "GPU and first-touch allocator must share their original physical machine");
        require(allocator.cpu.numa_node >= 0 && allocator.cpu.memory_bytes > 0 && allocator.cpu_cores > 0,
            "host first-touch requires observed NUMA identity");
        require((gpu.device.is_cuda() || gpu.device.is_rocm()) && gpu.device.ordinal >= 0 &&
            bulk_bytes > 1 && bulk_bytes <= INT_MAX, "invalid GPU or bounded bulk geometry");
        const DeviceInfo *found = nullptr;
        for (const auto &candidate : executor.gpus)
            if (candidate.type == gpu.device.type && candidate.local_device_id == gpu.device.ordinal)
            {
                require(!found && !candidate.uuid.empty(), "ambiguous GPU physical identity");
                found = &candidate;
            }
        require(found, "GPU is absent from its original executor");
        requireRankLocalGPUOwnership(executor, *found, "Planning GPU/host executor");
        return {gpu, host, executor.node_id, allocator.cpu.numa_node, found->uuid, bulk_bytes};
    }

    std::vector<uint8_t> PlanningHostDeviceRequest::serialize() const
    {
        const auto text = nlohmann::json{{"gpu_rank", gpuRank()}, {"host_rank", hostRank()},
            {"node", physicalNode()}, {"host_numa", hostNumaNode()},
            {"device", {int(device().type), device().ordinal}}, {"uuid", uuid()}, {"bulk_bytes", bulkBytes()}}.dump();
        return {text.begin(), text.end()};
    }

    size_t PlanningHostDeviceMeasurement::hostAllocationBytes(const PlanningHostDeviceRequest &request)
    {
        const auto one = TransferEngine::mappedHostRegionAllocationBytes(request.bulkBytes());
        if (one > std::numeric_limits<size_t>::max() / 2)
            throw std::overflow_error("Planning shared host-page extent overflows size_t");
        return 2 * one;
    }

    void PlanningHostDeviceMeasurement::contributeHostMemory(const PlanningHostDeviceRequest &request,
        PhysicalMemoryResource host, PhysicalMemoryPlanBuilder &builder)
    {
        require(host.world_rank == request.hostRank() && host.device == DeviceId::cpu(),
            "first-touch BOM resource identity mismatch");
        builder.add(host, PhysicalMemoryOwner::ActivationTransportStaging, hostAllocationBytes(request));
    }

    void PlanningHostDeviceMeasurement::contributeExecutionMemory(const PlanningHostDeviceRequest &request,
        PhysicalMemoryResource execution, PhysicalMemoryPlanBuilder &builder)
    {
        require(execution.world_rank == request.gpuRank() && execution.device == request.device(),
            "GPU executor BOM resource identity mismatch");
        builder.add(execution, PhysicalMemoryOwner::ActivationTransportStaging, request.bulkBytes());
        builder.add(execution, PhysicalMemoryOwner::NativeGraphExecutable,
            2 * GPUGraphMemoryContract::reservationBytesPerExecutable(request.device()));
    }

    namespace
    {
        /** @brief Original session-owned shared pages; no GPU registration or alternate byte ledger.
         * The host role touches both buffers before agreement permits the GPU
         * role to register them. File identity is authenticated before mmap.
         */
        class SharedHostDevicePages final
        {
        public:
            /** @brief Map the same exact file on the two declared roles.
             * @param request Original role and payload identity.
             * @param mpi Existing discovery context, absent only for one local role.
             * @param rank Actual caller rank, never guessed from ordinal.
             */
            SharedHostDevicePages(const PlanningHostDeviceRequest &request,
                const std::shared_ptr<IMPIContext> &mpi, int rank)
                : bytes_(PlanningHostDeviceMeasurement::hostAllocationBytes(request))
            {
                const auto name = "planning-host-device-" + std::to_string(request.gpuRank()) + "-" +
                    std::to_string(request.hostRank()) + "-" + std::to_string(int(request.device().type)) +
                    "-" + std::to_string(request.device().ordinal);
                if (mpi)
                {
                    require(mpi->topology(), "shared first-touch sample requires its original topology");
                    std::vector<int> roles{request.gpuRank(), request.hostRank()};
                    std::sort(roles.begin(), roles.end());
                    roles.erase(std::unique(roles.begin(), roles.end()), roles.end());
                    file_ = establishMoEOverlaySharedBacking(*mpi,
                        {.kind=MoEOverlaySharedBackingKind::PlanningHostDevice,
                         .node_namespace=mpi->topology()->node_shared_memory_namespace(), .generation=1,
                         .fingerprint_low=(uint64_t(request.gpuRank()) << 32) | uint32_t(request.hostRank()),
                         .fingerprint_high=(uint64_t(int(request.device().type)) << 32) | uint32_t(request.device().ordinal)},
                        request.hostRank(), roles, name, bytes_);
                }
                else
                {
                    // The same file/mmap owner has zero peer messages in the
                    // existing null-context one-rank transaction.
                    require(rank == 0 && request.gpuRank() == 0 && request.hostRank() == 0,
                        "null context cannot adopt foreign calibration roles");
                    auto root = MPISessionOwner::sharedDirectory();
                    require(root.has_value(), "shared first-touch sample requires its original owned session directory");
                    file_ = root->createExclusive(name);
                    require(bytes_ <= uint64_t(std::numeric_limits<off_t>::max()), "shared sample extent exceeds off_t");
                    if (::ftruncate(file_.descriptor(), off_t(bytes_)) != 0)
                        throw std::system_error(errno, std::generic_category(), "Cannot size planning shared host pages");
                }
                file_.requireExtent(bytes_);
                base_ = ::mmap(nullptr, bytes_, PROT_READ | PROT_WRITE, MAP_SHARED, file_.descriptor(), 0);
                if (base_ == MAP_FAILED)
                {
                    base_ = nullptr;
                    throw std::system_error(errno, std::generic_category(), "Cannot map planning shared host pages");
                }
                if (rank == request.hostRank()) std::memset(base_, 0, bytes_);
            }
            /** @brief Unmap after native retirement; original file RAII owns exact-name removal. */
            ~SharedHostDevicePages() { if (base_ && ::munmap(base_, bytes_) != 0) std::terminate(); }
            /** @return Stable page-aligned first/second buffer address. */
            void *buffer(bool output) const noexcept { return static_cast<uint8_t *>(base_) + (output ? bytes_ / 2 : 0); }
        private:
            MPISessionOwner::SharedFile file_;
            size_t bytes_;
            void *base_ = nullptr;
        };

        /** @brief Execute the original eight-coordinate native body on already-first-touched pages. */
        PlanningHostDeviceObservations measurePrepared(const PlanningHostDeviceRequest &request,
            const std::shared_ptr<PhysicalMemoryAuthority> &memory,
            const std::shared_ptr<SharedHostDevicePages> &pages,
            const std::function<void(MappedTransferDirection, DeviceTransferBuffer &,
                const MappedHostTransferRegion &, size_t, void *)> &enqueue_kernel_copy)
        {
            require(bool(memory), "missing physical admission");
            const auto device = request.device();
            auto &worker = GPUDeviceContextPool::instance().getContext(device);
            require(!worker.ownsCurrentThread(), "host pages must be first-touched on the rank setup caller");
            auto &transfer = TransferEngine::instance();
            const size_t host_bytes = TransferEngine::mappedHostRegionAllocationBytes(request.bulkBytes());
            auto device_claim = memory->claimNewAllocation(device, PhysicalMemoryOwner::ActivationTransportStaging, request.bulkBytes());
            const std::array devices{device};
            auto input = transfer.registerExternalMappedHostRegion(pages->buffer(false), host_bytes, devices, pages);
            auto output = transfer.registerExternalMappedHostRegion(pages->buffer(true), host_bytes, devices, pages);
            require(input && output && input->sizeBytes() == host_bytes && output->sizeBytes() == host_bytes,
                "mapped allocation extent differs from canonical geometry");
            PlanningHostDeviceObservations result{request, {}};
            worker.submitAndWait([&] {
                auto buffer = transfer.allocateDeviceTransferBuffer(request.bulkBytes(), device);
                require(bool(buffer), "device staging allocation failed");
                auto *backend = getBackendFor(device);
                require(backend != nullptr, "missing exact GPU backend");
                // This isolated setup measurement has no competing inference.
                // Reuse its original worker stream; a background pool would add
                // an unrelated lifetime and priority to primitive byte timing.
                void *const stream = worker.defaultStream();
                require(stream != nullptr, "calibration requires its exact worker stream");
                require(!isGraphCaptureActive() && backend->prepareMappedHostCopyKernels(device.gpu_ordinal()),
                    "cold mapped-copy function preparation failed");
                const auto graph_bytes = GPUGraphMemoryContract::reservationBytesPerExecutable(device);
                for (size_t payload : {size_t{1}, request.bulkBytes()})
                {
                    auto readback_claim = memory->reserveNewAllocations(device, PhysicalMemoryOwner::NativeGraphExecutable, graph_bytes);
                    auto readback = worker.createGraphCapture(stream);
                    require(bool(readback), "readback graph creation failed");
                    {
                        ScopedBackendGraphCapture capture(worker, *readback, "planning host-device byte verification");
                        require(capture.begin(), "readback capture failed");
                        transfer.enqueueDeviceToMappedHost(*buffer, 0, *output, 0, payload, device, stream);
                        capture.finish();
                    }
                    require(readback->prepareRuntimeContextStorage(memory), "readback context storage preparation failed");
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
                                transfer.enqueueMappedHostToDevice(*input, 0, *buffer, 0, payload, device, stream);
                            auto claim = memory->reserveNewAllocations(device, PhysicalMemoryOwner::NativeGraphExecutable, graph_bytes);
                            auto graph = worker.createGraphCapture(stream);
                            require(bool(graph), "copy graph creation failed");
                            {
                                ScopedBackendGraphCapture capture(worker, *graph, "planning host-device primitive");
                                require(capture.begin(), "copy capture failed");
                                const bool inbound = direction == MappedTransferDirection::HostToDevice;
                                const auto &mapped = inbound ? *input : *output;
                                if (mechanism == PlanningHostTransferMechanism::MappedKernel)
                                    enqueue_kernel_copy(direction, *buffer, mapped, payload, stream);
                                else if (inbound)
                                    transfer.enqueueMappedHostToDevice(mapped, 0, *buffer, 0, payload, device, stream);
                                else transfer.enqueueDeviceToMappedHost(*buffer, 0, mapped, 0, payload, device, stream);
                                capture.finish();
                            }
                            require(graph->prepareRuntimeContextStorage(memory), "copy context storage preparation failed");
                            require(graph->nodeCount() && graph->instantiate() &&
                                GPUGraphMemoryContract::acceptsFamilyObservation(device, graph->residentMemoryBytes(), graph_bytes),
                                "copy graph incomplete or outside admission");
                            const PlanningMeasurementWork work(PlanningWorkUnit::Bytes, payload,
                                "GPU-host byte primitive; device=" + device.toString() +
                                "; gpu-executor-rank=" + std::to_string(request.gpuRank()) +
                                "; first-touch-rank=" + std::to_string(request.hostRank()) +
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
            PlanningHostDeviceMeasurement::validate(result);
            return result;
        }
    }

    std::optional<PlanningHostDeviceObservations> PlanningHostDeviceMeasurement::collect(
        const PlanningHostDeviceRequest &request, const std::shared_ptr<IMPIContext> &mpi,
        const std::shared_ptr<PhysicalMemoryAuthority> &memory)
    {
        const int rank = mpi ? mpi->rank() : 0;
        const auto identity = request.serialize();
        PhysicalMemoryAllocationLease host_claim;
        std::shared_ptr<SharedHostDevicePages> pages;
        std::optional<PlanningHostDeviceObservations> result;
        // Every fallible stage completes the original all-rank phase before
        // another stage can use its pages. The retained owner survives a peer
        // failure until the phase's completion consensus has drained all work.
        const auto phase = [&](const auto &execute) {
            exchangePlanningSamples(mpi, [&](int count) {
                return std::vector<std::vector<uint8_t>>(count, identity);
            }, [&](auto wire) {
                require(std::vector<uint8_t>(wire.begin(), wire.end()) == identity,
                    "GPU/host calibration roles disagree across ranks");
                execute();
                return std::vector<uint8_t>{1};
            }, [&](auto receipts) {
                for (const auto &receipt : receipts)
                    require(receipt.bytes.size() == 1 && receipt.bytes[0] == 1, "incomplete GPU/host calibration phase");
            });
        };
        phase([&] {
            require(memory && memory->worldRank() == rank, "missing this rank's physical admission");
            if (mpi)
            {
                const auto original = mpi->clusterInventory();
                require(original && request == PlanningHostDeviceRequest::fromInventory(*original,
                    {request.gpuRank(), request.device()}, {request.hostRank()}, request.bulkBytes()),
                    "sample roles do not match this context's original discovery");
            }
            require(request.gpuRank() < (mpi ? mpi->world_size() : 1) &&
                request.hostRank() < (mpi ? mpi->world_size() : 1), "foreign calibration participant");
            if (rank == request.gpuRank())
            {
                auto &worker = GPUDeviceContextPool::instance().getContext(request.device());
                require(!worker.ownsCurrentThread(), "GPU/host calibration requires the rank setup caller");
            }
            if (rank == request.hostRank()) host_claim = memory->claimNewAllocation(DeviceId::cpu(),
                PhysicalMemoryOwner::ActivationTransportStaging, hostAllocationBytes(request));
        });
        phase([&] {
            // topology() is lazy and constructs WORLD collectives. Every
            // original rank must enter before the role-filtered file mapping,
            // including an idle rank for a same-rank GPU/first-touch sample.
            // The preceding phase first drains all fallible identity/PMA/role
            // validation: a local rejection must not meet this construction
            // collective while a healthy peer enters phase-failure consensus.
            if (mpi)
                require(mpi->topology(), "GPU/host calibration requires its original all-rank topology");
            if (rank == request.gpuRank() || rank == request.hostRank())
                pages = std::make_shared<SharedHostDevicePages>(request, mpi, rank);
        });
        phase([&] {
            if (rank == request.gpuRank())
                result = measurePrepared(request, memory, pages,
                    [&](MappedTransferDirection direction, DeviceTransferBuffer &buffer,
                        const MappedHostTransferRegion &mapped, size_t bytes, void *stream) {
                        TransferEngine::instance().enqueueMappedKernelCopyOnPreparedStream(
                            request.device(), stream, direction, buffer, 0, mapped, 0, bytes);
                    });
        });
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
