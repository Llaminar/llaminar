/**
 * @file MoERouterOwnedExchangeFixture.cpp
 * @brief Counted router exchange using production arithmetic and TransferEngine.
 *
 * Captures retain all pointers while request lengths and inputs change. Every
 * selected bit is compared to the unchanged complete router on each GPU. The
 * acquired count and every unused receive byte are observed too: numerically
 * correct output alone cannot certify that only live routes crossed the wire.
 * Test-only uploads/readbacks and final joins occur outside captured execution.
 */
#include "MoERouterOwnedExchangeFixture.h"
#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "collective/DeviceCountedAllGather.h"
#include "collective/LocalTPCollectiveInventory.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "kernels/common/MoERouterOwnedRows.h"
#include "planning/PlanningObservedResource.h"
#include "utils/MPIContext.h"
#include "utils/TestTensorFactory.h"
#include "integration/transfer/CapturedChannelGraphProof.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <unordered_set>

extern "C" bool cudaMoE_route_logits(const float *, const float *, float *, int, int, int, int, void *);
extern "C" bool cudaMoE_softmax_topk(float *, float *, float *, int, int, int, bool, int, void *, const int *, void *);
extern "C" bool hipMoE_gate_logits_q8_weights_decode_equivalent_rows(const float *, int8_t *, float *,
    const int8_t *, const float *, float *, int, int, int, int, void *, const int *);
extern "C" bool hipMoE_softmax_topk_decode_equivalent_rows(float *, float *, float *, int, int, int,
    bool, int, void *, const int *, void *);

namespace llaminar2::test
{
namespace
{
    /** @brief Preserve a failed invariant through ordinary resource unwinding. */
    void require(bool ok, const char *message)
    { if (!ok) throw std::runtime_error(message); }

    /** @brief Select the declared backend only, never another implementation on failure. */
    bool route(DeviceId device, const MoERouterOwnedRowsLaunch &p, void *stream)
    {
#ifdef HAVE_CUDA
        if (device.is_cuda()) return cuda::routeOwnedRows(p, stream);
#endif
#ifdef HAVE_ROCM
        if (device.is_rocm()) return rocm::routeOwnedRows(p, stream);
#endif
        throw std::invalid_argument("Router exchange backend is not compiled");
    }

    /** @brief Restore complete selections using the production publication kernel. */
    bool publish(DeviceId device, const MoERouterRowPublication &p, void *stream)
    {
#ifdef HAVE_CUDA
        if (device.is_cuda()) return cuda::publishRouterOwnedRows(p, stream);
#endif
#ifdef HAVE_ROCM
        if (device.is_rocm()) return rocm::publishRouterOwnedRows(p, stream);
#endif
        throw std::invalid_argument("Router publication backend is not compiled");
    }

    /** @brief Complete unmodified router, with independent route/logit destinations. */
    bool reference(DeviceId device, MoERouterOwnedRowsLaunch p, float *logits,
                   float *indices, float *weights, void *stream)
    {
#ifdef HAVE_CUDA
        if (device.is_cuda())
            return cudaMoE_route_logits(p.hidden, static_cast<const float *>(p.gate), logits,
                p.capacity, p.width, p.experts, device.ordinal, stream) &&
                cudaMoE_softmax_topk(logits, indices, weights, p.capacity, p.experts, p.top_k,
                    p.normalize, device.ordinal, stream, p.live_rows, nullptr);
#endif
#ifdef HAVE_ROCM
        if (device.is_rocm())
            return hipMoE_gate_logits_q8_weights_decode_equivalent_rows(p.hidden, p.hidden_q8,
                p.hidden_scales, static_cast<const int8_t *>(p.gate), p.gate_scales, logits,
                p.capacity, p.width, p.experts, device.ordinal, stream, p.live_rows) &&
                hipMoE_softmax_topk_decode_equivalent_rows(logits, indices, weights, p.capacity,
                    p.experts, p.top_k, p.normalize, device.ordinal, stream, p.live_rows, nullptr);
#endif
        throw std::invalid_argument("Router oracle backend is not compiled");
    }

    /** @brief Retained local test storage; graphs retire before captured addresses. */
    struct Participant
    {
        DeviceId device;
        IWorkerGPUContext *worker = nullptr;
        IBackend *backend = nullptr;
        void *stream = nullptr;
        std::vector<std::shared_ptr<TensorBase>> tensors;
        std::shared_ptr<TensorBase> packet, peers;
        MoERouterOwnedRowsLaunch routing;
        MoERouterRowPublication publication;
        float *oracle_logits = nullptr, *oracle_indices = nullptr, *oracle_weights = nullptr;
        std::vector<PhysicalMemoryOwnerReservation> graph_reservations;
        DeviceCountedAllGatherBinding exchange;
        std::array<std::unique_ptr<IGPUGraphCapture>, 2> graphs;
        void *begin = nullptr, *end = nullptr;

        /** @brief Join submitted work before any normal or exceptional retirement. */
        ~Participant()
        {
            if (!backend || !stream) return;
            (void)backend->synchronizeStream(stream, device.ordinal);
            for (auto &graph : graphs) graph.reset();
            if (end) backend->destroyEvent(end, device.ordinal);
            if (begin) backend->destroyEvent(begin, device.ordinal);
        }

        /** @brief Materialize canonical tensor owners outside graph capture. */
        std::shared_ptr<TensorBase> allocate(std::size_t bytes)
        {
            std::shared_ptr<TensorBase> tensor = TestTensorFactory::createFP32({(bytes + 3) / 4});
            require(tensor->ensureOnDevice(device, stream), "Router fixture tensor preparation");
            tensors.push_back(tensor);
            return tensor;
        }

        /** @brief Upload changed request state outside the captured/timed interval. */
        void upload(void *destination, const void *source, std::size_t bytes)
        { require(backend->hostToDeviceOnStream(destination, source, bytes, device.ordinal, stream), "Router fixture upload"); }

        /** @brief Complete terminal test observation; never called by production replay. */
        void read(void *destination, const void *source, std::size_t bytes)
        { require(backend->deviceToHostOnStream(destination, source, bytes, device.ordinal, stream), "Router terminal readback"); }
    };
}

void proveMoERouterOwnedExchange(DeviceId first, int participants, int capacity,
    int width, int experts, const std::vector<int> &live_counts, MoERouterOwnedExchangeTiming *timing)
{
    constexpr int top_k = 8;
    require(participants >= 2 && capacity >= 2 && width > 0 && width % 32 == 0 && experts >= top_k &&
        !live_counts.empty() && (!timing || live_counts.size() == 1) &&
        std::all_of(live_counts.begin(), live_counts.end(), [&](int rows) { return rows >= 0 && rows <= capacity; }),
        "Router fixture geometry");
    auto mpi = MPIContextFactory::global();
    const auto inventory = mpi->clusterInventory();
    require(bool(inventory), "Router fixture canonical topology");
    const auto &rank = inventory->ranks.at(mpi->rank());
    std::vector<DeviceId> devices;
    for (int i = participants - 1; i >= 0; --i)
        devices.push_back(first.is_cuda() ? DeviceId::cuda(first.ordinal + i) : DeviceId::rocm(first.ordinal + i));
    require(localTPPeerAccessCoverage(rank, devices) == PeerAccessCoverage::None,
        "Counted router fixture must not bypass enabled native P2P");
    const MoERouterRowPacketLayout geometry{DeviceRowPartition::balanced(0, participants), capacity, top_k};
    const auto stride = geometry.packetBytes();
    const auto memory = DeviceCountedAllGather::memoryFor(participants, stride);
    PhysicalMemoryPlanBuilder bom;
    bom.add(planningObservedResource(rank, DeviceId::cpu()), PhysicalMemoryOwner::ActivationTransportStaging,
        memory.host_bytes * participants);
    for (const auto device : devices)
    {
        bom.add(planningObservedResource(rank, device), PhysicalMemoryOwner::ActivationTransportStaging, memory.device_bytes);
        bom.add(planningObservedResource(rank, device), PhysicalMemoryOwner::NativeGraphExecutable,
            2 * GPUGraphMemoryContract::reservationBytesPerExecutable(device));
    }
    auto authority = std::make_shared<PhysicalMemoryAuthority>(
        std::make_shared<const PhysicalMemoryPlanAdmissionCertificate>(bom.build()), rank.rank);
    std::vector<std::unique_ptr<Participant>> owners;
    std::vector<void *> streams;
    const auto input_count = std::size_t(capacity) * width;
    const auto route_count = std::size_t(capacity) * top_k;
    for (const auto device : devices)
    {
        auto owner = std::make_unique<Participant>();
        owner->device = device;
        owner->backend = getBackendFor(device);
        require(owner->backend && owner->backend->deviceCount() > device.ordinal, "Router fixture GPU count");
        owner->worker = &GPUDeviceContextPool::instance().getContext(device);
        owner->worker->submitAndWait([&] {
            owner->stream = owner->worker->getOrCreateAuxiliaryStream("owned_router_exchange");
            require(owner->stream, "Router fixture exact stream");
        });
        streams.push_back(owner->stream);
        owners.push_back(std::move(owner));
    }
    auto fabric = DeviceCountedAllGather::create(*authority, DeviceId::cpu(), devices, streams,
        PeerAccessCoverage::None, stride);
    // Eight copies amortize host launch costs in the explicit economy run.
    // Correctness always records precisely one complete transaction per replay.
    const int copies = timing ? 8 : 1;
    for (int member = 0; member < participants; ++member)
    {
        auto &p = *owners[member];
        p.worker->submitAndWait([&] {
            auto &r = p.routing;
            r.partition = DeviceRowPartition::balanced(member, participants);
            r.capacity = capacity; r.width = width; r.experts = experts; r.top_k = top_k;
            r.hidden = static_cast<const float *>(p.allocate(input_count * 4)->gpu_data_ptr());
            r.live_rows = static_cast<const int32_t *>(p.allocate(4)->gpu_data_ptr());
            r.logits = static_cast<float *>(p.allocate(std::size_t(capacity) * experts * 4)->gpu_data_ptr());
            r.hidden_q8 = static_cast<int8_t *>(p.allocate(input_count)->gpu_data_ptr());
            r.hidden_scales = static_cast<float *>(p.allocate(input_count / 32 * 4)->gpu_data_ptr());
            p.packet = p.allocate(stride + 64);
            p.peers = p.allocate(stride * (participants - 1) + 64);
            p.exchange = fabric->bind(member, {0x726f75746572524full, stride}, p.packet, p.peers);
            r.selected = static_cast<MoERouterSelectedRoute *>(p.packet->gpu_data_ptr());
            r.selected_bytes = p.exchange.extent(member);
            p.publication = {{r.partition, capacity, top_k}, r.live_rows, r.selected,
                p.peers->gpu_data_ptr(), p.exchange.extent(0),
                static_cast<float *>(p.allocate(route_count * 4)->gpu_data_ptr()),
                static_cast<float *>(p.allocate(route_count * 4)->gpu_data_ptr())};
            p.oracle_logits = static_cast<float *>(p.allocate(std::size_t(capacity) * experts * 4)->gpu_data_ptr());
            p.oracle_indices = static_cast<float *>(p.allocate(route_count * 4)->gpu_data_ptr());
            p.oracle_weights = static_cast<float *>(p.allocate(route_count * 4)->gpu_data_ptr());
            if (p.device.is_rocm())
            {
                r.format = MoERouterPreparedFormat::BlockQ8;
                auto gate = p.allocate(std::size_t(experts) * width);
                auto scales = p.allocate(std::size_t(experts) * width / 32 * 4);
                r.gate = gate->gpu_data_ptr(); r.gate_scales = static_cast<const float *>(scales->gpu_data_ptr());
                std::vector<int8_t> values(std::size_t(experts) * width);
                std::vector<float> scale(values.size() / 32);
                for (size_t i = 0; i < values.size(); ++i) values[i] = int(i * 71 % 255) - 127;
                for (size_t i = 0; i < scale.size(); ++i) scale[i] = std::ldexp(float(1 + i % 7), -12);
                p.upload(gate->gpu_data_ptr(), values.data(), values.size());
                p.upload(scales->gpu_data_ptr(), scale.data(), scale.size() * 4);
                require(p.worker->synchronizeStreamChecked(p.stream), "Retire quantized gate upload storage");
            }
            else
            {
                r.format = MoERouterPreparedFormat::FP32;
                auto gate = p.allocate(std::size_t(experts) * width * 4);
                r.gate = gate->gpu_data_ptr();
                std::vector<float> values(std::size_t(experts) * width);
                for (size_t i = 0; i < values.size(); ++i) values[i] = .11f * std::sin(float(i * 13 + 5) * .0037f);
                p.upload(gate->gpu_data_ptr(), values.data(), values.size() * 4);
                require(p.worker->synchronizeStreamChecked(p.stream), "Retire floating gate upload storage");
            }
            p.upload(const_cast<int32_t *>(r.live_rows), &capacity, sizeof(capacity));
            require(p.worker->synchronizeStreamChecked(p.stream), "Router fixture setup completion");
            // This raw native-graph fixture has no stage dependency ledger.
            // Authenticate its setup publication on the exact capture stream
            // before recording, as the production graph executor does.
            TransferEngine::requireDeviceInput(p.packet.get(), p.device, p.stream);
            require(!publish(p.device, p.publication, nullptr), "Router publication rejects null stream");
            for (int kind = 0; kind < 2; ++kind)
            {
                p.graph_reservations.push_back(authority->reserveNewAllocations(p.device,
                    PhysicalMemoryOwner::NativeGraphExecutable, GPUGraphMemoryContract::reservationBytesPerExecutable(p.device)));
                auto &graph = p.graphs[kind];
                graph = p.worker->createGraphCapture(p.stream);
                require(bool(graph), "Router graph owner");
                ScopedBackendGraphCapture recording(*p.worker, *graph, "owned router exchange");
                require(recording.begin(), "Router graph begin");
                for (int copy = 0; copy < copies; ++copy)
                    if (kind == 0)
                        require(reference(p.device, r, p.oracle_logits, p.oracle_indices, p.oracle_weights, p.stream), "Complete router oracle");
                    else
                    {
                        require(route(p.device, r, p.stream), "Owned router producer");
                        TransferEngine::publishDeviceWrite(p.packet.get(), p.device, p.stream);
                        p.exchange.enqueue(p.stream);
                        require(publish(p.device, p.publication, p.stream), "Owned router publication");
                    }
                recording.finish();
                std::vector<GPUGraphKernelNodeInfo> nodes;
                require(graph->inspectKernelNodes(nodes), "Router graph native resource evidence");
                std::unordered_set<std::string> reported;
                for (const auto &node : nodes)
                {
                    // Native attributes combine spills with diagnostic call
                    // frames. Existing CUDA channel failures call printf; the
                    // final linked DSO adds its 32-byte frame although the
                    // object has zero stack/local allocation. The compiler
                    // spill guard owns that distinction. Enforce zero private
                    // storage on every router/math node introduced here; keep
                    // complete transport geometry/resource evidence below.
                    if (node.name.find("capturedTransfer") == std::string::npos &&
                        node.local_memory_bytes_per_thread != 0)
                        throw std::runtime_error("Router transaction scratch: " + node.name + " bytes=" +
                            std::to_string(node.local_memory_bytes_per_thread));
                    if (timing && member == 0 && reported.insert(node.name).second)
                        std::printf("ROUTER_RESOURCE kind=%d kernel=%s grid=%u,%u,%u threads=%u registers=%u shared=%zu local=%zu active_blocks=%u\n",
                            kind, node.name.c_str(), node.grid_x, node.grid_y, node.grid_z,
                            node.block_x * node.block_y * node.block_z, node.registers_per_thread,
                            node.static_shared_memory_bytes + node.dynamic_shared_memory_bytes,
                            node.local_memory_bytes_per_thread, node.max_active_blocks_per_sm);
                }
                if (kind == 1) counted_channel_test::verifyCapturedChannelGraph(*graph, stride, 2 * (participants - 1) * copies);
                require(graph->instantiate(), "Router graph instantiate");
            }
            if (timing)
            {
                p.begin = p.backend->createTimingEvent(p.device.ordinal);
                p.end = p.backend->createTimingEvent(p.device.ordinal);
                require(p.begin && p.end, "Router native timing events");
            }
        });
    }
    std::vector<float> hidden(input_count);
    int replay = 0;
    for (const int live : live_counts)
    {
        ++replay;
        for (size_t i = 0; i < hidden.size(); ++i)
            hidden[i] = .081f * std::sin(float(i + replay * 7 + 7) * .021f) +
                        .053f * std::cos(float(i + replay * 19 + 11) * .039f);
        for (auto &owner : owners) owner->worker->submitAndWait([&] {
            auto &p = *owner;
            p.upload(const_cast<float *>(p.routing.hidden), hidden.data(), hidden.size() * 4);
            p.upload(const_cast<int32_t *>(p.routing.live_rows), &live, sizeof(live));
            require(p.backend->memset(p.packet->gpu_data_ptr(), 0xa5, p.packet->size_bytes(), p.device.ordinal, p.stream) &&
                p.backend->memset(p.peers->gpu_data_ptr(), 0xa5, p.peers->size_bytes(), p.device.ordinal, p.stream), "Router packet guard reset");
        });
        // Submit every participant before any potentially blocking readback.
        // Reverse submission order on alternate replays to stress empty owners.
        for (int i = 0; i < participants; ++i)
        {
            auto &p = *owners[replay % 2 ? i : participants - 1 - i];
            p.worker->submitAndWait([&] {
                require(p.graphs[0]->launch() && p.graphs[1]->launch(), "Router retained transaction");
            });
        }
        for (int member = 0; member < participants; ++member)
        {
            auto &p = *owners[member];
            p.worker->submitAndWait([&] {
                std::vector<uint32_t> actual(route_count), expected(route_count);
                for (int column = 0; column < 2; ++column)
                {
                    p.read(actual.data(), column ? p.publication.weights : p.publication.indices, actual.size() * 4);
                    p.read(expected.data(), column ? p.oracle_weights : p.oracle_indices, expected.size() * 4);
                    require(p.worker->synchronizeStreamChecked(p.stream), "Router route observation");
                    require(actual == expected, "Router exchange changed a selected ID/probability bit or inactive suffix");
                }
                std::vector<uint64_t> extents(participants);
                std::vector<uint8_t> received(p.peers->size_bytes()), local(p.packet->size_bytes());
                p.read(extents.data(), p.publication.acquired_bytes, extents.size() * 8);
                p.read(received.data(), p.peers->gpu_data_ptr(), received.size());
                p.read(local.data(), p.packet->gpu_data_ptr(), local.size());
                require(p.worker->synchronizeStreamChecked(p.stream), "Router extent observation");
                for (int source = 0; source < participants; ++source)
                {
                    const auto span = DeviceRowPartition::resolveMember(capacity, live, source, participants);
                    const auto bytes = size_t(span.count) * top_k * sizeof(MoERouterSelectedRoute);
                    require(extents[source] == bytes, "Router collective sent capacity instead of its live routes");
                    const size_t offset = source == member ? 0 : (source < member ? source : source - 1) * stride;
                    const auto *data = source == member ? local.data() : received.data() + offset;
                    for (size_t byte = bytes; byte < stride; ++byte)
                        require(data[byte] == 0xa5, "Router collective overwrote an unused packet tail");
                }
                require(std::all_of(local.begin() + stride, local.end(), [](uint8_t b) { return b == 0xa5; }) &&
                    std::all_of(received.begin() + stride * (participants - 1), received.end(), [](uint8_t b) { return b == 0xa5; }),
                    "Router collective crossed admitted packet capacity");
            });
        }
    }
    if (timing)
    {
        constexpr int replays = 16, warmups = 2, samples = 9;
        for (int sample = -warmups; sample < samples; ++sample)
            for (int experiment = 0; experiment < 2; ++experiment)
            {
                const int kind = (sample + warmups) % 2 ? 1 - experiment : experiment;
                for (auto &owner : owners) owner->worker->submitAndWait([&] {
                    auto &p = *owner;
                    require(p.backend->recordEvent(p.begin, p.device.ordinal, p.stream), "Router timer begin");
                    for (int replay = 0; replay < replays; ++replay) require(p.graphs[kind]->launch(), "Router timed replay");
                    require(p.backend->recordEvent(p.end, p.device.ordinal, p.stream), "Router timer end");
                });
                double slowest_us = 0;
                for (auto &owner : owners) owner->worker->submitAndWait([&] {
                    auto &p = *owner;
                    float milliseconds = 0;
                    require(p.backend->waitForEvent(p.end, p.device.ordinal) &&
                        p.backend->eventElapsedTimeMs(p.begin, p.end, p.device.ordinal, &milliseconds), "Router event interval");
                    slowest_us = std::max(slowest_us, double(milliseconds) * 1000 / (copies * replays));
                });
                if (sample >= 0) (kind == 0 ? timing->replicated_us : timing->distributed_us).push_back(slowest_us);
            }
    }
    owners.clear(); // Graphs/bindings before their fabric and physical authority.
    fabric.reset();
    for (auto device : devices)
    {
        require(authority->claimedBytes(device, PhysicalMemoryOwner::ActivationTransportStaging,
            PhysicalMemoryMaterializationKind::NewAllocation) == 0, "Router transport storage retirement");
        require(authority->reservedBytes(device, PhysicalMemoryOwner::NativeGraphExecutable) == 0,
            "Router graph reservation retirement");
    }
}
}
