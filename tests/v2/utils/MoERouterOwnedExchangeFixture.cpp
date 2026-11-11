/**
 * @file MoERouterOwnedExchangeFixture.cpp
 * @brief Counted router exchange using production arithmetic and TransferEngine.
 *
 * Captures retain all pointers while request lengths and inputs change. Every
 * selected bit is compared to the unchanged complete router on each GPU. The
 * acquired count and every unused receive byte are observed too: numerically
 * correct output alone cannot certify that only live routes crossed the wire.
 * Test-only uploads/readbacks and final joins occur outside captured execution.
 * Each participant retains hardware metadata prepared before graph recording.
 */
#include "kernels/common/RouterQ8Launch.h"
#include "MoERouterOwnedExchangeFixture.h"
#include "fixtures/backends/OriginalRankLocalNativeReplay.h"
#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "collective/DeviceCountedAllGather.h"
#include "collective/ILocalTPContext.h"
#include "collective/LocalTPCollectiveInventory.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "kernels/common/MoERouterOwnedRows.h"
#include "kernels/cuda/moe/CUDARouterQ8Prefill.h"
#include "planning/PlanningObservedResource.h"
#include "utils/MPIContext.h"
#include "utils/TestTensorFactory.h"
#include "integration/transfer/CapturedChannelGraphProof.h"

#include <algorithm>
#include <array>
#include <future>
#include <exception>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <unordered_set>

extern "C" bool cudaMoE_route_logits(const float *, const float *, float *, int, int, int, int, void *);
extern "C" bool cudaMoE_softmax_topk(float *, float *, float *, int, int, int, bool, int, void *, const int *, void *);

extern "C" bool hipMoE_softmax_topk_decode_equivalent_rows(float *, float *, float *, int, int, int,
    bool, int, void *, const int *, void *);

namespace llaminar2::test
{
namespace
{
    /** @brief Preserve a failed invariant through ordinary resource unwinding. */
    void require(bool ok, const char *message)
    { if (!ok) throw std::runtime_error(message); }

    /**
     * @brief Launch the declared backend using immutable participant setup.
     * @param device Exact owning GPU participant.
     * @param p Retained routing buffers and device-owned live row count.
     * @param stream Exact participant stream used for recording or execution.
     * @param hardware CUDA launch geometry prepared before capture; unused by ROCm.
     * @return True when the selected backend accepts the routing launch.
     * @throws std::invalid_argument If the requested backend is not compiled.
     */
    bool route(DeviceId device, const MoERouterOwnedRowsLaunch &p, void *stream,
               const cuda::RouterQ8PrefillHardware &hardware)
    {
#ifdef HAVE_CUDA
        if (device.is_cuda()) return cuda::routeOwnedRows(p, stream, hardware);
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

    /**
     * @brief Real receive tensor exposing only its actual publication evidence.
     *
     * All coherence/event behavior remains in TensorBase and TransferEngine.
     * These passive protected-state views never override publication, replace
     * a backend, author coherence state, or fabricate completion evidence.
     */
    class NativeCountedPublicationTensor final : public FP32Tensor
    {
    public:
        /** @brief Retain ordinary FP32 storage for the exact receive-bank bytes. */
        explicit NativeCountedPublicationTensor(std::size_t bytes)
            : FP32Tensor({(bytes + 3) / 4}) {}
        /** @return The actual retained completion event protecting device bytes. */
        void *publicationEvent() const
        {
            std::lock_guard lock(coherence_mutex_);
            return completionEventProtectsDeviceValue_()?device_completion_event_:nullptr;
        }
        /** @return Whether TransferEngine imported that exact event onto the named consumer stream. */
        bool acquiredOn(void *event,void *stream) const
        {
            std::lock_guard lock(coherence_mutex_);
            return event && last_joined_completion_event_==event && last_joined_consumer_stream_==stream;
        }
        /** @return Whether a new publication protects this exact device before any consumer import. */
        bool freshPublicationOn(DeviceId device) const
        {
            std::lock_guard lock(coherence_mutex_);
            return completionEventProtectsDeviceValue_() && device_completion_event_ &&
                event_device_==std::optional{device} && authoritative_device_==std::optional{device} &&
                !last_joined_completion_event_ && !last_joined_consumer_stream_;
        }
    };

    /** @brief Retained local test storage; graphs retire before captured addresses. */
    struct Participant
    {
        DeviceId device;
        IWorkerGPUContext *worker = nullptr;
        IBackend *backend = nullptr;
        void *stream = nullptr;
        std::vector<std::shared_ptr<TensorBase>> tensors;
        std::shared_ptr<TensorBase> packet, peers;
        cuda::RouterQ8PrefillHardware router_hardware; ///< Prepared on the owning worker before capture.
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

/** @brief Replay the exchange using only the actual owner's selected physical members. */
void proveMoERouterOwnedExchangeOnCohort(const std::vector<DeviceId> &selected, int capacity,
    int width, int experts, const std::vector<int> &live_counts, MoERouterOwnedExchangeTiming *timing)
{
    const int participants = static_cast<int>(selected.size());
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
        devices.push_back(selected.at(i));
    const auto topology = DeviceCountedAllGatherTopology::resolve(rank, devices);
    const MoERouterRowPacketLayout geometry{DeviceRowPartition::balanced(0, participants), capacity, top_k};
    const auto stride = geometry.packetBytes();
    std::vector<GlobalDeviceAddress> native_members;
    for(const auto device:devices)native_members.push_back(GlobalDeviceAddress::fromLocalDeviceId(device));
    // The existing original rank owns this complete homogeneous communicator.
    // It outlives graph owners, packet owners and the collective's borrowed context.
    auto native_context=createLocalTPContext(native_members,{},devices.front().is_cuda()?
        CollectiveBackendType::NCCL:CollectiveBackendType::RCCL);
    require(bool(native_context),"Router original native collective context");
    PhysicalMemoryPlanBuilder bom;
    for (std::size_t member = 0; member < devices.size(); ++member)
    {
        const auto device = devices[member];
        const auto memory = topology.memoryFor(member, stride);
        bom.add(planningObservedResource(rank, DeviceId::cpu()), PhysicalMemoryOwner::ActivationTransportStaging, memory.host_bytes);
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
#ifdef HAVE_CUDA
            if (device.is_cuda())
                require(cuda::prepareRouterQ8PrefillHardware(device.ordinal, owner->stream, owner->router_hardware),
                    "Router fixture hardware preparation");
#endif
        });
        streams.push_back(owner->stream);
        owners.push_back(std::move(owner));
    }
    auto fabric = DeviceCountedAllGather::create(*authority, *native_context, DeviceId::cpu(), topology, streams, stride);
    for (std::size_t from = 0; from < devices.size(); ++from)
        for (std::size_t to = 0; to < devices.size(); ++to)
            if (from != to) require(fabric->payloadStorage(from, to) == topology.payloadStorage(from, to),
                "Router actual counted payload storage differs from original peer route");
    // Eight copies amortize host launch costs in the explicit economy run.
    // Correctness always records precisely one complete transaction per replay.
    const int copies = timing ? 8 : 1;
    std::vector<std::future<void>> preparation;
    for (int member = 0; member < participants; ++member)
    {
        auto &p = *owners[member];
        preparation.push_back(p.worker->submitAsync([&, member] {
            native_context->prepareOriginalNativeParticipant([&] {
            auto &p=*owners[member];
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
            p.exchange = fabric->bind(member, {0x726f75746572524full, stride},
                p.packet, p.peers);
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
                        require(route(p.device, r, p.stream, p.router_hardware), "Owned router producer");
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
                        node.name.find("nccl") == std::string::npos &&
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
                if(kind==1) {
                    std::size_t preparations=0;
                    for(const auto &node:nodes) {
                        require(node.name.find("capturedTransfer")==std::string::npos,"Native counted composition retained a custom peer kernel");
                        if(node.name.find("prepareNativeCountedAllGatherFrame")!=std::string::npos) {
                            ++preparations;
                            require(node.grid_x==1 && node.block_x==1,"Native completion preparation geometry");
                        }
                    }
                    require(preparations==std::size_t(copies),"Missing native counted invocation preparation");
                }
                require(graph->instantiate(), "Router graph instantiate");
            }
            if (timing)
            {
                p.begin = p.backend->createTimingEvent(p.device.ordinal);
                p.end = p.backend->createTimingEvent(p.device.ordinal);
                require(p.begin && p.end, "Router native timing events");
            }
            });
        }));
    }
    std::exception_ptr preparation_error;
    for(auto &work:preparation)try{work.get();}catch(...){
        if(!preparation_error){preparation_error=std::current_exception();(void)native_context->signalNativeCompletionAbort();}
    }
    if(preparation_error)std::rethrow_exception(preparation_error);
    std::vector<float> hidden(input_count);
    int replay = 0;
    std::vector<unsigned long long> expected_wire(participants,0);
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
        // Terminal diagnostics retain each source's actual compact packet and
        // independent full oracle. A receiver-local oracle alone conflates a
        // heterogeneous arithmetic difference with transfer/publication damage.
        struct SourceObservation
        {
            std::array<std::vector<float>, 2> oracle;
            std::vector<MoERouterSelectedRoute> packet;
        };
        std::vector<SourceObservation> source_observations(participants);
        for (int source = 0; source < participants; ++source)
        {
            auto &p = *owners[source];
            auto &observation = source_observations[source];
            p.worker->submitAndWait([&] {
                const auto span = DeviceRowPartition::balanced(source, participants).resolveFor(capacity, live);
                observation.packet.resize(std::size_t(span.count) * top_k);
                for (int column = 0; column < 2; ++column)
                {
                    observation.oracle[column].resize(route_count);
                    p.read(observation.oracle[column].data(), column ? p.oracle_weights : p.oracle_indices,
                        route_count * sizeof(float));
                }
                if (!observation.packet.empty())
                    p.read(observation.packet.data(), p.packet->gpu_data_ptr(),
                        observation.packet.size() * sizeof(MoERouterSelectedRoute));
                require(p.worker->synchronizeStreamChecked(p.stream), "Router source diagnostics completion");
            });
        }
        std::vector<unsigned long long> actual_wire(participants,0);
        for(int source=0;source<participants;++source) {
            const auto span=DeviceRowPartition::resolveMember(capacity,live,source,participants);
            expected_wire[source]+=static_cast<unsigned long long>(span.count)*top_k*sizeof(MoERouterSelectedRoute)*
                (participants-1)*copies;
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
                    if (actual != expected)
                    {
                        std::size_t index = 0;
                        while (index < actual.size() && actual[index] == expected[index]) ++index;
                        const int row = int(index / top_k);
                        const int source = DeviceRowPartition::balanced(member, participants).ownerFor(capacity, live, row);
                        std::ostringstream detail;
                        detail << "Router exchange changed a selected ID/probability bit or inactive suffix"
                            << " receiver=" << p.device.toString() << " receiver_member=" << member
                            << " replay=" << replay << " live=" << live << " capacity=" << capacity
                            << " width=" << width << " experts=" << experts
                            << " row=" << row << " slot=" << index % top_k
                            << " column=" << (column ? "weight" : "expert")
                            << " actual=0x" << std::hex << actual[index]
                            << " receiver_oracle=0x" << expected[index] << std::dec;
                        if (source >= 0)
                        {
                            const auto span = DeviceRowPartition::balanced(source, participants).resolveFor(capacity, live);
                            const auto &observation = source_observations[source];
                            const auto &route = observation.packet[(row - span.first) * top_k + index % top_k];
                            detail << " source=" << owners[source]->device.toString() << " source_member=" << source
                                << " source_oracle=0x" << std::hex
                                << std::bit_cast<std::uint32_t>(observation.oracle[column][index])
                                << " source_packet=0x"
                                << std::bit_cast<std::uint32_t>(column ? route.weight : route.expert);
                        }
                        else detail << " source=inactive_suffix";
                        throw std::runtime_error(detail.str());
                    }
                }
                std::vector<unsigned long long> counters(participants);
                std::vector<ncclLlaminarByteCompletion> completions(participants);
                p.read(counters.data(),fabric->payloadBytesAddress(member,0),counters.size()*sizeof(counters[0]));
                p.read(completions.data(),fabric->completionAddress(member,0),completions.size()*sizeof(completions[0]));
                require(p.worker->synchronizeStreamChecked(p.stream),"Native counted terminal wire evidence");
                const auto table=0x726f75746572524full;
                for(int source=0;source<participants;++source) {
                    const auto span=DeviceRowPartition::resolveMember(capacity,live,source,participants);
                    const auto bytes=std::uint64_t(span.count)*top_k*sizeof(MoERouterSelectedRoute);
                    const auto key=((table^(std::uint64_t(source)+1))*1099511628211ULL)|1ULL;
                    const auto authority=native_context->nativeCompletionAuthority(member).binding();
                    require(ncclLlaminarCompletionMatches(completions[source],table,stride,std::uint64_t(replay*copies),
                        bytes,key,authority,0,0),"Native counted receipt has stale identity, count or disposition");
                    actual_wire[source]+=counters[source];
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
        require(actual_wire==expected_wire,"Native counted composition sent capacity, omitted live bytes or duplicated links");
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
/** @brief Select the complete original owner before any native fixture allocation. */
void proveMoERouterOwnedExchange(DeviceId first, int participants, int capacity,
    int width, int experts, const std::vector<int> &live_counts, MoERouterOwnedExchangeTiming *timing)
{
    require(participants >= 2, "Router fixture participant count");
    originalRankLocalReplay(first.type, static_cast<size_t>(participants), [&](const auto &selected) {
        proveMoERouterOwnedExchangeOnCohort(selected, capacity, width, experts, live_counts, timing);
    });
}

/** @brief Real original-rank native composition, independent of router-balanced count policy. */
void proveNativeCountedAllGather(DeviceId first,int participants)
{
    originalRankLocalReplay(first.type,participants,[&](const auto &selected) {
        constexpr std::size_t capacity=1024*1024+3;
        constexpr std::uint64_t key=0x4e41544956454347ULL;
        const auto mpi=MPIContextFactory::global();
        const auto inventory=mpi->clusterInventory();
        require(bool(inventory),"Counted native original inventory");
        const auto &rank=inventory->ranks.at(mpi->rank());
        std::vector<DeviceId> devices(selected.rbegin(),selected.rend());
        std::vector<GlobalDeviceAddress> addresses;
        for(auto device:devices)addresses.push_back(GlobalDeviceAddress::fromLocalDeviceId(device));
        auto native_context=createLocalTPContext(addresses,{},first.is_cuda()?CollectiveBackendType::NCCL:CollectiveBackendType::RCCL);
        const auto topology=DeviceCountedAllGatherTopology::resolve(rank,devices);
        PhysicalMemoryPlanBuilder bom;
        for(std::size_t p=0;p<devices.size();++p) {
            const auto memory=topology.memoryFor(p,capacity);
            require(memory.host_bytes==0,"Counted native allocated mapped transfer storage");
            bom.add(planningObservedResource(rank,devices[p]),PhysicalMemoryOwner::ActivationTransportStaging,memory.device_bytes);
            bom.add(planningObservedResource(rank,devices[p]),PhysicalMemoryOwner::NativeGraphExecutable,
                GPUGraphMemoryContract::reservationBytesPerExecutable(devices[p]));
        }
        auto authority=std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<const PhysicalMemoryPlanAdmissionCertificate>(bom.build()),rank.rank);
        std::vector<std::unique_ptr<Participant>> owners;
        std::vector<void*> streams;
        for(auto device:devices) {
            auto p=std::make_unique<Participant>();p->device=device;
            p->worker=&GPUDeviceContextPool::instance().getContext(device);p->backend=getBackendFor(device);
            p->worker->submitAndWait([&]{p->stream=p->worker->getOrCreateAuxiliaryStream("native_counted_replay");});
            streams.push_back(p->stream);owners.push_back(std::move(p));
        }
        auto fabric=DeviceCountedAllGather::create(*authority,*native_context,DeviceId::cpu(),topology,streams,capacity);
        std::vector<std::future<void>> preparation;
        for(int member=0;member<participants;++member)
            preparation.push_back(owners[member]->worker->submitAsync([&,member] {
                native_context->prepareOriginalNativeParticipant([&] {
                auto &p=*owners[member];
                p.packet=p.allocate(capacity+64);
                p.peers=std::make_shared<NativeCountedPublicationTensor>(capacity*(participants-1)+64);
                require(p.peers->ensureOnDevice(p.device,p.stream),"Native counted observed receive storage");
                p.tensors.push_back(p.peers);
                p.begin=p.backend->createEvent(p.device.ordinal);
                p.end=p.backend->createEvent(p.device.ordinal);
                require(p.begin && p.end,"Native counted retained fork events");
                p.exchange=fabric->bind(member,{key,capacity},p.packet,p.peers);
                TransferEngine::requireDeviceInput(p.packet.get(),p.device,p.stream);
                p.graph_reservations.push_back(authority->reserveNewAllocations(p.device,
                    PhysicalMemoryOwner::NativeGraphExecutable,GPUGraphMemoryContract::reservationBytesPerExecutable(p.device)));
                p.graphs[1]=p.worker->createGraphCapture(p.stream);
                GraphCaptureDependencyLedger ledger(p.device,p.stream,{{
                    .stage_identity=&p.exchange,.stage_name="native counted exchange",
                    .external_inputs={p.packet.get()},.outputs={p.peers.get()},
                }},"native counted output publication");
                ScopedBackendGraphCapture recording(*p.worker,*p.graphs[1],"native counted replay",&ledger);
                require(recording.begin(),"Native counted capture begin");
                {
                    ScopedGraphCaptureStage stage(&p.exchange);
                    p.exchange.enqueue(p.stream);
                    // A pure output may be consumed inside this compound
                    // stage only after the actual enqueue publishes it.
                    TransferEngine::requireDeviceInput(p.peers.get(),p.device,p.stream);
                    stage.complete();
                }
                recording.finish();
                std::vector<GPUGraphKernelNodeInfo> nodes;
                require(p.graphs[1]->inspectKernelNodes(nodes),"Native counted captured nodes");
                std::size_t prepare=0,validate=0;
                for(const auto &node:nodes) {
                    require(node.name.find("capturedTransfer")==std::string::npos,"Counted native retained custom peer transport");
                    const bool preparation=node.name.find("prepareNativeCountedAllGatherFrame")!=std::string::npos;
                    const bool validation=node.name.find("validateNativeCountedAllGatherFrame")!=std::string::npos;
                    prepare+=preparation;
                    validate+=validation;
                    if(preparation || validation)
                        require(node.name_resolved && node.valid() &&
                            node.grid_x==1 && node.grid_y==1 && node.grid_z==1 &&
                            node.block_x==1 && node.block_y==1 && node.block_z==1 &&
                            node.dynamic_shared_memory_bytes==0 && node.static_shared_memory_bytes==0 &&
                            node.local_memory_bytes_per_thread==0,"Native counted scalar node geometry or resource contract");
                }
                require(prepare==1 && validate==1 && p.graphs[1]->instantiate(),"Native counted complete captured composition");
                });
        }));
        std::exception_ptr failed;
        for(auto &work:preparation)try{work.get();}catch(...){
            if(!failed){failed=std::current_exception();(void)native_context->signalNativeCompletionAbort();}
        }
        if(failed)std::rethrow_exception(failed);
        std::vector<unsigned long long> expected_wire(participants,0);
        constexpr std::array<std::size_t,8> sizes{capacity,0,1,257,4099,2,0,17};
        for(int replay=0;replay<64;++replay) {
            std::vector<std::uint64_t> extents(participants);
            std::vector<std::vector<std::uint8_t>> input(participants);
            for(int member=0;member<participants;++member) {
                // All-empty, one highly skewed owner, then rotated independent
                // counts. Repeated large-to-small transitions reuse one graph.
                extents[member]=replay%8==0?0:replay%8==1?(member==replay%participants?capacity:0):sizes[(replay+member)%sizes.size()];
                input[member].resize(capacity+64);
                for(std::size_t i=0;i<input[member].size();++i)input[member][i]=std::uint8_t(i*31+member*13+(replay%2?15:217));
                auto &p=*owners[member];
                p.worker->submitAndWait([&] {
                    p.upload(p.packet->gpu_data_ptr(),input[member].data(),input[member].size());
                    p.upload(p.exchange.extent(member),&extents[member],sizeof(extents[member]));
                    require(p.backend->memset(p.peers->gpu_data_ptr(),0xa5,p.peers->size_bytes(),p.device.ordinal,p.stream),"Native counted poison reset");
                });
                expected_wire[member]+=extents[member]*(participants-1);
            }
            for(int n=0;n<participants;++n) {
                auto &p=*owners[(n+replay)%participants];
                p.worker->submitAndWait([&]{require(p.graphs[1]->launch(),"Native counted retained replay");});
            }
            std::vector<unsigned long long> actual_wire(participants,0);
            for(int member=0;member<participants;++member) {
                auto &p=*owners[member];
                std::vector<std::uint8_t> output(p.peers->size_bytes()),unchanged(input[member].size());
                std::vector<std::uint64_t> acquired(participants);
                std::vector<unsigned long long> counters(participants);
                std::vector<ncclLlaminarByteCompletion> receipts(participants);
                p.worker->submitAndWait([&] {
                    p.read(output.data(),p.peers->gpu_data_ptr(),output.size());
                    p.read(unchanged.data(),p.packet->gpu_data_ptr(),unchanged.size());
                    p.read(acquired.data(),fabric->extentAddress(member,0),acquired.size()*sizeof(acquired[0]));
                    p.read(counters.data(),fabric->payloadBytesAddress(member,0),counters.size()*sizeof(counters[0]));
                    p.read(receipts.data(),fabric->completionAddress(member,0),receipts.size()*sizeof(receipts[0]));
                    require(p.worker->synchronizeStreamChecked(p.stream),"Native counted terminal observation");
                });
                require(acquired==extents,"Native counted changed live extents");
                require(unchanged==input[member],"Native counted modified its root packet or guard");
                for(int source=0;source<participants;++source) {
                    const auto message=((key^(std::uint64_t(source)+1))*1099511628211ULL)|1ULL;
                    require(ncclLlaminarCompletionMatches(receipts[source],key,capacity,replay+1,extents[source],message,
                        native_context->nativeCompletionAuthority(member).binding(),0,0),"Native counted stale completion identity");
                    actual_wire[source]+=counters[source];
                    if(source==member)continue;
                    const auto slot=source<member?source:source-1;
                    const auto begin=output.begin()+slot*capacity;
                    require(std::equal(begin,begin+extents[source],input[source].begin()),"Native counted changed a payload byte");
                    require(std::all_of(begin+extents[source],begin+capacity,[](auto byte){return byte==0xa5;}),"Native counted wrote unused capacity");
                }
                require(std::all_of(output.begin()+capacity*(participants-1),output.end(),[](auto byte){return byte==0xa5;}),"Native counted output guard changed");
            }
            require(actual_wire==expected_wire,"Native counted transmitted capacity or duplicated/omitted consumer bytes");
        }
        // Exercise the actual public enqueue boundary as well as recording.
        // Host authority deliberately invalidates the previous receive value;
        // raw backend readback alone would hide a missing output publication.
        // The second round borrows the same producer event on an auxiliary
        // stream and requires the canonical completion-event join to publish.
        for(const bool forked:{false,true}) {
            std::vector<std::vector<std::uint8_t>> inputs(participants);
            std::vector<std::uint64_t> live(participants);
            std::vector<void*> consumers(participants);
            for(int member=0;member<participants;++member) {
                auto &p=*owners[member];
                live[member]=member%2?257:17;
                inputs[member].assign(live[member],std::uint8_t(93+member*17+(forked?5:0)));
                p.worker->submitAndWait([&] {
                    consumers[member]=p.worker->getOrCreateAuxiliaryStream("native_counted_publication_consumer");
                    require(consumers[member] && consumers[member]!=p.stream,"Native counted distinct consumer frontier");
                    p.upload(p.packet->gpu_data_ptr(),inputs[member].data(),inputs[member].size());
                    p.upload(p.exchange.extent(member),&live[member],sizeof(live[member]));
                    require(p.backend->memset(p.peers->gpu_data_ptr(),0xa5,p.peers->size_bytes(),p.device.ordinal,p.stream),
                        "Native counted publication poison reset");
                    TransferEngine::publishDeviceWrite(p.packet.get(),p.device,p.stream);
                    TransferEngine::publishHostWrite(p.peers.get());
                    require(!p.peers->deviceValid(),"Native counted output must require fresh publication");
                    require(!static_cast<NativeCountedPublicationTensor&>(*p.peers).publicationEvent(),
                        "Native counted old output event survived host authority");
                });
            }
            std::vector<std::future<void>> submissions;
            for(int member=0;member<participants;++member)
                submissions.push_back(owners[member]->worker->submitAsync([&,member] {
                    native_context->prepareOriginalNativeParticipant([&] {
                        auto &p=*owners[member];
                        auto &output=static_cast<NativeCountedPublicationTensor&>(*p.peers);
                        if(!forked)p.exchange.enqueue(p.stream);
                        else {
                            auto &transfer=TransferEngine::instance();
                            const auto source=transfer.recordDeviceInputFork(p.packet.get(),p.device,p.stream,p.begin);
                            const auto acquired=transfer.acquireDeviceInputFork(source,consumers[member]);
                            p.exchange.enqueueAcquiredInput(acquired);
                            require(!output.publicationEvent() && !p.peers->deviceValid(),
                                "Native counted auxiliary fork published before its canonical join");
                            require(p.worker->recordEventChecked(p.end,acquired.consumerStream()) &&
                                p.worker->waitEventChecked(p.end,p.stream),"Native counted exact fork completion join");
                            TransferEngine::publishDeviceWrite(p.peers.get(),p.device,p.stream);
                        }
                        require(output.freshPublicationOn(p.device) && p.peers->deviceValid(),
                            "Native counted lost its new exact-device producer event");
                        const auto event=output.publicationEvent();
                        TransferEngine::requireDeviceInput(p.peers.get(),p.device,p.stream);
                        require(output.acquiredOn(event,p.stream),"Native counted producer frontier did not import its exact event");
                    });
                }));
            std::exception_ptr submission_error;
            for(auto &submission:submissions)try{submission.get();}catch(...){if(!submission_error)submission_error=std::current_exception();}
            if(submission_error)std::rethrow_exception(submission_error);
            for(int member=0;member<participants;++member) {
                auto &p=*owners[member];
                p.worker->submitAndWait([&] {
                    auto &output=static_cast<NativeCountedPublicationTensor&>(*p.peers);
                    const auto event=output.publicationEvent();
                    TransferEngine::requireDeviceInput(p.peers.get(),p.device,consumers[member]);
                    require(output.acquiredOn(event,consumers[member]),"Native counted consumer did not acquire the exact output event");
                    require(p.peers->ensureOnHost(consumers[member]),"Native counted public terminal observation");
                });
                const auto *output=static_cast<const std::uint8_t*>(p.peers->raw_data());
                for(int source=0;source<participants;++source) {
                    if(source==member)continue;
                    const auto slot=source<member?source:source-1;
                    const auto *begin=output+slot*capacity;
                    require(std::equal(begin,begin+live[source],inputs[source].begin()),"Native counted public consumer saw stale live bytes");
                    require(std::all_of(begin+live[source],begin+capacity,[](auto byte){return byte==0xa5;}),
                        "Native counted public consumer changed inactive capacity");
                }
                require(std::all_of(output+capacity*(participants-1),output+p.peers->size_bytes(),[](auto byte){return byte==0xa5;}),
                    "Native counted public consumer changed output guard");
            }
        }
        // Release all native graph owners, retain a genuine direct-created
        // binding, then destroy its original context. No installation was
        // needed to register lifetime revocation with the original owner.
        auto retained=owners.front()->exchange;
        auto replacement=owners.back()->exchange;
        const auto first_extent=retained.extent(0);
        const auto replacement_extent=replacement.extent(0);
        owners.clear();
        native_context.reset();
        bool rejected=false;
        try{retained.enqueue(streams.front());}catch(const std::logic_error&){rejected=true;}
        require(rejected,"Retained direct-create counted binding dereferenced its destroyed native context");
        // Real graph owners and their tensor vectors are gone. Assignment is
        // now the final lifetime authority for these actual native banks: old
        // capture leases must retire before their allocation owners can free.
        auto &same=retained;
        retained=same;
        require(retained.extent(0)==first_extent,"Native counted copy self-assignment changed its binding");
        retained=std::move(same);
        require(retained.extent(0)==first_extent,"Native counted move self-assignment lost its binding");
        auto copied=replacement;
        auto moved=std::move(copied);
        require(moved.extent(0)==replacement_extent,"Native counted move construction changed its binding");
        retained=replacement; // Populated copy replacement releases the sole first-bank bundle.
        require(retained.extent(0)==replacement_extent,"Native counted copy replacement changed its participant");
        replacement=std::move(moved);
        require(replacement.extent(0)==replacement_extent,"Native counted populated move replacement lost its binding");
        moved=retained;
        moved={};
        replacement={};
        retained={}; // The last real second-bank bundle must release leases before owners.
        fabric.reset();
        for(auto device:devices)
            require(authority->claimedBytes(device,PhysicalMemoryOwner::ActivationTransportStaging,
                PhysicalMemoryMaterializationKind::NewAllocation)==0,"Native counted metadata retirement");
    });
}

}
