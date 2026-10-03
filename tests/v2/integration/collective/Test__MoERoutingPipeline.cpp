/**
 * @file Test__MoERoutingPipeline.cpp
 * @brief Real-device regression of the production owned-router graph lowerer.
 *
 * Tests capture actual stages and their TransferEngine dependency ledger, not
 * a hand-written replacement of the graph. A separate replicated tensor API
 * oracle shares the prepared-format policy but owns independent scratch.
 * Replays change inputs/counts, include empty peers, and validate exact route
 * bits, producer probability snapshots and unused communication tails.
 */
#include "execution/compute_stages/stages/MoERoutingPipeline.h"
#include "execution/local_execution/graph/ComputeGraph.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/moe/MoEWorkspaceRequirements.h"
#include "collective/DeviceCountedAllGather.h"
#include "collective/LocalTPCollectiveInventory.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/BackendManager.h"
#include "planning/PlanningObservedResource.h"
#include "kernels/KernelFactory.h"
#include "utils/MPIContext.h"
#include "utils/TestTensorFactory.h"
#include <gtest/gtest.h>
#include <array>
#include <cmath>
#include <cstring>
#include <map>
using namespace llaminar2;

namespace
{
/** @brief Turn an invariant failure into an unwinding diagnostic outside device execution. */
void require(bool value, const char *message) { if (!value) throw std::runtime_error(message); }

/** @brief Input is uploaded before replay; this boundary owns no duplicate math. */
class InputBoundary final : public IComputeStage
{
public:
    /** @brief Bind the actual participant. */
    explicit InputBoundary(DeviceId device) : IComputeStage(device) {}
    /** @return No inference operation is performed by this test input boundary. */
    bool execute(IDeviceContext *) override { return true; }
    /** @return Harmless graph input classification. */
    ComputeStageType type() const override { return ComputeStageType::MOE_ROUTER; }
    /** @return Every native backend supports a no-work boundary. */
    bool supportsBackend(ComputeBackendType) const override { return true; }
    /** @return No setup or eager execution is needed. */
    bool isGraphCapturable() const override { return true; }
    /** @return Inputs remain external producers in the real capture ledger. */
    StageBufferContract bufferContract() const override { return {}; }
    /** @return This synthetic boundary contributes no diagnostic values. */
    StageDumpInfo buildDumpInfoImpl() const override { return {}; }
};

/** @brief Resource owner whose submitted work retires before every captured address. */
struct Participant
{
    DeviceId device;
    IWorkerGPUContext *worker = nullptr;
    IBackend *backend = nullptr;
    void *stream = nullptr;
    std::shared_ptr<TensorBase> hidden, gate, live, packet, peers, indices, weights, expected_ids, expected_weights;
    std::unique_ptr<DeviceWorkspaceManager> workspace, oracle_workspace;
    std::shared_ptr<MoERoutedPipelineKernelOwner> kernel_owner = std::make_shared<MoERoutedPipelineKernelOwner>();
    std::unique_ptr<IMoEKernel> oracle;
    std::unique_ptr<IDeviceContext> context;
    ComputeGraph model;
    std::array<std::unique_ptr<IGPUGraphCapture>, 2> graphs;
    std::vector<PhysicalMemoryOwnerReservation> reservations;
    /** @brief Join test observations before retiring graph/storage, also on failure. */
    ~Participant()
    {
        if (backend && stream) (void)backend->synchronizeStream(stream, device.ordinal);
        for (auto &g : graphs) g.reset();
    }
    /** @brief Allocate fixture-only FP32 tensor geometry and publish initial storage. */
    std::shared_ptr<TensorBase> tensor(std::vector<std::size_t> shape)
    {
        std::shared_ptr<TensorBase> result = test::TestTensorFactory::createFP32(shape);
        require(result->ensureOnDevice(device, stream), "Router stage tensor admission");
        return result;
    }
    /** @brief Upload one changed request outside capture and timing. */
    void upload(void *dst, const void *src, std::size_t bytes)
    { require(backend->hostToDeviceOnStream(dst, src, bytes, device.ordinal, stream), "Router stage input upload"); }
    /** @brief Observe exactly one terminal diagnostic buffer. */
    std::vector<float> read(const void *src, std::size_t count)
    {
        std::vector<float> result(count);
        require(backend->deviceToHostOnStream(result.data(), src, count * 4, device.ordinal, stream) &&
            worker->synchronizeStreamChecked(stream), "Router stage terminal observation");
        return result;
    }
};

/** @brief Derive internal/external event edges from production arena contracts. */
std::vector<GraphCaptureDependencyLedger::StagePlan> plan(Participant &p)
{
    const std::map<BufferId, TensorBase *> buffers{
        {BufferId::NORMALIZED, p.hidden.get()}, {BufferId::MOE_EXPERT_INDICES, p.indices.get()},
        {BufferId::MOE_EXPERT_WEIGHTS, p.weights.get()}, {BufferId::MOE_PROJECTION_LOCAL_PACKET, p.packet.get()},
        {BufferId::MOE_PROJECTION_GATHERED_PACKETS, p.peers.get()}};
    std::map<BufferId, std::size_t> writers;
    std::vector<GraphCaptureDependencyLedger::StagePlan> plans;
    for (const auto &name : p.model.getExecutionOrder())
    {
        auto *stage = p.model.getNode(name)->stage.get();
        GraphCaptureDependencyLedger::StagePlan stage_plan;
        stage_plan.stage_identity = stage; stage_plan.stage_name = name;
        const auto contract = stage->bufferContract();
        for (const auto &input : contract.allArenaReads())
            if (auto found = writers.find(input.id); found != writers.end())
                stage_plan.internal_inputs.push_back({buffers.at(input.id), found->second});
            else stage_plan.external_inputs.push_back(buffers.at(input.id));
        for (auto *weight : contract.weight_tensors)
            stage_plan.external_inputs.push_back(dynamic_cast<TensorBase *>(weight));
        for (const auto &output : contract.allWrites())
        { stage_plan.outputs.push_back(buffers.at(output.id)); writers[output.id] = plans.size(); }
        plans.push_back(std::move(stage_plan));
    }
    return plans;
}

/** @brief Run the exact graph lowering on reversed logical membership and changing live prefixes. */
void prove(DeviceId first, int degree)
{
    constexpr int capacity = 512, width = 96, experts = 17, top_k = 8;
    const auto mpi = MPIContextFactory::global();
    const auto inventory = mpi->clusterInventory();
    require(bool(inventory), "Router stage inventory");
    const auto &rank = inventory->ranks.at(mpi->rank());
    std::vector<DeviceId> devices;
    for (int p = degree - 1; p >= 0; --p)
        devices.push_back(first.is_cuda() ? DeviceId::cuda(p) : DeviceId::rocm(p));
    require(localTPPeerAccessCoverage(rank, devices) == PeerAccessCoverage::None, "Router stage no-P2P admission");
    const MoERouterRowPacketLayout layout{DeviceRowPartition::balanced(0, degree), capacity, top_k};
    const auto stride = layout.packetBytes();
    const auto memory = DeviceCountedAllGather::memoryFor(degree, stride);
    PhysicalMemoryPlanBuilder bom;
    bom.add(planningObservedResource(rank, DeviceId::cpu()), PhysicalMemoryOwner::ActivationTransportStaging,
        memory.host_bytes * degree);
    for (auto device : devices)
    {
        bom.add(planningObservedResource(rank, device), PhysicalMemoryOwner::ActivationTransportStaging, memory.device_bytes);
        bom.add(planningObservedResource(rank, device), PhysicalMemoryOwner::NativeGraphExecutable,
            2 * GPUGraphMemoryContract::reservationBytesPerExecutable(device));
    }
    PhysicalMemoryAuthority authority(std::make_shared<const PhysicalMemoryPlanAdmissionCertificate>(bom.build()), rank.rank);
    std::vector<std::unique_ptr<Participant>> owners;
    std::vector<void *> streams;
    for (auto device : devices)
    {
        auto p = std::make_unique<Participant>(); p->device = device;
        p->backend = getBackendFor(device); p->worker = &GPUDeviceContextPool::instance().getContext(device);
        p->worker->submitAndWait([&] { p->stream = p->worker->getOrCreateAuxiliaryStream("router_pipeline"); });
        streams.push_back(p->stream); owners.push_back(std::move(p));
    }
    auto fabric = DeviceCountedAllGather::create(authority, DeviceId::cpu(), devices, streams, PeerAccessCoverage::None, stride);
    for (int member = 0; member < degree; ++member)
    {
        auto &p = *owners[member];
        p.worker->submitAndWait([&] {
            p.context = IDeviceContext::create(p.device, 1);
            p.hidden = p.tensor({capacity, width}); p.gate = p.tensor({experts, width}); p.live = p.tensor({1});
            p.indices = p.tensor({capacity, top_k}); p.weights = p.tensor({capacity, top_k});
            p.expected_ids = p.tensor({capacity, top_k}); p.expected_weights = p.tensor({capacity, top_k});
            p.packet = p.tensor({stride / 4 + 16}); p.peers = p.tensor({stride / 4 * (degree - 1) + 16});
            std::vector<float> gate(experts * width);
            for (std::size_t i = 0; i < gate.size(); ++i) gate[i] = .11f * std::sin(float(i * 13 + 5) * .0037f);
            p.upload(p.gate->gpu_data_ptr(), gate.data(), gate.size() * 4);
            p.upload(p.live->gpu_data_ptr(), &capacity, 4);
            require(p.worker->synchronizeStreamChecked(p.stream), "Router stage gate lifetime");
            const auto requirements = p.device.is_cuda() ? MoEWorkspaceBuffers::cudaRouting(capacity, width, experts)
                : MoEWorkspaceBuffers::rocmRouting(capacity, width, experts);
            p.workspace = std::make_unique<DeviceWorkspaceManager>(p.device, requirements.total_bytes_with_alignment());
            p.oracle_workspace = std::make_unique<DeviceWorkspaceManager>(p.device, requirements.total_bytes_with_alignment());
            require(p.workspace->allocate(requirements) && p.oracle_workspace->allocate(requirements), "Router stage workspace");
            MoERoutingStage::Params route;
            route.device_id = p.device; route.input = p.hidden.get(); route.gate_weights = p.gate.get();
            route.seq_len = capacity; route.d_model = width; route.num_experts = experts; route.top_k = top_k;
            route.active_row_count_device = static_cast<const std::int32_t *>(p.live->gpu_data_ptr());
            route.output_indices = p.indices.get(); route.output_weights = p.weights.get();
            route.routed_pipeline_kernel_owner = p.kernel_owner;
            p.model.addNode("input", std::make_unique<InputBoundary>(p.device), p.device);
            appendMoERoutingPipeline(p.model, route, "layer0_moe_routing", "input",
                {fabric, member, {p.packet.get(), BufferId::MOE_PROJECTION_LOCAL_PACKET},
                    {p.peers.get(), BufferId::MOE_PROJECTION_GATHERED_PACKETS}});
            require(p.model.size() == 4, "Router stage lowering must contain exactly three production nodes");
            for (const auto &name : p.model.getExecutionOrder())
            {
                auto *stage = p.model.getNode(name)->stage.get();
                stage->setGPUStream(p.stream);
                if (auto *consumer = dynamic_cast<IWorkspaceConsumer *>(stage)) consumer->bindWorkspace(p.workspace.get());
                require(stage->prepareGraphLaunch(p.context.get(), p.stream) && stage->isGraphCapturable(), "Router stage preparation");
            }
            auto *local = p.model.getNode("layer0_moe_routing_owned")->stage.get();
            const auto dump = local->getDumpInfoSnapshot();
            require(dump.outputs.size() == 1 && std::holds_alternative<SnapshotCompactRows>(dump.outputs[0].row_layout),
                "Router probabilities must declare compact ownership, not claim complete values");
            p.oracle = llaminar::v2::kernels::KernelFactory::createMoEKernel(p.device);
            p.oracle->setGPUStream(p.stream);
            dynamic_cast<IWorkspaceConsumer *>(p.oracle.get())->bindWorkspace(p.oracle_workspace.get());
            require(p.oracle->prepareRouteLaunch(p.gate.get(),
                {.kind = MoERouteLaunchKind::GroupedPrefill, .physical_rows = capacity,
                 .d_model = width, .num_experts = experts, .top_k = top_k}), "Router oracle preparation");
            for (auto *tensor : {p.hidden.get(), p.gate.get(), p.packet.get()})
                TransferEngine::requireDeviceInput(tensor, p.device, p.stream);
            for (int kind = 0; kind < 2; ++kind)
            {
                p.reservations.push_back(authority.reserveNewAllocations(p.device, PhysicalMemoryOwner::NativeGraphExecutable,
                    GPUGraphMemoryContract::reservationBytesPerExecutable(p.device)));
                p.graphs[kind] = p.worker->createGraphCapture(p.stream);
                auto capture_plan = plan(p);
                GraphCaptureDependencyLedger ledger(p.device, p.stream, capture_plan, "router_pipeline");
                ScopedBackendGraphCapture recording(*p.worker, *p.graphs[kind], "router_pipeline", kind ? &ledger : nullptr);
                require(recording.begin(), "Router pipeline begin capture");
                if (!kind)
                {
                    MoERoutingResult unused;
                    require(p.oracle->routeWithTensorsEffectiveSeqLen(p.hidden.get(), p.gate.get(), capacity, width, experts, top_k,
                        true, p.expected_ids.get(), p.expected_weights.get(), unused, route.active_row_count_device),
                        "Router oracle capture");
                }
                else for (const auto &name : p.model.getExecutionOrder())
                {
                    auto *stage = p.model.getNode(name)->stage.get();
                    ScopedGraphCaptureStage scope(stage);
                    require(stage->execute(p.context.get()), "Router production stage capture");
                    scope.complete();
                }
                recording.finish();
                require(p.graphs[kind]->instantiate(), "Router pipeline instantiate");
            }
        });
    }
    int iteration = 0;
    for (const int live : {512, 448, 1, 0, 3, 65, 511, 2, 0, 512})
    {
        ++iteration;
        std::vector<float> hidden(capacity * width);
        for (std::size_t i = 0; i < hidden.size(); ++i) hidden[i] = .09f * std::cos(float(i + iteration) * .032f);
        for (int index = 0; index < degree; ++index)
        {
            auto &p = *owners[(iteration + index) % degree];
            p.worker->submitAndWait([&] {
                p.upload(p.hidden->gpu_data_ptr(), hidden.data(), hidden.size() * 4);
                p.upload(p.live->gpu_data_ptr(), &live, 4);
                require(p.backend->memset(p.packet->gpu_data_ptr(), 0xa5, p.packet->size_bytes(), p.device.ordinal, p.stream) &&
                    p.backend->memset(p.peers->gpu_data_ptr(), 0xa5, p.peers->size_bytes(), p.device.ordinal, p.stream),
                    "Router pipeline tail poisoning");
                require(p.graphs[0]->launch() && p.graphs[1]->launch(), "Router pipeline replay");
            });
        }
        for (int member = 0; member < degree; ++member)
        {
            auto &p = *owners[member];
            p.worker->submitAndWait([&] {
                for (const auto pair : {std::pair{p.indices, p.expected_ids}, std::pair{p.weights, p.expected_weights}})
                {
                    const auto actual = p.read(pair.first->gpu_data_ptr(), capacity * top_k);
                    const auto expected = p.read(pair.second->gpu_data_ptr(), capacity * top_k);
                    require(std::memcmp(actual.data(), expected.data(), actual.size() * 4) == 0, "Router pipeline selected bits drifted");
                }
                const auto owner = DeviceRowPartition::balanced(member, degree);
                const auto owned = owner.resolveFor(capacity, live);
                const auto probs = p.read(p.workspace->getBuffer(MoEWorkspaceBuffers::ROUTE_LOGITS), owner.capacityFor(capacity) * experts);
                const auto expected = p.read(p.oracle_workspace->getBuffer(MoEWorkspaceBuffers::ROUTE_LOGITS), capacity * experts);
                require(std::memcmp(probs.data(), expected.data() + owned.first * experts, owned.count * experts * 4) == 0,
                    "Router pipeline probability snapshot drifted");
                const auto local = p.read(p.packet->gpu_data_ptr(), p.packet->numel());
                const auto peers = p.read(p.peers->gpu_data_ptr(), p.peers->numel());
                for (int source = 0; source < degree; ++source)
                {
                    std::uint64_t bytes = 0;
                    require(p.backend->deviceToHostOnStream(&bytes, fabric->extentAddress(member, source), 8, p.device.ordinal, p.stream) &&
                        p.worker->synchronizeStreamChecked(p.stream), "Router pipeline byte-count observation");
                    const auto span = DeviceRowPartition::balanced(source, degree).resolveFor(capacity, live);
                    require(bytes == std::size_t(span.count) * top_k * sizeof(MoERouterSelectedRoute), "Router pipeline padded wire extent");
                    const auto *payload = reinterpret_cast<const std::uint8_t *>(
                        source == member ? local.data() : peers.data());
                    const auto offset = source == member ? 0 : std::size_t(source < member ? source : source - 1) * stride;
                    for (auto byte = std::size_t(bytes); byte < stride; ++byte)
                        require(payload[offset + byte] == 0xa5, "Router pipeline overwrote an unused packet suffix");
                }
                for (const auto &bank : {std::pair{&local, stride}, std::pair{&peers, stride * (degree - 1)}})
                {
                    const auto *bytes = reinterpret_cast<const std::uint8_t *>(bank.first->data());
                    for (auto byte = bank.second; byte < bank.first->size() * 4; ++byte)
                        require(bytes[byte] == 0xa5, "Router pipeline crossed a packet bank guard");
                }
            });
        }
    }
    owners.clear();
    fabric.reset();
}
}
#ifdef HAVE_CUDA
/** @test Capture the production CUDA lowerer and prepared router owner. */
TEST(MoERoutingPipeline, CUDA2) { prove(DeviceId::cuda(0), 2); }
#endif
#ifdef HAVE_ROCM
/** @test Capture the production HIP lowerer with the installed Q8 router policy. */
TEST(MoERoutingPipeline, ROCm2) { prove(DeviceId::rocm(0), 2); }
/** @test Empty owners and odd live prefixes retain four-participant correctness. */
TEST(MoERoutingPipeline, ROCm4) { prove(DeviceId::rocm(0), 4); }
#endif
