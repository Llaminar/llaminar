/**
 * @file MoEProjectionPipelineFixture.cpp
 * @brief Shared correctness and isolated economy fixture for the projection lowerer.
 *
 * The candidate executes graph stages and native collectives over prepared
 * gate/up runtime banks plus fixed down columns. A separate complete-expert
 * captured pipeline provides the serial-arithmetic oracle. Fixture-only full
 * weights permit adversarial owner-bank changes; this is not a movement-copy
 * or production-memory-footprint certificate. All comparisons are byte exact.
 * Optional timings include runtime publication, regrouping and both explicit
 * collectives. Setup, request uploads and oracle readback stay outside timing.
 * The real down stage also exposes its existing route-column buffer through
 * the production snapshot mapper. Every replay verifies that diagnostic route
 * addends fold byte-exactly to the captured output for its owned columns.
 * The shared-column cohort instead checks the complete reduce-scatter/gate/
 * gather DAG against a full-row gate oracle. Its route-addend bank is already
 * reused at the terminal boundary, so that cohort observes the retained folded
 * columns, never pretends overwritten storage is a route-producer snapshot.
 * Decode's full-row cohort preserves the separate native shared allreduce and
 * verifies its bytes while routed down and the following native gather execute
 * in the production fork/join order. It never substitutes a column reduction.
 */
#include "MoEProjectionPipelineFixture.h"
#include "execution/moe/MoEProjectionRuntimeBinding.h"
#include <gtest/gtest.h>
#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "collective/LocalTPContext.h"
#include "collective/DeviceCountedAllGather.h"
#include "collective/LocalTPCollectiveInventory.h"
#include "planning/PlanningObservedResource.h"
#include "utils/MPIContext.h"
#include "execution/compute_stages/stages/MoEProjectionPipeline.h"
#include "execution/compute_stages/stages/TPLocalReduceOverlap.h"
#include "execution/compute_stages/stages/MoERoutingStage.h"
#include "execution/moe/MoEGroupedVerifierHistogramBoundarySet.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/local_execution/graph/ComputeGraph.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "execution/moe/DeviceMoEExpertDescriptorBuilder.h"
#include "execution/moe/MoEWorkspaceRequirements.h"
#include "interfaces/IWorkspaceConsumer.h"
#include "kernels/IMoEKernel.h"
#include "kernels/KernelFactory.h"
#include "loaders/ExpertGemmRegistry.h"
#include "models/GraphTypes.h"
#include "snapshots/SnapshotCapture.h"
#include "transfer/TransferEngine.h"
#include "utils/GpuPreparedGemmHarness.h"
#include "utils/QuantizedVerifierFormats.h"

#include <algorithm>
#include <array>
#include <barrier>
#include <bit>
#include <cmath>
#include <cstdio>
#include <exception>
#include <limits>
#include <map>
#include <thread>
#include <vector>

namespace llaminar2::test
{
namespace
{
/** @brief A fixture check which preserves C++ resource unwinding. */
void require(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }

/** @brief Source factory and exact floating/native execution family. */
struct Format
{
    std::string name;
    DeviceMoEWeightFormat family;
    QuantizedVerifierWeightCreator create;
    QuantizedVerifierWeightCreator create_down; ///< Empty means the same source format as gate/up.
};

/** @return Every supported prepared codebook plus each floating expert precision. */
std::vector<Format> formats()
{
    std::vector<Format> result;
    for (const auto &format : quantizedMoEVerifierFormats())
        result.push_back({format.label, DeviceMoEWeightFormat::NativeVNNI, format.create});
    // The target model mixes gate/up and down codebooks. Equal-format sweeps
    // alone would not prove that the fixed bank retains its independent type.
    result.push_back({"IQ2_S/IQ4_NL", DeviceMoEWeightFormat::NativeVNNI,
        [](const auto &shape, uint32_t seed) { return TestTensorFactory::createIQ2_SRandom(shape, seed); },
        [](const auto &shape, uint32_t seed) { return TestTensorFactory::createIQ4_NLRandom(shape, seed); }});
    result.push_back({"FP16", DeviceMoEWeightFormat::FP16, [](const auto &shape, uint32_t seed) {
        return TestTensorFactory::createFP16Random(shape, -.125f, .125f, seed); }});
    result.push_back({"BF16", DeviceMoEWeightFormat::BF16, [](const auto &shape, uint32_t seed) {
        return TestTensorFactory::createBF16Random(shape, -.125f, .125f, seed); }});
    result.push_back({"FP32", DeviceMoEWeightFormat::FP32, [](const auto &shape, uint32_t seed) {
        return TestTensorFactory::createFP32Random(shape, -.125f, .125f, seed); }});
    return result;
}

/** @brief Name the fixture's already published router inputs without replacing any candidate math. */
class FixtureRouterBoundary final : public IComputeStage
{
public:
    /** @brief Bind the input boundary to the same device as its downstream graph. */
    explicit FixtureRouterBoundary(DeviceId device) : IComputeStage(device) {}
    /** @return Router identity for the already populated diagnostic inputs. */
    ComputeStageType type() const override { return ComputeStageType::MOE_ROUTER; }
    /** @return The stable producer name consumed by the projection lowerer. */
    std::string name() const override { return "fixture_router"; }
    /** @brief Input publication happens before capture; this boundary adds no work. */
    bool execute(IDeviceContext *) override { return true; }
    /** @return True because this boundary performs no backend computation. */
    bool supportsBackend(ComputeBackendType) const override { return true; }
    /** @return True because request input storage is stable before capture. */
    bool isGraphCapturable() const override { return true; }
    /** @return No new owner; the fixture publishes the borrowed input tensors. */
    StageBufferContract bufferContract() const override { return {}; }
    /** @return No diagnostic snapshots for the synthetic input boundary. */
    StageDumpInfo buildDumpInfoImpl() const override { return {}; }
};

/** @brief Join one diagnostic stream before any graph/tensor owner can retire. */
struct Join
{
    IWorkerGPUContext &gpu; void *stream; ILocalTPContext &tp;
    const int exceptions = std::uncaught_exceptions();
    /** @brief Abort peers before joining a failed collective, then retire queued work. */
    ~Join()
    {
        if (std::uncaught_exceptions() > exceptions) tp.requestAbort();
        (void)gpu.synchronizeStreamChecked(stream);
    }
};

/** @brief Exact-stream timing resources; never created by the functional gate. */
class TimingEvents final
{
public:
    /** @brief Allocate two backend timing events outside every captured graph. */
    TimingEvents(IBackend &backend, DeviceId device, void *stream)
        : backend_(backend), device_(device), stream_(stream)
    {
        begin_ = backend_.createTimingEvent(device_.ordinal);
        end_ = backend_.createTimingEvent(device_.ordinal);
        if (!begin_ || !end_)
        {
            backend_.destroyEvent(end_, device_.ordinal);
            backend_.destroyEvent(begin_, device_.ordinal);
            throw std::runtime_error("projection timing event admission");
        }
    }
    /** @brief Destroy completed timing handles; the fixture joins before resource retirement. */
    ~TimingEvents()
    {
        backend_.destroyEvent(end_, device_.ordinal);
        backend_.destroyEvent(begin_, device_.ordinal);
    }
    /** @return Captured device interval in microseconds, excluding host setup/readback. */
    double measure(IGPUGraphCapture &capture)
    {
        require(backend_.recordEvent(begin_, device_.ordinal, stream_), "projection timer begin");
        require(capture.launch(), "projection timed replay");
        require(backend_.recordEvent(end_, device_.ordinal, stream_) &&
            backend_.waitForEvent(end_, device_.ordinal), "projection timer completion");
        float ms = 0;
        require(backend_.eventElapsedTimeMs(begin_, end_, device_.ordinal, &ms), "projection timer elapsed");
        return double(ms) * 1000.;
    }
private:
    IBackend &backend_;
    DeviceId device_;
    void *stream_, *begin_ = nullptr, *end_ = nullptr;
};

/** @brief Build ledger inputs from the same arena contracts used by the production executor. */
std::vector<GraphCaptureDependencyLedger::StagePlan> capturePlan(
    ComputeGraph &graph, const std::map<BufferId, TensorBase *> &buffers)
{
    std::map<BufferId, std::size_t> writers;
    std::vector<GraphCaptureDependencyLedger::StagePlan> plans;
    for (const auto &name : graph.getExecutionOrder())
    {
        auto *stage = graph.getNode(name)->stage.get();
        GraphCaptureDependencyLedger::StagePlan plan;
        plan.stage_identity = stage; plan.stage_name = name;
        const auto contract = stage->bufferContract();
        for (const auto &input : contract.allArenaReads())
        {
            auto *tensor = buffers.at(input.id);
            if (const auto writer = writers.find(input.id); writer != writers.end())
                plan.internal_inputs.push_back({tensor, writer->second});
            else plan.external_inputs.push_back(tensor);
        }
        for (auto *weight : contract.weight_tensors)
            plan.external_inputs.push_back(dynamic_cast<TensorBase *>(weight));
        for (const auto &output : contract.allWrites())
        {
            plan.outputs.push_back(buffers.at(output.id));
            writers[output.id] = plans.size();
        }
        plans.push_back(std::move(plan));
    }
    return plans;
}

/** @brief Exercise full/empty/sparse rows and changed owners on a retained six-node transaction. */
void proveFormat(DeviceId device, int participant, ILocalTPContext &tp, std::barrier<> &rendezvous,
    const Format &format, const ProjectionPipelineFixtureConfig &config,
    std::vector<double> &candidate_times, std::vector<double> &oracle_times)
{
    const auto [experts, width, intermediate, top_k] = config.geometry;
    const int capacity = *std::max_element(config.rows.begin(), config.rows.end());
    auto &gpu = GPUDeviceContextPool::instance().getContext(device);
    gpu.submitAndWait([&]
    {
        const int participants = tp.degree();
        const int local_columns = width / participants;
        auto *backend = getBackendFor(device);
        auto *stream = gpu.getOrCreateAuxiliaryStream("projection_pipeline_regression");
        auto context = IDeviceContext::create(device, 1);
        const bool floating = deviceMoEWeightFormatIsFloating(format.family);
        std::vector<std::shared_ptr<TensorBase>> sources;
        std::vector<std::shared_ptr<GpuPreparedGemm>> engines;
        auto prepare = [&](std::shared_ptr<TensorBase> source) {
            sources.push_back(source);
            auto prepared = std::make_shared<GpuPreparedGemm>(floating
                ? makeGpuPreparedFloatingPointGemm(source.get(), device)
                : makeGpuPreparedGemm(source.get(), device));
            engines.push_back(prepared);
            return std::shared_ptr<ITensorGemm>(prepared, prepared->kernel);
        };
        const auto ownership = MoEExpertProjectionOwnership::gateUpOwnedDownColumns(
            {experts, width, intermediate}, participant, participants);
        MoERoutedExpertPlacementPlan initial_plan;
        initial_plan.enabled = true;
        initial_plan.topology = RoutedExpertPlacementTopology::SingleDomain;
        initial_plan.continuation_domain = "native";
        initial_plan.shared_expert_domain = "native";
        initial_plan.residency_policy = RoutedExpertResidencyPolicy::StaticById;
        RoutedExpertDomain initial_domain;
        initial_domain.name = "native";
        initial_domain.scope = ExecutionDomainScope::RANK_LOCAL;
        initial_domain.owner_rank = 0;
        initial_domain.backend = device.is_cuda() ? CollectiveBackendType::NCCL : CollectiveBackendType::RCCL;
        initial_domain.routed_compute_policy = RoutedExpertComputePolicy::GateUpOwnedDownColumns;
        initial_domain.routed_phase_policy = RoutedExpertPhasePolicy::Uniform;
        initial_domain.participants = tp.devices();
        initial_plan.domains.push_back(std::move(initial_domain));
        initial_plan.routed_tiers.push_back({.name = "native", .domain = "native", .priority = 0});
        initial_plan.placements.push_back({.layer = 0, .routed_expert_tier = std::vector<int>(experts, 0)});
        const auto initial_owners = MoEExpertOwnerMap::build(initial_plan);
        MoEExpertOwnerParticipant endpoint;
        endpoint.participant_id = participant; endpoint.domain_participant_index = participant;
        endpoint.domain_name = "native"; endpoint.device = device;
        endpoint.address = GlobalDeviceAddress::fromLocalDeviceId(device);
        endpoint.world_rank = 0; endpoint.world_rank_known = true;
        ExpertGemmRegistry registry;
        std::vector<MoEOverlayPreparedExpertPayload> movable, complete;
        for (int expert = 0; expert < experts; ++expert)
        {
            auto gate = prepare(format.create({std::size_t(intermediate), std::size_t(width)}, 51800 + expert));
            auto up = prepare(format.create({std::size_t(intermediate), std::size_t(width)}, 51900 + expert));
            std::shared_ptr<TensorBase> down_source = (format.create_down ? format.create_down : format.create)(
                {std::size_t(width), std::size_t(intermediate)}, 52000 + expert);
            auto down = prepare(down_source);
            auto slice = prepare(down_source->create_view({std::size_t(local_columns), std::size_t(intermediate)},
                std::size_t(participant) * local_columns * intermediate));
            registry.registerEngineForParticipant("native", device, 0, participant, 0, expert,
                ExpertGemmRegistry::WeightRole::DOWN, slice.get(), slice, ownership);
            movable.push_back(MoEOverlayPreparedExpertPayload::gateUp(gate, up));
            complete.emplace_back(gate, up, down);
        }
        auto fixed = MoEOverlayFixedDownProjectionBank::resolve(registry, endpoint, 0, ownership);
        DeviceMoERuntimeTable::Config runtime_config;
        runtime_config.device_id = device; runtime_config.num_layers = 1;
        runtime_config.num_experts = experts; runtime_config.top_k = top_k;
        runtime_config.prefill_token_capacity = capacity; runtime_config.mirror_to_device = true;
        const bool prove_history = config.history == ProjectionHistoryProof::AllDemandBoundaries;
        runtime_config.deferred_verifier_token_capacity = prove_history ? capacity : 0;
        runtime_config.grouped_verifier_histogram_publication = prove_history
            ? GroupedVerifierHistogramPublicationMode::AcceptedRows
            : GroupedVerifierHistogramPublicationMode::Disabled;
        runtime_config.overlay_service_telemetry_coverage = prove_history
            ? MoEOverlayServiceTelemetryCoverage::AllRuntimeLayers
            : MoEOverlayServiceTelemetryCoverage::Disabled;
        runtime_config.fixed_down_banks = {fixed};
        DeviceMoERuntimeTable runtime(runtime_config);
        const auto requirements = device.is_cuda()
            ? MoEWorkspaceBuffers::cudaMoE(capacity, width, intermediate, experts, top_k)
            : MoEWorkspaceBuffers::rocmMoE(capacity, width, intermediate, experts, top_k);
        DeviceWorkspaceManager workspace(device, requirements.total_bytes_with_alignment() + 4 * 1024 * 1024);
        require(workspace.allocate(requirements), "projection graph workspace");
        const auto tensor = [&](std::size_t count) {
            std::shared_ptr<FP32Tensor> t = TestTensorFactory::createFP32({count + 16});
            std::fill_n(t->mutable_data(), t->numel(), 0.f);
            require(t->ensureOnDevice(device, stream), "projection graph buffer admission");
            return t;
        };
        const MoEGroupedIntermediateLayout maximum_layout{floating ? MoEGroupedIntermediateEncoding::FP32
            : MoEGroupedIntermediateEncoding::BlockQ8FP32Scales, static_cast<std::uint32_t>(intermediate),
            static_cast<std::uint32_t>(capacity * top_k),
            static_cast<std::uint32_t>(participants)};
        auto hidden = tensor(capacity * width), indices = tensor(capacity * top_k), weights = tensor(capacity * top_k);
        const auto packet_bytes = tp.deviceCountedAllGather() ? maximum_layout.compactCapacityBytes() : maximum_layout.packetBytes();
        auto packet = tensor(packet_bytes / sizeof(float));
        auto packets = tensor(participants * packet_bytes / sizeof(float));
        auto route_columns = tensor(capacity * top_k * local_columns);
        auto columns = tensor(capacity * local_columns), gathered = tensor(capacity * width), output = tensor(capacity * width);
        auto expected = tensor(capacity * width);
        const bool shared_columns = config.output == ProjectionOutputProof::SharedColumnCompletion;
        const bool shared_full_sum = config.output == ProjectionOutputProof::SharedFullRowsOverlap;
        const bool has_shared = shared_columns || shared_full_sum;
        // These extra banks belong only to the redundant diagnostic oracle.
        // The candidate lowerer reuses the production route/gather buffers.
        auto shared = has_shared ? tensor(capacity * width) : nullptr;
        auto full_shared = shared_columns ? tensor(capacity * width) : nullptr;
        auto full_routed = shared_columns ? tensor(capacity * width) : nullptr;
        auto shared_gate = shared_columns ? tensor(width) : nullptr;
        auto active_rows = TestTensorFactory::createINT32({1});
        require(active_rows->ensureOnDevice(device, stream), "projection live-row admission");
        auto accepted = TestTensorFactory::createINT32({1});
        auto publication_ok = TestTensorFactory::createINT32({1});
        require(accepted->ensureOnDevice(device, stream) && publication_ok->ensureOnDevice(device, stream),
            "projection accepted-state admission");
        auto oracle = llaminar::v2::kernels::KernelFactory::createMoEKernel(device);
        oracle->setGPUStream(stream);
        dynamic_cast<IWorkspaceConsumer *>(oracle.get())->bindWorkspace(&workspace);
        std::vector<DeviceNativeVNNIMatrixDesc> gates(experts), ups(experts), downs(experts);
        std::vector<DeviceMoEFloatingMatrixDesc> fp_gates(experts), fp_ups(experts), fp_downs(experts);
        for (int expert = 0; expert < experts; ++expert)
        {
            DeviceMoEExpertDescriptor desc{};
            require(exportDeviceMoEPreparedPayload(complete[expert], desc), "projection oracle descriptors");
            gates[expert] = desc.gate; ups[expert] = desc.up; downs[expert] = desc.down;
            fp_gates[expert] = desc.floating_gate; fp_ups[expert] = desc.floating_up; fp_downs[expert] = desc.floating_down;
        }
        const int gate_table = floating ? oracle->uploadGroupedExpertFloatingGateUpDescriptorTables(
            fp_gates.data(), fp_ups.data(), format.family, experts, width, intermediate)
            : oracle->uploadGroupedExpertGateUpDescriptorTables(gates.data(), ups.data(), experts, width, intermediate);
        const int down_table = floating ? oracle->uploadGroupedExpertFloatingDownDescriptorTable(
            fp_downs.data(), format.family, experts, width, intermediate)
            : oracle->uploadGroupedExpertDownDescriptorTable(downs.data(), experts, width, intermediate);
        require(gate_table >= 0 && down_table >= 0, "projection oracle table publication");
        uint32_t epoch = 0;
        std::vector<MoEOverlayPreparedExpertPayload> initial_payloads(experts);
        for (int expert = 0; expert < experts; ++expert)
            if (initial_owners.ownerFor(0, expert)->owner_participant == participant)
                initial_payloads[expert] = movable[expert];
        const auto changeOwners = [&](int replay) {
            if (epoch == 0)
            {
                // Exercise the same setup bridge used by a real model. It
                // must neither require a whole expert nor an eager launch.
                publishInitialMoEProjectionRuntimeLayer(runtime, initial_owners, 0, initial_payloads, stream);
                epoch = 1;
                return;
            }
            MoEPlacementUpdate placement;
            placement.epoch = ++epoch; placement.expert_count = experts;
            placement.participant_id = participant; placement.participant_count = participants;
            placement.experts.resize(experts); placement.local_compute_mask.resize(experts);
            placement.replica_role.assign(experts, static_cast<uint8_t>(DeviceMoEReplicaRole::Primary));
            for (int expert = 0; expert < experts; ++expert)
            {
                const int owner = replay == 0 ? initial_owners.ownerFor(0, expert)->owner_participant
                    : replay % 4 == 3 ? 0 : (expert + replay) % participants;
                auto &desc = placement.experts[expert];
                desc.logical_expert_id = expert; desc.owner_participant = owner;
                desc.local_slot = owner == participant ? expert : -1;
                desc.projection_set = DeviceMoEProjectionSet::GateUp;
                if (owner == participant)
                {
                    placement.local_compute_mask[expert] = 1;
                    desc.flags = toMoEExpertFlags(DeviceMoEExpertFlags::Valid | DeviceMoEExpertFlags::Resident | DeviceMoEExpertFlags::LocalCompute);
                    require(exportDeviceMoEPreparedPayload(movable[expert], desc), "projection owner payload");
                }
            }
            require(runtime.prepareInactiveBank(0, placement) && runtime.flipActiveBank(0, epoch, stream), "projection owner bank publication");
            // Late graph construction may revisit setup after real movement.
            // Idempotence must preserve the live epoch, not reset its owners.
            publishInitialMoEProjectionRuntimeLayer(runtime, initial_owners, 0, initial_payloads, stream);
            require(runtime.hostLayerState(0).active_epoch == epoch, "projection setup overwrote a moved epoch");
        };
        std::vector<float> h(hidden->numel(), .03125f), ids(indices->numel(), -1.f), probabilities(weights->numel(), 0.f);
        std::vector<float> actual(output->numel()), reference(expected->numel()), poison(output->numel(), -1234.5f);
        const auto upload = [&](ITensor *t, const std::vector<float> &v) {
            require(backend->hostToDeviceOnStream(t->gpu_data_ptr(), v.data(), v.size() * sizeof(float), device.ordinal, stream), "projection request input");
        };
        std::vector<float> shared_values(has_shared ? shared->numel() : 0);
        std::vector<float> summed_values(shared_values.size());
        std::vector<float> gate_values(shared_columns ? shared_gate->numel() : 0);
        std::vector<float> routed_reference(shared_columns ? full_routed->numel() : 0);
        std::vector<float> shared_reference(shared_values.size());
        std::vector<float> local_routed(shared_columns ? columns->numel() : 0);
        std::vector<float> local_shared(shared_values.size());
        const auto uploadShared = [&](int replay, int live) {
            if (!has_shared) return;
            // Exact binary partials isolate transport/layout from reduction
            // rounding, while different peer/row/column values expose swaps.
            for (std::size_t i = 0; i < shared_values.size(); ++i)
            {
                const float base = .03125f * (int((i * 7 + replay * 13) % 31) - 15);
                shared_values[i] = base + .125f * participant;
                summed_values[i] = 0.f;
                for (int peer = 0; peer < participants; ++peer)
                    summed_values[i] += base + .125f * peer;
            }
            for (std::size_t i = 0; i < gate_values.size(); ++i)
                gate_values[i] = .125f * (int((i * 17 + replay) % 23) - 11);
            upload(shared.get(), shared_values);
            if (shared_columns)
            {
                upload(full_shared.get(), summed_values);
                upload(shared_gate.get(), gate_values); upload(full_routed.get(), poison);
            }
            require(backend->hostToDeviceOnStream(active_rows->gpu_data_ptr(), &live, sizeof(live),
                device.ordinal, stream), "projection shared live-row publication");
        };
        const std::vector<MoEGroupedPlanDemand> demands = prove_history
            ? std::vector{MoEGroupedPlanDemand::None, MoEGroupedPlanDemand::OrdinaryPrefill,
                          MoEGroupedPlanDemand::DeferredAcceptedRows}
            : std::vector{MoEGroupedPlanDemand::None};
        for (int rows : config.rows)
        for (const auto demand : demands)
        {
            // A main verifier has at least a base row and one draft. M=1 is
            // covered separately for no-history and ordinary-plan semantics.
            if (rows == 1 && demand == MoEGroupedPlanDemand::DeferredAcceptedRows) continue;
            SCOPED_TRACE(format.name + "/rows=" + std::to_string(rows) + "/participant=" + std::to_string(participant));
            changeOwners(0);
            auto kernel_owner = std::make_shared<MoERoutedPipelineKernelOwner>();
            ComputeGraph graph;
            graph.addNode("fixture_router", std::make_unique<FixtureRouterBoundary>(device), device);
            MoEProjectionPipelineParams params;
            params.fixed_down = fixed;
            params.initial_gate_up.resize(experts);
            for (int expert = 0; expert < experts; ++expert)
                if (initial_owners.ownerFor(0, expert)->owner_participant == participant)
                    params.initial_gate_up[expert] = movable[expert];
            // Later epochs move previously absent experts here. Their runtime
            // descriptors, not this sparse startup table, must supply the dots.
            params.kernel_owner = kernel_owner; params.runtime = &runtime; params.tp = &tp;
            params.counted_exchange = tp.deviceCountedAllGather();
            params.rows = rows; params.top_k = top_k; params.prefix = "layer0_projection_"; params.router_node = "fixture_router";
            params.demand = demand;
            params.service_phase = demand == MoEGroupedPlanDemand::DeferredAcceptedRows
                ? MoEOverlayServicePhaseHint::GroupedVerifier : MoEOverlayServicePhaseHint::Prefill;
            ActivationBuffers arena;
            arena.normalized = dynamic_cast<TensorBase *>(hidden.get());
            arena.extensions = {
                {BufferId::MOE_EXPERT_INDICES, dynamic_cast<TensorBase *>(indices.get())},
                {BufferId::MOE_EXPERT_WEIGHTS, dynamic_cast<TensorBase *>(weights.get())},
                {BufferId::MOE_PROJECTION_LOCAL_PACKET, dynamic_cast<TensorBase *>(packet.get())},
                {BufferId::MOE_PROJECTION_GATHERED_PACKETS, dynamic_cast<TensorBase *>(packets.get())},
                {BufferId::MOE_PROJECTION_ROUTE_COLUMNS, dynamic_cast<TensorBase *>(route_columns.get())},
                {BufferId::MOE_PROJECTION_LOCAL_COLUMNS, dynamic_cast<TensorBase *>(columns.get())},
                {BufferId::MOE_PROJECTION_GATHERED_COLUMNS, dynamic_cast<TensorBase *>(gathered.get())},
                {BufferId::MOE_COMBINED_OUTPUT, dynamic_cast<TensorBase *>(output.get())}};
            // Exactly the production arena contract: common roles are not
            // extension entries. This caught the first model-launch defect.
            require(arena.get(BufferId::NORMALIZED) == nullptr, "common input must not be an extension buffer");
            params.bindArena(arena);
            require(params.hidden.tensor == hidden.get(), "production normalized activation binding");
            const auto terminal = appendMoEProjectionPipeline(graph, params);
            require(graph.size() == 7 && terminal == "layer0_projection_assemble", "projection graph structure");
            auto invalid = params; invalid.prefix = "invalid_"; invalid.local_packet = invalid.output;
            EXPECT_THROW(appendMoEProjectionPipeline(graph, invalid), std::invalid_argument);
            EXPECT_EQ(graph.size(), 7u);
            if (has_shared)
            {
                graph.addNode("fixture_shared", std::make_unique<FixtureRouterBoundary>(device), device);
                // Match the full production graph, including the first overlap:
                // packet exchange runs beside shared FFN before the later
                // shared reduce-scatter runs beside routed down projection.
                overlapTPLocalAllGather(graph, params.prefix + "intermediate_allgather", "fixture_shared");
                TPAllreduceStage::Params sum;
                sum.device_id = device; sum.tp_ctx = &tp; sum.tensor = shared.get();
                sum.count = std::size_t(rows) * width; sum.precision = "fp32";
                sum.tensor_buffer_id = BufferId::MOE_SHARED_EXPERT_OUTPUT;
                if (shared_full_sum)
                {
                    sum.stage_name = "fixture_shared_allreduce";
                    graph.addNode(sum.stage_name, std::make_unique<TPAllreduceStage>(sum), device);
                    graph.addDependency(sum.stage_name, "fixture_shared");
                    overlapMoEProjectionSharedAllreduce(graph, params.prefix, sum.stage_name);
                    EXPECT_THROW(overlapMoEProjectionSharedAllreduce(graph, params.prefix, sum.stage_name),
                                 std::invalid_argument);
                    require(graph.size() == 11, "projection full-row overlap duplicated a node");
                    // Pin the exact native submission order, not merely final
                    // equality: a peer may otherwise capture a different order.
                    const auto order = graph.getExecutionOrder();
                    const auto position = [&](const std::string &name) {
                        const auto found = std::find(order.begin(), order.end(), name);
                        require(found != order.end(), "projection full-row overlap omitted an edge");
                        return std::distance(order.begin(), found);
                    };
                    require(position(params.prefix + "intermediate_allgather") < position(sum.stage_name + "_submit") &&
                            position(sum.stage_name + "_submit") < position(params.prefix + "import_down") &&
                            position(params.prefix + "import_down") < position(sum.stage_name) &&
                            position(sum.stage_name) < position(params.prefix + "column_allgather"),
                            "projection full-row sum is outside its independent down window");
                }
                else
                {
                MoEProjectionSharedColumns completion{.reduction = sum, .producer = "fixture_shared",
                    .gate = shared_gate.get(), .active_rows = static_cast<const int *>(active_rows->gpu_data_ptr()),
                    .combined_output = params.output};
                require(finalizeMoEProjectionSharedColumns(graph, params.prefix, completion) == terminal,
                    "projection shared completion changed terminal identity");
                EXPECT_THROW(finalizeMoEProjectionSharedColumns(graph, params.prefix, completion), std::invalid_argument);
                require(graph.size() == 12, "projection shared completion duplicated a node");
                }
            }
            std::map<BufferId, TensorBase *> buffers;
            for (const auto binding : {params.hidden, params.routing_indices, params.routing_weights, params.local_packet,
                    params.gathered_packets, params.local_route_columns,
                    params.local_columns, params.gathered_columns, params.output})
                buffers.emplace(binding.id, dynamic_cast<TensorBase *>(binding.tensor));
            if (has_shared) buffers.emplace(BufferId::MOE_SHARED_EXPERT_OUTPUT, shared.get());
            // Exact and padded cold admission precede the producer's metadata
            // preparation. Support is known from the admitted fabric, whereas
            // capture readiness must wait for its concrete packet binding.
            for (const auto &name : graph.getExecutionOrder())
            {
                const auto *stage = graph.getNode(name)->stage.get();
                if (stage->type() == ComputeStageType::DEVICE_COUNTED_ALLGATHER)
                {
                    EXPECT_FALSE(stage->isGraphCapturable());
                    EXPECT_TRUE(stage->supportsGraphCaptureAfterLaunchPreparation());
                    EXPECT_TRUE(stage->supportsLazyPrefillGraphCapturePreflight());
                    EXPECT_TRUE(stage->supportsPaddedPrefillGraphCapturePreflight());
                }
            }
            int collectives = 0;
            for (const auto &name : graph.getExecutionOrder())
            {
                auto *stage = graph.getNode(name)->stage.get();
                if (auto *consumer = dynamic_cast<IWorkspaceConsumer *>(stage)) consumer->bindWorkspace(&workspace);
                require(stage->prepareGraphLaunch(context.get(), stream), "projection graph preparation");
                collectives += stage->type() == ComputeStageType::NATIVE_ALLGATHER ||
                    stage->type() == ComputeStageType::DEVICE_COUNTED_ALLGATHER;
            }
            require(collectives == (has_shared ? 3 : 2), "projection collective fork/join edges must be explicit");
            // Exercise the real producer descriptor, not a hand-written test
            // snapshot. Metadata borrows the admitted bank without a GPU copy.
            const auto down_stage_name = params.prefix + "import_down";
            auto route_dump = graph.getNode(down_stage_name)->stage->getDumpInfoSnapshot();
            const int local_width = width / participants;
            require(route_dump.outputs.size() == 1 &&
                route_dump.outputs.front().tensor == route_columns.get() &&
                route_dump.outputs.front().rows == static_cast<size_t>(rows * top_k) &&
                route_dump.outputs.front().cols == static_cast<size_t>(local_width),
                "projection route snapshot must borrow packed down-column output");
            const auto route_keys = SnapshotCapture::possibleKeysForStage(down_stage_name, route_dump);
            require(route_keys == std::vector<std::string>{"layer0_MOE_ROUTE_CONTRIBUTIONS"},
                "projection route producer missing canonical diagnostic key");
            std::vector<float> route_observation(static_cast<size_t>(rows) * top_k * local_width);
            auto *foreign_stream = gpu.getOrCreateAuxiliaryStream("projection_rejected_stream");
            for (const auto &name : graph.getExecutionOrder())
            {
                auto *stage = graph.getNode(name)->stage.get();
                if (stage->type() == ComputeStageType::MOE_PROJECTION_PHASE)
                    EXPECT_THROW(stage->prepareGraphLaunch(context.get(), foreign_stream), std::logic_error);
            }
            upload(hidden.get(), h); upload(indices.get(), ids); upload(weights.get(), probabilities);
            uploadShared(0, rows);
            require(gpu.synchronizeStreamChecked(stream), "projection admission join");
            // Admission joins every external producer to this exact capture
            // stream. A host completion alone is not a coherence publication.
            for (auto *input : {hidden.get(), indices.get(), weights.get()})
                TransferEngine::requireDeviceInput(input, device, stream);
            if (shared_columns)
                for (auto *input : {shared.get(), shared_gate.get(), full_shared.get(), full_routed.get()})
                    TransferEngine::requireDeviceInput(input, device, stream);
            if (shared_full_sum)
                TransferEngine::requireDeviceInput(shared.get(), device, stream);
            auto capture = gpu.createGraphCapture(stream), oracle_capture = gpu.createGraphCapture(stream);
            Join join{gpu, stream, tp};
            const auto record = [&] {
                GraphCaptureDependencyLedger ledger(device, stream, capturePlan(graph, buffers), "projection_pipeline");
                rendezvous.arrive_and_wait();
                ScopedBackendGraphCapture recording(gpu, *capture, "projection_pipeline", &ledger);
                require(recording.begin(), "projection capture begin");
                for (const auto &name : graph.getExecutionOrder())
                {
                    auto *stage = graph.getNode(name)->stage.get();
                    ScopedGraphCaptureStage scope(stage);
                    try { require(stage->execute(context.get()), "projection phase capture"); }
                    catch (const std::exception &error)
                    {
                        // A failure inside a fork can make abandoned native
                        // capture itself fatal. Preserve the original stage and
                        // exception before that strict retirement guard fires.
                        std::fprintf(stderr, "Projection capture failed at %s on %s: %s\n",
                            name.c_str(), device.toString().c_str(), error.what());
                        throw;
                    }
                    scope.complete();
                }
                recording.finish();
                require(capture->instantiate(), "projection capture instantiate");
            };
            record();
            // The production router owns accepted publication even though
            // grouping below it writes the final gate/up participant ledger.
            // Exercise that real publisher, not a fixture-specific commit.
            std::unique_ptr<MoERoutingStage> publisher;
            std::unique_ptr<IGPUGraphCapture> accepted_capture;
            void *publication_stream = nullptr;
            if (demand == MoEGroupedPlanDemand::DeferredAcceptedRows)
            {
                publication_stream = runtime.groupedVerifierHistogramPublicationStream();
                MoERoutingStage::Params route;
                route.device_id = device; route.layer_idx = 0; route.seq_len = rows;
                route.num_experts = experts; route.top_k = top_k; route.d_model = width;
                route.moe_runtime_table = &runtime;
                route.force_decode_equivalent_verifier_prefill = true;
                route.grouped_verifier_histogram_role = MoEGroupedVerifierHistogramRole::DeferredAcceptedRows;
                route.routed_pipeline_kernel_owner = kernel_owner;
                publisher = std::make_unique<MoERoutingStage>(route);
                publisher->setGPUStream(publication_stream);
                publisher->bindWorkspace(&workspace);
                MoEGroupedVerifierHistogramBoundarySet boundaries;
                MoEGroupedVerifierHistogramBoundaryResolution resolution;
                std::string error;
                require(boundaries.registerRoutedLayer(0, "fixture_router", &error) &&
                    boundaries.registerStage(publisher.get(), "fixture_router", &error) &&
                    boundaries.resolve(resolution, &error), "projection accepted-history boundary resolution");
                require(resolution.publishers.size() == 1 && resolution.publishers[0] == publisher.get() &&
                    resolution.publication_stream == publication_stream, "projection has one real history authority");
                require(publisher->prepareGroupedVerifierHistogramProducer(publication_stream), "projection history producer");
                accepted_capture = gpu.createGraphCapture(publication_stream);
                GraphCaptureGuard guard;
                require(publisher->transitionGroupedVerifierHistogramProducerCapture(publication_stream,
                        RuntimeHistogramProducerCaptureTransition::Entering) && accepted_capture->beginCapture() &&
                    publisher->enqueueCommittedGroupedVerifierHistograms(
                        static_cast<const int32_t *>(accepted->gpu_data_ptr()),
                        static_cast<const int32_t *>(publication_ok->gpu_data_ptr()), 1, rows, publication_stream) &&
                    accepted_capture->endCapture() &&
                    publisher->transitionGroupedVerifierHistogramProducerCapture(publication_stream,
                        RuntimeHistogramProducerCaptureTransition::Completed) && accepted_capture->instantiate(),
                    "projection accepted-history retained capture");
            }
            {
                ScopedBackendGraphCapture recording(gpu, *oracle_capture, "projection_full_row_oracle");
                require(recording.begin(), "projection oracle capture begin");
                require(oracle->prepareExpertGroupsAsync(indices.get(), weights.get(), rows, experts, top_k) &&
                    oracle->executeGroupedPrefillPipeline(hidden.get(), shared_columns ? full_routed.get() : expected.get(),
                        gate_table, down_table, rows, width, intermediate, experts, top_k), "projection oracle math capture");
                if (shared_columns)
                    require(oracle->sharedExpertGateFromTensorsEffectiveSeqLen(hidden.get(), shared_gate.get(),
                        full_shared.get(), rows, width,
                        static_cast<const int *>(active_rows->gpu_data_ptr())), "projection standalone full-row gate oracle");
                recording.finish();
            }
            require(oracle_capture->instantiate(), "projection oracle instantiate");
            using History = std::array<std::array<uint64_t, kDeviceMoEMaxExperts>, 6>;
            const auto read_history = [&] {
                History result{};
                constexpr std::array offsets{
                    offsetof(DeviceMoELayerRuntime, decode_histogram), offsetof(DeviceMoELayerRuntime, decode_local_histogram),
                    offsetof(DeviceMoELayerRuntime, prefill_histogram), offsetof(DeviceMoELayerRuntime, prefill_local_histogram),
                    offsetof(DeviceMoELayerRuntime, grouped_verifier_histogram), offsetof(DeviceMoELayerRuntime, grouped_verifier_local_histogram)};
                for (std::size_t phase = 0; phase < offsets.size(); ++phase)
                    require(backend->deviceToHostOnStream(result[phase].data(),
                        reinterpret_cast<const char *>(runtime.deviceLayerState(0)) + offsets[phase],
                        experts * sizeof(uint64_t), device.ordinal, stream), "projection diagnostic history read");
                require(gpu.synchronizeStreamChecked(stream), "projection diagnostic history join");
                return result;
            };
            for (int replay = 0; replay < 20; ++replay)
            {
                if (replay == 5 || replay == 15)
                {
                    // A normal request boundary changes data, never the
                    // retained native graph or its descriptor-table identity.
                    for (const auto &name : graph.getExecutionOrder())
                        graph.getNode(name)->stage->resetSessionStatePreservingCapturedReplay();
                }
                if (replay == 10)
                {
                    // Exercise the distinct hard-reset lifecycle as well:
                    // retire the graph, invalidate metadata, then prepare a new
                    // capture from the same declarative graph and prepared weights.
                    // This is a fixture reset test, not a production replay mode.
                    rendezvous.arrive_and_wait();
                    capture->reset();
                    for (const auto &name : graph.getExecutionOrder())
                        graph.getNode(name)->stage->invalidateKernelDynamicState();
                    for (const auto &name : graph.getExecutionOrder())
                    {
                        auto *stage = graph.getNode(name)->stage.get();
                        if (stage->type() == ComputeStageType::MOE_PROJECTION_PHASE)
                            require(!stage->isGraphCapturable(), "stale projection metadata remained capturable");
                    }
                    for (const auto &name : graph.getExecutionOrder())
                    {
                        auto *stage = graph.getNode(name)->stage.get();
                        require(stage->prepareGraphLaunch(context.get(), stream), "projection reset preparation");
                        require(stage->prepareGraphLaunch(context.get(), stream), "projection preparation must be idempotent");
                    }
                    record();
                }
                changeOwners(replay);
                for (std::size_t i = 0; i < h.size(); ++i) h[i] = .0071f * (int((i * 19 + replay * 13) % 101) - 50);
                const int live = replay % 3 == 1 ? 0 : replay % 3 == 2 ? (rows + 1) / 2 : rows;
                for (int slot = 0; slot < capacity * top_k; ++slot)
                {
                    ids[slot] = slot / top_k < live ? float((slot + replay) % experts) : -1.f;
                    probabilities[slot] = slot / top_k < live ? .25f * (1 + slot % top_k) : 0.f;
                }
                upload(hidden.get(), h); upload(indices.get(), ids); upload(weights.get(), probabilities);
                upload(output.get(), poison); upload(expected.get(), poison);
                uploadShared(replay, live);
                const auto before = prove_history ? read_history() : History{};
                rendezvous.arrive_and_wait();
                require(capture->launch() && oracle_capture->launch(), "projection retained replay");
                const int32_t accepted_rows = replay % 4 == 0 ? 0 : replay % 4 == 1 ? 1 : (rows + 1) / 2;
                const int32_t ok = replay % 5 != 0;
                if (accepted_capture)
                {
                    // These joins are diagnostic terminal observations only.
                    // Production orders the retained transaction graphs with
                    // its existing accepted-state producer/consumer events.
                    require(gpu.synchronizeStreamChecked(stream), "projection completed verifier observation");
                    require(backend->hostToDeviceOnStream(accepted->gpu_data_ptr(), &accepted_rows, sizeof(accepted_rows),
                            device.ordinal, publication_stream) &&
                        backend->hostToDeviceOnStream(publication_ok->gpu_data_ptr(), &ok, sizeof(ok),
                            device.ordinal, publication_stream) && accepted_capture->launch() &&
                        gpu.synchronizeStreamChecked(publication_stream), "projection accepted publication replay");
                }
                if (prove_history)
                {
                    const auto after = read_history();
                    History expected_counts{};
                    const int counted = demand == MoEGroupedPlanDemand::OrdinaryPrefill ? live
                        : demand == MoEGroupedPlanDemand::DeferredAcceptedRows && ok ? std::min(live, accepted_rows) : 0;
                    const int phase = demand == MoEGroupedPlanDemand::OrdinaryPrefill ? 2 : 4;
                    for (int slot = 0; slot < counted * top_k; ++slot)
                    {
                        const int expert = static_cast<int>(ids[slot]);
                        const int owner = replay == 0 ? initial_owners.ownerFor(0, expert)->owner_participant
                            : replay % 4 == 3 ? 0 : (expert + replay) % participants;
                        ++expected_counts[phase][expert];
                        if (owner == participant) ++expected_counts[phase + 1][expert];
                    }
                    for (int phase_index = 0; phase_index < 6; ++phase_index)
                        for (int expert = 0; expert < experts; ++expert)
                            require(after[phase_index][expert] - before[phase_index][expert] == expected_counts[phase_index][expert],
                                "projection demand counted wrong phase, rejected row, or participant");
                }
                require(backend->deviceToHostOnStream(actual.data(), output->gpu_data_ptr(), output->size_bytes(), device.ordinal, stream) &&
                    backend->deviceToHostOnStream(reference.data(), expected->gpu_data_ptr(), expected->size_bytes(), device.ordinal, stream) &&
                    backend->deviceToHostOnStream(route_observation.data(), route_columns->gpu_data_ptr(),
                        route_observation.size() * sizeof(float), device.ordinal, stream) &&
                    gpu.synchronizeStreamChecked(stream), "projection terminal observation");
                if (const auto fabric = tp.deviceCountedAllGather())
                {
                    std::vector<std::uint64_t> extents(participants), expected_extents(participants);
                    require(backend->deviceToHostOnStream(extents.data(), fabric->extentAddress(participant, 0),
                        extents.size() * sizeof(extents[0]), device.ordinal, stream) &&
                        gpu.synchronizeStreamChecked(stream), "projection terminal extent observation");
                    for (int slot = 0; slot < live * top_k; ++slot)
                    {
                        const int expert = static_cast<int>(ids[slot]);
                        const int owner = replay == 0 ? initial_owners.ownerFor(0, expert)->owner_participant
                            : replay % 4 == 3 ? 0 : (expert + replay) % participants;
                        expected_extents[owner] += maximum_layout.compactRecordWords() * sizeof(std::uint32_t);
                    }
                    require(extents == expected_extents, "projection communicated capacity rather than live owned routes");
                }
                if (shared_full_sum)
                {
                    // Observe only at the ordinary test terminal, never while
                    // capturing or between the sum's submission and its join.
                    require(backend->deviceToHostOnStream(local_shared.data(), shared->gpu_data_ptr(),
                            shared->size_bytes(), device.ordinal, stream) &&
                            gpu.synchronizeStreamChecked(stream), "projection full-row sum terminal observation");
                    for (std::size_t i = 0; i < local_shared.size(); ++i)
                    {
                        const float expected_sum = i < std::size_t(rows) * width ? summed_values[i] : shared_values[i];
                        require(std::bit_cast<std::uint32_t>(local_shared[i]) == std::bit_cast<std::uint32_t>(expected_sum),
                                "projection overlapped full-row sum changed arithmetic or touched its guard");
                    }
                }
                if (shared_columns)
                {
                    require(backend->deviceToHostOnStream(routed_reference.data(), full_routed->gpu_data_ptr(),
                            full_routed->size_bytes(), device.ordinal, stream) &&
                        backend->deviceToHostOnStream(shared_reference.data(), full_shared->gpu_data_ptr(),
                            full_shared->size_bytes(), device.ordinal, stream) &&
                        backend->deviceToHostOnStream(local_routed.data(), columns->gpu_data_ptr(),
                            columns->size_bytes(), device.ordinal, stream) &&
                        backend->deviceToHostOnStream(local_shared.data(), shared->gpu_data_ptr(),
                            shared->size_bytes(), device.ordinal, stream) &&
                        gpu.synchronizeStreamChecked(stream), "projection shared-column terminal observation");
                    // This is the old TP path's separate FP32 add, not the
                    // fused-gate kernel which may contract multiply/add. Only
                    // the test oracle executes this scalar observation fold.
                    for (int row = 0; row < rows; ++row)
                        for (int col = 0; col < width; ++col)
                        {
                            const auto index = std::size_t(row) * width + col;
                            reference[index] = row < live ? shared_reference[index] + routed_reference[index] : 0.f;
                        }
                    for (int row = 0; row < rows; ++row)
                        for (int col = 0; col < local_width; ++col)
                        {
                            const auto local = std::size_t(row) * local_width + col;
                            const auto full = std::size_t(row) * width + participant * local_width + col;
                            require(std::bit_cast<uint32_t>(local_routed[local]) ==
                                std::bit_cast<uint32_t>(routed_reference[full]), "projection routed-column oracle mismatch");
                            const float shared_value = row < live ? shared_reference[full] : 0.f;
                            require(std::bit_cast<uint32_t>(local_shared[local]) ==
                                std::bit_cast<uint32_t>(shared_value), "projection shared-column oracle mismatch");
                        }
                }
                else
                {
                // The terminal read is diagnostic-only. Capture the actual
                // producer view after the retained graph has completed, then
                // fold original router slots in the same ordered FP32 sum.
                route_dump.outputs.front().data = route_observation.data();
                SnapshotCapture snapshots;
                snapshots.captureStage(down_stage_name, route_dump);
                const auto route_snapshot = snapshots.getShared(route_keys.front());
                require(route_snapshot && route_snapshot->publication == SnapshotPublication::ColumnPartition &&
                    route_snapshot->data == route_observation, "projection route publication contract");
                for (int row = 0; row < rows; ++row)
                    for (int col = 0; col < local_width; ++col)
                    {
                        float sum = 0.f;
                        for (int slot = 0; slot < top_k; ++slot)
                            sum += route_snapshot->data[(static_cast<size_t>(row) * top_k + slot) * local_width + col];
                        require(std::bit_cast<uint32_t>(sum) ==
                            std::bit_cast<uint32_t>(actual[row * width + participant * local_width + col]),
                            "projection published route columns do not fold to the captured output");
                    }
                }
                for (std::size_t i = 0; i < actual.size(); ++i)
                    if (!std::isfinite(actual[i]) || std::bit_cast<uint32_t>(actual[i]) != std::bit_cast<uint32_t>(reference[i]))
                        throw std::runtime_error(format.name + " projection byte mismatch rows=" + std::to_string(rows) +
                            " replay=" + std::to_string(replay) + " element=" + std::to_string(i) +
                            " actual=" + std::to_string(actual[i]) + " expected=" + std::to_string(reference[i]));
                if (live > 0)
                    require(std::any_of(reference.begin(), reference.begin() + live * width,
                        [](float value) { return value != 0.f; }), "projection math requires a nonzero live-row witness");
            }
            if (config.timing_samples > 0)
            {
                // A balanced, full-live request follows the adversarial gate.
                // Both candidates see identical inputs and the same prepared
                // weights. The unsharded oracle is compute-only, not a timed
                // two-device control or a whole-model baseline.
                changeOwners(0);
                for (int slot = 0; slot < capacity * top_k; ++slot)
                {
                    ids[slot] = slot / top_k < rows ? float((slot * 17 + slot / top_k * 7) % experts) : -1.f;
                    probabilities[slot] = slot / top_k < rows ? 1.f / top_k : 0.f;
                }
                upload(indices.get(), ids); upload(weights.get(), probabilities);
                require(gpu.synchronizeStreamChecked(stream), "projection timed request admission");
                TimingEvents events(*backend, device, stream);
                for (int sample = -config.timing_warmups; sample < config.timing_samples; ++sample)
                {
                    for (int round = 0; round < 2; ++round)
                    {
                        const bool candidate = (sample + config.timing_warmups + round) % 2 == 0;
                        rendezvous.arrive_and_wait();
                        const double us = events.measure(candidate ? *capture : *oracle_capture);
                        if (sample >= 0) (candidate ? candidate_times : oracle_times).push_back(us);
                    }
                    require(backend->deviceToHostOnStream(actual.data(), output->gpu_data_ptr(), output->size_bytes(), device.ordinal, stream) &&
                        backend->deviceToHostOnStream(reference.data(), expected->gpu_data_ptr(), expected->size_bytes(), device.ordinal, stream) &&
                        gpu.synchronizeStreamChecked(stream), "projection timed result observation");
                    for (std::size_t i = 0; i < actual.size(); ++i)
                        require(std::isfinite(actual[i]) && std::bit_cast<uint32_t>(actual[i]) == std::bit_cast<uint32_t>(reference[i]),
                            "projection timed replay byte mismatch");
                    require(std::any_of(reference.begin(), reference.begin() + rows * width,
                        [](float value) { return value != 0.f; }), "projection timed nonzero witness");
                }
            }
            rendezvous.arrive_and_wait();
        }
        if (prove_history)
        {
            const auto binding = runtime.deviceOverlayServiceTelemetryBinding(0);
            std::array<DeviceMoEOverlayServiceTelemetryCell, kDeviceMoEOverlayServicePhaseCount> cells{};
            require(binding.valid() && backend->deviceToHostOnStream(cells.data(), binding.layer_telemetry,
                sizeof(cells), device.ordinal, stream) && gpu.synchronizeStreamChecked(stream),
                "projection service observation");
            for (const int phase : {1, 2})
                require(cells[phase].sample_count > 0 && cells[phase].activation_count > 0 &&
                    cells[phase].total_nanoseconds > 0 && !cells[phase].overflowed,
                    "projection gate/up economics requires real prefill and verifier service evidence");
        }
    });
}

} // namespace

void runProjectionPipelineFixture(DeviceId first, int participants,
    const ProjectionPipelineFixtureConfig &config, ProjectionPipelineTiming *timing)
{
    require(participants >= 2 && !config.rows.empty() &&
        std::all_of(config.rows.begin(), config.rows.end(), [](int rows) { return rows > 0; }) &&
        config.geometry.experts >= config.geometry.top_k && config.geometry.top_k > 0 &&
        *std::max_element(config.rows.begin(), config.rows.end()) <=
            std::numeric_limits<int>::max() / config.geometry.top_k &&
        config.geometry.width > 0 && config.geometry.width % participants == 0 &&
        config.geometry.intermediate > 0 && config.timing_warmups >= 0 && config.timing_samples >= 0 &&
        (config.output == ProjectionOutputProof::RoutedOnly ||
            (config.timing_samples == 0 && config.history == ProjectionHistoryProof::Disabled)) &&
        ((timing != nullptr) == (config.timing_samples > 0)) &&
        (!timing || (config.rows.size() == 1 && !config.format.empty())), "projection fixture configuration");
    auto *backend = getBackendFor(first);
    ASSERT_NE(backend, nullptr); ASSERT_GE(backend->deviceCount(), participants);
    std::vector<DeviceId> devices;
    std::vector<GlobalDeviceAddress> addresses;
    for (int ordinal = participants - 1; ordinal >= 0; --ordinal)
    {
        devices.push_back(first.is_cuda() ? DeviceId::cuda(ordinal) : DeviceId::rocm(ordinal));
        addresses.push_back(GlobalDeviceAddress::fromLocalDeviceId(devices.back()));
    }
    auto tp = createLocalTPContext(addresses, {},
        first.is_cuda() ? CollectiveBackendType::NCCL : CollectiveBackendType::RCCL);
    ASSERT_NE(tp, nullptr);
    std::shared_ptr<PhysicalMemoryAuthority> transport_authority;
    if (config.exchange == ProjectionExchangeProof::DeviceCounted)
    {
        const auto mpi = MPIContextFactory::global();
        const auto inventory = mpi->clusterInventory();
        require(bool(inventory), "projection counted transport inventory");
        const auto &rank = inventory->ranks.at(mpi->rank());
        const auto coverage = localTPPeerAccessCoverage(rank, devices);
        require(coverage == PeerAccessCoverage::None, "projection counted fixture must not bypass enabled native P2P");
        const MoEGroupedIntermediateLayout maximum{MoEGroupedIntermediateEncoding::FP32,
            static_cast<std::uint32_t>(config.geometry.intermediate),
            static_cast<std::uint32_t>(*std::max_element(config.rows.begin(), config.rows.end()) * config.geometry.top_k),
            static_cast<std::uint32_t>(participants)};
        const auto capacity = maximum.compactCapacityBytes();
        const auto memory = DeviceCountedAllGather::memoryFor(participants, capacity);
        PhysicalMemoryPlanBuilder bom;
        bom.add(planningObservedResource(rank, DeviceId::cpu()), PhysicalMemoryOwner::ActivationTransportStaging,
            memory.host_bytes * participants);
        for (auto device : devices) bom.add(planningObservedResource(rank, device),
            PhysicalMemoryOwner::ActivationTransportStaging, memory.device_bytes);
        transport_authority = std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<const PhysicalMemoryPlanAdmissionCertificate>(bom.build()), rank.rank);
        std::vector<void *> streams(devices.size());
        for (std::size_t i = 0; i < devices.size(); ++i)
        {
            auto &worker = GPUDeviceContextPool::instance().getContext(devices[i]);
            worker.submitAndWait([&] { streams[i] = worker.getOrCreateAuxiliaryStream("projection_counted_setup"); });
        }
        tp->installDeviceCountedAllGather(DeviceCountedAllGather::create(*transport_authority,
            DeviceId::cpu(), devices, streams, coverage, capacity));
    }
    std::barrier rendezvous(participants);
    std::vector<std::exception_ptr> errors(participants);
    std::vector<std::thread> threads(participants);
    std::vector<std::vector<double>> candidate_times(participants), oracle_times(participants);
    auto selected_formats = formats();
    if (!config.format.empty())
        std::erase_if(selected_formats, [&](const Format &format) { return format.name != config.format; });
    require(!selected_formats.empty(), "projection fixture format is not supported");
    for (int participant = 0; participant < participants; ++participant)
        threads[participant] = std::thread([&, participant] {
            try { for (const auto &format : selected_formats)
                proveFormat(devices[participant], participant, *tp, rendezvous, format, config,
                    candidate_times[participant], oracle_times[participant]); }
            catch (...) { errors[participant] = std::current_exception(); tp->requestAbort(); rendezvous.arrive_and_drop(); }
        });
    for (auto &thread : threads) thread.join();
    for (const auto &error : errors) if (error)
        try { std::rethrow_exception(error); } catch (const std::exception &failure) { ADD_FAILURE() << failure.what(); }
    if (timing && !::testing::Test::HasFailure())
    {
        // GPUs overlap: adding participant intervals would double count work.
        // Store one maximum-participant interval for each paired sample.
        std::vector<double> critical(config.timing_samples), oracle_critical(config.timing_samples);
        for (int participant = 0; participant < participants; ++participant)
        {
            require(candidate_times[participant].size() == critical.size() &&
                oracle_times[participant].size() == critical.size(), "projection timing coverage");
            for (int sample = 0; sample < config.timing_samples; ++sample)
            {
                critical[sample] = std::max(critical[sample], candidate_times[participant][sample]);
                oracle_critical[sample] = std::max(oracle_critical[sample], oracle_times[participant][sample]);
            }
        }
        std::sort(critical.begin(), critical.end());
        std::sort(oracle_critical.begin(), oracle_critical.end());
        *timing = {critical[critical.size() / 2], critical[critical.size() / 10],
            critical[critical.size() * 9 / 10], oracle_critical[oracle_critical.size() / 2]};
    }
}
} // namespace llaminar2::test
