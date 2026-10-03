/**
 * @file Test__MoEProjectionPreparationGPU.cpp
 * @brief Real-device, all-format proofs of projection-specific weight preparation.
 *
 * The candidate uses real GGUF source loading, WeightManager bindings and an admitted
 * PhysicalMemoryAuthority. Independent complete matrices provide byte oracles
 * for every native plane. No candidate-side GPU slicing, re-quantization or
 * complete down-matrix allocation may substitute for source-side slicing.
 * Captured projection arithmetic is covered by MoEProjectionBoundary; this
 * test closes production materialization, prepared lifetime, and captured route
 * publication for the exact gate/up-only movable payload.
 */
#include "execution/moe/ExpertPreparedMemoryGeometry.h"
#include "backends/GPUDeviceContextPool.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "execution/moe/DeviceMoEExpertDescriptorBuilder.h"
#include "execution/moe/MoEExpertOverlayPreparationPlan.h"
#include "execution/moe/MoEWorkspaceRequirements.h"
#include "interfaces/IWorkspaceConsumer.h"
#include "kernels/IMoEKernel.h"
#include "kernels/KernelFactory.h"
#include "execution/moe/MoEOverlayFixedDownProjectionBank.h"
#include "execution/moe/MoEOverlayParticipantResidency.h"
#include "loaders/MoEExpertSourceView.h"
#include "loaders/WeightManager.h"
#include "loaders/WeightMetadataRegistry.h"
#include "mocks/MockModelLoader.h"
#include "mocks/MockMPIContext.h"
#include "tensors/TensorFactory.h"
#include "utils/GpuPreparedGemmHarness.h"
#include "utils/QuantizedVerifierFormats.h"
#include "utils/PlanningGGUFFixture.h"
#include "utils/ScopedGPUStream.h"

#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>

namespace llaminar2::test
{
namespace
{
using Role = ExpertGemmRegistry::WeightRole;
// Matches the shared GGUF fixture. Eight experts exercise non-contiguous random
// owner selections; unequal N/K catches accidental transposes before packing.
constexpr MoEExpertProjectionOwnership::Geometry kGeometry{8, 256, 512};
constexpr std::array<WeightRole, 3> kRoles{
    WeightRole::MoEExpertGate, WeightRole::MoEExpertUp, WeightRole::MoEExpertDown};
constexpr std::array<const char *, 3> kNames{
    "blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_exps.weight", "blk.0.ffn_down_exps.weight"};

/** @brief Couple a source factory to its exact native allocation provenance. */
struct Format
{
    std::string name;
    TensorType type;
    ExpertWeightFormat identity;
    QuantizedVerifierWeightCreator create;
};

/** @return All model-loadable native formats and all three floating precisions. */
std::vector<Format> formats()
{
    std::vector<Format> result;
    for (const auto &entry : quantizedVerifierFormats())
    {
        // Q8_1 is accepted as a 2-D activation/prepared matrix, not as a
        // three-dimensional GGUF model parent. A separate negative test below
        // prevents it from being silently certified as a model expert source.
        if (entry.tensor_type == TensorType::Q8_1) continue;
        result.push_back({entry.label, entry.tensor_type,
            ExpertWeightFormat::nativeVnni({entry.source_codebook_id,
                entry.source_is_superblock, true}), entry.create});
    }
    result.push_back({"FP32", TensorType::FP32, ExpertWeightFormat::floating(TensorType::FP32),
        [](const auto &shape, uint32_t seed) { return TestTensorFactory::createFP32Random(shape, -1.f, 1.f, seed); }});
    result.push_back({"FP16", TensorType::FP16, ExpertWeightFormat::floating(TensorType::FP16),
        [](const auto &shape, uint32_t seed) { return TestTensorFactory::createFP16Random(shape, -1.f, 1.f, seed); }});
    result.push_back({"BF16", TensorType::BF16, ExpertWeightFormat::floating(TensorType::BF16),
        [](const auto &shape, uint32_t seed) { return TestTensorFactory::createBF16Random(shape, -1.f, 1.f, seed); }});
    return result;
}

/**
 * @brief Resolve an admitted native multi-GPU domain in reversed device order.
 * @param device Backend of the physical endpoints; not a slice coordinate.
 * @param degree Number of distinct local GPUs, at least two.
 * @return Exact runtime ownership used by materialization and its byte oracle.
 */
std::shared_ptr<MoEExpertOverlayRuntimePlan> runtime(DeviceId device, int degree)
{
    if (degree < 2)
        throw std::invalid_argument("Projection preparation fixture requires a native multi-GPU domain");
    auto source = std::make_shared<MoERoutedExpertPlacementPlan>();
    source->enabled = true;
    source->topology = RoutedExpertPlacementTopology::SingleDomain;
    source->continuation_domain = "compute";
    source->shared_expert_domain = "compute";
    source->residency_policy = RoutedExpertResidencyPolicy::StaticById;
    source->owner_order = RoutedExpertOwnerOrder::Random;
    RoutedExpertDomain domain;
    domain.name = "compute";
    domain.scope = ExecutionDomainScope::RANK_LOCAL;
    domain.owner_rank = 0;
    domain.backend = device.is_cuda() ? CollectiveBackendType::NCCL : CollectiveBackendType::RCCL;
    domain.routed_compute_policy = RoutedExpertComputePolicy::GateUpOwnedDownColumns;
    for (int index = 0; index < degree; ++index)
        domain.participants.push_back(device.is_cuda()
            ? GlobalDeviceAddress::cuda(degree - 1 - index)
            : GlobalDeviceAddress::rocm(degree - 1 - index));
    source->domains = {domain};
    source->routed_tiers = {RoutedExpertTier{.name = "capacity", .domain = "compute", .priority = 3}};
    source->placements = {{.layer = 0, .routed_expert_tier = std::vector<int>(kGeometry.experts, 0)}};
    return resolveMoEExpertOverlayRuntimePlan(source);
}

/** @brief Download only a named live plane at this test's terminal observation boundary. */
std::vector<uint8_t> download(DeviceId device, void *stream, const void *pointer, size_t bytes)
{
    std::vector<uint8_t> result(bytes);
    if (!pointer || !getBackendFor(device)->deviceToHost(result.data(), pointer, bytes, device.ordinal, stream))
        throw std::runtime_error("Projection preparation could not read the exact prepared plane");
    return result;
}

/** @brief Compare N slices in every K block, not an incorrect flat byte interval. */
void expectPlane(DeviceId device, void *stream, const void *candidate, const void *oracle,
    size_t bytes_per_block, const MoEExpertProjectionOwnership::Projection &projection)
{
    const size_t k_blocks = static_cast<size_t>(projection.source_columns) / 32;
    const auto actual = download(device, stream, candidate, k_blocks * projection.rows * bytes_per_block);
    const auto expected = download(device, stream, oracle, k_blocks * projection.source_rows * bytes_per_block);
    for (size_t block = 0; block < k_blocks; ++block)
        ASSERT_EQ(std::memcmp(actual.data() + block * projection.rows * bytes_per_block,
            expected.data() + (block * projection.source_rows + projection.first_row) * bytes_per_block,
            projection.rows * bytes_per_block), 0) << "K block=" << block;
}

/** @brief Join terminal observations before fixture storage or a graph retires. */
struct PublicationObservationJoin
{
    IBackend *backend;
    DeviceId device;
    void *stream;
    /** @brief This is a test-lifetime join, never a captured execution edge. */
    ~PublicationObservationJoin() { (void)backend->synchronizeStream(stream, device.ordinal); }
};

/**
 * @brief Exercise both grouped-publication geometries with real split weights.
 * @param fixed Authenticated model-lifetime down columns, not movable replicas.
 * @param movable Exact prepared gate/up pairs from the production registry.
 * @param owners Canonical participant identities, deliberately not device ordinals.
 *
 * Router and assigned-route publication must retain every locally owned pair.
 * The host oracle checks integer route geometry and exact descriptor bytes;
 * it does not compute, replay or repair any part of the captured transaction.
 */
void proveRuntimePublication(
    const std::shared_ptr<const MoEOverlayFixedDownProjectionBank> &fixed,
    const std::vector<MoEOverlayPreparedExpertPayload> &movable,
    const MoEExpertOwnerMap &owners)
{
    const auto device = fixed->device();
    auto &context = GPUDeviceContextPool::instance().getContext(device);
    context.submitAndWait([&]
    {
        constexpr int experts = kGeometry.experts, top_k = 2, capacity = 257;
        constexpr int d_model = kGeometry.model_columns, intermediate = kGeometry.intermediate_columns;
        auto *backend = getBackendFor(device);
        auto *stream = context.defaultStream();
        const auto require = [](bool ok) { if (!ok) throw std::runtime_error("Split runtime publication fixture failed"); };
        const auto requirements = device.is_cuda()
            ? MoEWorkspaceBuffers::cudaMoE(capacity, d_model, intermediate, experts, top_k)
            : MoEWorkspaceBuffers::rocmMoE(capacity, d_model, intermediate, experts, top_k);
        DeviceWorkspaceManager workspace(device, requirements.total_bytes_with_alignment() + 4 * 1024 * 1024);
        require(workspace.allocate(requirements));
        auto kernel = llaminar::v2::kernels::KernelFactory::createMoEKernel(device);
        require(kernel != nullptr);
        kernel->setGPUStream(stream);
        auto *consumer = dynamic_cast<IWorkspaceConsumer *>(kernel.get());
        require(consumer != nullptr);
        consumer->bindWorkspace(&workspace);

        DeviceMoERuntimeTable::Config config;
        config.device_id = device;
        config.num_layers = 1;
        config.num_experts = experts;
        config.top_k = top_k;
        config.mirror_to_device = true;
        config.prefill_token_capacity = capacity;
        config.fixed_down_banks = {fixed};
        DeviceMoERuntimeTable runtime(config);
        MoEPlacementUpdate placement;
        placement.epoch = 1;
        placement.expert_count = experts;
        placement.participant_id = fixed->participantId();
        placement.participant_count = owners.participants().size();
        placement.experts.resize(experts);
        placement.local_compute_mask.resize(experts);
        placement.replica_role.assign(experts, static_cast<uint8_t>(DeviceMoEReplicaRole::Primary));
        std::vector<DeviceNativeVNNIMatrixDesc> gates(experts), ups(experts);
        std::vector<DeviceMoEFloatingMatrixDesc> fp_gates(experts), fp_ups(experts);
        DeviceMoEWeightFormat family = DeviceMoEWeightFormat::NativeVNNI;
        for (int expert = 0; expert < experts; ++expert)
        {
            const auto *owner = owners.ownerFor(0, expert);
            require(owner != nullptr);
            const bool local = owner->owner_participant == placement.participant_id;
            placement.local_compute_mask[expert] = local;
            auto &desc = placement.experts[expert];
            desc.logical_expert_id = expert;
            desc.owner_participant = owner->owner_participant;
            desc.local_slot = local ? expert : -1;
            desc.projection_set = DeviceMoEProjectionSet::GateUp;
            if (!local) continue;
            desc.flags = toMoEExpertFlags(DeviceMoEExpertFlags::Valid |
                DeviceMoEExpertFlags::Resident | DeviceMoEExpertFlags::LocalCompute);
            require(exportDeviceMoEPreparedPayload(movable[expert], desc));
            family = desc.weight_format;
            gates[expert] = desc.gate; ups[expert] = desc.up;
            fp_gates[expert] = desc.floating_gate; fp_ups[expert] = desc.floating_up;
        }
        require(runtime.prepareInactiveBank(0, placement) && runtime.flipActiveBank(0, placement.epoch, stream));
        const auto state = runtime.hostLayerState(0);
        const int gate_table = deviceMoEWeightFormatIsFloating(family)
            ? kernel->uploadGroupedExpertFloatingGateUpDescriptorTables(fp_gates.data(), fp_ups.data(),
                family, experts, d_model, intermediate)
            : kernel->uploadGroupedExpertGateUpDescriptorTables(gates.data(), ups.data(), experts, d_model, intermediate);
        const int down_table = std::visit([&](const auto &table)
        {
            const auto columns = fixed->ownership().projection(WeightRole::MoEExpertDown).rows;
            if constexpr (std::is_same_v<std::decay_t<decltype(table)>,
                    MoEOverlayFixedDownProjectionBank::FloatingDescriptorTable>)
                return kernel->uploadGroupedExpertFloatingDownDescriptorTable(table.experts.data(),
                    table.format, experts, columns, intermediate);
            else
                return kernel->uploadGroupedExpertDownDescriptorTable(table.experts.data(), experts, columns, intermediate);
        }, fixed->exportDescriptorTable());
        require(gate_table >= 0 && down_table >= 0);
        auto indices = TestTensorFactory::createFP32({capacity * top_k});
        auto weights = TestTensorFactory::createFP32({capacity * top_k});
        std::fill_n(indices->mutable_data(), capacity * top_k, -1.f);
        std::fill_n(weights->mutable_data(), capacity * top_k, 0.f);
        require(indices->ensureOnDevice(device, stream) && weights->ensureOnDevice(device, stream));
        // Keep every host publication alive until the final observation join.
        std::vector<float> ids(capacity * top_k, -1.f), probabilities(capacity * top_k, 0.f);
        const auto publish_inputs = [&]
        {
            require(backend->hostToDevice(indices->gpu_data_ptr(), ids.data(), ids.size() * sizeof(float), device.ordinal, stream));
            require(backend->hostToDevice(weights->gpu_data_ptr(), probabilities.data(), probabilities.size() * sizeof(float), device.ordinal, stream));
        };
        for (const int rows : {16, 129, capacity})
            for (const bool assigned : {false, true})
            {
                SCOPED_TRACE(::testing::Message() << "rows=" << rows << " assigned=" << assigned);
                auto graph = context.createGraphCapture(stream);
                require(graph != nullptr);
                PublicationObservationJoin join{backend, device, stream};
                const auto enqueue = [&]
                {
                    if (!kernel->publishCompleteGroupedPrefillPlanFromRouter(runtime.deviceLayerState(0),
                        indices.get(), weights.get(), rows, rows, experts, top_k,
                        gate_table, down_table, true, MoEGroupedPlanDemand::None, DeviceMoEProjectionSet::GateUp)) return false;
                    return !assigned || kernel->publishCompleteGroupedPrefillPlanFromRuntimeAssignments(
                        runtime.deviceLayerState(0), rows, rows, experts, top_k, gate_table,
                        down_table, MoEGroupedPlanDemand::None, DeviceMoEProjectionSet::GateUp);
                };
                publish_inputs();
                require(enqueue() && backend->synchronizeStream(stream, device.ordinal));
                {
                    GraphCaptureGuard recording;
                    require(graph->beginCapture());
                    require(enqueue());
                    require(graph->endCapture());
                }
                require(graph->instantiate());
                for (int replay = 0; replay < 20; ++replay)
                {
                    SCOPED_TRACE(replay);
                    std::vector<int> counts(experts), offsets(experts), inverse(rows * top_k, -1);
                    for (int slot = 0; slot < rows * top_k; ++slot)
                    {
                        // Full -> empty -> sparse replay catches stale descriptors/maps.
                        const int expert = replay % 3 == 1 || (replay % 3 == 2 && slot % 2)
                            ? -1 : (slot * 3 + replay) % experts;
                        ids[slot] = static_cast<float>(expert);
                        probabilities[slot] = expert < 0 ? 0.f : 0.5f;
                        if (expert >= 0 && placement.local_compute_mask[expert]) ++counts[expert];
                    }
                    for (int expert = 1; expert < experts; ++expert)
                        offsets[expert] = offsets[expert - 1] + counts[expert - 1];
                    auto next = offsets;
                    for (int slot = 0; slot < rows * top_k; ++slot)
                    {
                        const int expert = static_cast<int>(ids[slot]);
                        if (expert >= 0 && placement.local_compute_mask[expert]) inverse[slot] = next[expert]++;
                    }
                    publish_inputs();
                    require(graph->launch());
                    const auto read = [&]<typename T>(const void *source, size_t count)
                    {
                        std::vector<T> values(count);
                        require(backend->deviceToHost(values.data(), source, count * sizeof(T), device.ordinal, stream));
                        require(backend->synchronizeStream(stream, device.ordinal));
                        return values;
                    };
                    EXPECT_EQ(read.template operator()<int>(state.expert_counts, experts), counts);
                    EXPECT_EQ(read.template operator()<int>(state.expert_offsets, experts), offsets);
                    EXPECT_EQ(read.template operator()<int>(workspace.getBuffer(MoEWorkspaceBuffers::GROUP_ORIGINAL_TO_GROUPED),
                        rows * top_k), inverse);
                    const auto local_weights = read.template operator()<float>(state.route_weights, rows * top_k);
                    for (int slot = 0; slot < rows * top_k; ++slot)
                        EXPECT_EQ(local_weights[slot], inverse[slot] < 0 ? 0.f : probabilities[slot]);
                    const auto verify_descriptors = [&]<typename Descriptor>(const std::vector<Descriptor> &expected)
                    {
                        const auto *key = device.is_cuda() ? MoEWorkspaceBuffers::CUDA_RUNTIME_PREFILL_GATE_DESC_TABLE
                            : MoEWorkspaceBuffers::ROCM_RUNTIME_PREFILL_GATE_DESC_TABLE;
                        const auto actual = read.template operator()<Descriptor>(workspace.getBuffer(key), experts);
                        const auto *down_key = device.is_cuda() ? MoEWorkspaceBuffers::CUDA_RUNTIME_PREFILL_DOWN_DESC_TABLE
                            : MoEWorkspaceBuffers::ROCM_RUNTIME_PREFILL_DOWN_DESC_TABLE;
                        const auto down = read.template operator()<Descriptor>(workspace.getBuffer(down_key), experts);
                        for (int expert = 0; expert < experts; ++expert)
                            if (counts[expert] > 0)
                            {
                                EXPECT_EQ(std::memcmp(&actual[expert], &expected[expert], sizeof(Descriptor)), 0);
                                EXPECT_FALSE(down[expert].valid());
                            }
                    };
                    if (deviceMoEWeightFormatIsFloating(family)) verify_descriptors(fp_gates);
                    else verify_descriptors(gates);
                }
            }
    });
}

/**
 * @brief Prove production preparation and retirement without allocating full down replicas.
 *
 * The oracle allocation is deliberately separate test storage. Only the
 * candidate's admitted ledger is checked against the unchanged whole-expert
 * footprint; no driver-free-memory delta is used as an accounting substitute.
 */
void prove(DeviceId first)
{
    auto *backend = getBackendFor(first);
    ASSERT_NE(backend, nullptr);
    ASSERT_GE(backend->deviceCount(), 2);
    {
        // A 2-D Q8_1 matrix is a valid kernel operand but not a model's
        // [K,N,E] parent. Reject it before admission or any device allocation.
        auto loader = MockModelLoader::createMinimal();
        WeightManager manager(*loader);
        WeightBinding binding;
        binding.binding_id = 1;
        binding.identity = makeSourceWeightIdentity(kNames[0], ModelContextId{8100}, 1);
        binding.tensor_owner = TestTensorFactory::createQ8_1Random({512, 256}, -1.f, 1.f, 71);
        loader->addTensor(kNames[0], binding.tensor_owner);
        binding.tensor = binding.tensor_owner.get();
        binding.residency.home_device = first;
        binding.residency.resident_device = first;
        FrozenModelWeightSet invalid({}, {binding});
        // Admission must succeed outside the negative assertion. Otherwise
        // an unrelated topology error could falsely certify source rejection.
        const auto admitted_runtime = runtime(first, 2);
        try
        {
            (void)manager.prepareMoEExpertOverlayWeights(
                *admitted_runtime, first, &invalid);
            FAIL() << "A two-dimensional activation matrix became a model expert source";
        }
        catch (const std::invalid_argument &error)
        {
            EXPECT_EQ(std::string(error.what()),
                "Projection preparation requires bounded GGUF [K,N,E] geometry: " +
                    std::string(kNames[0]));
        }
    }
    // Single-device execution is deliberately not this physical mode. Unit
    // source-view tests cover its unsliced mathematics and admission rejection;
    // real-device preparation exercises admitted two- and four-way domains.
    for (int degree = 2; degree <= std::min(4, backend->deviceCount()); degree *= 2)
    {
        const auto rt = runtime(first, degree);
        const auto owner_map = MoEExpertOwnerMap::build(rt->sourcePlan());
        for (const auto &format : formats())
        {
            SCOPED_TRACE(format.name + "/degree=" + std::to_string(degree));
            // Tensor views retain their source through shared_from_this.
            // Keep the independent oracle parents alive through every view.
            std::array<std::shared_ptr<TensorBase>, 3> original;
            const auto complete = MoEExpertProjectionOwnership::completeExperts(kGeometry);
            PlanningGGUFFixture file(true, false, PlanningGGUFFixture::sourceType(format.type),
                kGeometry.intermediate_columns);
            for (size_t role = 0; role < kRoles.size(); ++role)
            {
                const auto shape = complete.projection(kRoles[role]);
                original[role] = format.create({static_cast<size_t>(kGeometry.experts * shape.rows),
                    static_cast<size_t>(shape.source_columns)}, 91 + role);
                file.writePayload(kNames[role], 0, {static_cast<const uint8_t *>(original[role]->raw_data()),
                    original[role]->size_bytes()});
            }
            MockMPIContext mpi;
            TensorFactory factory(mpi);
            ModelLoader loader(&factory);
            ASSERT_TRUE(loader.loadModel(file.path()));
            const auto prep = MoEExpertOverlayPreparationPlan::build(*rt, loader);
            for (const auto &participant : rt->domainForTier(0).participants)
            {
                const auto device = participant.local_device;
                SCOPED_TRACE(device.to_string());
                ScopedGPUStream stream(device);
                const auto ownership = MoEExpertProjectionOwnership::gateUpOwnedDownColumns(
                    kGeometry, participant.participant_index, degree);
                const auto scoped = prep.filteredForDevice(device);
                auto manager = std::make_unique<WeightManager>(loader);
                size_t planned_bytes = 0, max_raw_bytes = 0, complete_bytes = 0;
                for (size_t role = 0; role < kRoles.size(); ++role)
                {
                    const auto ids = scoped.expertsForDeviceLayerRole(device, 0, static_cast<Role>(role));
                    const auto shape = ownership.projection(kRoles[role]);
                    const auto whole = complete.projection(kRoles[role]);
                    complete_bytes += resolveExpertPreparedProjectionMemoryGeometry(
                        whole.rows, whole.source_columns, format.identity).gpu_live_bytes;
                    planned_bytes += ids.size() * resolveExpertPreparedProjectionMemoryGeometry(
                        shape.rows, shape.source_columns, format.identity).gpu_live_bytes;
                    if (ids.empty()) continue;
                    // Whole gate/up selections are contiguous coalesced runs.
                    // Down slices leave gaps in the source between experts, so
                    // each slice is its own load run at degree > 1.
                    const size_t matrix_raw = original[role]->size_bytes() / kGeometry.experts
                        / static_cast<size_t>(whole.rows) * shape.rows;
                    max_raw_bytes = std::max(max_raw_bytes,
                        shape.rows == whole.rows ? matrix_raw * ids.size() : matrix_raw);
                }
                EXPECT_EQ(planned_bytes, complete_bytes * kGeometry.experts / degree);
                const PhysicalMemoryResource resource{.world_rank = 0, .device = device,
                    .total_bytes = 256u << 20, .admission_available_bytes = 256u << 20};
                const PhysicalMemoryResource host{.world_rank = 0, .device = DeviceId::cpu(),
                    .total_bytes = 256u << 20, .admission_available_bytes = 256u << 20};
                const auto staging = resolveGPUWeightLoadMemoryGeometry(max_raw_bytes, configuredGPUWeightLoadMemoryPolicy());
                PhysicalMemoryPlanBuilder builder;
                builder.add(resource, PhysicalMemoryOwner::RoutedExpertWeights, planned_bytes)
                    .add(resource, PhysicalMemoryOwner::WeightLoadStaging, staging.staging_bytes)
                    .add(host, PhysicalMemoryOwner::WeightLoadStaging, staging.host_staging_bytes);
                auto authority = std::make_shared<PhysicalMemoryAuthority>(
                    std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(builder.build()), 0);
                manager->installPhysicalMemoryAuthority(authority);
                const auto source_plan = scoped.sourceWeightPlan(loader, ModelContextId{8101});
                const auto frozen = manager->materialize(source_plan);
                ASSERT_EQ(frozen.bindings().size(), source_plan.size());
                std::vector<const TensorBase *> owned_sources;
                for (const auto &binding : frozen.bindings())
                {
                    EXPECT_EQ(binding.tensor->native_type(), format.type);
                    EXPECT_EQ(binding.identity.overlay_participant_index, participant.participant_index);
                    EXPECT_EQ(binding.identity.overlay_participant_world_rank, 0);
                    const auto role = binding.identity.role == WeightRole::MoEExpertGate ? Role::GATE :
                        binding.identity.role == WeightRole::MoEExpertUp ? Role::UP : Role::DOWN;
                    EXPECT_EQ(binding.slice.expert_ids, scoped.expertsForDeviceLayerRole(device, 0, role));
                    // Contiguous quantized selections borrow GGUF mappings;
                    // they are not duplicated heap payloads to retire. Random
                    // non-contiguous selections and floating sources own copies.
                    if (!binding.tensor->is_mmap_data())
                    {
                        ASSERT_FALSE(binding.tensor->is_view());
                        owned_sources.push_back(binding.tensor);
                    }
                }
                ASSERT_TRUE(manager->prepareMoEExpertOverlayWeights(*rt, device, &frozen));
                bool found_weight_owner = false;
                for (const auto &line : authority->rankAttestation())
                    if (line.identity.device == device && line.owner == PhysicalMemoryOwner::RoutedExpertWeights)
                    {
                        found_weight_owner = true;
                        EXPECT_EQ(line.materialized_new_bytes, planned_bytes);
                        EXPECT_EQ(line.planned_new_bytes, planned_bytes);
                    }
                ASSERT_TRUE(found_weight_owner);
                for (const auto *source : owned_sources) EXPECT_TRUE(source->is_raw_data_released());

                {
                    // Bind the same independent lifetimes that the graph uses:
                    // gate/up obey dynamic residency, while every fixed down
                    // slice is retained even when that expert is remote.
                    const auto endpoint = std::find_if(owner_map.participants().begin(),
                        owner_map.participants().end(), [&](const auto &candidate)
                        {
                            return candidate.device == device &&
                                candidate.domain_participant_index == participant.participant_index;
                        });
                    ASSERT_NE(endpoint, owner_map.participants().end());
                    auto fixed = MoEOverlayFixedDownProjectionBank::resolve(
                        manager->expertGemmRegistry(), *endpoint, 0, ownership);
                    ASSERT_EQ(fixed->engines().size(), static_cast<size_t>(kGeometry.experts));
                    const auto fixed_descriptors = fixed->exportDescriptorTable();
                    const auto down_shape = ownership.projection(WeightRole::MoEExpertDown);
                    std::visit([&](const auto &table)
                    {
                        ASSERT_EQ(table.experts.size(), static_cast<size_t>(kGeometry.experts));
                        for (const auto &entry : table.experts)
                        {
                            EXPECT_TRUE(entry.valid());
                            EXPECT_EQ(entry.n, down_shape.rows);
                            EXPECT_EQ(entry.k, down_shape.source_columns);
                        }
                    }, fixed_descriptors);
                    EXPECT_EQ(std::holds_alternative<MoEOverlayFixedDownProjectionBank::FloatingDescriptorTable>(
                        fixed_descriptors), format.identity.isFloating());
                    std::vector<MoEOverlayPreparedExpertPayload> movable;
                    std::string error;
                    const auto mask = owner_map.expertMaskForParticipant(0, endpoint->participant_id, kGeometry.experts);
                    ASSERT_TRUE(resolveMoEOverlayPreparedExpertPayloads(manager->expertGemmRegistry(),
                        *endpoint, 0, kGeometry.experts, mask, movable, &error, ownership)) << error;
                    MoEOverlayParticipantResidencyRegistry residency({
                        .owner_map = owner_map, .local_participant_ids = {endpoint->participant_id},
                        .num_layers = 1, .num_experts = kGeometry.experts, .initial_epoch = 1,
                        .projection_preparation = std::make_shared<const MoEExpertOverlayPreparationPlan>(scoped)});
                    ASSERT_TRUE(residency.finalizeInitialBanksFromPreparedRegistry(
                        manager->expertGemmRegistry(), &error)) << error;
                    EXPECT_EQ(residency.endpoint(endpoint->participant_id)->movableProjections(), DeviceMoEProjectionSet::GateUp);
                    ASSERT_TRUE(residency.registerInitialLayer(endpoint->participant_id, 0, mask, movable, &error)) << error;
                    for (int expert = 0; expert < kGeometry.experts; ++expert)
                    {
                        EXPECT_EQ(movable[expert].readyFor(DeviceMoEProjectionSet::GateUp), mask[expert]);
                        EXPECT_FALSE(movable[expert].down());
                        EXPECT_NE(fixed->engines()[expert], nullptr);
                    }
                    proveRuntimePublication(fixed, movable, owner_map);
                }

                for (const auto &request : scoped.requests())
                {
                    SCOPED_TRACE(::testing::Message() << "expert=" << request.expert_id << " role=" << static_cast<int>(request.role));
                    auto *kernel = manager->expertGemmRegistry().getEngineForParticipant("compute", device, 0,
                        participant.participant_index, 0, request.expert_id, request.role, request.projection_ownership);
                    ASSERT_NE(kernel, nullptr);
                    const size_t role = static_cast<size_t>(request.role);
                    const auto shape = ownership.projection(kRoles.at(role));
                    const auto whole = complete.projection(kRoles.at(role));
                    const auto &source = original.at(role);
                    if (format.identity.isFloating())
                    {
                        ContiguousFloatingPointWeightDescriptor descriptor;
                        ASSERT_TRUE(kernel->exportContiguousFloatingPointWeights(descriptor));
                        EXPECT_EQ(descriptor.type, format.type);
                        EXPECT_EQ(descriptor.n, shape.rows);
                        EXPECT_EQ(descriptor.k, shape.source_columns);
                        EXPECT_EQ(descriptor.bytes, static_cast<size_t>(shape.rows) * shape.source_columns
                            * format.identity.floatingElementBytes());
                        const auto actual = download(device, stream.get(), descriptor.data, descriptor.bytes);
                        const auto *expected = static_cast<const uint8_t *>(source->raw_data()) +
                            (request.expert_id * whole.rows + shape.first_row) * whole.source_columns
                            * format.identity.floatingElementBytes();
                        EXPECT_EQ(std::memcmp(actual.data(), expected, actual.size()), 0);
                    }
                    else
                    {
                        auto full_matrix = source->create_view({static_cast<size_t>(whole.rows),
                            static_cast<size_t>(whole.source_columns)},
                            static_cast<size_t>(request.expert_id) * whole.rows * whole.source_columns);
                        auto oracle = makeGpuPreparedGemm(full_matrix.get(), device);
                        DeviceNativeVNNIMatrixDesc actual, expected;
                        ASSERT_TRUE(kernel->exportNativeVNNIMatrixDesc(actual));
                        ASSERT_TRUE(oracle.kernel->exportNativeVNNIMatrixDesc(expected));
                        EXPECT_EQ(actual.n, shape.rows);
                        EXPECT_EQ(actual.k, shape.source_columns);
                        EXPECT_EQ(actual.codebook_id, expected.codebook_id);
                        EXPECT_EQ(actual.source_codebook_id, format.identity.native_vnni.codebook_id);
                        EXPECT_EQ(actual.source_is_superblock, format.identity.native_vnni.is_superblock);
                        EXPECT_TRUE(actual.source_identity_present);
                        const auto *metadata = native_vnni_formats::forQuantType(format.name);
                        ASSERT_NE(metadata, nullptr);
                        expectPlane(device, stream.get(), actual.payload, expected.payload, metadata->payload_bytes, shape);
                        expectPlane(device, stream.get(), actual.scales, expected.scales, sizeof(uint16_t), shape);
                        if (metadata->is_asymmetric)
                            expectPlane(device, stream.get(), actual.mins, expected.mins, sizeof(uint16_t), shape);
                        if (metadata->has_emins)
                            expectPlane(device, stream.get(), actual.emins, expected.emins, sizeof(uint32_t), shape);
                    }
                }
                // Prepared handles, not released sources, satisfy setup replay.
                ASSERT_TRUE(manager->prepareMoEExpertOverlayWeights(*rt, device, &frozen));
                manager.reset();
                for (const auto &line : authority->rankAttestation())
                    EXPECT_EQ(line.materialized_new_bytes, 0u);
            }
        }
    }
}

#ifdef HAVE_CUDA
/** @test All model expert formats retain exact CUDA planes and allocation bounds. */
TEST(MoEProjectionPreparationGPU, CUDA) { prove(DeviceId::cuda(0)); }
#endif
#ifdef HAVE_ROCM
/** @test ROCm consumes the same source ownership and canonical memory geometry. */
TEST(MoEProjectionPreparationGPU, ROCm) { prove(DeviceId::rocm(0)); }
#endif
} // namespace
} // namespace llaminar2::test
