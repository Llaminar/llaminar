/**
 * @file Test__MoEOverlayCPUCohortFabric.cpp
 * @brief Genuine two-socket whole/cohort physical staging and abort.
 *
 * One original GGUF/context/preparation supplies all real prepared banks and
 * PMA roots. Polling gates delay/fail an actual piece, retaining its copy, ACK
 * and abort/drain. Explicit native cases replace only the whole tier with a
 * genuine original inventory-owned GPU endpoint; no global candidate publication is invented.
 * Every native variant retains one original source-admitted GPU worker; joined calibration adds the
 * native CPU domain before weights/probe scratch are materialized.
 */
#include "execution/moe/MoEOverlayCPUServiceMeasurement.h"
#include "backends/NativeContextPreparation.h"
#include "backends/RetainedParticipantGraphExecutable.h"
#include "planning/ModelMemoryProfile.h"
#include "planning/WorkspaceMemoryEstimator.h"
#include "collective/GlobalTPContext.h"
#include "collective/NativeCPUCollectiveProgram.h"
#include "collective/backends/ShmemSpinBackend.h"
#include "kernels/cpu/gemm/CPUProjectionWorkspaceContract.h"
#include "kernels/cpu/CPUInvocationWorkspace.h"
#include "execution/moe/MoEOverlayPhysicalResidencyFabric.h"
#include "execution/moe/MoEOverlayPhysicalLaneDemand.h"
#include "execution/moe/MoEOverlayCapacityAdmission.h"
#include "execution/moe/MoEOverlayCapacityResolver.h"
#include "execution/moe/MoEExpertOverlayPreparationPlan.h"
#include "execution/moe/MoEOverlayResidencyAuthority.h"
#include "loaders/ModelContext.h"
#include "loaders/WeightManager.h"
#include "loaders/PreparedWeightStore.h"
#include "loaders/GPUVramPreflight.h"
#include "planning/RankHardwareOwnership.h"
#include "execution/moe/ExpertTierGpuPeerTransferLane.h"
#include "collective/CollectiveTimeoutPolicy.h"
#include <iostream>
#include "tensors/NativeWeightSourceRange.h"
#include "backends/HostMemoryCapacity.h"
#include "backends/BackendManager.h"
#include "utils/NodeTopology.h"
#include "utils/NUMATopology.h"
#include "utils/MPIContext.h"
#include "../../utils/PlanningGGUFFixture.h"
#include "../../utils/ServingGGUFFixturePayload.h"
#include "../../utils/CPUProjectionTestWorkspace.h"
#include "../../utils/TestTensorFactory.h"
#include "../../fixtures/collective/CPUExpertDomainSocketMembership.h"
#include "../../fixtures/moe/MoEOverlayJointPlacementCandidateTestAccess.h"
#include <gtest/gtest.h>
#include <mpi.h>
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace llaminar2::test {
namespace {
constexpr int kExperts=4,kH=256,kF=96;
constexpr size_t kStaging=4093;
using Progress=MoEOverlayResidencyWaveProgress;
/** @return Same-host distinct physical socket observations in actual world-rank order. */
std::vector<CPUExpertSocketEndpoint> originalFabricSockets(const MPIContext& mpi) {
    if(mpi.world_size()!=2)throw std::invalid_argument("CPU cohort fabric requires two real MPI ranks");
    const auto affinity=NUMATopology::detectLocalNUMANode();
    if(!affinity.detection_succeeded || affinity.local_numa_node<0)
        throw std::runtime_error("CPU cohort fabric rank has no original NUMA affinity");
    const auto topology=NodeTopology::detect();
    const auto node=std::find_if(topology.numaNodes().begin(),topology.numaNodes().end(),
        [&](const auto& item){return item.node_id==affinity.local_numa_node;});
    if(node==topology.numaNodes().end() || node->socket_id<0)
        throw std::runtime_error("CPU cohort fabric rank has no original physical socket");
    std::array<int,2> local{affinity.local_numa_node,node->socket_id};std::array<int,4> all{};
    if(MPI_Allgather(local.data(),2,MPI_INT,all.data(),2,MPI_INT,MPI_COMM_WORLD)!=MPI_SUCCESS)
        throw std::runtime_error("CPU cohort fabric socket observation failed");
    std::array<char,MPI_MAX_PROCESSOR_NAME> host{};int length=0;
    if(MPI_Get_processor_name(host.data(),&length)!=MPI_SUCCESS)
        throw std::runtime_error("CPU cohort fabric hostname observation failed");
    std::array<char,2*MPI_MAX_PROCESSOR_NAME> hosts{};
    if(MPI_Allgather(host.data(),MPI_MAX_PROCESSOR_NAME,MPI_CHAR,hosts.data(),MPI_MAX_PROCESSOR_NAME,
        MPI_CHAR,MPI_COMM_WORLD)!=MPI_SUCCESS || std::string(hosts.data())!=std::string(hosts.data()+MPI_MAX_PROCESSOR_NAME))
        throw std::runtime_error("CPU cohort fabric requires original same-host endpoints");
    std::vector<CPUExpertSocketEndpoint> result{{0,all[0],all[1]},{1,all[2],all[3]}};
    if(!validCPUExpertSocketMembership(result,2))
        throw std::runtime_error("CPU cohort fabric requires one actual MPI rank per distinct physical socket");
    return result;
}
/**
 * @brief Declare fixed tiers/cuts against observed original socket endpoints.
 * @param sockets Genuine world-rank ordered physical CPU endpoints.
 * @param whole_device Explicit whole-owner backend; CPU preserves the original host-only fixture.
 * @param whole_owner Original inventory rank owning the exact whole endpoint.
 * @return Fixed admitted tiers with no fabricated rank/socket execution.
 */
MoERoutedExpertPlacementPlan fabricOriginalPlan(const std::vector<CPUExpertSocketEndpoint>& sockets,
    DeviceId whole_device=DeviceId::cpu(),int whole_owner=0) {
    MoERoutedExpertPlacementPlan plan;
    plan.enabled=true;plan.first_model_layer=0;plan.topology=RoutedExpertPlacementTopology::TieredOverlay;
    plan.continuation_domain=plan.base_model_domain=plan.shared_expert_domain="whole";
    RoutedExpertDomain whole;
    whole.name="whole";whole.scope=ExecutionDomainScope::RANK_LOCAL;
    whole.backend=whole_device.is_cuda()?CollectiveBackendType::NCCL:
        whole_device.is_rocm()?CollectiveBackendType::RCCL:CollectiveBackendType::UPI;whole.owner_rank=whole_owner;
    whole.participants={whole_device.is_cuda()?GlobalDeviceAddress::cuda(whole_device.ordinal):
        whole_device.is_rocm()?GlobalDeviceAddress::rocm(whole_device.ordinal):
        GlobalDeviceAddress::cpu(sockets.at(size_t(whole_owner)).numa_node)};
    whole.routed_compute_policy=RoutedExpertComputePolicy::Apportioned;
    RoutedExpertDomain cohort;
    cohort.name="cohort";cohort.scope=ExecutionDomainScope::NODE_LOCAL;
    cohort.backend=CollectiveBackendType::UPI;cohort.owner_rank=-1;
    cohort.participants={GlobalDeviceAddress::cpu(sockets.at(0).numa_node),GlobalDeviceAddress::cpu(sockets.at(1).numa_node)};
    cohort.world_ranks={0,1};cohort.routed_compute_policy=RoutedExpertComputePolicy::TensorSharded;
    cohort.routed_phase_policy=RoutedExpertPhasePolicy::Uniform;
    cohort.routed_decode_assignment_policy=cohort.routed_prefill_assignment_policy=RoutedExpertAssignmentPolicy::StaticOwner;
    plan.domains={whole,cohort};
    plan.routed_tiers={{.name="whole",.domain="whole",.priority=0,.max_experts_per_layer=2},
        {.name="cohort",.domain="cohort",.priority=1,.max_experts_per_layer=2,.fallback=true}};
    plan.placements={{.layer=0,.routed_expert_tier={0,0,1,1}}};
    plan.residency_policy=RoutedExpertResidencyPolicy::StaticById;
    plan.authority_execution=resolveMoEOverlayAuthorityExecutionKind(plan);
    return plan;
}
/** @brief Real original source and physical owners, destroyed after every staged operation drains. */
struct FabricOriginal {
    PlanningGGUFFixture file{true,false,GGUFTensorType::F32,kF,std::nullopt,8,
        PlanningGGUFFixture::Geometry::Directory,1,kExperts};
    std::shared_ptr<MPIContext> mpi;
    std::shared_ptr<TensorFactory> tensors;
    // Native source outlives the model/engine owners during implicit teardown.
    std::shared_ptr<NativeContextPreparation> service_native_source;
    std::optional<PreparedGPUContextUsePermit> service_native_use;
    std::shared_ptr<ModelContext> context;
    std::shared_ptr<const MoERoutedExpertPlacementPlan> plan;
    std::shared_ptr<const MoEExpertOverlayPreparationPlan> preparation;
    std::shared_ptr<MoEOverlayResidencyAuthority> authority;
    std::shared_ptr<MoEOverlayParticipantResidencyRegistry> registry;
    std::shared_ptr<PhysicalMemoryAuthority> memory;
    std::shared_ptr<MoEOverlayMPIRemoteProjectionTransport> remote;
    std::shared_ptr<MoEOverlayPhysicalResidencyFabric> fabric;
    std::vector<MoEOverlayLayerWeightManifest> manifests;
    // These optional owners belong to the same original setup/PMA. Default
    // physical-only tests retain their established allocation/lifetime scope.
    std::optional<NativeCPUCollectiveProgram> service_program;
    std::shared_ptr<GlobalTPContext> service_domain;
    ModelMemoryProfile service_profile;
    RankInventory service_inventory;
    int whole=-1,last_shard=-1;
    ModelContextId model;
    DeviceId whole_device=DeviceId::cpu();
    int whole_owner=0; ///< Original observed GPU owner; CPU-only keeps its established rank-zero whole tier.
    std::shared_ptr<const MoEOverlayPhysicalLaneDemandPlan> physical_demand;
    /**
     * @brief Parse once, admit exact source/shadow/lane roots and publish genuine initial banks.
     * @param device Exact whole GPU identity; its original UUID/NUMA owner alone prepares native storage.
     * @param joined_service Admit the actual service probe/domain on the same original PMA.
     */
    explicit FabricOriginal(DeviceId device=DeviceId::cpu(),bool joined_service=false):whole_device(device) {
        try {
        if(!device.is_cpu() && !device.is_gpu())
            throw std::invalid_argument("Fabric whole endpoint must be CPU or explicit native GPU");
        int rank=-1,size=0;MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&size);
        mpi=MPIContextFactory::global();
        if(!mpi || mpi->rank()!=rank || mpi->world_size()!=size)
            throw std::invalid_argument("CPU cohort fabric lost its original owned MPI context");
        const auto sockets=originalFabricSockets(*mpi);
        if(device.is_gpu()) {
            const auto inventory=mpi->clusterInventory();
            if(!inventory || inventory->world_size!=size || inventory->ranks.size()!=size_t(size))
                throw std::invalid_argument("GPU/CPU fabric requires its complete original inventory");
            int matches=0;
            for(const auto& owner:inventory->ranks)for(const auto& gpu:owner.gpus)
                if(gpu.type==device.type && gpu.local_device_id==device.ordinal && rankOwnsGPUDevice(owner,gpu)) {
                    if(gpu.uuid.empty())throw std::invalid_argument("GPU/CPU fabric original device has no UUID");
                    whole_owner=owner.rank;++matches;
                }
            if(matches!=1)throw std::invalid_argument("GPU/CPU fabric requires one exact original GPU UUID/NUMA owner");
            service_inventory=inventory->getRank(rank);
            if(rank==whole_owner)requireRankLocalGPUOwnership(service_inventory,device,"GPU/CPU fabric original source");
        }
        if(joined_service && !device.is_gpu())
            throw std::invalid_argument("Joined prepared service requires disjoint GPUwhole and original CPUcohort");
        if(!hasCPUBackend())initCPUBackend(sockets.at(size_t(rank)).numa_node);
        populateServingGGUFFixture(file);
        tensors=std::make_shared<TensorFactory>(*mpi);
        ModelContextConfig source;source.mpi_ctx=mpi;source.factory=tensors.get();source.use_mmap=false;
        context=ModelContext::create(file.path(),source);
        if(!context)throw std::runtime_error("CPU cohort fabric original context did not load");
        model=ModelContextId{reinterpret_cast<uint64_t>(context.get())};
        plan=std::make_shared<const MoERoutedExpertPlacementPlan>(fabricOriginalPlan(sockets,whole_device,whole_owner));
        const auto runtime=resolveMoEExpertOverlayRuntimePlan(plan,{.current_world_rank=rank,.validate_mvp_root_reachability=false});
        if(!runtime)throw std::runtime_error("CPU cohort fabric original runtime is absent");
        auto& loader=context->concreteLoader();
        preparation=std::make_shared<const MoEExpertOverlayPreparationPlan>(MoEExpertOverlayPreparationPlan::build(*runtime,loader));
        authority=std::make_shared<MoEOverlayResidencyAuthority>(MoEOverlayResidencyAuthority::Config{
            .initial_plan=*plan,.model_metadata={.num_layers=1,.num_experts=kExperts,.d_model=kH,
                .routed_intermediate_size=kF,.main_inference_layer_count=1},
            .projection_preparation=preparation,.maintenance_mode=MoERebalanceRuntimeMode::Off,
            .participant_rebalance_policy={.enabled=false}});
        const auto& topology=authority->snapshot()->placementOwnerMap();
        std::vector<int> local;
        for(const auto& member:topology.participants()) {
            if(member.domain_name=="whole")whole=member.participant_id;
            if(member.domain_name=="cohort" && member.domain_participant_index==1)last_shard=member.participant_id;
            if(member.world_rank_known && member.world_rank==rank)local.push_back(member.participant_id);
        }
        if(whole<0 || last_shard<0 || local.empty())throw std::runtime_error("CPU cohort fabric lost original targets");
        std::sort(local.begin(),local.end());
        auto loading=preparation->filteredForDevice(DeviceId::cpu()).sourceWeightPlan(loader,model);
        size_t resident_bytes=0;
        const auto placement=loader.nativeWeightSourcePlacement();
        for(const auto& request:loading.requirements())
            for(const auto& projection:projectNativeWeightSource(request,loader.nativeWeightSourceGeometry(request.source_name.empty()?request.canonical_name:request.source_name)))
                resident_bytes+=CPUStorageRoot<float>(projection.finalRows()*projection.finalColumns(),
                    PhysicalMemoryOwner::RoutedExpertWeights,DeviceId::cpu(),placement).physicalBytes();
        manifests=buildMoEOverlayLayerWeightManifestFromLoader(loader,1,kExperts);
        size_t shadow_bytes=0,gpu_shadow_bytes=0;
        for(int id:local) {
            const auto* member=topology.participantForId(id);
            const auto selected=member->domain_name=="cohort"?preparation->cpuShardSourceFor(0,*member):nullptr;
            const auto local_manifest=selected?selected->cohort()->localManifest(selected->participantIndex()):manifests.front();
            const auto footprint=MoEOverlayCapacityResolver::preparedFootprints({local_manifest}).front();
            if(member->device.is_gpu())gpu_shadow_bytes+=footprint.gpu_shadow_bytes;
            else shadow_bytes+=footprint.cpu_shadow_bytes;
        }
        const MoEOverlayCapacityAdmissionPolicy storage{
            .migration_storage=MoEOverlayMigrationStorageKind::PhysicalResidencyFabric,
            .shadow_slots_per_endpoint_layer=1,.staging_capacity_bytes=kStaging,
            .maximum_concurrent_cycles=1,.maximum_execution_streams=1,.maximum_cycles_per_layer=1,
            .distributed_transport=true,.overlay_world_size=2};
        const auto staging=MoEOverlayCapacityAdmission::transferStagingBOM(*plan,2,storage,manifests,kExperts);
        size_t migration_bytes=0,gpu_migration_bytes=0;
        for(const auto& charge:staging)if(charge.world_rank==rank) {
            if(charge.device.is_cpu())migration_bytes+=charge.bytes;
            else if(charge.device==whole_device)gpu_migration_bytes+=charge.bytes;
            else throw std::runtime_error("Fabric staging introduced a foreign allocator");
        }
        const auto execution_counts=MoEOverlayCapacityAdmission::executionStreamCounts(*plan,rank,storage.maximum_execution_streams);
        if(!execution_counts.empty()) {
            auto demand=std::make_shared<MoEOverlayPhysicalLaneDemandPlan>(configuredMoEOverlayLaneDemand(
                *plan,manifests,kExperts,MoEOverlayPhysicalLaneScope::communicator(size),
                storage.maximum_concurrent_cycles,storage.maximum_cycles_per_layer));
            demand->original_execution_streams=execution_counts;
            bindOriginalMoEOverlayGpuChoices(*demand,rank,[&](DeviceId accessor,DeviceId allocation) {
                return originalMoEOverlayDirectedPeerAccess(service_inventory,accessor,allocation);
            });
            physical_demand=std::move(demand);
        }
        const auto capacity=observeSystemMemoryCapacity();
        const PhysicalMemoryResource cpu{rank,DeviceId::cpu(),capacity.total_bytes,capacity.admission_available_bytes};
        PhysicalMemoryPlanBuilder budget;
        budget.add(cpu,PhysicalMemoryOwner::RoutedExpertWeights,resident_bytes);
        budget.add(cpu,PhysicalMemoryOwner::ExpertShadowSlots,shadow_bytes);
        budget.add(cpu,PhysicalMemoryOwner::ExpertMigrationStaging,migration_bytes);
        size_t cpu_service_probe=0,gpu_service_host_probe=0;
        if(joined_service) {
            using Service=MoEOverlayCPUServiceMeasurement;
            service_profile=ModelMemoryProfile::fromLoader(loader);
            const auto inventory=mpi->clusterInventory();
            service_inventory=inventory->ranks.at(size_t(rank));
            std::vector<NativeCPUCollectiveProgram::Participant> members;
            for(int member=0;member<2;++member)
                members.push_back({member,GlobalDeviceAddress::cpu(sockets.at(member).numa_node,
                    inventory->ranks.at(member).hostname)});
            service_program.emplace(NativeCPUCollectiveProgram::expertShardRoutePartials(
                1,CollectiveBackendType::UPI,members));
            const auto member=std::find_if(topology.participants().begin(),topology.participants().end(),[&](const auto& value){
                return value.domain_name=="cohort" && value.world_rank_known && value.world_rank==rank;
            });
            if(member==topology.participants().end())throw std::logic_error("Joined service lost original cohort member");
            const auto selected=preparation->cpuShardSourceFor(0,*member);
            if(!selected)throw std::logic_error("Joined service lost original local cohort cuts");
            auto workspace=cpuSwiGLUWorkspaceRequirements(Service::kMaximumRows,kF/2);
            for(const auto role:{WeightRole::MoEExpertGate,WeightRole::MoEExpertUp,WeightRole::MoEExpertDown})
                workspace.merge(CPUProjectionWorkspaceContract::sourceNative(TensorType::FP32,
                    {.rows=Service::kMaximumRows,.n=role==WeightRole::MoEExpertDown?kH:kF/2,
                     .k=role==WeightRole::MoEExpertDown?kF/2:kH,.workers=1,
                     .execution=CPUExecutionGeometry::local(),.numerical_policy=CPUProjectionNumericalPolicy::GPUAlignedExpert}));
            size_t probe=0;
            for(const auto phase:{ExpertHistogramSource::PrefillChunk,ExpertHistogramSource::DecodeToken,ExpertHistogramSource::GroupedVerifier})
                probe=std::max(probe,Service::cpuJointAllocationBytes(selected->cohort(),selected->participantIndex(),phase,
                    workspace.total_bytes_with_alignment()));
            cpu_service_probe=probe;
            service_program->contributeDomainBOM(budget,cpu,rank);
            NativeCPUCollectiveProgram::contributeProcessPoolBOM(budget,cpu);
            if(rank==0)budget.add(cpu,PhysicalMemoryOwner::ActivationTransportStaging,
                ShmemSpinArena::allocationGeometry(2).mapped_bytes);
        }

        if(whole_device.is_gpu() && rank==whole_owner) {
            // The same preparation declares the genuine GPU source owners. GPU
            // upload rings coexist with CPU source/fabric staging and are additive.
            const auto footprint=MoEOverlayCapacityResolver::preparedFootprints(manifests).front();
            auto gpu_loading=preparation->filteredForDevice(whole_device).sourceWeightPlan(loader,model);
            size_t maximum_source_bytes=0;
            for(const auto& request:gpu_loading.requirements()) {
                const auto geometry=loader.nativeWeightSourceGeometry(request.source_name.empty()?request.canonical_name:request.source_name);
                size_t request_bytes=0;
                for(const auto& projection:projectNativeWeightSource(request,geometry))
                    request_bytes+=projection.range.nativeBytes()*projection.matrix_count;
                maximum_source_bytes=std::max(maximum_source_bytes,request_bytes);
            }
            const auto upload=resolveGPUWeightLoadMemoryGeometry(maximum_source_bytes,configuredGPUWeightLoadMemoryPolicy());
            auto* backend=getBackendFor(whole_device);
            if(!backend)throw std::runtime_error("Explicit floating Fabric backend is unavailable");
            const PhysicalMemoryResource gpu{rank,whole_device,backend->deviceMemoryTotal(whole_device.ordinal),
                backend->deviceMemoryFree(whole_device.ordinal)};
            // Native movement, even without service calibration, must borrow
            // the SAME original source which owns the final weights and PMA.
            const std::array observations{cpu,gpu};
            const std::array selected{whole_device};
            service_native_source=NativeContextPreparation::create(observations,rank,selected);
            service_native_use.emplace(service_native_source->acquire(whole_device));
            context->bindNativeContextPreparation(service_native_source);
            service_native_source->freezeForPlanning();
            for(const auto& row:service_native_source->mandatoryBOM().resources())budget.add(row);
            for(const auto& [endpoint,count]:execution_counts) {
                if(endpoint!=whole_device)throw std::invalid_argument("GPU/CPU fixture stream plan introduced a foreign GPU");
                const auto streams=TransferEngine::persistentTransferExecutionLanesBOM(count,cpu,gpu);
                for(const auto& row:streams.resources())budget.add(row);
            }
            for(const auto& [endpoint,count]:originalMoEOverlayPeerEventCounts(*physical_demand)) {
                if(endpoint!=whole_device)throw std::invalid_argument("GPU/CPU fixture event plan introduced a foreign GPU");
                const auto unit=ExpertTierGpuPeerTransferLane::nativeEventsBOM(cpu,gpu,false);
                for(size_t instance=0;instance<count;++instance)
                    for(const auto& row:unit.resources())budget.add(row);
            }
            if(joined_service) {
                const auto general=RetainedParticipantGraphExecutable::generalBOM(cpu,gpu);
                const auto calibration=RetainedParticipantGraphExecutable::calibrationBOM(cpu,gpu);
                for(const auto& row:general.resources())budget.add(row);
                for(const auto& row:calibration.resources())budget.add(row);
                const auto found=std::find_if(service_inventory.gpus.begin(),service_inventory.gpus.end(),[&](const auto& observed){
                    return observed.type==whole_device.type && observed.local_device_id==whole_device.ordinal;
                });
                if(found==service_inventory.gpus.end() || found->compute_units<=0)
                    throw std::runtime_error("Joined service lost actual GPU inventory geometry");
                const auto invocation=WorkspaceMemoryEstimator::estimatePreparedExpertService(service_profile,
                    {.device=whole_device,.device_compute_units=found->compute_units,.resident_graph_rows=MoEOverlayCPUServiceMeasurement::kMaximumRows,
                     .first_layer=0,.last_layer=0,.apportioned_routed_experts=true});
                const auto probe=MoEOverlayCPUServiceMeasurement::gpuAllocationBytes({kH,kF},invocation);
                gpu_service_host_probe=probe.host_bytes;
                budget.add(gpu,PhysicalMemoryOwner::ExecutionWorkspace,probe.device_bytes);
            }
            budget.add(gpu,PhysicalMemoryOwner::RoutedExpertWeights,2*footprint.gpu_live_bytes)
                .add(gpu,PhysicalMemoryOwner::ExpertShadowSlots,gpu_shadow_bytes)
                .add(gpu,PhysicalMemoryOwner::ExpertMigrationStaging,gpu_migration_bytes)
                .add(gpu,PhysicalMemoryOwner::WeightLoadStaging,upload.staging_bytes)
                .add(cpu,PhysicalMemoryOwner::WeightLoadStaging,upload.host_staging_bytes);
        }
        if(joined_service)
            // Target probes execute serially; shared CPU scratch admission is
            // the maximum actual family, not the sum of nonconcurrent roots.
            budget.add(cpu,PhysicalMemoryOwner::ExecutionWorkspace,std::max(cpu_service_probe,gpu_service_host_probe));
        const auto certificate=std::make_shared<const PhysicalMemoryPlanAdmissionCertificate>(budget.build());
        if(service_native_source) {
            service_native_source->admitFinal(certificate);
            memory=service_native_source->memory();
        } else memory=std::make_shared<PhysicalMemoryAuthority>(certificate,rank);
        auto manager=context->concreteWeightManager();manager->installPhysicalMemoryAuthority(memory);
        const auto frozen=manager->materialize(loading);
        if(!manager->prepareMoEExpertOverlayWeights(*runtime,DeviceId::cpu(),&frozen,nullptr,
            PreparedWeightAdmission::AllocateCompleteSet,preparation))
            throw std::runtime_error("CPU cohort fabric original preparation did not complete");
        if(whole_device.is_gpu() && rank==whole_owner) {
            auto gpu_loading=preparation->filteredForDevice(whole_device).sourceWeightPlan(loader,model);
            const auto gpu_frozen=manager->materialize(gpu_loading);
            if(!manager->prepareMoEExpertOverlayWeights(*runtime,whole_device,&gpu_frozen,nullptr,
                PreparedWeightAdmission::AllocateCompleteSet,preparation))
                throw std::runtime_error("Fabric genuine GPU source preparation did not complete");
        }
        registry=std::make_shared<MoEOverlayParticipantResidencyRegistry>(MoEOverlayParticipantResidencyRegistry::Config{
            .local_participant_ids=local,.num_layers=1,.num_experts=kExperts,.initial_epoch=1,
            .retained_epoch_capacity=3,.projection_preparation=preparation,.original_model=model});
        std::string error;
        if(!registry->bindOriginalPreparedContext(context,preparation,&error) ||
            !registry->finalizeInitialBanksFromPreparedRegistry(manager->expertGemmRegistry(),&error))throw std::runtime_error(error);
        const auto demand=physical_demand?*physical_demand:configuredMoEOverlayLaneDemand(
            *plan,manifests,kExperts,MoEOverlayPhysicalLaneScope::communicator(size),
            storage.maximum_concurrent_cycles,storage.maximum_cycles_per_layer);
        remote=std::make_shared<MoEOverlayMPIRemoteProjectionTransport>(MoEOverlayMPIRemoteProjectionTransport::Config{
            .mpi_context=mpi,.lane_budget=remoteMoEOverlayProjectionLaneBudget(demand),
            .staging_capacity_bytes=kStaging,.perf_device="native-cpu-cohort-fabric"});
        fabric=MoEOverlayPhysicalResidencyFabric::create({.memory_authority=memory,.registry=registry,
            .initial_snapshot=authority->snapshot(),.physical_gpu_lane_demand=physical_demand,
            .layer_weight_manifest=manifests,.remote_projection_transport=remote,
            .shadow_slots_per_endpoint_layer=1,.staging_capacity_bytes=kStaging,.maximum_concurrent_cycles=1,
            .maximum_execution_streams=1,.maximum_cycles_per_layer=1,.perf_device="native-cpu-cohort-fabric"});
        if(physical_demand) {
            size_t original_events=0;
            for(const auto& [endpoint,count]:originalMoEOverlayPeerEventCounts(*physical_demand)) {
                if(endpoint!=whole_device)throw std::logic_error("Physical event materialization introduced a foreign GPU");
                original_events=checkedMoEOverlayLaneAdd(original_events,count);
            }
            if(fabric->stats().direct_gpu_peer_lanes!=original_events)
                throw std::logic_error("Physical peer event incidence disagrees with the original shared demand");
            if(!service_native_source || service_native_source->memory()!=memory ||
                context->nativeContextPreparation()!=service_native_source)
                throw std::logic_error("Physical GPU fabric lost its original prepared source/PMA");
        }
        if(memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::RoutedExpertWeights,
            PhysicalMemoryMaterializationKind::NewAllocation)!=resident_bytes ||
            memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::ExpertShadowSlots,
                PhysicalMemoryMaterializationKind::NewAllocation)!=shadow_bytes)
            throw std::runtime_error("CPU cohort fabric source/shadow roots differ from their original exact PMA declarations");
        if(joined_service) {
            std::variant<NativeCPUCollectiveProgram::DomainUse,NativeCPUCollectiveProgram::ProcessUse> use(
                std::in_place_type<NativeCPUCollectiveProgram::DomainUse>,service_program->admit(memory,rank));
            service_domain=GlobalTPContext::createAdmittedCPUExpertDomain(MPI_COMM_WORLD,*service_program,std::move(use));
            if(!service_domain)throw std::runtime_error("Joined service original CPU domain was not created");
        }
        } catch(...) {
            try { retireOriginal(); } catch(const std::exception& error) {
                std::cerr<<"GPU/CPU fabric cold retirement failed: "<<error.what()<<'\n';std::terminate();
            }
            throw;
        }
    }
    /** @brief Retire every actual borrower before the one original native context. */
    ~FabricOriginal() noexcept {
        try { retireOriginal(); } catch(const std::exception& error) {
            std::cerr<<"GPU/CPU fabric original retirement failed: "<<error.what()<<'\n';std::terminate();
        }
    }
    /** @brief Existing joined cleanup also applies to ordinary native physical movement. */
    void retireOriginal() {
        if(service_domain) {
            if(!service_domain->retireOriginalCPUExpertDomain())
                throw std::runtime_error("GPU/CPU fabric original CPU domain did not retire");
            service_domain.reset();
        }
        fabric.reset();registry.reset();remote.reset();authority.reset();preparation.reset();context.reset();
        if(service_native_source) {
            const bool had_worker=service_native_use.has_value();service_native_use.reset();
            const auto retired=service_native_source->retireReleasedContexts();
            if(had_worker && retired!=1)throw std::runtime_error("GPU/CPU fabric failed exact original GPU retirement");
            service_native_source.reset();
        }
    }
    /** @brief Copy only an observed terminal tensor through the retained original worker stream. */
    void readTerminalWeights(DeviceId device,const void* source,size_t bytes,void* destination) const {
        if(!service_native_use || service_native_use->device()!=device || service_native_use->memory()!=memory)
            throw std::invalid_argument("GPU/CPU fabric terminal read lost original device/source/PMA");
        const auto worker=service_native_use->worker();
        worker->submitAndWait([&] {
            auto* backend=getBackendFor(device);void* stream=worker->defaultStream();
            if(!backend || reinterpret_cast<uintptr_t>(stream)<=2 ||
                !backend->deviceToHost(destination,source,bytes,device.ordinal,stream) ||
                !backend->synchronizeStream(stream,device.ordinal)) {
                std::cerr<<"GPU/CPU fabric exact terminal copy could not retire\n";std::terminate();
            }
        });
    }
    /** @return Original authority-authored swap, without publishing any candidate bank. */
    MoEOverlayResidencyTransaction transaction() const {
        auto placements=plan->placements;
        std::swap(placements.front().routed_expert_tier[0],placements.front().routed_expert_tier[2]);
        return MoEOverlayJointPlacementCandidateTestAccess::prepareTransaction(*authority,std::move(placements));
    }
};
/** @brief Own every real operation until normal completion or actual abort/drain. */
class ComponentWave {
public:
    MoEOverlayParticipantPreparedTransfers prepared;
    /** @brief Reserve full physical work, preserving the unchanged global authority. */
    ComponentWave(FabricOriginal& original,const MoEOverlayResidencyTransaction& transaction)
        :prepared(original.fabric->prepareTransfers(transaction,original.registry->localParticipantIds())) {
        if(prepared.status!=MoEOverlayResidencyStageStartStatus::Started) {
            const auto error=prepared.error;
            abortAndDrain(); // Failed prefixes are inert today, but still consume the real cleanup interface.
            throw std::runtime_error(error);
        }
    }
    /** @brief Drain before any assertion unwinds slots or MPI owners. */
    ~ComponentWave() {if(!drained_) {try{abortAndDrain();}catch(...){std::terminate();}}}
    /** @brief Delegate abort/drain to original operations, never manufacture readiness. */
    void abortAndDrain() {
        for(auto& migration:prepared.migrations)for(auto& piece:migration.operations)if(piece.operation)piece.operation->abort();
        const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);
        for(;;) {
            bool ready=true;std::string error;
            for(auto& migration:prepared.migrations)for(auto& piece:migration.operations) {
                if(!piece.operation)continue;
                const auto status=piece.operation->pollAbort(&error);
                if(status==Progress::Failed)throw std::runtime_error(error);
                ready=ready && status==Progress::Ready;
            }
            if(ready){drained_=true;return;}
            if(std::chrono::steady_clock::now()>=deadline)throw std::runtime_error("CPU cohort fabric abort exceeded standard deadline");
            std::this_thread::yield();
        }
    }
private:bool drained_=false;
};
/** @brief Gate visibility of a real piece; source/copy/ACK/abort remain production-owned. */
class LastPieceGate final:public IMoEOverlayTierTransferOperation {
public:
    /** @brief Retain the actual fabric operation and all its source lifetimes. */
    explicit LastPieceGate(std::unique_ptr<IMoEOverlayTierTransferOperation> operation):operation_(std::move(operation)){}
    bool released=false,failed=false;
    /** @brief Inject only the deterministic polling edge before genuine progress. */
    Progress poll(std::string* error) noexcept override {
        if(failed){if(error)*error="component injected final-piece failure";return Progress::Failed;}
        return released?operation_->poll(error):Progress::Pending;
    }
    /** @brief Preserve actual asynchronous cancellation. */
    void abort() noexcept override {operation_->abort();}
    /** @brief Preserve real cancellation quiescence. */
    Progress pollAbort(std::string* error) noexcept override{return operation_->pollAbort(error);}
    /** @return Original completed physical measurement. */
    std::optional<ExpertTierProjectionTransferMeasurement> completedMeasurement() const noexcept override{return operation_->completedMeasurement();}
private:std::unique_ptr<IMoEOverlayTierTransferOperation> operation_;
};
/** @return Exact remote last-member Down piece, named identically on both actual ranks. */
LastPieceGate* delayLastPiece(ComponentWave& wave,const FabricOriginal& original,const MoEOverlayResidencyTransaction& transaction) {
    for(size_t index=0;index<wave.prepared.migrations.size();++index)
        if(transaction.migrations[index].expert_id==2)
            for(auto& piece:wave.prepared.migrations[index].operations)
                if(piece.projection==ExpertTierWeightProjection::Down && piece.source_participant==original.last_shard && piece.destination_participant==original.whole) {
                    auto gate=std::make_unique<LastPieceGate>(std::move(piece.operation));
                    auto* result=gate.get();piece.operation=std::move(gate);return result;
                }
    throw std::runtime_error("CPU cohort fabric omitted its original remote last piece");
}
/** @brief Poll actual pieces with symmetric test-only consensus, excluding the named gate. */
void pollComponent(ComponentWave& wave,const LastPieceGate* withheld=nullptr) {
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);
    for(;;) {
        int all_local=1;std::string error;
        for(auto& migration:wave.prepared.migrations)for(auto& piece:migration.operations) {
            if(piece.operation.get()==withheld)continue;
            const auto progress=piece.operation->poll(&error);
            if(progress==Progress::Failed)throw std::runtime_error(error);
            if(progress!=Progress::Ready)all_local=0;
        }
        int all_ranks=0;
        if(MPI_Allreduce(&all_local,&all_ranks,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD)!=MPI_SUCCESS)
            throw std::runtime_error("CPU cohort fabric component progress vote failed");
        if(all_ranks)return;
        if(std::chrono::steady_clock::now()>=deadline)throw std::runtime_error("CPU cohort fabric staging exceeded standard deadline");
        std::this_thread::yield();
    }
}
/** @brief Actual previously observed terminal boundary, never an admission/readiness token. */
enum class ComponentTerminalQuery { TransferReady, AbortDrained };
/**
 * @brief Re-query only genuine terminal operations without consuming a new reservation.
 * @param wave Actual staged operations, already wholly Ready or abort-drained.
 * @param remote Original real transport supplying immutable traffic/counter observations.
 * @param phase Actual terminal boundary already observed through the original operation.
 */
void expectStableTerminalQueries(ComponentWave& wave,
    const MoEOverlayMPIRemoteProjectionTransport& remote,ComponentTerminalQuery phase)
{
    const auto before=remote.stats();
    for(int repeat=0;repeat<20;++repeat)
        for(auto& migration:wave.prepared.migrations)for(auto& piece:migration.operations) {
            ASSERT_TRUE(piece.operation);
            std::string error;
            EXPECT_EQ(phase==ComponentTerminalQuery::AbortDrained?piece.operation->pollAbort(&error):piece.operation->poll(&error),Progress::Ready)
                << error;
        }
    const auto after=remote.stats();
    EXPECT_EQ(after.bytes_sent,before.bytes_sent);EXPECT_EQ(after.bytes_received,before.bytes_received);
    EXPECT_EQ(after.operations_completed,before.operations_completed);
    EXPECT_EQ(after.operations_aborted,before.operations_aborted);
    EXPECT_EQ(after.protocol_failures,before.protocol_failures);
    EXPECT_EQ(after.active_mpi_requests,0u);
}
/** @return Independent originally authored float at exact original expert/N/K coordinates. */
float sourceValue(int expert,int row,int column,int columns,int rows) {
    const size_t index=(size_t(expert)*rows+size_t(row))*columns+size_t(column);
    return 0.001f*float(1+index%29)/29.0f;
}
/** @brief Check one final owner, original-coordinate bytes and real one-hot GEMM. */
void expectDestination(const FabricOriginal& original,const MoEOverlayResidencyTransaction& transaction,
    size_t migration_index,const MoEOverlayParticipantDestinationArrival& destination) {
    const auto& edge=transaction.migrations[migration_index];
    std::array<std::shared_ptr<ITensorGemm>,3> engines;
    std::array<MoEExpertProjectionOwnership::Projection,3> projections;
    std::string error;
    if(destination.arrival->isCPUJointFragment()) {
        MoEOverlayPreparedCPUFragmentPayload payload;
        ASSERT_TRUE(destination.arrival->completeCPUFragment(payload,&error))<<error;
        EXPECT_EQ(payload.model(),original.model);EXPECT_EQ(payload.preparation(),original.preparation);
        ASSERT_EQ(payload.cohort(),edge.destination.shard_cohort);
        const auto member=std::find_if(payload.cohort()->participants().begin(),payload.cohort()->participants().end(),
            [&](const auto& p){return p.participant_id==destination.participant_id;});
        ASSERT_NE(member,payload.cohort()->participants().end());
        const auto ordinal=size_t(member-payload.cohort()->participants().begin());
        engines={payload.gate(),payload.up(),payload.down()};
        const auto& owned=payload.cohort()->ownership(ordinal);
        projections={owned.projection(WeightRole::MoEExpertGate),owned.projection(WeightRole::MoEExpertUp),owned.projection(WeightRole::MoEExpertDown)};
        MoEOverlayPreparedExpertPayload invalid;
        EXPECT_FALSE(destination.arrival->completePayload(invalid,&error));
    } else {
        MoEOverlayPreparedExpertPayload payload;
        ASSERT_TRUE(destination.arrival->completePayload(payload,&error))<<error;
        engines={payload.gate(),payload.up(),payload.down()};
        for(size_t role=0;role<3;++role) {
            const auto& matrix=original.manifests.front().projections[role];
            projections[role]={MoEProjectionResidence::ExpertOwner,matrix.N,matrix.K,0,matrix.N,0,matrix.K};
        }
    }
    const auto* participant=transaction.candidate->placementOwnerMap().participantForId(destination.participant_id);
    ASSERT_TRUE(participant);
    const auto destination_device=participant->device;
    for(size_t role=0;role<3;++role) {
        const auto& engine=engines[role];ASSERT_TRUE(engine);
        ContiguousFloatingPointWeightDescriptor bytes;
        ASSERT_TRUE(engine->exportContiguousFloatingPointWeights(bytes));ASSERT_TRUE(bytes.valid());
        ASSERT_EQ(bytes.type,TensorType::FP32);const auto& owned=projections[role];
        EXPECT_EQ(bytes.n,owned.rows);EXPECT_EQ(bytes.k,owned.columns);
        std::vector<float> downloaded;
        if(destination_device.is_gpu()) {
            // Diagnostic final-byte observation only, after every actual Fabric
            // operation is Ready. This is not a production copy/substitute lane.
            const auto device=destination_device;
            downloaded.resize(size_t(bytes.n)*bytes.k);
            original.readTerminalWeights(device,bytes.data,bytes.bytes,downloaded.data());
        }
        const auto* values=downloaded.empty()?static_cast<const float*>(bytes.data):downloaded.data();
        for(int row=0;row<owned.rows;++row)for(int column=0;column<owned.columns;++column)
            ASSERT_EQ(std::bit_cast<uint32_t>(values[size_t(row)*owned.columns+column]),
                std::bit_cast<uint32_t>(sourceValue(edge.expert_id,owned.first_row+row,owned.first_column+column,
                    owned.source_columns,owned.source_rows)));
        if(destination_device.is_gpu())continue; // Actual CPU one-hot GEMM remains the independent execution oracle.
        auto input=TestTensorFactory::createFP32Zeros({1,size_t(bytes.k)});
        auto output=TestTensorFactory::createFP32Zeros({1,size_t(bytes.n)});
        const auto requirements=cpuProjectionTestRequirements(1,{engine.get()});
        ASSERT_EQ(requirements.total_bytes_with_alignment(),0u); // Native FP32 execution owns no transform scratch.
        for(int column:{0,bytes.k/2,bytes.k-1}) {
            std::fill_n(input->mutable_data(),bytes.k,0.f);input->mutable_data()[column]=1.f;
            ASSERT_TRUE(engine->multiply_tensor(input.get(),output.get(),1,bytes.n,bytes.k,true,1.f,0.f,nullptr,
                original.mpi.get(),-1,nullptr));
            for(int row=0;row<bytes.n;++row)
                EXPECT_EQ(std::bit_cast<uint32_t>(static_cast<const float*>(output->raw_data())[row]),
                    std::bit_cast<uint32_t>(values[size_t(row)*bytes.k+column]));
        }
    }
}
} // namespace
/** @test Genuine staged whole/cohort pieces withhold readiness until the last real piece completes. */
TEST(MoEOverlayCPUCohortFabric, RealTwoSocketStageDelaysFinalPieceAndExecutesExactDestinations) {
    FabricOriginal original;
    RecordProperty("physical_sockets",2);RecordProperty("mpi_ranks",2);
    const auto transaction=original.transaction();ASSERT_TRUE(transaction.valid());
    ASSERT_EQ(transaction.migrations.size(),2u);
    ComponentWave wave(original,transaction);auto* gate=delayLastPiece(wave,original,transaction);ASSERT_TRUE(gate);
    pollComponent(wave,gate);
    if(original.mpi->rank()==original.whole_owner)
        for(size_t index=0;index<wave.prepared.migrations.size();++index)if(transaction.migrations[index].expert_id==2)
            for(const auto& destination:wave.prepared.migrations[index].destination_arrivals) {
                MoEOverlayPreparedExpertPayload incomplete;std::string error;
                EXPECT_FALSE(destination.arrival->completePayload(incomplete,&error));
            }
    EXPECT_EQ(original.authority->snapshot()->epoch,1u);
    gate->released=true;pollComponent(wave);
    expectStableTerminalQueries(wave,*original.remote,ComponentTerminalQuery::TransferReady);
    for(size_t index=0;index<wave.prepared.migrations.size();++index)
        for(const auto& destination:wave.prepared.migrations[index].destination_arrivals)
            expectDestination(original,transaction,index,destination);
    const auto stats=original.remote->stats();
    const uint64_t expected_remote_bytes=uint64_t(3)*kH*kF*sizeof(float)/2;
    EXPECT_EQ(stats.bytes_sent,expected_remote_bytes);EXPECT_EQ(stats.bytes_received,expected_remote_bytes);
    EXPECT_EQ(stats.active_mpi_requests,0u);EXPECT_EQ(stats.inference_stream_waits,0u);
    EXPECT_EQ(stats.blocking_synchronizations,0u);
    wave.abortAndDrain();
    expectStableTerminalQueries(wave,*original.remote,ComponentTerminalQuery::AbortDrained);
    EXPECT_EQ(original.authority->snapshot()->epoch,1u);
    for(int id:original.registry->localParticipantIds())EXPECT_FALSE(original.registry->endpoint(id)->acquire(2));
}
/**
 * @test A real source-bound seal preserves fragment provenance for the next graph bank.
 *
 * Both ranks retain their original GGUF, PMA, member and prepared-source roots.
 * Initial placement is already canonical: sealing it requires no invented
 * movement/publication, yet its CPU cohort banks have no whole-expert vector.
 */
TEST(MoEOverlayCPUCohortFabric, RealTwoSocketReusableSealRebindsOriginalFragmentsAndSuccessorBanks) {
    FabricOriginal original;
    auto manager=original.context->concreteWeightManager();ASSERT_TRUE(manager);
    auto& engines=manager->expertGemmRegistry();
    const auto local=original.registry->localParticipantIds();
    const auto& owners=original.authority->snapshot()->placementOwnerMap();
    std::vector<std::pair<int,ExpertGemmRegistry::StaticProjectionLayer>> before;
    for(const int id:local) {
        if(!original.registry->isCPUJointParticipant(id))continue;
        const auto* participant=owners.participantForId(id);ASSERT_TRUE(participant);
        auto layer=engines.acquireCPUJointLayer(original.model,*participant,0);ASSERT_TRUE(layer);
        before.emplace_back(id,std::move(*layer));
    }
    ASSERT_EQ(before.size(),1u); // One actual member on each original physical socket.
    const auto claimed=original.memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::RoutedExpertWeights,
        PhysicalMemoryMaterializationKind::NewAllocation);
    std::string error;
    const auto seal=original.fabric->sealReusableInitialPlacement(1,&error);ASSERT_TRUE(seal)<<error;
    ASSERT_TRUE(seal->valid());
    for(const auto& [id,previous]:before) {
        const auto bank=std::find_if(seal->local_banks.begin(),seal->local_banks.end(),
            [id](const auto& value){return value.participant_id==id;});
        ASSERT_NE(bank,seal->local_banks.end());
        EXPECT_TRUE(bank->layerForModelLayer(0).experts.empty());
        ASSERT_TRUE(bank->layerForModelLayer(0).cpu_fragment_family);
    }
    auto torn=*seal;
    auto& mask=torn.local_banks.front().layers.front().resident_mask;
    ASSERT_FALSE(mask.empty());mask.front()=!mask.front();
    EXPECT_FALSE(engines.replaceParticipantResidencyFromSeal(torn,*original.registry,&error));
    for(const auto& [id,previous]:before) {
        const auto unchanged=engines.acquireCPUJointLayer(original.model,*owners.participantForId(id),0);
        ASSERT_TRUE(unchanged);
        for(const int expert:previous.cpu_joint_source->expertIds())
            EXPECT_EQ(unchanged->experts.at(size_t(expert)).gate,previous.experts.at(size_t(expert)).gate);
    }
    ASSERT_TRUE(engines.replaceParticipantResidencyFromSeal(*seal,*original.registry,&error))<<error;
    for(const auto& [id,previous]:before) {
        const auto* participant=owners.participantForId(id);ASSERT_TRUE(participant);
        const auto current=engines.acquireCPUJointLayer(original.model,*participant,0);ASSERT_TRUE(current);
        ASSERT_EQ(current->sources.size(),previous.sources.size());
        EXPECT_EQ(current->cpu_joint_source,previous.cpu_joint_source);
        EXPECT_FALSE(current->cpu_joint_source.owner_before(previous.cpu_joint_source));
        EXPECT_FALSE(previous.cpu_joint_source.owner_before(current->cpu_joint_source));
        for(size_t index=0;index<previous.sources.size();++index) {
            const auto& a=previous.sources[index];const auto& b=current->sources[index];
            EXPECT_EQ(a.binding_id,b.binding_id);EXPECT_EQ(a.native_source,b.native_source);
            EXPECT_EQ(a.projection_source_owner,b.projection_source_owner);
            EXPECT_EQ(a.projection_ownership,b.projection_ownership);
            EXPECT_EQ(a.tensor,b.tensor);EXPECT_EQ(a.tensor_owner,b.tensor_owner);
            EXPECT_FALSE(a.tensor_owner.owner_before(b.tensor_owner));EXPECT_FALSE(b.tensor_owner.owner_before(a.tensor_owner));
        }
        const auto bank=std::find_if(seal->local_banks.begin(),seal->local_banks.end(),
            [id](const auto& value){return value.participant_id==id;});
        ASSERT_NE(bank,seal->local_banks.end());
        for(const int expert:current->cpu_joint_source->expertIds()) {
            const auto& fragment=bank->layerForModelLayer(0).cpu_fragments.at(size_t(expert));
            EXPECT_EQ(current->experts.at(size_t(expert)).gate,fragment.gate());
            EXPECT_EQ(current->experts.at(size_t(expert)).up,fragment.up());
            EXPECT_EQ(current->experts.at(size_t(expert)).down,fragment.down());
        }
    }
    auto successor=std::make_shared<MoEOverlayParticipantResidencyRegistry>(MoEOverlayParticipantResidencyRegistry::Config{
        .local_participant_ids=local,.num_layers=1,.num_experts=kExperts,.initial_epoch=1,
        .retained_epoch_capacity=3,.projection_preparation=original.preparation,.original_model=original.model});
    ASSERT_TRUE(successor->bindOriginalPreparedContext(original.context,original.preparation,&error))<<error;
    ASSERT_TRUE(successor->finalizeInitialBanksFromPreparedRegistry(engines,&error))<<error;
    for(const auto& [id,previous]:before) {
        const auto frame=successor->initialCPUJointLayer(id,0);ASSERT_TRUE(frame);
        const auto lease=frame->acquireInitialConstruction();ASSERT_TRUE(lease);
        const auto current=engines.acquireCPUJointLayer(original.model,*owners.participantForId(id),0);ASSERT_TRUE(current);
        for(const int expert:current->cpu_joint_source->expertIds()) {
            const auto& fragment=lease->layerForModelLayer(0).cpu_fragments.at(size_t(expert));
            ASSERT_TRUE(fragment.valid());EXPECT_EQ(fragment.model(),original.model);
            EXPECT_EQ(fragment.preparation(),original.preparation);
            EXPECT_EQ(fragment.gate(),current->experts.at(size_t(expert)).gate);
            EXPECT_EQ(fragment.up(),current->experts.at(size_t(expert)).up);
            EXPECT_EQ(fragment.down(),current->experts.at(size_t(expert)).down);
        }
    }
    EXPECT_EQ(original.memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::RoutedExpertWeights,
        PhysicalMemoryMaterializationKind::NewAllocation),claimed);
    EXPECT_EQ(original.authority->snapshot()->epoch,1u);
}

/** @test Failed last-piece staging drains real operations before component source-node reclamation. */
TEST(MoEOverlayCPUCohortFabric, RealTwoSocketFailedFinalPieceDrainsAndRetainsOriginalSourceBank) {
    FabricOriginal original;const auto transaction=original.transaction();ASSERT_TRUE(transaction.valid());
    ComponentWave wave(original,transaction);auto* gate=delayLastPiece(wave,original,transaction);ASSERT_TRUE(gate);
    pollComponent(wave,gate);
    for(int id:original.registry->localParticipantIds())
        EXPECT_FALSE(original.registry->endpoint(id)->retiredEpochReaders(1)); // Still published, so not a retired-node observation.
    auto manager=original.context->concreteWeightManager();manager->clearCache();manager->preparedWeightStore()->clear();manager->expertGemmRegistry().clear();
    // Deliberate registry-lifetime adversary, never global publication or a
    // fabricated inference ticket. All actual source readers were acquired first.
    for(int id:original.registry->localParticipantIds()) {
        auto endpoint=original.registry->endpoint(id);(void)endpoint->retire(1);
        EXPECT_EQ(endpoint->retainedEpochCount(),0u); // Withdrawal is immediate, even with a real old reader.
        EXPECT_FALSE(endpoint->acquire(1)); // Observation must never resurrect source readiness.
        EXPECT_FALSE(endpoint->retiredEpochReaders(0));
        EXPECT_FALSE(endpoint->retiredEpochReaders(2)); // No candidate node was installed.
    }
    if(original.mpi->rank()==1) {
        const auto readers=original.registry->endpoint(original.last_shard)->retiredEpochReaders(1);
        ASSERT_TRUE(readers);EXPECT_EQ(*readers,1u); // The fabric's ONE original member borrow, not an engine ref.
    } else {
        EXPECT_FALSE(original.registry->endpoint(original.whole)->retiredEpochReaders(1));
    }
    gate->failed=true;std::string error;
    EXPECT_EQ(gate->poll(&error),Progress::Failed);EXPECT_NE(error.find("final-piece"),std::string::npos);
    wave.abortAndDrain();
    expectStableTerminalQueries(wave,*original.remote,ComponentTerminalQuery::AbortDrained);
    if(original.mpi->rank()==1) {
        const auto drained=original.registry->endpoint(original.last_shard)->retiredEpochReaders(1);
        ASSERT_TRUE(drained);EXPECT_EQ(*drained,0u); // Actual abort/drain released the original reader before reclamation.
    }
    for(int id:original.registry->localParticipantIds()) {
        auto endpoint=original.registry->endpoint(id);(void)endpoint->retire(1); // Existing maintenance sweep, no diagnostic mutation.
        EXPECT_EQ(endpoint->retainedEpochCount(),0u);EXPECT_FALSE(endpoint->acquire(2));
        EXPECT_FALSE(endpoint->retiredEpochReaders(1)); // The real retired node is now reclaimed.
    }
    EXPECT_EQ(original.remote->stats().active_mpi_requests,0u);
    EXPECT_EQ(original.authority->snapshot()->epoch,1u);
    wave.prepared={}; // Drop actual arrivals, operation callbacks and their engine owners after drain.
    original.fabric.reset();original.registry.reset();original.remote.reset();
    original.context.reset();manager.reset();
    EXPECT_EQ(original.memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::RoutedExpertWeights,
        PhysicalMemoryMaterializationKind::NewAllocation),0u);
    EXPECT_EQ(original.memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::ExpertShadowSlots,
        PhysicalMemoryMaterializationKind::NewAllocation),0u);
}
/** @test A retained whole-wave reservation rejects new setup and aborts without starting any copy. */
TEST(MoEOverlayCPUCohortFabric, RealTwoSocketSetupFailureDrainsWithoutStartingPhysicalWork) {
    FabricOriginal original;const auto transaction=original.transaction();ASSERT_TRUE(transaction.valid());
    ComponentWave reserved(original,transaction);
    // Same admitted slots/lanes cannot serve two overlapping waves. The failed
    // constructor drains its real cleanup prefix; neither wave has been polled.
    EXPECT_THROW((void)ComponentWave(original,transaction),std::runtime_error);
    const auto before=original.remote->stats();
    EXPECT_EQ(before.bytes_sent,0u);EXPECT_EQ(before.bytes_received,0u);
    EXPECT_EQ(before.active_mpi_requests,0u);
    EXPECT_EQ(original.fabric->stats().active_cpu_copy_operations,0u);
    reserved.abortAndDrain();
    expectStableTerminalQueries(reserved,*original.remote,ComponentTerminalQuery::AbortDrained);
    const auto after=original.remote->stats();
    EXPECT_EQ(after.bytes_sent,0u);EXPECT_EQ(after.bytes_received,0u);
    EXPECT_EQ(after.active_mpi_requests,0u);
    for(int id:original.registry->localParticipantIds()) {
        EXPECT_TRUE(original.registry->endpoint(id)->acquire(1));
        EXPECT_FALSE(original.registry->endpoint(id)->acquire(2));
    }
    EXPECT_EQ(original.authority->snapshot()->epoch,1u);
}

} // namespace llaminar2::test

namespace llaminar2::test {
namespace {
/** @brief Weak observation of an actual original bank engine, never a lifetime substitute. */
struct FloatingSourceObservation {
    std::weak_ptr<ITensorGemm> engine;
    DeviceId device;
    int expert;
    MoEExpertProjectionOwnership::Projection projection;
};
/**
 * @brief Observe original sources without retaining a bank or engine across staging.
 * @param original Same genuine source-bound registry and immutable preparation.
 * @param transaction Same authored wave selecting actual source experts.
 * @return Weak engine identities and original coordinates for unchanged-source checks.
 */
std::vector<FloatingSourceObservation> observeFloatingSources(const FabricOriginal& original,
    const MoEOverlayResidencyTransaction& transaction)
{
    std::vector<FloatingSourceObservation> result;
    const std::array<WeightRole,3> roles{WeightRole::MoEExpertGate,WeightRole::MoEExpertUp,WeightRole::MoEExpertDown};
    for(int id:original.registry->localParticipantIds()) {
        auto bank=original.registry->endpoint(id)->acquire(1);
        if(!bank)throw std::runtime_error("Floating Fabric original source bank is absent");
        const auto* participant=original.authority->snapshot()->placementOwnerMap().participantForId(id);
        if(!participant)throw std::runtime_error("Floating Fabric original source participant is absent");
        const auto& layer=bank->layers.front();
        for(int expert=0;expert<kExperts;++expert)if(layer.resident_mask.at(expert) &&
            std::any_of(transaction.migrations.begin(),transaction.migrations.end(),
                [expert](const auto& migration){return migration.expert_id==expert;})) {
            std::array<std::shared_ptr<ITensorGemm>,3> engines;
            if(layer.cpu_fragment_family) {
                const auto& payload=layer.cpu_fragments.at(expert);
                engines={payload.gate(),payload.up(),payload.down()};
            } else {
                const auto& payload=layer.experts.at(expert);
                engines={payload.gate(),payload.up(),payload.down()};
            }
            for(size_t role=0;role<3;++role) {
                const auto& matrix=original.manifests.front().projections[role];
                const auto owned=layer.cpu_fragment_family?
                    layer.cpu_fragment_family->cohort()->ownership(layer.cpu_fragment_family->participantIndex()).projection(roles[role]):
                    MoEExpertProjectionOwnership::Projection{MoEProjectionResidence::ExpertOwner,matrix.N,matrix.K,0,matrix.N,0,matrix.K};
                ContiguousFloatingPointWeightDescriptor actual;
                if(!engines[role] || !engines[role]->exportContiguousFloatingPointWeights(actual) ||
                    !actual.valid() || actual.type!=TensorType::FP32 ||
                    actual.n!=owned.rows || actual.k!=owned.columns)
                    throw std::runtime_error("Floating Fabric requires actual prepared FP32 source geometry/precision");
                result.push_back({engines[role],participant->device,expert,owned});
            }
        }
    }
    return result;
}
/**
 * @brief Check every original logical source byte after destination writes complete.
 * @param sources Weak observations: the actual Fabric bank leases must still own them.
 */
void expectFloatingSourcesUnchanged(const FabricOriginal& original,const std::vector<FloatingSourceObservation>& sources)
{
    for(const auto& source:sources) {
        auto engine=source.engine.lock();ASSERT_TRUE(engine);
        ContiguousFloatingPointWeightDescriptor bytes;
        ASSERT_TRUE(engine->exportContiguousFloatingPointWeights(bytes));ASSERT_TRUE(bytes.valid());
        ASSERT_EQ(bytes.type,TensorType::FP32);
        const auto& owned=source.projection;
        ASSERT_EQ(bytes.n,owned.rows);ASSERT_EQ(bytes.k,owned.columns);
        std::vector<float> downloaded;
        if(source.device.is_gpu()) {
            downloaded.resize(size_t(bytes.n)*bytes.k);
            original.readTerminalWeights(source.device,bytes.data,bytes.bytes,downloaded.data());
        }
        const auto* values=downloaded.empty()?static_cast<const float*>(bytes.data):downloaded.data();
        for(int row=0;row<bytes.n;++row)for(int column=0;column<bytes.k;++column)
            ASSERT_EQ(std::bit_cast<uint32_t>(values[size_t(row)*bytes.k+column]),
                std::bit_cast<uint32_t>(sourceValue(source.expert,owned.first_row+row,owned.first_column+column,
                    owned.source_columns,owned.source_rows)));
    }
}
/**
 * @brief Stage actual floating whole/cohort rectangles in both directions.
 * @param device Exact CUDA/ROCm whole-owner resolved from inventory; both ranks retain
 * their original physical CPU socket/member. No extra GPU-only rank exists.
 *
 * Gate/up split F rows and down splits K columns. Full destination-byte checks
 * detect writes outside each piece when its sibling carries different original
 * columns. Source banks remain borrowed by real operations until abort/drain.
 */
void proveFloatingGPUCohort(DeviceId device)
{
    FabricOriginal original(device);
    ::testing::Test::RecordProperty("physical_sockets",2);::testing::Test::RecordProperty("mpi_ranks",2);
    ::testing::Test::RecordProperty("source_format","FP32");
    const auto transaction=original.transaction();ASSERT_TRUE(transaction.valid());
    ASSERT_EQ(transaction.migrations.size(),2u);
    const auto sources=observeFloatingSources(original,transaction);
    ComponentWave wave(original,transaction);
    auto* gate=delayLastPiece(wave,original,transaction);ASSERT_TRUE(gate);
    pollComponent(wave,gate);
    // A finalized prefix cannot satisfy the complete destination triplet.
    if(original.mpi->rank()==original.whole_owner)
        for(size_t index=0;index<wave.prepared.migrations.size();++index)
            if(transaction.migrations[index].expert_id==2)
                for(const auto& destination:wave.prepared.migrations[index].destination_arrivals) {
                    MoEOverlayPreparedExpertPayload premature;std::string error;
                    EXPECT_FALSE(destination.arrival->completePayload(premature,&error));
                }
    // Withdraw original lookup owners while the last real piece is still held.
    // The Fabric's original bank leases, not preparation/cache pointers, must
    // keep all native source roots alive through this successful completion.
    auto manager=original.context->concreteWeightManager();
    manager->clearCache();manager->preparedWeightStore()->clear();manager->expertGemmRegistry().clear();
    for(int id:original.registry->localParticipantIds()) {
        auto endpoint=original.registry->endpoint(id);(void)endpoint->retire(1);
        EXPECT_FALSE(endpoint->acquire(1));
        const auto readers=endpoint->retiredEpochReaders(1);
        if(id==original.last_shard) {
            ASSERT_TRUE(readers);EXPECT_GT(*readers,0u); // The withheld genuine source piece still owns this reader.
        } else EXPECT_FALSE(readers); // Already-Ready members release their readers immediately.
    }
    gate->released=true;pollComponent(wave);
    expectStableTerminalQueries(wave,*original.remote,ComponentTerminalQuery::TransferReady);
    for(size_t index=0;index<wave.prepared.migrations.size();++index)
        for(const auto& destination:wave.prepared.migrations[index].destination_arrivals)
            expectDestination(original,transaction,index,destination);
    expectFloatingSourcesUnchanged(original,sources);
    for(const auto& migration:wave.prepared.migrations)
        for(const auto& piece:migration.operations) {
            SCOPED_TRACE("rank="+std::to_string(original.mpi->rank())+
                " source="+std::to_string(piece.source_participant)+
                " destination="+std::to_string(piece.destination_participant)+
                " projection="+std::to_string(static_cast<int>(piece.projection)));
            const auto measurement=piece.operation->completedMeasurement();
            const auto& owners=transaction.previous->placementOwnerMap();
            const auto* source=owners.participantForId(piece.source_participant);
            const auto* destination=owners.participantForId(piece.destination_participant);
            ASSERT_TRUE(source && destination && source->world_rank_known && destination->world_rank_known);
            if(source->world_rank!=original.mpi->rank() && destination->world_rank!=original.mpi->rank()) {
                // A rank with no endpoint retains protocol readiness only. It
                // neither transfers this piece nor invents a timing sample.
                EXPECT_FALSE(measurement);
                continue;
            }
            ASSERT_TRUE(measurement);
            EXPECT_TRUE(measurement->valid());
            EXPECT_EQ(measurement->bytes,uint64_t(kH)*kF*sizeof(float)/2);
        }
    const auto stats=original.remote->stats();
    const uint64_t remote_bytes=uint64_t(3)*kH*kF*sizeof(float)/2;
    EXPECT_EQ(stats.bytes_sent,remote_bytes);EXPECT_EQ(stats.bytes_received,remote_bytes);
    EXPECT_EQ(stats.active_mpi_requests,0u);EXPECT_EQ(stats.protocol_failures,0u);
    EXPECT_EQ(stats.inference_stream_waits,0u);EXPECT_EQ(stats.blocking_synchronizations,0u);
    wave.abortAndDrain();
    expectStableTerminalQueries(wave,*original.remote,ComponentTerminalQuery::AbortDrained);
    EXPECT_EQ(original.authority->snapshot()->epoch,1u);
    for(int id:original.registry->localParticipantIds()) {
        auto endpoint=original.registry->endpoint(id);
        EXPECT_FALSE(endpoint->acquire(2));
        const auto readers=endpoint->retiredEpochReaders(1);
        if(readers)EXPECT_EQ(*readers,0u);
        (void)endpoint->retire(1);
        EXPECT_FALSE(endpoint->retiredEpochReaders(1));
    }
    wave.prepared={};
    original.fabric.reset();original.registry.reset();original.remote.reset();
    original.context.reset();manager.reset();
    for(const auto& source:sources)EXPECT_TRUE(source.engine.expired());
    for(const auto allocator:{DeviceId::cpu(),device}) {
        if(allocator.is_gpu() && original.mpi->rank()!=original.whole_owner)continue; // Remote GPU is not a local PMA resource.
        EXPECT_EQ(original.memory->claimedBytes(allocator,PhysicalMemoryOwner::RoutedExpertWeights,
            PhysicalMemoryMaterializationKind::NewAllocation),0u);
        EXPECT_EQ(original.memory->claimedBytes(allocator,PhysicalMemoryOwner::ExpertShadowSlots,
            PhysicalMemoryMaterializationKind::NewAllocation),0u);
    }
}
/** @brief Unexpected native setup failure must wake its peer before owned resources unwind. */
void runFloatingGPUCohort(DeviceId device)
{
    try { proveFloatingGPUCohort(device); }
    catch(const std::exception& error) {
        ADD_FAILURE()<<error.what();MPI_Abort(MPI_COMM_WORLD,2);throw;
    }
}
} // namespace
#if defined(HAVE_CUDA)
TEST(MoEOverlayGPUCPUCohortFabric, CUDAFloatingOriginalRectangles)
{
    runFloatingGPUCohort(DeviceId::cuda(0));
}
#endif
#if defined(HAVE_ROCM)
TEST(MoEOverlayGPUCPUCohortFabric, ROCmFloatingOriginalRectangles)
{
    runFloatingGPUCohort(DeviceId::rocm(0));
}
#endif

} // namespace llaminar2::test

#include "CPUJointFabricMeasurementContract.inc"

#include "CPUJointShadowCalibrationContract.inc"

#include "CPUJointCalibrationControllerContract.inc"
