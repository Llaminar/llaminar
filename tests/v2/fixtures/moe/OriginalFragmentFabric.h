/**
 * @file OriginalFragmentFabric.h
 * @brief Shared test-only original GGUF physical-fabric owner and real transfer progress.
 *
 * Preserves the existing two-rank inventory, source preparation, sole PMA BOM,
 * physical transfer and abort/drain authorities. Optional setup callbacks add
 * actual source payloads and canonical BOM inputs before admission; they do not
 * fabricate descriptors, arrivals or routing publication.
 */
#pragma once
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
#include "../collective/CPUExpertDomainSocketMembership.h"
#include "MoEOverlayJointPlacementCandidateTestAccess.h"
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

#include "execution/moe/MoEOverlayPhysicalProjectionPieces.h"
#include "execution/moe/MoEOverlayPreparedWeightSource.h"
#include "execution/moe/ExpertPreparedMemoryGeometry.h"
#include "kernels/cpu/gemm/CPUNativeQuantisedWeightPacker.h"
#include "kernels/cpu/gemm/CPUWeightPreparationGeometry.h"
#include "execution/factory/GraphWeightBindingPreparation.h"
#include <functional>
#include <set>
#include <map>
#include <span>
#include <optional>
#include "../../utils/ObservedNativeReplayOwner.h"

namespace llaminar2::test::original_fragment_fixture {
using Progress=MoEOverlayResidencyWaveProgress;
inline constexpr size_t kOriginalFragmentStaging=4093;
/** @return Same-host distinct physical socket observations in actual world-rank order. */
inline std::vector<CPUExpertSocketEndpoint> originalFabricSockets(const MPIContext& mpi) {
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
/** @return Independent originally authored float at exact original expert/N/K coordinates. */
inline float sourceValue(int expert,int row,int column,int columns,int rows) {
    const size_t index=(size_t(expert)*rows+size_t(row))*columns+size_t(column);
    return 0.001f*float(1+index%29)/29.0f;
}
/** @brief Authored source dimensions; prepared fragments derive every cut from these values. */
struct OriginalFragmentSourceGeometry final {
    int hidden=256; ///< Original embedding width, never a prepared descriptor override.
    int routed=640; ///< Original full expert intermediate width before cohort slicing.
    int shared=256; ///< Original shared-expert source width.
    int experts=6; ///< Source expert population, including every declared owner.
    int top_k=2; ///< Source routing extent.
};
/** @brief Optional original-source preparation extension; callbacks never publish an epoch. */
struct OriginalFragmentPreparationHooks {
    std::optional<OriginalFragmentSourceGeometry> geometry; ///< Absent preserves the original mixed/floating fixture dimensions.
    std::function<void(const MoERoutedExpertPlacementPlan&,int)> authored_plan;
    GGUFTensorType shared_type=GGUFTensorType::F32; ///< Exact optional shared-source directory format; old owner remains FP32.
    /// Author additional genuine GGUF payloads before the ModelContext opens the file.
    std::function<void(PlanningGGUFFixture&)> source;
    /// Add exact canonical preparation/workspace BOM inputs before the sole PMA admission.
    std::function<void(ModelContext&,PhysicalMemoryPlanBuilder&,const std::vector<PhysicalMemoryResource>&,
        const std::vector<DeviceId>&)> before_admission;
    /// Prepare additional original bindings under the now-installed sole PMA authority.
    std::function<void(ModelContext&,const std::vector<DeviceId>&)> prepared;
};
/** @brief Authored target request; the inventory supplies original devices and rank ownership. */
struct OriginalFragmentTargetRequest {
    std::string name; DeviceType backend; size_t degree;
    std::string same_rank_as; ///< Optional original domain whose rank must own this distinct cohort.
    std::string pci_address; ///< Optional exact single-card test cohort, preserving original rank and inventory identity.
    std::vector<std::string_view> pci_addresses; ///< Optional exact cohort order, all from one actual NUMA owner.
};
/** @brief Actual original all-cohort source and existing physical fabric, with no serving publication. */
struct OriginalFragmentFabric final {
    const OriginalFragmentSourceGeometry geometry; ///< Immutable original file/placement authority.
    const int experts,hidden,width; ///< Source dimensions, shared by all existing preparation owners.
    std::shared_ptr<MPIContext> mpi;
    std::unique_ptr<PlanningGGUFFixture> file;
    std::shared_ptr<TensorFactory> tensors;
    std::shared_ptr<NativeContextPreparation> native_source;
    std::vector<PreparedGPUContextUsePermit> native_uses;
    std::shared_ptr<ModelContext> context;
    std::shared_ptr<const MoERoutedExpertPlacementPlan> plan;
    std::shared_ptr<const MoEExpertOverlayPreparationPlan> preparation;
    std::shared_ptr<MoEOverlayResidencyAuthority> authority;
    std::shared_ptr<MoEOverlayParticipantResidencyRegistry> registry;
    std::shared_ptr<PhysicalMemoryAuthority> memory;
    std::shared_ptr<MoEOverlayMPIRemoteProjectionTransport> remote;
    std::shared_ptr<MoEOverlayPhysicalResidencyFabric> fabric;
    std::vector<MoEOverlayLayerWeightManifest> manifests;
    std::vector<DeviceId> local_gpus;
    ModelContextId model;
    /** @brief Build the requested original cohorts, never renumbering inventory endpoints. */
    OriginalFragmentFabric(std::span<const OriginalFragmentTargetRequest> requests,bool mixed,const OriginalFragmentPreparationHooks& hooks={})
        :geometry(hooks.geometry.value_or(OriginalFragmentSourceGeometry{.routed=mixed?640:1024})),
         experts(geometry.experts),hidden(geometry.hidden),width(geometry.routed),mpi(MPIContextFactory::global()) {
        if(experts<6||hidden<=0||hidden%32||width<=0||width%32||geometry.shared<=0||geometry.shared%32||
           geometry.top_k<=0||geometry.top_k>experts)
            throw std::invalid_argument("Original fragment geometry requires aligned source dimensions and all six rotation witnesses");
        if(!mpi)throw std::invalid_argument("Fragment staging requires its original MPI context");
        const bool has_cpu_cohort=std::any_of(requests.begin(),requests.end(),
            [](const auto& request){return request.backend==DeviceType::CPU;});
        const auto sockets=has_cpu_cohort?originalFabricSockets(*mpi):std::vector<CPUExpertSocketEndpoint>{};
        const auto inventory=mpi->clusterInventory();
        if(!inventory || inventory->world_size!=mpi->world_size())
            throw std::invalid_argument("Fragment staging requires complete original rank inventory");
        const auto rank=mpi->rank();const auto& observed=inventory->getRank(rank);
        if(!hasCPUBackend()) {
            const auto affinity=NUMATopology::detectLocalNUMANode();
            if(!affinity.detection_succeeded||affinity.local_numa_node<0)
                throw std::runtime_error("Original physical fabric requires exact rank-local NUMA affinity");
            initCPUBackend(affinity.local_numa_node);
        }
        std::set<std::pair<int,DeviceId>> assigned_gpus;
        MoERoutedExpertPlacementPlan authored;
        authored.enabled=true;authored.first_model_layer=0;
        authored.topology=RoutedExpertPlacementTopology::TieredOverlay;
        for(const auto& request:requests) {
            if(request.name.empty() || !request.degree)throw std::invalid_argument("Invalid authored cohort request");
            RoutedExpertDomain domain;domain.name=request.name;
            domain.routed_compute_policy=RoutedExpertComputePolicy::TensorSharded;
            domain.routed_phase_policy=RoutedExpertPhasePolicy::Uniform;
            domain.routed_decode_assignment_policy=domain.routed_prefill_assignment_policy=RoutedExpertAssignmentPolicy::StaticOwner;
            if(request.backend==DeviceType::CPU) {
                if(request.degree!=sockets.size())throw std::invalid_argument("CPU target must retain every original socket");
                domain.scope=ExecutionDomainScope::NODE_LOCAL;domain.owner_rank=-1;
                domain.backend=CollectiveBackendType::UPI;
                for(const auto& endpoint:sockets) {
                    domain.participants.push_back(GlobalDeviceAddress::cpu(endpoint.numa_node));
                    domain.world_ranks.push_back(endpoint.rank);
                }
            } else {
                domain.scope=ExecutionDomainScope::RANK_LOCAL;
                domain.backend=request.backend==DeviceType::CUDA?CollectiveBackendType::NCCL:CollectiveBackendType::RCCL;
                // Co-location is an authored topology requirement. Inventory
                // remains the owner; no test invents ranks or reuses a card in
                // two supposedly distinct source cohorts.
                int required_rank=-1;
                if(!request.same_rank_as.empty()) {
                    const auto previous=std::find_if(authored.domains.begin(),authored.domains.end(),
                        [&](const auto& item){return item.name==request.same_rank_as;});
                    if(previous==authored.domains.end() || previous->owner_rank<0)
                        throw std::invalid_argument("Co-located cohort requires an earlier original GPU domain");
                    required_rank=previous->owner_rank;
                }
                // Multiple real ranks may each own a qualifying cohort. Use
                // the canonical first-complete observed owner, rather than
                // treating legitimate hardware on a second socket as ambiguity.
                // Reserve capacity for explicitly co-located later domains.
                size_t required_devices=request.degree;
                for(const auto& later:requests)
                    if(later.same_rank_as==request.name && later.backend==request.backend)
                        required_devices+=later.degree;
                if(!request.pci_address.empty()&&request.degree!=1)
                    throw std::invalid_argument("Exact single-card original fragment request requires degree1");
                const std::array<std::string_view,1> pci{request.pci_address};
                if(!request.pci_addresses.empty()&&(!request.pci_address.empty()||request.pci_addresses.size()!=request.degree))
                    throw std::invalid_argument("Exact original cohort roster must cover every member once");
                const std::span<const std::string_view> requested_pci=!request.pci_addresses.empty()
                    ?std::span<const std::string_view>{request.pci_addresses}
                    :request.pci_address.empty()?std::span<const std::string_view>{}:std::span<const std::string_view>{pci};
                const std::array family{NativeReplayFamily{request.backend,request.degree,{},requested_pci}};
                const auto chosen=selectObservedNativeReplayOwner(*inventory,family,[&](const RankInventory& owner) {
                    if(required_rank>=0 && owner.rank!=required_rank)return false;
                    size_t available=0;
                    for(const auto& gpu:owner.gpus)
                        if(gpu.type==request.backend && rankOwnsGPUDevice(owner,gpu) &&
                            !assigned_gpus.contains({owner.rank,DeviceId{gpu.type,gpu.local_device_id}}))++available;
                    return available>=required_devices;
                });
                if(!chosen)throw std::invalid_argument("Authored GPU cohort unavailable on original NUMA-local rank");
                for(const auto& owner:inventory->ranks) {
                    if(owner.rank!=chosen->discovery_owner)continue;
                    std::vector<DeviceId> selected;
                    if(!requested_pci.empty()) {
                        selected=chosen->devices;
                        for(const auto d:selected) {
                            if(assigned_gpus.contains({owner.rank,d}))
                                throw std::invalid_argument("Original requested PCI cohort overlaps an earlier target");
                            const auto gpu=std::find_if(owner.gpus.begin(),owner.gpus.end(),[&](const auto& g){
                                return g.type==d.type&&g.local_device_id==d.ordinal;});
                            if(gpu==owner.gpus.end()||gpu->uuid.empty()||!rankOwnsGPUDevice(owner,*gpu))
                                throw std::invalid_argument("Original requested PCI cohort lacks its native owner/UUID");
                        }
                    }
                    for(const auto& gpu:owner.gpus)
                        if(requested_pci.empty()&&gpu.type==request.backend && rankOwnsGPUDevice(owner,gpu) &&
                            !assigned_gpus.contains({owner.rank,DeviceId{gpu.type,gpu.local_device_id}})) {
                            if(gpu.uuid.empty())throw std::invalid_argument("Original GPU has no UUID");
                            selected.push_back({gpu.type,gpu.local_device_id});
                        }
                    if(selected.size()<request.degree)continue;
                    if(!domain.participants.empty())throw std::invalid_argument("GPU target has ambiguous original rank ownership");
                    domain.owner_rank=owner.rank;
                    for(size_t member=0;member<request.degree;++member) {
                        domain.participants.push_back(GlobalDeviceAddress::fromLocalDeviceId(selected[member]));
                        assigned_gpus.emplace(owner.rank,selected[member]);
                        if(owner.rank==rank) {
                            requireRankLocalGPUOwnership(observed,selected[member],"original fragment staging source");
                            local_gpus.push_back(selected[member]);
                        }
                    }
                }
                if(domain.participants.size()!=request.degree)throw std::invalid_argument("Authored GPU cohort unavailable on original rank");
            }
            authored.domains.push_back(std::move(domain));
            const int tier=int(authored.routed_tiers.size());
            authored.routed_tiers.push_back({.name=request.name,.domain=request.name,.priority=tier,
                .max_experts_per_layer=(experts+int(requests.size())-1)/int(requests.size()),.fallback=tier+1==int(requests.size())});
        }
        if(requests.size()!=3)throw std::invalid_argument("Closed staging rotation requires three authored targets");
        authored.continuation_domain=authored.base_model_domain=authored.shared_expert_domain=requests.front().name;
        // Retain the existing six closed-rotation witnesses. Remaining real
        // source experts have declared owners too; an uneven population is not
        // represented by absent descriptors or an invented all-local table.
        std::vector<int> placement{0,0,1,1,2,2};
        for(int expert=6;expert<experts;++expert)placement.push_back((expert-6)%int(requests.size()));
        authored.placements={{.layer=0,.routed_expert_tier=std::move(placement)}};
        authored.residency_policy=RoutedExpertResidencyPolicy::RoutedTierRebalanced;
        authored.authority_execution=resolveMoEOverlayAuthorityExecutionKind(authored);
        plan=std::make_shared<const MoERoutedExpertPlacementPlan>(std::move(authored));
        if(hooks.authored_plan)hooks.authored_plan(*plan,rank);
        const std::array role_types=mixed?std::array{GGUFTensorType::Q4_K,GGUFTensorType::Q4_K,GGUFTensorType::Q5_1}:
            std::array{GGUFTensorType::F32,GGUFTensorType::F32,GGUFTensorType::F32};
        file=std::make_unique<PlanningGGUFFixture>(true,false,role_types[0],width,std::nullopt,8,
            PlanningGGUFFixture::Geometry::Directory,1,experts,hooks.shared_type,role_types,
            PlanningGGUFFixture::MoEDimensions{unsigned(hidden),unsigned(geometry.shared),unsigned(geometry.top_k)});
        // These are genuine separately encoded GGUF projection parents. In particular
        // down Q5_1 has native32 K groups; no Q4_K source is reinterpreted at F640.
        const std::array names{"blk.0.ffn_gate_exps.weight","blk.0.ffn_up_exps.weight","blk.0.ffn_down_exps.weight"};
        for(size_t role=0;role<names.size();++role) {
            const size_t rows=role==2?hidden:width,columns=role==2?width:hidden;
            // Author large original fixtures in bounded expert chunks. The
            // conservative FP32 extent also bounds each quantizer's temporary
            // input; ordinary six-expert fixtures keep their original seed and
            // complete payload exactly. Runtime loading still uses its existing
            // source preparation and sole PMA, with no test-owned replacement.
            constexpr size_t source_chunk_bytes=512u*1024u*1024u;
            const size_t experts_per_chunk=std::max(size_t{1},source_chunk_bytes/(rows*columns*sizeof(float)));
            size_t source_offset=0;
            for(size_t first=0;first<size_t(experts);first+=experts_per_chunk) {
                const size_t count=std::min(experts_per_chunk,size_t(experts)-first);
                const auto seed=91+role+first*17;
                std::unique_ptr<TensorBase> source;
                if(mixed && role==2)source=TestTensorFactory::createQ5_1Random({count*rows,columns},seed);
                else if(mixed) {
                    const auto generated=TestTensorFactory::createQ4_KRandom({count*rows,columns},seed);
                    std::vector<uint8_t> native(generated->size_bytes());
                    std::memcpy(native.data(),generated->raw_data(),native.size());
                    // Include nonzero minima and both zero signs, retaining
                    // the original global block-coordinate pattern.
                    constexpr std::array<uint16_t,4> minima{0x0000u,0x8000u,0x2400u,0x2800u};
                    for(size_t index=0;index<native.size()/sizeof(Q4_KBlock);++index) {
                        Q4_KBlock block;
                        std::memcpy(&block,native.data()+index*sizeof(block),sizeof(block));
                        block.dmin=minima[(source_offset/sizeof(Q4_KBlock)+index+role)%minima.size()];
                        std::memcpy(native.data()+index*sizeof(block),&block,sizeof(block));
                    }
                    source=std::make_unique<Q4_KTensor>(std::vector<size_t>{count*rows,columns},native);
                } else {
                    auto values=TestTensorFactory::createFP32Zeros({count*rows,columns});
                    for(size_t expert=0;expert<count;++expert)for(size_t n=0;n<rows;++n)for(size_t k=0;k<columns;++k)
                        values->mutable_data()[(expert*rows+n)*columns+k]=sourceValue(int(first+expert),n,k,columns,rows);
                    source=std::move(values);
                }
                file->writePayload(names[role],source_offset,
                    {static_cast<const uint8_t*>(source->raw_data()),source->size_bytes()});
                source_offset+=source->size_bytes();
            }
        }
        if(hooks.source)hooks.source(*file);
        tensors=std::make_shared<TensorFactory>(*mpi);
        ModelContextConfig source;source.mpi_ctx=mpi;source.factory=tensors.get();source.use_mmap=false;
        context=ModelContext::create(file->path(),source);
        if(!context)throw std::runtime_error("Original fragment model failed to load");
        model=ModelContextId{reinterpret_cast<uint64_t>(context.get())};
        auto& loader=context->concreteLoader();
        const auto runtime=resolveMoEExpertOverlayRuntimePlan(plan,{.current_world_rank=rank,.validate_mvp_root_reachability=false});
        preparation=std::make_shared<const MoEExpertOverlayPreparationPlan>(MoEExpertOverlayPreparationPlan::build(*runtime,loader));
        authority=std::make_shared<MoEOverlayResidencyAuthority>(MoEOverlayResidencyAuthority::Config{
            .initial_plan=*plan,.model_metadata={.num_layers=1,.num_experts=experts,.d_model=hidden,
                .routed_intermediate_size=width,.main_inference_layer_count=1},.projection_preparation=preparation,
            .maintenance_mode=MoERebalanceRuntimeMode::Off,.participant_rebalance_policy={.enabled=false}});
        manifests=buildMoEOverlayLayerWeightManifestFromLoader(loader,1,experts);
        const auto& topology=authority->snapshot()->placementOwnerMap();
        std::vector<int> local;
        std::map<DeviceId,size_t> live,shadow;
        for(const auto& participant:topology.participants())if(participant.world_rank_known && participant.world_rank==rank) {
            local.push_back(participant.participant_id);
            const auto selected=preparation->shardSourceFor(0,participant);
            if(!selected)throw std::logic_error("Original TensorSharded member lacks its source family");
            const auto manifest=selected->cohort()->localManifest(selected->participantIndex());
            const auto footprint=MoEOverlayCapacityResolver::preparedFootprints({manifest}).front();
            shadow[participant.device]+=participant.device.is_gpu()?footprint.gpu_shadow_bytes:footprint.cpu_shadow_bytes;
            if(participant.device.is_cpu())live[participant.device]+=selected->expertIds().size()*footprint.cpu_live_bytes;
            else for(const auto& projection:manifest.projections)
                live[participant.device]+=selected->expertIds().size()*resolveExpertPreparedProjectionMemoryGeometry(
                    projection.N,projection.K,projection.format,GPUPreparedWeightPoolLayout::MigrationReusable).gpu_live_bytes;
        }
        std::sort(local.begin(),local.end());
        const MoEOverlayCapacityAdmissionPolicy storage{.migration_storage=MoEOverlayMigrationStorageKind::PhysicalResidencyFabric,
            .shadow_slots_per_endpoint_layer=1,.staging_capacity_bytes=kOriginalFragmentStaging,.maximum_concurrent_cycles=1,
            .maximum_execution_streams=1,.maximum_cycles_per_layer=1,.distributed_transport=mpi->world_size()>1,.overlay_world_size=mpi->world_size()};
        auto demand=std::make_shared<MoEOverlayPhysicalLaneDemandPlan>(configuredMoEOverlayLaneDemand(*plan,manifests,experts,
            MoEOverlayPhysicalLaneScope::communicator(mpi->world_size()),1,1));
        demand->original_execution_streams=MoEOverlayCapacityAdmission::executionStreamCounts(*plan,rank,1);
        bindOriginalMoEOverlayGpuChoices(*demand,rank,[&](DeviceId accessor,DeviceId allocation) {
            return originalMoEOverlayDirectedPeerAccess(observed,accessor,allocation);
        });
        const auto capacity=observeSystemMemoryCapacity();
        const PhysicalMemoryResource cpu{rank,DeviceId::cpu(),capacity.total_bytes,capacity.admission_available_bytes};
        std::vector<PhysicalMemoryResource> observations{cpu};
        for(const auto device:local_gpus) {
            auto* backend=getBackendFor(device);if(!backend)throw std::runtime_error("Original native backend unavailable");
            observations.push_back({rank,device,backend->deviceMemoryTotal(device.ordinal),backend->deviceMemoryFree(device.ordinal)});
        }
        if(!local_gpus.empty()) {
            native_source=NativeContextPreparation::create(observations,rank,local_gpus);
            for(const auto device:local_gpus)native_uses.push_back(native_source->acquire(device));
            context->bindNativeContextPreparation(native_source);native_source->freezeForPlanning();
        }
        PhysicalMemoryPlanBuilder budget;
        if(native_source)for(const auto& resource:native_source->mandatoryBOM().resources())budget.add(resource);
        for(const auto& resource:observations) {
            budget.add(resource,PhysicalMemoryOwner::RoutedExpertWeights,live[resource.device]);
            budget.add(resource,PhysicalMemoryOwner::ExpertShadowSlots,shadow[resource.device]);
        }
        // CPU native packing has its own bounded source/scratch root even on
        // a rank with no GPUs. Quote the canonical writer's geometry rather
        // than relying on otherwise unused GPU upload staging to cover it.
        size_t cpu_source_staging=0;
        const auto cpu_source_plan=preparation->filteredForDevice(DeviceId::cpu()).sourceWeightPlan(loader,model);
        for(const auto& request:cpu_source_plan.requirements()) {
            const auto geometry=loader.nativeWeightSourceGeometry(
                request.source_name.empty()?request.canonical_name:request.source_name);
            for(const auto& projection:projectNativeWeightSource(request,geometry)) {
                const auto quote=quoteCPUWeightPreparation(geometry.original_type,
                    projection.finalRows(),projection.finalColumns(),WeightPrecision::NATIVE,CPUWeightConsumer::PreparedGemm);
                if(quote)cpu_source_staging=std::max(cpu_source_staging,
                    resolveNativeWeightLoadStaging(*quote,loader.nativeWeightSourcePlacement(),
                        configuredNativeWeightLoadStagingBudgetBytes()).physical_bytes);
            }
        }
        budget.add(cpu,PhysicalMemoryOwner::WeightLoadStaging,cpu_source_staging);
        for(const auto& charge:MoEOverlayCapacityAdmission::transferStagingBOM(*plan,mpi->world_size(),storage,manifests,experts))
            if(charge.world_rank==rank) {
                const auto allocator=std::find_if(observations.begin(),observations.end(),[&](const auto& item){return item.device==charge.device;});
                if(allocator==observations.end())throw std::logic_error("Staging BOM names foreign local allocator");
                budget.add(*allocator,PhysicalMemoryOwner::ExpertMigrationStaging,charge.bytes);
            }
        for(const auto device:local_gpus) {
            const auto gpu=*std::find_if(observations.begin(),observations.end(),[&](const auto& item){return item.device==device;});
            size_t maximum_source=0;
            // requirements() borrows its plan; retain the plan through source
            // geometry lookup instead of iterating a destroyed temporary.
            const auto source_plan=preparation->filteredForDevice(device).sourceWeightPlan(loader,model);
            for(const auto& request:source_plan.requirements()) {
                size_t bytes=0;
                for(const auto& projection:projectNativeWeightSource(request,loader.nativeWeightSourceGeometry(
                    request.source_name.empty()?request.canonical_name:request.source_name)))bytes+=projection.range.nativeBytes()*projection.matrix_count;
                maximum_source=std::max(maximum_source,bytes);
            }
            const auto upload=resolveGPUWeightLoadMemoryGeometry(maximum_source,configuredGPUWeightLoadMemoryPolicy());
            budget.add(gpu,PhysicalMemoryOwner::WeightLoadStaging,upload.staging_bytes);
            budget.add(cpu,PhysicalMemoryOwner::WeightLoadStaging,upload.host_staging_bytes);
            const auto stream_count=demand->original_execution_streams.at(device);
            const auto lanes=TransferEngine::persistentTransferExecutionLanesBOM(stream_count,cpu,gpu);
            for(const auto& row:lanes.resources())budget.add(row);
            const auto events=originalMoEOverlayPeerEventCounts(*demand);
            const auto event=events.find(device);
            if(event!=events.end())for(size_t i=0;i<event->second;++i) {
                const auto native_events=ExpertTierGpuPeerTransferLane::nativeEventsBOM(cpu,gpu,false);
                for(const auto& row:native_events.resources())budget.add(row);
            }
        }
        if(hooks.before_admission)hooks.before_admission(*context,budget,observations,local_gpus);
        const auto certificate=std::make_shared<const PhysicalMemoryPlanAdmissionCertificate>(budget.build());
        if(native_source) {
            native_source->admitFinal(certificate);memory=native_source->memory();
        } else {
            // A genuine CPU-only rank owns no native runtime population. It
            // admits its same typed source/shadow/transport BOM directly.
            memory=std::make_shared<PhysicalMemoryAuthority>(certificate,rank);
        }
        auto manager=context->concreteWeightManager();manager->installPhysicalMemoryAuthority(memory);
        for(const auto device:[&]{auto all=local_gpus;all.push_back(DeviceId::cpu());return all;}()) {
            const auto weights=projectGraphExpertSourceBindings(
                manager->materialize(preparation->filteredForDevice(device).sourceWeightPlan(loader,model)));
            if(!manager->prepareMoEExpertOverlayWeights(*runtime,device,&weights,nullptr,
                PreparedWeightAdmission::AllocateCompleteSet,preparation))throw std::runtime_error("Original fragment source preparation failed");
        }
        if(hooks.prepared)hooks.prepared(*context,local_gpus);
        registry=std::make_shared<MoEOverlayParticipantResidencyRegistry>(MoEOverlayParticipantResidencyRegistry::Config{
            .local_participant_ids=local,.num_layers=1,.num_experts=experts,.initial_epoch=1,.retained_epoch_capacity=3,
            .projection_preparation=preparation,.original_model=model});
        std::string error;
        if(!registry->bindOriginalPreparedContext(context,preparation,&error) ||
           !registry->finalizeInitialBanksFromPreparedRegistry(manager->expertGemmRegistry(),&error))throw std::runtime_error(error);
        if(mpi->world_size()>1)remote=std::make_shared<MoEOverlayMPIRemoteProjectionTransport>(MoEOverlayMPIRemoteProjectionTransport::Config{
            .mpi_context=mpi,.lane_budget=remoteMoEOverlayProjectionLaneBudget(*demand),.staging_capacity_bytes=kOriginalFragmentStaging,
            .perf_device="original-gpu-fragment-staging"});
        fabric=MoEOverlayPhysicalResidencyFabric::create({.memory_authority=memory,.registry=registry,
            .initial_snapshot=authority->snapshot(),.physical_gpu_lane_demand=demand,.layer_weight_manifest=manifests,
            .remote_projection_transport=remote,.shadow_slots_per_endpoint_layer=1,.staging_capacity_bytes=kOriginalFragmentStaging,
            .maximum_concurrent_cycles=1,.maximum_execution_streams=1,.maximum_cycles_per_layer=1,
            .perf_device="original-gpu-fragment-staging"});
    }
    /** @brief Destroy actual physical and source borrowers before original worker retirement. */
    ~OriginalFragmentFabric() noexcept {
        try {
            fabric.reset();registry.reset();remote.reset();authority.reset();preparation.reset();context.reset();
            const auto expected=native_uses.size();native_uses.clear();
            if(native_source && native_source->retireReleasedContexts()!=expected)std::terminate();
        }catch(...){std::terminate();}
    }
    /** @return Same authority's closed three-target rotation, without publishing epoch2. */
    MoEOverlayResidencyTransaction transaction(bool reverse) const {
        auto placements=plan->placements;
        for(const int expert:{0,2,4})placements.front().routed_expert_tier[expert]=
            (placements.front().routed_expert_tier[expert]+(reverse?2:1))%int(plan->routed_tiers.size());
        return MoEOverlayJointPlacementCandidateTestAccess::prepareTransaction(*authority,std::move(placements));
    }
    /** @brief Observe only terminal native arrays through their exact original admitted worker. */
    void read(DeviceId device,const void* source,size_t bytes,void* destination) const {
        const auto use=std::find_if(native_uses.begin(),native_uses.end(),[&](const auto& item){return item.device()==device;});
        if(use==native_uses.end())throw std::invalid_argument("Terminal fragment read has no original worker");
        const auto worker=use->worker();worker->submitAndWait([&] {
            auto* backend=getBackendFor(device);const auto stream=worker->defaultStream();
            if(reinterpret_cast<uintptr_t>(stream)<=2 || !backend->deviceToHost(destination,source,bytes,device.ordinal,stream) ||
                !backend->synchronizeStream(stream,device.ordinal))throw std::runtime_error("Terminal fragment byte observation failed");
        });
    }
};
/** @brief Own every real operation until normal completion or actual abort/drain. */
class ComponentWave {
public:
    MoEOverlayParticipantPreparedTransfers prepared;
    /** @brief Reserve full physical work, preserving the unchanged global authority. */
    template<class Original>
    ComponentWave(Original& original,const MoEOverlayResidencyTransaction& transaction)
        :prepared(original.fabric->prepareTransfers(transaction,original.registry->localParticipantIds())),
         fabric_(original.fabric) {
        if(prepared.status!=MoEOverlayResidencyStageStartStatus::Started) {
            const auto error=prepared.error;
            abortAndDrain(); // Failed prefixes are inert today, but still consume the real cleanup interface.
            throw std::runtime_error(error);
        }
    }
    /** @brief Drain before any assertion unwinds slots or MPI owners. */
    ~ComponentWave() {if(!drained_) {try{abortAndDrain();}catch(...){std::terminate();}}}
    /** @brief Submit the real retained relay epochs without manufacturing completion or waiting on a GPU. */
    void progressTransfers() {
        const auto fabric=fabric_.lock();std::string error;
        if(!fabric || !fabric->submitOutstandingTransferProgress(&error))
            throw std::runtime_error(error.empty()?"Physical staging lost its original progress owner":error);
    }
    /** @brief Delegate abort/drain to original operations, never manufacture readiness. */
    void abortAndDrain() {
        for(auto& migration:prepared.migrations)for(auto& piece:migration.operations)if(piece.operation)piece.operation->abort();
        const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);
        for(;;) {
            progressTransfers();
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
private:
    std::weak_ptr<MoEOverlayPhysicalResidencyFabric> fabric_; ///< Fixture does not prolong a completed fabric's storage lifetime.
    bool drained_=false;
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
    /** @return The retained physical operation's exact admitted live payload extent. */
    std::optional<std::uint64_t> quotedPayloadBytes() const noexcept override{return operation_->quotedPayloadBytes();}
private:std::unique_ptr<IMoEOverlayTierTransferOperation> operation_;
};
/** @brief Poll actual pieces with symmetric test-only consensus, excluding the named gate. */
inline void pollComponent(ComponentWave& wave,const LastPieceGate* withheld=nullptr) {
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);
    for(;;) {
        int all_local=1;std::string error;
        for(auto& migration:wave.prepared.migrations)for(auto& piece:migration.operations) {
            if(piece.operation.get()==withheld)continue;
            const auto progress=piece.operation->poll(&error);
            if(progress==Progress::Failed)throw std::runtime_error(error);
            if(progress!=Progress::Ready)all_local=0;
        }
        wave.progressTransfers();
        int all_ranks=0;
        if(MPI_Allreduce(&all_local,&all_ranks,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD)!=MPI_SUCCESS)
            throw std::runtime_error("CPU cohort fabric component progress vote failed");
        if(all_ranks)return;
        if(std::chrono::steady_clock::now()>=deadline)throw std::runtime_error("CPU cohort fabric staging exceeded standard deadline");
        std::this_thread::yield();
    }
}

} // namespace llaminar2::test::original_fragment_fixture
