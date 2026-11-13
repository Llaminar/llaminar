/**
 * @file PhysicalProjectionLaneDemandContract.inc
 * @brief Device-free original-cut fanout, retained-port and cycle-bound proofs.
 */
#include "execution/moe/MoEOverlayPhysicalLaneDemand.h"
namespace llaminar2 { namespace {
/** @return Declared native source, not a loaded tensor or fabricated prepared engine. */
MoEOverlayLayerWeightManifest pieceDemandSource(const NativeQuantisedFormatInfo& format=native_quantised_formats::Q4_0) {
    return {.layer_idx=0,.projections={{projection(ExpertTierWeightProjection::Gate,8192,256,format),
        projection(ExpertTierWeightProjection::Up,8192,256,format),projection(ExpertTierWeightProjection::Down,256,8192,format)}}};
}
/** @return One original whole GPU target and one declared 2..8 CPU cohort. */
MoERoutedExpertPlacementPlan pieceDemandPlan(int cpu_members) {
    MoERoutedExpertPlacementPlan plan;plan.enabled=true;plan.topology=RoutedExpertPlacementTopology::TieredOverlay;
    plan.continuation_domain=plan.base_model_domain=plan.shared_expert_domain="gpu";
    RoutedExpertDomain gpu;gpu.name="gpu";gpu.scope=ExecutionDomainScope::RANK_LOCAL;gpu.owner_rank=0;
    gpu.backend=CollectiveBackendType::NCCL;gpu.participants={GlobalDeviceAddress::cuda(0)};
    gpu.routed_compute_policy=RoutedExpertComputePolicy::Apportioned;
    RoutedExpertDomain cpu;cpu.name="cpu";cpu.scope=ExecutionDomainScope::NODE_LOCAL;cpu.owner_rank=-1;
    cpu.backend=CollectiveBackendType::UPI;cpu.routed_compute_policy=RoutedExpertComputePolicy::TensorSharded;
    cpu.routed_phase_policy=RoutedExpertPhasePolicy::Uniform;
    cpu.routed_decode_assignment_policy=cpu.routed_prefill_assignment_policy=RoutedExpertAssignmentPolicy::StaticOwner;
    for(int member=0;member<cpu_members;++member){cpu.participants.push_back(GlobalDeviceAddress::cpu(member));cpu.world_ranks.push_back(member+1);}
    plan.domains={gpu,cpu};plan.routed_tiers={{.name="hot",.domain="gpu",.priority=0},
        {.name="cold",.domain="cpu",.priority=1,.fallback=true}};
    std::vector<int> tiers(512,1);tiers[0]=tiers[2]=0;
    plan.placements={{.layer=0,.routed_expert_tier=std::move(tiers)}};return plan;
}
/** @return Exact count of an existing pool key from immutable projected requirements. */
size_t pieceDemandCount(const MoEOverlayPhysicalLaneCounts& counts,MoEOverlayPhysicalLaneKind kind,
    int rank,DeviceId source,DeviceId destination,ExpertTierWeightProjection role) {
    const auto found=counts.find({.kind=kind,.world_rank=rank,.source=source,.destination=destination,.projection=role});
    return found==counts.end()?0:found->second;
}
}
/** @test Whole-to-cohort fanout prices actual P pieces, not one scalar owner or P-squared padding. */
TEST(MoEOverlayPhysicalPieceDemand, WholeCohortFanoutUsesOriginalCutsAndOneMaterializedPlan) {
    for(const auto& format:native_quantised_formats::kAllSourceFormats) {
    SCOPED_TRACE(format.quant_type);
    const std::vector<MoEOverlayLayerWeightManifest> source{pieceDemandSource(*format.metadata)};
    for(int members=2;members<=8;++members) {
        SCOPED_TRACE(members);const auto plan=pieceDemandPlan(members);
        const auto targets=configuredMoEOverlayProjectionTargets(plan,source.front(),512);
        ASSERT_EQ(targets.size(),2u);ASSERT_EQ(targets[1].members.size(),size_t(members));
        const auto pieces=physicalMoEOverlayProjectionPieces(targets[0],targets[1]);
        ASSERT_EQ(pieces.size(),size_t(3*members));
        for(const auto& piece:pieces) {
            EXPECT_FALSE(piece.intersection.empty());EXPECT_EQ(piece.source_member,0u);
            EXPECT_EQ(piece.destination_participant,targets[1].members[piece.destination_member].participant_id);
        }
        constexpr size_t cycles=2;
        const auto demand=configuredMoEOverlayLaneDemand(plan,source,512,members+1,cycles,cycles);
        const auto capacity=demand.materializedCounts();
        const auto directory=remoteMoEOverlayProjectionLaneBudget(demand);
        ASSERT_TRUE(directory.projectionOperationCount());
        EXPECT_EQ(*directory.projectionOperationCount(),6u*size_t(members)*cycles);
        for(const auto role:{ExpertTierWeightProjection::Gate,ExpertTierWeightProjection::Up,ExpertTierWeightProjection::Down}) {
            EXPECT_EQ(pieceDemandCount(demand.migration_pieces,MoEOverlayPhysicalLaneKind::RemoteSource,
                0,DeviceId::cuda(0),DeviceId::invalid(),role),cycles*size_t(members));
            EXPECT_EQ(pieceDemandCount(demand.setup_ports,MoEOverlayPhysicalLaneKind::RemoteSource,
                0,DeviceId::cuda(0),DeviceId::invalid(),role),cycles);
            EXPECT_EQ(pieceDemandCount(capacity,MoEOverlayPhysicalLaneKind::RemoteProjectionDirectory,
                -1,DeviceId::invalid(),DeviceId::invalid(),role),cycles*size_t(2*members));
        }
        const auto actual=physicalMoEOverlayLaneCounts(pieces,targets[0],targets[1],members+1);
        EXPECT_NO_THROW(requireFitsMoEOverlayLaneDemand(demand,actual));
        auto excessive=actual;for(auto& [key,count]:excessive)count=capacity.at(key)+1;
        EXPECT_THROW(requireFitsMoEOverlayLaneDemand(demand,excessive),std::invalid_argument);
        MoEOverlayCapacityAdmissionPolicy policy{.migration_storage=MoEOverlayMigrationStorageKind::PhysicalResidencyFabric,
            .shadow_slots_per_endpoint_layer=cycles,.staging_capacity_bytes=4096,
            .maximum_concurrent_cycles=cycles,.maximum_cycles_per_layer=cycles,.distributed_transport=true,.overlay_world_size=members+1};
        const auto bom=MoEOverlayCapacityAdmission::transferStagingBOM(plan,members+1,policy,source,512);
        const auto gpu=std::find_if(bom.begin(),bom.end(),[](const auto& row){return row.world_rank==0 && row.device==DeviceId::cuda(0);});
        ASSERT_NE(gpu,bom.end());EXPECT_EQ(gpu->bytes,6u*size_t(members)*cycles*4096u);
        auto colocated=plan;
        for(int member=0;member<members;++member)colocated.domains.back().world_ranks[size_t(member)]=member;
        const auto colocated_demand=configuredMoEOverlayLaneDemand(colocated,source,512,members,cycles,cycles);
        const auto colocated_budget=remoteMoEOverlayProjectionLaneBudget(colocated_demand);
        EXPECT_EQ(colocated_budget.maximum_pieces_per_projection_wave,
            cycles*std::max(size_t(members+1),size_t(2*(members-1))));
        for(const auto role:{ExpertTierWeightProjection::Gate,ExpertTierWeightProjection::Up,ExpertTierWeightProjection::Down})
            EXPECT_EQ(pieceDemandCount(colocated_demand.migration_pieces,MoEOverlayPhysicalLaneKind::RemoteSource,
                0,DeviceId::cuda(0),DeviceId::invalid(),role),cycles*size_t(members-1));
    }
    }
}
/** @test Intermediate apportioned tiers remain separate targets, not additional cohort cycles. */
TEST(MoEOverlayPhysicalPieceDemand, TwoCUDAFourROCmTwoCPUDeclaredTargetsRemainExact) {
    auto plan=pieceDemandPlan(2);plan.domains.front().participants.push_back(GlobalDeviceAddress::cuda(1));
    auto warm=plan.domains.front();warm.name="warm";warm.backend=CollectiveBackendType::RCCL;
    warm.participants={GlobalDeviceAddress::rocm(0),GlobalDeviceAddress::rocm(1),GlobalDeviceAddress::rocm(2),GlobalDeviceAddress::rocm(3)};
    plan.domains.insert(plan.domains.begin()+1,warm);
    plan.routed_tiers.insert(plan.routed_tiers.begin()+1,{.name="warm",.domain="warm",.priority=1});
    plan.routed_tiers.back().priority=2;
    const std::vector<MoEOverlayLayerWeightManifest> source{pieceDemandSource()};
    const auto targets=configuredMoEOverlayProjectionTargets(plan,source.front(),512);
    ASSERT_EQ(targets.size(),7u);EXPECT_EQ(targets.back().members.size(),2u);
    const auto demand=configuredMoEOverlayLaneDemand(plan,source,512,3,1,1);
    const auto counts=demand.materializedCounts();
    for(const auto role:{ExpertTierWeightProjection::Gate,ExpertTierWeightProjection::Up,ExpertTierWeightProjection::Down}) {
        EXPECT_EQ(pieceDemandCount(counts,MoEOverlayPhysicalLaneKind::RemoteProjectionDirectory,-1,
            DeviceId::invalid(),DeviceId::invalid(),role),8u);
        for(const auto device:{DeviceId::cuda(0),DeviceId::cuda(1),DeviceId::rocm(0),DeviceId::rocm(1),DeviceId::rocm(2),DeviceId::rocm(3)})
            EXPECT_EQ(pieceDemandCount(counts,MoEOverlayPhysicalLaneKind::RemoteSource,0,device,DeviceId::invalid(),role),2u);
    }
    auto unsupported=plan;unsupported.domains.front().routed_compute_policy=RoutedExpertComputePolicy::TensorSharded;
    EXPECT_THROW((void)configuredMoEOverlayLaneDemand(unsupported,source,512,3,1,1),std::invalid_argument);
}
/** @test Retained setup ports are explicit capacities, never fictitious remote traffic. */
TEST(MoEOverlayPhysicalPieceDemand, RetainedPortsPreserveWholeOwnerBOMWithoutTrafficClaims) {
    auto plan=pieceDemandPlan(2);auto cuda=plan.domains.front();
    auto rocm=cuda;rocm.name="rocm";rocm.backend=CollectiveBackendType::RCCL;rocm.participants={GlobalDeviceAddress::rocm(2)};
    auto cpu=plan.domains.back();cpu.scope=ExecutionDomainScope::RANK_LOCAL;cpu.owner_rank=0;cpu.world_ranks.clear();
    cpu.participants={GlobalDeviceAddress::cpu(0)};cpu.routed_compute_policy=RoutedExpertComputePolicy::Apportioned;
    plan.domains={cuda,rocm,cpu};plan.routed_tiers={{.name="cuda",.domain="gpu",.priority=0},
        {.name="rocm",.domain="rocm",.priority=1},{.name="cpu",.domain="cpu",.priority=2,.fallback=true}};
    const std::vector<MoEOverlayLayerWeightManifest> source{pieceDemandSource()};
    const auto demand=configuredMoEOverlayLaneDemand(plan,source,512,2,1,1);
    for(const auto& [key,count]:demand.migration_pieces) {
        EXPECT_NE(key.kind,MoEOverlayPhysicalLaneKind::RemoteSource);
        EXPECT_NE(key.kind,MoEOverlayPhysicalLaneKind::RemoteDestination);
    }
    const MoEOverlayCapacityAdmissionPolicy policy{.migration_storage=MoEOverlayMigrationStorageKind::PhysicalResidencyFabric,
        .shadow_slots_per_endpoint_layer=1,.staging_capacity_bytes=4096,.distributed_transport=true,.overlay_world_size=2};
    const auto bom=MoEOverlayCapacityAdmission::transferStagingBOM(plan,2,policy,source,512);
    const auto bytes=[&](int rank,DeviceId device){const auto found=std::find_if(bom.begin(),bom.end(),[&](const auto& row){return row.world_rank==rank && row.device==device;});
        return found==bom.end()?size_t{0}:found->bytes;};
    EXPECT_EQ(bytes(0,DeviceId::cuda(0)),12u*4096u);EXPECT_EQ(bytes(0,DeviceId::rocm(2)),12u*4096u);
    EXPECT_EQ(bytes(0,DeviceId::cpu()),57u*4096u);EXPECT_EQ(bytes(1,DeviceId::cpu()),9u*4096u);
}
/** @test Different cohort cuts count only nonempty pieces; local CPU workers add no staging family. */
TEST(MoEOverlayPhysicalPieceDemand, UnequalCohortsKeepDirectCPUWorkersScratchFree) {
    for(const auto& format:native_quantised_formats::kAllSourceFormats) {
        SCOPED_TRACE(format.quant_type);
        auto plan=pieceDemandPlan(2);auto third=plan.domains.back();third.name="third";
        third.participants.push_back(GlobalDeviceAddress::cpu(2));third.world_ranks.push_back(3);
        plan.domains.push_back(third);plan.routed_tiers.back().fallback=false;
        plan.routed_tiers.push_back({.name="third",.domain="third",.priority=2,.fallback=true});
        const auto source=pieceDemandSource(*format.metadata);
        const auto targets=configuredMoEOverlayProjectionTargets(plan,source,512);
        ASSERT_EQ(targets.size(),3u);
        const auto pieces=physicalMoEOverlayProjectionPieces(targets[1],targets[2]);
        EXPECT_EQ(pieces.size(),12u); // Four actual interval intersections for each role, not2×3.
        const auto actual=physicalMoEOverlayLaneCounts(pieces,targets[1],targets[2],4);
        size_t workers=0,global=0;
        for(const auto& [key,count]:actual) {
            if(key.kind==MoEOverlayPhysicalLaneKind::LocalCPUWorker)workers+=count;
            else {EXPECT_EQ(key.kind,MoEOverlayPhysicalLaneKind::RemoteProjectionDirectory);global+=count;}
        }
        EXPECT_EQ(workers,6u);EXPECT_EQ(global,6u);
        std::fill(plan.placements.front().routed_expert_tier.begin(),plan.placements.front().routed_expert_tier.end(),0);
        const auto empty=configuredMoEOverlayProjectionTargets(plan,source,512);
        EXPECT_EQ(empty[1].members.size(),2u);EXPECT_EQ(empty[2].members.size(),3u);
        // Declared layout pricing is passive metadata; this is no arrival,
        // initial-source permit, physical reservation or native readiness proof.
    }
}
/** @test Global pools take feasible layer-cycle slots, not a sum of independent maxima. */
TEST(MoEOverlayPhysicalPieceDemand, LayerCapsProduceCheckedNonfactorableWaveBounds) {
    auto plan=pieceDemandPlan(2);auto fourth=plan.domains.back();fourth.name="fourth";
    for(int member=2;member<4;++member){fourth.participants.push_back(GlobalDeviceAddress::cpu(member));fourth.world_ranks.push_back(member+1);}
    // Reorder physical ranks and NUMA endpoints together. The same process
    // still names the same socket in both original domains.
    fourth.participants={GlobalDeviceAddress::cpu(2),GlobalDeviceAddress::cpu(0),GlobalDeviceAddress::cpu(1),GlobalDeviceAddress::cpu(3)};
    fourth.world_ranks={3,1,2,4};
    plan.domains.push_back(fourth);plan.routed_tiers.back().fallback=false;
    plan.routed_tiers.push_back({.name="fourth",.domain="fourth",.priority=2,.fallback=true});
    const auto source=[](int layer,int f){const auto format=ExpertWeightFormat::nativeQuantised({.codebook_id=native_quantised_formats::Q4_0.codebook_id,
        .is_superblock=native_quantised_formats::Q4_0.is_superblock,.present=true});
        return MoEOverlayLayerWeightManifest{.layer_idx=layer,.projections={{{ExpertTierWeightProjection::Gate,f,256,format},
            {ExpertTierWeightProjection::Up,f,256,format},{ExpertTierWeightProjection::Down,256,f,format}}}};};
    // Six32-column blocks cut2↔4 at unequal origins: five intersections.
    // Eight blocks cut2↔4 at shared boundaries: four intersections.
    const std::vector<MoEOverlayLayerWeightManifest> profiles{source(0,192),source(1,256)};
    const auto demand=configuredMoEOverlayLaneDemand(plan,profiles,512,5,3,2);
    const auto budget=remoteMoEOverlayProjectionLaneBudget(demand);
    EXPECT_EQ(budget.maximum_pieces_per_projection_wave,26u); //9+9+8, not9×3 or(9+8)×3.
    ASSERT_TRUE(budget.projectionOperationCount());EXPECT_EQ(*budget.projectionOperationCount(),78u);
    const auto one=configuredMoEOverlayLaneDemand(plan,{profiles.data(),1},512,5,4,1);
    EXPECT_EQ(remoteMoEOverlayProjectionLaneBudget(one).maximum_pieces_per_projection_wave,9u);
    const auto one_global=configuredMoEOverlayLaneDemand(plan,{profiles.data(),1},512,5,1,1);
    auto duplicates=profiles;duplicates[1]=profiles[0];duplicates[1].layer_idx=1;
    const auto two_layers_one_global=configuredMoEOverlayLaneDemand(plan,duplicates,512,5,1,1);
    EXPECT_EQ(one_global.materializedCounts(),two_layers_one_global.materializedCounts());
    duplicates[1].layer_idx=0;
    EXPECT_THROW((void)configuredMoEOverlayLaneDemand(plan,duplicates,512,5,1,1),std::invalid_argument);
}
/** @test Local-only staging preserves its original nonzero process rank without a fake communicator. */
TEST(MoEOverlayPhysicalPieceDemand, RankLocalScopeUsesOriginalNonzeroRank) {
    auto plan=pieceDemandPlan(2);auto& cpu=plan.domains.back();
    cpu.participants.resize(1);cpu.world_ranks.clear();cpu.owner_rank=3;
    cpu.scope=ExecutionDomainScope::RANK_LOCAL;cpu.routed_compute_policy=RoutedExpertComputePolicy::Apportioned;
    plan.domains.front().owner_rank=3;
    const std::vector<MoEOverlayLayerWeightManifest> source{pieceDemandSource()};
    const auto local=MoEOverlayPhysicalLaneScope::rankLocal(3);
    const auto demand=configuredMoEOverlayLaneDemand(plan,source,512,local,1,1);
    EXPECT_TRUE(demand.setup_ports.empty());
    for(const auto& [key,count]:demand.migration_pieces) {
        EXPECT_EQ(key.world_rank,3);EXPECT_NE(key.kind,MoEOverlayPhysicalLaneKind::RemoteProjectionDirectory);
    }
    EXPECT_THROW((void)configuredMoEOverlayLaneDemand(plan,source,512,1,1,1),std::invalid_argument);
    const auto inactive=configuredMoEOverlayLaneDemand(plan,source,512,MoEOverlayPhysicalLaneScope::rankLocal(0),1,1);
    EXPECT_TRUE(inactive.materializedCounts().empty());
    auto remote=pieceDemandPlan(2);
    const auto remote_targets=configuredMoEOverlayProjectionTargets(remote,source.front(),512);
    const auto remote_pieces=physicalMoEOverlayProjectionPieces(remote_targets[0],remote_targets[1]);
    EXPECT_THROW((void)physicalMoEOverlayLaneCounts(remote_pieces,remote_targets[0],remote_targets[1],
        MoEOverlayPhysicalLaneScope::rankLocal(0)),std::invalid_argument);
    EXPECT_THROW((void)MoEOverlayPhysicalLaneScope::rankLocal(-1),std::invalid_argument);
}
/** @test Repeated complete target vertices cannot masquerade as one admitted simple cycle. */
TEST(MoEOverlayPhysicalPieceDemand, RepeatedTargetCycleAndTornPieceRejectBeforeSubmission) {
    auto plan=pieceDemandPlan(2);const auto source=pieceDemandSource();
    const auto owners=MoEExpertOwnerMap::buildShardPlacement(plan,{source});
    std::vector<MoEOverlayTierMigration> edges;
    for(int expert=0;expert<4;++expert) {
        auto from=*owners.placementFor(0,expert);auto to=*owners.placementFor(0,expert%2?0:1);to.expert_id=expert;
        edges.push_back({.layer_idx=0,.expert_id=expert,.source=std::move(from),.destination=std::move(to)});
    }
    const std::vector<MoEOverlayTierMigrationCycle> repeated{{.layer_idx=0,.migration_indices={0,1,2,3}}};
    EXPECT_TRUE(repeated.front().valid(edges));
    EXPECT_THROW(requireSimpleMoEOverlayPhysicalCycles(edges,repeated),std::invalid_argument);
    const std::vector<MoEOverlayTierMigrationCycle> canonical{{.layer_idx=0,.migration_indices={0,1}},
        {.layer_idx=0,.migration_indices={2,3}}};
    EXPECT_NO_THROW(requireSimpleMoEOverlayPhysicalCycles(edges,canonical));
    const auto targets=configuredMoEOverlayProjectionTargets(plan,source,512);
    auto pieces=physicalMoEOverlayProjectionPieces(targets[0],targets[1]);
    pieces.push_back(pieces.back());
    EXPECT_THROW((void)physicalMoEOverlayLaneCounts(pieces,targets[0],targets[1],3),std::invalid_argument);
}
}
