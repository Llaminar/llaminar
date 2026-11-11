/**
 * @file Test__RecordedJointPacketSourceMap.cpp
 * @brief Device-free original CPU cohort source-map and zero-hole replay.
 *
 * Recorded participants describe declared CPU NUMA/rank identity; no backend,
 * native context or physical capacity is impersonated. This keeps the original
 * source transition and exact route-selection oracles available in CPU-only
 * Unit and preflight builds, independent of native component hardware admission.
 */
#include <gtest/gtest.h>
#include "execution/moe/MoEExpertOwnerMap.h"
#include "execution/moe/MoERuntimeTable.h"
#include "execution/moe/MoEExpertShardCohort.h"
#include "execution/moe/MoEOverlaySharedDispatchSelection.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
using namespace llaminar2;
namespace {
    constexpr std::array<std::int32_t, 3> kLayers{5, 9, 14};
    constexpr std::int32_t kTopK = 2;
    constexpr std::uint32_t kExpertCount = 18u;
    constexpr std::uint64_t kPlacementEpoch = 41u;
    constexpr std::array<float, 8> kRouteWeightPattern{
        0.75f, 0.25f, 0.6f, 0.4f,
        0.55f, 0.45f, 0.9f, 0.1f};

    /** @return Cold original source cohort, including mixed ordinary GPU experts. */
    std::shared_ptr<const MoEExpertOwnerMap>
    jointPacketOwners(int degree, DeviceId device,
                      const MoEExpertOwnerMap *original = nullptr, bool swapped = false)
    {
        MoERoutedExpertPlacementPlan p; p.enabled=true; p.first_model_layer=kLayers.front();
        p.topology=RoutedExpertPlacementTopology::TieredOverlay;
        p.continuation_domain=p.base_model_domain=p.shared_expert_domain="compute";
        RoutedExpertDomain gpu;gpu.name="compute";gpu.scope=ExecutionDomainScope::RANK_LOCAL;
        gpu.backend=device.is_cuda()?CollectiveBackendType::NCCL:CollectiveBackendType::RCCL;gpu.owner_rank=0;
        gpu.participants={device.is_cuda()?GlobalDeviceAddress::cuda(0):GlobalDeviceAddress::rocm(0)};
        gpu.routed_compute_policy=RoutedExpertComputePolicy::Apportioned;
        RoutedExpertDomain cpu;cpu.name="joint";cpu.scope=ExecutionDomainScope::NODE_LOCAL;
        cpu.backend=CollectiveBackendType::UPI;cpu.owner_rank=-1;
        cpu.routed_compute_policy=RoutedExpertComputePolicy::TensorSharded;
        cpu.routed_phase_policy=RoutedExpertPhasePolicy::Uniform;
        cpu.routed_decode_assignment_policy=cpu.routed_prefill_assignment_policy=RoutedExpertAssignmentPolicy::StaticOwner;
        for (int member = 0; member < degree; ++member)
        {
            cpu.participants.push_back(GlobalDeviceAddress::cpu(member));
            cpu.world_ranks.push_back(member + 1);
        }
        RoutedExpertDomain outsider=cpu;outsider.name="outside";outsider.routed_compute_policy=RoutedExpertComputePolicy::Apportioned;
        outsider.participants={GlobalDeviceAddress::cpu(degree)};outsider.world_ranks={degree+1};
        p.domains={gpu,cpu,outsider};
        p.routed_tiers={{.name="whole",.domain="compute",.priority=0,.max_experts_per_layer=14},
                        {.name="fragments",.domain="joint",.priority=3,.max_experts_per_layer=4,.fallback=true},
                        {.name="outsider",.domain="outside",.priority=2,.max_experts_per_layer=1}};
        std::vector<MoEOverlayLayerWeightManifest> sources;
        const auto fp=ExpertWeightFormat::floating(TensorType::FP32);
        for(const int layer:kLayers){
            std::vector<int> tiers(kExpertCount, 0);
            if (swapped)
            {
                for (int e : {11, 12, 13, 16})
                    tiers[e] = 1;
            }
            else
            {
                for (int e : {10, 14, 15, 17})
                    tiers[e] = 1;
            }
            p.placements.push_back({.layer=layer,.routed_expert_tier=tiers});
            sources.push_back({.layer_idx=layer,.projections={{{ExpertTierWeightProjection::Gate,64,8,fp},
                {ExpertTierWeightProjection::Up,64,8,fp},{ExpertTierWeightProjection::Down,8,64,fp}}}});
        }
        return std::make_shared<const MoEExpertOwnerMap>(
            original ? MoEExpertOwnerMap::buildShardTransition(p, *original, *original)
                     : MoEExpertOwnerMap::buildShardPlacement(p, sources));
    }

}

// Native CPUJoint fanout and zero-weight compaction are owned by the genuine
// original MPI/NUMA CPUExpertDomainNative fixture. The metadata-only cohort
// tests below deliberately exercise arbitrary declared counts without claiming
// that those physical sockets are installed.

/** @test Cold fixture obeys original fallback ordering and retains every real source member. */
TEST(Test__MappedActivationPacketCUDAAndROCm, JointSourceFixtureHasCompleteOriginalColdPlacement)
{
    for(const auto device:{DeviceId::cuda(0),DeviceId::rocm(0)})
    for(const int degree:{2,4,8}) {
        SCOPED_TRACE("degree="+std::to_string(degree)+" device="+device.toString());
        const auto owners=jointPacketOwners(degree,device);
        ASSERT_EQ(owners->participants().size(),static_cast<std::size_t>(degree+2));
        for(const int layer:kLayers) {
            ASSERT_EQ(owners->expertsForParticipant(layer,0).size(),14u);
            for(int expert:{10,14,15,17}) {
                const auto* row=owners->placementFor(layer,expert);
                ASSERT_NE(row,nullptr);ASSERT_TRUE(row->shard_cohort);
                ASSERT_EQ(row->shard_cohort->participants().size(),static_cast<std::size_t>(degree));
                EXPECT_EQ(row->owner_participant,-1);
                const auto members=row->shard_cohort->participants();
                for(size_t index=0;index<members.size();++index) {
                    const auto& member=members[index];
                    const auto* original=owners->participantForId(member.participant_id);
                    ASSERT_NE(original,nullptr);EXPECT_EQ(original->domain_name,"joint");
                    EXPECT_EQ(original->world_rank,member.world_rank);EXPECT_EQ(original->address,member.address);
                    EXPECT_EQ(member.world_rank,static_cast<int>(index)+1);
                    EXPECT_EQ(member.address.numa_node,static_cast<int>(index));
                    for(size_t previous=0;previous<index;++previous) {
                        EXPECT_NE(member.world_rank,members[previous].world_rank);
                        EXPECT_NE(member.address.numa_node,members[previous].address.numa_node);
                        EXPECT_NE(member.participant_id,members[previous].participant_id);
                    }
                }
            }
        }
        const auto candidate = jointPacketOwners(degree, device, owners.get(), true);
        for (const int layer : kLayers)
        {
            const auto *original_cohort = owners->placementFor(layer, 10);
            for (int expert : {11, 12, 13, 16})
            {
                const auto *incoming = candidate->placementFor(layer, expert);
                ASSERT_TRUE(incoming && incoming->shard_cohort);
                EXPECT_EQ(incoming->shard_cohort, original_cohort->shard_cohort);
                EXPECT_FALSE(incoming->shard_cohort.owner_before(original_cohort->shard_cohort));
                EXPECT_FALSE(original_cohort->shard_cohort.owner_before(incoming->shard_cohort));
            }
            EXPECT_EQ(candidate->expertsForParticipant(layer, 0).size(), 14u);
            EXPECT_TRUE(owners->placementFor(layer, 10)->shard_cohort);
        }
        const auto outside=std::find_if(owners->participants().begin(),owners->participants().end(),[](const auto& p){return p.domain_name=="outside";});
        ASSERT_NE(outside,owners->participants().end());
        for(const int layer:kLayers)EXPECT_TRUE(owners->expertsForParticipant(layer,outside->participant_id).empty());
    }
}


/** @test Recorded swapped epoch41 source banks retain every declared member without fabricating native CPUs. */
TEST(Test__MappedActivationPacketCUDAAndROCm, JointRecordedSourceMapEpochAndZeroHoleContract)
{
    for(const auto device:{DeviceId::cuda(0),DeviceId::rocm(0)})
    for(const int degree:{2,4,8}) {
        const auto original=jointPacketOwners(degree,device);
        const auto candidate=jointPacketOwners(degree,device,original.get(),true);
        for(const bool swapped:{false,true})for(const int pattern:{0,1,2,3})
        for(const int physical:{1,65})for(const int recorded_live:{0,1,9,65}) {
            if(recorded_live>physical)continue;
            const auto& owners=swapped?candidate:original;
            std::array<float,65*kTopK> ids{},weights{};
            for(size_t slot=0;slot<ids.size();++slot) {
                ids[slot]=float(10+slot%8);
                weights[slot]=pattern==2 || (pattern==1 && slot%3!=1)?0.f:
                    pattern==3?std::bit_cast<float>(std::uint32_t{1}):kRouteWeightPattern[slot%kRouteWeightPattern.size()];
            }
            for(const auto& member:original->participants()) {
                if(member.domain_name!="joint" && member.domain_name!="outside")continue;
                const auto layer=kLayers.front();
                DeviceMoEPlacementBank bank{};bank.epoch=kPlacementEpoch+pattern;
                for(int expert=0;expert<kExpertCount;++expert) {
                    const auto* placement=owners->placementFor(layer,expert);ASSERT_NE(placement,nullptr);
                    bank.overlay_route_participant[expert]=placement->shard_cohort?
                        MoEOverlayRouteTarget::cpuCohort(placement->tier_idx).word():
                        MoEOverlayRouteTarget::whole(placement->owner_participant).word();
                }
                MoEOverlayActivationDispatchPackLaunch launch;
                launch.physical_rows=physical;launch.active_row_count_device=&recorded_live;
                launch.route_expert_ids_fp32=ids.data();launch.route_weights=weights.data();
                launch.placement.expert_count=kExpertCount;launch.packet.top_k=kTopK;
                launch.target_participant_id=member.participant_id;launch.model_layer_index=layer;
                if(member.domain_name=="joint")launch.joint_route_target=MoEOverlayJointRouteTarget::fromOriginalCohort(
                    kExpertCount,member.participant_id,layer,original->placementFor(layer,10)->tier_idx);
                for(int row=0;row<physical;++row) {
                    bool expected=false;
                    for(int slot=0;slot<kTopK && row<recorded_live;++slot) {
                        const auto coordinate=size_t(row*kTopK+slot);
                        const auto expert=int(ids[coordinate]);
                        const bool cohort=swapped?(expert==11 || expert==12 || expert==13 || expert==16):
                            (expert==10 || expert==14 || expert==15 || expert==17);
                        const bool useful=pattern!=2 && (pattern!=1 || coordinate%3==1);
                        expected|=member.domain_name=="joint" && cohort && useful;
                    }
                    EXPECT_EQ(selectMoEOverlaySharedRow(launch,bank.overlay_route_participant,bank.epoch,row),
                        expected?MoEOverlaySharedRowDecision::Needed:MoEOverlaySharedRowDecision::Unneeded);
                    EXPECT_EQ(selectMoEOverlaySharedRow(launch,bank.overlay_route_participant,0,row),MoEOverlaySharedRowDecision::Invalid);
                }
                if(member.domain_name=="joint") {
                    ++launch.model_layer_index;
                    if(recorded_live)EXPECT_EQ(selectMoEOverlaySharedRow(launch,bank.overlay_route_participant,bank.epoch,0),
                        MoEOverlaySharedRowDecision::Invalid);
                }
            }
        }
    }
}
