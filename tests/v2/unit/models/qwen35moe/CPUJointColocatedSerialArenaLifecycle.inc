/**
 * @file CPUJointColocatedSerialArenaLifecycle.inc
 * @brief Actual joint CPU serial arena materialization with admitted original local F.
 */
#include "memory/NUMAAllocator.h"

/** @test Original root and contributor arenas use local F, stable rows and exact admitted storage. */
TEST(Test__Qwen35MoEGraph, CPUJointColocatedSerialArenaUsesOriginalFragmentWidth)
{
    const int node=NUMAAllocator::instance().getCurrentNUMANode();
    ASSERT_GE(node,0);
    const auto format=ExpertWeightFormat::floating(TensorType::FP32);
    const MoEOverlayLayerWeightManifest layer{.layer_idx=0,.projections={{{ExpertTierWeightProjection::Gate,64,32,format},
        {ExpertTierWeightProjection::Up,64,32,format},{ExpertTierWeightProjection::Down,32,64,format}}}};
    for (const size_t local_member : {0u,1u})
    {
        SCOPED_TRACE(local_member);
        const int peer_node=node==0?1:0;
        const std::vector<MoEExpertShardCohort::Participant> members{
            {7,0,GlobalDeviceAddress::cpu(local_member==0?node:peer_node)},
            {19,1,GlobalDeviceAddress::cpu(local_member==1?node:peer_node)}};
        const int participant=members[local_member].participant_id;
        const int foreign_participant=members[1-local_member].participant_id;
        const auto cohort=std::make_shared<const MoEExpertShardCohort>(MoEExpertShardCohort::checked(
            layer,4,members,MoEExpertShardCohort::balancedIntervals(layer,2)));
        CPUJointExpertRouteStorageLayout joint(cohort,local_member,8,2);
        ASSERT_EQ(joint.intermediate(),32);
        auto admission=std::make_shared<CPUSerialExpertStorageAdmission>();
        admission->allocator={static_cast<int>(local_member),DeviceId::cpu()};admission->participant_ids={participant};
        admission->controller_devices={DeviceId::cpu()};
        admission->envelope=std::make_shared<const CPUSerialExpertStorageEnvelope>(DeviceId::cpu(),8,1,32,2,
            std::vector<int>{1,2,4,8},1,8,joint.isRoot(),CPUSerialExpertFamilyPolicy::LocalPowerOfTwo,32,4,joint);
        const auto declared=admission->envelope->forController(DeviceId::cpu());
        const size_t arenaBytes=admission->physicalBytes(),workspaceBytes=declared->workspaceBytes();
        const size_t total=CPUSerialExpertStorageLayout::add(arenaBytes,workspaceBytes);
        PhysicalMemoryPlanBuilder plan;
        const PhysicalMemoryResource resource{static_cast<int>(local_member),DeviceId::cpu(),total,total};
        plan.add(resource,PhysicalMemoryOwner::ActivationArena,arenaBytes)
            .add(resource,PhysicalMemoryOwner::ExecutionWorkspace,workspaceBytes);
        auto memory=std::make_shared<PhysicalMemoryAuthority>(std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()),static_cast<int>(local_member));
        auto live=[&] {size_t bytes=0;for(const auto& row:memory->rankAttestation())bytes+=row.materialized_new_bytes;return bytes;};
        PreparedWeightStore store;store.installPhysicalMemoryAuthority(memory);
        {
            auto config=makeMoEConfig();config.d_model=32;config.head_dim=16;config.d_ff=64;
            config.moe.num_experts=4;config.moe.top_k=2;config.moe.intermediate_size=64;
            config.max_seq_len=config.max_activation_rows=8;config.cpu_serial_expert_storage={admission};
            Qwen35MoEGraph graph(config);graph.setPreparedWeightStore(&store);
            const auto original=Qwen35MoEPrefixRuntimeTestPeer::serialArena(graph,participant,8);
            ASSERT_NE(original,nullptr);ASSERT_NE(original->cpuGroupedWorkspace(),nullptr);
            ASSERT_NE(original->cpuJointPackedRecords(),nullptr);
            EXPECT_EQ(bool(original->cpuCanonicalRoutes()),joint.isRoot());
            EXPECT_EQ(original->allocationBytes(),total);EXPECT_EQ(live(),total);
            for(const size_t rows:{size_t(8),size_t(1),size_t(4),size_t(8)}) {
                auto reused=Qwen35MoEPrefixRuntimeTestPeer::serialArena(graph,participant,rows);
                EXPECT_EQ(reused,original);
                auto lease=reused->cpuGroupedWorkspace()->acquire(rows,2,32,32,4,0);
                EXPECT_EQ(live(),total);
            }
            EXPECT_THROW((void)original->cpuGroupedWorkspace()->acquire(1,2,32,64,4,0),std::invalid_argument);
            EXPECT_THROW((void)Qwen35MoEPrefixRuntimeTestPeer::serialArena(graph,participant,9),std::logic_error);
            EXPECT_THROW((void)Qwen35MoEPrefixRuntimeTestPeer::serialArena(graph,foreign_participant,1),std::invalid_argument);
            EXPECT_THROW((void)Qwen35MoEPrefixRuntimeTestPeer::serialArena(graph,participant,0),std::invalid_argument);
            EXPECT_EQ(live(),total);
        }
        EXPECT_EQ(live(),0u);
    }
}
