/**
 * @file Test__MoEOptimizationMovementEndpoint.cpp
 * @brief Complete original target/lifetime/export replay without native readiness.
 *
 * Widths2..8 are checked source metadata, not claims of native sockets or ranks.
 * Actual transport completion is independently replayed through per-piece joins.
 */
#include "execution/moe/MoEOptimizationMovementEndpoint.h"
#include "execution/moe/MoEExpertOwnerMap.h"
#include "execution/moe/MoEExpertShardCohort.h"
#include "app/modes/MoEMovementLedgerJson.h"
#include <gtest/gtest.h>
namespace llaminar2::test {
/** @test Every original member/cut survives repeated publication and checked scalar projection. */
TEST(MoEOptimizationMovementEndpoint, OriginalCohortLifetimeAndLosslessExportReplay)
{
    for(int width=2;width<=8;++width) {
        SCOPED_TRACE(width);
        const auto format=ExpertWeightFormat::floating(TensorType::FP32);
        const MoEOverlayLayerWeightManifest source{.layer_idx=3,.projections={{{ExpertTierWeightProjection::Gate,64*width,32,format},
            {ExpertTierWeightProjection::Up,64*width,32,format},{ExpertTierWeightProjection::Down,32,64*width,format}}}};
        std::vector<MoEExpertShardCohort::Participant> members;
        for(int member=0;member<width;++member)members.push_back({member+1,member,GlobalDeviceAddress::cpu(member)});
        auto original=std::make_shared<const MoEExpertShardCohort>(MoEExpertShardCohort::checked(source,4,members,
            MoEExpertShardCohort::balancedIntervals(source,width)));
        MoEExpertOwner owner{.layer_idx=3,.expert_id=1,.tier_idx=1,.owner_participant=-1,
            .device=DeviceId::invalid(),.shard_cohort=original};
        auto endpoint=MoEOptimizationMovementEndpoint::fromOwner(owner);
        EXPECT_TRUE(endpoint.valid());EXPECT_TRUE(endpoint.isCPUCohort());
        // Public default moves leave a cohort arm with an empty shared borrow.
        auto moved_from=endpoint;
        auto moved_to=std::move(moved_from);
        EXPECT_TRUE(moved_to==endpoint);
        EXPECT_FALSE(moved_from.valid());
        EXPECT_TRUE(moved_from==moved_from);
        EXPECT_FALSE(moved_from==endpoint);
        EXPECT_FALSE(endpoint==moved_from);
        auto second_moved_from=endpoint;
        auto second_moved_to=std::move(second_moved_from);
        EXPECT_TRUE(moved_from==second_moved_from);
        EXPECT_TRUE(second_moved_to==moved_to);
        moved_from=endpoint;
        EXPECT_TRUE(moved_from.valid());
        EXPECT_TRUE(moved_from==endpoint);
        EXPECT_EQ(endpoint.target().word(),MoEOverlayRouteTarget::cpuCohort(1).word());
        EXPECT_EQ(endpoint.cohort().original,original);
        EXPECT_THROW(endpoint.whole(),std::logic_error);
        auto alias=owner;alias.owner_participant=members.front().participant_id;alias.device=DeviceId::cpu();
        EXPECT_THROW(MoEOptimizationMovementEndpoint::fromOwner(alias),std::invalid_argument);
        const auto whole=MoEOptimizationMovementEndpoint::whole(0,DeviceId::cuda(0),0);
        EXPECT_THROW(whole.cohort(),std::logic_error);
        for(int replay=0;replay<20;++replay) {
            const auto json=movement_json_detail::endpointRecord(endpoint);
            ASSERT_EQ(json.at("members").size(),size_t(width));
            EXPECT_EQ(json.at("kind"),"cpu_cohort");EXPECT_EQ(json.at("tier"),1);EXPECT_EQ(json.at("layer"),3);
            for(int member=0;member<width;++member) {
                const auto& row=json.at("members").at(member);
                EXPECT_EQ(row.at("participant"),member+1);EXPECT_EQ(row.at("world_rank"),member);
                EXPECT_EQ(row.at("address"),members[member].address.toString());
                EXPECT_EQ(row.at("numa_node"),member);
                ASSERT_EQ(row.at("projections").size(),3u);
                EXPECT_EQ(row.at("projections").at(0).at("first_row"),64*member);
                EXPECT_EQ(row.at("projections").at(2).at("first_column"),64*member);
            }
            MoEOptimizationMovementEdge edge{.authority=MoEOptimizationAuthority::Host,.transaction=2,.candidate_epoch=2,
                .layer=3,.expert=1,.cycle_index=0,.cycle_size=2,.direction=MoEOptimizationMovementDirection::Demotion,
                .source=whole,.destination=endpoint,.source_priority=0,.destination_priority=1,.estimated_weight_bytes=1024};
            ASSERT_TRUE(edge.valid());
            auto foreign_layer=edge;foreign_layer.layer=4;EXPECT_FALSE(foreign_layer.valid());
            EXPECT_THROW(movement_json_detail::edgeRecord(foreign_layer),std::invalid_argument);
            const auto exported=movement_json_detail::edgeRecord(edge);
            EXPECT_EQ(exported.at("destination"),json);
            EXPECT_FALSE(exported.contains("destination_participant"));
            std::swap(edge.source,edge.destination);std::swap(edge.source_priority,edge.destination_priority);
            edge.direction=MoEOptimizationMovementDirection::Promotion;EXPECT_TRUE(edge.valid());
            EXPECT_EQ(movement_json_detail::edgeRecord(edge).at("source"),json);
        }
        auto recut=MoEExpertShardCohort::balancedIntervals(source,width);
        recut[0].columns-=16;recut[1].first_column-=16;recut[1].columns+=16;
        auto changed=owner;changed.shard_cohort=std::make_shared<const MoEExpertShardCohort>(
            MoEExpertShardCohort::checked(source,4,members,recut));
        EXPECT_FALSE(endpoint==MoEOptimizationMovementEndpoint::fromOwner(changed));
        std::weak_ptr<const MoEExpertShardCohort> lifetime=original;original.reset();owner.shard_cohort.reset();
        // The rejected representative and public move replay still hold genuine
        // shared borrows. Release each original owner before claiming final retirement.
        EXPECT_FALSE(lifetime.expired());
        alias.shard_cohort.reset();
        moved_from={};
        moved_to={};
        second_moved_to={};
        EXPECT_FALSE(lifetime.expired())<<"The original endpoint still owns the complete cohort";
        endpoint={};
        EXPECT_TRUE(lifetime.expired())<<"The final original borrow must release the cohort";
    }
    EXPECT_FALSE(MoEOptimizationMovementEndpoint{}.valid());
    EXPECT_THROW(MoEOptimizationMovementEndpoint::whole(-1,DeviceId::cpu()),std::invalid_argument);
    EXPECT_THROW(MoEOptimizationMovementEndpoint::whole(0,DeviceId::invalid()),std::invalid_argument);
    EXPECT_THROW(MoEOptimizationMovementEndpoint::whole(0,DeviceId::cpu(),-1),std::invalid_argument);
    EXPECT_TRUE(movement_json_detail::endpointRecord(MoEOptimizationMovementEndpoint::whole(0,DeviceId::cpu())).at("world_rank").is_null());
}
}
