/**
 * @file Test__MoEOverlayPhysicalProjectionPieces.cpp
 * @brief Original-geometry contracts for the actual physical fabric normalizer.
 *
 * These metadata cases use the existing original source-plan builder and
 * participant map. Widths beyond physical hardware are geometry only: no
 * transport, readiness, model load, socket or native execution is invented.
 */
#include "execution/moe/MoEOverlayPhysicalProjectionPieces.h"
#include "execution/moe/MoEOverlayPhysicalRemotePieceOrdinals.h"
#include "../../../utils/CPUJointSourceTestPlan.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cstdint>

namespace llaminar2::test
{
namespace
{
/** @return One complete original floating layer, with unequal positive F cuts. */
MoEOverlayLayerWeightManifest pieceSource()
{
    const auto format = ExpertWeightFormat::floating(TensorType::FP32);
    return {.layer_idx = 7, .projections = {{
        {.projection = ExpertTierWeightProjection::Gate, .N = 544, .K = 64, .format = format},
        {.projection = ExpertTierWeightProjection::Up, .N = 544, .K = 64, .format = format},
        {.projection = ExpertTierWeightProjection::Down, .N = 64, .K = 544, .format = format}}}};
}
/** @return The existing original plan with a complete metadata-only CPU membership. */
MoERoutedExpertPlacementPlan piecePlan(int width)
{
    auto plan = cpuJointSourceTestPlan();
    auto &cpu = plan.domains[1];
    cpu.participants.clear(); cpu.world_ranks.clear();
    for (int member = 0; member < width; ++member)
    {
        cpu.participants.push_back(GlobalDeviceAddress::cpu(member));
        cpu.world_ranks.push_back(member);
    }
    return plan;
}
/** @return Same logical expert with exact original source and destination alternatives. */
MoEOverlayTierMigration pieceEdge(MoEExpertOwner from, MoEExpertOwner to)
{
    to.expert_id = from.expert_id;
    return {.layer_idx = 7, .expert_id = from.expert_id,
        .source = std::move(from), .destination = std::move(to)};
}
}

TEST(MoEOverlayPhysicalProjectionPieces, WholeAndCompleteCohortsNormalizeWithoutScalarAliases)
{
    const auto source = pieceSource();
    for (int width = 2; width <= 8; ++width)
    {
        SCOPED_TRACE(width);
        const auto topology = MoEExpertOwnerMap::buildShardPlacement(piecePlan(width), {source});
        const auto whole = *topology.placementFor(7, 0);
        const auto cohort = *topology.placementFor(7, 64);
        const auto edge = pieceEdge(whole, cohort);
        const auto pieces = physicalMoEOverlayProjectionPieces(edge, source, 512, topology);
        ASSERT_EQ(pieces.size(), size_t(3 * width));
        std::array<std::uint64_t, 3> area{};
        for (size_t index = 0; index < pieces.size(); ++index)
        {
            const auto &piece = pieces[index];
            EXPECT_EQ(size_t(piece.projection), index / width);
            EXPECT_EQ(piece.destination_member, index % width);
            EXPECT_EQ(piece.source_member, 0u);
            EXPECT_EQ(piece.source_participant, whole.owner_participant);
            EXPECT_EQ(piece.destination_participant,
                cohort.shard_cohort->participants()[index % width].participant_id);
            EXPECT_GT(piece.intersection.rows(), 0);
            EXPECT_GT(piece.intersection.columns(), 0);
            area[size_t(piece.projection)] += uint64_t(piece.intersection.rows()) * piece.intersection.columns();
        }
        for (const auto &projection : source.projections)
            EXPECT_EQ(area[size_t(projection.projection)], uint64_t(projection.N) * projection.K);
        const auto reverse = physicalMoEOverlayProjectionPieces(pieceEdge(edge.destination, whole), source, 512, topology);
        ASSERT_EQ(reverse.size(), pieces.size());
        for (const auto &piece : reverse)
        {
            EXPECT_EQ(piece.destination_member, 0u);
            EXPECT_EQ(piece.destination_participant, whole.owner_participant);
        }
        const auto singleton = physicalMoEOverlayProjectionPieces(pieceEdge(whole, whole), source, 512, topology);
        ASSERT_EQ(singleton.size(), 3u);
        for (const auto &piece : singleton)
        {
            EXPECT_EQ(piece.source_participant, whole.owner_participant);
            EXPECT_EQ(piece.destination_participant, whole.owner_participant);
            EXPECT_EQ(piece.intersection.sourceRowOrigin(), 0);
            EXPECT_EQ(piece.intersection.destinationColumnOrigin(), 0);
        }
    }
}

TEST(MoEOverlayPhysicalProjectionPieces, DifferentOriginalCutsProduceOnlyNonemptyIntersections)
{
    const auto source = pieceSource();
    const auto topology = MoEExpertOwnerMap::buildShardPlacement(piecePlan(2), {source});
    auto from = *topology.placementFor(7, 64);
    auto to = from;
    const auto original_members = from.shard_cohort->participants();
    auto recut = MoEExpertShardCohort::checked(source, 512,
        {original_members.begin(), original_members.end()}, {{0, 256}, {256, 288}});
    to.shard_cohort = std::make_shared<const MoEExpertShardCohort>(std::move(recut));
    const auto pieces = physicalMoEOverlayProjectionPieces(pieceEdge(from, to), source, 512, topology);
    // Original [0,272),[272,544) intersects destination [0,256),[256,544):
    // three actual pieces per role, never a padded 2x2 transport Cartesian product.
    ASSERT_EQ(pieces.size(), 9u);
    for (size_t role = 0; role < 3; ++role)
    {
        const auto offset = role * 3;
        EXPECT_EQ(pieces[offset].destination_member, 0u);
        EXPECT_EQ(pieces[offset].source_member, 0u);
        EXPECT_EQ(pieces[offset + 1].destination_member, 1u);
        EXPECT_EQ(pieces[offset + 1].source_member, 0u);
        EXPECT_EQ(pieces[offset + 2].destination_member, 1u);
        EXPECT_EQ(pieces[offset + 2].source_member, 1u);
        EXPECT_FALSE(pieces[offset + 1].intersection.empty());
        EXPECT_EQ(role == 2 ? pieces[offset + 1].intersection.columns() : pieces[offset + 1].intersection.rows(), 16);
    }
}

TEST(MoEOverlayPhysicalProjectionPieces, ForeignSourceExpertAndEndpointRejectBeforePhysicalWork)
{
    const auto source = pieceSource();
    const auto topology = MoEExpertOwnerMap::buildShardPlacement(piecePlan(2), {source});
    const auto from = *topology.placementFor(7, 0);
    const auto to = *topology.placementFor(7, 64);
    auto edge = pieceEdge(from, to);
    EXPECT_THROW((void)physicalMoEOverlayProjectionPieces(edge, source, 448, topology), std::invalid_argument);
    auto foreign = source;
    foreign.projections[0].format = ExpertWeightFormat::floating(TensorType::FP16);
    EXPECT_THROW((void)physicalMoEOverlayProjectionPieces(edge, foreign, 512, topology), std::invalid_argument);
    edge.source.address = GlobalDeviceAddress::cpu(0);
    EXPECT_THROW((void)physicalMoEOverlayProjectionPieces(edge, source, 512, topology), std::invalid_argument);
    edge = pieceEdge(from, to);
    edge.destination.owner_participant = edge.destination.shard_cohort->participants()[0].participant_id;
    EXPECT_THROW((void)physicalMoEOverlayProjectionPieces(edge, source, 512, topology), std::invalid_argument);
}
/** @test MPI slots follow the complete original wave, never a rank-local physical subset. */
TEST(MoEOverlayPhysicalProjectionPieces, GlobalRemoteOrdinalsRemainUniqueThroughEightSkewedMembers)
{
    const auto source=pieceSource();
    for(int width=2;width<=8;++width) {
        SCOPED_TRACE(width);
        const auto topology=MoEExpertOwnerMap::buildShardPlacement(piecePlan(width),{source});
        const auto whole=*topology.placementFor(7,0);
        const auto cohort=*topology.placementFor(7,64);
        auto recut=cohort;
        std::vector<MoEExpertShardCohort::Interval> cuts;
        int frontier=0;
        for(int member=0;member<width;++member) {
            const int count=member+1==width?544-frontier:1;
            cuts.push_back({frontier,count});frontier+=count;
        }
        const auto original_members=cohort.shard_cohort->participants();
        recut.shard_cohort=std::make_shared<const MoEExpertShardCohort>(MoEExpertShardCohort::checked(source,512,
            {original_members.begin(),original_members.end()},std::move(cuts)));
        std::vector<std::vector<MoEOverlayPhysicalProjectionPiece>> wave;
        wave.push_back(physicalMoEOverlayProjectionPieces(pieceEdge(whole,cohort),source,512,topology));
        wave.push_back(physicalMoEOverlayProjectionPieces(pieceEdge(cohort,recut),source,512,topology));
        wave.push_back(physicalMoEOverlayProjectionPieces(pieceEdge(cohort,whole),source,512,topology));
        const auto global=physicalMoEOverlayRemotePieceOrdinals(wave,topology,width);
        size_t expected=0;
        for(size_t edge=0;edge<wave.size();++edge)for(size_t index=0;index<wave[edge].size();++index) {
            const auto& piece=wave[edge][index];
            const auto sr=topology.participantForId(piece.source_participant)->world_rank;
            const auto dr=topology.participantForId(piece.destination_participant)->world_rank;
            const auto ordinal=global.by_migration[edge][index];
            if(sr==dr)EXPECT_FALSE(ordinal);
            else {ASSERT_TRUE(ordinal);EXPECT_EQ(*ordinal,expected++);}
        }
        EXPECT_EQ(global.remote_piece_count,expected);
        // Rank1 has only one source/destination per role in the first edge,
        // while rank0 sees every remote member. Their original slot IDs must
        // retain gaps rather than restarting from each local binding count.
        for(int rank=0;rank<width;++rank) {
            const auto local_view=physicalMoEOverlayRemotePieceOrdinals(wave,topology,width);
            EXPECT_EQ(local_view.by_migration,global.by_migration);
            for(size_t role=0;role<3;++role) {
                const auto index=role*size_t(width)+1;
                ASSERT_TRUE(local_view.by_migration[0][index]);
                EXPECT_EQ(*local_view.by_migration[0][index],role*size_t(width-1));
            }
        }
        auto missing=wave;missing.front().front().source_participant=-91;
        EXPECT_THROW((void)physicalMoEOverlayRemotePieceOrdinals(missing,topology,width),std::invalid_argument);
        EXPECT_THROW((void)physicalMoEOverlayRemotePieceOrdinals(wave,topology,width-1),std::invalid_argument);
        EXPECT_THROW((void)physicalMoEOverlayRemotePieceOrdinals(wave,topology,0),std::invalid_argument);
    }
}

}
