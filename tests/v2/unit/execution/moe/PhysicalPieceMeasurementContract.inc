/**
 * @file PhysicalPieceMeasurementContract.inc
 * @brief Recorded piece-observation coverage, identity and empty-rank reduction contracts.
 *
 * These are device-free retained-observation fixtures, not measured performance
 * samples or service certification. Actual events/ACKs remain production-owned.
 */
#include "execution/moe/MoEOverlayPhysicalProjectionPieces.h"
namespace llaminar2::test {
namespace {
MoEOverlayPhysicalProjectionTarget measurementTarget(int members, int firstId, int tier)
{
    MoEOverlayPhysicalProjectionTarget target{.tier_idx=tier};
    for(int member=0;member<members;++member) {
        MoEOverlayPhysicalProjectionMember value;
        value.participant_id=firstId+member;
        value.world_rank=firstId+member;
        value.device=DeviceId::cpu();
        value.address=GlobalDeviceAddress::cpu(member);
        const int width=215040/members,first=member*width;
        const auto rows = members == 1 ? MoEProjectionResidence::ExpertOwner : MoEProjectionResidence::ParticipantOutputSlice;
        const auto columns = members == 1 ? MoEProjectionResidence::ExpertOwner : MoEProjectionResidence::ParticipantInputSlice;
        value.projections={{{rows,215040,256,first,width,0,256},
            {rows,215040,256,first,width,0,256},
            {columns,256,215040,0,256,first,width}}};
        target.members.push_back(value);
    }
    return target;
}
MoEOverlayCompletedMigrationMeasurement pieceRow(int sourceMembers,int destinationMembers,ExpertWeightFormat format)
{
    const auto from=measurementTarget(sourceMembers,0,0),to=measurementTarget(destinationMembers,8,1);
    MoEOverlayCompletedMigrationMeasurement row{.expected_epoch=1,.candidate_epoch=2,.layer=7,.expert=11,
        .wave_wall_nanoseconds=101};
    row.source_participant=sourceMembers==1?0:-1;row.destination_participant=destinationMembers==1?8:-1;
    std::array<ExpertWeightFormat,3> formats;formats.fill(format);
    row.geometry=MoEOverlayMigrationMeasurementGeometry::checked(std::pair{from,to},formats);
    row.piece_observations.resize(row.geometry->pieces().size());
    return row;
}
ExpertTierProjectionTransferMeasurement recordedPiece(std::size_t piece)
{
    return {.sequence=3,.bytes=128+piece,.wall_nanoseconds=51+piece,.device_nanoseconds=0,.host_nanoseconds=21+piece};
}
}

TEST(PhysicalPieceMeasurementContract, CanonicalUnequalCutsMergeComplementaryAndEmptyRanks)
{
    std::vector<std::pair<int,int>> populations{{2,3},{3,2}};
    for(int members=2;members<=8;++members) { populations.emplace_back(1,members);populations.emplace_back(members,1); }
    for(const auto &format:native_vnni_formats::kAllSourceFormats) {
        const auto native=ExpertWeightFormat::nativeVnni({.codebook_id=format.metadata->codebook_id,
            .is_superblock=format.metadata->is_superblock,.present=true});
        for(const auto [sourceMembers,destinationMembers]:populations) {
            auto first=pieceRow(sourceMembers,destinationMembers,native),second=first,empty=first;
            ASSERT_TRUE(first.valid());
            for(std::size_t i=0;i<first.piece_observations.size();++i) {
                auto &slot=(i%2?second:first).piece_observations[i];
                slot.quoted_payload_bytes=128+i;slot.observation=recordedPiece(i);
            }
            const auto merged=MoEOverlayMigrationMeasurementMerger::merge({{first},{second},{empty}});
            ASSERT_EQ(merged.size(),1u);ASSERT_TRUE(merged.front().valid());
            EXPECT_EQ(merged.front().geometry,first.geometry);
            for(std::size_t i=0;i<first.piece_observations.size();++i) {
                ASSERT_TRUE(merged.front().piece_observations[i].observation);
                EXPECT_EQ(merged.front().piece_observations[i].observation->bytes,128+i);
            }
            auto missing=first;missing.piece_observations.pop_back();EXPECT_FALSE(missing.valid());
            EXPECT_THROW((void)MoEOverlayMigrationMeasurementMerger::merge({{missing},{second}}),std::invalid_argument);
        }
    }
}
TEST(PhysicalPieceMeasurementContract, MissingMemberAndOverlappingGeometryRejectBeforePolling)
{
    auto source=measurementTarget(2,0,0),destination=measurementTarget(3,8,1);
    std::array<ExpertWeightFormat,3> formats;formats.fill(ExpertWeightFormat::floating(TensorType::FP32));
    destination.members.pop_back();
    EXPECT_THROW((void)MoEOverlayMigrationMeasurementGeometry::checked({source,destination},formats),std::invalid_argument);
    destination=measurementTarget(3,8,1);destination.members[1].projections=destination.members[0].projections;
    EXPECT_THROW((void)MoEOverlayMigrationMeasurementGeometry::checked({source,destination},formats),std::invalid_argument);
    destination=measurementTarget(3,8,1);formats[1]={};
    EXPECT_THROW((void)MoEOverlayMigrationMeasurementGeometry::checked({source,destination},formats),std::invalid_argument);
}
TEST(PhysicalPieceMeasurementContract, SingleAndAllIdenticalWrongBytesCannotOverrideBoundQuote)
{
    auto row=pieceRow(2,3,ExpertWeightFormat::floating(TensorType::FP32));
    const auto empty=row;
    for(std::size_t i=0;i<row.piece_observations.size();++i) {
        row.piece_observations[i].quoted_payload_bytes=128+i;row.piece_observations[i].observation=recordedPiece(i);
    }
    ASSERT_TRUE(row.valid());
    auto bad=row;++bad.piece_observations.front().observation->bytes;
    EXPECT_FALSE(bad.valid());
    EXPECT_THROW((void)MoEOverlayMigrationMeasurementMerger::merge({{bad}}),std::invalid_argument);
    EXPECT_THROW((void)MoEOverlayMigrationMeasurementMerger::merge({{bad},{bad}}),std::invalid_argument);
    EXPECT_THROW((void)MoEOverlayMigrationMeasurementMerger::merge({{row},{bad}}),std::invalid_argument);
    bad=row;bad.piece_observations.front().quoted_payload_bytes.reset();EXPECT_FALSE(bad.valid());
    bad=row;bad.piece_observations.front().observation.reset();
    EXPECT_THROW((void)MoEOverlayMigrationMeasurementMerger::merge({{bad},{empty}}),std::invalid_argument);
}
TEST(PhysicalPieceMeasurementContract, SourceOnlyObserverCannotOmitOneDestinationMember)
{
    auto source=pieceRow(1,3,ExpertWeightFormat::floating(TensorType::FP32));const auto empty=source;
    for(std::size_t i=0;i<source.piece_observations.size();++i) {
        source.piece_observations[i].quoted_payload_bytes=128+i;source.piece_observations[i].observation=recordedPiece(i);
    }
    const auto merged=MoEOverlayMigrationMeasurementMerger::merge({{source},{empty},{empty}});
    ASSERT_EQ(merged.front().piece_observations.size(),9u);
    for(std::size_t i=0;i<source.piece_observations.size();++i)
        if(source.geometry->pieces()[i].destination_participant==10)source.piece_observations[i].observation.reset();
    EXPECT_THROW((void)MoEOverlayMigrationMeasurementMerger::merge({{source},{empty},{empty}}),std::invalid_argument);
}
TEST(PhysicalPieceMeasurementContract, OwnedJournalHandoffPreservesExactOriginalAllocations)
{
    auto row=pieceRow(2,3,ExpertWeightFormat::floating(TensorType::FP32));
    std::vector<MoEOverlayCompletedMigrationMeasurement> wave;wave.push_back(std::move(row));
    const auto *rows=wave.data();const auto *slots=wave[0].piece_observations.data();const auto geometry=wave[0].geometry;
    MoEOverlayMigrationMeasurementJournal journal({.maximum_migrations_per_wave=1});
    EXPECT_FALSE(journal.recordCompletedWave(static_cast<const std::vector<MoEOverlayCompletedMigrationMeasurement>&>(wave),nullptr));
    ASSERT_TRUE(journal.recordCompletedWave(std::move(wave),nullptr));
    std::vector<MoEOverlayCompletedMigrationMeasurement> received;
    ASSERT_TRUE(journal.consume(&received,nullptr));
    EXPECT_EQ(received.data(),rows);EXPECT_EQ(received[0].piece_observations.data(),slots);EXPECT_EQ(received[0].geometry,geometry);
}
}
#include "execution/moe/MoEOverlayEconomyEvidenceExchange.h"
namespace llaminar2::test {
TEST(PhysicalPieceMeasurementContract, ScalarPricesRejectCohortsAndPartialSingletons)
{
    for(int members:{1,2,3,8}) {
        auto forward=pieceRow(1,members,ExpertWeightFormat::floating(TensorType::FP32));
        auto reverse=forward;auto targets=forward.geometry->targets();std::swap(targets.first,targets.second);
        reverse.geometry=MoEOverlayMigrationMeasurementGeometry::checked(std::move(targets),forward.geometry->formats());
        std::swap(reverse.source_participant,reverse.destination_participant);reverse.expert=12;
        for(std::size_t i=0;i<forward.piece_observations.size();++i) {
            forward.piece_observations[i]={128+i,recordedPiece(i)};reverse.piece_observations[i]={128+i,recordedPiece(i)};
        }
        ASSERT_TRUE(forward.valid());ASSERT_TRUE(reverse.valid());
        // Complete evidence is transportable for both targets; scalar prices only describe whole owners.
        MoEOverlayMigrationProfileResult profile{.measurements={forward,reverse}};ASSERT_TRUE(profile.valid());
        if(members==1) {
            const auto scalar=scalarMoEOverlayMigrationMeasurement(forward);
            EXPECT_FALSE(scalar.geometry);EXPECT_TRUE(scalar.piece_observations.empty());
            for(std::size_t role=0;role<3;++role)EXPECT_EQ(scalar.projections[role],forward.piece_observations[role].observation);
            auto partial=forward;partial.piece_observations[0].observation.reset();
            EXPECT_THROW((void)scalarMoEOverlayMigrationMeasurement(partial),std::invalid_argument);
            auto inconsistent=forward;++inconsistent.piece_observations[0].observation->bytes;
            EXPECT_THROW((void)scalarMoEOverlayMigrationMeasurement(inconsistent),std::invalid_argument);
        } else {
            EXPECT_THROW((void)scalarMoEOverlayMigrationMeasurement(forward),std::invalid_argument);
            EXPECT_THROW((void)scalarMoEOverlayMigrationMeasurement(reverse),std::invalid_argument);
        }
    }
}
TEST(PhysicalPieceMeasurementContract, MixedFormatManifestPermutationResolvesByRole)
{
    const auto native=[](std::size_t index){const auto *metadata=native_vnni_formats::kAllSourceFormats[index].metadata;
        return ExpertWeightFormat::nativeVnni({.codebook_id=metadata->codebook_id,.is_superblock=metadata->is_superblock,.present=true});};
    MoEOverlayLayerWeightManifest layer{7,{{{ExpertTierWeightProjection::Gate,256,256,native(0)},
        {ExpertTierWeightProjection::Up,256,256,ExpertWeightFormat::floating(TensorType::FP16)},
        {ExpertTierWeightProjection::Down,256,256,native(20)}}}};
    const auto original=layer;
    do {
        ASSERT_TRUE(layer.valid());
        for(const auto &projection:original.projections)
            EXPECT_EQ(layer.projectionForRole(projection.projection).format,projection.format);
    }while(std::next_permutation(layer.projections.begin(),layer.projections.end(),[](const auto&a,const auto&b){return unsigned(a.projection)<unsigned(b.projection);}));
    layer.projections[1]=layer.projections[0];
    EXPECT_THROW((void)layer.projectionForRole(layer.projections[0].projection),std::invalid_argument);
}
}
