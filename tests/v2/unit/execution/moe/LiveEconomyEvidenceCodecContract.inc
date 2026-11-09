/**
 * @file LiveEconomyEvidenceCodecContract.inc
 * @brief Actual live codec replay over explicitly recorded original geometry.
 *
 * These fixtures certify protocol identity/extents and no physical timing or
 * source readiness. Native MPI integration separately proves real collectives.
 */
#include "execution/moe/MoEOverlayEconomyEvidenceCodec.h"
#include "execution/moe/MoEOverlayEconomyEvidenceExchange.h"
#include "utils/RecordedMigrationEvidence.h"
namespace llaminar2::test {
namespace {
/** @brief Recorded whole/cohort reciprocal wave with distinct actual endpoint ranks. */
auto liveCodecContext(int members,ExpertWeightFormat format=ExpertWeightFormat::floating(TensorType::FP32)) {
    auto a=recordedMigrationTarget(1,0,0,0),b=recordedMigrationTarget(members,8,1,1);
    for(int i=0;i<members;++i)b.members[i].world_rank=i+1;
    std::array<ExpertWeightFormat,3> formats;formats.fill(format);
    const auto forward=MoEOverlayMigrationMeasurementGeometry::checked({a,b},formats);
    const auto reverse=MoEOverlayMigrationMeasurementGeometry::checked({b,a},formats);
    return MoEOverlayMigrationEvidenceContext::checked({{1,2,0,members==1?8:-1,7,11,forward},
        {1,2,members==1?8:-1,0,7,12,reverse}},MoEOverlayResidencyTransactionPurpose::EconomyCalibration);
}
/** @brief Replay exactly the original Fabric endpoint observation rule. */
auto liveCodecRows(const MoEOverlayMigrationEvidenceContext &context,int rank) {
    auto rows=MoEOverlayEconomyEvidenceCodec::decode(context,{},99,101);
    for(auto &row:rows)for(std::size_t p=0;p<row.geometry->pieces().size();++p){
        const auto &piece=row.geometry->pieces()[p];bool owns=false;
        for(const auto *target:{&row.geometry->targets().first,&row.geometry->targets().second})for(const auto &member:target->members)
            if(member.world_rank==rank && (member.participant_id==piece.source_participant || member.participant_id==piece.destination_participant))owns=true;
        if(owns)row.piece_observations[p]={.quoted_payload_bytes=128+p,
            .observation=ExpertTierProjectionTransferMeasurement{.sequence=3,.bytes=128+p,.wall_nanoseconds=51+p,.host_nanoseconds=21+p}};
    }
    return rows;
}
}
TEST(LiveEconomyEvidenceCodec, ExactIncidenceAllFormatsAndLargeSmallEmptyReplay) {
    for(const auto &entry:native_quantised_formats::kAllSourceFormats)for(int members:{8,3,1,2}) {
        const auto format=ExpertWeightFormat::nativeQuantised({.codebook_id=entry.metadata->codebook_id,
            .is_superblock=entry.metadata->is_superblock,.present=true});
        const auto context=liveCodecContext(members,format);
        std::vector<std::vector<MoEOverlayCompletedMigrationMeasurement>> ranks;
        for(int rank=0;rank<=members+1;++rank) {
            const auto original=liveCodecRows(*context,rank);
            const auto payload=MoEOverlayEconomyEvidenceCodec::encode(*context,original,rank);
            EXPECT_EQ(payload.size(),MoEOverlayEconomyEvidenceCodec::maximumEncodedBytes(*context,rank));
            if(rank>members)EXPECT_TRUE(payload.empty());
            const auto decoded=MoEOverlayEconomyEvidenceCodec::decode(*context,payload,rank,101);
            ASSERT_TRUE(context->matches(decoded));
            for(std::size_t m=0;m<decoded.size();++m)for(std::size_t p=0;p<decoded[m].piece_observations.size();++p)
                EXPECT_EQ(decoded[m].piece_observations[p].observation,original[m].piece_observations[p].observation);
            ranks.push_back(decoded);
        }
        const auto merged=MoEOverlayMigrationMeasurementMerger::merge(ranks);
        for(const auto &row:merged)for(const auto &slot:row.piece_observations)ASSERT_TRUE(slot.observation);
    }
}
TEST(LiveEconomyEvidenceCodec, EndpointOmissionCannotBeHiddenByOtherEndpoint) {
    const auto context=liveCodecContext(3);const auto valid=liveCodecRows(*context,0);
    const auto payload=MoEOverlayEconomyEvidenceCodec::encode(*context,valid,0);
    auto missing=valid;missing[0].piece_observations[0]={};
    EXPECT_THROW((void)MoEOverlayEconomyEvidenceCodec::encode(*context,missing,0),std::invalid_argument);
    missing=valid;missing[0].piece_observations[0].observation.reset();
    EXPECT_THROW((void)MoEOverlayEconomyEvidenceCodec::encode(*context,missing,0),std::invalid_argument);
    // Another endpoint has a real duplicate; it does not authorize rank0 to omit its own evidence.
    const auto other=liveCodecRows(*context,1);ASSERT_TRUE(other[0].piece_observations[0].observation);
    EXPECT_THROW((void)MoEOverlayEconomyEvidenceCodec::decode(*context,{},0,101),std::invalid_argument);
    EXPECT_THROW((void)MoEOverlayEconomyEvidenceCodec::decode(*context,payload,9,101),std::invalid_argument);
}
TEST(LiveEconomyEvidenceCodec, TruncatedDuplicateForeignAndQuoteMismatchReject) {
    const auto context=liveCodecContext(2);auto rows=liveCodecRows(*context,0);
    const auto payload=MoEOverlayEconomyEvidenceCodec::encode(*context,rows,0);
    auto truncated=payload;truncated.pop_back();
    EXPECT_THROW((void)MoEOverlayEconomyEvidenceCodec::decode(*context,truncated,0,101),std::invalid_argument);
    auto duplicate=payload;duplicate.insert(duplicate.end(),payload.begin(),payload.end());
    EXPECT_THROW((void)MoEOverlayEconomyEvidenceCodec::decode(*context,duplicate,0,101),std::invalid_argument);
    ++rows[0].piece_observations[0].observation->bytes;
    EXPECT_THROW((void)MoEOverlayEconomyEvidenceCodec::encode(*context,rows,0),std::invalid_argument);
    rows=liveCodecRows(*context,0);++rows[0].expert;
    EXPECT_THROW((void)MoEOverlayEconomyEvidenceCodec::encode(*context,rows,0),std::invalid_argument);
    const auto different=liveCodecContext(2,ExpertWeightFormat::floating(TensorType::FP16));
    EXPECT_THROW((void)MoEOverlayEconomyEvidenceCodec::decode(*different,payload,0,101),std::invalid_argument);
    const auto smaller=liveCodecContext(1);
    EXPECT_THROW((void)MoEOverlayEconomyEvidenceCodec::decode(*smaller,payload,0,101),std::invalid_argument);
}
TEST(LiveEconomyEvidenceCodec, MandatoryContextAndCoordinateRemainBound) {
    const auto context=liveCodecContext(1);auto rows=liveCodecRows(*context,0);
    MoEOverlayMigrationProfileEvidence evidence{.profile_sequence=3,.coordinate=recordedWholeMigrationCoordinate(0,8,7),.local_measurements=rows,.context=context};
    ASSERT_TRUE(evidence.valid());evidence.context.reset();EXPECT_FALSE(evidence.valid());
    evidence.context=context;evidence.coordinate=recordedWholeMigrationCoordinate(3,4,7);EXPECT_FALSE(evidence.valid());
    const auto scalar=scalarMoEOverlayMigrationMeasurement(rows[0]);
    EXPECT_FALSE(scalar.geometry);for(const auto &sample:scalar.projections)EXPECT_TRUE(sample);
    const auto cohort=liveCodecContext(2);
    EXPECT_THROW((void)scalarMoEOverlayMigrationMeasurement(liveCodecRows(*cohort,0)[0]),std::invalid_argument);
    rows[0].piece_observations[0].observation.reset();
    EXPECT_THROW((void)scalarMoEOverlayMigrationMeasurement(rows[0]),std::invalid_argument);
}

TEST(LiveEconomyEvidenceCodec, CompleteCohortIdentityRejectsRepresentativeMemberAlias) {
    const auto original=liveCodecContext(2);
    auto identities=original->identities();
    // Complete-target ordering places the negative cohort tag before a whole owner.
    std::reverse(identities.begin(),identities.end());
    const auto context=MoEOverlayMigrationEvidenceContext::checked(
        std::move(identities),MoEOverlayResidencyTransactionPurpose::EconomyCalibration);
    auto rows=liveCodecRows(*context,0);
    MoEOverlayMigrationProfileEvidence evidence{
        .profile_sequence=3,
        .coordinate={MoEOverlayRouteTarget::cpuCohort(1),MoEOverlayRouteTarget::whole(0),7},
        .local_measurements=rows,.context=context};
    ASSERT_TRUE(evidence.valid());
    evidence.coordinate.source_target=MoEOverlayRouteTarget::cpuCohort(2);
    EXPECT_FALSE(evidence.valid());
    // The previous member-based coordinate check admitted this exact alias.
    evidence.context=original;evidence.local_measurements=liveCodecRows(*original,0);
    evidence.coordinate=recordedWholeMigrationCoordinate(0,8,7);
    EXPECT_FALSE(evidence.valid());
    evidence.coordinate=recordedWholeMigrationCoordinate(0,9,7);
    EXPECT_FALSE(evidence.valid());
}
}
