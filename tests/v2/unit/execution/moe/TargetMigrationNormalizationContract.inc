/**
 * @file TargetMigrationNormalizationContract.inc
 * @brief Preserve actual whole-wave max-of-medians with complete physical piece evidence.
 */
#include "utils/RecordedTargetLedgerEvidence.h"
#include "execution/moe/MoEOverlayEconomyCalibrationPlanner.h"
namespace llaminar2::test {
TEST(TargetMigrationNormalization, CompleteTargetIdentityRejectsSingletonAndMemberAliases) {
    const auto format=ExpertWeightFormat::floating(TensorType::FP32);
    const auto whole=recordedTargetLedgerDeclaration(1,1,format);
    ASSERT_TRUE(MoEOverlayTargetLayerMigrationMeasurement::geometryMatches(whole.coordinate,*whole.geometry));
    auto alias=whole;
    alias.coordinate.source_target=MoEOverlayRouteTarget::cpuCohort(0);
    EXPECT_FALSE(MoEOverlayTargetLayerMigrationMeasurement::geometryMatches(alias.coordinate,*alias.geometry));
    EXPECT_THROW((MoEOverlayMigrationMeasurementLedger({.required_target_measurements={alias},
        .measurement_identity="singleton-cohort-alias"})),std::invalid_argument);
    for(int members=2;members<=8;++members) {
        const auto cohort=recordedTargetLedgerDeclaration(members,1,format);
        ASSERT_TRUE(MoEOverlayTargetLayerMigrationMeasurement::geometryMatches(cohort.coordinate,*cohort.geometry));
        auto member_alias=cohort;
        member_alias.coordinate.source_target=MoEOverlayRouteTarget::whole(0);
        EXPECT_FALSE(MoEOverlayTargetLayerMigrationMeasurement::geometryMatches(member_alias.coordinate,*member_alias.geometry));
        EXPECT_THROW((MoEOverlayMigrationMeasurementLedger({.required_target_measurements={member_alias},
            .measurement_identity="cohort-member-alias"})),std::invalid_argument);
    }
}
TEST(TargetMigrationNormalization, ConcurrentPiecesUseMaxOfMediansNotSumOrMedianOfMax) {
    const auto declaration=recordedTargetLedgerDeclaration(1,2,ExpertWeightFormat::floating(TensorType::FP32));
    MoEOverlayMigrationMeasurementLedger ledger({.required_target_measurements={declaration},.warmup_samples_per_coordinate=0,.measurement_identity="max-medians"});
    for(std::size_t i=0;i<3;++i){auto row=recordedTargetLedgerObservation(declaration,10,1);
        row.piece_observations[i].observation->wall_nanoseconds=100;ASSERT_TRUE(recordTargetLedgerRow(ledger,declaration,row));}
    const auto sealed=ledger.seal();const auto prices=MoEOverlayEconomyProfileComposer::normalizeMigrationMeasurements(sealed.rows);
    ASSERT_EQ(prices.size(),1u);EXPECT_EQ(prices[0].transfer_and_repack_ns,10u);
    auto piece_dominates=sealed.rows;piece_dominates[0].pieces[0].measurement.wall_nanoseconds=200;
    EXPECT_EQ(MoEOverlayEconomyProfileComposer::normalizeMigrationMeasurements(piece_dominates)[0].transfer_and_repack_ns,200u);
    auto bad=piece_dominates;bad[0].pieces.pop_back();
    EXPECT_THROW((void)MoEOverlayEconomyProfileComposer::normalizeMigrationMeasurements(bad),std::invalid_argument);
    bad=piece_dominates;++bad[0].pieces[0].measurement.bytes;
    EXPECT_THROW((void)MoEOverlayEconomyProfileComposer::normalizeMigrationMeasurements(bad),std::invalid_argument);
    bad=piece_dominates;bad.push_back(bad[0]);
    EXPECT_THROW((void)MoEOverlayEconomyProfileComposer::normalizeMigrationMeasurements(bad),std::invalid_argument);
}
TEST(TargetMigrationNormalization, CheckedWholeAdaptersKeepOriginalNumericalCost) {
    MoEOverlayParticipantLayerMigrationMeasurement scalar{.source_participant=0,.destination_participant=1,.layer=7,
        .wave_wall_nanoseconds=110,.wave_sample_count=3};
    for(auto &piece:scalar.projections)piece={.sequence=1,.bytes=100,.wall_nanoseconds=120,.host_nanoseconds=1};
    const auto old=MoEOverlayEconomyProfileComposer::normalizeMigrationMeasurements(std::vector{scalar});
    const auto typed=MoEOverlayTargetLayerMigrationMeasurement::fromWhole(scalar);
    const auto prices=MoEOverlayEconomyProfileComposer::normalizeMigrationMeasurements(std::vector{typed});
    const auto projected=MoEOverlayEconomyProfileComposer::checkedWholeMigrationCosts(prices);
    ASSERT_EQ(projected.size(),1u);EXPECT_EQ(projected[0].source_participant,old[0].source_participant);
    EXPECT_EQ(projected[0].destination_participant,old[0].destination_participant);
    EXPECT_EQ(projected[0].transfer_and_repack_ns,old[0].transfer_and_repack_ns);
    EXPECT_EQ(typed.whole().projections,scalar.projections);
}
}

namespace llaminar2::test {
TEST(TargetMigrationNormalization, CatalogPreservesWholeExpansionButRejectsUnresolvedCohortCuts) {
    const auto format=ExpertWeightFormat::floating(TensorType::FP32);
    const MoEOverlayLayerWeightManifest first{32,{{{ExpertTierWeightProjection::Gate,215040,256,format},
        {ExpertTierWeightProjection::Up,215040,256,format},{ExpertTierWeightProjection::Down,256,215040,format}}}};
    auto second=first;second.layer_idx=33;
    const MoEOverlayEconomyCalibrationLayerCatalog catalog({first,second});
    for(int members:{1,2}) {
        const auto declaration=recordedTargetLedgerDeclaration(1,members,format,32);
        MoEOverlayMigrationMeasurementLedger ledger({.required_target_measurements={declaration},.warmup_samples_per_coordinate=0,.measurement_identity="expand-original"});
        for(int i=0;i<3;++i)ASSERT_TRUE(recordTargetLedgerRow(ledger,declaration,recordedTargetLedgerObservation(declaration,100,10)));
        const auto sealed=ledger.seal();
        if(members==1) {
            const auto expanded=catalog.expand(sealed);ASSERT_TRUE(expanded.valid());ASSERT_EQ(expanded.rows.size(),2u);
            EXPECT_EQ(expanded.rows[0].coordinate.layer,32);EXPECT_EQ(expanded.rows[1].coordinate.layer,33);
            EXPECT_EQ(expanded.rows[0].geometry,sealed.rows[0].geometry);
            EXPECT_EQ(expanded.rows[1].wave_wall_nanoseconds,sealed.rows[0].wave_wall_nanoseconds);
        }else EXPECT_THROW((void)catalog.expand(sealed),std::invalid_argument);
    }
}
}
