/**
 * @file TargetMigrationLedgerContract.inc
 * @brief Replay exact target geometry, robust samples and atomic rejection through the actual ledger.
 */
#include "utils/RecordedTargetLedgerEvidence.h"
namespace llaminar2::test {
TEST(TargetMigrationLedger, AllFormatsAndCutsRetainEveryPieceAndRobustWaveMedian) {
    std::vector<ExpertWeightFormat> formats;
    for(const auto &entry:native_vnni_formats::kAllSourceFormats)formats.push_back(ExpertWeightFormat::nativeVnni(
        {.codebook_id=entry.metadata->codebook_id,.is_superblock=entry.metadata->is_superblock,.present=true}));
    for(auto type:{TensorType::FP16,TensorType::BF16,TensorType::FP32})formats.push_back(ExpertWeightFormat::floating(type));
    std::vector<std::pair<int,int>> populations{{2,3}};
    for(int members=2;members<=8;++members){populations.emplace_back(1,members);populations.emplace_back(members,1);}
    for(const auto format:formats)for(const auto [sources,destinations]:populations) {
        const auto declaration=recordedTargetLedgerDeclaration(sources,destinations,format);
        MoEOverlayMigrationMeasurementLedger ledger({.required_target_measurements={declaration},.warmup_samples_per_coordinate=1,
            .measurement_identity="recorded-target-ledger"});
        EXPECT_EQ(ledger.requiredTargetCoordinates(),std::vector{declaration.coordinate});
        EXPECT_THROW((void)ledger.requiredCoordinates(),std::invalid_argument);
        const std::uint64_t waves[]{9999,300,100,200};
        for(auto wave:waves)ASSERT_TRUE(recordTargetLedgerRow(ledger,declaration,recordedTargetLedgerObservation(declaration,wave,17)));
        ASSERT_TRUE(ledger.ready());const auto sealed=ledger.seal();ASSERT_TRUE(sealed.valid());ASSERT_EQ(sealed.rows.size(),1u);
        const auto &row=sealed.rows[0];EXPECT_EQ(row.coordinate,declaration.coordinate);EXPECT_EQ(row.geometry,declaration.geometry);
        EXPECT_EQ(row.wave_wall_nanoseconds,200u);EXPECT_EQ(row.wave_sample_count,3u);
        ASSERT_EQ(row.pieces.size(),declaration.geometry->pieces().size());
        for(std::size_t p=0;p<row.pieces.size();++p){EXPECT_EQ(row.pieces[p].quoted_payload_bytes,100+p);EXPECT_EQ(row.pieces[p].measurement.wall_nanoseconds,17u);}
        EXPECT_THROW((void)sealed.wholeRows(),std::invalid_argument);
        const auto costs=MoEOverlayEconomyProfileComposer::normalizeMigrationMeasurements(sealed.rows);
        ASSERT_EQ(costs.size(),1u);EXPECT_EQ(costs[0].transfer_and_repack_ns,200u);
        EXPECT_THROW((void)MoEOverlayEconomyProfileComposer::checkedWholeMigrationCosts(costs),std::invalid_argument);
        EXPECT_EQ(ledger.stats().warmup_migration_observations,1u);EXPECT_EQ(ledger.stats().stored_migration_observations,3u);
        EXPECT_THROW((void)ledger.seal(),std::logic_error);
    }
}
TEST(TargetMigrationLedger, EntireWaveRejectsMissingForeignAndConflictingPiecesBeforeMutation) {
    const auto declaration=recordedTargetLedgerDeclaration(2,3,ExpertWeightFormat::floating(TensorType::FP32));
    MoEOverlayMigrationMeasurementLedger ledger({.required_target_measurements={declaration},.warmup_samples_per_coordinate=0,
        .measurement_identity="target-rejections"});
    const auto valid=recordedTargetLedgerObservation(declaration,100,10);const auto before=ledger.stats();
    auto missing=valid;missing.piece_observations[0].observation.reset();EXPECT_FALSE(recordTargetLedgerRow(ledger,declaration,missing));
    missing=valid;missing.piece_observations.pop_back();EXPECT_FALSE(recordTargetLedgerRow(ledger,declaration,missing));
    auto wrong=valid;++wrong.piece_observations[0].observation->bytes;EXPECT_FALSE(recordTargetLedgerRow(ledger,declaration,wrong));
    wrong=valid;++wrong.piece_observations[0].observation->bytes;++*wrong.piece_observations[0].quoted_payload_bytes;
    EXPECT_FALSE(recordTargetLedgerRow(ledger,declaration,wrong));
    const auto foreign=recordedTargetLedgerDeclaration(2,3,ExpertWeightFormat::floating(TensorType::FP16));
    EXPECT_FALSE(recordTargetLedgerRow(ledger,declaration,recordedTargetLedgerObservation(foreign,100,10)));
    auto wrong_key=declaration;++wrong_key.coordinate.layer;EXPECT_FALSE(recordTargetLedgerRow(ledger,wrong_key,valid));
    const std::array<MoEOverlayBorrowedTargetMigrationObservation,2> duplicate{{{declaration.coordinate,valid},{declaration.coordinate,valid}}};
    EXPECT_FALSE(ledger.recordCompletedTargetWave(duplicate));
    const auto after=ledger.stats();EXPECT_EQ(after.completed_waves_observed,before.completed_waves_observed);
    EXPECT_EQ(after.migration_observations_seen,before.migration_observations_seen);EXPECT_FALSE(ledger.ready());
    ASSERT_TRUE(recordTargetLedgerRow(ledger,declaration,valid));EXPECT_EQ(ledger.stats().stored_migration_observations,1u);
}
TEST(TargetMigrationLedger, SetupCapacityAndPresealedReuseRequireOriginalGeometryAndQuotes) {
    auto declaration=recordedTargetLedgerDeclaration(1,2,ExpertWeightFormat::floating(TensorType::FP32));
    auto no_quotes=declaration;no_quotes.quoted_payload_bytes.clear();
    MoEOverlayMigrationMeasurementLedger recorder({.required_target_measurements={no_quotes},.warmup_samples_per_coordinate=0,.measurement_identity="sampled"});
    for(int i=0;i<3;++i)ASSERT_TRUE(recordTargetLedgerRow(recorder,declaration,recordedTargetLedgerObservation(declaration,100,10)));
    const auto sealed=recorder.seal();
    MoEOverlayMigrationMeasurementLedger unquoted({.required_target_measurements={no_quotes},.measurement_identity="unquoted"});
    std::string error;EXPECT_FALSE(unquoted.matchesDeclarations(sealed,&error));EXPECT_NE(error.find("setup-bound"),std::string::npos);
    MoEOverlayMigrationMeasurementLedger exact({.required_target_measurements={declaration},.measurement_identity="exact"});
    EXPECT_TRUE(exact.matchesDeclarations(sealed));
    auto conflict=sealed;++conflict.rows[0].pieces[0].quoted_payload_bytes;++conflict.rows[0].pieces[0].measurement.bytes;
    ASSERT_TRUE(conflict.valid());EXPECT_FALSE(exact.matchesDeclarations(conflict));
    auto changed=sealed;auto targets=declaration.geometry->targets();
    for(int role=0;role<3;++role){auto &a=targets.second.members[0].projections[role];auto &b=targets.second.members[1].projections[role];
        if(role==2){a.columns-=32;b.first_column-=32;b.columns+=32;}else{a.rows-=32;b.first_row-=32;b.rows+=32;}}
    changed.rows[0].geometry=MoEOverlayMigrationMeasurementGeometry::checked(std::move(targets),declaration.geometry->formats());
    ASSERT_TRUE(changed.valid());EXPECT_FALSE(exact.matchesDeclarations(changed));
    auto missing=declaration;missing.geometry.reset();
    EXPECT_THROW((MoEOverlayMigrationMeasurementLedger({.required_target_measurements={missing},.measurement_identity="missing"})),std::invalid_argument);
    auto partial=declaration;partial.quoted_payload_bytes.pop_back();
    EXPECT_THROW((MoEOverlayMigrationMeasurementLedger({.required_target_measurements={partial},.measurement_identity="partial"})),std::invalid_argument);
    EXPECT_THROW((MoEOverlayMigrationMeasurementLedger({.required_target_measurements={declaration,declaration},.measurement_identity="duplicate"})),std::invalid_argument);
    EXPECT_THROW((MoEOverlayMigrationMeasurementLedger({.required_target_measurements={declaration},.warmup_samples_per_coordinate=UINT64_MAX,.measurement_identity="overflow"})),std::invalid_argument);
    EXPECT_THROW((MoEOverlayMigrationMeasurementLedger({.required_target_measurements={declaration},.warmup_samples_per_coordinate=0,
        .measured_samples_per_coordinate=UINT64_MAX,.measurement_identity="sample-overflow"})),std::overflow_error);
}
TEST(TargetMigrationLedger, GlobalLayerKeysAndSignedTargetOrderRemainDistinct) {
    const auto base=recordedTargetLedgerDeclaration(1,2,ExpertWeightFormat::floating(TensorType::BF16),32);
    auto later=base;later.coordinate.layer=40;
    auto reverse=base;std::swap(reverse.coordinate.source_target,reverse.coordinate.destination_target);
    auto targets=base.geometry->targets();std::swap(targets.first,targets.second);
    reverse.geometry=MoEOverlayMigrationMeasurementGeometry::checked(std::move(targets),base.geometry->formats());
    MoEOverlayMigrationMeasurementLedger ledger({.required_target_measurements={later,base,reverse},.warmup_samples_per_coordinate=0,.measurement_identity="layers"});
    EXPECT_EQ(ledger.requiredTargetCoordinates().front(),reverse.coordinate);
    for(const auto &d:{base,reverse,later})for(int i=0;i<3;++i)ASSERT_TRUE(recordTargetLedgerRow(ledger,d,recordedTargetLedgerObservation(d,100,10)));
    ASSERT_TRUE(ledger.ready());const auto sealed=ledger.seal();EXPECT_TRUE(ledger.matchesDeclarations(sealed));
}
}
