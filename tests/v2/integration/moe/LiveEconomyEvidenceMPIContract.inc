/**
 * @file LiveEconomyEvidenceMPIContract.inc
 * @brief Real two-rank count/payload collectives over explicit recorded metadata.
 *
 * No physical transfer timing or prepared-source permit is minted here. These
 * cases exercise the actual MPI lane, counts, payload reuse and empty observer.
 */
namespace llaminar2::test {
namespace {
/** @brief Retain four explicitly declared endpoint identities over two actual ranks. */
MoEExpertOwnerMap liveEconomyInventory() {
    auto plan=distributedCpuOverlayPlan();plan.domains.clear();plan.routed_tiers.clear();
    for(int i=0;i<4;++i) {
        const auto name="evidence_"+std::to_string(i);
        plan.domains.push_back(domain(name,GlobalDeviceAddress::cpu(i%2),i/2,CollectiveBackendType::MPI));
        plan.routed_tiers.push_back(tier(name,name,i,1,i==3));
    }
    plan.continuation_domain=plan.shared_expert_domain=plan.base_model_domain=plan.domains.front().name;
    plan.placements={{.layer=0,.routed_expert_tier={0,1,2,3}}};
    return MoEExpertOwnerMap::build(plan);
}
/** @brief Recorded reciprocal whole/cohort metadata with real observer rank identities. */
auto liveMPIContext(int members,bool empty_rank_one) {
    auto a=recordedMigrationTarget(1,0,0,0);
    auto b=recordedMigrationTarget(members,empty_rank_one?1:2,empty_rank_one?1:2,empty_rank_one?0:1);
    if(empty_rank_one){b.members[0].device=DeviceId::cpu();b.members[0].address=GlobalDeviceAddress::cpu(1);}
    if(members>1)std::swap(a,b); // Same canonical typed-target order used by calibration.
    const auto format=ExpertWeightFormat::floating(TensorType::FP32);
    const auto forward=MoEOverlayMigrationMeasurementGeometry::checked({a,b},{format,format,format});
    const auto reverse=MoEOverlayMigrationMeasurementGeometry::checked({b,a},{format,format,format});
    const int source=a.members.size()==1?a.members.front().participant_id:-1;
    const int destination=b.members.size()==1?b.members.front().participant_id:-1;
    return MoEOverlayMigrationEvidenceContext::checked({{1,2,source,destination,0,0,forward},{1,2,destination,source,0,1,reverse}},
        MoEOverlayResidencyTransactionPurpose::EconomyCalibration);
}
/** @brief Supply all and only the original endpoint's recorded observations. */
auto liveMPIRows(const MoEOverlayMigrationEvidenceContext &context,int rank) {
    auto rows=MoEOverlayEconomyEvidenceCodec::decode(context,{},99,901);
    for(auto &row:rows)for(std::size_t p=0;p<row.geometry->pieces().size();++p){
        const auto &piece=row.geometry->pieces()[p];bool owns=false;
        for(const auto *target:{&row.geometry->targets().first,&row.geometry->targets().second})for(const auto &member:target->members)
            if(member.world_rank==rank && (member.participant_id==piece.source_participant || member.participant_id==piece.destination_participant))owns=true;
        if(owns)row.piece_observations[p]={.quoted_payload_bytes=8192+p,
            .observation=ExpertTierProjectionTransferMeasurement{.sequence=31,.bytes=8192+p,.wall_nanoseconds=301,.host_nanoseconds=251}};
    }
    return rows;
}
/** @brief Poll the existing nonblocking profile lane to completion. */
auto finishLiveProfile(MoEOverlayMPIEconomyEvidenceExchange &exchange) {
    MoEOverlayMigrationProfileResult result;std::string error;
    for(;;){const auto progress=exchange.pollMigrationProfile(&result,&error);
        if(progress==MoEOverlayResidencyWaveProgress::Ready)return result;
        EXPECT_EQ(progress,MoEOverlayResidencyWaveProgress::Pending)<<error;
        if(progress==MoEOverlayResidencyWaveProgress::Failed)return result;
        std::this_thread::yield();}
}
}
TEST(Test__MoEOverlayMPIResidencyConsensus, LiveEconomyEvidenceCountsPayloadAndEmptyReuse) {
    auto mpi=worldContext();if(!requireTwoRanks(*mpi))GTEST_SKIP()<<"Live evidence protocol requires two ranks";
    const auto inventory=liveEconomyInventory();
    MoEOverlayMPIEconomyEvidenceExchange exchange({.mpi_context=mpi,.owner_map=inventory,
        .production_topology=ExpertHistogramProductionTopology::forRetainedExecution(1,1,ExpertHistogramServingRegime::PositiveDepthMTP),
        .perf_device="mpi_live_recorded_evidence"});
    std::uint64_t sequence=71;
    for(const auto [members,empty]:{std::pair{2,false},std::pair{1,false},std::pair{1,true},std::pair{2,false}}) {
        const auto context=liveMPIContext(members,empty);const auto rows=liveMPIRows(*context,mpi->rank());
        const auto local_bytes=MoEOverlayEconomyEvidenceCodec::encode(*context,rows,mpi->rank()).size();
        std::size_t global_bytes=0;for(int rank=0;rank<2;++rank)global_bytes+=MoEOverlayEconomyEvidenceCodec::maximumEncodedBytes(*context,rank);
        if(empty&&mpi->rank()==1)EXPECT_EQ(local_bytes,0u);
        const auto before=exchange.stats();
        const auto coordinate=members==1
            ? recordedWholeMigrationCoordinate(0,empty?1:2,0)
            : MoEOverlayMigrationTargetCoordinate{
                MoEOverlayRouteTarget::cpuCohort(2),MoEOverlayRouteTarget::whole(0),0};
        MoEOverlayMigrationProfileEvidence profile{.profile_sequence=sequence++,
            .coordinate=coordinate,.local_measurements=rows,.context=context};
        std::string error;ASSERT_TRUE(exchange.beginMigrationProfile(profile,&error))<<error;
        const auto merged=finishLiveProfile(exchange);ASSERT_TRUE(merged.valid());ASSERT_TRUE(exchange.idle());
        const auto after=exchange.stats();
        EXPECT_GT(after.evidence_count_bytes_sent,before.evidence_count_bytes_sent);
        EXPECT_EQ(after.evidence_payload_bytes_sent-before.evidence_payload_bytes_sent,local_bytes);
        EXPECT_EQ(after.evidence_payload_bytes_received-before.evidence_payload_bytes_received,global_bytes);
        EXPECT_EQ(after.evidence_payload_collectives_started-before.evidence_payload_collectives_started,1u);
        for(const auto &row:merged.measurements)for(const auto &slot:row.piece_observations)ASSERT_TRUE(slot.observation);
        auto attempt=economyAttempt(mpi->rank(),true,inventory);attempt.calibration_sequence=sequence++;
        attempt.coordinate=profile.coordinate;attempt.local_measurements=rows;attempt.context=context;
        const auto start=exchange.stats();ASSERT_TRUE(exchange.beginAttempt(attempt,&error))<<error;
        const auto result=finishEconomyAttempt(exchange);ASSERT_TRUE(result.valid());ASSERT_TRUE(result.accepted);ASSERT_TRUE(exchange.idle());
        EXPECT_EQ(exchange.stats().evidence_payload_bytes_sent-start.evidence_payload_bytes_sent,local_bytes);
        EXPECT_EQ(exchange.stats().evidence_payload_bytes_received-start.evidence_payload_bytes_received,global_bytes);
    }
}
}
