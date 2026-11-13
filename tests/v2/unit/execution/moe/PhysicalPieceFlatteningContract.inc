/**
 * @file PhysicalPieceFlatteningContract.inc
 * @brief Actual recorded-component factory preserves operation coordinates while flattening.
 *
 * The existing identity-only provider supplies no native readiness or timing.
 * This test checks ownership mapping, not transfer arithmetic or certification.
 */
#include "execution/moe/MoEOverlayPhysicalProjectionPieces.h"
namespace {
class DescribedIdentityProvider final:public IMoEOverlayParticipantTransferProvider {
public:
    /** @brief Identity-only component owns no physical source slots. */
    void retirePreviousSources(std::uint64_t, const std::vector<MoEOverlayTierMigration>&) noexcept override {}
    MoEOverlayParticipantPreparedTransfers prepareTransfers(const MoEOverlayResidencyTransaction &transaction,
        const std::vector<int>&local) override {
        IdentityTransferProvider original;
        auto result=original.prepareTransfers(transaction,local);
        for(std::size_t migration=0;migration<result.migrations.size();++migration) {
            auto &record=result.migrations[migration];const auto &edge=transaction.migrations[migration];
            const MoEExpertProjectionOwnership::Projection whole{MoEProjectionResidence::ExpertOwner,1,1,0,1,0,1};
            MoEOverlayPhysicalProjectionMember source,destination;
            source.participant_id=edge.source.owner_participant;destination.participant_id=edge.destination.owner_participant;
            source.device=destination.device=DeviceId::cpu();source.projections.fill(whole);destination.projections.fill(whole);
            record.measurement_geometry=MoEOverlayMigrationMeasurementGeometry::checked(
                std::pair{MoEOverlayPhysicalProjectionTarget{edge.source.tier_idx,{source}},
                    MoEOverlayPhysicalProjectionTarget{edge.destination.tier_idx,{destination}}},
                {ExpertWeightFormat::floating(TensorType::FP32),ExpertWeightFormat::floating(TensorType::FP32),ExpertWeightFormat::floating(TensorType::FP32)});
        }
        return result;
    }
};
}
TEST(Test__MoEOverlayParticipantMigration, FlattenedOperationsRetainOriginalMeasurementCoordinates)
{
    auto fixture=participantMigrationFixture();
    auto factory=test::MoEOverlayParticipantMigrationTestAccess::recordedComponentFactory({
        .registry=fixture.registry,.transfer_provider=std::make_shared<DescribedIdentityProvider>(),.perf_device="piece-mapping"});
    auto result=factory.prepare(fixture.transaction);
    ASSERT_EQ(result.status,MoEOverlayResidencyStageStartStatus::Started)<<result.error;
    ASSERT_TRUE(result.valid(fixture.transaction.migrations.size()*3));
    ASSERT_EQ(result.measurement_operations.size(),result.transfers.size());
    for(std::size_t i=0;i<result.transfers.size();++i) {
        EXPECT_EQ(result.measurement_operations[i].migration,i/3);
        EXPECT_EQ(result.measurement_operations[i].piece,i%3);
    }
    const auto originalCoordinates=result.measurement_operations;
    result.measurement_operations.back()=result.measurement_operations.front();
    EXPECT_FALSE(result.valid(fixture.transaction.migrations.size()*3));
    result.measurement_operations=originalCoordinates;
    result.measurement_geometries.pop_back();
    EXPECT_FALSE(result.valid(fixture.transaction.migrations.size()*3));
    for(auto &operation:result.transfers)operation->abort();
    result.inactive_bank->abort();
}
