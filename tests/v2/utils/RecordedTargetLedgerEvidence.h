/**
 * @file RecordedTargetLedgerEvidence.h
 * @brief Explicit recorded geometry and operation-quote metadata for ledger replay.
 *
 * These fixtures own no engine, model source permit, readiness or performance
 * certification. Native transfer qualification remains separate.
 */
#pragma once
#include "utils/RecordedMigrationEvidence.h"
#include "execution/moe/MoEOverlayMigrationMeasurementLedger.h"
namespace llaminar2::test {
/** @brief Declare exact original pieces and independently recorded command quotes.
 * @param sources Original source member count. @param destinations Original destination member count.
 * @param format Recorded source format. @param layer Global model layer.
 * @return Complete setup declaration, never allocation capacity or physical readiness. */
inline auto recordedTargetLedgerDeclaration(int sources,int destinations,ExpertWeightFormat format,int layer=7) {
    auto a=recordedMigrationTarget(sources,0,0,0),b=recordedMigrationTarget(destinations,8,1,1);
    const auto geometry=MoEOverlayMigrationMeasurementGeometry::checked({a,b},{format,format,format});
    MoEOverlayRequiredTargetMigrationMeasurement result{{sources==1?MoEOverlayRouteTarget::whole(0):MoEOverlayRouteTarget::cpuCohort(0),
        destinations==1?MoEOverlayRouteTarget::whole(8):MoEOverlayRouteTarget::cpuCohort(1),layer},geometry,{}};
    for(std::size_t p=0;p<geometry->pieces().size();++p)result.quoted_payload_bytes.push_back(100+p);
    return result;
}
/** @brief Replay all actual canonical piece samples with quoted bytes unchanged.
 * @param declaration Original checked setup declaration. @param wave Recorded whole-wave duration.
 * @param piece_wall Recorded piece duration, identical across pieces for this fixture.
 * @return Complete recorded observation; no physical source event is fabricated. */
inline auto recordedTargetLedgerObservation(const MoEOverlayRequiredTargetMigrationMeasurement &declaration,
    std::uint64_t wave,std::uint64_t piece_wall) {
    const auto &targets=declaration.geometry->targets();
    MoEOverlayCompletedMigrationMeasurement row{.expected_epoch=1,.candidate_epoch=2,
        .source_participant=targets.first.members.size()==1?targets.first.members[0].participant_id:-1,
        .destination_participant=targets.second.members.size()==1?targets.second.members[0].participant_id:-1,
        .layer=declaration.coordinate.layer,.expert=11,.wave_wall_nanoseconds=wave,.geometry=declaration.geometry};
    for(std::size_t p=0;p<declaration.geometry->pieces().size();++p) {
        const auto quote=declaration.quoted_payload_bytes.empty()?100+p:declaration.quoted_payload_bytes[p];
        row.piece_observations.push_back({quote,ExpertTierProjectionTransferMeasurement{.sequence=1,.bytes=quote,
            .wall_nanoseconds=piece_wall,.host_nanoseconds=1}});
    }
    return row;
}
/** @brief Record one borrowed row without constructing a dynamic view vector.
 * @param ledger Existing single ledger. @param declaration Original coordinate.
 * @param row Borrowed actual recorded observation. @param error Optional rejection diagnostic.
 * @return Atomic acceptance result. */
inline bool recordTargetLedgerRow(MoEOverlayMigrationMeasurementLedger &ledger,
    const MoEOverlayRequiredTargetMigrationMeasurement &declaration,const MoEOverlayCompletedMigrationMeasurement &row,
    std::string *error=nullptr) {
    const std::array<MoEOverlayBorrowedTargetMigrationObservation,1> view{{{declaration.coordinate,row}}};
    return ledger.recordCompletedTargetWave(view,error);
}
}
