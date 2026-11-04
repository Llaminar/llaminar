/**
 * @file RecordedMigrationEvidence.h
 * @brief Explicit passive geometry for recorded economy protocol fixtures.
 *
 * These declarations create no engine, source permit, readiness or measured
 * physical performance. Quotes are recorded fixture evidence only. Actual
 * fabric integration owns physical operation provenance and event/ACK proof.
 */
#pragma once
#include "execution/moe/MoEOverlayMigrationEvidenceContext.h"
#include "execution/moe/MoEOverlayMigrationTargetCoordinate.h"
#include "execution/moe/ExpertTierWeightStream.h"
namespace llaminar2::test {
/**
 * @brief Declare a whole-owner protocol coordinate without a cohort alias.
 * @param source Original whole participant. @param destination Original whole participant.
 * @param layer Original model-global layer. @return Typed recorded whole-target identity.
 */
inline auto recordedWholeMigrationCoordinate(int source, int destination, int layer) {
    return MoEOverlayMigrationTargetCoordinate{
        MoEOverlayRouteTarget::whole(source), MoEOverlayRouteTarget::whole(destination), layer};
}
/** @brief Build a complete recorded target using equal original cuts.
 * @param members Participant count. @param first_id First canonical participant ID.
 * @param tier Recorded tier index. @param rank Actual observer rank assigned to this target.
 * @return Complete passive original geometry. */
inline MoEOverlayPhysicalProjectionTarget recordedMigrationTarget(int members,int first_id,int tier,int rank) {
    MoEOverlayPhysicalProjectionTarget target{.tier_idx=tier};
    for(int i=0;i<members;++i) {
        const int width=215040/members,first=i*width;
        const auto rows=members==1?MoEProjectionResidence::ExpertOwner:MoEProjectionResidence::ParticipantOutputSlice;
        const auto columns=members==1?MoEProjectionResidence::ExpertOwner:MoEProjectionResidence::ParticipantInputSlice;
        target.members.push_back({.participant_id=first_id+i,.world_rank=rank,.device=DeviceId::cpu(),
            .address=GlobalDeviceAddress::cpu(i),.projections={{{rows,215040,256,first,width,0,256},
                {rows,215040,256,first,width,0,256},{columns,256,215040,0,256,first,width}}}});
    }
    return target;
}
/** @brief Checked recorded whole geometry for legacy scalar-sample fixture migration.
 * @param source Source participant. @param destination Destination participant.
 * @param source_rank Actual source observer. @param destination_rank Actual destination observer.
 * @return Passive whole-target geometry, without a physical source claim. */
inline auto recordedWholeMigrationGeometry(int source,int destination,int source_rank,int destination_rank) {
    std::array<ExpertWeightFormat,3> formats;formats.fill(ExpertWeightFormat::floating(TensorType::FP32));
    auto a=recordedMigrationTarget(1,source,source,source_rank),b=recordedMigrationTarget(1,destination,destination,destination_rank);
    a.members[0].address=b.members[0].address=GlobalDeviceAddress::cpu();
    return MoEOverlayMigrationMeasurementGeometry::checked({a,b},formats);
}
/** @brief Attach explicit whole geometry and independently recorded quotes to scalar fixture samples.
 * @param rows Recorded transaction-ordered rows, converted in place before protocol use.
 * @param inventory Original exact MPI inventory when these records enter a real collective;
 *        absent only for standalone recorded metadata tests with explicitly participant-indexed ranks.
 * @return Checked measurement-only context. */
inline auto attachRecordedMigrationContext(std::vector<MoEOverlayCompletedMigrationMeasurement> &rows,
    const MoEExpertOwnerMap *inventory=nullptr) {
    std::vector<MoEOverlayMigrationEvidenceIdentity> identities;
    for(auto &row:rows) {
        row.geometry=recordedWholeMigrationGeometry(row.source_participant,row.destination_participant,
            row.source_participant,row.destination_participant);
        if(inventory) {
            auto targets=row.geometry->targets();
            for(auto *target:{&targets.first,&targets.second}) {
                auto &member=target->members.front();
                const auto *original=inventory->participantForId(member.participant_id);
                if(!original || !original->world_rank_known)
                    throw std::invalid_argument("Recorded MPI evidence requires an original resolved inventory member");
                target->tier_idx=original->tier_idx;
                member.world_rank=original->world_rank;member.device=original->device;member.address=original->address;
            }
            row.geometry=MoEOverlayMigrationMeasurementGeometry::checked(std::move(targets),row.geometry->formats());
        }
        row.piece_observations.resize(3);
        for(std::size_t role=0;role<3;++role) {
            auto &slot=row.piece_observations[role];slot.observation=row.projections[role];
            if(slot.observation)slot.quoted_payload_bytes=slot.observation->bytes;
        }
        row.projections={};
        identities.push_back({row.expected_epoch,row.candidate_epoch,row.source_participant,
            row.destination_participant,row.layer,row.expert,row.geometry});
    }
    return MoEOverlayMigrationEvidenceContext::checked(std::move(identities),MoEOverlayResidencyTransactionPurpose::EconomyCalibration);
}
}
