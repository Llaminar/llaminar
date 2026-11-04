/**
 * @file MoEOverlayMigrationEvidenceContext.h
 * @brief Checked local measurement geometry for one prepared migration wave.
 *
 * This immutable metadata is not a source, preparation, readiness or publication
 * permit. Production retains the actual prepared wave geometry; recorded tests
 * may explicitly describe metadata but cannot acquire any physical authority.
 */
#pragma once
#include "MoEOverlayTierMigrationTransport.h"
#include <stdexcept>
#include <limits>
namespace llaminar2 {
/** @brief One transaction-ordered, observation-free migration identity. */
struct MoEOverlayMigrationEvidenceIdentity {
    std::uint64_t expected_epoch=0, candidate_epoch=0;
    int source_participant=-1, destination_participant=-1, layer=-1, expert=-1;
    std::shared_ptr<const MoEOverlayMigrationMeasurementGeometry> geometry;
};
/** @brief Original local checked geometry used to authenticate received ordinals. */
class MoEOverlayMigrationEvidenceContext final {
public:
    /**
     * @brief Check explicit measurement metadata; this grants no physical permit.
     * @param identities Exact transaction order and original prepared geometry owners.
     * @param purpose Original transaction purpose; no execution authority is created.
     * @return Immutable checked metadata owner.
     * @throws std::invalid_argument For missing geometry, duplicate experts or inconsistent epochs.
     */
    static std::shared_ptr<const MoEOverlayMigrationEvidenceContext> checked(
        std::vector<MoEOverlayMigrationEvidenceIdentity> identities,
        MoEOverlayResidencyTransactionPurpose purpose) {
        if(identities.empty()) throw std::invalid_argument("Migration evidence requires original wave context");
        for(std::size_t i=0;i<identities.size();++i) {
            const auto &v=identities[i];
            if(!v.geometry || v.layer<0 || v.expert<0 || !v.candidate_epoch ||
               v.expected_epoch==0 || v.expected_epoch==std::numeric_limits<std::uint64_t>::max() || v.candidate_epoch!=v.expected_epoch+1 || v.geometry->pieces().empty() ||
               v.expected_epoch!=identities[0].expected_epoch || v.candidate_epoch!=identities[0].candidate_epoch)
                throw std::invalid_argument("Migration evidence context has invalid original geometry or epochs");
            const auto &targets=v.geometry->targets();
            for(const auto *target:{&targets.first,&targets.second})for(const auto &member:target->members)
                if(member.world_rank<0)throw std::invalid_argument("Migration evidence requires original resolved observer ranks");
            const auto owner=[](const auto &t){return t.members.size()==1?t.members.front().participant_id:-1;};
            if(v.source_participant!=owner(targets.first) || v.destination_participant!=owner(targets.second))
                throw std::invalid_argument("Migration evidence context target identity differs from original geometry");
            for(std::size_t j=0;j<i;++j) if(v.layer==identities[j].layer && v.expert==identities[j].expert)
                throw std::invalid_argument("Migration evidence context repeats logical expert");
        }
        return std::shared_ptr<const MoEOverlayMigrationEvidenceContext>(new MoEOverlayMigrationEvidenceContext(std::move(identities),purpose));
    }
    /** @return Original transaction purpose, without granting permission to execute it. */
    auto purpose() const noexcept {return purpose_;}
    /** @return Exact original transaction identities; never sender-authored layouts. */
    const auto &identities() const noexcept {return identities_;}
    /** @param rows Completed local or decoded rows. @return Exact checked context identity. */
    bool matches(const std::vector<MoEOverlayCompletedMigrationMeasurement> &rows) const noexcept {
        if(rows.size()!=identities_.size())return false;
        for(std::size_t i=0;i<rows.size();++i){const auto &a=rows[i];const auto &b=identities_[i];
            if(!a.valid() || a.expected_epoch!=b.expected_epoch || a.candidate_epoch!=b.candidate_epoch ||
               a.source_participant!=b.source_participant || a.destination_participant!=b.destination_participant ||
               a.layer!=b.layer || a.expert!=b.expert || !a.geometry ||
               !a.geometry->sameIdentity(*b.geometry))return false;
        }
        return true;
    }
private:
    /** @brief Retain already checked metadata without retaining physical engines.
     * @param identities Checked transaction-ordered original metadata. @param purpose Original transaction purpose. */
    explicit MoEOverlayMigrationEvidenceContext(std::vector<MoEOverlayMigrationEvidenceIdentity> identities,MoEOverlayResidencyTransactionPurpose purpose)
        :identities_(std::move(identities)),purpose_(purpose){}
    std::vector<MoEOverlayMigrationEvidenceIdentity> identities_;
    MoEOverlayResidencyTransactionPurpose purpose_;
};
/**
 * @brief Borrow a checked singleton's samples into a scalar value without allocation.
 * @param row Original complete evidence. @param output Empty scalar result, overwritten only on success.
 * @return False for cohorts, incomplete samples or invalid identity; no ledger mutation occurs.
 */
inline bool projectScalarMoEOverlayMigrationMeasurement(
    const MoEOverlayCompletedMigrationMeasurement &row,MoEOverlayCompletedMigrationMeasurement &output) noexcept {
    if(!row.valid())return false;
    std::array<std::optional<ExpertTierProjectionTransferMeasurement>,3> samples=row.projections;
    if(row.geometry) {
        const auto &targets=row.geometry->targets();
        if(targets.first.members.size()!=1 || targets.second.members.size()!=1 ||
           row.geometry->pieces().size()!=3 || row.source_participant<0 || row.destination_participant<0)return false;
        std::array<bool,3> seen{};
        for(std::size_t i=0;i<row.piece_observations.size();++i) {
            const auto role=static_cast<std::size_t>(row.geometry->pieces()[i].projection);
            const auto &slot=row.piece_observations[i];
            if(role>=3 || seen[role] || !slot.valid() || !slot.observation)return false;
            seen[role]=true;samples[role]=slot.observation;
        }
    }
    output={.expected_epoch=row.expected_epoch,.candidate_epoch=row.candidate_epoch,
        .source_participant=row.source_participant,.destination_participant=row.destination_participant,
        .layer=row.layer,.expert=row.expert,.projections=std::move(samples),.wave_wall_nanoseconds=row.wave_wall_nanoseconds};
    return true;
}
/**
 * @brief Project a checked singleton at the cold scalar profile-consumer boundary.
 * @param row Complete described whole migration, or direct scalar ledger row.
 * @return Scalar three-role observation, without copying piece vectors.
 * @throws std::invalid_argument For cohort or incomplete/noncanonical samples.
 */
inline MoEOverlayCompletedMigrationMeasurement scalarMoEOverlayMigrationMeasurement(
    const MoEOverlayCompletedMigrationMeasurement &row) {
    MoEOverlayCompletedMigrationMeasurement result;
    if(!projectScalarMoEOverlayMigrationMeasurement(row,result))
        throw std::invalid_argument("Complete-target cohort or incomplete evidence cannot be projected into scalar prices");
    return result;
}
}
