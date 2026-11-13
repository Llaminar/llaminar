/**
 * @file CPUExpertShardSourceSelection.h
 * @brief Closed original-map selection of native source bytes for one joint CPU expert member.
 *
 * This immutable value retains the original cohort and selected logical IDs.
 * It owns no allocation or live placement state and grants no execution/native
 * readiness. Existing frozen source bindings add actual model/tensor ownership;
 * the source-bound prepared registry retains those facts through execution.
 */
#pragma once
#include "MoEExpertOwnerMap.h"
#include "MoEExpertShardCohort.h"
#include <memory>
#include <span>
#include <vector>
namespace llaminar2 {
/** @brief One original CPU joint-fragment source recipe, never a whole-expert residency claim. */
class CPUExpertShardSourceSelection final {
public:
    /**
     * @brief Derive exactly this member's selected IDs from the original canonical placement.
     * @param original Source-authenticated joint placement, not an independently apportioned map.
     * @param layer Original source layer.
     * @param participant_id Original global participant descriptor identity.
     * @return Shared immutable recipe retaining the SAME original cohort.
     * @throws std::invalid_argument For missing/foreign/non-CPU/whole-owner or inconsistent source membership.
     */
    static std::shared_ptr<const CPUExpertShardSourceSelection> fromOriginalPlacement(
        const MoEExpertOwnerMap& original,int layer,int participant_id);
    /** @return Original source layer, not a compact tensor index. */
    int layer()const noexcept{return cohort_->source().layer_idx;}
    /** @return Exact original cohort owner, retaining complete E/H/F/formats and ordered rank/NUMA members. */
    const std::shared_ptr<const MoEExpertShardCohort>& cohort()const noexcept{return cohort_;}
    /** @return Original domain coordinate used by the existing N/K projection geometry. */
    std::size_t participantIndex()const noexcept{return member_;}
    /** @return Same original global descriptor, including domain, rank, NUMA and CPU device0. */
    const MoEExpertOwnerParticipant& participant()const noexcept{return participant_;}
    /** @return Exact selected original expert IDs; source E remains in the cohort geometry. */
    std::span<const int> expertIds()const noexcept{return experts_;}
    /** @return Checked original source geometry for this member, never inferred from compact shape. */
    const MoEExpertProjectionOwnership& ownership()const noexcept{return cohort_->ownership(member_);}
    /** @return Whether an original logical expert belongs to this exact frozen source selection. */
    bool contains(int expert)const noexcept;
    /** @return Original GGUF[K,N,E] request consumed by the native loader, with presliced=false. */
    WeightSliceSpec sourceLoadSlice(WeightRole role)const{return cohort_->sourceLoadSlice(member_,role,experts_);}
    /** @return Exact post-compaction source identity, with presliced=true and original selected IDs. */
    WeightSliceSpec compactSlice(WeightRole role)const{return cohort_->sourceSlice(member_,role,experts_);}
    /** @return Structural source/member/selection equality, never pointer-only cohort equality. */
    bool operator==(const CPUExpertShardSourceSelection& other)const noexcept;
private:
    /** @brief Retain only the already-authenticated canonical map facts. */
    CPUExpertShardSourceSelection(std::shared_ptr<const MoEExpertShardCohort> cohort,
        std::size_t member,MoEExpertOwnerParticipant participant,std::vector<int> experts)
        :cohort_(std::move(cohort)),member_(member),participant_(std::move(participant)),experts_(std::move(experts)){}
    std::shared_ptr<const MoEExpertShardCohort> cohort_;
    std::size_t member_;
    MoEExpertOwnerParticipant participant_;
    std::vector<int> experts_;
};
} // namespace llaminar2
