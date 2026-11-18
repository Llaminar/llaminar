/**
 * @file ExpertShardSourceSelection.cpp
 * @brief Derive immutable CPU/GPU fragment source selection from the original canonical map.
 *
 * Original logical IDs are never apportioned again. Each member retains the
 * same selected experts and cohort definition, while the existing projection
 * geometry supplies its distinct native source rectangle. No source payload,
 * allocation, placement transition or execution readiness is manufactured.
 * Empty selections retain canonical family metadata only and cannot request a
 * native source slice, which still requires actual selected expert IDs.
 */
#include "ExpertShardSourceSelection.h"
#include <algorithm>
#include <stdexcept>
namespace llaminar2 {
std::shared_ptr<const ExpertShardSourceSelection> ExpertShardSourceSelection::fromOriginalPlacement(
    const MoEExpertOwnerMap& original,int layer,int participant_id)
{
    const auto* participant=original.participantForId(participant_id);
    if(layer<0 || !participant || !participant->world_rank_known || participant->world_rank<0 ||
       !participant->device.is_valid() || participant->address.toLocalDeviceId()!=participant->device ||
       (participant->device.is_cpu() &&
        (participant->address.device_ordinal!=0 || !participant->address.hasValidNuma())))
        throw std::invalid_argument("Joint source requires its original rank/device participant and CPU NUMA identity");
    // Source provenance is independent of initial residency. The canonical
    // original map retains this block even when no expert selects the tier.
    auto cohort=original.originalShardCohort(participant->tier_idx,layer);
    if(!cohort || cohort->source().layer_idx!=layer)
        throw std::invalid_argument("Joint source has no original tier/layer cohort declaration");
    auto experts=original.expertsForParticipant(layer,participant_id);
    std::sort(experts.begin(),experts.end());
    if(std::adjacent_find(experts.begin(),experts.end())!=experts.end())
        throw std::invalid_argument("Joint source repeats an original logical expert");
    for(const int expert:experts) {
        const auto* row=original.placementFor(layer,expert);
        if(!row || !row->shard_cohort || row->tier_idx!=participant->tier_idx ||
           row->domain_name!=participant->domain_name || row->tier_name!=participant->tier_name ||
           row->layer_idx!=layer || row->expert_id!=expert || row->shard_cohort->source().layer_idx!=layer)
            throw std::invalid_argument("Joint source cannot reinterpret a whole/foreign expert owner");
        if(cohort!=row->shard_cohort || cohort.owner_before(row->shard_cohort) ||
           row->shard_cohort.owner_before(cohort))
            throw std::invalid_argument("Joint source selection does not retain its original cohort authority");
        if(expert<0 || expert>=cohort->ownership(0).geometry().experts)
            throw std::invalid_argument("Joint source selection lies outside the original expert axis");
    }
    const auto members=cohort->participants();
    const auto found=std::find_if(members.begin(),members.end(),
        [&](const auto& member){return member.participant_id==participant_id;});
    if(found==members.end() || found->world_rank!=participant->world_rank ||
       found->address!=participant->address || participant->domain_participant_index<0 ||
       static_cast<std::size_t>(participant->domain_participant_index)!=static_cast<std::size_t>(found-members.begin()))
        throw std::invalid_argument("Joint source participant changed its original cohort coordinate");
    return std::shared_ptr<const ExpertShardSourceSelection>(new ExpertShardSourceSelection(
        std::move(cohort),static_cast<std::size_t>(found-members.begin()),*participant,std::move(experts)));
}
bool ExpertShardSourceSelection::contains(int expert)const noexcept
{
    return std::binary_search(experts_.begin(),experts_.end(),expert);
}
bool ExpertShardSourceSelection::operator==(const ExpertShardSourceSelection& other)const noexcept
{
    return member_==other.member_ && participant_==other.participant_ && experts_==other.experts_ &&
        (cohort_==other.cohort_ || *cohort_==*other.cohort_);
}
} // namespace llaminar2
