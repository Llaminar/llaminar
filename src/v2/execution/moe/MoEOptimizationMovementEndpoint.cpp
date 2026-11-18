/**
 * @file MoEOptimizationMovementEndpoint.cpp
 * @brief Checked cold projection of canonical whole/cohort movement endpoints.
 *
 * The original shared cohort remains the only source of layout and membership.
 * These methods observe metadata and mint neither residency nor native readiness.
 * A checked GPU singleton is still a complete cohort; the source factory owns
 * population validation, including the distinct CPU multi-rank requirement.
 */
#include "MoEOptimizationMovementEndpoint.h"
#include "MoEExpertOwnerMap.h"
#include "MoEExpertShardCohort.h"
#include <stdexcept>
#include <limits>
namespace llaminar2 {
MoEOptimizationMovementEndpoint MoEOptimizationMovementEndpoint::whole(int participant,DeviceId device,std::optional<int> rank) {
    if(participant<0 || !device.is_valid() || (rank && *rank<0))
        throw std::invalid_argument("Completed whole movement endpoint has invalid original identity");
    return MoEOptimizationMovementEndpoint(Whole{participant,device,rank});
}
MoEOptimizationMovementEndpoint MoEOptimizationMovementEndpoint::fromOwner(const MoEExpertOwner& owner) {
    if(!owner.shard_cohort)return whole(owner.owner_participant,owner.device,
        owner.owner_world_rank_known?std::optional{owner.owner_world_rank}:std::nullopt);
    if(owner.tier_idx<0 || owner.tier_idx>std::numeric_limits<int32_t>::max()-1 || owner.layer_idx<0 || owner.shard_cohort->source().layer_idx!=owner.layer_idx ||
       owner.owner_participant!=-1 || owner.device.is_valid())
        throw std::invalid_argument("Completed cohort movement endpoint differs from original complete placement");
    return MoEOptimizationMovementEndpoint(Cohort{owner.tier_idx,owner.shard_cohort});
}
const MoEOptimizationMovementEndpoint::Whole& MoEOptimizationMovementEndpoint::whole() const {
    if(!isWhole())throw std::logic_error("Whole movement projection rejects complete cohort or absent endpoint");
    return std::get<Whole>(value_);
}
const MoEOptimizationMovementEndpoint::Cohort& MoEOptimizationMovementEndpoint::cohort() const {
    if(!isCohort() || !std::get<Cohort>(value_).original)throw std::logic_error("Cohort movement projection requires complete original cohort");
    return std::get<Cohort>(value_);
}
MoEOverlayRouteTarget MoEOptimizationMovementEndpoint::target() const noexcept {
    if(isWhole())return MoEOverlayRouteTarget::whole(std::get<Whole>(value_).participant);
    if(isCohort())return MoEOverlayRouteTarget::cohort(std::get<Cohort>(value_).tier);
    return {};
}
bool MoEOptimizationMovementEndpoint::valid() const noexcept {
    if(isWhole()){const auto& w=std::get<Whole>(value_);return w.participant>=0&&w.device.is_valid()&&(!w.world_rank||*w.world_rank>=0);}
    if(isCohort()){const auto& c=std::get<Cohort>(value_);return c.tier>=0&&c.tier<=std::numeric_limits<int32_t>::max()-1&&bool(c.original)&&!c.original->participants().empty();}
    return false;
}
bool MoEOptimizationMovementEndpoint::validForLayer(int layer) const noexcept {
    return layer>=0 && valid() && (!isCohort() || std::get<Cohort>(value_).original->source().layer_idx==layer);
}
bool MoEOptimizationMovementEndpoint::operator==(const MoEOptimizationMovementEndpoint& other) const {
    if(value_.index()!=other.value_.index())return false;
    if(isWhole())return whole()==other.whole();
    if(isCohort()){const auto& a=std::get<Cohort>(value_);const auto& b=std::get<Cohort>(other.value_);
        if(a.tier!=b.tier)return false;
        // Default moves preserve the variant arm but transfer its shared borrow.
        // Null arms remain comparable without projecting invalid ownership.
        if(!a.original || !b.original)return a.original==b.original;
        return a.original==b.original || *a.original==*b.original;}
    return true;
}
}
