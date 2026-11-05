/**
 * @file MoEOptimizationMovementEndpoint.cpp
 * @brief Checked cold projection of canonical whole/cohort movement endpoints.
 *
 * The original shared cohort remains the only source of layout and membership.
 * These methods observe metadata and mint neither residency nor native readiness.
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
    return MoEOptimizationMovementEndpoint(CPUCohort{owner.tier_idx,owner.shard_cohort});
}
const MoEOptimizationMovementEndpoint::Whole& MoEOptimizationMovementEndpoint::whole() const {
    if(!isWhole())throw std::logic_error("Whole movement projection rejects complete cohort or absent endpoint");
    return std::get<Whole>(value_);
}
const MoEOptimizationMovementEndpoint::CPUCohort& MoEOptimizationMovementEndpoint::cohort() const {
    if(!isCPUCohort() || !std::get<CPUCohort>(value_).original)throw std::logic_error("Cohort movement projection requires complete original cohort");
    return std::get<CPUCohort>(value_);
}
MoEOverlayRouteTarget MoEOptimizationMovementEndpoint::target() const noexcept {
    if(isWhole())return MoEOverlayRouteTarget::whole(std::get<Whole>(value_).participant);
    if(isCPUCohort())return MoEOverlayRouteTarget::cpuCohort(std::get<CPUCohort>(value_).tier);
    return {};
}
bool MoEOptimizationMovementEndpoint::valid() const noexcept {
    if(isWhole()){const auto& w=std::get<Whole>(value_);return w.participant>=0&&w.device.is_valid()&&(!w.world_rank||*w.world_rank>=0);}
    if(isCPUCohort()){const auto& c=std::get<CPUCohort>(value_);return c.tier>=0&&c.tier<=std::numeric_limits<int32_t>::max()-1&&bool(c.original)&&c.original->participants().size()>=2;}
    return false;
}
bool MoEOptimizationMovementEndpoint::validForLayer(int layer) const noexcept {
    return layer>=0 && valid() && (!isCPUCohort() || std::get<CPUCohort>(value_).original->source().layer_idx==layer);
}
bool MoEOptimizationMovementEndpoint::operator==(const MoEOptimizationMovementEndpoint& other) const {
    if(value_.index()!=other.value_.index())return false;
    if(isWhole())return whole()==other.whole();
    if(isCPUCohort()){const auto& a=std::get<CPUCohort>(value_);const auto& b=std::get<CPUCohort>(other.value_);
        if(a.tier!=b.tier)return false;
        // Default moves preserve the variant arm but transfer its shared borrow.
        // Null arms remain comparable without projecting invalid ownership.
        if(!a.original || !b.original)return a.original==b.original;
        return a.original==b.original || *a.original==*b.original;}
    return true;
}
}
