/** @file MoEOverlayHostServiceContract.cpp
 * @brief Preserve pre-quota service demand and project immutable admitted execution.
 *
 * An empty runtime route mask is not evidence that an endpoint can disappear.
 * Only an original, complete, frozen zero quota removes an external tier from
 * capture; its conservative pre-quota service reservation remains unchanged.
 */
#include "MoEOverlayHostServiceContract.h"
#include "execution/local_execution/graph/ComputeGraph.h"
#include "MoEOverlayRankBatchTransport.h"
#include "MoEOverlayNodeLocalRankBatchTransport.h"
#include "interfaces/IMPIContext.h"
#include "interfaces/IMPITopology.h"
#include "execution/mpi_orchestration/DeviceInventory.h"
#include <algorithm>
#include <stdexcept>
namespace llaminar2 {
MoEOverlayHostServiceContract MoEOverlayHostServiceContract::resolve(const MoERoutedExpertPlacementPlan& plan,std::span<const MoEExpertOwnerParticipant> participants,const IMPIContext& world){
    return resolveImpl(plan,participants,world.world_size(),[&](int a,int b){
        if(!world.topology())throw std::logic_error("CPU service remote declaration requires original physical topology");
        return world.topology()->same_node(a,b);
    });
}
MoEOverlayHostServiceContract MoEOverlayHostServiceContract::resolve(const MoERoutedExpertPlacementPlan& plan,std::span<const MoEExpertOwnerParticipant> participants,const ClusterInventory& inventory){
    return resolveImpl(plan,participants,inventory.world_size,[&](int a,int b){return inventory.connectionBetweenRanks(a,b).locality()!=RankConnectionLocality::CrossNode;});
}
std::vector<MoEExpertOwnerParticipant> MoEOverlayHostServiceContract::admittedExpertParticipants(
    const MoERoutedExpertPlacementPlan& plan, const MoEExpertOwnerMap& owners)
{
    std::vector<bool> permanently_empty(plan.routed_tiers.size(), false);
    for (std::size_t tier = 0; tier < plan.routed_tiers.size(); ++tier) {
        const auto& declaration = plan.routed_tiers[tier];
        if (declaration.domain == plan.continuation_domain) continue;
        const auto& quotas = declaration.resolved_live_experts_per_layer;
        if (quotas.empty() || !std::all_of(quotas.begin(), quotas.end(), [](auto value) { return value == 0; }))
            continue;
        if (plan.placements.empty() || std::any_of(plan.placements.begin(), plan.placements.end(), [&](const auto& layer) {
                return layer.layer < plan.first_model_layer ||
                    static_cast<std::size_t>(layer.layer - plan.first_model_layer) >= quotas.size();
            }))
            throw std::logic_error("Admitted empty host tier has no complete frozen layer quota");
        permanently_empty[tier] = true;
    }
    for (const auto& owner : owners.owners()) {
        if (owner.tier_idx < 0 || static_cast<std::size_t>(owner.tier_idx) >= permanently_empty.size() ||
            owner.domain_name != plan.routed_tiers[static_cast<std::size_t>(owner.tier_idx)].domain)
            throw std::logic_error("Admitted host service changed original expert tier identity");
        if (permanently_empty[static_cast<std::size_t>(owner.tier_idx)])
            throw std::logic_error("Admitted zero host tier still owns an expert contribution");
    }
    std::vector<MoEExpertOwnerParticipant> result;
    result.reserve(owners.participants().size());
    for (const auto& participant : owners.participants()) {
        if (participant.tier_idx < 0 || static_cast<std::size_t>(participant.tier_idx) >= permanently_empty.size() ||
            participant.domain_name != plan.routed_tiers[static_cast<std::size_t>(participant.tier_idx)].domain)
            throw std::logic_error("Admitted host service changed original participant tier identity");
        if (!permanently_empty[static_cast<std::size_t>(participant.tier_idx)])
            result.push_back(participant);
    }
    return result;
}
MoEOverlayHostServiceContract MoEOverlayHostServiceContract::resolveAdmitted(
    const MoERoutedExpertPlacementPlan& plan, const MoEExpertOwnerMap& owners, const IMPIContext& world)
{
    const auto participants = admittedExpertParticipants(plan, owners);
    return resolve(plan, participants, world);
}
MoEOverlayHostServiceContract MoEOverlayHostServiceContract::resolveAdmitted(
    const MoERoutedExpertPlacementPlan& plan, const MoEExpertOwnerMap& owners, const ClusterInventory& inventory)
{
    const auto participants = admittedExpertParticipants(plan, owners);
    return resolve(plan, participants, inventory);
}
MoEOverlayHostServiceContract MoEOverlayHostServiceContract::resolveImpl(const MoERoutedExpertPlacementPlan& plan,std::span<const MoEExpertOwnerParticipant> participants,int worldSize,const std::function<bool(int,int)>& sameNode){
    if(worldSize<=0)throw std::invalid_argument("CPU service topology world population is inconsistent");
    MoEOverlayHostServiceContract result;result.distributed_=worldSize>1;result.rootParticipant_=std::max(0,plan.continuation_domain_spec.logical_root_participant);
    for(std::size_t tier=0;tier<plan.routed_tiers.size();++tier)if(plan.routed_tiers[tier].domain==plan.continuation_domain){
        if(result.continuationTier_>=0)throw std::logic_error("Routed continuation names multiple tiers");
        result.continuationTier_=static_cast<int>(tier);const auto found=std::find_if(plan.domains.begin(),plan.domains.end(),[&](const auto& d){return d.name==plan.routed_tiers[tier].domain;});
        const auto* domain=found==plan.domains.end()?nullptr:&*found;
        if(!domain)throw std::logic_error("Routed continuation has no authored domain");
        result.continuation_=ContinuationGeometry{domain->scope,domain->participants.size(),domain->routed_compute_policy};
    }
    if(!result.continuation_)return result;
    const auto source=std::find_if(participants.begin(),participants.end(),[&](const auto& p){return p.participant_id==result.rootParticipant_;});
    if(source==participants.end()||!source->world_rank_known||source->tier_idx!=result.continuationTier_)
        throw std::logic_error("Routed host service has no original continuation rank/root descriptor");
    for(const auto& p:participants){
        if(!p.world_rank_known||p.world_rank<0||p.world_rank>=worldSize)throw std::logic_error("Routed host service topology has unresolved physical rank");
        if(p.tier_idx==result.continuationTier_){result.continuationParticipants_.push_back(p);continue;}
        if(p.world_rank==source->world_rank && p.device.is_cpu())
            result.local_.push_back(p.participant_id);
        else {result.external_.push_back(p.participant_id);
            if (p.world_rank == source->world_rank && p.device.is_gpu() && source->device.is_gpu() &&
                p.device.type != source->device.type)
                result.rankLocalCrossVendor_ = true;
            if(p.world_rank!=source->world_rank && !sameNode(source->world_rank,p.world_rank))
                result.allExternalNodeLocal_=false;}
    }
    return result;
}
GraphNativeCaptureEnvelope MoEOverlayHostServiceContract::captureEnvelope(bool dispatchAuthority) const noexcept {
    if (!hasHostBoundary() && !requiresRankLocalInferencePublisher())
        return GraphNativeCaptureEnvelope::DeviceOwnedTimelineTransaction;
    return dispatchAuthority
        ? GraphNativeCaptureEnvelope::HeterogeneousTicketAuthorityTransaction
        : GraphNativeCaptureEnvelope::HeterogeneousTicketFollowerTransaction;
}
bool MoEOverlayHostServiceContract::requiresExecutorService(int rank,DeviceId device,int degree,int index)const{
    if(!continuation_||!device.is_gpu()||!hasHostBoundary())return false;
    const auto participant=std::find_if(continuationParticipants_.begin(),continuationParticipants_.end(),[&](const auto& p){return p.world_rank==rank&&p.device==device;});
    if(participant==continuationParticipants_.end())return false;
    if(std::find_if(std::next(participant),continuationParticipants_.end(),[&](const auto& p){return p.world_rank==rank&&p.device==device;})!=continuationParticipants_.end())
        throw std::logic_error("Routed host service resolves duplicate continuation device");
    const bool single=continuation_->scope==ExecutionDomainScope::SINGLE&&continuation_->participants==1&&continuation_->compute!=RoutedExpertComputePolicy::TensorSharded;
    const bool localTP=degree>1&&continuation_->scope==ExecutionDomainScope::RANK_LOCAL&&continuation_->compute==RoutedExpertComputePolicy::Apportioned&&continuation_->participants==static_cast<std::size_t>(degree);
    if(localTP&&participant->domain_participant_index!=index)throw std::logic_error("Routed host service LocalTP coordinate changed");
    return (single||localTP)&&(!distributed_||participant->participant_id==rootParticipant_);
}
}
