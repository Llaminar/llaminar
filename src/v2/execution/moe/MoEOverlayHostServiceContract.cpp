/** @file MoEOverlayHostServiceContract.cpp
 * @brief Preserve existing routed endpoint/transport decisions before quotas and during lowering.
 */
#include "MoEOverlayHostServiceContract.h"
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
        if(p.world_rank==source->world_rank)result.local_.push_back(p.participant_id);
        else {result.remote_.push_back(p.participant_id);
            if(!sameNode(source->world_rank,p.world_rank))
                result.allRemoteNodeLocal_=false;}
    }
    return result;
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
