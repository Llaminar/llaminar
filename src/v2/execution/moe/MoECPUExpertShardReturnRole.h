/**
 * @file MoECPUExpertShardReturnRole.h
 * @brief Cold source-map projection of the one original CPU joint return publisher.
 *
 * Logical expert identity remains joint. This projection only names which
 * original member publishes the already-combined canonical route payload;
 * every contributor still computes, gathers and participates in terminal ACK.
 */
#pragma once
#include "MoEExpertOwnerMap.h"
#include "MoEExpertShardCohort.h"
#include "MoEOverlayActivationEpochProtocol.h"
#include <span>
#include <stdexcept>
namespace llaminar2 {
/**
 * @brief Validate source-cohort membership across an original retained graph family.
 * @param source Original immutable source-prepared map; null is the unchanged ordinary path.
 * @param target Exact authored target endpoint, not first-arriving participant.
 * @param layers Original main/sidecar model-layer manifest in the channel.
 * @return Ordinary, or the fixed original cohort root/contributor publication role.
 * @throws std::invalid_argument For missing, mixed, foreign or replaced joint declarations.
 */
inline MoEOverlayActivationReturnRole originalCPUExpertShardReturnRole(const MoEExpertOwnerMap* source,
    const MoEOverlayActivationLaneEndpoint &target,std::span<const int32_t> layers) {
    if(!source)return MoEOverlayActivationReturnRole::Ordinary;
    const auto* participant=source->participantForId(target.participant_id);
    if(!participant || !participant->world_rank_known || participant->world_rank!=target.world_rank || layers.empty())
        throw std::invalid_argument("CPU joint channel has foreign original endpoint/family membership");
    if(!participant->device.is_cpu())return MoEOverlayActivationReturnRole::Ordinary;
    std::optional<MoEOverlayActivationReturnRole> role;
    size_t ordinary_layers=0;
    for(size_t ordinal=0;ordinal<layers.size();++ordinal) {
        const int layer=layers[ordinal];
        if(layer<0 || std::find(layers.begin(),layers.begin()+ordinal,layer)!=layers.begin()+ordinal)
            throw std::invalid_argument("CPU joint channel repeats or changes an original source layer");
        std::shared_ptr<const MoEExpertShardCohort> cohort;
        bool whole=false;
        for(const auto &row:source->owners())if(row.layer_idx==layer) {
            if(!row.shard_cohort) {whole=whole || row.owner_participant==target.participant_id;continue;}
            const auto members=row.shard_cohort->participants();
            const auto member=std::find_if(members.begin(),members.end(),[&](const auto& item){return item.participant_id==target.participant_id;});
            if(member==members.end())continue;
            if(member->world_rank!=target.world_rank || member->address!=participant->address)
                throw std::invalid_argument("CPU joint channel replaced original physical member identity");
            if(cohort && (cohort->participants().size()!=members.size() ||
               !std::equal(cohort->participants().begin(),cohort->participants().end(),members.begin())))
                throw std::invalid_argument("CPU joint channel source changed original reduction membership");
            cohort=row.shard_cohort;
        }
        if(!cohort) {
            if(role)throw std::invalid_argument("CPU joint channel omitted an original source layer");
            ++ordinary_layers;continue;
        }
        if(ordinary_layers)throw std::invalid_argument("CPU joint channel omitted an original source layer");
        if(whole)throw std::invalid_argument("CPU joint channel cannot alias a whole-expert return publisher");
        const auto current=cohort->participants().front().participant_id==target.participant_id
            ?MoEOverlayActivationReturnRole::StaticJointRoot:MoEOverlayActivationReturnRole::StaticJointContributor;
        if(role && *role!=current)throw std::invalid_argument("CPU joint channel changed original root across source layers");
        role=current;
    }
    return role.value_or(MoEOverlayActivationReturnRole::Ordinary);
}
}
