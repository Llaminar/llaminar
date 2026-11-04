/**
 * @file MoEOverlayMTPSourceContract.h
 * @brief Original whole/cohort source completeness before MTP materialization.
 *
 * This passive contract validates the retained source recipe, not prepared
 * engines or native readiness. Registry publication later authenticates actual
 * Store/PMA lifetimes and every prepared projection independently.
 */
#pragma once
#include "MoEExpertOverlayPreparationPlan.h"
#include "MoEExpertOwnerMap.h"
#include "CPUExpertShardSourceSelection.h"
#include <map>
#include <set>
#include <span>
#include <tuple>

namespace llaminar2 {
/**
 * @brief Require one complete original placement/source recipe per predictor expert.
 * @param owners Original authority map; never reconstructed from compact weights.
 * @param preparation Same original rank-scoped source recipe, required for joint rows.
 * @param layers Actual model transaction family's original predictor layer IDs.
 * @param experts Original complete expert axis.
 * @param world_rank Rank whose local source requests will be materialized.
 * @throws std::invalid_argument For absent, foreign or partial source/cohort coverage.
 * @note Remote members are authenticated by the sealed original cohort. Only
 * local requests are required here; no rank claims another rank's prepared bytes.
 */
inline void requireMoEOverlayMTPSourceCoverage(const MoEExpertOwnerMap &owners,
    const MoEExpertOverlayPreparationPlan *preparation,std::span<const int> layers,
    int experts,int world_rank)
{
    if(experts<=0 || world_rank<0)throw std::invalid_argument("MTP source requires original rank/expert geometry");
    const std::set<int> source_layers(layers.begin(),layers.end());
    if(source_layers.size()!=layers.size())throw std::invalid_argument("MTP source repeats an original predictor layer");
    using Coordinate=std::tuple<int,int,int>;
    std::map<Coordinate,unsigned> local_roles;
    if(preparation)for(const auto &request:preparation->requests()) {
        if(!source_layers.contains(request.layer) || request.participant_world_rank!=world_rank || !request.cpu_joint_source)continue;
        const auto &selected=*request.cpu_joint_source;
        const auto *row=owners.placementFor(request.layer,request.expert_id);
        if(!row || !row->shard_cohort || selected.cohort()!=row->shard_cohort ||
           selected.layer()!=request.layer || !selected.contains(request.expert_id) ||
           selected.participant().world_rank!=world_rank ||
           selected.participant().domain_participant_index!=request.participant_index ||
           selected.participant().domain_name!=request.domain_name ||
           selected.participant().device!=request.device || !request.projection_ownership ||
           *request.projection_ownership!=selected.ownership())
            throw std::invalid_argument("MTP local source request changed original predictor/cohort geometry");
        unsigned bit=0;
        switch(request.role) {
            case ExpertGemmRegistry::WeightRole::GATE:bit=1;break;
            case ExpertGemmRegistry::WeightRole::UP:bit=2;break;
            case ExpertGemmRegistry::WeightRole::DOWN:bit=4;break;
            default:throw std::invalid_argument("MTP source has an unknown expert projection");
        }
        auto &roles=local_roles[{request.layer,request.expert_id,selected.participant().participant_id}];
        if(roles&bit)throw std::invalid_argument("MTP source repeats an original expert projection");
        roles|=bit;
    }
    for(const int layer:layers)for(int expert=0;expert<experts;++expert) {
        const auto *placement=owners.placementFor(layer,expert);
        if(!placement)throw std::invalid_argument("MTP predictor expert has no original placement");
        if(!placement->shard_cohort) {
            if(!owners.ownerFor(layer,expert))throw std::invalid_argument("MTP predictor expert lost its original whole owner");
            continue;
        }
        if(!preparation || preparation->cpuJointOwnerMap().get()!=&owners)
            throw std::invalid_argument("MTP joint source requires the same original preparation/map authority");
        const auto &cohort=*placement->shard_cohort;
        if(cohort.source().layer_idx!=layer || cohort.participants().size()<2 || cohort.participants().size()>8)
            throw std::invalid_argument("MTP predictor cohort changed original layer/population");
        for(size_t member=0;member<cohort.participants().size();++member) {
            const auto &original=cohort.participants()[member];
            const auto *participant=owners.participantForId(original.participant_id);
            if(!participant || !participant->world_rank_known || participant->world_rank!=original.world_rank ||
               participant->address!=original.address || participant->domain_participant_index!=int(member) ||
               participant->tier_idx!=placement->tier_idx || cohort.ownership(member).geometry().experts!=experts)
                throw std::invalid_argument("MTP predictor cohort lost an original source member");
            if(original.world_rank==world_rank && local_roles[{layer,expert,original.participant_id}]!=7)
                throw std::invalid_argument("MTP predictor local fragment requires all original Gate/Up/Down sources");
        }
    }
}
}
