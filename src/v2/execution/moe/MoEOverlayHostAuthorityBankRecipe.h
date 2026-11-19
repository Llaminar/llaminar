/**
 * @file MoEOverlayHostAuthorityBankRecipe.h
 * @brief Checked host-authority device-bank recipes from original participant banks.
 *
 * This cold metadata boundary exports the bank's closed whole or fragment arm.
 * It never manufactures a complete expert from fragment engines, performs a
 * transfer, or grants an epoch. Recipe construction accepts cold metadata;
 * the actual publisher separately requires hostPublicationValid() and retains
 * every original stream, event, inference boundary and epoch authority. The existing publisher retains the exact bank
 * reader through device publication and old-reader retirement.
 */
#pragma once
#include "DeviceMoEExpertDescriptorBuilder.h"
#include "MoEOverlayDeviceControllerRuntimeBinding.h"
#include "MoEOverlayParticipantResidency.h"
#include "MoEExpertShardCohort.h"
#include "MoEExpertOverlayPreparationPlan.h"
#include "MoEOverlayResidencyAuthority.h"
#include "MoEOverlayRouteTarget.h"
#include "MoERuntimeTable.h"
#include <algorithm>
#include <span>
#include <limits>
#include <stdexcept>

namespace llaminar2 {
namespace moe_host_authority_recipe_detail {
/** @return Exact cold runtime identity; this does not grant any device publication right. */
inline bool metadataMatches(const MoEOverlayDeviceControllerRuntimeBinding& binding)
{
    const auto* runtime=binding.runtime_table_host;
    return runtime && binding.device.is_gpu() && binding.layerScopeValid() &&
        binding.overlay_participant_id>=0 && binding.domain_participant_count>0 &&
        binding.domain_participant_id<binding.domain_participant_count && binding.expert_count>0 && binding.top_k>0 &&
        runtime->deviceId()==binding.device && runtime->expertCount()==int(binding.expert_count) &&
        runtime->topK()==int(binding.top_k) && binding.matchesLayerScope(runtime->firstModelLayer(),uint32_t(runtime->layerCount()));
}
} // namespace moe_host_authority_recipe_detail

/**
 * @brief Require every original local GPU cohort member before any bank reserve.
 * @param snapshot Same candidate placement and original participant catalog.
 * @param bindings Actual process-local publication endpoints, with no aliases.
 * @throws std::invalid_argument For omitted, duplicated, foreign or mixed-rank bindings.
 */
inline void requireMoEOverlayHostAuthorityGPUCohortBindings(
    const MoEOverlayResidencySnapshot& snapshot,
    std::span<const MoEOverlayDeviceControllerRuntimeBinding> bindings)
{
    if(bindings.empty())throw std::invalid_argument("GPU bank publication requires its original endpoints");
    const auto& owners=snapshot.placementOwnerMap();
    const auto* first=owners.participantForId(bindings.front().overlay_participant_id);
    if(!first || !first->world_rank_known)
        throw std::invalid_argument("GPU bank publication lost its original local rank");
    for(const auto& binding:bindings) {
        const auto* member=owners.participantForId(binding.overlay_participant_id);
        if(!moe_host_authority_recipe_detail::metadataMatches(binding) || !member || !member->world_rank_known ||
           member->world_rank!=first->world_rank || member->device!=binding.device ||
           std::count_if(bindings.begin(),bindings.end(),[&](const auto& other){
               return other.overlay_participant_id==binding.overlay_participant_id;})!=1)
            throw std::invalid_argument("GPU bank publication has a foreign or repeated original endpoint");
    }
    const auto require_cohort=[&](const std::shared_ptr<const MoEExpertShardCohort>& cohort) {
        if(!cohort || !cohort->participants().front().address.isGPU())return;
        for(const auto& member:cohort->participants()) {
            if(member.world_rank!=first->world_rank)continue;
            const auto found=std::find_if(bindings.begin(),bindings.end(),[&](const auto& binding){
                return binding.overlay_participant_id==member.participant_id &&
                       binding.device==member.address.toLocalDeviceId();});
            if(found==bindings.end())
                throw std::invalid_argument("GPU bank publication omitted an original local cohort member");
        }
    };
    // Original empty cohorts still own an execution lane and receive remote
    // routes. Coverage comes from original source declarations, not live IDs.
    const auto& original=snapshot.originalOwnerMap();
    for(const auto& participant:original.participants()) {
        // Each canonical tier has exactly one original domain coordinate0.
        // Borrow its source declaration once per layer, independent of E or
        // preparation request count, including a zero-expert original target.
        if(participant.domain_participant_index!=0)continue;
        for(int row=0;row<snapshot.layerCount();++row)
            require_cohort(original.originalShardCohort(participant.tier_idx,
                snapshot.firstModelLayer()+row));
    }

}

/**
 * @brief Export one exact candidate layer through its original closed payload arm.
 * @param candidate Immutable candidate placement and prepared-source authority.
 * @param bank Original endpoint's installed candidate bank, retained by its reader.
 * @param binding Same original GPU runtime identity and model-layer scope.
 * @param layer Original global model layer, never a compact storage index.
 * @return Complete update for the existing inactive-bank publication machinery.
 * @throws std::invalid_argument For source/family/member/geometry/selection mismatch.
 */
[[nodiscard]] inline MoEPlacementUpdate buildMoEOverlayHostAuthorityDeviceLayerUpdate(
    const MoEOverlayResidencySnapshot& candidate,
    const MoEOverlayParticipantResidencyBank& bank,
    const MoEOverlayDeviceControllerRuntimeBinding& binding,int layer)
{
    const auto& owners=candidate.placementOwnerMap();
    const auto* participant=owners.participantForId(binding.overlay_participant_id);
    if(!candidate.valid() || !moe_host_authority_recipe_detail::metadataMatches(binding) || !participant ||
       participant->device!=binding.device || !bank.valid(binding.overlay_participant_id,binding.device,
           int(binding.layer_count),int(binding.expert_count),binding.first_model_layer) ||
       bank.epoch!=candidate.epoch || candidate.epoch>std::numeric_limits<uint32_t>::max() ||
       !bank.containsModelLayer(layer))
        throw std::invalid_argument("GPU candidate recipe lost its original endpoint, epoch or layer scope");
    const auto& local=bank.layerForModelLayer(layer);
    const auto* runtime_family=binding.runtime_table_host->fragmentFamily(layer);
    const auto& family=local.fragment_family;
    if(!family && binding.domain_participant_count>kDeviceMoEMaxParticipants)
        throw std::invalid_argument("Whole GPU candidate recipe exceeds its native resident-mask width");
    if(bool(family)!=bool(runtime_family) || (family &&
       (runtime_family!=family.get() || family->preparation()!=candidate.originalProjectionPreparation())))
        throw std::invalid_argument("GPU candidate recipe changed its original fragment family");
    if(family) {
        const auto member=family->participantIndex();
        const auto members=family->cohort()->participants();
        if(member>=members.size() || members[member].participant_id!=binding.overlay_participant_id ||
           members[member].address.toLocalDeviceId()!=binding.device ||
           member!=binding.domain_participant_id || members.size()!=binding.domain_participant_count ||
           local.fragments.size()!=binding.expert_count || !local.experts.empty())
            throw std::invalid_argument("GPU candidate recipe changed its original fragment member");
    }
    MoEPlacementUpdate update;
    update.epoch=uint32_t(candidate.epoch);update.expert_count=binding.expert_count;
    update.participant_id=binding.domain_participant_id;update.participant_count=binding.domain_participant_count;
    update.experts.resize(binding.expert_count);update.local_compute_mask.assign(binding.expert_count,0u);
    update.replica_role.assign(binding.expert_count,uint8_t(DeviceMoEReplicaRole::None));
    update.resident_participant_mask.assign(binding.expert_count,0u);
    update.overlay_route_participant.assign(binding.expert_count,-1);
    for(uint32_t expert=0;expert<binding.expert_count;++expert) {
        const auto* placement=owners.placementFor(layer,int(expert));
        if(!placement)throw std::invalid_argument("GPU candidate recipe has an unowned logical expert");
        auto& descriptor=update.experts[expert];descriptor.logical_expert_id=int32_t(expert);
        descriptor.projection_set=binding.runtime_table_host->movableProjections();
        if(placement->shard_cohort) {
            const auto& cohort=*placement->shard_cohort;
            update.overlay_route_participant[expert]=MoEOverlayRouteTarget::cohort(placement->tier_idx).word();
            descriptor.owner_participant=-1;
            const auto members=cohort.participants();
            const auto member=std::find_if(members.begin(),members.end(),[&](const auto& value){
                return value.participant_id==binding.overlay_participant_id;});
            const bool local_member=member!=members.end();
            if(local_member!=local.resident_mask.at(expert))
                throw std::invalid_argument("GPU candidate recipe has an incomplete or foreign cohort selection");
            if(!local_member)continue; // Genuine CPU/GPU remote cohort; no whole-owner alias.
            if(!family || family->cohort().get()!=placement->shard_cohort.get() ||
               family->participantIndex()!=size_t(member-members.begin()) ||
               !family->accepts(local.fragments[expert],int(expert)) ||
               !exportDeviceMoEFragmentWeightDescriptors(local.fragments[expert],descriptor))
                throw std::invalid_argument("GPU candidate recipe rejected foreign original fragment provenance");
            descriptor.local_slot=int32_t(expert);
            descriptor.flags=toMoEExpertFlags(DeviceMoEExpertFlags::Valid |
                DeviceMoEExpertFlags::Resident | DeviceMoEExpertFlags::LocalCompute);
            update.local_compute_mask[expert]=1u;
            continue;
        }
        if(placement->owner_participant<0)
            throw std::invalid_argument("GPU candidate recipe has no complete target owner");
        update.overlay_route_participant[expert]=placement->owner_participant;
        if(placement->domain_name==participant->domain_name) {
            if(family || placement->domain_participant_index<0 ||
               uint32_t(placement->domain_participant_index)>=binding.domain_participant_count)
                throw std::invalid_argument("GPU candidate recipe cannot alias a fragment as a whole owner");
            descriptor.owner_participant=placement->domain_participant_index;
            update.resident_participant_mask[expert]=1u<<uint32_t(placement->domain_participant_index);
        } else descriptor.owner_participant=-1;
        if(!local.resident_mask.at(expert))continue;
        if(family || placement->owner_participant!=binding.overlay_participant_id ||
           !local.experts.at(expert).complete() ||
           !exportDeviceMoEExpertWeightDescriptors(local.experts[expert].gate().get(),
               local.experts[expert].up().get(),local.experts[expert].down().get(),descriptor))
            throw std::invalid_argument("GPU candidate recipe rejected its complete original resident source");
        descriptor.logical_expert_id=int32_t(expert);descriptor.owner_participant=int32_t(binding.domain_participant_id);
        descriptor.local_slot=int32_t(expert);
        descriptor.flags=toMoEExpertFlags(DeviceMoEExpertFlags::Valid | DeviceMoEExpertFlags::Resident |
            DeviceMoEExpertFlags::LocalCompute | DeviceMoEExpertFlags::PreferredOwner);
        update.local_compute_mask[expert]=1u;update.replica_role[expert]=uint8_t(DeviceMoEReplicaRole::Primary);
        update.resident_participant_mask[expert]|=1u<<binding.domain_participant_id;
    }
    return update;
}
} // namespace llaminar2
