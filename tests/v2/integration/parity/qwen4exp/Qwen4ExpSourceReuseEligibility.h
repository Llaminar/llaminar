/**
 * @file Qwen4ExpSourceReuseEligibility.h
 * @brief Original rank-bound role selection for cold source lifetime tests.
 * The source harness and source-only MTP projection use this same owner-membership
 * check. Expert-owner selection separately uses canonical known-rank routed
 * participants; equal local ordinals cannot authorize a foreign rank. Neither
 * selector grants admission, creates workers or changes runtime placement.
 */
#pragma once
#include "loaders/ResolvedImmutableRowTablePlacement.h"
#include "../ModelParityDefinition.h"
#include "execution/moe/MoEExpertOwnerMap.h"
#include <algorithm>
namespace llaminar2::test::parity::qwen4exp {
/** @brief Test admission obligation, never a runtime or fallback execution policy. */
enum class Qwen4ExpSourceRecreationProof { FromOriginalRoles, Required };
/**
 * @brief Recognize a nonempty original declared compute-owner subset.
 * @param declared Authored GPU cache owner/coordinator request.
 * @param consumers Exact original continuation consumer identities.
 * @return Whether every distinct declared owner and the coordinator belong to the original consumers.
 */
inline bool qwen4ExpSharedContinuationOwnerSet(const AdaptiveGPUCachePlacementRequest& declared,
    std::span<const PhysicalMemoryAllocatorIdentity> consumers)
{
    return declared.mode==AdaptiveGPUCachePlacementMode::DeclaredRankLocal &&
        !declared.owners.empty() && declared.owners.size()<=consumers.size() && declared.coordinator &&
        std::find(consumers.begin(),consumers.end(),PhysicalMemoryAllocatorIdentity{
            declared.coordinator->world_rank,declared.coordinator->device()})!=consumers.end() &&
        std::all_of(declared.owners.begin(),declared.owners.end(),[&](const auto& owner){
            return std::count_if(declared.owners.begin(),declared.owners.end(),[&](const auto& candidate){
                return candidate.participant==owner.participant;})==1;
        }) &&
        std::all_of(declared.owners.begin(),declared.owners.end(),[&](const auto& owner){
            return std::find(consumers.begin(),consumers.end(),
                PhysicalMemoryAllocatorIdentity{owner.participant.world_rank,owner.participant.device()})!=consumers.end();
        });
}
/**
 * @brief Select original shared-continuation source reuse, including a pure-reader coordinator.
 * @param placement Original resolved rank placement, absent for expert-only CPU followers.
 * @return True for nonempty owner subsets of original compute readers; expert attachments remain separate.
 */
inline bool qwen4ExpSourceReusesSharedContinuationOwners(const ResolvedImmutableRowTablePlacement* placement)
{
    return placement && placement->isGPUAdaptive() &&
        qwen4ExpSharedContinuationOwnerSet(placement->gpuAdaptiveRequest().placement,placement->consumers());
}
/**
 * @brief Select original continuation owners for actual runner retirement and recreation.
 * @param placement Same resolved source placement, absent on an expert-only CPU follower.
 * @return True for independent consumer-local owners or a declared shared continuation subset.
 *
 * ConsumerLocal has one retained source boundary per GPU and one shared host
 * provider. It needs the same cold recreation proof as shared GPU owners;
 * absence of a shared coordinator is not absence of a reusable source.
 */
inline bool qwen4ExpSourceReusesContinuationOwners(const ResolvedImmutableRowTablePlacement* placement)
{
    return placement && placement->isGPUAdaptive() && !placement->consumers().empty() &&
        (placement->gpuAdaptiveRequest().placement.mode==AdaptiveGPUCachePlacementMode::ConsumerLocal ||
         qwen4ExpSourceReusesSharedContinuationOwners(placement));
}
/**
 * @brief Select expert lookup owners using complete original rank/device/domain identity.
 * @param cell Original canonical model/topology and owner declaration.
 * @param placement SAME resolved model table placement, never a newly admitted owner.
 * @return Nonconsumer owners whose known original rank and device match a routed expert participant.
 * @throws std::exception If the original routed participant declaration is invalid.
 * @note Unknown ranks cannot authorize an attachment. Host/address resolution is
 * separately authenticated by the existing source execution-plan proof; this
 * selector never collapses a foreign rank into a local device ordinal.
 */
inline std::vector<PhysicalMemoryAllocatorIdentity> qwen4ExpSourceExpertLookupOwners(
    const ModelParityCase& cell,const ResolvedImmutableRowTablePlacement* placement)
{
    std::vector<PhysicalMemoryAllocatorIdentity> result;
    if(!placement || !placement->isGPUAdaptive() || !cell.topology.expert_overlay_plan)return result;
    const auto& plan=*cell.topology.expert_overlay_plan;
    const auto participants=MoEExpertOwnerMap::declaredParticipants(plan,{.reject_tensor_sharded_domains=false});
    for(const auto& owner:placement->gpuAdaptiveRequest().placement.owners){
        const PhysicalMemoryAllocatorIdentity allocator{owner.participant.world_rank,owner.participant.device()};
        if(std::find(placement->consumers().begin(),placement->consumers().end(),allocator)!=placement->consumers().end())continue;
        const bool expert=std::any_of(participants.begin(),participants.end(),[&](const auto& participant){
            return participant.world_rank_known && participant.world_rank==allocator.world_rank &&
                participant.device==allocator.device && participant.domain_name!=plan.base_model_domain &&
                participant.domain_name!=plan.continuation_domain;
        });
        if(expert)result.push_back(allocator);
    }
    return result;
}
}
