/**
 * @file Qwen4ExpPooledCurrentDemandCases.h
 * @brief Four explicit ordinary current-demand candidates over unchanged canonical seeds.
 *
 * CUDA TP2 with its original opposite-vendor storage owner has a pure result
 * reader at C0/C64. CUDA and ROCm matched256MiB shared-compute owners instead
 * have actual inline-owner readers. ROCm pure-reader owner subsets have their
 * own qualification obligation and are not invented by this projection.
 * Every candidate still needs native admission and separate numerical evidence.
 */
#pragma once
#include "Qwen4ExpMatchedCacheEconomyCases.h"
#include "Qwen4ExpSourceAdmissionCases.h"
#include "execution/local_execution/graph/ReadAheadScheduling.h"
#include <set>

namespace llaminar2::test::parity::qwen4exp {
/** @brief Distinct actual ownership obligations; neither grants runtime authority. */
enum class Qwen4ExpCurrentDemandRole { PureReader, InlineOwner };
/**
 * @brief Classify only the authored original owner/consumer geometry.
 * @param cell One selected canonical seed or its current-demand sibling.
 * @return The foreign reader's actual role; no device ordinal/test-name inference occurs.
 * @throws std::invalid_argument For unsupported or ambiguous ownership products.
 */
inline Qwen4ExpCurrentDemandRole qwen4ExpCurrentDemandRole(const ModelParityCase& cell)
{
    if(!cell.ngram_table_policy || !cell.ngram_table_policy->isGPUAdaptive())
        throw std::invalid_argument("Current-demand candidates require original adaptive shared owners");
    const auto& placement=cell.ngram_table_policy->gpu_adaptive.placement;
    if(placement.mode!=AdaptiveGPUCachePlacementMode::DeclaredRankLocal || !placement.coordinator ||
       placement.owners.empty())throw std::invalid_argument("Current-demand candidate lost its authored coordinator/owners");
    const auto config=cell.makeOrchestrationConfig(cell.model.model_path,0);
    const auto domains=config.executionDomainDefinitions();
    if(!cell.topology.expert_overlay_plan)throw std::invalid_argument("Current-demand seed lost original expert domains");
    const auto& continuation=cell.topology.expert_overlay_plan->continuation_domain;
    const auto found=std::find_if(domains.begin(),domains.end(),[&](const auto& domain){return domain.name==continuation;});
    if(found==domains.end() || found->participants.size()!=2)
        throw std::invalid_argument("Current-demand shared candidate requires original TP2 consumers");
    std::vector<AdaptiveGPUCacheParticipant> consumers;
    for(const auto& participant:found->participants){const auto device=participant.toLocalDeviceId();consumers.push_back({0,device.type,device.ordinal});}
    const auto member=[&](const auto& owner){return std::find(consumers.begin(),consumers.end(),owner.participant)!=consumers.end();};
    if(placement.owners.size()==consumers.size() && std::all_of(placement.owners.begin(),placement.owners.end(),member))
        return Qwen4ExpCurrentDemandRole::InlineOwner;
    if(placement.owners.size()==1 && !member(placement.owners.front()))
        return Qwen4ExpCurrentDemandRole::PureReader;
    throw std::invalid_argument("Current-demand owner subsets are outside this explicit original four-seed projection");
}
/**
 * @return CUDA pure-reader C0/C64 and CUDA/ROCm inline-owner256MiB candidates.
 * @note Only diagnostic identity changes. Original topology, sources, precision,
 * storage bytes, expert policy and Off/capacity0 requests remain byte-identical.
 */
inline const std::vector<ModelParityCase>& qwen4ExpPooledCurrentDemandCases()
{
    static const auto cases=[]{
        std::vector<ModelParityCase> result;
        const auto append=[&](const ModelParityCase& seed,Qwen4ExpCurrentDemandRole expected){
            if(seed.mtp!=ModelParityMTP::Off || seed.retained_mtp_draft_capacity!=0 ||
               seed.expert_overlay!=std::optional{ModelParityExpertOverlayPolicy{}} ||
               qwen4ExpCurrentDemandRole(seed)!=expected)
                throw std::logic_error("Current-demand projection changed its original StaticOrdinal/Off role");
            auto candidate=seed;
            candidate.topology.test_id+="_PooledEarlyCurrent";
            candidate.inventory_role=ModelParityInventoryRole::NumericalDiagnostic;
            result.push_back(std::move(candidate));
        };
        for(const auto& seed:qwen4ExpSharedCoordinatorCUDATP2AdaptiveSourceAdmissionSeeds())
            if(seed.ngram_table_policy->gpu_adaptive.source_backing==AdaptiveImmutableRowBacking::SSD)
                append(seed,Qwen4ExpCurrentDemandRole::PureReader);
        for(const auto& seed:qwen4ExpHardwareMatchedCacheEconomyCases()) {
            if(!seed.ngram_table_policy || !seed.ngram_table_policy->isGPUAdaptive())continue;
            const auto& placement=seed.ngram_table_policy->gpu_adaptive.placement;
            if(placement.mode==AdaptiveGPUCachePlacementMode::DeclaredRankLocal &&
               std::all_of(placement.owners.begin(),placement.owners.end(),[](const auto& owner){return owner.cache_physical_byte_limit==(std::size_t{256}<<20);}))
                append(seed,Qwen4ExpCurrentDemandRole::InlineOwner);
        }
        std::set<std::string> ids;for(const auto& cell:result)if(!ids.insert(cell.testName()).second)
            throw std::logic_error("Current-demand projection duplicated a canonical candidate identity");
        if(result.size()!=4)throw std::logic_error("Current-demand projection lost its four supported original seeds");
        return result;
    }();return cases;
}
/** @return Only active retained3 siblings from the existing source expander; no duplicate Off launch. */
inline const std::vector<ModelParityCase>& qwen4ExpPooledCurrentMTPDemandCases()
{
    static const auto cases=[]{
        auto expanded=qwen4ExpExpandSourceAdmissionMTP(qwen4ExpPooledCurrentDemandCases(), Qwen4ExpSourceMTPEnvelope::Retained3);
        std::erase_if(expanded,[](const auto& cell){return cell.mtp==ModelParityMTP::Off;});
        if(expanded.size()!=12)throw std::logic_error("Current-demand active MTP product lost its original four seed/three policy closure");
        return expanded;
    }();return cases;
}

}
