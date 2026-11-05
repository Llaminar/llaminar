/**
 * @file Qwen4ExpSharedContinuationMTPSourceCases.h
 * @brief Source-only active MTP projection of original matched shared cache seeds.
 * SSD and full-RAM backings retain their separately authored matched seeds.
 * Original cache-economy numerical/performance and Off source inventories remain
 * unchanged. Physical fit is a fresh native admission obligation, never inferred.
 */
#pragma once
#include "Qwen4ExpMatchedCacheEconomyCases.h"
#include "Qwen4ExpSourceAdmissionCases.h"
#include "Qwen4ExpSourceReuseEligibility.h"
namespace llaminar2::test::parity::qwen4exp {
/**
 * @brief Authenticate authored continuation membership before source-only expansion.
 * @param cell Original canonical seed; expert-only followers have no continuation placement.
 * @return Whether the existing shared compute-owner reuse contract applies.
 */
inline bool qwen4ExpSharedContinuationSourceSeed(const ModelParityCase& cell)
{
    if(cell.topology.mpi_ranks!=1 || !cell.topology.expert_overlay_plan ||
       !cell.ngram_table_policy || !cell.ngram_table_policy->isGPUAdaptive())return false;
    const auto config=cell.makeOrchestrationConfig(cell.model.model_path,0);
    const auto domains=config.executionDomainDefinitions();
    const auto& name=cell.topology.expert_overlay_plan->continuation_domain;
    const auto found=std::find_if(domains.begin(),domains.end(),[&](const auto& domain){return domain.name==name;});
    if(found==domains.end())throw std::logic_error("Original shared source seed lost its continuation domain");
    std::vector<PhysicalMemoryAllocatorIdentity> consumers;
    for(const auto& participant:found->participants){
        const auto device=participant.toLocalDeviceId();
        if(!device.is_gpu())return false;
        consumers.push_back({0,device});
    }
    return qwen4ExpSharedContinuationOwnerSet(cell.ngram_table_policy->gpu_adaptive.placement,consumers);
}
/**
 * @return Active1/3/dynamic retained15 siblings from existing SSD/RAM matched shared seeds.
 * @note Off already executes in its original family; no duplicate Off launch or new topology list.
 */
inline const std::vector<ModelParityCase>& qwen4ExpSharedContinuationMTPSourceAdmissionCases()
{
    static const auto cases=[]{
        std::vector<ModelParityCase> seeds;
        // The Off cases already belong to their original two source families.
        // Project only eligible shared-compute owners, preserving original
        // cache byte requests, expert quotas and immutable backing identity.
        for(const auto* original:{&qwen4ExpHardwareMatchedCacheEconomyCases(),
                                  &qwen4ExpHardwareMatchedCacheRAMBackingEconomyCases()})
            for(const auto& cell:*original)
                if(qwen4ExpSharedContinuationSourceSeed(cell))seeds.push_back(cell);
        std::vector<ModelParityCase> result;
        for(auto cell:qwen4ExpExpandSourceAdmissionMTP(seeds))
            if(cell.mtp!=ModelParityMTP::Off)result.push_back(std::move(cell));
        return result;
    }();return cases;
}
}
