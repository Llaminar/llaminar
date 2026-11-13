/**
 * @file Qwen4ExpMatchedCacheEconomyCases.h
 * @brief Callable TP2 policy comparisons with one explicitly matched expert workload.
 *
 * These experiments compare whole RAM/mmap, independent consumer caches and an
 * original shared pool. GPU128/CPU384 is a declared experimental tier split,
 * never an admitted capacity or production default. PMA may reject it; source
 * admission and fresh numerical panels precede any timing. AutoExperts remains
 * the separate original HardwareProductionDefaults baseline. Cache ceilings are
 * byte requests, not claims of positive granted slots or favorable hit rates.
 */
#pragma once
#include "Qwen4ExpGPUAdaptiveParityCases.h"
#include <limits>
#include "kernels/qwen4exp/AdaptiveImmutableRowCacheState.h"
namespace llaminar2::test::parity::qwen4exp {
/**
 * @brief Keep the original callable TP2 topology while replacing only authored experiment fields.
 * @param target Existing CUDA2 or ROCm2 continuation, with its original CPU expert domain.
 * @param policy Existing public whole-source, ConsumerLocal or exact shared-pool policy.
 * @param gpuExperts Declared group expert ceiling; default128 is an experiment awaiting admission.
 * @param cpuExperts Matching original CPU tier ceiling; default384 completes the reviewed model workload.
 * @return Original model/history/arithmetic and full functional MTP/movement definition.
 * @throws std::invalid_argument For unsupported topology, ownership, or malformed declared ceilings.
 */
inline ModelParityDefinition qwen4ExpMatchedCacheDefinition(Qwen4ExpHardwareTarget target,
    const ImmutableRowTablePolicy& policy, int gpuExperts=128, int cpuExperts=384)
{
    if((target!=Qwen4ExpHardwareTarget::CUDA2 && target!=Qwen4ExpHardwareTarget::ROCm2) ||
       gpuExperts<=0 || cpuExperts<=0 || gpuExperts>std::numeric_limits<int>::max()-cpuExperts)
        throw std::invalid_argument("Matched cache experiments require original TP2 and explicit representable expert ceilings");
    policy.validate();
    auto definition=qwen4ExpHardwareTargetDefinition(target);
    if(policy.storage!=ImmutableRowStoragePreference::RAM && policy.storage!=ImmutableRowStoragePreference::MemoryMapped &&
       policy.storage!=ImmutableRowStoragePreference::AdaptiveVRAMRAMSSD)
        throw std::invalid_argument("Matched cache slice has no resident/hybrid/follower execution advertisement");
    if(policy.isGPUAdaptive() && policy.gpu_adaptive.placement.mode==AdaptiveGPUCachePlacementMode::DeclaredRankLocal){
        const auto& original=definition.features.ngram_storage.back().gpu_adaptive.placement;
        const auto& proposed=policy.gpu_adaptive.placement;
        if(proposed.coordinator!=original.coordinator || proposed.owners.size()!=original.owners.size())
            throw std::invalid_argument("Matched shared pool changed its original coordinator/physical owners");
        for(std::size_t index=0;index<original.owners.size();++index)
            if(proposed.owners[index].participant!=original.owners[index].participant)
                throw std::invalid_argument("Matched shared pool changed original participant order/identity");
    }
    auto plan=std::make_shared<MoERoutedExpertPlacementPlan>(*definition.topology.expert_overlay_plan);
    if(plan->routed_tiers.size()!=2 || !plan->routed_tiers.back().fallback)
        throw std::invalid_argument("Matched cache workload lost its original GPU/CPU expert tiers");
    plan->routed_tiers.front().max_experts_per_layer=gpuExperts;
    plan->routed_tiers.back().max_experts_per_layer=cpuExperts;
    for(auto& tier:plan->routed_tiers)tier.memory_budget_bytes=0;
    if(!validateMoERoutedExpertPlacementPlan(*plan).ok())
        throw std::invalid_argument("Matched cache workload has invalid original expert placement");
    definition.topology.expert_overlay_plan=std::move(plan);
    const std::string old="_GpuExperts8";const auto at=definition.topology.test_id.find(old);
    if(at==std::string::npos)throw std::logic_error("Matched cache topology lost original workload label");
    definition.topology.test_id.replace(at,old.size(),"_GpuExperts"+std::to_string(gpuExperts)+"_CPUExperts"+std::to_string(cpuExperts));
    definition.topology.test_id+="_MatchedCacheExperiment";
    definition.features.ngram_storage={policy};
    return definition;
}
/**
 * @brief Request a bounded actual cache using the existing placement and PMA geometry.
 * @param target Exact authored CUDA2 or ROCm2 continuation.
 * @param placement ConsumerLocal or original two-owner shared pool; no hybrid alias exists.
 * @param gpuBytes Explicit experimental physical-byte ceiling on EACH participating GPU.
 * @return SSD-backed GPU policy with no redundant RAM row cache and original production mmap windows.
 * @throws std::invalid_argument For unsupported placement, topology or a zero byte ceiling.
 */
inline ImmutableRowTablePolicy qwen4ExpMatchedGPUCachePolicy(Qwen4ExpHardwareTarget target,
    AdaptiveGPUCachePlacementMode placement,std::size_t gpuBytes)
{
    if((target!=Qwen4ExpHardwareTarget::CUDA2 && target!=Qwen4ExpHardwareTarget::ROCm2) || !gpuBytes)
        throw std::invalid_argument("Matched GPU cache needs original TP2 and explicit positive byte ceiling");
    ImmutableRowTablePolicy policy{.storage=ImmutableRowStoragePreference::AdaptiveVRAMRAMSSD};
    constexpr std::uint64_t maximumRows=::llaminar2::qwen4exp::adaptive_lookup::MaximumSlots; // PMA/source extent chooses actual rows.
    if(placement==AdaptiveGPUCachePlacementMode::ConsumerLocal){
        policy.gpu_adaptive.maximum_gpu_cache_rows=maximumRows;
        policy.gpu_adaptive.gpu_cache_physical_byte_limit=gpuBytes;
    }else if(placement==AdaptiveGPUCachePlacementMode::DeclaredRankLocal){
        const auto original=qwen4ExpHardwareTargetDefinition(target);
        policy.gpu_adaptive.placement=original.features.ngram_storage.back().gpu_adaptive.placement;
        for(auto& owner:policy.gpu_adaptive.placement.owners){owner.maximum_cache_rows=maximumRows;owner.cache_physical_byte_limit=gpuBytes;}
    }else throw std::invalid_argument("Matched GPU cache has no unsupported ownership alias");
    policy.validate();return policy;
}
/**
 * @return Twelve original TP2 cases: two hardware rows x RAM/mmap controls and two cache policies x two byte ceilings.
 * @details Every case declares the SAME GPU128/CPU384 expert workload, Off and
 *          zero retained draft capacity. The optional established panel binder
 *          may add its explicitly inactive source dependency. A C0 grant is
 *          admission evidence, not positive-cache performance. Fitting source,
 *          exact resolved quotas and fresh panel acceptance remain required.
 * @throws std::logic_error If original case expansion loses or duplicates the declared product.
 */
inline const std::vector<ModelParityCase>& qwen4ExpHardwareMatchedCacheEconomyCases()
{
    static const auto cases=[] {
        std::vector<ModelParityCase> result;
        for(const auto target:{Qwen4ExpHardwareTarget::CUDA2,Qwen4ExpHardwareTarget::ROCm2}){
            std::vector<ImmutableRowTablePolicy> policies{{.storage=ImmutableRowStoragePreference::RAM},
                                                        {.storage=ImmutableRowStoragePreference::MemoryMapped}};
            for(const auto placement:{AdaptiveGPUCachePlacementMode::ConsumerLocal,AdaptiveGPUCachePlacementMode::DeclaredRankLocal})
                for(const std::size_t bytes:{std::size_t{256}<<20,std::size_t{1}<<30})
                    policies.push_back(qwen4ExpMatchedGPUCachePolicy(target,placement,bytes));
            for(const auto& policy:policies){
                auto definition=qwen4ExpMatchedCacheDefinition(target,policy);
                definition.features.mtp=ModelParityAxisProfile::Disabled;
                definition.features.dynamic_speedup_witness=ModelParityDynamicSpeedupWitness::Disabled;
                definition.model.shared_predictor.reset();definition.model.maximum_mtp_draft_depth=0;
                for(auto cell:expandModelParityDefinition(definition))
                    if(cell.mtp==ModelParityMTP::Off && cell.expert_overlay==std::optional{ModelParityExpertOverlayPolicy{}})
                        result.push_back(std::move(cell));
            }
        }
        if(result.size()!=12)throw std::logic_error("Matched cache economy lost its exact original policy product");
        return result;
    }();return cases;
}
/**
 * @brief Derive RAM-backed GPU-cache comparisons from the same original matched workload.
 * @return Eight exact siblings (two TP2 rows, two placements, two byte ceilings), no new topology/product list.
 * @throws std::logic_error If original expansion loses its complete adaptive comparison subset.
 * @note Existing twelve cases stay unchanged; every sibling needs fresh source/accuracy/performance qualification.
 */
inline const std::vector<ModelParityCase>& qwen4ExpHardwareMatchedCacheRAMBackingEconomyCases(){
    static const auto cases=[]{
        std::vector<ModelParityCase> result;
        for(auto cell:qwen4ExpHardwareMatchedCacheEconomyCases())if(cell.ngram_table_policy->isGPUAdaptive()){
            cell.ngram_table_policy->gpu_adaptive.source_backing=AdaptiveImmutableRowBacking::RAM;
            cell.ngram_table_policy->gpu_adaptive.host={};cell.ngram_table_policy->validate();result.push_back(std::move(cell));
        }
        if(result.size()!=8)throw std::logic_error("RAM-backed matched comparisons lost an original adaptive cell");return result;
    }();return cases;
}
} // namespace llaminar2::test::parity::qwen4exp
