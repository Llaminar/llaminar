/**
 * @file Qwen4ExpGPUAdaptiveParityCases.h
 * @brief Four explicit single-GPU/local-source adaptive diagnostics in the canonical matrix.
 *
 * The original48 main GGUF, every checkpoint, prefix lifecycle and threshold are
 * inherited unchanged. Eight GPU routed experts per layer is an authored input,
 * not a response to cache pressure. CPU retains complete final expert coverage.
 * Cell declaration grants no numerical, serving, performance or E2E certificate.
 */
#pragma once
#include "Qwen4ExpProductionParityCases.h"

namespace llaminar2::test::parity::qwen4exp
{
/** @return Exact C0 and bounded positive GPU/RAM requests; SSD remains mandatory in both. */
inline std::vector<ImmutableRowTablePolicy> qwen4ExpGPUAdaptiveStorageAxes()
{
    ImmutableRowTablePolicy zero{.storage=ImmutableRowStoragePreference::AdaptiveVRAMRAMSSD};
    auto bounded=zero;
    bounded.gpu_adaptive={.maximum_gpu_cache_rows=64,.gpu_cache_physical_byte_limit=1u<<20,
        .host={.maximum_cache_rows=64,.cache_physical_byte_limit=1u<<20}};
    return {zero,bounded};
}
/**
 * @brief Select only the already-supported one-GPU continuation with its local CPU source/expert tier.
 * @param mode CUDA+CPU or ROCm+CPU; CPU/PP/hybrid do not enter this diagnostic arm.
 * @return Existing canonical source and numerical definition, with explicit fixed higher-priority quota.
 * @throws std::invalid_argument For another topology, rather than substituting its continuation.
 */
inline ModelParityDefinition qwen4ExpGPUAdaptiveParityDefinition(Qwen4ExpParityTopology mode)
{
    if(mode!=Qwen4ExpParityTopology::CUDACPU && mode!=Qwen4ExpParityTopology::ROCmCPU)
        throw std::invalid_argument("GPU adaptive diagnostics require one CUDA/ROCm full-main continuation and local CPU source");
    auto definition=qwen4ExpProductionDefinition(mode);
    auto plan=std::make_shared<MoERoutedExpertPlacementPlan>(*definition.topology.expert_overlay_plan);
    plan->routed_tiers.front().max_experts_per_layer=8;
    definition.topology.expert_overlay_plan=std::move(plan);
    definition.topology.test_id+="_GpuExperts8";
    definition.features.ngram_storage=qwen4ExpGPUAdaptiveStorageAxes();
    return definition;
}
/**
 * @return Two native backends times C0/positive caches, retaining only existing StaticOrdinal/MTP-off cells.
 * @note This filters existing typed expanded policy values, never parses a generated test name or creates another expander.
 */
inline const std::vector<ModelParityCase> &qwen4ExpGPUAdaptiveProductionCases()
{
    static const auto cases=[] {
        std::vector<ModelParityCase> result;
        for(const auto topology:{Qwen4ExpParityTopology::CUDACPU,Qwen4ExpParityTopology::ROCmCPU})
        {
            auto expanded=expandModelParityDefinition(qwen4ExpGPUAdaptiveParityDefinition(topology));
            for(auto &cell:expanded)
                if(cell.expert_overlay==std::optional{ModelParityExpertOverlayPolicy{}} && cell.mtp==ModelParityMTP::Off)
                    result.push_back(std::move(cell));
        }
        if(result.size()!=4)throw std::logic_error("GPU adaptive diagnostic inventory must contain exactly four canonical cells");
        return result;
    }();
    return cases;
}
} // namespace llaminar2::test::parity::qwen4exp
