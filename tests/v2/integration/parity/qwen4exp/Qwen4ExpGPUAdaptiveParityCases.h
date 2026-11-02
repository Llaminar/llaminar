/**
 * @file Qwen4ExpGPUAdaptiveParityCases.h
 * @brief Canonical local adaptive controls and explicit opposite-vendor storage-only diagnostics.
 *
 * The original48 main GGUF, every checkpoint, prefix lifecycle and threshold are
 * inherited unchanged. Eight GPU routed experts per layer is an authored input,
 * not a response to cache pressure. CPU retains complete final expert coverage.
 * Cell declaration grants no numerical, serving, performance or E2E certificate.
 */
#pragma once
#include "Qwen4ExpProductionParityCases.h"
#include "Qwen4ExpFixedHeadParityCases.h"

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
 * @brief Declare an opposite-vendor cache cohort without changing any compute/expert participant.
 * @param mode Original CUDA+CPU or ROCm+CPU full-main continuation.
 * @return C0 and bounded source-storage policies over the same original expert definition.
 * @note The separately registered ordinary-retained15 family below consumes this
 * declaration. Registration grants no numerical, movement or serving certificate.
 */
inline ModelParityDefinition qwen4ExpDeclaredGPUCacheSourceDefinition(Qwen4ExpParityTopology mode)
{
    auto definition=qwen4ExpGPUAdaptiveParityDefinition(mode);
    for(auto &policy:definition.features.ngram_storage)
    {
        auto &g=policy.gpu_adaptive;const auto rows=g.maximum_gpu_cache_rows;const auto bytes=g.gpu_cache_physical_byte_limit;
        const auto local=mode==Qwen4ExpParityTopology::CUDACPU?DeviceType::CUDA:DeviceType::ROCm;
        const auto remote=mode==Qwen4ExpParityTopology::CUDACPU?DeviceType::ROCm:DeviceType::CUDA;
        g.maximum_gpu_cache_rows=0;g.gpu_cache_physical_byte_limit=0;
        g.placement={.mode=AdaptiveGPUCachePlacementMode::DeclaredRankLocal,
            .owners={{{0,local,0},rows,bytes},{{0,remote,0},rows,bytes}},.coordinator=AdaptiveGPUCacheParticipant{0,local,0}};
        policy.validate();
    }
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
/**
 * @return Four explicit opposite-vendor storage cases: two singleton compute backends times C0/positive.
 * @details Reuse the exact original definition/expander, main/predictor and numerical
 * gates. Only StaticOrdinal and ordinary Off execute; retained15 is authored storage
 * capacity, never an advertisement of speculative execution or dynamic movement.
 */
inline const std::vector<ModelParityCase>& qwen4ExpOppositeVendorOrdinaryRetained15Cases(){
    static const auto cases=[] {
        std::vector<ModelParityCase> result;
        for(const auto mode:{Qwen4ExpParityTopology::CUDACPU,Qwen4ExpParityTopology::ROCmCPU})
            for(auto cell:expandModelParityDefinition(qwen4ExpDeclaredGPUCacheSourceDefinition(mode)))
                if(cell.expert_overlay==std::optional{ModelParityExpertOverlayPolicy{}} && cell.mtp==ModelParityMTP::Off){
                    cell.model.shared_predictor=SharedPredictorGGUFSource{std::string(kPredictorModelPath)};
                    cell.retained_mtp_draft_capacity=kModelParityRequiredMaximumMTPDepth;
                    cell.topology.test_id+="_R15";
                    result.push_back(std::move(cell));
                }
        if(result.size()!=4)throw std::logic_error("Opposite-vendor ordinary retained15 inventory lost its four authored cells");
        return result;
    }();
    return cases;
}
/**
 * @return Fixed1, fixed3 and Dynamic over both original vendor/cache pairs.
 * @details The canonical expander owns all policy construction. These active
 * cases retain the same fifteen-draft storage and serial-control identities as
 * the ordinary family above. Static expert ownership isolates the speculative
 * execution boundary; dynamic expert movement remains a separate obligation.
 * Registration is not a claim of native execution or numerical qualification.
 */
inline const std::vector<ModelParityCase>& qwen4ExpOppositeVendorMTPRetained15Cases(){
    static const auto cases=[] {
        std::vector<ModelParityCase> result;
        for(const auto mode:{Qwen4ExpParityTopology::CUDACPU,Qwen4ExpParityTopology::ROCmCPU}){
            auto definition=qwen4ExpDeclaredGPUCacheSourceDefinition(mode);
            definition.model.shared_predictor=SharedPredictorGGUFSource{std::string(kPredictorModelPath)};
            definition.model.maximum_mtp_draft_depth=kModelParityRequiredMaximumMTPDepth;
            definition.features.mtp=ModelParityAxisProfile::Standard;
            // Validate the expander's complete potential movement product. The
            // selected static cells do not enable maintenance or change quotas.
            for(auto* movement:{&definition.dynamic_rebalance.economic_movement,
                               &definition.dynamic_rebalance.economic_movement_and_observed_speedup}){
                movement->device_min_maintenance_period_tokens=kModelParityRequiredMaximumMTPDepth+1;
                movement->device_initial_maintenance_period_tokens=kModelParityRequiredMaximumMTPDepth+1;
            }
            for(auto cell:expandModelParityDefinition(definition)){
                if(cell.expert_overlay!=std::optional{ModelParityExpertOverlayPolicy{}} ||
                    (cell.mtp!=ModelParityMTP::Depth1 && cell.mtp!=ModelParityMTP::Depth3 &&
                     cell.mtp!=ModelParityMTP::DynamicDepth))continue;
                cell.topology.test_id+="_R15";
                result.push_back(std::move(cell));
            }
        }
        if(result.size()!=12)throw std::logic_error("Opposite-vendor active MTP inventory lost its twelve authored cells");
        return result;
    }();
    return cases;
}
/**
 * @return Canonical Dynamic-expert cells, independently selecting the original owner order.
 * @param order Original Ordinal or Random physical expert-bank order.
 * @details Each family retains the same source/cache/compute definitions and
 * Off/Fixed1/Fixed3/Dynamic MTP policies. Physical movement and economy evidence
 * remain mandatory even for MTP Off; no eligibility follows from declaration.
 */
inline const std::vector<ModelParityCase>& qwen4ExpOppositeVendorDynamicExpertRetained15Cases(RoutedExpertOwnerOrder order){
    if(order!=RoutedExpertOwnerOrder::Ordinal && order!=RoutedExpertOwnerOrder::Random)
        throw std::invalid_argument("Opposite-vendor Dynamic expert family requires an original owner-order policy");
    const auto expand=[](RoutedExpertOwnerOrder selected){
        std::vector<ModelParityCase> result;
        for(const auto mode:{Qwen4ExpParityTopology::CUDACPU,Qwen4ExpParityTopology::ROCmCPU}){
            auto definition=qwen4ExpDeclaredGPUCacheSourceDefinition(mode);
            definition.model.shared_predictor=SharedPredictorGGUFSource{std::string(kPredictorModelPath)};
            definition.model.maximum_mtp_draft_depth=kModelParityRequiredMaximumMTPDepth;
            definition.features.mtp=ModelParityAxisProfile::Standard;
            // Preserve the authored active family's complete transaction
            // cadence; neither storage placement nor a test wake changes it.
            for(auto* movement:{&definition.dynamic_rebalance.economic_movement,
                                &definition.dynamic_rebalance.economic_movement_and_observed_speedup}){
                movement->device_min_maintenance_period_tokens=kModelParityRequiredMaximumMTPDepth+1;
                movement->device_initial_maintenance_period_tokens=kModelParityRequiredMaximumMTPDepth+1;
            }
            const ModelParityExpertOverlayPolicy policy{.owner_order=selected,.movement=ModelParityExpertMovement::Dynamic};
            for(auto cell:expandModelParityDefinition(definition)){
                if(cell.expert_overlay!=std::optional{policy} ||
                   (cell.mtp!=ModelParityMTP::Off && cell.mtp!=ModelParityMTP::Depth1 &&
                    cell.mtp!=ModelParityMTP::Depth3 && cell.mtp!=ModelParityMTP::DynamicDepth))continue;
                cell.topology.test_id+="_R15";
                result.push_back(std::move(cell));
            }
        }
        if(result.size()!=16)throw std::logic_error("Opposite-vendor Dynamic expert family lost its sixteen canonical cases");
        return result;
    };
    static const auto ordinal=expand(RoutedExpertOwnerOrder::Ordinal);
    static const auto random=expand(RoutedExpertOwnerOrder::Random);
    return order==RoutedExpertOwnerOrder::Ordinal?ordinal:random;
}
/** @return Active-only native-exact views of the same canonical Dynamic expert families; Off controls remain in their numerical/export family. */
inline const std::vector<ModelParityCase>& qwen4ExpOppositeVendorDynamicExpertMTPRetained15Cases(RoutedExpertOwnerOrder order){
    if(order!=RoutedExpertOwnerOrder::Ordinal && order!=RoutedExpertOwnerOrder::Random)
        throw std::invalid_argument("Opposite-vendor Dynamic expert replay requires an original owner order");
    const auto active=[](RoutedExpertOwnerOrder selected){
        std::vector<ModelParityCase> result;
        for(const auto& cell:qwen4ExpOppositeVendorDynamicExpertRetained15Cases(selected))
            if(cell.mtp!=ModelParityMTP::Off)result.push_back(cell);
        if(result.size()!=12)throw std::logic_error("Opposite-vendor Dynamic expert replay lost its active twelve-case view");
        return result;
    };
    static const auto ordinal=active(RoutedExpertOwnerOrder::Ordinal);
    static const auto random=active(RoutedExpertOwnerOrder::Random);
    return order==RoutedExpertOwnerOrder::Ordinal?ordinal:random;
}

/** @brief Reuse one authored ConsumerLocal TP2 definition for source, execution and economy obligations.
 * @param backend Actual homogeneous CUDA or ROCm continuation with CPU expert coverage.
 * @return Original strict numerical definition and C0/C64 policy axes; no qualification follows.
 */
inline ModelParityDefinition qwen4ExpConsumerLocalTP2AdaptiveDefinition(DeviceType backend)
{
    const std::array ordinals{0,1};
    auto definition=qwen4ExpProductionDefinition(Qwen4ExpParityTopology::CPU);
    definition.topology=qwen4ExpFixedHeadParityTopology(backend,ordinals);
    definition.topology.test_id+="_ConsumerLocal";
    definition.features.ngram_storage=qwen4ExpGPUAdaptiveStorageAxes();
    return definition;
}

/** @return Canonical execution cells over the original TP2 topology, independent of source-only admission.
 * @param policy Exact static or dynamic expert policy; physical movement remains a separate obligation from MTP.
 * @note Off/1/3/Dynamic all retain15. Off dynamic representatives carry the existing matched speed proof;
 * active modes retain economic movement and their exact original Off control. No tolerance changes occur.
 */
inline const std::vector<ModelParityCase>& qwen4ExpConsumerLocalTP2ExecutionCases(ModelParityExpertOverlayPolicy policy)
{
    const auto expand=[](ModelParityExpertOverlayPolicy selected){
        std::vector<ModelParityCase> result;
        for(const auto backend:{DeviceType::CUDA,DeviceType::ROCm}){
            auto definition=qwen4ExpConsumerLocalTP2AdaptiveDefinition(backend);
            definition.model.shared_predictor=SharedPredictorGGUFSource{std::string(kPredictorModelPath)};
            definition.model.maximum_mtp_draft_depth=kModelParityRequiredMaximumMTPDepth;
            definition.features.mtp=ModelParityAxisProfile::Standard;
            definition.features.dynamic_speedup_witness=ModelParityDynamicSpeedupWitness::BothOwnerOrders;
            for(auto* movement:{&definition.dynamic_rebalance.economic_movement,
                                &definition.dynamic_rebalance.economic_movement_and_observed_speedup}){
                movement->device_min_maintenance_period_tokens=kModelParityRequiredMaximumMTPDepth+1;
                movement->device_initial_maintenance_period_tokens=kModelParityRequiredMaximumMTPDepth+1;
            }
            for(auto cell:expandModelParityDefinition(definition)){
                if(cell.expert_overlay!=std::optional{selected} ||
                   (cell.mtp!=ModelParityMTP::Off && cell.mtp!=ModelParityMTP::Depth1 &&
                    cell.mtp!=ModelParityMTP::Depth3 && cell.mtp!=ModelParityMTP::DynamicDepth))continue;
                cell.topology.test_id+="_R15";
                result.push_back(std::move(cell));
            }
        }
        if(result.size()!=16)throw std::logic_error("ConsumerLocal TP2 lost its sixteen execution obligations");
        return result;
    };
    static const auto ordinalStatic=expand(ModelParityExpertOverlayPolicy{});
    static const auto ordinalDynamic=expand({.owner_order=RoutedExpertOwnerOrder::Ordinal,.movement=ModelParityExpertMovement::Dynamic});
    static const auto randomDynamic=expand({.owner_order=RoutedExpertOwnerOrder::Random,.movement=ModelParityExpertMovement::Dynamic});
    if(policy==ModelParityExpertOverlayPolicy{})return ordinalStatic;
    if(policy.movement==ModelParityExpertMovement::Dynamic && policy.owner_order==RoutedExpertOwnerOrder::Ordinal)return ordinalDynamic;
    if(policy.movement==ModelParityExpertMovement::Dynamic && policy.owner_order==RoutedExpertOwnerOrder::Random)return randomDynamic;
    throw std::invalid_argument("ConsumerLocal TP2 requires its explicitly authored expert family");
}

/** @return Active-only native exact views of the SAME canonical execution cells; Off controls stay in numerical/export family. */
inline std::vector<ModelParityCase> qwen4ExpConsumerLocalTP2NativeExactCases(ModelParityExpertOverlayPolicy policy)
{
    std::vector<ModelParityCase> result;
    for(const auto& cell:qwen4ExpConsumerLocalTP2ExecutionCases(policy))if(cell.mtp!=ModelParityMTP::Off)result.push_back(cell);
    return result;
}
} // namespace llaminar2::test::parity::qwen4exp
