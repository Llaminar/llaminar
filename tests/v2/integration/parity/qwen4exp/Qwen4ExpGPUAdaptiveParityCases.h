/**
 * @file Qwen4ExpGPUAdaptiveParityCases.h
 * @brief Canonical local adaptive controls and explicit opposite-vendor storage-only diagnostics.
 *
 * The original48 main GGUF, every checkpoint, prefix lifecycle and threshold are
 * inherited unchanged. Pressure families retain their authored eight GPU experts;
 * production hardware experiments retain the original PMA-selected expert policy.
 * Cache admission never changes either expert declaration.
 * Cell declaration grants no numerical, serving, performance or E2E certificate.
 */
#pragma once
#include "Qwen4ExpProductionParityCases.h"
#include "Qwen4ExpFixedHeadParityCases.h"
#include "kernels/qwen4exp/AdaptiveImmutableRowCacheState.h"

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
/** @return Original ROCm TP2 compute/expert policy with one declared third storage-only owner. */
inline ModelParityDefinition qwen4ExpSharedCoordinatorTP2AdaptiveDefinition()
{
    const std::array ordinals{0,1};
    auto definition=qwen4ExpProductionDefinition(Qwen4ExpParityTopology::CPU);
    definition.topology=qwen4ExpFixedHeadParityTopology(DeviceType::ROCm,ordinals);
    definition.topology.test_id+="_SharedCoordinatorStorage2";
    definition.features.ngram_storage=qwen4ExpGPUAdaptiveStorageAxes();
    for(auto& policy:definition.features.ngram_storage){
        auto& request=policy.gpu_adaptive;
        const auto rows=request.maximum_gpu_cache_rows,bytes=request.gpu_cache_physical_byte_limit;
        request.maximum_gpu_cache_rows=0;request.gpu_cache_physical_byte_limit=0;
        request.placement={.mode=AdaptiveGPUCachePlacementMode::DeclaredRankLocal,
            .owners={{{0,DeviceType::ROCm,2},rows,bytes}},.coordinator=AdaptiveGPUCacheParticipant{0,DeviceType::ROCm,0}};
        policy.validate();
    }
    return definition;
}
/** @return The same original model/math/expert axes with CUDA TP2 and a declared ROCm-only storage owner. */
inline ModelParityDefinition qwen4ExpSharedCoordinatorCUDATP2AdaptiveDefinition()
{
    auto definition=qwen4ExpSharedCoordinatorTP2AdaptiveDefinition();
    const std::array ordinals{0,1};
    definition.topology=qwen4ExpFixedHeadParityTopology(DeviceType::CUDA,ordinals);
    definition.topology.test_id+="_SharedCoordinatorROCmStorage0";
    for(auto& policy:definition.features.ngram_storage){
        policy.gpu_adaptive.placement.coordinator=AdaptiveGPUCacheParticipant{0,DeviceType::CUDA,0};
        policy.gpu_adaptive.placement.owners.front().participant={0,DeviceType::ROCm,0};
        policy.validate();
    }
    return definition;
}


/**
 * @brief Expand execution obligations from the same declared shared source topology.
 * @param policy Authored StaticOrdinal, DynamicOrdinal or DynamicRandom expert policy.
 * @return C0/C64 crossed with Off, fixed1, fixed3 and dynamic MTP, all retaining15.
 * @throws std::invalid_argument For an unauthored expert policy.
 * @throws std::logic_error If the canonical product loses an execution obligation.
 * @details Source-admission Off remains its original zero-draft request. These
 * separately named execution controls retain the predictor and full graph
 * capacity so active MTP, numerical panels and matched timing have the same
 * physical configuration. Shared storage never grants movement or accuracy.
 */
inline const std::vector<ModelParityCase>& qwen4ExpSharedCoordinatorTP2ExecutionCases(
    ModelParityExpertOverlayPolicy policy)
{
    const auto expand=[](ModelParityExpertOverlayPolicy selected){
        auto definition=qwen4ExpSharedCoordinatorTP2AdaptiveDefinition();
        definition.model.shared_predictor=SharedPredictorGGUFSource{std::string(kPredictorModelPath)};
        definition.model.maximum_mtp_draft_depth=kModelParityRequiredMaximumMTPDepth;
        definition.features.mtp=ModelParityAxisProfile::Standard;
        definition.features.dynamic_speedup_witness=ModelParityDynamicSpeedupWitness::BothOwnerOrders;
        for(auto* movement:{&definition.dynamic_rebalance.economic_movement,
                            &definition.dynamic_rebalance.economic_movement_and_observed_speedup}){
            movement->device_min_maintenance_period_tokens=kModelParityRequiredMaximumMTPDepth+1;
            movement->device_initial_maintenance_period_tokens=kModelParityRequiredMaximumMTPDepth+1;
        }
        std::vector<ModelParityCase> result;
        for(auto cell:expandModelParityDefinition(definition)){
            if(cell.expert_overlay!=std::optional{selected} ||
               (cell.mtp!=ModelParityMTP::Off && cell.mtp!=ModelParityMTP::Depth1 &&
                cell.mtp!=ModelParityMTP::Depth3 && cell.mtp!=ModelParityMTP::DynamicDepth))continue;
            cell.topology.test_id+="_R15";
            result.push_back(std::move(cell));
        }
        if(result.size()!=8)throw std::logic_error("Shared coordinator TP2 lost its eight execution obligations");
        return result;
    };
    static const auto ordinalStatic=expand(ModelParityExpertOverlayPolicy{});
    static const auto ordinalDynamic=expand({.owner_order=RoutedExpertOwnerOrder::Ordinal,.movement=ModelParityExpertMovement::Dynamic});
    static const auto randomDynamic=expand({.owner_order=RoutedExpertOwnerOrder::Random,.movement=ModelParityExpertMovement::Dynamic});
    if(policy==ModelParityExpertOverlayPolicy{})return ordinalStatic;
    if(policy.movement==ModelParityExpertMovement::Dynamic && policy.owner_order==RoutedExpertOwnerOrder::Ordinal)return ordinalDynamic;
    if(policy.movement==ModelParityExpertMovement::Dynamic && policy.owner_order==RoutedExpertOwnerOrder::Random)return randomDynamic;
    throw std::invalid_argument("Shared coordinator TP2 requires its explicitly authored expert family");
}

/**
 * @brief Select native verifier byte replay from the original shared execution cells.
 * @param policy Same expert policy as the associated numerical and serial controls.
 * @return Six active cells; Off controls remain in the execution/export family.
 * @throws std::invalid_argument For an unauthored expert policy.
 */
inline std::vector<ModelParityCase> qwen4ExpSharedCoordinatorTP2NativeExactCases(
    ModelParityExpertOverlayPolicy policy)
{
    std::vector<ModelParityCase> result;
    for(const auto& cell:qwen4ExpSharedCoordinatorTP2ExecutionCases(policy))
        if(cell.mtp!=ModelParityMTP::Off)result.push_back(cell);
    return result;
}
/** @brief Exact six requested hardware rows; actual board identity remains native evidence. */
enum class Qwen4ExpHardwareTarget { ROCm1, ROCm2, ROCm4, CUDA1, CUDA2, CUDA2ROCm4 };
/** @brief Requested physical cache roles, never inferred from spare hardware. */
enum class Qwen4ExpHardwareNgramOwners { Continuation, Follower, Both };

/**
 * @brief Describe one of the six canonical hardware workloads through existing topology owners.
 * @param target One/two/four MI50, one/two RTX3090, or two RTX3090 plus four MI50 followers.
 * @param owners Mixed-row owner selection; homogeneous rows require Continuation.
 * @return Original model/math, RAM/SSD and bounded RAM/VRAM tiers, retained15 and existing MTP axes.
 * @throws std::invalid_argument For an unknown row or follower selection without a follower domain.
 * @details The four ROCm followers are actual expert workers in one rank-local
 *          RCCL domain. Continuation quota8 remains; follower quota8 is an
 *          additional real expert tier before original CPU backing. Declarations
 *          do not remove runtime guards or grant numerical/performance support.
 */
inline ModelParityDefinition qwen4ExpHardwareTargetDefinition(
    Qwen4ExpHardwareTarget target, Qwen4ExpHardwareNgramOwners owners = Qwen4ExpHardwareNgramOwners::Continuation)
{
    const bool mixed = target == Qwen4ExpHardwareTarget::CUDA2ROCm4;
    const bool cuda = target == Qwen4ExpHardwareTarget::CUDA1 ||
        target == Qwen4ExpHardwareTarget::CUDA2 || mixed;
    const int degree = target == Qwen4ExpHardwareTarget::ROCm4 ? 4 :
        (target == Qwen4ExpHardwareTarget::ROCm1 || target == Qwen4ExpHardwareTarget::CUDA1 ? 1 : 2);
    if (target != Qwen4ExpHardwareTarget::ROCm1 && target != Qwen4ExpHardwareTarget::ROCm2 &&
        target != Qwen4ExpHardwareTarget::ROCm4 && target != Qwen4ExpHardwareTarget::CUDA1 &&
        target != Qwen4ExpHardwareTarget::CUDA2 && !mixed)
        throw std::invalid_argument("Unknown canonical QFN hardware row");
    if ((owners != Qwen4ExpHardwareNgramOwners::Continuation &&
         owners != Qwen4ExpHardwareNgramOwners::Follower && owners != Qwen4ExpHardwareNgramOwners::Both) ||
        (!mixed && owners != Qwen4ExpHardwareNgramOwners::Continuation))
        throw std::invalid_argument("Canonical QFN follower cache selection requires its actual follower row");
    const auto backend = cuda ? DeviceType::CUDA : DeviceType::ROCm;
    auto definition = qwen4ExpGPUAdaptiveParityDefinition(
        cuda ? Qwen4ExpParityTopology::CUDACPU : Qwen4ExpParityTopology::ROCmCPU);
    if (degree != 1) {
        const std::array ordinals{0, 1, 2, 3};
        definition.topology = qwen4ExpFixedHeadParityTopology(backend,
            std::span<const int>(ordinals.data(), static_cast<std::size_t>(degree)));
    }
    if (mixed) {
        auto plan = std::make_shared<MoERoutedExpertPlacementPlan>(*definition.topology.expert_overlay_plan);
        auto follower = qwen4ExpParityDomain("rocm_followers", GlobalDeviceAddress::rocm(0), CollectiveBackendType::RCCL);
        follower.scope = ExecutionDomainScope::RANK_LOCAL;
        follower.participants.clear();
        follower.weights.assign(4, 0.25f);
        for (int ordinal = 0; ordinal < 4; ++ordinal) {
            follower.participants.push_back(GlobalDeviceAddress::rocm(ordinal));
            definition.topology.participants.insert(definition.topology.participants.end() - 1,
                {GlobalDeviceAddress::rocm(ordinal), 0});
        }
        plan->domains.insert(plan->domains.end() - 1, std::move(follower));
        plan->routed_tiers.insert(plan->routed_tiers.end() - 1,
            {.name = "rocm_follower_tier", .domain = "rocm_followers", .priority = 10,
             .max_experts_per_layer = 8, .memory_budget_bytes = 0, .fallback = false});
        plan->routed_tiers.back().priority = 20;
        if (!validateMoERoutedExpertPlacementPlan(*plan).ok())
            throw std::invalid_argument("Invalid canonical QFN four-follower expert workload");
        definition.topology.expert_overlay_plan = std::move(plan);
        definition.topology.test_id += "_ROCm0123Followers";
    }
    const char* role = owners == Qwen4ExpHardwareNgramOwners::Continuation ? "Continuation" :
        owners == Qwen4ExpHardwareNgramOwners::Follower ? "Follower" : "Both";
    definition.topology.test_id += std::string("_CanonicalHardware_Ngram") + role;
    auto gpu_axes = qwen4ExpGPUAdaptiveStorageAxes();
    // Isolate the VRAM row-cache contribution from the combined RAM+VRAM tier.
    // C64 is a pressure control in rows, never a tuned production memory budget.
    auto vram_only = gpu_axes.back();
    vram_only.gpu_adaptive.host = {};
    gpu_axes.insert(gpu_axes.end() - 1, std::move(vram_only));
    for (auto& policy : gpu_axes) {
        auto& request = policy.gpu_adaptive;
        const auto rows = request.maximum_gpu_cache_rows;
        const auto bytes = request.gpu_cache_physical_byte_limit;
        request.maximum_gpu_cache_rows = 0;
        request.gpu_cache_physical_byte_limit = 0;
        request.placement = {.mode = AdaptiveGPUCachePlacementMode::DeclaredRankLocal,
            .owners = {}, .coordinator = AdaptiveGPUCacheParticipant{0, backend, 0}};
        if (owners != Qwen4ExpHardwareNgramOwners::Follower)
            for (int ordinal = 0; ordinal < degree; ++ordinal)
                request.placement.owners.push_back({{0, backend, ordinal}, rows, bytes});
        if (mixed && owners != Qwen4ExpHardwareNgramOwners::Continuation)
            for (int ordinal = 0; ordinal < 4; ++ordinal)
                request.placement.owners.push_back({{0, DeviceType::ROCm, ordinal}, rows, bytes});
        policy.validate();
    }
    definition.features.ngram_storage = {
        ImmutableRowTablePolicy{.storage = ImmutableRowStoragePreference::RAM},
        ImmutableRowTablePolicy{.storage = ImmutableRowStoragePreference::MemoryMapped}};
    const auto host_axes = qwen4ExpHostAdaptiveStorageAxes();
    definition.features.ngram_storage.insert(definition.features.ngram_storage.end(), host_axes.begin(), host_axes.end());
    definition.features.ngram_storage.insert(definition.features.ngram_storage.end(), gpu_axes.begin(), gpu_axes.end());
    definition.model.shared_predictor = SharedPredictorGGUFSource{std::string(kPredictorModelPath)};
    definition.model.maximum_mtp_draft_depth = kModelParityRequiredMaximumMTPDepth;
    definition.features.mtp = ModelParityAxisProfile::Standard;
    definition.features.dynamic_speedup_witness = ModelParityDynamicSpeedupWitness::BothOwnerOrders;
    for (auto* movement : {&definition.dynamic_rebalance.economic_movement,
                           &definition.dynamic_rebalance.economic_movement_and_observed_speedup}) {
        movement->device_min_maintenance_period_tokens = kModelParityRequiredMaximumMTPDepth + 1;
        movement->device_initial_maintenance_period_tokens = kModelParityRequiredMaximumMTPDepth + 1;
    }
    return definition;
}
/**
 * @brief Supported whole-source controls for the six hardware rows, without colocation admission.
 * @return Twelve Off/StaticOrdinal RAM or SSD-mmap cases with production expert ceilings.
 * @details Zero expert ceilings use original PMA residency planning, not zero
 *          experts. No predictor, retained draft envelope or forced movement is
 *          inherited from correctness pressure controls. Numerical qualification
 *          of each exact resolved policy precedes performance eligibility.
 */
inline const std::vector<ModelParityCase>& qwen4ExpHardwareProductionSourceCases()
{
    static const auto cases = [] {
        std::vector<ModelParityCase> result;
        for (const auto target : {Qwen4ExpHardwareTarget::ROCm1, Qwen4ExpHardwareTarget::ROCm2,
                Qwen4ExpHardwareTarget::ROCm4, Qwen4ExpHardwareTarget::CUDA1,
                Qwen4ExpHardwareTarget::CUDA2, Qwen4ExpHardwareTarget::CUDA2ROCm4}) {
            auto definition = qwen4ExpHardwareTargetDefinition(target);
            auto plan = std::make_shared<MoERoutedExpertPlacementPlan>(*definition.topology.expert_overlay_plan);
            for (auto& tier : plan->routed_tiers) {
                tier.max_experts_per_layer = 0;
                tier.memory_budget_bytes = 0;
            }
            definition.topology.expert_overlay_plan = std::move(plan);
            // This new production policy must not retain the pressure family's quota8 label.
            const std::string quota_label = "_GpuExperts8";
            const auto quota_position = definition.topology.test_id.find(quota_label);
            if (quota_position == std::string::npos)
                throw std::logic_error("Production expert policy lost its authored pressure-label boundary");
            definition.topology.test_id.replace(quota_position, quota_label.size(), "_PMAExpertResidency");
            definition.topology.test_id += "_ProductionDefaults";
            definition.features.ngram_storage = {
                ImmutableRowTablePolicy{.storage = ImmutableRowStoragePreference::RAM},
                ImmutableRowTablePolicy{.storage = ImmutableRowStoragePreference::MemoryMapped}};
            definition.features.mtp = ModelParityAxisProfile::Disabled;
            definition.features.dynamic_speedup_witness = ModelParityDynamicSpeedupWitness::Disabled;
            definition.model.shared_predictor.reset();
            definition.model.maximum_mtp_draft_depth = 0;
            for (auto cell : expandModelParityDefinition(definition))
                if (cell.mtp == ModelParityMTP::Off &&
                    cell.expert_overlay == std::optional{ModelParityExpertOverlayPolicy{}})
                    result.push_back(std::move(cell));
        }
        if (result.size() != 12)
            throw std::logic_error("Six hardware production defaults lost RAM/SSD source controls");
        return result;
    }();
    return cases;
}
/**
 * @brief Derive requested ROCm4 cache comparisons from the original production hardware workload.
 * @return Original TP4 continuation, PMA-selected expert residency and real RAM/SSD backing controls.
 * @details Cache ceilings are experimental requests, not grants. All policies share
 * the same expert-placement declaration; actual resolved equality and fitting
 * positive cache slots remain admission/benchmark evidence. No TP2 quota is copied.
 */
inline ModelParityDefinition qwen4ExpConsumerLocalTP4AdaptiveDefinition()
{
    auto definition=qwen4ExpHardwareTargetDefinition(Qwen4ExpHardwareTarget::ROCm4);
    auto plan=std::make_shared<MoERoutedExpertPlacementPlan>(*definition.topology.expert_overlay_plan);
    for(auto& tier:plan->routed_tiers){tier.max_experts_per_layer=0;tier.memory_budget_bytes=0;}
    definition.topology.expert_overlay_plan=std::move(plan);
    const std::string old="_GpuExperts8";const auto at=definition.topology.test_id.find(old);
    if(at==std::string::npos)throw std::logic_error("TP4 cache workload lost original quota-label boundary");
    definition.topology.test_id.replace(at,old.size(),"_PMAExpertResidency");
    definition.topology.test_id+="_ConsumerLocalCacheExperiment";
    definition.features.ngram_storage={ImmutableRowTablePolicy{.storage=ImmutableRowStoragePreference::RAM},
        ImmutableRowTablePolicy{.storage=ImmutableRowStoragePreference::MemoryMapped}};
    for(const auto backing:{AdaptiveImmutableRowBacking::SSD,AdaptiveImmutableRowBacking::RAM})
        for(const std::size_t bytes:{std::size_t{256}<<20,std::size_t{1}<<30}){
            ImmutableRowTablePolicy policy{.storage=ImmutableRowStoragePreference::AdaptiveVRAMRAMSSD};
            policy.gpu_adaptive.maximum_gpu_cache_rows=::llaminar2::qwen4exp::adaptive_lookup::MaximumSlots;
            policy.gpu_adaptive.gpu_cache_physical_byte_limit=bytes;
            policy.gpu_adaptive.source_backing=backing;policy.gpu_adaptive.host={};policy.validate();
            definition.features.ngram_storage.push_back(std::move(policy));
        }
    return definition;
}
/**
 * @return Exact original AutoExperts TP4 controls and cache requests crossed with Off/1/3/Dynamic MTP.
 * @details AutoExperts can admit all experts on GPUs, leaving no actual tier
 * movement. Therefore this callable family is StaticOrdinal only. A separate
 * explicitly fitting nonzero CPU/GPU residency workload must retain the strict
 * physical-movement proof before dynamic-expert cases can be registered.
 */
inline const std::vector<ModelParityCase>& qwen4ExpConsumerLocalTP4ExecutionCases()
{
    static const auto cases=[]{
        std::vector<ModelParityCase> result;
        for(auto cell:expandModelParityDefinition(qwen4ExpConsumerLocalTP4AdaptiveDefinition()))
            if(cell.expert_overlay==std::optional{ModelParityExpertOverlayPolicy{}} &&
               (cell.mtp==ModelParityMTP::Off || cell.mtp==ModelParityMTP::Depth1 ||
                cell.mtp==ModelParityMTP::Depth3 || cell.mtp==ModelParityMTP::DynamicDepth)){
                cell.topology.test_id+="_R15";result.push_back(std::move(cell));
            }
        if(result.size()!=24)throw std::logic_error("ConsumerLocal TP4 lost its original backing/cache/MTP product");
        return result;
    }();return cases;
}
/** @return Active-only exact same cells; original Off remains a mandatory serial control. */
inline std::vector<ModelParityCase> qwen4ExpConsumerLocalTP4NativeExactCases()
{
    std::vector<ModelParityCase> result;
    for(const auto& cell:qwen4ExpConsumerLocalTP4ExecutionCases())if(cell.mtp!=ModelParityMTP::Off)result.push_back(cell);
    return result;
}
/**
 * @brief Declare a separate bounded TP4 workload with genuinely positive expert tiers.
 * @param backing Original SSD or full-RAM source; row formats and table identity are unchanged.
 * @return Original ROCm4 topology with GPU384/CPU128 ceilings and ConsumerLocal256MiB requests.
 * @throws std::invalid_argument For an unknown immutable source backing.
 * @details The canonical capacity planner must still admit every live runtime.
 * Metadata admission against recorded observations is not native movement or
 * economy proof. This workload never changes the all-resident AutoExperts family.
 */
inline ModelParityDefinition qwen4ExpConsumerLocalTP4MovementDefinition(AdaptiveImmutableRowBacking backing)
{
    if(backing!=AdaptiveImmutableRowBacking::SSD && backing!=AdaptiveImmutableRowBacking::RAM)
        throw std::invalid_argument("TP4 movement requires an original SSD or full-RAM source");
    auto definition=qwen4ExpConsumerLocalTP4AdaptiveDefinition();
    auto plan=std::make_shared<MoERoutedExpertPlacementPlan>(*definition.topology.expert_overlay_plan);
    if(plan->routed_tiers.size()!=2 || !plan->routed_tiers.back().fallback)
        throw std::logic_error("TP4 movement lost its original GPU/CPU expert tiers");
    plan->routed_tiers.front().max_experts_per_layer=384;
    plan->routed_tiers.back().max_experts_per_layer=128;
    if(!validateMoERoutedExpertPlacementPlan(*plan).ok())
        throw std::logic_error("TP4 movement has an invalid original expert placement");
    definition.topology.expert_overlay_plan=std::move(plan);
    const std::string original="_PMAExpertResidency";
    const auto at=definition.topology.test_id.find(original);
    if(at==std::string::npos)throw std::logic_error("TP4 movement lost its original workload label");
    definition.topology.test_id.replace(at,original.size(),"_GpuExperts384_CPUExperts128");
    definition.topology.test_id+="_Movement";
    std::erase_if(definition.features.ngram_storage,[&](const auto& policy){
        return !policy.isGPUAdaptive() || policy.gpu_adaptive.source_backing!=backing ||
            policy.gpu_adaptive.gpu_cache_physical_byte_limit!=(std::size_t{256}<<20);
    });
    if(definition.features.ngram_storage.size()!=1)
        throw std::logic_error("TP4 movement lost its exact original bounded ConsumerLocal source policy");
    definition.features.dynamic_speedup_witness=ModelParityDynamicSpeedupWitness::BothOwnerOrders;
    for(auto* movement:{&definition.dynamic_rebalance.economic_movement,
                        &definition.dynamic_rebalance.economic_movement_and_observed_speedup}){
        movement->device_min_maintenance_period_tokens=kModelParityRequiredMaximumMTPDepth+1;
        movement->device_initial_maintenance_period_tokens=kModelParityRequiredMaximumMTPDepth+1;
    }
    return definition;
}
/**
 * @brief Expand genuine two-tier movement independently of MTP execution depth.
 * @param order Existing ordinal or seeded-random original expert placement.
 * @return Eight canonical cells: two original backings crossed with Off/1/3/Dynamic, retained15.
 * @throws std::invalid_argument For an unsupported owner order.
 * @throws std::logic_error For a changed original policy product.
 * @note Off is the exact serial control and retains the existing speed witness;
 * active cells retain physical movement and native serial/grouped byte proof.
 */
inline const std::vector<ModelParityCase>& qwen4ExpConsumerLocalTP4MovementCases(RoutedExpertOwnerOrder order)
{
    const auto expand=[](RoutedExpertOwnerOrder selected){
        std::vector<ModelParityCase> result;
        const ModelParityExpertOverlayPolicy expert{.owner_order=selected,.movement=ModelParityExpertMovement::Dynamic};
        for(const auto backing:{AdaptiveImmutableRowBacking::SSD,AdaptiveImmutableRowBacking::RAM})
            for(auto cell:expandModelParityDefinition(qwen4ExpConsumerLocalTP4MovementDefinition(backing)))
                if(cell.expert_overlay==std::optional{expert} &&
                   (cell.mtp==ModelParityMTP::Off || cell.mtp==ModelParityMTP::Depth1 ||
                    cell.mtp==ModelParityMTP::Depth3 || cell.mtp==ModelParityMTP::DynamicDepth)){
                    cell.topology.test_id+="_R15";result.push_back(std::move(cell));
                }
        if(result.size()!=8)throw std::logic_error("TP4 movement lost its exact backing/MTP product");
        return result;
    };
    static const auto ordinal=expand(RoutedExpertOwnerOrder::Ordinal);
    static const auto random=expand(RoutedExpertOwnerOrder::Random);
    if(order==RoutedExpertOwnerOrder::Ordinal)return ordinal;
    if(order==RoutedExpertOwnerOrder::Random)return random;
    throw std::invalid_argument("TP4 movement requires original ordinal or seeded-random placement");
}
/**
 * @brief Select native exact views without relabeling active MTP as a direct PyTorch panel.
 * @param order Original ordinal or seeded-random expert placement.
 * @return Six active cells from the same canonical product; Off remains their exact serial control.
 * @throws std::invalid_argument For an unsupported original owner order.
 */
inline std::vector<ModelParityCase> qwen4ExpConsumerLocalTP4MovementNativeExactCases(RoutedExpertOwnerOrder order)
{
    std::vector<ModelParityCase> result;
    for(const auto& cell:qwen4ExpConsumerLocalTP4MovementCases(order))
        if(cell.mtp!=ModelParityMTP::Off)result.push_back(cell);
    return result;
}
} // namespace llaminar2::test::parity::qwen4exp
