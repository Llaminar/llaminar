/**
 * @file Qwen4ExpOwnerSubsetParityCases.h
 * @brief Real-model source, numerical and exact-MTP requests for original ngram owner subsets.
 *
 * Subsets select original continuation coordinates; they never resize a compute
 * communicator or replace an expert domain. Mixed rows retain all four actual
 * ROCm expert followers in the same declared cache owner population. Source
 * admission, numerical panels and native exact replay remain separate mandatory
 * obligations. Registration proves no hardware support or inference correctness.
 * The forty source, forty numerical and thirty active exact requests below are
 * unqualified candidates. Moving-GPU exact comparisons additionally require a
 * generic placement-comparison scope; the current CPUJoint-only scope is not
 * claimed as their qualification.
 */
#pragma once
#include "Qwen4ExpGPUAdaptiveParityCases.h"
#include <algorithm>
#include <cstdint>
#include <limits>

namespace llaminar2::test::parity::qwen4exp {
/**
 * @brief Select a nonempty original continuation subset without changing its coordinator.
 * @param target One authored hardware row, including its actual expert followers.
 * @param mask Nonzero bitset in original continuation participant order.
 * @return Four C0/C64 RAM/SSD pressure policies on the original model and expert topology.
 * @throws std::invalid_argument For empty/foreign selection or an unresolved native rank.
 * @throws std::logic_error If the original continuation or pressure product changes.
 * Mixed rows retain their original follower owners; a missing coordinator bit
 * deliberately exercises a pure-reader coordinator. C64 is a correctness ceiling,
 * never a performance budget or an admitted allocation grant.
 */
inline ModelParityDefinition qwen4ExpOwnerSubsetPressureDefinition(
    Qwen4ExpHardwareTarget target, std::uint64_t mask)
{
    const auto role = target == Qwen4ExpHardwareTarget::CUDA2ROCm4
        ? Qwen4ExpHardwareNgramOwners::Both : Qwen4ExpHardwareNgramOwners::Continuation;
    auto definition = qwen4ExpHardwareTargetDefinition(target, role);
    const auto& plan = *definition.topology.expert_overlay_plan;
    const auto domain = std::find_if(plan.domains.begin(), plan.domains.end(),
        [&](const auto& candidate) { return candidate.name == plan.continuation_domain; });
    if (domain == plan.domains.end() || domain->participants.empty())
        throw std::logic_error("Owner subset lost its original continuation domain");
    const auto degree = domain->participants.size();
    if (degree >= std::numeric_limits<std::uint64_t>::digits || !mask || (mask >> degree))
        throw std::invalid_argument("Owner subset requires only nonempty original continuation coordinates");
    if (domain->owner_rank < 0 || !domain->world_ranks.empty())
        throw std::invalid_argument("Owner subset requires an original known rank-local continuation");
    std::vector<ImmutableRowTablePolicy> axes;
    for (const auto& original : definition.features.ngram_storage) {
        if (!original.isGPUAdaptive() || original.gpu_adaptive.host.maximum_cache_rows) continue;
        for (const auto backing : {AdaptiveImmutableRowBacking::SSD, AdaptiveImmutableRowBacking::RAM}) {
            auto policy = original;
            auto& owners = policy.gpu_adaptive.placement.owners;
            std::erase_if(owners, [&](const auto& owner) {
                const auto participant = std::find_if(domain->participants.begin(), domain->participants.end(),
                    [&](const auto& address) { return owner.participant.world_rank == domain->owner_rank &&
                        owner.participant.device() == DeviceId{address.device_type,address.device_ordinal}; });
                return participant != domain->participants.end() &&
                    !(mask & (std::uint64_t{1} << std::distance(domain->participants.begin(), participant)));
            });
            policy.gpu_adaptive.source_backing = backing;
            policy.validate();
            axes.push_back(std::move(policy));
        }
    }
    if (axes.size() != 4) throw std::logic_error("Owner subset lost original C0/C64 RAM/SSD policies");
    definition.features.ngram_storage = std::move(axes);
    definition.topology.test_id += "_ContinuationOwnerSubset" + std::to_string(mask) + "Candidate";
    return definition;
}

/**
 * @return All26 nonempty continuation subsets across the six original hardware rows.
 * The mixed row keeps its actual four followers; masks enumerate only original
 * continuation membership. Cardinality follows the original domain, never TP2.
 */
inline std::vector<ModelParityDefinition> qwen4ExpOwnerSubsetDefinitions()
{
    std::vector<ModelParityDefinition> result;
    for (const auto target : kQwen4ExpHardwareTargets) {
        const auto first = qwen4ExpOwnerSubsetPressureDefinition(target, 1);
        const auto& plan = *first.topology.expert_overlay_plan;
        const auto domain = std::find_if(plan.domains.begin(), plan.domains.end(),
            [&](const auto& value) { return value.name == plan.continuation_domain; });
        const auto end = std::uint64_t{1} << domain->participants.size();
        for (std::uint64_t mask = 1; mask < end; ++mask)
            result.push_back(qwen4ExpOwnerSubsetPressureDefinition(target, mask));
    }
    return result;
}

/** @brief Original cache-axis projection, not a new storage mode. */
enum class Qwen4ExpOwnerSubsetCachePressure { Zero, Positive };
/** @brief One bounded real-model request; exhaustive masks stay in fast selector tests. */
struct Qwen4ExpOwnerSubsetModelRequest {
    Qwen4ExpHardwareTarget target;
    std::uint64_t continuation_mask;
    AdaptiveImmutableRowBacking backing;
    Qwen4ExpOwnerSubsetCachePressure cache;
    ModelParityExpertMovement movement;
};
/**
 * @return Ten authored model requests, orthogonal to the unchanged full-owner product.
 * Eight static requests cover both singleton roles, ROCm4 noncontiguous owners
 * and original mixed followers. Two moving siblings retain actual ordinal
 * movement. Existing single-device/full-owner families remain authoritative.
 */
inline constexpr std::array kQwen4ExpOwnerSubsetModelRequests{
    Qwen4ExpOwnerSubsetModelRequest{Qwen4ExpHardwareTarget::CUDA2,1,AdaptiveImmutableRowBacking::SSD,Qwen4ExpOwnerSubsetCachePressure::Zero,ModelParityExpertMovement::Static},
    Qwen4ExpOwnerSubsetModelRequest{Qwen4ExpHardwareTarget::CUDA2,2,AdaptiveImmutableRowBacking::RAM,Qwen4ExpOwnerSubsetCachePressure::Positive,ModelParityExpertMovement::Static},
    Qwen4ExpOwnerSubsetModelRequest{Qwen4ExpHardwareTarget::ROCm2,1,AdaptiveImmutableRowBacking::SSD,Qwen4ExpOwnerSubsetCachePressure::Positive,ModelParityExpertMovement::Static},
    Qwen4ExpOwnerSubsetModelRequest{Qwen4ExpHardwareTarget::ROCm2,2,AdaptiveImmutableRowBacking::RAM,Qwen4ExpOwnerSubsetCachePressure::Zero,ModelParityExpertMovement::Static},
    Qwen4ExpOwnerSubsetModelRequest{Qwen4ExpHardwareTarget::ROCm4,5,AdaptiveImmutableRowBacking::RAM,Qwen4ExpOwnerSubsetCachePressure::Positive,ModelParityExpertMovement::Static},
    Qwen4ExpOwnerSubsetModelRequest{Qwen4ExpHardwareTarget::ROCm4,2,AdaptiveImmutableRowBacking::SSD,Qwen4ExpOwnerSubsetCachePressure::Zero,ModelParityExpertMovement::Static},
    Qwen4ExpOwnerSubsetModelRequest{Qwen4ExpHardwareTarget::CUDA2ROCm4,1,AdaptiveImmutableRowBacking::SSD,Qwen4ExpOwnerSubsetCachePressure::Positive,ModelParityExpertMovement::Static},
    Qwen4ExpOwnerSubsetModelRequest{Qwen4ExpHardwareTarget::CUDA2ROCm4,2,AdaptiveImmutableRowBacking::RAM,Qwen4ExpOwnerSubsetCachePressure::Zero,ModelParityExpertMovement::Static},
    Qwen4ExpOwnerSubsetModelRequest{Qwen4ExpHardwareTarget::CUDA2,2,AdaptiveImmutableRowBacking::RAM,Qwen4ExpOwnerSubsetCachePressure::Positive,ModelParityExpertMovement::Dynamic},
    Qwen4ExpOwnerSubsetModelRequest{Qwen4ExpHardwareTarget::CUDA2ROCm4,1,AdaptiveImmutableRowBacking::SSD,Qwen4ExpOwnerSubsetCachePressure::Positive,ModelParityExpertMovement::Dynamic}};

/**
 * @return Ten Off/capacity0 source seeds projected from original typed definitions.
 * The existing source MTP expander adds explicit retained3 active requests. Each
 * source fixture also requires actual cold recreation from its original source
 * contract. Forty setup candidates are separate from inference qualification.
 */
inline const std::vector<ModelParityCase>& qwen4ExpOwnerSubsetSourceAdmissionSeeds()
{
    static const auto cases = [] {
        std::vector<ModelParityCase> result;
        for (const auto& request : kQwen4ExpOwnerSubsetModelRequests) {
            auto definition = qwen4ExpOwnerSubsetPressureDefinition(request.target,request.continuation_mask);
            definition.features.mtp = ModelParityAxisProfile::Disabled;
            definition.features.dynamic_speedup_witness = ModelParityDynamicSpeedupWitness::Disabled;
            definition.model.shared_predictor.reset();
            definition.model.maximum_mtp_draft_depth = 0;
            const auto before = result.size();
            for (auto cell : expandModelParityDefinition(definition)) {
                const auto& storage = cell.ngram_table_policy->gpu_adaptive;
                const bool positive = std::any_of(storage.placement.owners.begin(),storage.placement.owners.end(),
                    [](const auto& owner){return owner.maximum_cache_rows != 0;});
                if (cell.expert_overlay == std::optional{ModelParityExpertOverlayPolicy{
                        .owner_order=RoutedExpertOwnerOrder::Ordinal,.movement=request.movement}} &&
                    storage.source_backing == request.backing && positive ==
                        (request.cache == Qwen4ExpOwnerSubsetCachePressure::Positive))
                    result.push_back(std::move(cell));
            }
            if (result.size()!=before+1) throw std::logic_error("Owner subset representative lost its unique original projection");
        }
        return result;
    }();
    return cases;
}

/**
 * @brief Author the same real inference workload with a bounded retained3 predictor envelope.
 * @param policy Original StaticOrdinal or DynamicOrdinal policy; movement is never substituted.
 * @return32 static or8 moving Off/fixed1/fixed3/dynamic requests retaining the same original source and full model.
 * @throws std::invalid_argument For a policy outside these explicit candidate families.
 * @note Off retains3 too, so serial/grouped and local paired numerical comparisons
 * use the same physical request. Retained15 families remain unchanged elsewhere.
 */
inline std::vector<ModelParityCase> qwen4ExpOwnerSubsetExecutionCases(ModelParityExpertOverlayPolicy policy)
{
    if (policy.owner_order != RoutedExpertOwnerOrder::Ordinal ||
        (policy.movement != ModelParityExpertMovement::Static && policy.movement != ModelParityExpertMovement::Dynamic))
        throw std::invalid_argument("Owner subset inference candidates require original static/moving ordinal policies");
    std::vector<ModelParityCase> result;
    for (const auto& seed : qwen4ExpOwnerSubsetSourceAdmissionSeeds()) {
        if (seed.expert_overlay != std::optional{policy}) continue;
        for (const auto mtp : {ModelParityMTP::Off, ModelParityMTP::Depth1, ModelParityMTP::Depth3, ModelParityMTP::DynamicDepth}) {
            auto cell = seed;
            cell.inventory_role = ModelParityInventoryRole::NumericalDiagnostic;
            cell.model.shared_predictor = SharedPredictorGGUFSource{std::string(kPredictorModelPath)};
            cell.model.maximum_mtp_draft_depth = 3;
            cell.retained_mtp_draft_capacity = 3;
            cell.mtp = mtp;
            cell.topology.test_id += "_R3";
            result.push_back(std::move(cell));
        }
    }
    return result;
}

/** @brief Preserve actual native exact-MTP obligations independently from source and numerical panels.
 * @param policy The same explicit expert policy as execution controls.
 * @return24 static or6 moving active retained3 requests; moving candidates keep their real movement obligations.
 */
inline std::vector<ModelParityCase> qwen4ExpOwnerSubsetNativeExactCases(ModelParityExpertOverlayPolicy policy)
{
    auto result = qwen4ExpOwnerSubsetExecutionCases(policy);
    std::erase_if(result, [](const auto& cell) { return cell.mtp == ModelParityMTP::Off; });
    return result;
}
} // namespace llaminar2::test::parity::qwen4exp
