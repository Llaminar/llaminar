/**
 * @file Qwen4ExpFixedHeadParityCases.h
 * @brief Exact full-model local-TP qualification of native ngram head shards.
 *
 * These cases retain the original main-model source, numerical checkpoints,
 * prefix proof and expert movement axes. An explicit GPU expert limit precedes
 * planning: lowest-priority table admission never changes expert grants to make
 * a case fit. Paired RAM cells use the same limit and ordered GPU membership.
 * Declaring a cell is not numerical, hardware or performance certification.
 */
#pragma once
#include "Qwen4ExpProductionParityCases.h"
#include <set>
#include <span>

namespace llaminar2::test::parity::qwen4exp
{
/**
 * @brief Declare homogeneous full-main TP with an independent CPU expert tier.
 * @param backend CUDA or ROCm; the continuation cannot mix backend types.
 * @param ordinals Actual communicator order, retained without ordinal sorting.
 * @return One public ExpertOverlay topology with eight GPU experts per layer.
 * @throws std::invalid_argument For unsupported degree, backend or duplicate IDs.
 *
 * The eight-expert limit is an explicit workload parameter shared by both table
 * modes. It neither estimates table bytes nor reserves capacity outside PMA.
 */
inline ModelParityTopologyDefinition qwen4ExpFixedHeadParityTopology(
    DeviceType backend, std::span<const int> ordinals)
{
    if ((backend != DeviceType::CUDA && backend != DeviceType::ROCm) ||
        (ordinals.size() != 2 && ordinals.size() != 4 && ordinals.size() != 8))
        throw std::invalid_argument("Fixed-head model qualification requires CUDA/ROCm TP2/4/8");
    std::set<int> unique;
    for (const int ordinal : ordinals)
        if (ordinal < 0 || !unique.insert(ordinal).second)
            throw std::invalid_argument("Fixed-head model qualification requires distinct nonnegative ordinals");
    const bool cuda = backend == DeviceType::CUDA;
    auto topology = qwen4ExpProductionTopology(cuda ? Qwen4ExpParityTopology::CUDACPU
                                                   : Qwen4ExpParityTopology::ROCmCPU);
    auto plan = std::make_shared<MoERoutedExpertPlacementPlan>(*topology.expert_overlay_plan);
    auto &continuation = plan->domains.front();
    continuation.scope = ExecutionDomainScope::RANK_LOCAL;
    continuation.participants.clear();
    continuation.weights.assign(ordinals.size(), 1.0f / static_cast<float>(ordinals.size()));
    topology.participants.clear();
    topology.test_id.clear();
    for (const int ordinal : ordinals)
    {
        const auto address = cuda ? GlobalDeviceAddress::cuda(ordinal) : GlobalDeviceAddress::rocm(ordinal);
        continuation.participants.push_back(address);
        topology.participants.push_back({address, 0});
        topology.test_id += std::string(cuda ? "CUDA" : "ROCm") + std::to_string(ordinal) + "_";
    }
    topology.participants.push_back({GlobalDeviceAddress::cpu(0), 0});
    topology.test_id += "CPU0_TP" + std::to_string(ordinals.size()) + "_ExpertOverlay_GpuExperts8";
    plan->continuation_domain_spec.setDensePolicy(DenseParallelPolicy::TensorParallel);
    plan->routed_tiers.front().max_experts_per_layer = 8;
    const auto validation = validateMoERoutedExpertPlacementPlan(*plan);
    if (!validation.ok()) throw std::invalid_argument("Invalid fixed-head model qualification topology");
    topology.expert_overlay_plan = std::move(plan);
    return topology;
}

/**
 * @brief Pair original RAM lookup with native fixed-head GPU shards.
 * @param backend Homogeneous GPU continuation backend.
 * @param ordinals Exact ordered local TP participants.
 * @return Original strict main-model definition with explicit matching topology.
 */
inline ModelParityDefinition qwen4ExpFixedHeadParityDefinition(
    DeviceType backend, std::span<const int> ordinals)
{
    auto definition = qwen4ExpProductionDefinition(Qwen4ExpParityTopology::CPU);
    definition.topology = qwen4ExpFixedHeadParityTopology(backend, ordinals);
    ImmutableRowTablePolicy shards{.storage = ImmutableRowStoragePreference::VRAM};
    shards.distribution = ImmutableRowTableDistribution::FixedHeadSharded;
    definition.features.ngram_storage = {
        ImmutableRowTablePolicy{.storage = ImmutableRowStoragePreference::RAM}, shards};
    return definition;
}

/**
 * @return Two backends x TP2/4/8 plus reversed TP2, retaining both table modes
 * and the canonical static/dynamic, ordinal/random expert axes (64 cells).
 * @note TP4/8 cases require actual corresponding hardware. Their registration
 * is a pending qualification obligation on smaller workers, never a green proof.
 */
inline const std::vector<ModelParityCase> &qwen4ExpFixedHeadProductionCases()
{
    static const auto cases = [] {
        std::vector<ModelParityCase> result;
        const std::array<std::vector<int>, 4> orders{{{0,1}, {0,1,2,3}, {0,1,2,3,4,5,6,7}, {1,0}}};
        for (const auto backend : {DeviceType::CUDA, DeviceType::ROCm})
            for (const auto &order : orders)
            {
                auto expanded = expandModelParityDefinition(qwen4ExpFixedHeadParityDefinition(backend, order));
                result.insert(result.end(), std::make_move_iterator(expanded.begin()),
                              std::make_move_iterator(expanded.end()));
            }
        return result;
    }();
    return cases;
}
} // namespace llaminar2::test::parity::qwen4exp
