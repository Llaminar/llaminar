/**
 * @file Qwen4ExpLocalHotHeldSourceCases.h
 * @brief Four original TP2 policies for causal all-hit progress under held source I/O.
 *
 * Pool C0 forces a real source request from the C0 local consumer. The other
 * original consumer retains positive local capacity across public request reset.
 * No source, topology, token history, provider or arithmetic is synthesized.
 * Registration does not qualify native progress, numerical accuracy or timing.
 */
#pragma once
#include "Qwen4ExpCombinedLocalHotCases.h"

namespace llaminar2::test::parity::qwen4exp {
/**
 * @return Original CUDA/ROCm RAM/StaticOrdinal/R3 Off requests, one per positive
 * local coordinate. Source backing, mmap bounds, expert and host policy remain.
 */
inline std::vector<ModelParityCase> qwen4ExpLocalHotHeldSourceCases()
{
    std::vector<ModelParityCase> result;
    for (const auto target : {Qwen4ExpHardwareTarget::ROCm2, Qwen4ExpHardwareTarget::CUDA2})
    {
        const auto pressure = target == Qwen4ExpHardwareTarget::ROCm2
            ? Qwen4ExpOwnerSubsetCachePressure::Zero : Qwen4ExpOwnerSubsetCachePressure::Positive;
        const auto seed = qwen4ExpCombinedLocalHotC0CasesFromOriginalSeed(
            {target, 2, AdaptiveImmutableRowBacking::RAM, pressure, ModelParityExpertMovement::Static},
            {ModelParityMTP::Off}).front();
        const auto consumers = qwen4ExpLocalHotOriginalConsumers(seed);
        if (consumers.size() != 2)
            throw std::logic_error("Held-source proof requires its exact original two consumers");
        for (std::size_t positive = 0; positive < consumers.size(); ++positive)
        {
            auto cell = seed;
            auto& gpu = cell.ngram_table_policy->gpu_adaptive;
            gpu.local_hot_cache = AdaptiveGPULocalHotCacheRequest{
                .maximum_cache_rows = 0, .cache_physical_byte_limit = 0, .policy = gpu.gpu_policy,
                .consumer_overrides = {{consumers[positive], std::uint64_t{1} << 20, std::size_t{64} << 20}}};
            cell.ngram_table_policy->validate();
            const std::string old = "_CombinedLocalHotC0Candidate";
            if (!cell.topology.test_id.ends_with(old))
                throw std::logic_error("Held-source proof lost its original C0 projection");
            cell.topology.test_id.resize(cell.topology.test_id.size() - old.size());
            cell.topology.test_id += "_LocalHotPoolC0PositiveConsumer" + std::to_string(positive) + "HeldSourceCandidate";
            result.push_back(std::move(cell));
        }
    }
    return result;
}
} // namespace llaminar2::test::parity::qwen4exp
