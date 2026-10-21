/**
 * @file Qwen4ExpNgramAdmissionCases.h
 * @brief Focused full-source storage admission declarations, outside mathematical certification.
 *
 * These three cases reuse the canonical typed expander without appending to
 * the twenty mathematical cells. The explicit GPU/CPU pipeline is selected
 * before planning: table admission may not reduce its weights, rows or cache.
 */
#pragma once
#include "Qwen4ExpProductionParityCases.h"

namespace llaminar2::test::parity::qwen4exp
{
/** @return Full-model admission cases; no numerical campaign is declared for them. */
inline const std::vector<ModelParityCase> &qwen4ExpNgramAdmissionCases()
{
    static const auto cases = [] {
        std::vector<ModelParityCase> result;
        for (const auto mode : {Qwen4ExpParityTopology::CPU, Qwen4ExpParityTopology::CUDACPU})
        {
            auto definition = qwen4ExpProductionDefinition(mode);
            if (mode == Qwen4ExpParityTopology::CUDACPU)
                definition.topology.test_id += "_24GiB";
            definition.features.ngram_storage = {
                ImmutableRowTablePolicy{.storage = ImmutableRowStoragePreference::Automatic}};
            auto expanded = expandModelParityDefinition(definition);
            for (auto &cell : expanded)
            {
                // A focused admission declaration chooses one existing overlay
                // policy, rather than expanding four unrelated movement checks.
                if (cell.expert_overlay &&
                    *cell.expert_overlay != ModelParityExpertOverlayPolicy{}) continue;
                result.push_back(std::move(cell));
            }
        }
        auto resident = qwen4ExpProductionDefinition(Qwen4ExpParityTopology::CPU);
        resident.topology = {
            .test_id = "ROCm0_ROCm1_CPU0_PLEOnly_PP",
            .kind = ModelParityTopologyKind::RankLocalPipelineParallel,
            .participants = {{GlobalDeviceAddress::rocm(0), 0},
                             {GlobalDeviceAddress::rocm(1), 0},
                             {GlobalDeviceAddress::cpu(0), 0}},
            .collective = Collective::HETEROGENEOUS,
            .mpi_ranks = 1,
            .pipeline_stage_sizes = {1, 1, 1},
            .pipeline_weights = {1.0f, 1.0f, 46.0f}};
        resident.features.ngram_storage = {
            ImmutableRowTablePolicy{.storage = ImmutableRowStoragePreference::VRAM}};
        auto expanded = expandModelParityDefinition(resident);
        result.insert(result.end(), std::make_move_iterator(expanded.begin()),
                      std::make_move_iterator(expanded.end()));
        return result;
    }();
    return cases;
}

/**
 * @brief Project one selected case through the public runner configuration.
 * @param cell Exact declaration from qwen4ExpNgramAdmissionCases().
 * @param model_path Existing harness-resolved original model root.
 * @param rank Existing MPI owner.
 * @return Unchanged production policies plus the explicit CUDA24GiB ceiling.
 *
 * A larger CUDA card is deliberately capped at24GiB before planning so this
 * case proves Automatic cannot place the26.8GiB native table there. This is
 * a fixed fixture input, never a response to a failed admission candidate.
 */
inline OrchestrationConfig qwen4ExpNgramAdmissionConfig(
    const ModelParityCase &cell, const std::string &model_path, int rank)
{
    auto config = cell.makeOrchestrationConfig(model_path, rank);
    if (cell.topology.kind == ModelParityTopologyKind::RankLocalMultiDomain)
        config.max_gpu_memory_mb = 24u * 1024u;
    return config;
}

/** @return The concrete tier required by these explicit admission declarations. */
inline ImmutableRowStorage qwen4ExpNgramAdmissionExpectedStorage(const ModelParityCase &cell)
{
    if (!cell.ngram_table_policy) throw std::invalid_argument("Ngram admission case lacks requested storage");
    switch (cell.ngram_table_policy->storage)
    {
    case ImmutableRowStoragePreference::Automatic: return ImmutableRowStorage::RAM;
    case ImmutableRowStoragePreference::VRAM: return ImmutableRowStorage::VRAM;
    default: throw std::invalid_argument("Unexpected requested tier in focused ngram admission case");
    }
}
} // namespace llaminar2::test::parity::qwen4exp
