/**
 * @file Qwen4ExpMTPNumericalCandidateCases.h
 * @brief Private candidate projection of canonical paired-source numerical MTP cells.
 *
 * This declaration adds no default eligibility. The original main/admission
 * inventories remain immutable; a separately registered matrix would require
 * all original four main shards and the explicit source48 predictor. Each
 * active cell retains every production numerical, accepted-prefix and native
 * serial-row obligation. Declaring a cell is not qualification evidence.
 */
#pragma once
#include "Qwen4ExpProductionParityCases.h"

namespace llaminar2::test::parity::qwen4exp
{
/**
 * @brief Derive the unqualified paired-source definition without editing main cells.
 * @param topology Existing exact CPU, overlay or mixed-pipeline declaration.
 * @return Standard six-policy MTP definition, explicit mmap and depth15 capacity.
 *
 * The standard expander validates its complete potential overlay product before
 * selecting Static/Ordinal below. Therefore both inherited Dynamic projections
 * must admit one complete maximum-depth transaction even though these candidate
 * cells never select Dynamic expert movement. No numeric/economic threshold is
 * relaxed and no runtime maintenance is inserted into a Static cell.
 */
inline ModelParityDefinition qwen4ExpMTPNumericalCandidateDefinition(Qwen4ExpParityTopology topology)
{
    auto definition = qwen4ExpProductionDefinition(topology);
    // Its paired-source retained15 Off control is distinct from main-only Off.
    // Keep the complete numerical campaign without advertising the still
    // unqualified sidecar as a canonical runtime/control inventory authority.
    definition.inventory_role = ModelParityInventoryRole::NumericalDiagnostic;
    definition.model.shared_predictor = SharedPredictorGGUFSource{std::string(kPredictorModelPath)};
    definition.model.maximum_mtp_draft_depth = kModelParityRequiredMaximumMTPDepth;
    definition.features.mtp = ModelParityAxisProfile::Standard;
    definition.features.ngram_storage = {
        ImmutableRowTablePolicy{.storage = ImmutableRowStoragePreference::MemoryMapped}};
    for (auto *movement : {&definition.dynamic_rebalance.economic_movement,
                           &definition.dynamic_rebalance.economic_movement_and_observed_speedup})
    {
        movement->device_min_maintenance_period_tokens = kModelParityRequiredMaximumMTPDepth + 1;
        movement->device_initial_maintenance_period_tokens = kModelParityRequiredMaximumMTPDepth + 1;
    }
    return definition;
}

/**
 * @return Four canonical topologies ×six canonical MTP policies, Static/Ordinal only.
 * @note Off is a paired-source retained15 control, distinct from the main-only Off cell.
 *
 * The canonical expander remains the only axis engine. Filtering already typed
 * expert policy avoids a second matrix, fabricated R1 capacity or name parsing.
 * No HTTP/E2E tag or public default is added. Depth1 is the first diagnostic;
 * Depth2/3/15 and Dynamic remain separate required future qualification cells.
 */
inline const std::vector<ModelParityCase> &qwen4ExpMTPNumericalCandidateCases()
{
    static const auto cases = [] {
        std::vector<ModelParityCase> result;
        for (const auto topology : {Qwen4ExpParityTopology::CPU, Qwen4ExpParityTopology::CUDACPU,
                                   Qwen4ExpParityTopology::ROCmCPU, Qwen4ExpParityTopology::CUDAROCmCPU})
            for (auto &cell : expandModelParityDefinition(qwen4ExpMTPNumericalCandidateDefinition(topology)))
            {
                if (cell.expert_overlay && *cell.expert_overlay != ModelParityExpertOverlayPolicy{})
                    continue;
                result.push_back(std::move(cell));
            }
        return result;
    }();
    return cases;
}
} // namespace llaminar2::test::parity::qwen4exp
