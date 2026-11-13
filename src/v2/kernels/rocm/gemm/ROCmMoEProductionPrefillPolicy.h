/**
 * @file ROCmMoEProductionPrefillPolicy.h
 * @brief Qualify installed complete-MoE measurements before exact lookup.
 *
 * Generated corpus/table bytes retain their original complete, full-width,
 * token-row, all-route meaning. Split phases, compact down columns, canonical
 * route publication and selected ownership must use the existing generic
 * policy until independently measured. This wrapper changes no numeric family,
 * generic geometry, K partition or generated artifact.
 */
#pragma once

#include "../../common/MoEPrefillProjectionExecution.h"
#include "ROCmMoEProductionPrefillOverlayGenerated.inc"

namespace llaminar2::rocm
{
    /**
     * @brief Look up an existing exact row only for its measured operation.
     * @param operation Qualification projected from the actual phase/publication/grouping.
     * @param gateup_codebook Existing gate/up policy codebook, without new aliases.
     * @param down_codebook Existing down policy codebook, without new aliases.
     * @param hidden_size Original model width; the arithmetic key remains full.
     * @param expert_width Complete intermediate width.
     * @param expert_count Logical descriptor-table width, never active experts.
     * @param top_k Original routes per token.
     * @param m Captured original token rows.
     * @param config Receives an exact match; untouched on every miss.
     * @return Whether the unchanged generated table matched this qualified key.
     */
    [[nodiscard]] inline bool selectROCmMoEProductionPrefillOverlayForOperation(
        MoEPrefillMeasuredOperation operation,
        std::uint8_t gateup_codebook, std::uint8_t down_codebook,
        int hidden_size, int expert_width, int expert_count, int top_k, int m,
        generated::ROCmMoEProductionPrefillOverlayConfig &config)
    {
        if (operation != MoEPrefillMeasuredOperation::CompleteFullWidthTokenRowsAllRoutes)
            return false;
        return generated::selectROCmMoEProductionPrefillOverlay(
            gateup_codebook, down_codebook, hidden_size, expert_width,
            expert_count, top_k, m, config);
    }
} // namespace llaminar2::rocm
