/**
 * @file LocalMegakernelLowering.h
 * @brief Declarative region-contraction helper for fusing local stage chains into megakernels.
 *
 * Replaces duplicated hand-rolled consumer-scanning and export-building loops
 * across Attention and FFN megakernel lowerers (ROCm & CUDA). Automatically
 * resolves exact outside exports through @ref ComputeGraph::discoverLocalRegionExports
 * and selects the matching @ref LocalRegionCapturePolicy from the graph's
 * active capture envelope unless explicitly overridden.
 */

#pragma once

#include "GraphComposer.h"
#include <memory>
#include <optional>
#include <vector>

namespace llaminar2
{
    /**
     * @brief Declarative contract for contracting a closed participant-local region into one megakernel stage.
     */
    struct LocalMegakernelRegionSpec
    {
        std::vector<GraphNodeRef> members;                 ///< Distinct nodes inside the region (must include @ref terminal).
        GraphNodeRef terminal;                             ///< Retained completion node whose identity survives contraction.
        std::unique_ptr<IComputeStage> replacement;        ///< Complete participant-local replacement stage.
        std::optional<LocalRegionCapturePolicy> capture_policy; ///< Defaults to @ref ComputeGraph::defaultLocalRegionCapturePolicy().
    };

    /**
     * @brief Atomically validate and contract a declared local region into one megakernel node.
     * @param graph Participant-local ComputeGraph under cold construction.
     * @param spec Complete typed region specification.
     * @return Typed @ref GraphNodeRef naming the retained replacement terminal.
     * @throws std::invalid_argument On invalid/foreign handles, duplicate members, missing terminal,
     *         collectives/manual boundaries inside the region, or cycles introduced by contraction.
     */
    GraphNodeRef lowerLocalMegakernelRegion(ComputeGraph &graph, LocalMegakernelRegionSpec spec);

} // namespace llaminar2
