/**
 * @file MoEOverlayRetainedParentComposer.cpp
 * @brief Implementation of shared ExpertOverlay retained-parent discovery.
 */

#include "MoEOverlayRetainedParentComposer.h"

#include "MoEOverlayRetainedActivationTransaction.h"
#include "../compute_stages/stages/MoEOverlayActivationPacketStages.h"
#include "../local_execution/graph/ComputeGraph.h"

#include <stdexcept>
#include <utility>
#include <vector>

namespace llaminar2
{
    std::optional<DeviceGraphExecutor::RetainedParentCompositionHook>
    makeMoEOverlayRetainedParentComposer(const ComputeGraph &graph)
    {
        if (graph.nativeCaptureEnvelope() ==
            GraphNativeCaptureEnvelope::DeviceOwnedTimelineTransaction)
        {
            /*
             * Timeline-aware packet stages already splice their waits and
             * publications into the top-level stream capture. Re-wrapping
             * that graph as children is both redundant and illegal on CUDA
             * when adaptive attention contributes conditional nodes.
             */
            return std::nullopt;
        }
        size_t continuation_dispatches = 0u;
        size_t continuation_returns = 0u;
        size_t follower_dispatches = 0u;
        size_t follower_returns = 0u;
        for (const std::string &node_name : graph.getExecutionOrder())
        {
            const ComputeNode *const node = graph.getNode(node_name);
            if (!node || !node->stage)
            {
                throw std::runtime_error(
                    "Retained ExpertOverlay parent discovery found an unresolved graph stage");
            }
            switch (node->stage->type())
            {
            case ComputeStageType::MOE_OVERLAY_ACTIVATION_DISPATCH_PACK:
                ++continuation_dispatches;
                break;
            case ComputeStageType::MOE_OVERLAY_ACTIVATION_RETURN_CONSUME:
                ++continuation_returns;
                break;
            case ComputeStageType::MOE_OVERLAY_ACTIVATION_DISPATCH_CONSUME:
                ++follower_dispatches;
                break;
            case ComputeStageType::MOE_OVERLAY_ACTIVATION_RETURN_PACK:
                ++follower_returns;
                break;
            default:
                break;
            }
        }

        const bool has_continuation_marker =
            continuation_dispatches != 0u || continuation_returns != 0u;
        const bool has_follower_marker =
            follower_dispatches != 0u || follower_returns != 0u;
        if (!has_continuation_marker && !has_follower_marker)
            return std::nullopt;
        if (has_continuation_marker && has_follower_marker)
        {
            throw std::runtime_error(
                "One physical endpoint graph cannot mix continuation and follower activation packet authority");
        }

        const bool continuation = has_continuation_marker;
        if ((continuation &&
             (continuation_dispatches == 0u || continuation_returns == 0u)) ||
            (!continuation &&
             (follower_dispatches == 0u || follower_returns == 0u)))
        {
            throw std::runtime_error(
                "Retained ExpertOverlay endpoint graph contains an incomplete activation packet protocol");
        }

        return DeviceGraphExecutor::RetainedParentCompositionHook{
            [continuation](
                IGPUGraphCapture &destination,
                const ComputeGraph &source_graph,
                std::span<const DeviceGraphExecutor::GraphSegmentCache::
                                    RetainedCaptureUnitTemplateView> units)
            {
                /*
                 * Convert executor-owned views without copying captures or
                 * stage names. Parent construction completes synchronously,
                 * before either borrowed span can change.
                 */
                std::vector<MoEOverlayRetainedCaptureUnit> source_units;
                source_units.reserve(units.size());
                for (const auto &unit : units)
                {
                    source_units.push_back({
                        .capture = unit.capture,
                        .stage_names = unit.stage_names,
                    });
                }
                if (continuation)
                {
                    MoEOverlayRetainedActivationTransaction::
                        buildContinuationFromCapturedUnits(
                            destination, source_graph, source_units);
                }
                else
                {
                    MoEOverlayRetainedActivationTransaction::
                        buildFollowerFromCapturedUnits(
                            destination, source_graph, source_units);
                }
                return true;
            }};
    }
} // namespace llaminar2
