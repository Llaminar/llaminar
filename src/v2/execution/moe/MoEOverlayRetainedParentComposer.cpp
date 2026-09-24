/**
 * @file MoEOverlayRetainedParentComposer.cpp
 * @brief Implementation of shared ExpertOverlay retained-parent discovery.
 */

#include "MoEOverlayRetainedParentComposer.h"

#include "MoEOverlayRetainedActivationTransaction.h"
#include "../compute_stages/stages/MoEOverlayActivationPacketStages.h"
#include "../compute_stages/stages/MoEOverlayTicketConsumeStage.h"
#include "../local_execution/graph/ComputeGraph.h"

#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace llaminar2
{
    namespace
    {
        /** @brief Exact endpoint lowering selected from concrete packet stages. */
        enum class RetainedEndpointCompositionKind : std::uint8_t
        {
            ScalarContinuation, ///< Parent inserts scalar lane timeline edges.
            ScalarFollower, ///< Parent inserts scalar follower timeline edges.
            StageOwned, ///< Child stages already own their complete ordering DAG.
        };
    } // namespace

    std::optional<MoEOverlayRetainedParentPlan>
    makeMoEOverlayRetainedParentPlan(const ComputeGraph &graph)
    {
        const GraphNativeCaptureEnvelope envelope =
            graph.nativeCaptureEnvelope();
        if (envelope ==
            GraphNativeCaptureEnvelope::DeviceOwnedTimelineTransaction)
        {
            /*
             * A device-owned timeline already captures every wait/publication in
             * one indivisible executable. It has neither child boundaries nor a
             * host ticket service for this lowerer to own.
             */
            return std::nullopt;
        }
        const bool heterogeneous_ticket_transaction =
            requiresHeterogeneousTicketSegmentation(envelope);
        const bool heterogeneous_ticket_authority =
            ownsHeterogeneousTicketBoundary(envelope);
        const bool heterogeneous_ticket_follower =
            followsHeterogeneousTicketBoundary(envelope);
        size_t continuation_dispatches = 0u;
        size_t continuation_returns = 0u;
        size_t follower_dispatches = 0u;
        size_t follower_returns = 0u;
        size_t stage_owned_dispatches = 0u;
        size_t stage_owned_returns = 0u;
        size_t manual_stages = 0u;
        size_t device_ingress_publishers = 0u;
        for (const std::string &node_name : graph.getExecutionOrder())
        {
            const ComputeNode *const node = graph.getNode(node_name);
            if (!node || !node->stage)
            {
                throw std::runtime_error(
                    "Retained ExpertOverlay parent discovery found an unresolved graph stage");
            }
            if (node->stage->isManualGraphBoundary())
            {
                ++manual_stages;
                if (heterogeneous_ticket_transaction)
                {
                    const StageBufferContract contract =
                        node->stage->bufferContract();
                    if (node->stage->manualGraphBoundaryScheduling() !=
                            ManualGraphBoundaryScheduling::
                                ConcurrentTicketService ||
                        !contract.allArenaReads().empty() ||
                        !contract.allWrites().empty())
                    {
                        /*
                         * The graph remains a valid explicitly segmented ticket
                         * transaction, but it is not safe to submit ahead of this
                         * host boundary. Declining the optimization preserves its
                         * declared production lifecycle without inventing a path.
                         */
                        return std::nullopt;
                    }
                    if (node->stage->concurrentManualFailureRole() ==
                        ConcurrentManualFailureRole::DeviceIngressPublisher)
                    {
                        ++device_ingress_publishers;
                    }
                }
            }
            if (dynamic_cast<const
                    MoEOverlayActivationDispatchPackStage *>(
                    node->stage.get()))
            {
                ++continuation_dispatches;
            }
            else if (dynamic_cast<const
                         MoEOverlayActivationReturnConsumeStage *>(
                         node->stage.get()))
            {
                ++continuation_returns;
            }
            else if (dynamic_cast<const
                         MoEOverlayActivationDispatchConsumeStage *>(
                         node->stage.get()))
            {
                ++follower_dispatches;
            }
            else if (dynamic_cast<const
                         MoEOverlayActivationReturnPackStage *>(
                         node->stage.get()))
            {
                ++follower_returns;
            }
            else if (dynamic_cast<const
                         MoEOverlayActivationDispatchPackBatchStage *>(
                         node->stage.get()))
            {
                ++stage_owned_dispatches;
            }
            else if (dynamic_cast<const
                         MoEOverlayActivationReturnConsumeBatchStage *>(
                         node->stage.get()))
            {
                ++stage_owned_returns;
            }

            if (node->stage->type() ==
                ComputeStageType::MOE_OVERLAY_TICKET_CONSUME)
            {
                if (!heterogeneous_ticket_transaction)
                    continue;
                const auto *const consumer =
                    dynamic_cast<const MoEOverlayTicketConsumeStage *>(
                        node->stage.get());
                if (!consumer ||
                    !consumer->params().canonical_route_ticket_storage ||
                    consumer->params().ticket_storage)
                {
                    /* Dense H2D ingress has no captured mapped readiness wait. */
                    return std::nullopt;
                }
            }
        }

        if (heterogeneous_ticket_authority &&
            (manual_stages == 0u || device_ingress_publishers == 0u))
        {
            return std::nullopt;
        }
        if (heterogeneous_ticket_follower && manual_stages != 0u)
        {
            return std::nullopt;
        }

        const bool has_continuation_marker =
            continuation_dispatches != 0u || continuation_returns != 0u;
        const bool has_follower_marker =
            follower_dispatches != 0u || follower_returns != 0u;
        const bool has_stage_owned_continuation =
            stage_owned_dispatches != 0u || stage_owned_returns != 0u;
        const unsigned endpoint_forms =
            static_cast<unsigned>(has_continuation_marker) +
            static_cast<unsigned>(has_follower_marker) +
            static_cast<unsigned>(has_stage_owned_continuation);
        if (endpoint_forms > 1u)
        {
            throw std::runtime_error(
                "One physical endpoint graph cannot mix scalar continuation, scalar follower, and stage-owned packet authority");
        }
        if ((has_continuation_marker &&
             (continuation_dispatches == 0u || continuation_returns == 0u)) ||
            (has_follower_marker &&
             (follower_dispatches == 0u || follower_returns == 0u)) ||
            (has_stage_owned_continuation &&
             (stage_owned_dispatches == 0u || stage_owned_returns == 0u)))
        {
            throw std::runtime_error(
                "Retained ExpertOverlay endpoint graph contains an incomplete activation packet protocol");
        }
        if ((heterogeneous_ticket_authority && has_follower_marker) ||
            (heterogeneous_ticket_follower &&
             (has_continuation_marker || has_stage_owned_continuation)))
        {
            throw std::runtime_error(
                "Retained ExpertOverlay packet stages disagree with the graph's heterogeneous authority role");
        }

        std::optional<RetainedEndpointCompositionKind> composition_kind;
        if (heterogeneous_ticket_transaction)
        {
            /*
             * Packet stages record their mapped wait/publication nodes while
             * their native child is captured. The retained parent therefore
             * orders complete children only at the real host-ticket cutpoints;
             * splitting scalar packet stages would duplicate their authority
             * and can separate adjacent layers into an uncomposable shape.
             */
            composition_kind = RetainedEndpointCompositionKind::StageOwned;
        }
        else if (has_continuation_marker)
        {
            composition_kind =
                RetainedEndpointCompositionKind::ScalarContinuation;
        }
        else if (has_follower_marker)
        {
            composition_kind =
                RetainedEndpointCompositionKind::ScalarFollower;
        }
        else if (has_stage_owned_continuation)
        {
            /*
             * A device-owned batch transaction is handled by its complete
             * graph envelope. An ordinary batch graph has no host cutpoint for
             * this retained-child composer to own.
             */
            return std::nullopt;
        }
        if (!composition_kind)
            return std::nullopt;

        DeviceGraphExecutor::RetainedParentCompositionHook composer{
            [kind = *composition_kind](
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
                switch (kind)
                {
                case RetainedEndpointCompositionKind::ScalarContinuation:
                    MoEOverlayRetainedActivationTransaction::
                        buildContinuationFromCapturedUnits(
                            destination, source_graph, source_units);
                    break;
                case RetainedEndpointCompositionKind::ScalarFollower:
                    MoEOverlayRetainedActivationTransaction::
                        buildFollowerFromCapturedUnits(
                            destination, source_graph, source_units);
                    break;
                case RetainedEndpointCompositionKind::StageOwned:
                    MoEOverlayRetainedActivationTransaction::
                        buildStageOwnedTransactionFromCapturedUnits(
                            destination, source_graph, source_units);
                    break;
                }
                return true;
            }};
        MoEOverlayRetainedParentPlan plan{
            .replay_policy = heterogeneous_ticket_authority
                                 ? DeviceGraphExecutor::GraphReplayPlanPolicy::
                                       RequireRetainedParentWithConcurrentTicketService
                                 : DeviceGraphExecutor::GraphReplayPlanPolicy::
                                       RequireRetainedParentComposition,
            .composer = std::move(composer),
        };
        if (!plan.valid())
        {
            throw std::runtime_error(
                "Retained ExpertOverlay parent discovery produced an invalid typed replay plan");
        }
        return plan;
    }
} // namespace llaminar2
