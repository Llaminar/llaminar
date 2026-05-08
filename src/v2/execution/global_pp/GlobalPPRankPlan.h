/**
 * @file GlobalPPRankPlan.h
 * @brief Per-rank execution plan for Global Pipeline Parallelism
 *
 * Derived from GlobalPPTopology, this tells each rank exactly what to do:
 * - Which stages to execute (and in what order)
 * - What transfers (sends/recvs) to perform between stages
 * - Weight shard info for each executed stage
 *
 * Key design principle: Each rank receives its own GlobalPPRankPlan and
 * executes it independently, without needing cluster-wide context.
 * The plan is deterministic — identical topologies produce identical plans.
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#pragma once

#include "GlobalPPTopology.h"
#include "../../execution/mpi_orchestration/RankExecutionPlan.h"
#include <string>
#include <vector>
#include <sstream>

namespace llaminar2
{

    // =========================================================================
    // RankStageAction
    // =========================================================================

    /**
     * @brief What this rank does for a specific Global PP stage
     */
    struct RankStageAction
    {
        int stage_id = -1;
        std::string domain_name; ///< Optional named execution domain for this stage

        /** @brief Role of this rank in the stage */
        enum class Role
        {
            EXECUTE,  ///< This rank executes computation for this stage
            IDLE,     ///< This rank is idle during this stage
        };
        Role role = Role::IDLE;

        // For EXECUTE role:
        int first_layer = -1;    ///< First layer index (inclusive)
        int last_layer = -1;     ///< Last layer index (inclusive)
        bool has_embedding = false;
        bool has_lm_head = false;

        // Inner parallelism (from stage spec)
        InnerParallelism inner_mode = InnerParallelism::SINGLE_DEVICE;
        std::vector<GlobalDeviceAddress> devices;
        std::vector<float> tp_weights;
        CollectiveBackendType backend = CollectiveBackendType::AUTO;

        // For global TP EXECUTE:
        bool is_global_tp = false;       ///< This is a multi-rank global TP stage
        int tp_rank_in_domain = -1;      ///< This rank's index in the TP domain
        int tp_domain_size = 0;          ///< Total participants in the TP domain
        WeightShardInfo weight_shard;    ///< Shard info for weight loading
        GlobalDeviceAddress device;      ///< Device for execution

        /** @brief Number of layers in this action */
        int layerCount() const
        {
            return (first_layer >= 0 && last_layer >= first_layer) ? (last_layer - first_layer + 1) : 0;
        }
    };

    // =========================================================================
    // RankTransferAction
    // =========================================================================

    /**
     * @brief What this rank sends/receives between stages
     */
    struct RankTransferAction
    {
        /** @brief Direction of the transfer for this rank */
        enum class Direction
        {
            SEND,  ///< This rank sends data to peer
            RECV,  ///< This rank receives data from peer
            LOCAL_HANDOFF, ///< This rank hands activations between local stage/domain runners
            NONE,  ///< No transfer needed (data already available)
        };

        Direction direction = Direction::NONE;
        int peer_rank = -1;    ///< Send to or receive from
        int mpi_tag = 0;       ///< MPI tag for this transfer
        int from_stage = -1;   ///< Source stage
        int to_stage = -1;     ///< Destination stage
    };

    // =========================================================================
    // GlobalPPRankPlan
    // =========================================================================

    /**
     * @brief Complete execution plan for one rank in Global PP
     *
     * Contains an ordered sequence of steps that the rank must execute.
     * Steps alternate between stage executions and transfers.
     *
     * Example for rank 0 in a 3-stage pipeline (stages 0, 1, 2):
     *   1. EXECUTE_STAGE (stage 0, layers 0-9)
     *   2. TRANSFER (SEND hidden_state to rank 1)
     *   3. TRANSFER (RECV result from rank 1, after stage 1)
     *   4. EXECUTE_STAGE (stage 2, layers 20-23, global TP)
     *
     * Example for rank 1:
     *   1. TRANSFER (RECV hidden_state from rank 0)
     *   2. EXECUTE_STAGE (stage 1, layers 10-19)
     *   3. TRANSFER (SEND result to rank 0)
     *   4. EXECUTE_STAGE (stage 2, layers 20-23, global TP)
     */
    struct GlobalPPRankPlan
    {
        int rank = -1; ///< This rank's MPI rank number

        /**
         * @brief One step in the execution plan
         */
        struct Step
        {
            enum class Type
            {
                EXECUTE_STAGE,
                TRANSFER,
            };
            Type type = Type::EXECUTE_STAGE;

            // For EXECUTE_STAGE:
            RankStageAction stage_action;

            // For TRANSFER:
            RankTransferAction transfer_action;
        };

        std::vector<Step> steps; ///< Ordered execution steps

        // =====================================================================
        // Queries
        // =====================================================================

        /** @brief Get all EXECUTE stages in this plan */
        std::vector<const RankStageAction *> executeStages() const
        {
            std::vector<const RankStageAction *> result;
            for (const auto &step : steps)
            {
                if (step.type == Step::Type::EXECUTE_STAGE &&
                    step.stage_action.role == RankStageAction::Role::EXECUTE)
                {
                    result.push_back(&step.stage_action);
                }
            }
            return result;
        }

        /** @brief Get all TRANSFER steps in this plan */
        std::vector<const RankTransferAction *> transferActions() const
        {
            std::vector<const RankTransferAction *> result;
            for (const auto &step : steps)
            {
                if (step.type == Step::Type::TRANSFER &&
                    step.transfer_action.direction != RankTransferAction::Direction::NONE)
                {
                    result.push_back(&step.transfer_action);
                }
            }
            return result;
        }

        /** @brief Check if this rank executes any stages */
        bool hasWork() const { return !executeStages().empty(); }

        /** @brief Human-readable summary */
        std::string toString() const
        {
            std::ostringstream oss;
            oss << "GlobalPPRankPlan { rank=" << rank << ", steps=" << steps.size() << "\n";
            for (size_t i = 0; i < steps.size(); ++i)
            {
                const auto &step = steps[i];
                oss << "  [" << i << "] ";
                if (step.type == Step::Type::EXECUTE_STAGE)
                {
                    const auto &sa = step.stage_action;
                    oss << "EXECUTE stage=" << sa.stage_id
                        << " layers=[" << sa.first_layer << "-" << sa.last_layer << "]";
                    if (!sa.domain_name.empty()) oss << " domain=" << sa.domain_name;
                    if (sa.is_global_tp) oss << " global_tp(rank_in_domain=" << sa.tp_rank_in_domain << ")";
                    if (sa.has_embedding) oss << " +emb";
                    if (sa.has_lm_head) oss << " +lm";
                }
                else
                {
                    const auto &ta = step.transfer_action;
                    if (ta.direction == RankTransferAction::Direction::SEND)
                    {
                        oss << "SEND to rank " << ta.peer_rank << " tag=" << ta.mpi_tag;
                    }
                    else if (ta.direction == RankTransferAction::Direction::RECV)
                    {
                        oss << "RECV from rank " << ta.peer_rank << " tag=" << ta.mpi_tag;
                    }
                    else if (ta.direction == RankTransferAction::Direction::LOCAL_HANDOFF)
                    {
                        oss << "LOCAL_HANDOFF rank " << ta.peer_rank << " tag=" << ta.mpi_tag;
                    }
                    else
                    {
                        oss << "TRANSFER (no-op)";
                    }
                    oss << " (stage " << ta.from_stage << " → " << ta.to_stage << ")";
                }
                oss << "\n";
            }
            oss << "}";
            return oss.str();
        }
    };

} // namespace llaminar2
