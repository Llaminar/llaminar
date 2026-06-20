/**
 * @file GlobalPPRankPlanBuilder.cpp
 * @brief Implementation of per-rank plan builder for Global PP
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#include "GlobalPPRankPlanBuilder.h"
#include "../../utils/Logger.h"
#include <algorithm>
#include <cassert>

namespace llaminar2
{

    GlobalPPRankPlan GlobalPPRankPlanBuilder::build(const GlobalPPTopology &topology, int rank)
    {
        GlobalPPRankPlan plan;
        plan.rank = rank;

        // Walk through stages in order
        for (size_t i = 0; i < topology.stages.size(); ++i)
        {
            const auto &stage = topology.stages[i];

            // Before executing this stage, check if we need transfers from previous stage
            if (i > 0)
            {
                auto transfers = buildTransfersForRank(
                    topology,
                    topology.stages[i - 1].stage_id,
                    stage.stage_id,
                    rank);

                for (auto &ta : transfers)
                {
                    if (ta.direction != RankTransferAction::Direction::NONE)
                    {
                        GlobalPPRankPlan::Step step;
                        step.type = GlobalPPRankPlan::Step::Type::TRANSFER;
                        step.transfer_action = std::move(ta);
                        plan.steps.push_back(std::move(step));
                    }
                }
            }

            // Add stage execution (or idle)
            auto action = buildStageAction(stage, rank);

            // Only add execute steps for stages this rank participates in
            if (action.role == RankStageAction::Role::EXECUTE)
            {
                GlobalPPRankPlan::Step step;
                step.type = GlobalPPRankPlan::Step::Type::EXECUTE_STAGE;
                step.stage_action = std::move(action);
                plan.steps.push_back(std::move(step));
            }
        }

        LOG_DEBUG("[GlobalPPRankPlanBuilder] Built plan for rank " << rank
                                                                    << ": " << plan.steps.size() << " steps, "
                                                                    << plan.executeStages().size() << " execute stages, "
                                                                    << plan.transferActions().size() << " transfers");

        return plan;
    }

    RankStageAction GlobalPPRankPlanBuilder::buildStageAction(const GlobalPPStageSpec &stage, int rank)
    {
        RankStageAction action;
        action.stage_id = stage.stage_id;
        action.domain_name = stage.domain_name;

        if (!stage.rankParticipates(rank))
        {
            action.role = RankStageAction::Role::IDLE;
            return action;
        }

        action.role = RankStageAction::Role::EXECUTE;
        action.first_layer = stage.first_layer;
        action.last_layer = stage.last_layer;
        action.has_embedding = stage.has_embedding;
        action.has_lm_head = stage.has_lm_head;
        action.backend = stage.backend;

        if (stage.is_global_tp)
        {
            action.is_global_tp = true;
            action.tp_domain_size = static_cast<int>(stage.participating_ranks.size());

            // Find this rank's index in the participating ranks
            for (size_t i = 0; i < stage.participating_ranks.size(); ++i)
            {
                if (stage.participating_ranks[i] == rank)
                {
                    action.tp_rank_in_domain = static_cast<int>(i);
                    break;
                }
            }

            // Weight shard info for global TP
            action.weight_shard.shard_index = action.tp_rank_in_domain;
            action.weight_shard.total_shards = action.tp_domain_size;
            action.weight_shard.work_fraction = 1.0f / static_cast<float>(action.tp_domain_size);

            // Device for this rank. Prefer explicit per-rank domain devices
            // aligned with participating_ranks; fall back to the legacy single
            // per_rank_device field for older specs/tests.
            if (action.tp_rank_in_domain >= 0 &&
                action.tp_rank_in_domain < static_cast<int>(stage.per_rank_devices.size()))
            {
                action.device = stage.per_rank_devices[action.tp_rank_in_domain];
            }
            else
            {
                action.device = stage.per_rank_device;
            }
        }
        else
        {
            // Single-rank stage
            action.is_global_tp = false;
            action.inner_mode = stage.inner_mode;
            action.devices = stage.devices;
            action.tp_weights = stage.tp_weights;

            // No global TP sharding for single-rank stages
            action.weight_shard.shard_index = 0;
            action.weight_shard.total_shards = 1;
            action.weight_shard.work_fraction = 1.0f;

            // Device is the first device (or primary device for inner parallelism)
            if (!stage.devices.empty())
            {
                action.device = stage.devices[0];
            }
        }

        return action;
    }

    std::vector<RankTransferAction> GlobalPPRankPlanBuilder::buildTransfersForRank(
        const GlobalPPTopology &topology,
        int from_stage_id,
        int to_stage_id,
        int rank)
    {
        std::vector<RankTransferAction> result;

        // Find all transfers for this stage transition
        for (const auto &t : topology.transfers)
        {
            if (t.from_stage != from_stage_id || t.to_stage != to_stage_id)
            {
                continue;
            }

            if (t.kind == GlobalPPTransferKind::LOCAL_HANDOFF)
            {
                if (t.sender_rank == rank && t.receiver_rank == rank)
                {
                    RankTransferAction ta;
                    ta.direction = RankTransferAction::Direction::LOCAL_HANDOFF;
                    ta.peer_rank = rank;
                    ta.mpi_tag = t.mpi_tag;
                    ta.from_stage = t.from_stage;
                    ta.to_stage = t.to_stage;
                    result.push_back(ta);
                }
                continue;
            }

            if (t.isNoop())
            {
                continue;
            }

            if (t.sender_rank == rank)
            {
                RankTransferAction ta;
                ta.direction = RankTransferAction::Direction::SEND;
                ta.peer_rank = t.receiver_rank;
                ta.mpi_tag = t.mpi_tag;
                ta.from_stage = t.from_stage;
                ta.to_stage = t.to_stage;
                result.push_back(ta);
            }
            else if (t.receiver_rank == rank)
            {
                RankTransferAction ta;
                ta.direction = RankTransferAction::Direction::RECV;
                ta.peer_rank = t.sender_rank;
                ta.mpi_tag = t.mpi_tag;
                ta.from_stage = t.from_stage;
                ta.to_stage = t.to_stage;
                result.push_back(ta);
            }
        }

        return result;
    }

} // namespace llaminar2
