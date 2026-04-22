/**
 * @file GlobalPPTopology.cpp
 * @brief Implementation of Global PP topology data structures
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#include "GlobalPPTopology.h"
#include "../../utils/Logger.h"
#include <algorithm>
#include <set>
#include <sstream>

namespace llaminar2
{

    // =========================================================================
    // GlobalPPStageSpec
    // =========================================================================

    std::vector<std::string> GlobalPPStageSpec::validate() const
    {
        std::vector<std::string> errors;

        if (stage_id < 0)
        {
            errors.push_back("stage_id must be >= 0");
        }

        if (first_layer < 0)
        {
            errors.push_back("first_layer must be >= 0 (stage " + std::to_string(stage_id) + ")");
        }

        if (last_layer < first_layer)
        {
            errors.push_back("last_layer (" + std::to_string(last_layer) +
                             ") must be >= first_layer (" + std::to_string(first_layer) +
                             ") (stage " + std::to_string(stage_id) + ")");
        }

        if (is_global_tp)
        {
            if (participating_ranks.size() < 2)
            {
                errors.push_back("Global TP stage " + std::to_string(stage_id) +
                                 " must have >= 2 participating ranks, got " +
                                 std::to_string(participating_ranks.size()));
            }
            if (owning_rank >= 0)
            {
                errors.push_back("Global TP stage " + std::to_string(stage_id) +
                                 " should not have owning_rank set (got " +
                                 std::to_string(owning_rank) + ")");
            }
        }
        else
        {
            if (owning_rank < 0)
            {
                errors.push_back("Single-rank stage " + std::to_string(stage_id) +
                                 " must have a valid owning_rank");
            }
            if (!participating_ranks.empty())
            {
                errors.push_back("Single-rank stage " + std::to_string(stage_id) +
                                 " should not have participating_ranks set");
            }
        }

        // Validate inner parallelism config
        if (!is_global_tp && inner_mode == InnerParallelism::LOCAL_TP)
        {
            if (devices.size() < 2)
            {
                errors.push_back("LocalTP stage " + std::to_string(stage_id) +
                                 " must have >= 2 devices, got " +
                                 std::to_string(devices.size()));
            }
        }

        return errors;
    }

    // =========================================================================
    // GlobalPPTopology
    // =========================================================================

    GlobalPPTopology GlobalPPTopology::build(std::vector<GlobalPPStageSpec> specs,
                                            int num_layers, int num_ranks)
    {
        GlobalPPTopology topo;
        topo.total_layers = num_layers;
        topo.world_size = num_ranks;

        // Sort stages by stage_id
        std::sort(specs.begin(), specs.end(),
                  [](const GlobalPPStageSpec &a, const GlobalPPStageSpec &b)
                  { return a.stage_id < b.stage_id; });

        topo.stages = std::move(specs);

        // Derive transfers between adjacent stages
        for (size_t i = 0; i + 1 < topo.stages.size(); ++i)
        {
            const auto &from_stage = topo.stages[i];
            const auto &to_stage = topo.stages[i + 1];

            GlobalPPTransfer transfer;
            transfer.from_stage = from_stage.stage_id;
            transfer.to_stage = to_stage.stage_id;
            transfer.mpi_tag = 1000 + static_cast<int>(i); // Unique tag per transfer

            // Determine sender and receiver ranks
            if (from_stage.is_global_tp && to_stage.is_global_tp)
            {
                // Both are global TP — check which receiver ranks need data
                std::set<int> sender_set(from_stage.participating_ranks.begin(),
                                         from_stage.participating_ranks.end());

                std::vector<int> needs_data;
                for (int r : to_stage.participating_ranks)
                {
                    if (sender_set.find(r) == sender_set.end())
                    {
                        needs_data.push_back(r);
                    }
                }

                if (needs_data.empty())
                {
                    // All receiver ranks already have data — no transfer needed
                    continue;
                }

                // Fan-out from designated sender (first rank in sender domain)
                int src = from_stage.participating_ranks[0];
                for (int target_rank : needs_data)
                {
                    GlobalPPTransfer fan_out;
                    fan_out.from_stage = from_stage.stage_id;
                    fan_out.to_stage = to_stage.stage_id;
                    fan_out.sender_rank = src;
                    fan_out.receiver_rank = target_rank;
                    fan_out.mpi_tag = 1000 + static_cast<int>(i) * 100 + target_rank;
                    topo.transfers.push_back(fan_out);
                }
                continue;
            }
            else if (from_stage.is_global_tp)
            {
                // Global TP → single-rank: the owning rank of the next stage
                // already participated in global TP (or needs data from it)
                int dest = to_stage.owning_rank;
                if (from_stage.rankParticipates(dest))
                {
                    // Destination rank already has the data — no transfer needed
                    transfer.sender_rank = dest;
                    transfer.receiver_rank = dest;
                }
                else
                {
                    // Need to pick a sender from the global TP participants
                    transfer.sender_rank = from_stage.participating_ranks[0];
                    transfer.receiver_rank = dest;
                }
            }
            else if (to_stage.is_global_tp)
            {
                // Single-rank → global TP: the owning rank needs to send to all others
                // Actually, we only need to send to participants that DON'T have the data
                int src = from_stage.owning_rank;

                // Each non-source participant needs to receive
                // We create one transfer per non-source participant
                // For simplicity, we just record the primary transfer here;
                // the rank plan builder handles fan-out
                for (int target_rank : to_stage.participating_ranks)
                {
                    if (target_rank != src)
                    {
                        GlobalPPTransfer fan_out;
                        fan_out.from_stage = from_stage.stage_id;
                        fan_out.to_stage = to_stage.stage_id;
                        fan_out.sender_rank = src;
                        fan_out.receiver_rank = target_rank;
                        fan_out.mpi_tag = 1000 + static_cast<int>(i) * 100 + target_rank;
                        topo.transfers.push_back(fan_out);
                    }
                }
                // Mark the primary transfer as handled (skip the push below)
                continue;
            }
            else
            {
                // Single-rank → single-rank
                transfer.sender_rank = from_stage.owning_rank;
                transfer.receiver_rank = to_stage.owning_rank;
            }

            topo.transfers.push_back(transfer);
        }

        return topo;
    }

    std::vector<int> GlobalPPTopology::stagesForRank(int rank) const
    {
        std::vector<int> result;
        for (const auto &stage : stages)
        {
            if (stage.rankParticipates(rank))
            {
                result.push_back(stage.stage_id);
            }
        }
        return result;
    }

    const GlobalPPStageSpec *GlobalPPTopology::stageForLayer(int layer) const
    {
        for (const auto &stage : stages)
        {
            if (stage.hasLayer(layer))
            {
                return &stage;
            }
        }
        return nullptr;
    }

    bool GlobalPPTopology::rankParticipatesInStage(int rank, int stage_id) const
    {
        if (stage_id < 0 || stage_id >= static_cast<int>(stages.size()))
        {
            return false;
        }
        return stages[stage_id].rankParticipates(rank);
    }

    const GlobalPPTransfer *GlobalPPTopology::transferBetween(int from_stage, int to_stage) const
    {
        for (const auto &t : transfers)
        {
            if (t.from_stage == from_stage && t.to_stage == to_stage)
            {
                return &t;
            }
        }
        return nullptr;
    }

    std::vector<std::string> GlobalPPTopology::validate() const
    {
        std::vector<std::string> errors;

        if (stages.empty())
        {
            errors.push_back("Topology has no stages");
            return errors;
        }

        if (total_layers <= 0)
        {
            errors.push_back("total_layers must be > 0");
        }

        if (world_size <= 0)
        {
            errors.push_back("world_size must be > 0");
        }

        // Validate individual stages
        for (const auto &stage : stages)
        {
            auto stage_errors = stage.validate();
            errors.insert(errors.end(), stage_errors.begin(), stage_errors.end());
        }

        // Validate stage IDs are contiguous starting from 0
        for (size_t i = 0; i < stages.size(); ++i)
        {
            if (stages[i].stage_id != static_cast<int>(i))
            {
                errors.push_back("Stage IDs must be contiguous from 0. Expected " +
                                 std::to_string(i) + ", got " +
                                 std::to_string(stages[i].stage_id));
            }
        }

        // Validate layer coverage: all layers 0..total_layers-1 must be covered exactly once
        std::vector<int> layer_coverage(total_layers, 0);
        for (const auto &stage : stages)
        {
            for (int l = stage.first_layer; l <= stage.last_layer && l < total_layers; ++l)
            {
                layer_coverage[l]++;
            }
        }

        for (int l = 0; l < total_layers; ++l)
        {
            if (layer_coverage[l] == 0)
            {
                errors.push_back("Layer " + std::to_string(l) + " is not covered by any stage");
            }
            else if (layer_coverage[l] > 1)
            {
                errors.push_back("Layer " + std::to_string(l) + " is covered by " +
                                 std::to_string(layer_coverage[l]) + " stages (must be exactly 1)");
            }
        }

        // Validate layer ranges are contiguous across stages
        for (size_t i = 0; i + 1 < stages.size(); ++i)
        {
            if (stages[i].last_layer + 1 != stages[i + 1].first_layer)
            {
                errors.push_back("Gap or overlap between stage " + std::to_string(i) +
                                 " (last_layer=" + std::to_string(stages[i].last_layer) +
                                 ") and stage " + std::to_string(i + 1) +
                                 " (first_layer=" + std::to_string(stages[i + 1].first_layer) + ")");
            }
        }

        // Validate embedding and LM head
        if (!stages.empty())
        {
            if (!stages.front().has_embedding)
            {
                errors.push_back("First stage must have has_embedding = true");
            }
            if (!stages.back().has_lm_head)
            {
                errors.push_back("Last stage must have has_lm_head = true");
            }
        }

        // Validate rank references are within world_size
        for (const auto &stage : stages)
        {
            if (!stage.is_global_tp && stage.owning_rank >= world_size)
            {
                errors.push_back("Stage " + std::to_string(stage.stage_id) +
                                 " owning_rank (" + std::to_string(stage.owning_rank) +
                                 ") >= world_size (" + std::to_string(world_size) + ")");
            }
            if (stage.is_global_tp)
            {
                for (int r : stage.participating_ranks)
                {
                    if (r >= world_size)
                    {
                        errors.push_back("Stage " + std::to_string(stage.stage_id) +
                                         " participating_rank " + std::to_string(r) +
                                         " >= world_size (" + std::to_string(world_size) + ")");
                    }
                }
            }
        }

        return errors;
    }

    std::string GlobalPPTopology::toString() const
    {
        std::ostringstream oss;
        oss << "GlobalPPTopology {\n";
        oss << "  total_layers=" << total_layers << ", world_size=" << world_size
            << ", stages=" << stages.size() << ", transfers=" << transfers.size() << "\n";

        for (const auto &stage : stages)
        {
            oss << "  Stage " << stage.stage_id << ": layers ["
                << stage.first_layer << "-" << stage.last_layer << "]";

            if (stage.has_embedding) oss << " +embedding";
            if (stage.has_lm_head) oss << " +lm_head";

            if (stage.is_global_tp)
            {
                oss << " global_tp ranks={";
                for (size_t i = 0; i < stage.participating_ranks.size(); ++i)
                {
                    if (i > 0) oss << ",";
                    oss << stage.participating_ranks[i];
                }
                oss << "} device=" << stage.per_rank_device.toString();
            }
            else
            {
                oss << " rank=" << stage.owning_rank
                    << " inner=" << innerParallelismName(stage.inner_mode);
                if (!stage.devices.empty())
                {
                    oss << " devices={";
                    for (size_t i = 0; i < stage.devices.size(); ++i)
                    {
                        if (i > 0) oss << ",";
                        oss << stage.devices[i].toString();
                    }
                    oss << "}";
                }
            }

            oss << "\n";
        }

        for (const auto &t : transfers)
        {
            oss << "  Transfer: stage " << t.from_stage << " → " << t.to_stage
                << " (rank " << t.sender_rank << " → " << t.receiver_rank
                << " tag=" << t.mpi_tag;
            if (t.isNoop()) oss << " NO-OP";
            oss << ")\n";
        }

        oss << "}";
        return oss.str();
    }

} // namespace llaminar2
