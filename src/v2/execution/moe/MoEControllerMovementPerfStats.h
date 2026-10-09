/**
 * @file MoEControllerMovementPerfStats.h
 * @brief Constant-cardinality observations of completed controller movement.
 *
 * The immutable physical publication owns the per-wave command, capacity and
 * economy evidence. PerfStats mirrors its ordered metadata with twelve fixed
 * counter families per participant. No transaction/expert ID becomes a key,
 * and no model payload or live device state is accessed by this projection.
 */
#pragma once

#include "MoEOptimizationStatus.h"
#include "utils/PerfStatsCollector.h"
#include <map>
#include <set>
#include <span>
#include <stdexcept>

namespace llaminar2
{
    /**
     * @brief Validate the complete retired wave before publishing any observation.
     * @param publication Authenticated physical and controller completion receipt.
     * @param edges Exact canonical edge span of that same wave.
     * @throws std::invalid_argument for mismatched identity, direction, capacity or cycles.
     */
    inline void validateMoEControllerCompletedMovement(
        const MoEOptimizationDeviceMovementPublication &publication,
        std::span<const MoEOptimizationMovementEdge> edges)
    {
        if (!publication.valid() || !publication.controller || publication.command_count != edges.size())
            throw std::invalid_argument("Controller movement requires one complete physical publication");
        const auto &receipt = *publication.controller;
        std::uint64_t promotions = 0, demotions = 0, same = 0, cross_rank = 0, cross_backend = 0;
        std::map<std::size_t, std::vector<const MoEOptimizationMovementEdge *>> cycles;
        std::set<std::pair<int, int>> experts;
        for (const auto &edge : edges)
        {
            const bool direction =
                (edge.direction == MoEOptimizationMovementDirection::SamePriority && edge.source_priority == edge.destination_priority) ||
                (edge.direction == MoEOptimizationMovementDirection::Promotion && edge.destination_priority < edge.source_priority) ||
                (edge.direction == MoEOptimizationMovementDirection::Demotion && edge.destination_priority > edge.source_priority);
            const bool axis = edge.axis == MoEOptimizationMovementAxis::TierResidency ||
                edge.axis == MoEOptimizationMovementAxis::ParticipantPlacement || edge.axis == MoEOptimizationMovementAxis::Combined;
            if (!edge.valid() || edge.authority != MoEOptimizationAuthority::Device ||
                edge.transaction != publication.transaction || edge.candidate_epoch != publication.candidate_epoch ||
                !direction || !axis || edge.blocking_inference || edge.estimated_weight_bytes == 0 || edge.source_participant == edge.destination_participant ||
                !edge.source_device.is_gpu() || !edge.destination_device.is_gpu() ||
                (edge.source_world_rank_known && edge.source_world_rank < 0) ||
                (edge.destination_world_rank_known && edge.destination_world_rank < 0) ||
                !experts.emplace(edge.layer, edge.expert).second)
                throw std::invalid_argument("Controller movement edge disagrees with its physical publication");
            promotions += edge.direction == MoEOptimizationMovementDirection::Promotion;
            demotions += edge.direction == MoEOptimizationMovementDirection::Demotion;
            same += edge.direction == MoEOptimizationMovementDirection::SamePriority;
            cross_rank += edge.source_world_rank_known && edge.destination_world_rank_known &&
                edge.source_world_rank != edge.destination_world_rank;
            cross_backend += edge.source_device.type != edge.destination_device.type;
            cycles[edge.cycle_index].push_back(&edge);
        }
        if (promotions != receipt.promotions || demotions != receipt.demotions || same != receipt.same_priority_moves ||
            cross_rank != receipt.cross_rank_moves || cross_backend != receipt.cross_backend_moves ||
            cycles.size() != receipt.physical_cycles)
            throw std::invalid_argument("Controller movement classification disagrees with completed edges");
        for (const auto &[index, members] : cycles)
        {
            std::map<int, std::int64_t> balance;
            for (const auto *edge : members)
            {
                if (edge->cycle_size != members.size() || edge->layer != members.front()->layer)
                    throw std::invalid_argument("Controller movement cycle has inconsistent geometry");
                ++balance[edge->source_participant];
                --balance[edge->destination_participant];
            }
            for (const auto &[participant, flow] : balance)
                if (flow != 0)
                    throw std::invalid_argument("Controller movement cycle does not conserve participant slots");
        }
    }

    /**
     * @brief Visit the versioned numeric evidence for one completed wave.
     * @param publication Completed physical receipt with time-policy metadata.
     * @param edges Exact canonical edge order belonging to the publication.
     * @param observe Receives name, counter value and a call-scoped uint64 word list.
     * @throws std::invalid_argument before invoking observe if the wave is invalid.
     *
     * All wave families bind the same full receipt. Edge identity encodes wire
     * enums explicitly: direction same/promotion/demotion=0/1/2, axis
     * tier/participant/combined=0/1/2, backend CUDA/ROCm=1/2. Unknown ranks
     * encode a zero presence word and zero value, independently of C++ padding.
     */
    template <typename Observer>
    void visitMoEControllerMovementEvidence(const MoEOptimizationDeviceMovementPublication &publication,
        std::span<const MoEOptimizationMovementEdge> edges, Observer &&observe)
    {
        validateMoEControllerCompletedMovement(publication, edges);
        const auto &r = *publication.controller;
        const auto words = {publication.transaction, publication.candidate_epoch, publication.command_count,
            publication.physical_payload_bytes, r.base_epoch, r.promotions, r.demotions, r.same_priority_moves,
            r.cross_domain_moves, r.cross_rank_moves, r.cross_backend_moves, r.snapshot_observations,
            r.priority_cost_before, r.priority_cost_after, r.same_priority_makespan_before, r.same_priority_makespan_after,
            r.accepted_cycles, r.physical_cycles, r.rejected_cycles, r.phase_tradeoff_candidates, r.improvement_floor_rejected_cycles,
            r.payoff_rejected_cycles, r.residency_rejected_cycles, r.economy.projected_service_gain_ns,
            r.economy.projected_transfer_and_repack_ns, r.economy.projected_inference_interference_ns,
            r.economy.projected_net_benefit_ns, r.changed_layers, r.layer_scan_start, r.layer_scan_next,
            r.edges_checked, r.participant_coordinates_checked, r.tier_coordinates_checked,
            r.malformed_edges, r.participant_flow_violations, r.tier_flow_violations};
        observe("dynamic_movement_transactions", 1.0, words);
        observe("dynamic_movement_commands", static_cast<double>(publication.command_count), words);
        observe("dynamic_physical_bytes", static_cast<double>(publication.physical_payload_bytes), words);
        observe("dynamic_promotions", static_cast<double>(r.promotions), words);
        observe("dynamic_demotions", static_cast<double>(r.demotions), words);
        observe("dynamic_same_priority_moves", static_cast<double>(r.same_priority_moves), words);
        observe("dynamic_cross_domain_moves", static_cast<double>(r.cross_domain_moves), words);
        observe("dynamic_cross_rank_moves", static_cast<double>(r.cross_rank_moves), words);
        observe("dynamic_cross_backend_moves", static_cast<double>(r.cross_backend_moves), words);
        observe("dynamic_capacity_conservation_certifications", 1.0, words);
        observe("dynamic_migration_edges", static_cast<double>(publication.command_count), words);
        for (const auto &e : edges)
            observe("dynamic_migration_edge_identities", 1.0, std::initializer_list<std::uint64_t>{
                e.transaction, e.candidate_epoch, static_cast<std::uint64_t>(e.layer), static_cast<std::uint64_t>(e.expert),
                e.cycle_index, e.cycle_size,
                e.direction == MoEOptimizationMovementDirection::SamePriority ? 0u :
                    e.direction == MoEOptimizationMovementDirection::Promotion ? 1u : 2u,
                e.axis == MoEOptimizationMovementAxis::TierResidency ? 0u :
                    e.axis == MoEOptimizationMovementAxis::ParticipantPlacement ? 1u : 2u,
                static_cast<std::uint64_t>(e.source_participant), static_cast<std::uint64_t>(e.destination_participant),
                static_cast<std::uint64_t>(static_cast<std::int64_t>(e.source_priority)),
                static_cast<std::uint64_t>(static_cast<std::int64_t>(e.destination_priority)),
                e.source_device.is_cuda() ? 1u : 2u, static_cast<std::uint64_t>(e.source_device.ordinal),
                e.destination_device.is_cuda() ? 1u : 2u, static_cast<std::uint64_t>(e.destination_device.ordinal),
                e.source_world_rank_known ? 1u : 0u,
                e.source_world_rank_known ? static_cast<std::uint64_t>(e.source_world_rank) : 0u,
                e.destination_world_rank_known ? 1u : 0u,
                e.destination_world_rank_known ? static_cast<std::uint64_t>(e.destination_world_rank) : 0u,
                e.estimated_weight_bytes, e.activation_count, 0});
    }

    /** @return Exact stable dimensions of the controller's completed-wave contract. */
    inline PerfStatsCollector::Tags moeControllerMovementTags()
    {
        return {{"policy_owner", "device"}, {"policy", "time_ns"}, {"encoding", "controller_wave_v1"},
            {"parallel_submission", "true"}, {"bounded_device_phases", "true"}, {"resident_external_waits", "0"},
            {"prearmed_cross_device_fanin", "true"}, {"retirement_attempts", "1"}, {"retirement_busy_retries", "0"},
            {"blocking_inference", "false"}, {"direction_counts_are_capacity_proof", "false"}};
    }

    /**
     * @brief Mirror completed work with fixed counter keys, independent of model lifetime.
     * @param publication Actual retired command and physical capacity receipt.
     * @param edges Exact live edge span; validated before the first counter mutation.
     * @param device Stable participant label owned by the controller configuration.
     */
    inline void recordMoEControllerCompletedMovement(const MoEOptimizationDeviceMovementPublication &publication,
        std::span<const MoEOptimizationMovementEdge> edges, const std::string &device)
    {
        if (!PerfStatsCollector::isDomainEnabled("moe_overlay_controller"))
            return;
        visitMoEControllerMovementEvidence(publication, edges,
            [&](const char *name, double value, std::initializer_list<std::uint64_t> words)
            {
                PerfStatsCollector::addCounterWithSequence("moe_overlay_controller", name, value, words,
                    "maintenance", device, moeControllerMovementTags());
            });
    }
}
