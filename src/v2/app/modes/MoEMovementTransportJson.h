/**
 * @file MoEMovementTransportJson.h
 * @brief Terminal projection of the owner's complete movement and transport history.
 *
 * HTTP snapshots may precede the last drained maintenance transaction. This
 * export preserves the final canonical history so diagnostic observers can
 * bind an earlier HTTP prefix to bounded PerfStats evidence without mistaking
 * matching totals or endpoint epochs for complete transaction membership.
 * It reads only immutable metadata, never model or prefix-cache payloads.
 */
#pragma once

#include "MoEMovementLedgerJson.h"
#include "execution/moe/MoEControllerMovementPerfStats.h"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <map>
#include <utility>
#include <vector>

namespace llaminar2
{
    /**
     * @brief Export one drained model-lifetime ledger, including actual device bytes.
     * @param ledger Immutable owner snapshot acquired after maintenance retirement.
     * @return Versioned terminal evidence; follower economics remain leader-owned.
     * @throws std::invalid_argument if receipts are missing, reordered, duplicated,
     *         truncated, or disagree with their completed physical edge counts.
     *
     * The diagnostic does not invent an economy proof for a transport follower.
     * Every device participant instead retains the same completed publication
     * identity and exact physical extent alongside its existing edge history.
     */
    inline nlohmann::json moeMovementTransportJson(const MoEOptimizationMovementLedger &ledger)
    {
        if (!ledger.complete())
            throw std::invalid_argument("Terminal movement transport history is incomplete");
        if (!ledger.stages.empty())
        {
            auto stages = nlohmann::json::array();
            for (const auto &stage : ledger.stages.entries())
                stages.push_back({{"identity", movement_json_detail::stageIdentityJson(stage.identity)},
                    {"transport", moeMovementTransportJson(stage.value)}});
            return {{"schema", 2}, {"scope", "terminal_model_lifetime"}, {"stages", std::move(stages)}};
        }
        using Wave = std::pair<std::uint64_t, std::uint64_t>;
        std::map<Wave, std::uint64_t> edge_counts;
        std::vector<Wave> wave_order;
        for (const auto &edge : ledger.edges)
        {
            if (edge.authority != MoEOptimizationAuthority::Device)
                continue;
            const Wave identity{edge.transaction, edge.candidate_epoch};
            if (wave_order.empty() || wave_order.back() != identity)
            {
                if (edge_counts.contains(identity))
                    throw std::invalid_argument("Terminal device movement edges interleave completed waves");
                wave_order.push_back(identity);
            }
            ++edge_counts[identity];
        }
        if (ledger.device_publications.size() != wave_order.size())
            throw std::invalid_argument("Terminal device movement lost a physical publication receipt");

        auto publications = nlohmann::json::array();
        for (std::size_t i = 0; i < ledger.device_publications.size(); ++i)
        {
            const auto &record = ledger.device_publications[i];
            const Wave identity{record.transaction, record.candidate_epoch};
            if (!record.valid() || identity != wave_order[i] ||
                record.command_count != edge_counts.at(identity))
                throw std::invalid_argument("Terminal device movement receipt disagrees with completed edges");
            publications.push_back({{"transaction", record.transaction},
                {"candidate_epoch", record.candidate_epoch},
                {"command_count", record.command_count},
                {"physical_payload_bytes", record.physical_payload_bytes}});
            if (record.controller)
            {
                const auto &r = *record.controller;
                const auto first = std::find_if(ledger.edges.begin(), ledger.edges.end(), [&](const auto &edge)
                    { return edge.transaction == record.transaction && edge.candidate_epoch == record.candidate_epoch; });
                validateMoEControllerCompletedMovement(record,
                    std::span<const MoEOptimizationMovementEdge>(&*first, record.command_count));
                const auto economy = std::find_if(ledger.economy.begin(), ledger.economy.end(), [&](const auto &proof)
                    { return proof.transaction == record.transaction && proof.candidate_epoch == record.candidate_epoch; });
                if (!ledger.economy.empty() && (economy == ledger.economy.end() || !economy->valid() ||
                    economy->command_count != record.command_count || economy->cycle_count != r.physical_cycles ||
                    !std::holds_alternative<MoEOptimizationTimeEconomy>(economy->proof) ||
                    std::get<MoEOptimizationTimeEconomy>(economy->proof) != r.economy))
                    throw std::invalid_argument("Controller transport receipt disagrees with the policy owner's economy");
                publications.back()["controller"] = {
                    {"base_epoch", r.base_epoch},
                    {"promotions", r.promotions},
                    {"demotions", r.demotions},
                    {"same_priority_moves", r.same_priority_moves},
                    {"cross_domain_moves", r.cross_domain_moves},
                    {"cross_rank_moves", r.cross_rank_moves},
                    {"cross_backend_moves", r.cross_backend_moves},
                    {"snapshot_observations", r.snapshot_observations},
                    {"priority_cost_before", r.priority_cost_before},
                    {"priority_cost_after", r.priority_cost_after},
                    {"same_priority_makespan_before", r.same_priority_makespan_before},
                    {"same_priority_makespan_after", r.same_priority_makespan_after},
                    {"accepted_cycles", r.accepted_cycles},
                    {"physical_cycles", r.physical_cycles},
                    {"rejected_cycles", r.rejected_cycles},
                    {"phase_tradeoff_candidates", r.phase_tradeoff_candidates},
                    {"improvement_floor_rejected_cycles", r.improvement_floor_rejected_cycles},
                    {"payoff_rejected_cycles", r.payoff_rejected_cycles},
                    {"residency_rejected_cycles", r.residency_rejected_cycles},
                    {"changed_layers", r.changed_layers},
                    {"layer_scan_start", r.layer_scan_start},
                    {"layer_scan_next", r.layer_scan_next},
                    {"edges_checked", r.edges_checked},
                    {"participant_coordinates_checked", r.participant_coordinates_checked},
                    {"tier_coordinates_checked", r.tier_coordinates_checked},
                    {"malformed_edges", r.malformed_edges},
                    {"participant_flow_violations", r.participant_flow_violations},
                    {"tier_flow_violations", r.tier_flow_violations},
                    {"projected_service_gain_ns", r.economy.projected_service_gain_ns},
                    {"projected_transfer_and_repack_ns", r.economy.projected_transfer_and_repack_ns},
                    {"projected_inference_interference_ns", r.economy.projected_inference_interference_ns},
                    {"projected_net_benefit_ns", r.economy.projected_net_benefit_ns},
                };
            }
        }
        return {{"schema", 1}, {"scope", "terminal_model_lifetime"},
            {"movement", moeMovementLedgerJson(ledger)},
            {"device_publications", std::move(publications)}};
    }

    /**
     * @brief Publish complete drained movement evidence next to its counter artifact.
     * @param perf_json_path Already-resolved, process-owned PerfStats JSON path.
     * @param rank Actual publishing MPI rank; an unranked process uses zero.
     * @param ledger Canonical immutable snapshot after native maintenance retirement.
     * @throws std::invalid_argument for an empty path, invalid rank or bad history.
     * @throws std::filesystem::filesystem_error or std::ios_base::failure on I/O failure.
     *
     * Validate before opening the temporary file. Atomic rename prevents readers
     * from accepting a partial document, and an unsuccessful export never replaces
     * earlier evidence. No cache/model payload is opened or inspected.
     */
    inline void writeMoEMovementTransportJson(const std::filesystem::path &perf_json_path,
        int rank, const MoEOptimizationMovementLedger &ledger)
    {
        if (perf_json_path.empty() || rank < 0)
            throw std::invalid_argument("Terminal movement export requires an owned path and MPI rank");
        auto document = moeMovementTransportJson(ledger);
        document["rank"] = rank;
        const std::filesystem::path destination = perf_json_path.string() + ".movement.json";
        const std::filesystem::path temporary = destination.string() + ".tmp";
        if (!destination.parent_path().empty())
            std::filesystem::create_directories(destination.parent_path());
        try
        {
            std::ofstream output;
            output.exceptions(std::ios::failbit | std::ios::badbit);
            output.open(temporary, std::ios::binary | std::ios::trunc);
            output << document.dump(2) << '\n';
            output.close();
            std::filesystem::rename(temporary, destination);
        }
        catch (...)
        {
            std::error_code ignored;
            std::filesystem::remove(temporary, ignored);
            throw;
        }
    }
}
