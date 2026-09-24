/**
 * @file NativeMoEMovementArchive.cpp
 * @brief Validate terminal native movement receipts without shadowing placement.
 *
 * Request generations authenticate reset; model epochs authenticate publication.
 * The archive preserves the native load units and actual copied payload bytes.
 * It performs no device I/O and has no dependency on the profiling collector.
 */
#include "NativeMoEMovementArchive.h"
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace llaminar2
{
    namespace
    {
        /** @brief Reject corrupt evidence with one precise owner diagnostic. */
        void require(bool condition, const char *message)
        {
            if (!condition)
                throw std::invalid_argument(std::string("Native MoE movement archive: ") + message);
        }

        /** @return Saturated missing-evidence count; it can never wrap to zero. */
        std::uint64_t saturatingAdd(std::uint64_t a, std::uint64_t b) noexcept
        {
            return b > UINT64_MAX - a ? UINT64_MAX : a + b;
        }

        /** @return Equality of meaningful wave fields, excluding struct padding. */
        bool sameWave(const DeviceMoERebalanceMovementWave &a, const DeviceMoERebalanceMovementWave &b)
        {
            return a.candidate_epoch == b.candidate_epoch && a.command_epoch == b.command_epoch &&
                a.first_edge == b.first_edge && a.edge_count == b.edge_count && a.reserved == b.reserved &&
                a.physical_payload_bytes == b.physical_payload_bytes && a.proof == b.proof;
        }

        /** @return Exact semantic command identity, not a byte/padding checksum. */
        bool sameEdge(const DeviceMoERebalanceMovementEdge &a, const DeviceMoERebalanceMovementEdge &b)
        {
            return a.layer == b.layer && a.expert == b.expert &&
                a.source_participant == b.source_participant && a.destination_participant == b.destination_participant &&
                a.activation_count == b.activation_count && a.estimated_weight_bytes == b.estimated_weight_bytes;
        }
    }

    NativeMoEMovementArchive::NativeMoEMovementArchive(NativeMoEMovementArchiveConfig config)
        : config_(std::move(config))
    {
        require(config_.workspace_generation && config_.layers && config_.experts &&
            config_.layers <= INT32_MAX && config_.experts <= INT32_MAX &&
            config_.wave_capacity && config_.edge_capacity >= 2 &&
            config_.wave_capacity <= config_.edge_capacity / 2 &&
            config_.participants.size() >= 2 && config_.participants.size() <= INT32_MAX,
            "incomplete arena/domain geometry");
        for (std::size_t i = 0; i < config_.participants.size(); ++i)
        {
            const auto &p = config_.participants[i];
            require(p.device.is_gpu() && p.world_rank >= 0 &&
                p.device.type == config_.participants.front().device.type &&
                p.priority == config_.participants.front().priority, "domain is not one homogeneous GPU tier");
            for (std::size_t j = 0; j < i; ++j)
                require(p.world_rank != config_.participants[j].world_rank ||
                    p.device != config_.participants[j].device, "duplicate physical participant");
        }
    }

    void NativeMoEMovementArchive::validate(std::uint64_t session_epoch, std::uint64_t workspace_generation,
        const DeviceMoERebalanceMovementJournalState &state,
        std::span<const DeviceMoERebalanceMovementWave> waves,
        std::span<const DeviceMoERebalanceMovementEdge> edges) const
    {
        require(session_epoch && session_epoch >= session_epoch_ &&
            workspace_generation == config_.workspace_generation, "stale request or rebound arena");
        require(state.committed_waves == waves.size() && state.committed_edges == edges.size() &&
            waves.size() <= config_.wave_capacity && edges.size() <= config_.edge_capacity &&
            (waves.empty() == edges.empty()) && edges.size() % 2 == 0 &&
            ((state.discarded_waves == 0) == (state.discarded_edges == 0)) &&
            ((waves.empty() && state.discarded_waves == 0) == (state.last_candidate_epoch == 0)),
            "invalid populated journal bounds");
        if (session_epoch == session_epoch_)
        {
            require(waves.size() >= previous_waves_.size() && edges.size() >= previous_edges_.size() &&
                state.last_candidate_epoch >= previous_state_.last_candidate_epoch &&
                state.discarded_waves >= previous_state_.discarded_waves &&
                state.discarded_edges >= previous_state_.discarded_edges &&
                std::equal(previous_waves_.begin(), previous_waves_.end(), waves.begin(), sameWave) &&
                std::equal(previous_edges_.begin(), previous_edges_.end(), edges.begin(), sameEdge),
                "committed request history regressed or changed");
            require(!previous_state_.discarded_waves || waves.size() == previous_waves_.size(),
                "device resumed recording after journal exhaustion");
            const bool changed_counts = waves.size() != previous_waves_.size() ||
                state.discarded_waves != previous_state_.discarded_waves ||
                state.discarded_edges != previous_state_.discarded_edges;
            // Once both loss counters saturate they can no longer witness each
            // later missing wave; the epoch still advances, and loss is sticky.
            const bool saturated_loss = previous_state_.discarded_waves == UINT64_MAX &&
                previous_state_.discarded_edges == UINT64_MAX;
            require(saturated_loss || changed_counts ==
                (state.last_candidate_epoch > previous_state_.last_candidate_epoch),
                "publication counts and terminal epoch disagree");
        }
        std::size_t next_edge = 0;
        std::uint64_t prior_epoch = session_epoch == session_epoch_ ? 0 : last_model_epoch_;
        for (const auto &wave : waves)
        {
            if (session_epoch == session_epoch_ && next_edge >= previous_edges_.size())
                require(wave.candidate_epoch > last_model_epoch_, "model epoch reused after request reset");
            require(wave.candidate_epoch > prior_epoch && wave.command_epoch && wave.reserved == 0 &&
                wave.first_edge == next_edge && wave.edge_count > 0 && wave.edge_count % 2 == 0 &&
                wave.edge_count <= edges.size() - next_edge && wave.physical_payload_bytes > 0 &&
                wave.proof.valid() && wave.proof.ownership_swap_accepts == wave.edge_count / 2,
                "invalid committed wave identity, payload or policy");
            for (std::size_t i = next_edge; i < next_edge + wave.edge_count; i += 2)
            {
                const auto &a = edges[i];
                const auto &b = edges[i + 1];
                require(a.layer < config_.layers && a.layer == b.layer &&
                    a.expert < config_.experts && b.expert < config_.experts && a.expert != b.expert &&
                    a.source_participant < config_.participants.size() &&
                    a.destination_participant < config_.participants.size() &&
                    a.source_participant != a.destination_participant &&
                    a.source_participant == b.destination_participant &&
                    a.destination_participant == b.source_participant &&
                    a.estimated_weight_bytes && b.estimated_weight_bytes,
                    "invalid reciprocal ownership cycle");
                // One expert cannot change owners twice in an atomic wave.
                // Candidate lists are bounded; this audit is terminal-only.
                for (std::size_t j = next_edge; j < i; ++j)
                    require(edges[j].layer != a.layer ||
                        (edges[j].expert != a.expert && edges[j].expert != b.expert),
                        "duplicate expert in committed wave");
            }
            next_edge += wave.edge_count;
            prior_epoch = wave.candidate_epoch;
        }
        require(next_edge == edges.size() &&
            (state.discarded_waves ? state.last_candidate_epoch > prior_epoch :
                state.last_candidate_epoch == (waves.empty() ? 0 : prior_epoch)),
            "journal terminal epoch does not authenticate its prefix");
    }

    std::size_t NativeMoEMovementArchive::observe(std::uint64_t session_epoch, std::uint64_t workspace_generation,
        const DeviceMoERebalanceMovementJournalState &state,
        std::span<const DeviceMoERebalanceMovementWave> waves,
        std::span<const DeviceMoERebalanceMovementEdge> edges)
    {
        validate(session_epoch, workspace_generation, state, waves, edges);
        const bool same_request = session_epoch == session_epoch_;
        const std::size_t first_wave = same_request ? previous_waves_.size() : 0;
        const std::size_t first_public_edge = ledger_.edges.size();
        // Prepare complete values before mutating the accepted archive. An
        // allocation/validation failure cannot publish a partial HTTP history.
        auto next = ledger_;
        auto totals = totals_;
        std::vector<DeviceMoERebalanceMovementWave> saved_waves(waves.begin(), waves.end());
        std::vector<DeviceMoERebalanceMovementEdge> saved_edges(edges.begin(), edges.end());
        for (std::size_t index = first_wave; index < waves.size(); ++index)
        {
            const auto &wave = waves[index];
            require(totals.transactions < UINT64_MAX && wave.edge_count <= UINT64_MAX - totals.commands &&
                wave.physical_payload_bytes <= UINT64_MAX - totals.physical_bytes,
                "completed movement totals overflow");
            ++totals.transactions;
            totals.commands += wave.edge_count;
            totals.physical_bytes += wave.physical_payload_bytes;
            totals.same_priority_moves += wave.edge_count;
            if (!next.complete() || next.economy.size() == config_.wave_capacity ||
                wave.edge_count > config_.edge_capacity - next.edges.size())
            {
                next.discarded_edges = saturatingAdd(next.discarded_edges, wave.edge_count);
                next.discarded_economy_records = saturatingAdd(next.discarded_economy_records, 1);
                continue;
            }
            next.economy.push_back({.authority = MoEOptimizationAuthority::Device,
                .transaction = wave.candidate_epoch, .candidate_epoch = wave.candidate_epoch,
                .command_count = wave.edge_count, .cycle_count = wave.edge_count / 2, .proof = wave.proof});
            for (std::uint32_t i = 0; i < wave.edge_count; ++i)
            {
                const auto &edge = edges[wave.first_edge + i];
                const auto &source = config_.participants[edge.source_participant];
                const auto &destination = config_.participants[edge.destination_participant];
                next.edges.push_back({.authority = MoEOptimizationAuthority::Device,
                    .transaction = wave.candidate_epoch, .candidate_epoch = wave.candidate_epoch,
                    .layer = static_cast<int>(edge.layer), .expert = static_cast<int>(edge.expert),
                    .cycle_index = i / 2, .cycle_size = 2,
                    .direction = MoEOptimizationMovementDirection::SamePriority,
                    .axis = MoEOptimizationMovementAxis::ParticipantPlacement,
                    .source_participant = static_cast<int>(edge.source_participant),
                    .destination_participant = static_cast<int>(edge.destination_participant),
                    .source_priority = source.priority, .destination_priority = destination.priority,
                    .source_device = source.device, .destination_device = destination.device,
                    .source_world_rank = source.world_rank, .destination_world_rank = destination.world_rank,
                    .source_world_rank_known = true, .destination_world_rank_known = true,
                    .estimated_weight_bytes = edge.estimated_weight_bytes, .activation_count = edge.activation_count});
            }
        }
        next.discarded_edges = saturatingAdd(next.discarded_edges,
            state.discarded_edges - (same_request ? previous_state_.discarded_edges : 0));
        next.discarded_economy_records = saturatingAdd(next.discarded_economy_records,
            state.discarded_waves - (same_request ? previous_state_.discarded_waves : 0));
        ledger_ = std::move(next);
        totals_ = totals;
        previous_waves_ = std::move(saved_waves);
        previous_edges_ = std::move(saved_edges);
        previous_state_ = state;
        session_epoch_ = session_epoch;
        last_model_epoch_ = std::max(last_model_epoch_, state.last_candidate_epoch);
        return first_public_edge;
    }
}
