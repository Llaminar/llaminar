/**
 * @file ParityRunnerEvidence.h
 * @brief Separate retained-runner construction proof from each cell's work.
 *
 * Compatible process-campaign cells may retain the exact production runner,
 * including its graphs and physical expert banks. Clearing their construction
 * records would demand fictional reconstruction; retaining whole domains would
 * let an earlier cell falsely certify replay, prefix restore, or movement.
 * Keep only the named immutable contracts in the existing collector. No second
 * evidence cache, runtime state mirror, or allocation ledger is introduced.
 */
#pragma once

#include "utils/PerfStatsCollector.h"
#include <stdexcept>
#include <array>
#include <numeric>
#include <span>

namespace llaminar2::test::parity
{
    /** Ownership transition authenticated by the fixture's complete runner key. */
    enum class ParityRunnerEvidenceLifetime
    {
        FreshRunner,
        RetainedRunner,
    };

    /**
     * @brief Project the four required terminal movement facts for one parity cell.
     * @param records Current-cell collector snapshot, after original native work completed.
     * @return Accepted placement, copied payload, physical bytes and applied placement totals.
     *
     * Native controller families originate only in the authenticated completed
     * physical-publication producer. Proposals and source arrivals do not
     * certify completion. CPU/mapped families retain their existing meanings.
     * Runner reset erases these request counters; setup evidence cannot supply
     * movement credit to a later cell. These are observations, never capacity.
     */
    inline std::array<float, 4> parityMoEMovementFacts(
        std::span<const PerfStatRecord> records)
    {
        const auto counter = [&](const char *domain, const char *name)
        {
            return std::accumulate(
                records.begin(), records.end(), 0.0,
                [&](double total, const PerfStatRecord &record)
                {
                    return total +
                           (record.kind == PerfStatRecord::Kind::Counter &&
                                    record.domain == domain &&
                                    record.name == name
                                ? record.value
                                : 0.0);
                });
        };

        /*
         * Host-authoritative multi-tier movement and device-authoritative
         * GPU movement publish different counters, but both must
         * prove the same four facts. Keep the mapping here at the common
         * parity boundary rather than teaching each model fixture backend
         * implementation details.
         */
        return {
            static_cast<float>(
                counter("moe_overlay_residency", "committed_expert_migrations") +
                counter("moe_overlay_controller", "dynamic_movement_transactions") +
                counter("moe_rebalance", "device_rebalance_dynamic_ownership_swap_accepts") +
                counter("moe_rebalance", "device_rebalance_selected_replicas")),
            static_cast<float>(
                counter("moe_overlay_residency", "cpu_native_copy_bytes") +
                counter("moe_overlay_residency", "remote_projection_payload_bytes_completed") +
                counter("moe_overlay_residency", "placement_transfer_operations_completed") +
                counter("moe_overlay_residency", "device_physical_destinations_staged") +
                counter("moe_overlay_controller", "dynamic_movement_commands") +
                counter("moe_rebalance", "cpu_llep_weight_transfers") +
                counter("moe_rebalance", "weight_transfer_outgoing_entries") +
                counter("moe_rebalance", "weight_transfer_incoming_entries") +
                counter("moe_rebalance", "mask_apply_received_entries") +
                counter("moe_rebalance", "device_rebalance_copy_copied_arrivals") +
                counter("moe_rebalance", "device_rebalance_transfer_current_copied_arrivals") +
                counter("moe_rebalance", "device_rebalance_wave_copied_arrivals_total") +
                counter("moe_rebalance", "device_rebalance_request_copied_payload_lower_bound")),
            static_cast<float>(
                counter("moe_overlay_residency", "cpu_native_copy_bytes") +
                counter("moe_overlay_residency", "remote_projection_payload_bytes_completed") +
                counter("moe_overlay_residency", "placement_transfer_payload_bytes_completed") +
                counter("moe_overlay_controller", "dynamic_physical_bytes") +
                counter("moe_rebalance", "cpu_llep_weight_transfer_incoming_bytes") +
                counter("moe_rebalance", "cpu_llep_weight_transfer_outgoing_bytes") +
                counter("moe_rebalance", "weight_transfer_incoming_bytes") +
                counter("moe_rebalance", "weight_transfer_outgoing_bytes") +
                counter("moe_rebalance", "mask_apply_received_bytes") +
                counter("moe_rebalance", "device_rebalance_transfer_useful_payload_bytes") +
                counter("moe_rebalance", "device_rebalance_request_useful_payload_bytes_lower_bound")),
            static_cast<float>(
                counter("moe_overlay_residency", "expert_migration_edges") +
                counter("moe_overlay_residency", "device_physical_epoch_published") +
                counter("moe_overlay_controller", "dynamic_migration_edges") +
                counter("moe_rebalance", "cpu_llep_weight_transfers") +
                counter("moe_rebalance", "ownership_change_entries") +
                counter("moe_rebalance", "replica_arrivals") +
                counter("moe_rebalance", "mask_apply_received_entries") +
                counter("moe_rebalance", "device_rebalance_apply_applied_arrivals") +
                counter("moe_rebalance", "device_rebalance_transfer_current_applied_arrivals") +
                counter("moe_rebalance", "device_rebalance_wave_applied_arrivals_total") +
                counter("moe_rebalance", "device_rebalance_request_applied_payload_lower_bound")),
        };
    }

    /**
     * @brief Start one cell's measurement window without inventing setup work.
     * @param lifetime Retained only while the exact, previously green runner is
     *                 still owned and its full physical/graph identity matches.
     *
     * The Static policy check is a once-per-authority certification, not a
     * request's no-movement proof. Every cell still requires a fresh execution
     * trace, zero movement counters, and an empty authoritative movement ledger.
     * All request counters/timers/ordered sequences are erased. In particular,
     * MTP read leases, ticket selections and graph replay may never survive.
     */
    inline void resetParityRunnerEvidence(ParityRunnerEvidenceLifetime lifetime)
    {
        switch (lifetime)
        {
        case ParityRunnerEvidenceLifetime::FreshRunner:
            PerfStatsCollector::reset();
            return;
        case ParityRunnerEvidenceLifetime::RetainedRunner:
            // Match exact families, never a domain shared with runtime work.
            // Keep original device, phase, tags and counts: do not republish
            // old allocations/captures as if the next cell constructed them.
            PerfStatsCollector::resetPreserving({}, {
                {"mtp", "sidecar_terminal_hidden_read_only_contracts"},
                {"memory", "moe_serial_local_expert_buffer_arena_allocations"},
                {"memory", "moe_serial_local_expert_buffer_arena_bytes"},
                {"moe_placement", "routed_expert_weight_selection"},
                {"moe_overlay_controller", "static_no_movement_transactions"},
                {"moe_overlay_residency", "static_no_movement_checks"},
                {"moe_overlay_controller", "follower_runtime_tables_materialized"},
                {"moe_overlay_participant_graph", "materialized_mapped_follower_families"},
                {"moe_overlay_participant_graph", "materialized_rank_local_canonical_ticket_consumers"},
                // These launches record nodes during capture, not inference.
                {"moe_overlay_participant_graph", "rank_local_canonical_ticket_consumer_launches"},
                {"forward_graph", "heterogeneous_ticket_transactions"},
                {"forward_graph", "segmented_plan_segments"},
                {"forward_graph", "segmented_graph_capture_segments"},
                {"forward_graph", "retained_parent_transaction_zero_launches"},
            });
            return;
        }
        throw std::logic_error("Unclassified parity runner evidence lifetime");
    }
}
