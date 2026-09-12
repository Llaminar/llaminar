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

namespace llaminar2::test::parity
{
    /** Ownership transition authenticated by the fixture's complete runner key. */
    enum class ParityRunnerEvidenceLifetime
    {
        FreshRunner,
        RetainedRunner,
    };

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
