/**
 * @file HeterogeneousTicketParityEvidence.h
 * @brief Interpret validated ticket lifecycles without confusing compiler shards.
 *
 * Capture planning proves marker adjacency before retained-parent lowering.
 * Lowering may combine many CPU boundaries into one service program or split
 * GPU work into extra graph-only children. Neither transformation changes the
 * logical ticket lifecycle. Its versioned evidence remains distinct from the
 * separate materialization/replay counters that prove actual GPU execution.
 */
#pragma once

#include "utils/PerfStatsCollector.h"

#include <charconv>
#include <cmath>
#include <cstdint>
#include <optional>
#include <string_view>
#include <system_error>

namespace llaminar2::test::parity
{
    /** @brief Exact disposition of one observed ticket-lifecycle record. */
    enum class HeterogeneousTicketEvidenceDisposition : std::uint8_t
    {
        Unrelated, ///< Another metric; supplies no ticket-lifecycle proof.
        Authority, ///< Validated device cutpoints and their CPU services.
        Follower, ///< Validated captured cutpoints with no local CPU service.
        Malformed, ///< Missing validation provenance or inconsistent geometry.
    };

    /**
     * @brief Authenticate the planner's pre-lowering lifecycle evidence.
     * @param record One immutable production PerfStats observation.
     * @return Its validated participant role, unrelated, or malformed.
     *
     * Counts alone cannot prove ordering. The versioned contract is emitted
     * only by the production marker-adjacency validator after it succeeds.
     * Additional captured children are legal; a missing terminal, cutpoint,
     * or authority service is not. This observer never controls execution.
     */
    inline HeterogeneousTicketEvidenceDisposition
    classifyHeterogeneousTicketLifecycleEvidence(const PerfStatRecord &record)
    {
        using Disposition = HeterogeneousTicketEvidenceDisposition;
        if (record.domain != "forward_graph" ||
            record.name != "heterogeneous_ticket_transactions")
            return Disposition::Unrelated;

        const auto tag = [&](const char *name) -> std::string_view
        {
            const auto found = record.tags.find(name);
            return found == record.tags.end() ? std::string_view{} : found->second;
        };
        const auto count = [&](const char *name) -> std::optional<std::uint64_t>
        {
            const auto value = tag(name);
            if (value.empty())
                return std::nullopt;
            std::uint64_t result = 0u;
            const auto parsed = std::from_chars(
                value.data(), value.data() + value.size(), result);
            if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size())
                return std::nullopt;
            return result;
        };
        const auto captured = count("capturable_segments");
        const auto manual = count("manual_segments");
        const auto boundaries = count("unit_boundaries");
        const auto terminals = count("terminal_units");
        if (record.kind != PerfStatRecord::Kind::Counter ||
            record.phase != "capture" || record.count == 0u ||
            !std::isfinite(record.value) || record.value <= 0.0 ||
            tag("lifecycle_contract") != "typed_marker_adjacency_v1" ||
            tag("ticket_publication_authority") != "stage_owned_mapped_timeline" ||
            !captured || !manual || !boundaries || !terminals ||
            *boundaries == 0u || *terminals != 1u || *captured <= *boundaries)
            return Disposition::Malformed;

        if (tag("role") == "authority" && *manual == *boundaries)
            return Disposition::Authority;
        if (tag("role") == "follower" && *manual == 0u)
            return Disposition::Follower;
        return Disposition::Malformed;
    }
}
