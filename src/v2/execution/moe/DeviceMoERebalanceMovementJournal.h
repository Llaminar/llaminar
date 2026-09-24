/**
 * @file DeviceMoERebalanceMovementJournal.h
 * @brief Bounded device-authored history of durable native ownership swaps.
 *
 * One finalizer owns append; terminal observation happens after its completion
 * event. There is no host polling, allocation, ring overwrite, or placement
 * decision here. A wave reserves a complete range, cooperatively writes its
 * immutable edges, and commits only after the placement selector publishes.
 * Exhaustion records an explicit missing-evidence count without obstructing
 * inference. A partial history can never certify completed movement.
 */
#pragma once

#include "DeviceMoERebalanceLoadSpreadProof.h"
#include <cstdint>
#include <type_traits>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define LLAMINAR_MOE_JOURNAL_HD __host__ __device__
#else
#define LLAMINAR_MOE_JOURNAL_HD
#endif

namespace llaminar2
{
    /**
     * @brief Authenticate one native ownership-swap pair before publication.
     * @param outgoing First immutable command in the planner's paired order.
     * @param returning Its reciprocal capacity-preserving command.
     * @param layers Declared runtime layer extent.
     * @param experts Declared experts per layer.
     * @param participants Domain-local participant extent, independent of rank.
     * @param ownership_op Shared caller's OwnershipTransfer wire value.
     * @return True only for two distinct experts exchanged within one layer.
     *
     * GPU ABI views and CPU records share this predicate. It checks command
     * identity, not a mutable routing histogram or inferred transport direction.
     */
    template <class Plan>
    [[nodiscard]] LLAMINAR_MOE_JOURNAL_HD inline bool nativeMovementPairValid(
        const Plan &outgoing, const Plan &returning, std::uint32_t layers,
        std::uint32_t experts, std::uint32_t participants,
        std::uint32_t ownership_op) noexcept
    {
        return outgoing.op == ownership_op && returning.op == ownership_op &&
            outgoing.layer < layers && outgoing.layer == returning.layer &&
            outgoing.expert < experts && returning.expert < experts &&
            outgoing.expert != returning.expert &&
            outgoing.source_participant < participants &&
            outgoing.destination_participant < participants &&
            outgoing.source_participant != outgoing.destination_participant &&
            outgoing.source_participant == returning.destination_participant &&
            outgoing.destination_participant == returning.source_participant;
    }

    /** @brief Immutable command identity retained before its reusable slot retires. */
    struct DeviceMoERebalanceMovementEdge
    {
        std::uint32_t layer = 0;
        std::uint32_t expert = 0;
        std::uint32_t source_participant = 0;
        std::uint32_t destination_participant = 0;
        std::uint64_t activation_count = 0; ///< Sealed demand, not a later histogram.
        std::uint64_t estimated_weight_bytes = 0; ///< Descriptor extent, not wire telemetry.
    };

    /** @brief One complete publication plus the exact decision that admitted it. */
    struct DeviceMoERebalanceMovementWave
    {
        std::uint64_t candidate_epoch = 0; ///< Durable model-lifetime selector generation.
        std::uint32_t command_epoch = 0; ///< Request-local reusable-command identity.
        std::uint32_t first_edge = 0;
        std::uint32_t edge_count = 0;
        std::uint32_t reserved = 0;
        std::uint64_t physical_payload_bytes = 0; ///< Actual complete-copy bytes from all participants.
        DeviceMoERebalanceLoadSpreadProof proof;
    };

    /**
     * @brief Single-writer publication prefix; capacities belong to the arena view.
     *
     * Request reset zeros this small header, never the entire payload storage.
     * Previously archived records are immutable diagnostic values, not a host
     * mirror used for planning. Model-lifetime epochs cannot be reused after
     * reset, so terminal archives can authenticate their append boundary.
     */
    struct DeviceMoERebalanceMovementJournalState
    {
        std::uint32_t committed_waves = 0;
        std::uint32_t committed_edges = 0;
        std::uint64_t last_candidate_epoch = 0;
        std::uint64_t discarded_waves = 0;
        std::uint64_t discarded_edges = 0;
    };

    /** @brief Exact arena-owned storage passed to the single device publisher. */
    struct DeviceMoERebalanceMovementJournalView
    {
        DeviceMoERebalanceMovementJournalState *state = nullptr;
        DeviceMoERebalanceMovementWave *waves = nullptr;
        DeviceMoERebalanceMovementEdge *edges = nullptr;
        std::uint32_t wave_capacity = 0;
        std::uint32_t edge_capacity = 0;

        /** @return Whether the complete initialized publication prefix is in bounds. */
        [[nodiscard]] LLAMINAR_MOE_JOURNAL_HD bool valid() const noexcept
        {
            return state && waves && edges && wave_capacity && edge_capacity &&
                state->committed_waves <= wave_capacity &&
                state->committed_edges <= edge_capacity &&
                ((state->committed_waves == 0) == (state->committed_edges == 0)) &&
                state->committed_edges % 2u == 0 &&
                state->committed_waves <= state->committed_edges / 2u &&
                ((state->discarded_waves == 0) == (state->discarded_edges == 0)) &&
                ((state->committed_waves == 0 && state->discarded_waves == 0) ==
                 (state->last_candidate_epoch == 0));
        }
    };

    /** Result of reserving a complete immutable wave, never a partial edge prefix. */
    enum class DeviceMoEMovementJournalDisposition : std::uint32_t
    {
        Invalid, ///< Broken owner identity/binding: publication must fail fatally.
        Record, ///< Complete prebound range is available for cooperative writes.
        Exhausted, ///< No space; serving may publish, but evidence is incomplete.
    };

    /**
     * @brief Pure append reservation bound to one immutable owner publication.
     *
     * Only the finalizer may consume this value; it is not a second transaction
     * state machine. Before selector publication it authorizes scratch writes,
     * not a visible committed record. A failed finalizer simply abandons it.
     */
    struct DeviceMoEMovementJournalAppend
    {
        DeviceMoEMovementJournalDisposition disposition = DeviceMoEMovementJournalDisposition::Invalid;
        DeviceMoERebalanceMovementJournalView binding; ///< Exact prebound pointer/capacity identity.
        std::uint32_t wave_index = 0;
        std::uint32_t first_edge = 0;
        std::uint32_t edge_count = 0;
        std::uint64_t previous_epoch = 0;
        std::uint64_t candidate_epoch = 0;
        std::uint32_t command_epoch = 0;
        DeviceMoERebalanceLoadSpreadProof proof; ///< Frozen with the reserved identity.
        std::uint64_t physical_payload_bytes = 0; ///< Sealed with the authenticated arrival count.
    };

    /**
     * @brief Reserve a whole wave without mutating the visible journal prefix.
     * @param view Exact persistent arena binding; no optional logging enablement.
     * @param candidate_epoch New durable selector generation, strictly monotonic.
     * @param command_epoch Actual prepared command generation, not a host counter.
     * @param edge_count Complete command count, two entries per accepted swap.
     * @param proof Device-sealed native policy inputs for those commands.
     * @param physical_payload_bytes Actual destination bytes after arrival authentication.
     * @return Validated range or explicit exhaustion; malformed inputs are Invalid.
     */
    [[nodiscard]] LLAMINAR_MOE_JOURNAL_HD inline DeviceMoEMovementJournalAppend
    prepareDeviceMoEMovementJournalAppend(
        const DeviceMoERebalanceMovementJournalView &view,
        std::uint64_t candidate_epoch, std::uint32_t command_epoch,
        std::uint32_t edge_count, const DeviceMoERebalanceLoadSpreadProof &proof,
        std::uint64_t physical_payload_bytes) noexcept
    {
        DeviceMoEMovementJournalAppend result;
        if (!view.valid() || command_epoch == 0 || !proof.valid() || physical_payload_bytes == 0 ||
            candidate_epoch <= view.state->last_candidate_epoch ||
            proof.ownership_swap_accepts > UINT32_MAX / 2u ||
            edge_count != proof.ownership_swap_accepts * 2u)
            return result;
        result.wave_index = view.state->committed_waves;
        result.binding = view;
        result.first_edge = view.state->committed_edges;
        result.edge_count = edge_count;
        result.previous_epoch = view.state->last_candidate_epoch;
        result.candidate_epoch = candidate_epoch;
        result.command_epoch = command_epoch;
        result.proof = proof;
        result.physical_payload_bytes = physical_payload_bytes;
        // Subtraction after bound validation avoids integer overflow. Once a
        // wave is discarded the prefix remains closed: do not resume recording
        // behind a missing generation, even if a later wave would be smaller.
        result.disposition = view.state->discarded_waves == 0 &&
            result.wave_index < view.wave_capacity &&
            edge_count <= view.edge_capacity - result.first_edge
                ? DeviceMoEMovementJournalDisposition::Record
                : DeviceMoEMovementJournalDisposition::Exhausted;
        return result;
    }

    /**
     * @brief Commit a reserved journal prefix after successful selector publication.
     * @param view The same single-writer arena binding used by preparation.
     * @param append Frozen range; stale/double commits are rejected without writes.
     * @return False for an invalid/stale reservation; true includes explicit exhaustion.
     *
     * The device caller must join edge writers and fence before this lane-zero
     * operation. It must fence again before advertising command-slot retirement.
     * No host or concurrently polling reader may consume this non-atomic header;
     * terminal readers require the exact finalizer completion event.
     */
    [[nodiscard]] LLAMINAR_MOE_JOURNAL_HD inline bool commitDeviceMoEMovementJournalAppend(
        const DeviceMoERebalanceMovementJournalView &view,
        const DeviceMoEMovementJournalAppend &append) noexcept
    {
        // A coincidentally equal prefix in another arena is not this reserved
        // transaction. Reject rebinding before reading any proposed owner.
        if (view.state != append.binding.state || view.waves != append.binding.waves ||
            view.edges != append.binding.edges || view.wave_capacity != append.binding.wave_capacity ||
            view.edge_capacity != append.binding.edge_capacity)
            return false;
        const auto current = prepareDeviceMoEMovementJournalAppend(
            view, append.candidate_epoch, append.command_epoch, append.edge_count, append.proof,
            append.physical_payload_bytes);
        if (append.disposition == DeviceMoEMovementJournalDisposition::Invalid ||
            current.disposition != append.disposition || current.wave_index != append.wave_index ||
            current.first_edge != append.first_edge || current.previous_epoch != append.previous_epoch)
            return false;
        if (append.disposition == DeviceMoEMovementJournalDisposition::Exhausted)
        {
            // Saturation retains the only required meaning: evidence is
            // incomplete. It cannot wrap to zero and masquerade as complete.
            if (view.state->discarded_waves != UINT64_MAX)
                ++view.state->discarded_waves;
            view.state->discarded_edges = append.edge_count > UINT64_MAX - view.state->discarded_edges
                ? UINT64_MAX : view.state->discarded_edges + append.edge_count;
        }
        else
        {
            auto &wave = view.waves[append.wave_index];
            wave.candidate_epoch = append.candidate_epoch;
            wave.command_epoch = append.command_epoch;
            wave.first_edge = append.first_edge;
            wave.edge_count = append.edge_count;
            wave.reserved = 0;
            wave.physical_payload_bytes = append.physical_payload_bytes;
            wave.proof = append.proof;
            view.state->committed_edges += append.edge_count;
            ++view.state->committed_waves;
        }
        view.state->last_candidate_epoch = append.candidate_epoch;
        return true;
    }

    static_assert(std::is_trivially_copyable_v<DeviceMoERebalanceMovementWave>);
    static_assert(sizeof(DeviceMoERebalanceMovementEdge) == 32);
    static_assert(sizeof(DeviceMoERebalanceMovementWave) == 120);
    static_assert(sizeof(DeviceMoERebalanceMovementJournalState) == 32);
}

#undef LLAMINAR_MOE_JOURNAL_HD
