/**
 * @file LeastLoadedExpertAssignment.h
 * @brief Backend-neutral least-loaded-resident routed-row assignment policy.
 *
 * This header owns the small deterministic LLA/LLAS policy from
 * least-loaded-resident assignment in a form that CPU, CUDA, and ROCm code can
 * share. It intentionally operates on caller-owned fixed buffers: graph-captured
 * device code must not allocate, and host tests should exercise the same
 * capacity/overflow behavior as the eventual kernels.
 */

#pragma once

#include <cstdint>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define LLAMINAR_LLEP_HD __host__ __device__ __forceinline__
#else
#define LLAMINAR_LLEP_HD inline
#endif

namespace llaminar2::least_loaded_ep
{
    inline constexpr uint32_t kInvalidParticipant = 0xffffffffu;

    struct LeastLoadedExpertAssignmentConfig
    {
        uint32_t expert_count = 0;
        uint32_t participant_count = 0;
        uint32_t min_chunk_tokens = 0;
        uint32_t alpha_numerator = 1;
        uint32_t alpha_denominator = 1;
        uint32_t lambda_numerator = 13;
        uint32_t lambda_denominator = 10;
        uint64_t min_spread_improvement = 0;
        uint32_t min_spread_improvement_divisor = 0;
        /**
         * Additional spread improvement required for each serialized payload
         * slot on the transfer wave's busiest participant lane.
         *
         * Reciprocal transfers use opposite full-duplex lanes and therefore
         * contribute one critical-path slot, not two. Multiple payloads sent
         * or received by the same participant remain additive.
         */
        uint64_t min_spread_improvement_per_critical_path_slot = 0;
        /**
         * Minimum useful foreign rows required for each serialized payload
         * slot on the transfer wave's busiest participant lane.
         */
        uint64_t min_foreign_rows_per_critical_path_slot = 0;
        /// Optional hard cap on missing expert-weight arrivals the assignment
        /// may request. Zero means use the caller-provided transfer buffer
        /// capacity. When the cap is exhausted, remaining rows stay on a
        /// resident participant instead of publishing an impossible plan.
        uint32_t max_weight_transfers = 0;
        /**
         * @brief Maximum distinct non-owner experts assigned to one participant.
         *
         * `max_weight_transfers` bounds new payload copies in this planning
         * wave. This field bounds the complete non-owner expert working set
         * consumed by the resulting assignment, including already-resident
         * replicas. Retaining one cached replica and requesting one new arrival
         * still requires two simultaneously valid expert payloads.
         *
         * Zero leaves the working set unbounded for resident-only policies.
         * Graph-native GPU callers set this to the physical transfer-directory
         * capacity so the planner cannot publish an assignment that the
         * materializer would have to truncate or repair.
         */
        uint32_t max_non_owner_experts_per_participant = 0;
        bool enable_balanced_skip = true;
    };

    struct LeastLoadedExpertAssignmentWorkspace
    {
        uint32_t *sorted_experts = nullptr;
        uint64_t *pending_load = nullptr;
        uint64_t *assigned_load = nullptr;
    };

    struct LeastLoadedExpertAssignmentSpan
    {
        uint32_t expert = 0;
        uint32_t owner_participant = 0;
        uint32_t destination_participant = 0;
        uint64_t route_row_begin = 0;
        uint64_t route_row_end = 0;
        uint8_t needs_foreign_weight = 0;
        uint8_t forced = 0;
        uint16_t reserved = 0;
    };

    /**
     * @brief Logical request to make one expert resident on a participant.
     *
     * The record deliberately contains no physical destination-slot identity.
     * GPU projection leases that graph-lifetime resource from the destination
     * device's live directory. Sixteen-byte alignment preserves the single
     * vector load/store used while staging assignment results in shared memory.
     */
    struct alignas(16) LeastLoadedExpertWeightTransfer
    {
        uint32_t expert = 0;
        uint32_t source_participant = 0;
        uint32_t destination_participant = 0;
        /**
         * Explicitly owned vector-lane tail.
         *
         * The transfer is copied and compared as one 16-byte device ABI
         * record. Naming the final word keeps aggregate assignment from
         * publishing indeterminate C++ padding bytes, so independently built
         * but logically identical plans remain byte deterministic.
         */
        uint32_t reserved = 0;
    };

    struct LeastLoadedExpertAssignmentStatus
    {
        uint64_t total_load = 0;
        uint64_t max_expert_load = 0;
        uint64_t capacity_per_participant = 0;
        uint64_t standard_load_min = 0;
        uint64_t standard_load_max = 0;
        uint64_t standard_load_spread = 0;
        uint64_t assigned_load_min = 0;
        uint64_t assigned_load_max = 0;
        uint64_t assigned_load_spread = 0;
        uint64_t assigned_load_spread_improvement = 0;
        uint64_t required_spread_improvement = 0;
        uint64_t required_foreign_rows = 0;
        uint64_t native_rows = 0;
        uint64_t spilled_rows = 0;
        uint32_t span_count = 0;
        uint32_t weight_transfer_count = 0;
        /**
         * Serialized payload width of the busiest source or destination lane.
         *
         * This is the transfer-cost multiplier. It can be smaller than
         * `weight_transfer_count` when independent or reciprocal edges run in
         * parallel.
         */
        uint32_t critical_path_transfer_slots = 0;
        uint32_t min_chunk_skips = 0;
        uint32_t forced_spills = 0;
        uint32_t overflow = 0;
        uint32_t invalid_config = 0;
        uint32_t skipped_balanced = 0;
        uint32_t skipped_insufficient_spread_improvement = 0;
        uint32_t skipped_insufficient_foreign_rows = 0;
        uint32_t standard_ep_selected = 0;
    };

    /**
     * @brief Report whether a published assignment redistributes any routed row.
     *
     * LLEP can improve participant balance without copying an expert payload
     * when the selected destination already owns a resident replica. In that
     * case @ref LeastLoadedExpertAssignmentStatus::spilled_rows and
     * @ref LeastLoadedExpertAssignmentStatus::weight_transfer_count both remain
     * zero even though production execution deliberately sends routed rows away
     * from the expert's authoritative owner. This predicate names that semantic
     * distinction directly so host tests and both GPU consumers use the same
     * definition.
     *
     * Empty spans are ignored because they do not assign executable work. A
     * null span array is valid only for an empty publication and therefore
     * reports no redistribution.
     *
     * @param spans Canonical assignment-span publication.
     * @param span_count Number of readable entries in @p spans.
     * @return `true` when at least one non-empty span targets a non-owner
     *         participant.
     */
    LLAMINAR_LLEP_HD bool containsNonOwnerAssignmentRows(
        const LeastLoadedExpertAssignmentSpan *spans,
        uint32_t span_count) noexcept
    {
        if (spans == nullptr)
            return false;

        for (uint32_t span_index = 0; span_index < span_count; ++span_index)
        {
            const auto &span = spans[span_index];
            if (span.route_row_begin < span.route_row_end &&
                span.destination_participant != span.owner_participant)
            {
                return true;
            }
        }
        return false;
    }

    LLAMINAR_LLEP_HD uint64_t saturatedMul(uint64_t lhs, uint64_t rhs) noexcept
    {
        if (lhs != 0ULL && rhs > (~0ULL) / lhs)
            return ~0ULL;
        return lhs * rhs;
    }

    /**
     * @brief Compute the serialized payload width of one transfer wave.
     *
     * Every participant can issue and receive on its own transport lane while
     * the wave is in flight. Transfers incident to the same source or the same
     * destination serialize on that participant's lane; independent edges and
     * the two directions of a reciprocal pair overlap. The economic multiplier
     * is consequently the maximum outgoing or incoming edge count of any
     * participant, rather than the global number of logical transfers.
     *
     * A malformed or unavailable manifest returns the conservative global
     * count. Callers still validate the manifest separately before publishing
     * it; this fallback only prevents an invalid diagnostic input from
     * understating cost.
     *
     * @param transfers Canonical logical payload-transfer manifest.
     * @param transfer_count Number of readable manifest entries.
     * @param participant_count Number of participants in the transfer domain.
     * @return Maximum serialized outgoing or incoming payload slots.
     */
    LLAMINAR_LLEP_HD uint32_t transferCriticalPathSlotCount(
        const LeastLoadedExpertWeightTransfer *transfers,
        uint32_t transfer_count,
        uint32_t participant_count) noexcept
    {
        if (transfer_count == 0u)
            return 0u;
        if (transfers == nullptr || participant_count == 0u ||
            participant_count > 64u)
        {
            return transfer_count;
        }

        uint32_t critical_path_slots = 0u;
        for (uint32_t participant = 0u;
             participant < participant_count;
             ++participant)
        {
            uint32_t outgoing = 0u;
            uint32_t incoming = 0u;
            for (uint32_t transfer_index = 0u;
                 transfer_index < transfer_count;
                 ++transfer_index)
            {
                const auto &transfer = transfers[transfer_index];
                if (transfer.source_participant >= participant_count ||
                    transfer.destination_participant >= participant_count ||
                    transfer.source_participant ==
                        transfer.destination_participant)
                {
                    return transfer_count;
                }
                if (transfer.source_participant == participant)
                    ++outgoing;
                if (transfer.destination_participant == participant)
                    ++incoming;
            }
            if (outgoing > critical_path_slots)
                critical_path_slots = outgoing;
            if (incoming > critical_path_slots)
                critical_path_slots = incoming;
        }
        return critical_path_slots;
    }

    LLAMINAR_LLEP_HD uint64_t ceilDiv(uint64_t numerator, uint64_t denominator) noexcept
    {
        if (denominator == 0ULL)
            return ~0ULL;
        return numerator / denominator + ((numerator % denominator) != 0ULL ? 1ULL : 0ULL);
    }

    /**
     * @brief Decide whether static-owner participant work is already balanced.
     *
     * LLEP changes which participant executes routed rows, so its admission
     * decision must compare participant loads.  Comparing the hottest expert
     * with the mean expert load answers a different question: an individual
     * expert can be very popular while ordinal ownership still distributes
     * total work evenly, and many individually ordinary experts can accumulate
     * on one participant.  Either mismatch can make an otherwise correct LLEP
     * plan slower than static-owner execution.
     *
     * The comparison is `max_participant_load / mean_participant_load <
     * lambda`, expressed with saturating integer products so CPU, CUDA, and
     * ROCm take exactly the same branch.
     *
     * @param total_load Total routed rows in the current assignment window.
     * @param max_participant_load Static-owner load on the busiest participant.
     * @param participant_count Number of participants in the execution domain.
     * @param lambda_numerator Numerator of the accepted imbalance ratio.
     * @param lambda_denominator Denominator of the accepted imbalance ratio.
     * @return `true` when transport-backed redistribution should be skipped.
     */
    LLAMINAR_LLEP_HD bool isStaticOwnerParticipantLoadBalancedEnough(
        uint64_t total_load,
        uint64_t max_participant_load,
        uint32_t participant_count,
        uint32_t lambda_numerator,
        uint32_t lambda_denominator) noexcept
    {
        if (total_load == 0ULL || participant_count == 0u)
            return true;
        if (lambda_denominator == 0u || lambda_numerator == 0u)
            return false;

        // max(load) / mean(load) < lambda, rewritten without floating point:
        // max * participants * denominator < total * numerator.
        const uint64_t lhs = saturatedMul(
            saturatedMul(
                max_participant_load,
                static_cast<uint64_t>(participant_count)),
            static_cast<uint64_t>(lambda_denominator));
        const uint64_t rhs = saturatedMul(
            total_load,
            static_cast<uint64_t>(lambda_numerator));
        return lhs < rhs;
    }

    LLAMINAR_LLEP_HD uint64_t availableCapacity(
        uint64_t capacity,
        uint64_t assigned,
        uint64_t pending) noexcept
    {
        if (assigned >= capacity)
            return 0ULL;
        const uint64_t remaining_after_assigned = capacity - assigned;
        return pending >= remaining_after_assigned ? 0ULL : remaining_after_assigned - pending;
    }

    LLAMINAR_LLEP_HD void summarizeParticipantLoads(
        const uint64_t *loads,
        uint32_t participant_count,
        uint64_t *min_out,
        uint64_t *max_out,
        uint64_t *spread_out) noexcept
    {
        uint64_t min_value = 0ULL;
        uint64_t max_value = 0ULL;
        if (loads != nullptr && participant_count > 0u)
        {
            min_value = loads[0];
            max_value = loads[0];
            for (uint32_t participant = 1; participant < participant_count; ++participant)
            {
                const uint64_t load = loads[participant];
                if (load < min_value)
                    min_value = load;
                if (load > max_value)
                    max_value = load;
            }
        }

        if (min_out)
            *min_out = min_value;
        if (max_out)
            *max_out = max_value;
        if (spread_out)
            *spread_out = max_value >= min_value ? max_value - min_value : 0ULL;
    }

    LLAMINAR_LLEP_HD uint64_t spreadImprovement(uint64_t before, uint64_t after) noexcept
    {
        return before > after ? before - after : 0ULL;
    }

    LLAMINAR_LLEP_HD uint64_t requiredSpreadImprovement(
        const LeastLoadedExpertAssignmentConfig &config,
        uint64_t total_load,
        uint32_t critical_path_transfer_slots) noexcept
    {
        const uint64_t transfer_component = saturatedMul(
            static_cast<uint64_t>(critical_path_transfer_slots),
            config.min_spread_improvement_per_critical_path_slot);
        uint64_t base = config.min_spread_improvement;
        if (config.min_spread_improvement_divisor > 0u)
        {
            const uint64_t relative_floor =
                total_load / static_cast<uint64_t>(config.min_spread_improvement_divisor);
            if (relative_floor > base)
                base = relative_floor;
        }
        if (transfer_component > (~0ULL) - base)
            return ~0ULL;
        return base + transfer_component;
    }

    LLAMINAR_LLEP_HD uint64_t requiredForeignRows(
        const LeastLoadedExpertAssignmentConfig &config,
        uint32_t critical_path_transfer_slots) noexcept
    {
        return saturatedMul(
            static_cast<uint64_t>(critical_path_transfer_slots),
            config.min_foreign_rows_per_critical_path_slot);
    }

    LLAMINAR_LLEP_HD void selectStandardEP(
        LeastLoadedExpertAssignmentStatus &status) noexcept
    {
        status.span_count = 0u;
        status.weight_transfer_count = 0u;
        status.native_rows = 0ULL;
        status.spilled_rows = 0ULL;
        status.min_chunk_skips = 0u;
        status.forced_spills = 0u;
        status.assigned_load_min = status.standard_load_min;
        status.assigned_load_max = status.standard_load_max;
        status.assigned_load_spread = status.standard_load_spread;
        status.assigned_load_spread_improvement = 0ULL;
        status.standard_ep_selected = 1u;
    }

    LLAMINAR_LLEP_HD uint32_t participantMaskLimit(uint32_t participant_count) noexcept
    {
        if (participant_count == 0u)
            return 0u;
        if (participant_count >= 32u)
            return 0xffffffffu;
        return (1u << participant_count) - 1u;
    }

    LLAMINAR_LLEP_HD uint32_t normalizeResidentParticipantMask(
        uint32_t resident_mask,
        int32_t owner_participant,
        uint32_t local_participant,
        uint32_t participant_count) noexcept
    {
        const uint32_t valid_mask = participantMaskLimit(participant_count);
        resident_mask &= valid_mask;
        if (resident_mask != 0u)
            return resident_mask;

        if (owner_participant >= 0 &&
            static_cast<uint32_t>(owner_participant) < participant_count)
        {
            return 1u << static_cast<uint32_t>(owner_participant);
        }

        const uint32_t fallback =
            local_participant < participant_count ? local_participant : 0u;
        return participant_count == 0u ? 0u : (1u << fallback);
    }

    LLAMINAR_LLEP_HD uint32_t residentParticipantMaskOrOwner(
        uint32_t resident_mask,
        uint32_t owner_participant,
        uint32_t participant_count) noexcept
    {
        resident_mask &= participantMaskLimit(participant_count);
        if (resident_mask != 0u)
            return resident_mask;
        if (owner_participant < participant_count && owner_participant < 32u)
            return 1u << owner_participant;
        return 0u;
    }

    LLAMINAR_LLEP_HD uint32_t selectWeightSourceParticipant(
        uint32_t owner_participant,
        uint32_t destination_participant,
        uint32_t resident_participant_mask,
        uint32_t participant_count) noexcept
    {
        const uint32_t resident_mask = residentParticipantMaskOrOwner(
            resident_participant_mask,
            owner_participant,
            participant_count);
        if (destination_participant < participant_count &&
            destination_participant < 32u &&
            (resident_mask & (1u << destination_participant)) != 0u)
        {
            return destination_participant;
        }
        if (owner_participant < participant_count &&
            owner_participant < 32u &&
            (resident_mask & (1u << owner_participant)) != 0u)
        {
            return owner_participant;
        }
        for (uint32_t participant = 0; participant < participant_count && participant < 32u; ++participant)
        {
            if ((resident_mask & (1u << participant)) != 0u)
                return participant;
        }
        if (owner_participant < participant_count)
            return owner_participant;
        if (destination_participant < participant_count)
            return destination_participant;
        return 0u;
    }

    LLAMINAR_LLEP_HD int32_t selectHighestLoadUnassignedExpert(
        const int32_t *expert_loads,
        const int32_t *assigned_participants,
        uint32_t expert_count) noexcept
    {
        if (!expert_loads || !assigned_participants)
            return -1;

        int32_t best_expert = -1;
        int32_t best_load = 0;
        for (uint32_t expert = 0; expert < expert_count; ++expert)
        {
            const int32_t load = expert_loads[expert];
            if (load <= 0 || assigned_participants[expert] >= 0)
                continue;
            if (best_expert < 0 ||
                load > best_load ||
                (load == best_load && static_cast<int32_t>(expert) < best_expert))
            {
                best_expert = static_cast<int32_t>(expert);
                best_load = load;
            }
        }
        return best_expert;
    }

    LLAMINAR_LLEP_HD uint32_t selectLeastLoadedResidentParticipant(
        uint32_t resident_mask,
        const uint64_t *participant_loads,
        uint32_t participant_count,
        uint32_t fallback_participant) noexcept
    {
        if (!participant_loads || participant_count == 0u)
            return 0u;

        uint32_t best_participant = kInvalidParticipant;
        uint64_t best_load = 0ULL;
        resident_mask &= participantMaskLimit(participant_count);
        for (uint32_t participant = 0; participant < participant_count; ++participant)
        {
            if ((resident_mask & (1u << participant)) == 0u)
                continue;
            const uint64_t load = participant_loads[participant];
            if (best_participant == kInvalidParticipant ||
                load < best_load ||
                (load == best_load && participant < best_participant))
            {
                best_participant = participant;
                best_load = load;
            }
        }
        if (best_participant != kInvalidParticipant)
            return best_participant;
        return fallback_participant < participant_count ? fallback_participant : 0u;
    }

    /**
     * @brief Build the batch-invariant tie turn for one logical route.
     *
     * Replica placement changes the grouping of floating-point partial sums
     * before the participant allreduce. Its tie-break key must therefore be a
     * pure function of committed model semantics, never of how many speculative
     * rows happened to execute. Adjacent logical positions advance by one so a
     * repeatedly hot expert still rotates across equally loaded residents.
     * Expert and route-slot terms decorrelate different routes in the same row.
     *
     * @param logical_position Absolute sequence position consumed by RoPE.
     * @param expert_id Selected global expert id.
     * @param route_slot Original top-k route slot within the row.
     */
    LLAMINAR_LLEP_HD uint64_t residentAssignmentTieTurn(
        int32_t logical_position,
        int expert_id,
        int route_slot) noexcept
    {
        const uint64_t position =
            logical_position >= 0
                ? static_cast<uint64_t>(
                      static_cast<uint32_t>(logical_position))
                : 0ULL;
        const uint64_t expert =
            expert_id >= 0
                ? static_cast<uint64_t>(
                      static_cast<uint32_t>(expert_id))
                : 0ULL;
        const uint64_t slot =
            route_slot >= 0
                ? static_cast<uint64_t>(
                      static_cast<uint32_t>(route_slot))
                : 0ULL;
        return position + expert * 131ULL + slot;
    }

    /**
     * @brief Select a batch-invariant least-loaded resident participant.
     *
     * `tie_turn` affects only participants tied at the minimum row-local load;
     * a genuinely lighter resident always wins. Resident participants are
     * enumerated in ascending id order. Serial CUDA/ROCm decode, grouped
     * CUDA/ROCm verification, and CPU test oracles all call this same primitive
     * with `residentAssignmentTieTurn()`, so changing verifier M or executing
     * rejected speculative rows cannot alter a committed row's partition.
     *
     * @param resident_mask Resident participants for the selected expert.
     * @param participant_loads Row-local loads accumulated by earlier top-k
     *        route slots.
     * @param participant_count Number of active collective participants.
     * @param fallback_participant Owner/local fallback used only when the
     *        placement mask is empty.
     * @param tie_turn Position-derived ordinal used only for equal-load ties.
     */
    LLAMINAR_LLEP_HD uint32_t
    selectBatchInvariantResidentParticipant(
        uint32_t resident_mask,
        const int *participant_loads,
        uint32_t participant_count,
        uint32_t fallback_participant,
        uint64_t tie_turn) noexcept
    {
        if (!participant_loads || participant_count == 0u)
            return 0u;

        resident_mask &= participantMaskLimit(participant_count);
        if (resident_mask == 0u)
        {
            const uint32_t fallback =
                fallback_participant < participant_count
                    ? fallback_participant
                    : 0u;
            resident_mask = 1u << fallback;
        }

        uint32_t resident_count = 0u;
        for (uint32_t participant = 0u;
             participant < participant_count;
             ++participant)
        {
            if ((resident_mask & (1u << participant)) != 0u)
                ++resident_count;
        }

        const uint32_t preferred_ordinal =
            static_cast<uint32_t>(
                tie_turn % static_cast<uint64_t>(resident_count));
        uint32_t preferred_participant = kInvalidParticipant;
        uint32_t resident_ordinal = 0u;
        for (uint32_t participant = 0u;
             participant < participant_count;
             ++participant)
        {
            if ((resident_mask & (1u << participant)) == 0u)
                continue;
            if (resident_ordinal == preferred_ordinal)
            {
                preferred_participant = participant;
                break;
            }
            ++resident_ordinal;
        }

        uint32_t best_participant = kInvalidParticipant;
        int best_load = 0;
        for (uint32_t participant = 0u;
             participant < participant_count;
             ++participant)
        {
            if ((resident_mask & (1u << participant)) == 0u)
                continue;
            const int load = participant_loads[participant];
            if (best_participant == kInvalidParticipant ||
                load < best_load ||
                (load == best_load &&
                 participant == preferred_participant))
            {
                best_participant = participant;
                best_load = load;
            }
        }
        if (best_participant != kInvalidParticipant)
            return best_participant;
        return fallback_participant < participant_count
                   ? fallback_participant
                   : 0u;
    }

    LLAMINAR_LLEP_HD void assignLeastLoadedResidentSplitCounts(
        uint64_t route_rows,
        uint32_t resident_mask,
        uint64_t *participant_loads,
        uint32_t participant_count,
        uint32_t fallback_participant,
        uint32_t *destination_counts) noexcept
    {
        if (route_rows == 0ULL ||
            !participant_loads ||
            !destination_counts ||
            participant_count == 0u ||
            participant_count > 32u)
        {
            return;
        }

        resident_mask &= participantMaskLimit(participant_count);
        if (resident_mask == 0u)
        {
            const uint32_t fallback =
                fallback_participant < participant_count ? fallback_participant : 0u;
            resident_mask = 1u << fallback;
        }

        uint64_t remaining = route_rows;
        while (remaining > 0ULL)
        {
            uint64_t min_load = 0ULL;
            bool have_min = false;
            for (uint32_t participant = 0; participant < participant_count; ++participant)
            {
                if ((resident_mask & (1u << participant)) == 0u)
                    continue;
                const uint64_t load = participant_loads[participant];
                if (!have_min || load < min_load)
                {
                    min_load = load;
                    have_min = true;
                }
            }
            if (!have_min)
                return;

            uint32_t min_count = 0u;
            uint64_t next_load = ~0ULL;
            bool have_next = false;
            for (uint32_t participant = 0; participant < participant_count; ++participant)
            {
                if ((resident_mask & (1u << participant)) == 0u)
                    continue;
                const uint64_t load = participant_loads[participant];
                if (load == min_load)
                {
                    ++min_count;
                }
                else if (load > min_load && (!have_next || load < next_load))
                {
                    next_load = load;
                    have_next = true;
                }
            }
            if (min_count == 0u)
                return;

            uint64_t chunk = remaining;
            if (have_next)
            {
                const uint64_t delta = next_load - min_load;
                const uint64_t fill_to_next =
                    saturatedMul(delta, static_cast<uint64_t>(min_count));
                if (fill_to_next > 0ULL && fill_to_next < chunk)
                    chunk = fill_to_next;
            }
            if (chunk == 0ULL)
                chunk = 1ULL;

            const uint64_t base = chunk / static_cast<uint64_t>(min_count);
            uint64_t extra = chunk % static_cast<uint64_t>(min_count);
            for (uint32_t participant = 0; participant < participant_count; ++participant)
            {
                if ((resident_mask & (1u << participant)) == 0u ||
                    participant_loads[participant] != min_load)
                {
                    continue;
                }
                uint64_t add = base;
                if (extra > 0ULL)
                {
                    ++add;
                    --extra;
                }
                if (add == 0ULL)
                    continue;
                participant_loads[participant] += add;
                destination_counts[participant] += static_cast<uint32_t>(add);
            }

            remaining -= chunk;
        }
    }

    LLAMINAR_LLEP_HD void sortExpertsByLoadDescending(
        const uint64_t *expert_loads,
        uint32_t expert_count,
        uint32_t *sorted_experts) noexcept
    {
        if (sorted_experts == nullptr)
            return;
        for (uint32_t i = 0; i < expert_count; ++i)
            sorted_experts[i] = i;

        for (uint32_t i = 1; i < expert_count; ++i)
        {
            const uint32_t expert = sorted_experts[i];
            const uint64_t load = expert_loads ? expert_loads[expert] : 0ULL;
            uint32_t j = i;
            while (j > 0)
            {
                const uint32_t previous_expert = sorted_experts[j - 1u];
                const uint64_t previous_load =
                    expert_loads ? expert_loads[previous_expert] : 0ULL;
                if (previous_load > load ||
                    (previous_load == load && previous_expert < expert))
                {
                    break;
                }
                sorted_experts[j] = previous_expert;
                --j;
            }
            sorted_experts[j] = expert;
        }
    }

    LLAMINAR_LLEP_HD bool appendWeightTransferIfMissing(
        LeastLoadedExpertWeightTransfer *transfers,
        uint32_t transfer_capacity,
        LeastLoadedExpertAssignmentStatus &status,
        uint32_t expert,
        uint32_t source_participant,
        uint32_t destination_participant) noexcept
    {
        if (source_participant == destination_participant)
            return true;

        /*
         * Transfers are emitted while visiting each expert exactly once in
         * canonical sorted order.  Entries for one expert are consequently a
         * contiguous suffix.  Walking that suffix backwards avoids an O(T^2)
         * scan across unrelated experts without changing duplicate semantics.
         */
        for (uint32_t i = status.weight_transfer_count; i > 0u; --i)
        {
            const auto &transfer = transfers[i - 1u];
            if (transfer.expert != expert)
                break;
            if (transfer.expert == expert &&
                transfer.source_participant == source_participant &&
                transfer.destination_participant == destination_participant)
            {
                return true;
            }
        }

        if (status.weight_transfer_count >= transfer_capacity || transfers == nullptr)
        {
            status.overflow = 1u;
            return false;
        }

        transfers[status.weight_transfer_count++] =
            LeastLoadedExpertWeightTransfer{
                expert,
                source_participant,
                destination_participant};
        return true;
    }

    LLAMINAR_LLEP_HD bool destinationNeedsForeignWeight(
        uint32_t owner_participant,
        uint32_t destination_participant,
        uint32_t resident_participant_mask,
        uint32_t participant_count = 32u) noexcept
    {
        const uint32_t resident_mask = residentParticipantMaskOrOwner(
            resident_participant_mask,
            owner_participant,
            participant_count);
        return destination_participant >= 32u ||
               (resident_mask & (1u << destination_participant)) == 0u;
    }

    LLAMINAR_LLEP_HD bool hasWeightTransfer(
        const LeastLoadedExpertWeightTransfer *transfers,
        const LeastLoadedExpertAssignmentStatus &status,
        uint32_t expert,
        uint32_t source_participant,
        uint32_t destination_participant) noexcept
    {
        if (source_participant == destination_participant)
            return true;
        if (!transfers)
            return false;
        for (uint32_t i = status.weight_transfer_count; i > 0u; --i)
        {
            const auto &transfer = transfers[i - 1u];
            if (transfer.expert != expert)
                break;
            if (transfer.expert == expert &&
                transfer.source_participant == source_participant &&
                transfer.destination_participant == destination_participant)
            {
                return true;
            }
        }
        return false;
    }

    LLAMINAR_LLEP_HD bool canAssignDestinationWithTransferCapacity(
        const LeastLoadedExpertWeightTransfer *transfers,
        uint32_t transfer_capacity,
        const LeastLoadedExpertAssignmentStatus &status,
        uint32_t expert,
        uint32_t owner_participant,
        uint32_t destination_participant,
        uint32_t resident_participant_mask,
        uint32_t participant_count = 32u) noexcept
    {
        if (!destinationNeedsForeignWeight(
                owner_participant,
                destination_participant,
                resident_participant_mask,
                participant_count))
        {
            return true;
        }
        const uint32_t source_participant = selectWeightSourceParticipant(
            owner_participant,
            destination_participant,
            resident_participant_mask,
            participant_count);
        return hasWeightTransfer(
                   transfers,
                   status,
                   expert,
                   source_participant,
                   destination_participant) ||
               status.weight_transfer_count < transfer_capacity;
    }

    /**
     * @brief Report whether an expert already occupies a participant work-set slot.
     *
     * Multiple row chunks for the same `(expert, destination)` pair consume one
     * expert payload. This predicate distinguishes another chunk from a new
     * logical payload.
     */
    LLAMINAR_LLEP_HD bool hasNonOwnerExpertAssignment(
        const LeastLoadedExpertAssignmentSpan *spans,
        const LeastLoadedExpertAssignmentStatus &status,
        uint32_t expert,
        uint32_t destination_participant) noexcept
    {
        if (!spans)
            return false;
        for (uint32_t i = 0; i < status.span_count; ++i)
        {
            const auto &span = spans[i];
            if (span.expert == expert &&
                span.destination_participant == destination_participant &&
                span.owner_participant != destination_participant)
            {
                return true;
            }
        }
        return false;
    }

    /**
     * @brief Count distinct non-owner experts assigned to one participant.
     *
     * The fixed caller-owned span array doubles as the deterministic work-set
     * ledger. The duplicate check avoids adding scratch storage to every
     * CPU/CUDA/ROCm caller and remains small because production transfer-slot
     * capacities are intentionally tight.
     */
    LLAMINAR_LLEP_HD uint32_t countNonOwnerExpertAssignments(
        const LeastLoadedExpertAssignmentSpan *spans,
        const LeastLoadedExpertAssignmentStatus &status,
        uint32_t destination_participant) noexcept
    {
        if (!spans)
            return 0u;

        uint32_t count = 0u;
        for (uint32_t i = 0; i < status.span_count; ++i)
        {
            const auto &candidate = spans[i];
            if (candidate.destination_participant != destination_participant ||
                candidate.owner_participant == destination_participant)
            {
                continue;
            }

            bool already_counted = false;
            for (uint32_t previous = 0u; previous < i; ++previous)
            {
                const auto &prior = spans[previous];
                if (prior.expert == candidate.expert &&
                    prior.destination_participant == destination_participant &&
                    prior.owner_participant != destination_participant)
                {
                    already_counted = true;
                    break;
                }
            }
            if (!already_counted)
                ++count;
        }
        return count;
    }

    /**
     * @brief Validate the complete non-owner payload working-set capacity.
     */
    LLAMINAR_LLEP_HD bool canAssignDestinationWithWorkingSetCapacity(
        const LeastLoadedExpertAssignmentSpan *spans,
        const LeastLoadedExpertAssignmentStatus &status,
        uint32_t expert,
        uint32_t owner_participant,
        uint32_t destination_participant,
        uint32_t max_non_owner_experts_per_participant) noexcept
    {
        if (destination_participant == owner_participant ||
            max_non_owner_experts_per_participant == 0u ||
            hasNonOwnerExpertAssignment(
                spans,
                status,
                expert,
                destination_participant))
        {
            return true;
        }

        return countNonOwnerExpertAssignments(
                   spans,
                   status,
                   destination_participant) <
               max_non_owner_experts_per_participant;
    }

    LLAMINAR_LLEP_HD void countConceptualAssignmentSpan(
        LeastLoadedExpertAssignmentStatus &status) noexcept
    {
        if (status.span_count != ~0u)
            ++status.span_count;
    }

    LLAMINAR_LLEP_HD bool appendAssignmentSpan(
        LeastLoadedExpertAssignmentSpan *spans,
        uint32_t span_capacity,
        LeastLoadedExpertWeightTransfer *transfers,
        uint32_t transfer_capacity,
        LeastLoadedExpertAssignmentStatus &status,
        uint32_t expert,
        uint32_t owner_participant,
        uint32_t destination_participant,
        uint32_t resident_participant_mask,
        uint64_t begin,
        uint64_t end,
        bool forced,
        uint32_t max_non_owner_experts_per_participant,
        uint32_t participant_count = 32u) noexcept
    {
        if (end <= begin)
            return true;
        if (status.span_count >= span_capacity || spans == nullptr)
        {
            status.overflow = 1u;
            return false;
        }

        /*
         * Candidate selection normally checks this constraint before append.
         * Recheck at the publication boundary so future policy changes cannot
         * accidentally emit an over-capacity assignment by bypassing selection.
         */
        if (!canAssignDestinationWithWorkingSetCapacity(
                spans,
                status,
                expert,
                owner_participant,
                destination_participant,
                max_non_owner_experts_per_participant))
        {
            status.overflow = 1u;
            return false;
        }

        const bool foreign = destinationNeedsForeignWeight(
            owner_participant,
            destination_participant,
            resident_participant_mask,
            participant_count);
        const uint64_t rows = end - begin;
        if (foreign)
        {
            const uint32_t source_participant = selectWeightSourceParticipant(
                owner_participant,
                destination_participant,
                resident_participant_mask,
                participant_count);
            if (!appendWeightTransferIfMissing(
                transfers,
                transfer_capacity,
                status,
                expert,
                source_participant,
                destination_participant))
            {
                return false;
            }
        }

        spans[status.span_count++] = LeastLoadedExpertAssignmentSpan{
            expert,
            owner_participant,
            destination_participant,
            begin,
            end,
            static_cast<uint8_t>(foreign ? 1u : 0u),
            static_cast<uint8_t>(forced ? 1u : 0u),
            0u};

        if (foreign)
            status.spilled_rows += rows;
        else
            status.native_rows += rows;
        return true;
    }

    LLAMINAR_LLEP_HD uint32_t leastLoadedOtherParticipant(
        uint32_t native_participant,
        uint32_t participant_count,
        const uint64_t *pending_load,
        const uint64_t *assigned_load) noexcept
    {
        uint32_t best = kInvalidParticipant;
        uint64_t best_load = 0ULL;
        for (uint32_t participant = 0; participant < participant_count; ++participant)
        {
            if (participant == native_participant && participant_count > 1u)
                continue;
            const uint64_t load =
                (assigned_load ? assigned_load[participant] : 0ULL) +
                (pending_load ? pending_load[participant] : 0ULL);
            if (best == kInvalidParticipant ||
                load < best_load ||
                (load == best_load && participant < best))
            {
                best = participant;
                best_load = load;
            }
        }
        return best == kInvalidParticipant ? native_participant : best;
    }

    LLAMINAR_LLEP_HD bool spillLeastLoaded(
        LeastLoadedExpertAssignmentSpan *spans,
        uint32_t span_capacity,
        LeastLoadedExpertWeightTransfer *transfers,
        uint32_t transfer_capacity,
        LeastLoadedExpertAssignmentStatus &status,
        const LeastLoadedExpertAssignmentConfig &config,
        const LeastLoadedExpertAssignmentWorkspace &workspace,
        uint32_t expert,
        uint32_t owner_participant,
        uint32_t resident_participant_mask,
        uint64_t remaining_rows,
        uint64_t route_row_offset) noexcept
    {
        while (remaining_rows > 0ULL)
        {
            bool assigned = false;
            uint32_t skipped_participants = 0;
            uint64_t skipped_mask = 0ULL;

            while (skipped_participants < config.participant_count)
            {
                uint32_t best = kInvalidParticipant;
                uint64_t best_load = 0ULL;
                for (uint32_t participant = 0; participant < config.participant_count; ++participant)
                {
                    if (participant == owner_participant && config.participant_count > 1u)
                        continue;
                    if (participant < 64u && (skipped_mask & (1ULL << participant)) != 0ULL)
                        continue;
                    const uint64_t load =
                        workspace.assigned_load[participant] +
                        workspace.pending_load[participant];
                    if (best == kInvalidParticipant ||
                        load < best_load ||
                        (load == best_load && participant < best))
                    {
                        best = participant;
                        best_load = load;
                    }
                }

                if (best == kInvalidParticipant)
                    break;
                if (!canAssignDestinationWithTransferCapacity(
                        transfers,
                        transfer_capacity,
                        status,
                        expert,
                        owner_participant,
                        best,
                        resident_participant_mask,
                        config.participant_count) ||
                    !canAssignDestinationWithWorkingSetCapacity(
                        spans,
                        status,
                        expert,
                        owner_participant,
                        best,
                        config.max_non_owner_experts_per_participant))
                {
                    ++skipped_participants;
                    if (best < 64u)
                        skipped_mask |= (1ULL << best);
                    else
                        break;
                    continue;
                }

                const uint64_t available = availableCapacity(
                    status.capacity_per_participant,
                    workspace.assigned_load[best],
                    workspace.pending_load[best]);
                const uint64_t chunk = remaining_rows < available ? remaining_rows : available;
                if (chunk == 0ULL ||
                    (config.min_chunk_tokens > 0u &&
                     chunk < static_cast<uint64_t>(config.min_chunk_tokens) &&
                     remaining_rows > chunk))
                {
                    ++status.min_chunk_skips;
                    ++skipped_participants;
                    if (best < 64u)
                        skipped_mask |= (1ULL << best);
                    else
                        break;
                    continue;
                }

                if (!appendAssignmentSpan(
                        spans,
                        span_capacity,
                        transfers,
                        transfer_capacity,
                        status,
                        expert,
                        owner_participant,
                        best,
                        resident_participant_mask,
                        route_row_offset,
                        route_row_offset + chunk,
                        false,
                        config.max_non_owner_experts_per_participant,
                        config.participant_count))
                {
                    return false;
                }
                workspace.assigned_load[best] += chunk;
                remaining_rows -= chunk;
                route_row_offset += chunk;
                assigned = true;
                break;
            }

            if (!assigned)
            {
                uint32_t forced_participant = leastLoadedOtherParticipant(
                    owner_participant,
                    config.participant_count,
                    workspace.pending_load,
                    workspace.assigned_load);
                if (!canAssignDestinationWithTransferCapacity(
                        transfers,
                        transfer_capacity,
                        status,
                        expert,
                        owner_participant,
                        forced_participant,
                        resident_participant_mask,
                        config.participant_count) ||
                    !canAssignDestinationWithWorkingSetCapacity(
                        spans,
                        status,
                        expert,
                        owner_participant,
                        forced_participant,
                        config.max_non_owner_experts_per_participant))
                {
                    forced_participant = owner_participant;
                }
                if (!appendAssignmentSpan(
                        spans,
                        span_capacity,
                        transfers,
                        transfer_capacity,
                        status,
                        expert,
                        owner_participant,
                        forced_participant,
                        resident_participant_mask,
                        route_row_offset,
                        route_row_offset + remaining_rows,
                        true,
                        config.max_non_owner_experts_per_participant,
                        config.participant_count))
                {
                    return false;
                }
                workspace.assigned_load[forced_participant] += remaining_rows;
                route_row_offset += remaining_rows;
                remaining_rows = 0ULL;
                ++status.forced_spills;
            }
        }

        return true;
    }

    LLAMINAR_LLEP_HD bool planLeastLoadedExpertAssignment(
        const uint64_t *expert_loads,
        const uint32_t *expert_owner_participants,
        const LeastLoadedExpertAssignmentConfig &config,
        const LeastLoadedExpertAssignmentWorkspace &workspace,
        LeastLoadedExpertAssignmentSpan *spans,
        uint32_t span_capacity,
        LeastLoadedExpertWeightTransfer *transfers,
        uint32_t transfer_capacity,
        LeastLoadedExpertAssignmentStatus &status,
        const uint32_t *expert_resident_participant_masks = nullptr,
        bool workspace_experts_are_sorted = false) noexcept
    {
        status = {};

        if (expert_loads == nullptr ||
            expert_owner_participants == nullptr ||
            workspace.sorted_experts == nullptr ||
            workspace.pending_load == nullptr ||
            workspace.assigned_load == nullptr ||
            config.expert_count == 0u ||
            config.participant_count == 0u ||
            config.participant_count > 64u ||
            config.alpha_numerator == 0u ||
            config.alpha_denominator == 0u)
        {
            status.invalid_config = 1u;
            return false;
        }

        for (uint32_t participant = 0; participant < config.participant_count; ++participant)
        {
            workspace.pending_load[participant] = 0ULL;
            workspace.assigned_load[participant] = 0ULL;
        }

        for (uint32_t expert = 0; expert < config.expert_count; ++expert)
        {
            const uint32_t owner = expert_owner_participants[expert];
            if (owner >= config.participant_count)
            {
                status.invalid_config = 1u;
                return false;
            }
            const uint64_t load = expert_loads[expert];
            status.total_load += load;
            if (load > status.max_expert_load)
                status.max_expert_load = load;
            workspace.pending_load[owner] += load;
        }

        summarizeParticipantLoads(
            workspace.pending_load,
            config.participant_count,
            &status.standard_load_min,
            &status.standard_load_max,
            &status.standard_load_spread);

        if (config.enable_balanced_skip &&
            isStaticOwnerParticipantLoadBalancedEnough(
                status.total_load,
                status.standard_load_max,
                config.participant_count,
                config.lambda_numerator,
                config.lambda_denominator))
        {
            status.skipped_balanced = 1u;
            status.standard_ep_selected = 1u;
            status.assigned_load_min = status.standard_load_min;
            status.assigned_load_max = status.standard_load_max;
            status.assigned_load_spread = status.standard_load_spread;
            return true;
        }

        const uint64_t capacity_numerator =
            saturatedMul(status.total_load, static_cast<uint64_t>(config.alpha_numerator));
        const uint64_t capacity_denominator =
            static_cast<uint64_t>(config.participant_count) *
            static_cast<uint64_t>(config.alpha_denominator);
        status.capacity_per_participant =
            ceilDiv(capacity_numerator, capacity_denominator);
        const uint32_t effective_transfer_capacity =
            config.max_weight_transfers == 0u
                ? transfer_capacity
                : (config.max_weight_transfers < transfer_capacity
                       ? config.max_weight_transfers
                       : transfer_capacity);

        /*
         * Device planners can build this canonical ordering cooperatively
         * before the ordered assignment pass begins.  Keeping that fact in
         * the shared API prevents a GPU kernel from silently sorting the same
         * experts again on one lane.  Host callers retain the compact scalar
         * sort by accepting the default value.
         */
        if (!workspace_experts_are_sorted)
        {
            sortExpertsByLoadDescending(
                expert_loads,
                config.expert_count,
                workspace.sorted_experts);
        }

        for (uint32_t order = 0; order < config.expert_count; ++order)
        {
            const uint32_t expert = workspace.sorted_experts[order];
            const uint64_t load = expert_loads[expert];
            if (load == 0ULL)
                continue;

            const uint32_t owner = expert_owner_participants[expert];
            const uint32_t resident_mask =
                expert_resident_participant_masks
                    ? (expert_resident_participant_masks[expert] &
                       participantMaskLimit(config.participant_count))
                    : residentParticipantMaskOrOwner(
                          0u,
                          owner,
                          config.participant_count);
            if (expert_resident_participant_masks && resident_mask == 0u)
            {
                status.invalid_config = 1u;
                return false;
            }
            workspace.pending_load[owner] =
                workspace.pending_load[owner] >= load
                    ? workspace.pending_load[owner] - load
                    : 0ULL;

            const uint64_t native_available = availableCapacity(
                status.capacity_per_participant,
                workspace.assigned_load[owner],
                workspace.pending_load[owner]);
            if (native_available >= load)
            {
                if (!appendAssignmentSpan(
                        spans,
                        span_capacity,
                        transfers,
                        effective_transfer_capacity,
                        status,
                        expert,
                        owner,
                        owner,
                        resident_mask,
                        0ULL,
                        load,
                        false,
                        config.max_non_owner_experts_per_participant,
                        config.participant_count))
                {
                    return false;
                }
                workspace.assigned_load[owner] += load;
                continue;
            }

            uint64_t route_offset = 0ULL;
            uint64_t remaining = load;
            if (native_available > 0ULL)
            {
                if (!appendAssignmentSpan(
                        spans,
                        span_capacity,
                        transfers,
                        effective_transfer_capacity,
                        status,
                        expert,
                        owner,
                        owner,
                        resident_mask,
                        0ULL,
                        native_available,
                        false,
                        config.max_non_owner_experts_per_participant,
                        config.participant_count))
                {
                    return false;
                }
                workspace.assigned_load[owner] += native_available;
                route_offset = native_available;
                remaining -= native_available;
            }

            if (!spillLeastLoaded(
                    spans,
                    span_capacity,
                    transfers,
                    effective_transfer_capacity,
                    status,
                    config,
                    workspace,
                    expert,
                    owner,
                    resident_mask,
                    remaining,
                    route_offset))
            {
                return false;
            }
        }

        summarizeParticipantLoads(
            workspace.assigned_load,
            config.participant_count,
            &status.assigned_load_min,
            &status.assigned_load_max,
            &status.assigned_load_spread);
        status.assigned_load_spread_improvement =
            spreadImprovement(status.standard_load_spread, status.assigned_load_spread);

        status.critical_path_transfer_slots =
            transferCriticalPathSlotCount(
                transfers,
                status.weight_transfer_count,
                config.participant_count);
        const uint64_t required_improvement =
            requiredSpreadImprovement(
                config,
                status.total_load,
                status.critical_path_transfer_slots);
        status.required_spread_improvement = required_improvement;
        if (required_improvement > 0ULL &&
            status.assigned_load_spread_improvement < required_improvement)
        {
            selectStandardEP(status);
            status.skipped_insufficient_spread_improvement = 1u;
        }

        const uint64_t required_rows =
            requiredForeignRows(
                config,
                status.critical_path_transfer_slots);
        status.required_foreign_rows = required_rows;
        if (required_rows > 0ULL && status.spilled_rows < required_rows)
        {
            selectStandardEP(status);
            status.skipped_insufficient_foreign_rows = 1u;
        }

        return status.overflow == 0u;
    }

    LLAMINAR_LLEP_HD bool planLeastLoadedExpertWeightTransfers(
        const uint64_t *expert_loads,
        const uint32_t *expert_owner_participants,
        const LeastLoadedExpertAssignmentConfig &config,
        const LeastLoadedExpertAssignmentWorkspace &workspace,
        LeastLoadedExpertWeightTransfer *transfers,
        uint32_t transfer_capacity,
        LeastLoadedExpertAssignmentStatus &status,
        const uint32_t *expert_resident_participant_masks = nullptr,
        bool workspace_experts_are_sorted = false) noexcept
    {
        status = {};

        if (expert_loads == nullptr ||
            expert_owner_participants == nullptr ||
            workspace.sorted_experts == nullptr ||
            workspace.pending_load == nullptr ||
            workspace.assigned_load == nullptr ||
            config.expert_count == 0u ||
            config.participant_count == 0u ||
            config.participant_count > 64u ||
            config.alpha_numerator == 0u ||
            config.alpha_denominator == 0u)
        {
            status.invalid_config = 1u;
            return false;
        }

        for (uint32_t participant = 0; participant < config.participant_count; ++participant)
        {
            workspace.pending_load[participant] = 0ULL;
            workspace.assigned_load[participant] = 0ULL;
        }

        for (uint32_t expert = 0; expert < config.expert_count; ++expert)
        {
            const uint32_t owner = expert_owner_participants[expert];
            if (owner >= config.participant_count)
            {
                status.invalid_config = 1u;
                return false;
            }

            const uint64_t load = expert_loads[expert];
            status.total_load += load;
            if (load > status.max_expert_load)
                status.max_expert_load = load;
            workspace.pending_load[owner] += load;
        }

        summarizeParticipantLoads(
            workspace.pending_load,
            config.participant_count,
            &status.standard_load_min,
            &status.standard_load_max,
            &status.standard_load_spread);

        if (config.enable_balanced_skip &&
            isStaticOwnerParticipantLoadBalancedEnough(
                status.total_load,
                status.standard_load_max,
                config.participant_count,
                config.lambda_numerator,
                config.lambda_denominator))
        {
            status.skipped_balanced = 1u;
            status.standard_ep_selected = 1u;
            status.assigned_load_min = status.standard_load_min;
            status.assigned_load_max = status.standard_load_max;
            status.assigned_load_spread = status.standard_load_spread;
            return true;
        }

        const uint64_t capacity_numerator =
            saturatedMul(status.total_load, static_cast<uint64_t>(config.alpha_numerator));
        const uint64_t capacity_denominator =
            static_cast<uint64_t>(config.participant_count) *
            static_cast<uint64_t>(config.alpha_denominator);
        status.capacity_per_participant =
            ceilDiv(capacity_numerator, capacity_denominator);
        const uint32_t effective_transfer_capacity =
            config.max_weight_transfers == 0u
                ? transfer_capacity
                : (config.max_weight_transfers < transfer_capacity
                       ? config.max_weight_transfers
                       : transfer_capacity);

        /*
         * GPU planners may populate this deterministic order cooperatively
         * before entering the inherently ordered assignment pass.  Host
         * callers retain the compact scalar sort, while GPU callers avoid
         * repeating an O(E^2) insertion sort on a single device lane.
         */
        if (!workspace_experts_are_sorted)
        {
            sortExpertsByLoadDescending(
                expert_loads,
                config.expert_count,
                workspace.sorted_experts);
        }

        for (uint32_t order = 0; order < config.expert_count; ++order)
        {
            const uint32_t expert = workspace.sorted_experts[order];
            const uint64_t load = expert_loads[expert];
            if (load == 0ULL)
                continue;

            const uint32_t owner = expert_owner_participants[expert];
            const uint32_t resident_mask =
                expert_resident_participant_masks
                    ? (expert_resident_participant_masks[expert] &
                       participantMaskLimit(config.participant_count))
                    : residentParticipantMaskOrOwner(
                          0u,
                          owner,
                          config.participant_count);
            if (expert_resident_participant_masks && resident_mask == 0u)
            {
                status.invalid_config = 1u;
                return false;
            }
            workspace.pending_load[owner] =
                workspace.pending_load[owner] >= load
                    ? workspace.pending_load[owner] - load
                    : 0ULL;

            const uint64_t native_available = availableCapacity(
                status.capacity_per_participant,
                workspace.assigned_load[owner],
                workspace.pending_load[owner]);
            if (native_available >= load)
            {
                const bool foreign = destinationNeedsForeignWeight(
                    owner,
                    owner,
                    resident_mask,
                    config.participant_count);
                if (foreign)
                {
                    if (!appendWeightTransferIfMissing(
                            transfers,
                            effective_transfer_capacity,
                            status,
                            expert,
                            selectWeightSourceParticipant(
                                owner,
                                owner,
                                resident_mask,
                                config.participant_count),
                            owner))
                    {
                        return false;
                    }
                    status.spilled_rows += load;
                }
                else
                {
                    status.native_rows += load;
                }
                workspace.assigned_load[owner] += load;
                countConceptualAssignmentSpan(status);
                continue;
            }

            uint64_t remaining = load;
            if (native_available > 0ULL)
            {
                const bool foreign = destinationNeedsForeignWeight(
                    owner,
                    owner,
                    resident_mask,
                    config.participant_count);
                if (foreign)
                {
                    if (!appendWeightTransferIfMissing(
                            transfers,
                            effective_transfer_capacity,
                            status,
                            expert,
                            selectWeightSourceParticipant(
                                owner,
                                owner,
                                resident_mask,
                                config.participant_count),
                            owner))
                    {
                        return false;
                    }
                    status.spilled_rows += native_available;
                }
                else
                {
                    status.native_rows += native_available;
                }
                workspace.assigned_load[owner] += native_available;
                countConceptualAssignmentSpan(status);
                remaining -= native_available;
            }

            while (remaining > 0ULL)
            {
                bool assigned = false;
                uint32_t skipped_participants = 0;
                uint64_t skipped_mask = 0ULL;

                while (skipped_participants < config.participant_count)
                {
                    uint32_t best = kInvalidParticipant;
                    uint64_t best_load = 0ULL;
                    for (uint32_t participant = 0; participant < config.participant_count; ++participant)
                    {
                        if (participant == owner && config.participant_count > 1u)
                            continue;
                        if (participant < 64u && (skipped_mask & (1ULL << participant)) != 0ULL)
                            continue;
                        const uint64_t participant_load =
                            workspace.assigned_load[participant] +
                            workspace.pending_load[participant];
                        if (best == kInvalidParticipant ||
                            participant_load < best_load ||
                            (participant_load == best_load && participant < best))
                        {
                            best = participant;
                            best_load = participant_load;
                        }
                    }

                    if (best == kInvalidParticipant)
                        break;
                    if (!canAssignDestinationWithTransferCapacity(
                            transfers,
                            effective_transfer_capacity,
                            status,
                            expert,
                            owner,
                            best,
                            resident_mask,
                            config.participant_count))
                    {
                        ++skipped_participants;
                        if (best < 64u)
                            skipped_mask |= (1ULL << best);
                        else
                            break;
                        continue;
                    }

                    const uint64_t available = availableCapacity(
                        status.capacity_per_participant,
                        workspace.assigned_load[best],
                        workspace.pending_load[best]);
                    const uint64_t chunk = remaining < available ? remaining : available;
                    if (chunk == 0ULL ||
                        (config.min_chunk_tokens > 0u &&
                         chunk < static_cast<uint64_t>(config.min_chunk_tokens) &&
                         remaining > chunk))
                    {
                        ++status.min_chunk_skips;
                        ++skipped_participants;
                        if (best < 64u)
                            skipped_mask |= (1ULL << best);
                        else
                            break;
                        continue;
                    }

                    if (!appendWeightTransferIfMissing(
                            transfers,
                            effective_transfer_capacity,
                            status,
                            expert,
                            selectWeightSourceParticipant(
                                owner,
                                best,
                                resident_mask,
                                config.participant_count),
                            best))
                    {
                        return false;
                    }
                    workspace.assigned_load[best] += chunk;
                    status.spilled_rows += chunk;
                    countConceptualAssignmentSpan(status);
                    remaining -= chunk;
                    assigned = true;
                    break;
                }

                if (!assigned)
                {
                    uint32_t forced_participant = leastLoadedOtherParticipant(
                        owner,
                        config.participant_count,
                        workspace.pending_load,
                        workspace.assigned_load);
                    if (!canAssignDestinationWithTransferCapacity(
                            transfers,
                            effective_transfer_capacity,
                            status,
                            expert,
                            owner,
                            forced_participant,
                            resident_mask,
                            config.participant_count))
                    {
                        forced_participant = owner;
                    }
                    const bool forced_foreign = destinationNeedsForeignWeight(
                        owner,
                        forced_participant,
                        resident_mask,
                        config.participant_count);
                    if (forced_foreign &&
                        !appendWeightTransferIfMissing(
                            transfers,
                            effective_transfer_capacity,
                            status,
                            expert,
                            selectWeightSourceParticipant(
                                owner,
                                forced_participant,
                                resident_mask,
                                config.participant_count),
                            forced_participant))
                    {
                        return false;
                    }
                    workspace.assigned_load[forced_participant] += remaining;
                    if (!forced_foreign)
                        status.native_rows += remaining;
                    else
                        status.spilled_rows += remaining;
                    countConceptualAssignmentSpan(status);
                    remaining = 0ULL;
                    ++status.forced_spills;
                }
            }
        }

        summarizeParticipantLoads(
            workspace.assigned_load,
            config.participant_count,
            &status.assigned_load_min,
            &status.assigned_load_max,
            &status.assigned_load_spread);
        status.assigned_load_spread_improvement =
            spreadImprovement(status.standard_load_spread, status.assigned_load_spread);

        status.critical_path_transfer_slots =
            transferCriticalPathSlotCount(
                transfers,
                status.weight_transfer_count,
                config.participant_count);
        const uint64_t required_improvement =
            requiredSpreadImprovement(
                config,
                status.total_load,
                status.critical_path_transfer_slots);
        status.required_spread_improvement = required_improvement;
        if (required_improvement > 0ULL &&
            status.assigned_load_spread_improvement < required_improvement)
        {
            selectStandardEP(status);
            status.skipped_insufficient_spread_improvement = 1u;
        }

        const uint64_t required_rows =
            requiredForeignRows(
                config,
                status.critical_path_transfer_slots);
        status.required_foreign_rows = required_rows;
        if (required_rows > 0ULL && status.spilled_rows < required_rows)
        {
            selectStandardEP(status);
            status.skipped_insufficient_foreign_rows = 1u;
        }

        return status.overflow == 0u;
    }

} // namespace llaminar2::least_loaded_ep
