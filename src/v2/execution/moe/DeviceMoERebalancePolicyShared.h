/**
 * @file DeviceMoERebalancePolicyShared.h
 * @brief Host/device shared MoE rebalance policy helpers.
 *
 * This header owns the small policy decisions that must stay identical across
 * the CPU mirror, CUDA kernels, and ROCm kernels. Keep it dependency-light so it
 * can be included from normal C++, nvcc, and hipcc translation units.
 */

#pragma once

#include <cstdint>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define LLAMINAR_MOE_REBALANCE_HD __host__ __device__ __forceinline__
#else
#define LLAMINAR_MOE_REBALANCE_HD inline
#endif

namespace llaminar2::moe_rebalance_policy
{
    struct LoadSpreadDelta
    {
        uint64_t current_total = 0;
        uint64_t current_min = 0;
        uint64_t current_max = 0;
        uint64_t proposed_total = 0;
        uint64_t proposed_min = 0;
        uint64_t proposed_max = 0;
        uint64_t current_spread = 0;
        uint64_t proposed_spread = 0;
        uint64_t improvement = 0;
        uint64_t required_improvement = 0;
        bool total_preserved = false;
        bool improves = false;
        bool meets_floor = false;
    };

    struct DestinationChoice
    {
        uint32_t destination_participant = 0;
        uint32_t proposed_resident_mask = 0;
        LoadSpreadDelta delta{};
        bool valid = false;
    };

    LLAMINAR_MOE_REBALANCE_HD uint32_t participantBit(uint32_t participant) noexcept
    {
        return 1u << participant;
    }

    LLAMINAR_MOE_REBALANCE_HD uint64_t directedParticipantEdgeBit(
        uint32_t source_participant,
        uint32_t destination_participant,
        uint32_t participant_stride) noexcept
    {
        if (participant_stride == 0u ||
            source_participant >= participant_stride ||
            destination_participant >= participant_stride)
        {
            return 0ULL;
        }
        const uint32_t bit_index =
            source_participant * participant_stride + destination_participant;
        return bit_index < 64u ? (1ULL << bit_index) : 0ULL;
    }

    LLAMINAR_MOE_REBALANCE_HD uint32_t validParticipantMask(uint32_t participant_count) noexcept
    {
        return (1u << participant_count) - 1u;
    }

    LLAMINAR_MOE_REBALANCE_HD uint32_t residentCount(
        uint32_t resident_mask,
        uint32_t participant_count) noexcept
    {
        uint32_t count = 0;
        for (uint32_t participant = 0; participant < participant_count; ++participant)
        {
            if ((resident_mask & participantBit(participant)) != 0u)
                ++count;
        }
        return count;
    }

    LLAMINAR_MOE_REBALANCE_HD uint32_t payloadBucketSlots(
        uint32_t requested_slots,
        uint32_t max_slot_capacity) noexcept
    {
        if (requested_slots == 0 || max_slot_capacity == 0)
            return 0;

        uint32_t bucket = 1u;
        while (bucket < requested_slots && bucket < max_slot_capacity)
        {
            const uint32_t next = bucket << 1u;
            if (next <= bucket)
                break;
            bucket = next;
        }
        return bucket > max_slot_capacity ? max_slot_capacity : bucket;
    }

    LLAMINAR_MOE_REBALANCE_HD uint32_t payloadBucketIndex(
        uint32_t bucket_slots) noexcept
    {
        uint32_t index = 0;
        while (bucket_slots > 1u)
        {
            bucket_slots >>= 1u;
            ++index;
        }
        return index;
    }

    LLAMINAR_MOE_REBALANCE_HD bool transferWaveMeetsSpreadImprovementFloor(
        uint64_t accepted_spread_improvement,
        uint32_t requested_payload_slots,
        uint32_t min_spread_improvement_per_payload_slot) noexcept
    {
        if (requested_payload_slots == 0u ||
            min_spread_improvement_per_payload_slot == 0u)
        {
            return true;
        }
        return accepted_spread_improvement >=
               static_cast<uint64_t>(requested_payload_slots) *
                   static_cast<uint64_t>(min_spread_improvement_per_payload_slot);
    }

    LLAMINAR_MOE_REBALANCE_HD bool transferWaveMeetsRealizedRouterBenefitFloor(
        uint64_t realized_router_spread_improvement,
        uint32_t requested_payload_slots,
        uint32_t min_router_spread_improvement_per_payload_slot,
        bool existing_hot_cache_active) noexcept
    {
        if (!existing_hot_cache_active ||
            requested_payload_slots == 0u ||
            min_router_spread_improvement_per_payload_slot == 0u)
        {
            return true;
        }
        return realized_router_spread_improvement >=
               static_cast<uint64_t>(requested_payload_slots) *
                   static_cast<uint64_t>(min_router_spread_improvement_per_payload_slot);
    }

    LLAMINAR_MOE_REBALANCE_HD uint32_t residentOrdinal(
        uint32_t resident_mask,
        uint32_t participant_count,
        uint32_t target_participant) noexcept
    {
        uint32_t ordinal = 0;
        for (uint32_t participant = 0; participant < participant_count; ++participant)
        {
            if ((resident_mask & participantBit(participant)) == 0u)
                continue;
            if (participant == target_participant)
                return ordinal;
            ++ordinal;
        }
        return participant_count;
    }

    LLAMINAR_MOE_REBALANCE_HD uint64_t projectedParticipantLoadForExpert(
        uint64_t expert_count,
        uint32_t resident_mask,
        uint32_t participant_count,
        uint32_t participant) noexcept
    {
        if (expert_count == 0 ||
            participant >= participant_count ||
            (resident_mask & participantBit(participant)) == 0u)
        {
            return 0;
        }
        const uint32_t count = residentCount(resident_mask, participant_count);
        if (count == 0)
            return 0;
        const uint64_t base = expert_count / static_cast<uint64_t>(count);
        const uint64_t remainder = expert_count % static_cast<uint64_t>(count);
        const uint32_t ordinal = residentOrdinal(resident_mask, participant_count, participant);
        return base + (ordinal < remainder ? 1u : 0u);
    }

    LLAMINAR_MOE_REBALANCE_HD void finalizeLoadSpread(
        const uint64_t *participant_load,
        uint32_t participant_count,
        uint64_t &total,
        uint64_t &min_load,
        uint64_t &max_load) noexcept
    {
        total = 0;
        min_load = UINT64_MAX;
        max_load = 0;
        for (uint32_t participant = 0; participant < participant_count; ++participant)
        {
            const uint64_t load = participant_load[participant];
            total += load;
            if (load < min_load)
                min_load = load;
            if (load > max_load)
                max_load = load;
        }
        if (participant_count == 0 || min_load == UINT64_MAX)
            min_load = 0;
    }

    LLAMINAR_MOE_REBALANCE_HD uint64_t requiredLoadSpreadImprovement(
        uint64_t current_total,
        uint32_t min_improvement,
        uint32_t min_improvement_divisor) noexcept
    {
        uint64_t required_improvement = static_cast<uint64_t>(min_improvement);
        if (min_improvement_divisor > 0)
        {
            const uint64_t relative_floor =
                current_total / static_cast<uint64_t>(min_improvement_divisor);
            if (relative_floor > required_improvement)
                required_improvement = relative_floor;
        }
        return required_improvement;
    }

    LLAMINAR_MOE_REBALANCE_HD bool expertCountCanMeetLoadSpreadFloor(
        uint64_t expert_count,
        uint64_t current_total,
        uint32_t min_improvement,
        uint32_t min_improvement_divisor) noexcept
    {
        const uint64_t required_improvement = requiredLoadSpreadImprovement(
            current_total,
            min_improvement,
            min_improvement_divisor);
        return required_improvement == 0 || expert_count >= required_improvement;
    }

    LLAMINAR_MOE_REBALANCE_HD bool candidateValueIsBetter(
        uint64_t candidate_value,
        uint64_t candidate_count,
        uint32_t candidate_expert,
        uint64_t best_value,
        uint64_t best_count,
        uint32_t best_expert) noexcept
    {
        if (candidate_value == 0)
            return false;
        if (best_value == 0)
            return true;
        if (candidate_value != best_value)
            return candidate_value > best_value;
        if (candidate_count != best_count)
            return candidate_count > best_count;
        return candidate_expert < best_expert;
    }

    LLAMINAR_MOE_REBALANCE_HD bool addingResidentCanMeetLoadSpreadFloorByCount(
        const uint64_t *current_participant_load,
        uint64_t expert_count,
        uint32_t current_resident_mask,
        uint32_t proposed_resident_mask,
        uint32_t participant_count,
        uint32_t min_improvement,
        uint32_t min_improvement_divisor) noexcept
    {
        if (!current_participant_load ||
            expert_count == 0 ||
            participant_count == 0 ||
            current_resident_mask == proposed_resident_mask)
        {
            return false;
        }

        uint64_t current_total = 0;
        uint64_t current_min = 0;
        uint64_t current_max = 0;
        finalizeLoadSpread(
            current_participant_load,
            participant_count,
            current_total,
            current_min,
            current_max);
        (void)current_min;
        (void)current_max;

        return expertCountCanMeetLoadSpreadFloor(
            expert_count,
            current_total,
            min_improvement,
            min_improvement_divisor);
    }

    LLAMINAR_MOE_REBALANCE_HD LoadSpreadDelta evaluateAddingResidentLoadSpread(
        const uint64_t *current_participant_load,
        uint64_t expert_count,
        uint32_t current_resident_mask,
        uint32_t proposed_resident_mask,
        uint32_t participant_count,
        uint32_t min_improvement = 0,
        uint32_t min_improvement_divisor = 0) noexcept
    {
        LoadSpreadDelta delta{};
        if (!current_participant_load ||
            expert_count == 0 ||
            participant_count == 0 ||
            current_resident_mask == proposed_resident_mask)
        {
            return delta;
        }

        finalizeLoadSpread(
            current_participant_load,
            participant_count,
            delta.current_total,
            delta.current_min,
            delta.current_max);
        delta.current_spread = delta.current_max - delta.current_min;
        delta.required_improvement = requiredLoadSpreadImprovement(
            delta.current_total,
            min_improvement,
            min_improvement_divisor);

        delta.proposed_total = 0;
        delta.proposed_min = UINT64_MAX;
        delta.proposed_max = 0;
        bool underflow = false;
        for (uint32_t participant = 0; participant < participant_count; ++participant)
        {
            const uint64_t old_contribution =
                projectedParticipantLoadForExpert(
                    expert_count,
                    current_resident_mask,
                    participant_count,
                    participant);
            const uint64_t new_contribution =
                projectedParticipantLoadForExpert(
                    expert_count,
                    proposed_resident_mask,
                    participant_count,
                    participant);
            if (current_participant_load[participant] < old_contribution)
                underflow = true;
            const uint64_t proposed_load =
                underflow
                    ? 0
                    : current_participant_load[participant] - old_contribution + new_contribution;
            delta.proposed_total += proposed_load;
            if (proposed_load < delta.proposed_min)
                delta.proposed_min = proposed_load;
            if (proposed_load > delta.proposed_max)
                delta.proposed_max = proposed_load;
        }
        if (participant_count == 0 || delta.proposed_min == UINT64_MAX)
            delta.proposed_min = 0;

        delta.total_preserved = !underflow && delta.proposed_total == delta.current_total;
        delta.proposed_spread = delta.proposed_max - delta.proposed_min;
        delta.improves =
            delta.total_preserved && delta.proposed_spread < delta.current_spread;
        delta.improvement =
            delta.improves ? (delta.current_spread - delta.proposed_spread) : 0;
        delta.meets_floor =
            delta.improves && delta.improvement >= delta.required_improvement;
        return delta;
    }

    LLAMINAR_MOE_REBALANCE_HD bool addingResidentImprovesLoadSpread(
        const uint64_t *current_participant_load,
        uint64_t expert_count,
        uint32_t current_resident_mask,
        uint32_t proposed_resident_mask,
        uint32_t participant_count,
        uint64_t *scratch_participant_load,
        uint32_t min_improvement = 0,
        uint32_t min_improvement_divisor = 0) noexcept
    {
        if (!current_participant_load ||
            !scratch_participant_load ||
            expert_count == 0 ||
            participant_count == 0 ||
            current_resident_mask == proposed_resident_mask)
        {
            return false;
        }

        const LoadSpreadDelta delta = evaluateAddingResidentLoadSpread(
            current_participant_load,
            expert_count,
            current_resident_mask,
            proposed_resident_mask,
            participant_count,
            min_improvement,
            min_improvement_divisor);
        if (!delta.meets_floor)
            return false;

        for (uint32_t participant = 0; participant < participant_count; ++participant)
        {
            const uint64_t old_contribution =
                projectedParticipantLoadForExpert(
                    expert_count,
                    current_resident_mask,
                    participant_count,
                    participant);
            const uint64_t new_contribution =
                projectedParticipantLoadForExpert(
                    expert_count,
                    proposed_resident_mask,
                    participant_count,
                    participant);
            scratch_participant_load[participant] =
                current_participant_load[participant] - old_contribution + new_contribution;
        }
        return true;
    }

    LLAMINAR_MOE_REBALANCE_HD bool destinationChoiceIsBetter(
        const DestinationChoice &candidate,
        const DestinationChoice &best,
        const uint64_t *current_participant_load) noexcept
    {
        if (!candidate.valid)
            return false;
        if (!best.valid)
            return true;
        if (candidate.delta.improvement != best.delta.improvement)
            return candidate.delta.improvement > best.delta.improvement;
        if (current_participant_load)
        {
            const uint64_t candidate_load =
                current_participant_load[candidate.destination_participant];
            const uint64_t best_load =
                current_participant_load[best.destination_participant];
            if (candidate_load != best_load)
                return candidate_load < best_load;
        }
        return candidate.destination_participant < best.destination_participant;
    }

    LLAMINAR_MOE_REBALANCE_HD DestinationChoice bestMissingResidentDestination(
        const uint64_t *current_participant_load,
        uint64_t expert_count,
        uint32_t current_resident_mask,
        uint32_t participant_count,
        const uint32_t *planned_replicas_per_participant = nullptr,
        uint32_t max_replicas_per_participant = 0,
        uint32_t min_improvement = 0,
        uint32_t min_improvement_divisor = 0) noexcept
    {
        DestinationChoice best{};
        if (!current_participant_load ||
            expert_count == 0 ||
            participant_count == 0)
        {
            return best;
        }

        const uint32_t valid_mask = validParticipantMask(participant_count);
        current_resident_mask &= valid_mask;
        if (current_resident_mask == valid_mask)
            return best;

        for (uint32_t participant = 0; participant < participant_count; ++participant)
        {
            const uint32_t bit = participantBit(participant);
            if ((current_resident_mask & bit) != 0u)
                continue;
            if (planned_replicas_per_participant &&
                max_replicas_per_participant > 0u &&
                planned_replicas_per_participant[participant] >= max_replicas_per_participant)
            {
                continue;
            }

            DestinationChoice candidate{};
            candidate.destination_participant = participant;
            candidate.proposed_resident_mask = (current_resident_mask | bit) & valid_mask;
            candidate.delta = evaluateAddingResidentLoadSpread(
                current_participant_load,
                expert_count,
                current_resident_mask,
                candidate.proposed_resident_mask,
                participant_count,
                min_improvement,
                min_improvement_divisor);
            candidate.valid = candidate.delta.meets_floor;
            if (destinationChoiceIsBetter(candidate, best, current_participant_load))
                best = candidate;
        }

        return best;
    }

    LLAMINAR_MOE_REBALANCE_HD bool commandBufferFull(
        uint32_t command_count,
        uint32_t command_capacity) noexcept
    {
        return command_count >= command_capacity;
    }

    template <typename RebalanceConfig>
    LLAMINAR_MOE_REBALANCE_HD bool hasValidRootParticipant(
        const RebalanceConfig &config) noexcept
    {
        return config.root_participant < config.participant_count;
    }

    template <typename RebalanceConfig>
    LLAMINAR_MOE_REBALANCE_HD bool isRootParticipant(
        const RebalanceConfig &config) noexcept
    {
        return config.participant_id == config.root_participant;
    }

    LLAMINAR_MOE_REBALANCE_HD int32_t firstResidentParticipant(
        uint32_t resident_mask,
        uint32_t participant_count,
        int32_t preferred_participant = -1,
        int32_t excluded_participant = -1) noexcept
    {
        if (preferred_participant >= 0 &&
            preferred_participant < static_cast<int32_t>(participant_count) &&
            preferred_participant != excluded_participant &&
            (resident_mask & participantBit(static_cast<uint32_t>(preferred_participant))) != 0u)
        {
            return preferred_participant;
        }

        for (uint32_t participant = 0; participant < participant_count; ++participant)
        {
            if (static_cast<int32_t>(participant) == excluded_participant)
                continue;
            if ((resident_mask & participantBit(participant)) != 0u)
                return static_cast<int32_t>(participant);
        }
        return -1;
    }

    template <typename ExpertDescriptor, typename RebalanceConfig>
    LLAMINAR_MOE_REBALANCE_HD bool candidateCanAffectLocalCompute(
        const ExpertDescriptor &desc,
        uint32_t resident_mask,
        const RebalanceConfig &config,
        bool plan_missing_arrivals) noexcept
    {
        const uint32_t participant_bit = participantBit(config.participant_id);
        const bool local_resident = (resident_mask & participant_bit) != 0u;
        const bool owner_local =
            desc.owner_participant == static_cast<int32_t>(config.participant_id);
        const bool replicated = (resident_mask & (resident_mask - 1u)) != 0u;
        if (replicated && local_resident)
            return true;
        if (!plan_missing_arrivals || local_resident || owner_local)
            return false;
        return firstResidentParticipant(
                   resident_mask,
                   config.participant_count,
                   desc.owner_participant,
                   static_cast<int32_t>(config.participant_id)) >= 0;
    }

    template <typename ExpertDescriptor, typename RebalanceConfig>
    LLAMINAR_MOE_REBALANCE_HD bool candidateCanAffectDomainCompute(
        const ExpertDescriptor &desc,
        uint32_t resident_mask,
        const RebalanceConfig &config,
        bool plan_missing_arrivals) noexcept
    {
        const uint32_t valid_mask = validParticipantMask(config.participant_count);
        resident_mask &= valid_mask;
        if (desc.owner_participant >= 0 &&
            desc.owner_participant < static_cast<int32_t>(config.participant_count))
        {
            resident_mask |= participantBit(static_cast<uint32_t>(desc.owner_participant));
        }
        const bool replicated = (resident_mask & (resident_mask - 1u)) != 0u;
        if (replicated)
            return true;
        if (!plan_missing_arrivals || resident_mask == 0u || resident_mask == valid_mask)
            return false;
        return firstResidentParticipant(
                   resident_mask,
                   config.participant_count,
                   desc.owner_participant,
                   -1) >= 0;
    }
} // namespace llaminar2::moe_rebalance_policy

#undef LLAMINAR_MOE_REBALANCE_HD
