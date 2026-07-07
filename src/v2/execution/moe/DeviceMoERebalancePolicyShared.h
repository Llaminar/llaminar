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

#if defined(__CUDACC__)
#define LLAMINAR_MOE_REBALANCE_HD __host__ __device__ __forceinline__
#elif defined(__HIPCC__)
#define LLAMINAR_MOE_REBALANCE_HD __host__ __device__ inline __attribute__((always_inline))
#else
#define LLAMINAR_MOE_REBALANCE_HD inline
#endif

namespace llaminar2::moe_rebalance_policy
{
    constexpr uint32_t kPlanExpertPayloadArrival = 1u;
    constexpr uint32_t kPlanResidentExpertAssignment = 2u;
    constexpr uint32_t kPlanOwnershipTransfer = 3u;
    constexpr uint32_t kMaxPolicyParticipants = 8u;
    constexpr uint32_t kDefaultDynamicImbalanceThresholdPerMille = 1300u;
    constexpr uint32_t kDefaultDynamicMinImprovementPerMille = 50u;
    constexpr uint32_t kDefaultDynamicMaxSwapsPerLayer = 4u;
    constexpr uint32_t kDefaultDynamicMaxPlanEntriesPerWave = 16u;
    constexpr uint64_t kDefaultDynamicMinWindowActivations = 64u;
    constexpr uint32_t kDefaultDeviceMinLoadSpreadImprovementDivisor = 15u;
    constexpr float kDefaultDynamicImbalanceThresholdRatio =
        static_cast<float>(kDefaultDynamicImbalanceThresholdPerMille) / 1000.0f;
    constexpr float kDefaultDynamicMinImprovementRatio =
        static_cast<float>(kDefaultDynamicMinImprovementPerMille) / 1000.0f;

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
        uint32_t source_participant = 0;
        uint64_t projected_shift = 0;
        LoadSpreadDelta delta{};
        bool valid = false;
    };

    struct OwnershipSwapChoice
    {
        uint32_t overloaded_participant = 0;
        uint32_t underloaded_participant = 0;
        uint32_t heavy_expert = 0;
        uint32_t light_expert = 0;
        uint64_t heavy_count = 0;
        uint64_t light_count = 0;
        uint64_t old_min_load = 0;
        uint64_t old_max_load = 0;
        uint64_t new_min_load = 0;
        uint64_t new_max_load = 0;
        uint64_t improvement = 0;
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

    LLAMINAR_MOE_REBALANCE_HD bool ratioAtLeastPerMille(
        uint64_t numerator,
        uint64_t denominator,
        uint32_t threshold_per_mille) noexcept
    {
        if (denominator == 0u)
            return numerator > 0u;
        if (threshold_per_mille == 0u)
            return true;
        if (numerator > UINT64_MAX / 1000ULL)
            return true;
        return numerator * 1000ULL >=
               denominator * static_cast<uint64_t>(threshold_per_mille);
    }

    LLAMINAR_MOE_REBALANCE_HD bool finiteRatioImprovesByPerMille(
        uint64_t old_max,
        uint64_t old_min,
        uint64_t new_max,
        uint64_t new_min,
        uint32_t min_improvement_per_mille) noexcept
    {
        if (old_min == 0u || new_min == 0u || old_max == 0u)
            return false;
        if (min_improvement_per_mille == 0u)
            return new_max * old_min < old_max * new_min;

        const uint64_t keep_per_mille =
            min_improvement_per_mille >= 1000u
                ? 0ULL
                : 1000ULL - static_cast<uint64_t>(min_improvement_per_mille);
        if (new_max != 0u && old_min > UINT64_MAX / new_max)
            return false;
        const uint64_t lhs0 = new_max * old_min;
        if (lhs0 > UINT64_MAX / 1000ULL)
            return false;
        const uint64_t lhs = lhs0 * 1000ULL;

        if (old_max != 0u && new_min > UINT64_MAX / old_max)
            return true;
        const uint64_t rhs0 = old_max * new_min;
        if (keep_per_mille != 0u && rhs0 > UINT64_MAX / keep_per_mille)
            return true;
        const uint64_t rhs = rhs0 * keep_per_mille;
        return lhs <= rhs;
    }

    LLAMINAR_MOE_REBALANCE_HD OwnershipSwapChoice bestDynamicOwnershipSwap(
        const uint64_t *participant_load,
        const uint64_t *expert_counts,
        const int32_t *expert_owner,
        uint32_t num_experts,
        uint32_t participant_count,
        uint32_t imbalance_threshold_per_mille = kDefaultDynamicImbalanceThresholdPerMille,
        uint32_t min_improvement_per_mille = kDefaultDynamicMinImprovementPerMille,
        uint64_t min_window_activations = kDefaultDynamicMinWindowActivations) noexcept
    {
        OwnershipSwapChoice choice{};
        if (!participant_load ||
            !expert_counts ||
            !expert_owner ||
            num_experts == 0u ||
            participant_count < 2u ||
            participant_count > kMaxPolicyParticipants)
        {
            return choice;
        }

        uint64_t total = 0;
        uint32_t overloaded = 0;
        uint32_t underloaded = 0;
        uint64_t max_load = 0;
        uint64_t min_load = UINT64_MAX;
        for (uint32_t participant = 0; participant < participant_count; ++participant)
        {
            const uint64_t load = participant_load[participant];
            total += load;
            if (load > max_load)
            {
                max_load = load;
                overloaded = participant;
            }
            if (load < min_load)
            {
                min_load = load;
                underloaded = participant;
            }
        }
        if (total < min_window_activations ||
            max_load == 0u ||
            overloaded == underloaded)
        {
            return choice;
        }
        if (min_load != 0u &&
            !ratioAtLeastPerMille(max_load, min_load, imbalance_threshold_per_mille))
        {
            return choice;
        }

        uint32_t heavy_expert = num_experts;
        uint32_t light_expert = num_experts;
        uint64_t heavy_count = 0;
        uint64_t light_count = UINT64_MAX;
        for (uint32_t expert = 0; expert < num_experts; ++expert)
        {
            const int32_t owner = expert_owner[expert];
            const uint64_t count = expert_counts[expert];
            if (owner == static_cast<int32_t>(overloaded))
            {
                if (heavy_expert == num_experts ||
                    count > heavy_count ||
                    (count == heavy_count && expert < heavy_expert))
                {
                    heavy_expert = expert;
                    heavy_count = count;
                }
            }
            else if (owner == static_cast<int32_t>(underloaded))
            {
                if (light_expert == num_experts ||
                    count < light_count ||
                    (count == light_count && expert < light_expert))
                {
                    light_expert = expert;
                    light_count = count;
                }
            }
        }
        if (heavy_expert == num_experts || light_expert == num_experts)
            return choice;

        uint64_t new_loads[kMaxPolicyParticipants] = {};
        for (uint32_t participant = 0; participant < participant_count; ++participant)
            new_loads[participant] = participant_load[participant];
        new_loads[overloaded] = max_load - heavy_count + light_count;
        new_loads[underloaded] = min_load - light_count + heavy_count;

        uint64_t new_min = UINT64_MAX;
        uint64_t new_max = 0;
        for (uint32_t participant = 0; participant < participant_count; ++participant)
        {
            const uint64_t load = new_loads[participant];
            if (load < new_min)
                new_min = load;
            if (load > new_max)
                new_max = load;
        }
        if (new_min == UINT64_MAX)
            new_min = 0;

        bool accept = false;
        if (min_load > 0u)
        {
            accept = finiteRatioImprovesByPerMille(
                max_load,
                min_load,
                new_max,
                new_min,
                min_improvement_per_mille);
        }
        else if (new_min > 0u)
        {
            accept = true;
        }
        else
        {
            accept = new_max < max_load;
        }
        if (!accept)
            return choice;

        choice.overloaded_participant = overloaded;
        choice.underloaded_participant = underloaded;
        choice.heavy_expert = heavy_expert;
        choice.light_expert = light_expert;
        choice.heavy_count = heavy_count;
        choice.light_count = light_count;
        choice.old_min_load = min_load == UINT64_MAX ? 0u : min_load;
        choice.old_max_load = max_load;
        choice.new_min_load = new_min;
        choice.new_max_load = new_max;
        const uint64_t old_spread = max_load - choice.old_min_load;
        const uint64_t new_spread = new_max - new_min;
        choice.improvement = old_spread > new_spread ? old_spread - new_spread : 1u;
        choice.valid = true;
        return choice;
    }

    LLAMINAR_MOE_REBALANCE_HD bool applyDynamicOwnershipSwap(
        uint64_t *participant_load,
        int32_t *expert_owner,
        const OwnershipSwapChoice &choice) noexcept
    {
        if (!participant_load || !expert_owner || !choice.valid)
            return false;
        participant_load[choice.overloaded_participant] =
            choice.old_max_load - choice.heavy_count + choice.light_count;
        participant_load[choice.underloaded_participant] =
            choice.old_min_load - choice.light_count + choice.heavy_count;
        expert_owner[choice.heavy_expert] =
            static_cast<int32_t>(choice.underloaded_participant);
        expert_owner[choice.light_expert] =
            static_cast<int32_t>(choice.overloaded_participant);
        return true;
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

    LLAMINAR_MOE_REBALANCE_HD bool transferWaveMeetsPostLoadSpreadCeiling(
        uint64_t post_policy_load_spread,
        uint64_t post_policy_load_total,
        uint32_t requested_payload_slots,
        uint32_t max_post_load_spread_per_mille) noexcept
    {
        if (requested_payload_slots == 0u ||
            max_post_load_spread_per_mille == 0u ||
            post_policy_load_total == 0u)
        {
            return true;
        }
        if (post_policy_load_spread > UINT64_MAX / 1000ULL)
            return false;
        const uint64_t lhs = post_policy_load_spread * 1000ULL;
        const uint64_t rhs =
            post_policy_load_total *
            static_cast<uint64_t>(max_post_load_spread_per_mille);
        return lhs <= rhs;
    }

    LLAMINAR_MOE_REBALANCE_HD bool transferWaveImprovesAggregateLoadSpread(
        uint64_t pre_wave_load_spread,
        uint64_t post_wave_load_spread,
        uint64_t pre_wave_load_total,
        uint64_t post_wave_load_total,
        uint32_t requested_payload_slots) noexcept
    {
        if (requested_payload_slots == 0u)
            return true;
        if (pre_wave_load_total == 0u ||
            post_wave_load_total == 0u ||
            pre_wave_load_total != post_wave_load_total)
        {
            return false;
        }
        return post_wave_load_spread < pre_wave_load_spread;
    }

    LLAMINAR_MOE_REBALANCE_HD bool transferWaveParticipantSpreadIsAcceptable(
        uint64_t pre_policy_load_spread,
        uint64_t post_policy_load_spread,
        uint64_t pre_policy_load_total,
        uint64_t post_policy_load_total,
        uint32_t requested_payload_slots,
        bool realized_router_payback) noexcept
    {
        if (requested_payload_slots == 0u)
            return true;
        if (pre_policy_load_total == 0u ||
            post_policy_load_total == 0u ||
            pre_policy_load_total != post_policy_load_total)
        {
            return false;
        }
        return post_policy_load_spread < pre_policy_load_spread ||
               realized_router_payback;
    }

    template <typename PlanEntry>
    LLAMINAR_MOE_REBALANCE_HD uint32_t prunePayloadArrivalsPreservingResidentAssignments(
        PlanEntry *entries,
        uint32_t command_count,
        uint32_t *resident_command_count = nullptr) noexcept
    {
        uint32_t write_index = 0;
        uint32_t residents = 0;
        if (entries != nullptr)
        {
            for (uint32_t read_index = 0; read_index < command_count; ++read_index)
            {
                const PlanEntry entry = entries[read_index];
                if (entry.op != kPlanResidentExpertAssignment)
                    continue;
                entries[write_index++] = entry;
                ++residents;
            }
        }
        if (resident_command_count != nullptr)
            *resident_command_count = residents;
        return write_index;
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

    LLAMINAR_MOE_REBALANCE_HD bool addExpertLoadLeastLoaded(
        uint64_t *participant_load,
        uint64_t expert_count,
        uint32_t resident_mask,
        uint32_t participant_count) noexcept
    {
        if (!participant_load ||
            expert_count == 0 ||
            participant_count == 0 ||
            participant_count > kMaxPolicyParticipants)
        {
            return false;
        }

        resident_mask &= validParticipantMask(participant_count);
        if (resident_mask == 0u)
            return false;

        uint64_t remaining = expert_count;
        while (remaining > 0ULL)
        {
            uint64_t min_load = UINT64_MAX;
            uint64_t next_load = UINT64_MAX;
            uint32_t min_count = 0;
            for (uint32_t participant = 0; participant < participant_count; ++participant)
            {
                if ((resident_mask & participantBit(participant)) == 0u)
                    continue;
                const uint64_t load = participant_load[participant];
                if (load < min_load)
                {
                    next_load = min_load;
                    min_load = load;
                    min_count = 1u;
                }
                else if (load == min_load)
                {
                    ++min_count;
                }
                else if (load < next_load)
                {
                    next_load = load;
                }
            }

            if (min_count == 0u)
                return false;

            if (next_load != UINT64_MAX && next_load > min_load)
            {
                const uint64_t delta_to_next = next_load - min_load;
                if (delta_to_next <= UINT64_MAX / static_cast<uint64_t>(min_count))
                {
                    const uint64_t rows_to_next =
                        delta_to_next * static_cast<uint64_t>(min_count);
                    if (rows_to_next <= remaining)
                    {
                        for (uint32_t participant = 0; participant < participant_count; ++participant)
                        {
                            if ((resident_mask & participantBit(participant)) != 0u &&
                                participant_load[participant] == min_load)
                            {
                                participant_load[participant] += delta_to_next;
                            }
                        }
                        remaining -= rows_to_next;
                        continue;
                    }
                }
            }

            const uint64_t rows_per_min = remaining / static_cast<uint64_t>(min_count);
            uint64_t extra_rows = remaining % static_cast<uint64_t>(min_count);
            for (uint32_t participant = 0; participant < participant_count; ++participant)
            {
                if ((resident_mask & participantBit(participant)) == 0u ||
                    participant_load[participant] != min_load)
                {
                    continue;
                }
                participant_load[participant] +=
                    rows_per_min + (extra_rows > 0ULL ? 1ULL : 0ULL);
                if (extra_rows > 0ULL)
                    --extra_rows;
            }
            remaining = 0ULL;
        }

        return true;
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

    LLAMINAR_MOE_REBALANCE_HD LoadSpreadDelta evaluateAddingResidentLeastLoadedSpread(
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
            participant_count > kMaxPolicyParticipants ||
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

        uint64_t proposed_load[kMaxPolicyParticipants] = {};
        bool underflow = false;
        for (uint32_t participant = 0; participant < participant_count; ++participant)
        {
            const uint64_t old_contribution =
                projectedParticipantLoadForExpert(
                    expert_count,
                    current_resident_mask,
                    participant_count,
                    participant);
            if (current_participant_load[participant] < old_contribution)
                underflow = true;
            proposed_load[participant] =
                underflow ? 0ULL : current_participant_load[participant] - old_contribution;
        }

        if (underflow ||
            !addExpertLoadLeastLoaded(
                proposed_load,
                expert_count,
                proposed_resident_mask,
                participant_count))
        {
            return delta;
        }

        finalizeLoadSpread(
            proposed_load,
            participant_count,
            delta.proposed_total,
            delta.proposed_min,
            delta.proposed_max);
        delta.total_preserved = delta.proposed_total == delta.current_total;
        delta.proposed_spread = delta.proposed_max - delta.proposed_min;
        delta.improves =
            delta.total_preserved && delta.proposed_spread < delta.current_spread;
        delta.improvement =
            delta.improves ? (delta.current_spread - delta.proposed_spread) : 0;
        delta.meets_floor =
            delta.improves && delta.improvement >= delta.required_improvement;
        return delta;
    }

    LLAMINAR_MOE_REBALANCE_HD bool addingResidentImprovesLeastLoadedSpread(
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
            participant_count > kMaxPolicyParticipants ||
            current_resident_mask == proposed_resident_mask)
        {
            return false;
        }

        const LoadSpreadDelta delta = evaluateAddingResidentLeastLoadedSpread(
            current_participant_load,
            expert_count,
            current_resident_mask,
            proposed_resident_mask,
            participant_count,
            min_improvement,
            min_improvement_divisor);
        if (!delta.meets_floor)
            return false;

        bool underflow = false;
        for (uint32_t participant = 0; participant < participant_count; ++participant)
        {
            const uint64_t old_contribution =
                projectedParticipantLoadForExpert(
                    expert_count,
                    current_resident_mask,
                    participant_count,
                    participant);
            if (current_participant_load[participant] < old_contribution)
                underflow = true;
            scratch_participant_load[participant] =
                underflow ? 0ULL : current_participant_load[participant] - old_contribution;
        }
        if (underflow)
            return false;

        return addExpertLoadLeastLoaded(
            scratch_participant_load,
            expert_count,
            proposed_resident_mask,
            participant_count);
    }

    LLAMINAR_MOE_REBALANCE_HD uint64_t dynamicMinimumProjectedShift(
        uint32_t window_size_tokens,
        uint64_t current_total,
        uint32_t min_improvement = 0,
        uint32_t min_improvement_divisor = 0) noexcept
    {
        uint64_t floor = window_size_tokens > 0u
                             ? static_cast<uint64_t>(window_size_tokens / 16u)
                             : 0ULL;
        if (floor < 2ULL)
            floor = 2ULL;
        const uint64_t configured_floor = requiredLoadSpreadImprovement(
            current_total,
            min_improvement,
            min_improvement_divisor);
        return configured_floor > floor ? configured_floor : floor;
    }

    LLAMINAR_MOE_REBALANCE_HD LoadSpreadDelta evaluateAddingResidentDynamicSpread(
        const uint64_t *current_participant_load,
        uint64_t expert_count,
        uint32_t current_resident_mask,
        uint32_t proposed_resident_mask,
        uint32_t participant_count,
        uint32_t source_participant,
        uint32_t destination_participant,
        uint32_t window_size_tokens = 0,
        uint32_t min_improvement = 0,
        uint32_t min_improvement_divisor = 0) noexcept
    {
        LoadSpreadDelta delta{};
        if (!current_participant_load ||
            expert_count == 0 ||
            participant_count == 0 ||
            current_resident_mask == proposed_resident_mask ||
            source_participant >= participant_count ||
            destination_participant >= participant_count ||
            source_participant == destination_participant ||
            (current_resident_mask & participantBit(source_participant)) == 0u ||
            (proposed_resident_mask & participantBit(destination_participant)) == 0u)
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
        delta.required_improvement = dynamicMinimumProjectedShift(
            window_size_tokens,
            delta.current_total,
            min_improvement,
            min_improvement_divisor);

        const uint64_t source_load = current_participant_load[source_participant];
        const uint64_t destination_load = current_participant_load[destination_participant];
        if (source_load <= destination_load)
            return delta;

        const uint64_t equalizing_shift = (source_load - destination_load + 1ULL) / 2ULL;
        const uint64_t shift = equalizing_shift < expert_count ? equalizing_shift : expert_count;

        delta.proposed_total = 0;
        delta.proposed_min = UINT64_MAX;
        delta.proposed_max = 0;
        for (uint32_t participant = 0; participant < participant_count; ++participant)
        {
            uint64_t proposed_load = current_participant_load[participant];
            if (participant == source_participant)
            {
                proposed_load = proposed_load > shift ? proposed_load - shift : 0ULL;
            }
            else if (participant == destination_participant)
            {
                proposed_load += shift;
            }
            delta.proposed_total += proposed_load;
            if (proposed_load < delta.proposed_min)
                delta.proposed_min = proposed_load;
            if (proposed_load > delta.proposed_max)
                delta.proposed_max = proposed_load;
        }
        if (participant_count == 0 || delta.proposed_min == UINT64_MAX)
            delta.proposed_min = 0;

        delta.total_preserved = delta.proposed_total == delta.current_total;
        delta.proposed_spread = delta.proposed_max - delta.proposed_min;
        delta.improves =
            delta.total_preserved && delta.proposed_spread < delta.current_spread;
        delta.improvement = delta.improves ? shift : 0ULL;
        delta.meets_floor = delta.improves && shift >= delta.required_improvement;
        return delta;
    }

    LLAMINAR_MOE_REBALANCE_HD bool addingResidentImprovesDynamicSpread(
        const uint64_t *current_participant_load,
        uint64_t expert_count,
        uint32_t current_resident_mask,
        uint32_t proposed_resident_mask,
        uint32_t participant_count,
        uint32_t source_participant,
        uint32_t destination_participant,
        uint64_t *scratch_participant_load,
        uint32_t window_size_tokens = 0,
        uint32_t min_improvement = 0,
        uint32_t min_improvement_divisor = 0) noexcept
    {
        if (!current_participant_load || !scratch_participant_load)
            return false;

        const LoadSpreadDelta delta = evaluateAddingResidentDynamicSpread(
            current_participant_load,
            expert_count,
            current_resident_mask,
            proposed_resident_mask,
            participant_count,
            source_participant,
            destination_participant,
            window_size_tokens,
            min_improvement,
            min_improvement_divisor);
        if (!delta.meets_floor || delta.improvement == 0ULL)
            return false;

        for (uint32_t participant = 0; participant < participant_count; ++participant)
        {
            uint64_t proposed_load = current_participant_load[participant];
            if (participant == source_participant)
                proposed_load = proposed_load > delta.improvement ? proposed_load - delta.improvement : 0ULL;
            else if (participant == destination_participant)
                proposed_load += delta.improvement;
            scratch_participant_load[participant] = proposed_load;
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

	    LLAMINAR_MOE_REBALANCE_HD DestinationChoice bestLeastLoadedMissingResidentDestination(
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
            participant_count == 0 ||
            participant_count > kMaxPolicyParticipants)
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
            candidate.delta = evaluateAddingResidentLeastLoadedSpread(
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

    LLAMINAR_MOE_REBALANCE_HD int32_t firstResidentParticipant(
        uint32_t resident_mask,
        uint32_t participant_count,
        int32_t preferred_participant,
        int32_t excluded_participant) noexcept;

    LLAMINAR_MOE_REBALANCE_HD DestinationChoice bestDynamicMissingResidentDestination(
        const uint64_t *current_participant_load,
        uint64_t expert_count,
        uint32_t current_resident_mask,
        uint32_t participant_count,
        int32_t preferred_source_participant = -1,
        const uint32_t *planned_replicas_per_participant = nullptr,
        uint32_t max_replicas_per_participant = 0,
        uint32_t window_size_tokens = 0,
        uint32_t min_improvement = 0,
        uint32_t min_improvement_divisor = 0) noexcept
    {
        DestinationChoice best{};
        if (!current_participant_load ||
            expert_count == 0 ||
            participant_count == 0 ||
            participant_count > kMaxPolicyParticipants)
        {
            return best;
        }

        const uint32_t valid_mask = validParticipantMask(participant_count);
        current_resident_mask &= valid_mask;
        if (current_resident_mask == 0u || current_resident_mask == valid_mask)
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

            const int32_t source = firstResidentParticipant(
                current_resident_mask,
                participant_count,
                preferred_source_participant,
                static_cast<int32_t>(participant));
            if (source < 0)
                continue;

            DestinationChoice candidate{};
            candidate.destination_participant = participant;
            candidate.source_participant = static_cast<uint32_t>(source);
            candidate.proposed_resident_mask = (current_resident_mask | bit) & valid_mask;
            candidate.delta = evaluateAddingResidentDynamicSpread(
                current_participant_load,
                expert_count,
                current_resident_mask,
                candidate.proposed_resident_mask,
                participant_count,
                candidate.source_participant,
                candidate.destination_participant,
                window_size_tokens,
                min_improvement,
                min_improvement_divisor);
            candidate.projected_shift = candidate.delta.improvement;
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

        /*
         * Domain-root planning is allowed to prefer the static owner when a
         * resident source must be selected, but it must not convert ownership
         * into residency.  The GPU payload packer can only copy bytes from a
         * participant whose bit is already present in the runtime resident mask.
         */
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
