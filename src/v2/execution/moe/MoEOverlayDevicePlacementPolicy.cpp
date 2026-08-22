/**
 * @file MoEOverlayDevicePlacementPolicy.cpp
 * @brief Deterministic CPU oracle for device-owned ExpertOverlay policy.
 */

#include "MoEOverlayDevicePlacementPolicy.h"

#include "DeviceMoERebalancePolicyShared.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace llaminar2
{
    namespace
    {
        /** Lexicographic tier-local service objective for one layer. */
        struct PlacementScore
        {
            std::uint64_t priority_cost = 0u;
            std::uint64_t same_priority_makespan = 0u;
        };

        /** Measured payoff and committed-hysteresis verdict for one cycle. */
        struct CycleEconomyScore
        {
            std::uint64_t service_before_ns = 0u;
            std::uint64_t service_after_ns = 0u;
            std::uint64_t projected_service_gain_ns = 0u;
            std::uint64_t transfer_and_repack_ns = 0u;
            std::uint64_t inference_interference_ns = 0u;
            std::uint64_t projected_net_benefit_ns = 0u;
            bool residency_eligible = true;
            bool payoff_eligible = false;

            /** @return Whether both independent measured gates accept. */
            [[nodiscard]] bool eligible() const noexcept
            {
                return residency_eligible && payoff_eligible;
            }
        };

        /** Saturating addition keeps adversarial counter tests defined. */
        std::uint64_t saturatingAdd(
            std::uint64_t lhs,
            std::uint64_t rhs) noexcept
        {
            return rhs > std::numeric_limits<std::uint64_t>::max() - lhs
                       ? std::numeric_limits<std::uint64_t>::max()
                       : lhs + rhs;
        }

        /** Saturating multiplication for priority-rank objective terms. */
        std::uint64_t saturatingMultiply(
            std::uint64_t lhs,
            std::uint64_t rhs) noexcept
        {
            if (lhs == 0u || rhs == 0u)
                return 0u;
            return lhs > std::numeric_limits<std::uint64_t>::max() / rhs
                       ? std::numeric_limits<std::uint64_t>::max()
                       : lhs * rhs;
        }

        /** Flatten one measured phase/layer/expert history coordinate. */
        std::size_t demandOffset(
            const MoEOverlayDevicePlacementPolicyInput &input,
            std::uint32_t phase,
            std::uint32_t layer,
            std::uint32_t expert) noexcept
        {
            return (static_cast<std::size_t>(phase) * input.num_layers +
                    layer) *
                       input.num_experts +
                   expert;
        }

        /** Flatten one tier/layer/production-phase service coordinate. */
        std::size_t serviceOffset(
            const MoEOverlayDevicePlacementPolicyInput &input,
            std::uint32_t tier,
            std::uint32_t layer,
            std::uint32_t phase) noexcept
        {
            return (static_cast<std::size_t>(tier) * input.num_layers +
                    layer) *
                       kMoEOverlayDeviceControllerEconomyServicePhaseCount +
                   phase;
        }

        /** Flatten one directed participant-pair/layer movement coordinate. */
        std::size_t migrationOffset(
            const MoEOverlayDevicePlacementPolicyInput &input,
            std::uint32_t source,
            std::uint32_t destination,
            std::uint32_t layer) noexcept
        {
            return (static_cast<std::size_t>(source) *
                        input.participants.size() +
                    destination) *
                       input.num_layers +
                   layer;
        }

        /** Conservative floor of gain scaled from routed activations to tokens. */
        std::uint64_t projectServiceGain(
            std::uint64_t gain,
            std::uint64_t payoff_horizon_tokens,
            std::uint32_t routed_experts_per_token,
            std::uint64_t observed_activations) noexcept
        {
            if (gain == 0u || payoff_horizon_tokens == 0u ||
                routed_experts_per_token == 0u ||
                observed_activations == 0u)
            {
                return 0u;
            }
            const std::uint64_t scale = saturatingMultiply(
                payoff_horizon_tokens, routed_experts_per_token);
            return saturatingAdd(
                saturatingMultiply(gain / observed_activations, scale),
                saturatingMultiply(gain % observed_activations, scale) /
                    observed_activations);
        }

        /** Flatten one participant/layer/expert coordinate. */
        std::size_t stateOffset(
            const MoEOverlayDevicePlacementPolicyInput &input,
            std::uint32_t participant,
            std::uint32_t layer,
            std::uint32_t expert) noexcept
        {
            return (static_cast<std::size_t>(participant) * input.num_layers +
                    layer) *
                       input.num_experts +
                   expert;
        }

        /** Flatten one layer/expert owner coordinate. */
        std::size_t ownerOffset(
            const MoEOverlayDevicePlacementPolicyInput &input,
            std::uint32_t layer,
            std::uint32_t expert) noexcept
        {
            return static_cast<std::size_t>(layer) * input.num_experts + expert;
        }

        /** Sorted unique integer priorities, independent of tier names/vendors. */
        std::vector<std::int32_t> priorityOrder(
            const MoEOverlayDevicePlacementPolicyInput &input)
        {
            std::vector<std::int32_t> priorities;
            priorities.reserve(input.participants.size());
            for (const auto &participant : input.participants)
                priorities.push_back(participant.tier_priority);
            std::sort(priorities.begin(), priorities.end());
            priorities.erase(
                std::unique(priorities.begin(), priorities.end()),
                priorities.end());
            return priorities;
        }

        /** Dense rank of one opaque integer priority. */
        std::size_t priorityRank(
            const std::vector<std::int32_t> &priorities,
            std::int32_t priority)
        {
            const auto found = std::lower_bound(
                priorities.begin(), priorities.end(), priority);
            if (found == priorities.end() || *found != priority)
                throw std::logic_error("ExpertOverlay policy lost a participant priority");
            return static_cast<std::size_t>(found - priorities.begin());
        }

        /** Score one layer owner vector against exact routed counts. */
        PlacementScore scoreLayer(
            const MoEOverlayDevicePlacementPolicyInput &input,
            const std::vector<std::int32_t> &owners,
            const std::vector<std::uint64_t> &expert_counts,
            const std::vector<std::int32_t> &priorities)
        {
            PlacementScore score;
            std::vector<std::uint64_t> participant_load(
                input.participants.size(), 0u);
            for (std::uint32_t expert = 0u;
                 expert < input.num_experts;
                 ++expert)
            {
                const std::int32_t owner = owners[expert];
                if (owner < 0 ||
                    static_cast<std::size_t>(owner) >=
                        input.participants.size())
                {
                    throw std::logic_error(
                        "ExpertOverlay policy score received an invalid owner");
                }
                const std::uint64_t count = expert_counts[expert];
                participant_load[static_cast<std::size_t>(owner)] =
                    saturatingAdd(
                        participant_load[static_cast<std::size_t>(owner)],
                        count);
                score.priority_cost = saturatingAdd(
                    score.priority_cost,
                    saturatingMultiply(
                        count,
                        priorityRank(
                            priorities,
                            input.participants[static_cast<std::size_t>(owner)]
                                .tier_priority)));
            }

            for (const std::int32_t priority : priorities)
            {
                std::uint64_t maximum = 0u;
                for (std::size_t participant = 0u;
                     participant < input.participants.size();
                     ++participant)
                {
                    if (input.participants[participant].tier_priority ==
                        priority)
                    {
                        maximum = std::max(
                            maximum, participant_load[participant]);
                    }
                }
                score.same_priority_makespan = saturatingAdd(
                    score.same_priority_makespan, maximum);
            }
            return score;
        }

        /** @return Whether one cycle stays inside one materially skewed priority. */
        bool samePriorityCycleIsEligible(
            const MoEOverlayDevicePlacementPolicyInput &input,
            const std::vector<std::int32_t> &current,
            const std::vector<std::int32_t> &desired,
            const std::vector<std::uint64_t> &expert_counts,
            const std::vector<std::uint32_t> &cycle) noexcept
        {
            if (cycle.empty())
                return false;

            std::int32_t cycle_priority = 0;
            bool selected_priority = false;
            for (const std::uint32_t expert : cycle)
            {
                const auto source = static_cast<std::size_t>(current[expert]);
                const auto destination =
                    static_cast<std::size_t>(desired[expert]);
                const std::int32_t source_priority =
                    input.participants[source].tier_priority;
                const std::int32_t destination_priority =
                    input.participants[destination].tier_priority;
                if (source_priority != destination_priority)
                    return false;
                if (!selected_priority)
                {
                    cycle_priority = source_priority;
                    selected_priority = true;
                }
                else if (cycle_priority != source_priority)
                {
                    return false;
                }
            }

            std::vector<std::uint64_t> participant_load(
                input.participants.size(), 0u);
            std::vector<std::int32_t> participant_priority;
            participant_priority.reserve(input.participants.size());
            for (const auto &participant : input.participants)
                participant_priority.push_back(participant.tier_priority);
            for (std::uint32_t expert = 0u;
                 expert < input.num_experts;
                 ++expert)
            {
                const auto owner = static_cast<std::size_t>(current[expert]);
                participant_load[owner] = saturatingAdd(
                    participant_load[owner], expert_counts[expert]);
            }
            return moe_rebalance_policy::samePriorityLoadIsImbalanced(
                participant_load.data(),
                participant_priority.data(),
                static_cast<std::uint32_t>(input.participants.size()),
                cycle_priority,
                input.dynamic_imbalance_threshold_per_mille);
        }

        /** Price one cycle identically to the CUDA/HIP policy implementation. */
        CycleEconomyScore scoreEconomyCycle(
            const MoEOverlayDevicePlacementPolicyInput &input,
            const std::vector<std::int32_t> &current,
            const std::vector<std::int32_t> &desired,
            const std::vector<std::int32_t> &candidate,
            const std::vector<std::uint64_t> &combined_counts,
            const std::vector<std::uint32_t> &cycle,
            std::uint32_t layer)
        {
            const auto &economy = input.economy.value();
            CycleEconomyScore score;
            if (cycle.empty())
                return score;

            const auto first_source = static_cast<std::size_t>(
                current[cycle.front()]);
            const std::int32_t selected_tier =
                input.participants[first_source].tier_index;
            bool pure_same_tier = selected_tier >= 0;
            for (const std::uint32_t expert : cycle)
            {
                const auto source = static_cast<std::size_t>(current[expert]);
                const auto destination = static_cast<std::size_t>(
                    desired[expert]);
                pure_same_tier = pure_same_tier &&
                    input.participants[source].tier_index == selected_tier &&
                    input.participants[destination].tier_index ==
                        selected_tier;
            }

            if (pure_same_tier)
            {
                for (std::uint32_t phase = 0u;
                     phase < kMoEOverlayDeviceControllerDemandPhaseCount;
                     ++phase)
                {
                    std::vector<std::uint64_t> before_load(
                        input.participants.size(), 0u);
                    std::vector<std::uint64_t> after_load(
                        input.participants.size(), 0u);
                    for (std::uint32_t expert = 0u;
                         expert < input.num_experts;
                         ++expert)
                    {
                        const std::uint64_t demand =
                            economy.phase_expert_demand[demandOffset(
                                input, phase, layer, expert)];
                        const auto before_owner = static_cast<std::size_t>(
                            current[expert]);
                        const auto after_owner = static_cast<std::size_t>(
                            candidate[expert]);
                        if (input.participants[before_owner].tier_index ==
                            selected_tier)
                        {
                            before_load[before_owner] = saturatingAdd(
                                before_load[before_owner], demand);
                        }
                        if (input.participants[after_owner].tier_index ==
                            selected_tier)
                        {
                            after_load[after_owner] = saturatingAdd(
                                after_load[after_owner], demand);
                        }
                    }
                    std::uint64_t before_maximum = 0u;
                    std::uint64_t after_maximum = 0u;
                    for (std::size_t participant = 0u;
                         participant < input.participants.size();
                         ++participant)
                    {
                        if (input.participants[participant].tier_index ==
                            selected_tier)
                        {
                            before_maximum = std::max(
                                before_maximum, before_load[participant]);
                            after_maximum = std::max(
                                after_maximum, after_load[participant]);
                        }
                    }
                    const std::uint64_t service = economy.service_costs[
                        serviceOffset(
                            input,
                            static_cast<std::uint32_t>(selected_tier),
                            layer,
                            phase)];
                    score.service_before_ns = saturatingAdd(
                        score.service_before_ns,
                        saturatingMultiply(before_maximum, service));
                    score.service_after_ns = saturatingAdd(
                        score.service_after_ns,
                        saturatingMultiply(after_maximum, service));
                }
            }
            else
            {
                for (const std::uint32_t expert : cycle)
                {
                    const auto source = static_cast<std::size_t>(
                        current[expert]);
                    const auto destination = static_cast<std::size_t>(
                        desired[expert]);
                    const auto source_tier = static_cast<std::uint32_t>(
                        input.participants[source].tier_index);
                    const auto destination_tier = static_cast<std::uint32_t>(
                        input.participants[destination].tier_index);
                    for (std::uint32_t phase = 0u;
                         phase < kMoEOverlayDeviceControllerDemandPhaseCount;
                         ++phase)
                    {
                        const std::uint64_t demand =
                            economy.phase_expert_demand[demandOffset(
                                input, phase, layer, expert)];
                        score.service_before_ns = saturatingAdd(
                            score.service_before_ns,
                            saturatingMultiply(
                                demand,
                                economy.service_costs[serviceOffset(
                                    input,
                                    source_tier,
                                    layer,
                                    phase)]));
                        score.service_after_ns = saturatingAdd(
                            score.service_after_ns,
                            saturatingMultiply(
                                demand,
                                economy.service_costs[serviceOffset(
                                    input,
                                    destination_tier,
                                    layer,
                                    phase)]));
                    }
                }
            }

            const bool reciprocal_pair = cycle.size() == 2u &&
                current[cycle[0]] == desired[cycle[1]] &&
                current[cycle[1]] == desired[cycle[0]];
            for (const std::uint32_t expert : cycle)
            {
                const auto source = static_cast<std::uint32_t>(
                    current[expert]);
                const auto destination = static_cast<std::uint32_t>(
                    desired[expert]);
                const auto &movement = economy.migration_costs[
                    migrationOffset(
                        input, source, destination, layer)];
                if (reciprocal_pair)
                {
                    score.transfer_and_repack_ns = std::max(
                        score.transfer_and_repack_ns,
                        movement.transfer_and_repack_ns);
                    score.inference_interference_ns = std::max(
                        score.inference_interference_ns,
                        movement.inference_interference_ns);
                }
                else
                {
                    score.transfer_and_repack_ns = saturatingAdd(
                        score.transfer_and_repack_ns,
                        movement.transfer_and_repack_ns);
                    score.inference_interference_ns = saturatingAdd(
                        score.inference_interference_ns,
                        movement.inference_interference_ns);
                }
                const std::uint64_t last_moved =
                    economy.last_moved_generation[
                        ownerOffset(input, layer, expert)];
                if (last_moved !=
                        kMoEOverlayDeviceControllerNeverMovedGeneration &&
                    (economy.transaction_generation < last_moved ||
                     economy.transaction_generation - last_moved <
                         economy.minimum_residency_generations))
                {
                    score.residency_eligible = false;
                }
            }

            if (!moe_rebalance_policy::relativeReductionAtLeastPerMille(
                    score.service_before_ns,
                    score.service_after_ns,
                    input.dynamic_minimum_improvement_per_mille))
            {
                return score;
            }
            const std::uint64_t gain =
                score.service_before_ns - score.service_after_ns;
            std::uint64_t observed_activations = 0u;
            for (const std::uint64_t count : combined_counts)
            {
                observed_activations = saturatingAdd(
                    observed_activations, count);
            }
            score.projected_service_gain_ns = projectServiceGain(
                gain,
                economy.payoff_horizon_tokens,
                economy.routed_experts_per_token,
                observed_activations);
            const std::uint64_t measured_cost = saturatingAdd(
                score.transfer_and_repack_ns,
                score.inference_interference_ns);
            if (score.projected_service_gain_ns > measured_cost)
            {
                score.projected_net_benefit_ns =
                    score.projected_service_gain_ns - measured_cost;
            }
            score.payoff_eligible = score.projected_net_benefit_ns >
                economy.minimum_net_benefit_ns;
            return score;
        }

        /** @return Whether left outranks right under deterministic payoff. */
        bool economyScoreBetter(
            const CycleEconomyScore &left,
            const CycleEconomyScore &right) noexcept
        {
            return left.projected_net_benefit_ns !=
                           right.projected_net_benefit_ns
                       ? left.projected_net_benefit_ns >
                             right.projected_net_benefit_ns
                       : left.projected_service_gain_ns >
                             right.projected_service_gain_ns;
        }

        /**
         * Build the full hottest-first target while retaining every participant
         * quota. The target is planning scratch; only complete improving cycles
         * are later admitted into the candidate epoch.
         */
        std::vector<std::int32_t> desiredOwners(
            const MoEOverlayDevicePlacementPolicyInput &input,
            const std::vector<std::int32_t> &current,
            const std::vector<std::uint64_t> &expert_counts,
            const std::vector<std::int32_t> &priorities)
        {
            std::vector<std::uint32_t> quotas(
                input.participants.size(), 0u);
            for (const std::int32_t owner : current)
                ++quotas.at(static_cast<std::size_t>(owner));

            std::vector<std::uint32_t> ordered_experts(input.num_experts);
            std::iota(ordered_experts.begin(), ordered_experts.end(), 0u);
            std::sort(
                ordered_experts.begin(), ordered_experts.end(),
                [&](std::uint32_t lhs, std::uint32_t rhs)
                {
                    return expert_counts[lhs] != expert_counts[rhs]
                               ? expert_counts[lhs] > expert_counts[rhs]
                               : lhs < rhs;
                });

            std::vector<std::int32_t> desired(
                input.num_experts, -1);
            std::size_t expert_cursor = 0u;
            for (const std::int32_t priority : priorities)
            {
                std::vector<std::uint32_t> participants;
                std::uint32_t group_quota = 0u;
                for (std::uint32_t participant = 0u;
                     participant < input.participants.size();
                     ++participant)
                {
                    if (input.participants[participant].tier_priority ==
                        priority)
                    {
                        participants.push_back(participant);
                        group_quota += quotas[participant];
                    }
                }

                std::vector<std::uint32_t> remaining;
                std::vector<std::uint64_t> load;
                remaining.reserve(participants.size());
                load.assign(participants.size(), 0u);
                for (const std::uint32_t participant : participants)
                    remaining.push_back(quotas[participant]);

                for (std::uint32_t slot = 0u; slot < group_quota; ++slot)
                {
                    if (expert_cursor >= ordered_experts.size())
                    {
                        throw std::logic_error(
                            "ExpertOverlay policy priority quotas exceed the expert set");
                    }
                    const std::uint32_t expert =
                        ordered_experts[expert_cursor++];
                    std::size_t selected = participants.size();
                    for (std::size_t index = 0u;
                         index < participants.size();
                         ++index)
                    {
                        if (remaining[index] == 0u)
                            continue;
                        if (selected == participants.size() ||
                            load[index] < load[selected] ||
                            (load[index] == load[selected] &&
                             participants[index] < participants[selected]))
                        {
                            selected = index;
                        }
                    }
                    if (selected == participants.size())
                    {
                        throw std::logic_error(
                            "ExpertOverlay policy exhausted a priority quota early");
                    }
                    desired[expert] = static_cast<std::int32_t>(
                        participants[selected]);
                    --remaining[selected];
                    load[selected] = saturatingAdd(
                        load[selected], expert_counts[expert]);
                }
            }
            if (expert_cursor != ordered_experts.size() ||
                std::find(desired.begin(), desired.end(), -1) != desired.end())
            {
                throw std::logic_error(
                    "ExpertOverlay policy did not assign every expert target");
            }
            return desired;
        }

        /**
         * Find one canonical simple cycle in the remaining balanced move graph.
         * The returned expert ids name directed owner-to-desired-owner edges.
         */
        std::vector<std::uint32_t> findCycle(
            const std::vector<std::int32_t> &current,
            const std::vector<std::int32_t> &desired,
            const std::vector<bool> &excluded,
            std::size_t participant_count)
        {
            for (std::uint32_t first = 0u; first < current.size(); ++first)
            {
                if (excluded[first] || current[first] == desired[first])
                    continue;
                const std::uint32_t start = static_cast<std::uint32_t>(
                    current[first]);
                const std::uint32_t next = static_cast<std::uint32_t>(
                    desired[first]);
                std::vector<std::uint32_t> path{first};
                if (next == start)
                    return path;

                std::vector<bool> visited(participant_count, false);
                visited[start] = true;
                visited[next] = true;
                std::function<bool(std::uint32_t)> close =
                    [&](std::uint32_t source)
                {
                    for (std::uint32_t expert = 0u;
                         expert < current.size();
                         ++expert)
                    {
                        if (excluded[expert] || expert == first ||
                            current[expert] !=
                                static_cast<std::int32_t>(source) ||
                            current[expert] == desired[expert])
                        {
                            continue;
                        }
                        const auto destination =
                            static_cast<std::uint32_t>(desired[expert]);
                        path.push_back(expert);
                        if (destination == start)
                            return true;
                        if (!visited[destination])
                        {
                            visited[destination] = true;
                            if (close(destination))
                                return true;
                            visited[destination] = false;
                        }
                        path.pop_back();
                    }
                    return false;
                };
                if (close(next))
                    return path;
            }
            return {};
        }
    } // namespace

    bool MoEOverlayDevicePlacementEconomyInput::valid(
        std::uint32_t num_layers,
        std::uint32_t num_experts,
        std::uint32_t participant_count) const noexcept
    {
        if (tier_count == 0u || routed_experts_per_token == 0u ||
            routed_experts_per_token > num_experts ||
            transaction_generation == 0u || payoff_horizon_tokens == 0u)
        {
            return false;
        }
        const std::size_t history_words =
            static_cast<std::size_t>(
                kMoEOverlayDeviceControllerDemandPhaseCount) *
            num_layers * num_experts;
        const std::size_t service_words =
            static_cast<std::size_t>(tier_count) * num_layers *
            kMoEOverlayDeviceControllerEconomyServicePhaseCount;
        const std::size_t migration_entries =
            static_cast<std::size_t>(participant_count) *
            participant_count * num_layers;
        const std::size_t movement_words =
            static_cast<std::size_t>(num_layers) * num_experts;
        if (phase_expert_demand.size() != history_words ||
            service_costs.size() != service_words ||
            migration_costs.size() != migration_entries ||
            last_moved_generation.size() != movement_words)
        {
            return false;
        }
        for (std::uint32_t tier = 0u; tier < tier_count; ++tier)
        {
            for (std::uint32_t layer = 0u; layer < num_layers; ++layer)
            {
                for (std::uint32_t phase = 0u;
                     phase < kMoEOverlayDeviceControllerDemandPhaseCount;
                     ++phase)
                {
                    const std::size_t offset =
                        (static_cast<std::size_t>(tier) * num_layers + layer) *
                            kMoEOverlayDeviceControllerEconomyServicePhaseCount +
                        phase;
                    if (service_costs[offset] == 0u)
                        return false;
                }
            }
        }
        for (std::uint32_t source = 0u;
             source < participant_count;
             ++source)
        {
            for (std::uint32_t destination = 0u;
                 destination < participant_count;
                 ++destination)
            {
                for (std::uint32_t layer = 0u;
                     layer < num_layers;
                     ++layer)
                {
                    const auto &cost = migration_costs[
                        (static_cast<std::size_t>(source) *
                             participant_count +
                         destination) *
                            num_layers +
                        layer];
                    if (source != destination &&
                        cost.transfer_and_repack_ns == 0u &&
                        cost.inference_interference_ns == 0u)
                    {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    bool MoEOverlayDevicePlacementPolicyInput::valid() const noexcept
    {
        if (num_layers == 0u || num_experts == 0u ||
            participants.empty() ||
            participants.size() >
                moe_rebalance_policy::kMaxPolicyParticipants ||
            base_epoch == 0u ||
            base_epoch == std::numeric_limits<std::uint64_t>::max() ||
            layer_start_cursor >= num_layers ||
            minimum_window_activations == 0u ||
            maximum_cycles_per_wave == 0u || command_capacity == 0u ||
            payload_bytes_per_layer.size() != num_layers || !economy ||
            !economy->valid(
                num_layers,
                num_experts,
                static_cast<std::uint32_t>(participants.size())))
        {
            return false;
        }
        for (std::size_t index = 0u; index < participants.size(); ++index)
        {
            if (!participants[index].validAt(index) ||
                participants[index].tier_index < 0 ||
                static_cast<std::uint32_t>(
                    participants[index].tier_index) >=
                    economy->tier_count)
                return false;
        }
        const std::size_t expected =
            participants.size() * static_cast<std::size_t>(num_layers) *
            static_cast<std::size_t>(num_experts);
        return collected_state.size() == expected &&
               std::all_of(
                   payload_bytes_per_layer.begin(),
                   payload_bytes_per_layer.end(),
                   [](std::uint64_t bytes) { return bytes != 0u; });
    }

    bool MoEOverlayDevicePlacementPolicyEvidence::improves() const noexcept
    {
        const std::uint64_t measured_cost = saturatingAdd(
            projected_transfer_and_repack_ns,
            projected_inference_interference_ns);
        return accepted_cycles > 0u &&
               projected_service_gain_ns > measured_cost &&
               projected_net_benefit_ns ==
                   projected_service_gain_ns - measured_cost;
    }

    MoEOverlayDevicePlacementPolicyPlan
    MoEOverlayDevicePlacementPolicyReference::planDynamic(
        const MoEOverlayDevicePlacementPolicyInput &input)
    {
        if (!input.valid())
        {
            throw std::invalid_argument(
                "ExpertOverlay device placement policy input is incomplete");
        }

        MoEOverlayDevicePlacementPolicyPlan plan;
        plan.candidate_owner.assign(
            static_cast<std::size_t>(input.num_layers) * input.num_experts,
            -1);
        const auto priorities = priorityOrder(input);
        plan.evidence.layer_scan_start = input.layer_start_cursor;
        plan.evidence.layer_scan_next =
            (input.layer_start_cursor + 1u) % input.num_layers;

        for (std::uint32_t layer_offset = 0u;
             layer_offset < input.num_layers;
             ++layer_offset)
        {
            const std::uint32_t layer =
                (input.layer_start_cursor + layer_offset) % input.num_layers;
            std::vector<std::uint64_t> counts(input.num_experts, 0u);
            std::vector<std::int32_t> current(input.num_experts, -1);
            std::uint64_t observations = 0u;
            for (std::uint32_t expert = 0u;
                 expert < input.num_experts;
                 ++expert)
            {
                std::uint32_t owner_count = 0u;
                for (std::uint32_t participant = 0u;
                     participant < input.participants.size();
                     ++participant)
                {
                    const std::uint64_t word = input.collected_state[
                        stateOffset(input, participant, layer, expert)];
                    counts[expert] = saturatingAdd(
                        counts[expert],
                        moe_rebalance_policy::collectedStateActivationCount(
                            word));
                    if (moe_rebalance_policy::
                            collectedStateAuthoritativeOwner(word))
                    {
                        if (!moe_rebalance_policy::
                                collectedStatePhysicallyResident(word))
                        {
                            throw std::invalid_argument(
                                "ExpertOverlay snapshot owner is not physically resident");
                        }
                        current[expert] = static_cast<std::int32_t>(participant);
                        ++owner_count;
                    }
                }
                if (owner_count != 1u)
                {
                    throw std::invalid_argument(
                        "ExpertOverlay snapshot must publish exactly one durable owner per expert");
                }
                observations = saturatingAdd(observations, counts[expert]);
                plan.candidate_owner[ownerOffset(input, layer, expert)] =
                    current[expert];
            }
            plan.evidence.snapshot_observations = saturatingAdd(
                plan.evidence.snapshot_observations, observations);
            if (observations < input.minimum_window_activations)
                continue;

            /* The snapshot supplies this phase's delta and durable owner. The
             * measured-economy input represents the device's already-updated
             * model-lifetime phase histories, which are the actual planning
             * demand. A history smaller than the current window cannot have
             * been produced by the monotonic device accumulator. */
            for (std::uint32_t expert = 0u;
                 expert < input.num_experts;
                 ++expert)
            {
                const std::uint64_t historical = saturatingAdd(
                    input.economy->phase_expert_demand[demandOffset(
                        input,
                        kMoEOverlayDeviceControllerEconomyDecodePhase,
                        layer,
                        expert)],
                    input.economy->phase_expert_demand[demandOffset(
                        input,
                        kMoEOverlayDeviceControllerEconomyPrefillPhase,
                        layer,
                        expert)]);
                if (historical < counts[expert])
                {
                    throw std::invalid_argument(
                        "ExpertOverlay measured demand predates the current device snapshot");
                }
                counts[expert] = historical;
            }

            const auto desired = desiredOwners(
                input, current, counts, priorities);
            PlacementScore working_score = scoreLayer(
                input, current, counts, priorities);
            plan.evidence.priority_cost_before = saturatingAdd(
                plan.evidence.priority_cost_before,
                working_score.priority_cost);
            plan.evidence.same_priority_makespan_before = saturatingAdd(
                plan.evidence.same_priority_makespan_before,
                working_score.same_priority_makespan);

            bool layer_changed = false;
            std::uint32_t layer_cycles = 0u;
            std::vector<bool> excluded(input.num_experts, false);
            const std::uint32_t maximum_commands =
                input.dynamic_maximum_commands_per_wave == 0u
                    ? input.command_capacity
                    : std::min(
                          input.command_capacity,
                          input.dynamic_maximum_commands_per_wave);
            while (plan.evidence.accepted_cycles <
                       input.maximum_cycles_per_wave &&
                   layer_cycles < input.dynamic_maximum_cycles_per_layer)
            {
                std::vector<bool> enumerated = excluded;
                std::vector<std::uint32_t> best_cycle;
                CycleEconomyScore best_economy;
                while (true)
                {
                    const auto cycle = findCycle(
                        current,
                        desired,
                        enumerated,
                        input.participants.size());
                    if (cycle.empty())
                        break;
                    for (const std::uint32_t expert : cycle)
                        enumerated[expert] = true;
                    if (plan.commands.size() + cycle.size() >
                        maximum_commands)
                    {
                        continue;
                    }

                    auto candidate = current;
                    for (const std::uint32_t expert : cycle)
                        candidate[expert] = desired[expert];
                    const PlacementScore candidate_score = scoreLayer(
                        input, candidate, counts, priorities);
                    const bool improves_priority =
                        candidate_score.priority_cost <
                        working_score.priority_cost;
                    const bool targets_skew = !improves_priority &&
                        samePriorityCycleIsEligible(
                            input,
                            current,
                            desired,
                            counts,
                            cycle);
                    if (!improves_priority && !targets_skew)
                    {
                        ++plan.evidence.rejected_cycles;
                        for (const std::uint32_t expert : cycle)
                            excluded[expert] = true;
                        continue;
                    }

                    const CycleEconomyScore economy = scoreEconomyCycle(
                        input,
                        current,
                        desired,
                        candidate,
                        counts,
                        cycle,
                        layer);
                    plan.evidence.residency_rejected_cycles +=
                        economy.residency_eligible ? 0u : 1u;
                    plan.evidence.payoff_rejected_cycles +=
                        economy.payoff_eligible ? 0u : 1u;
                    if (!economy.eligible())
                    {
                        ++plan.evidence.rejected_cycles;
                        for (const std::uint32_t expert : cycle)
                            excluded[expert] = true;
                        continue;
                    }
                    if (best_cycle.empty() ||
                        economyScoreBetter(economy, best_economy))
                    {
                        best_cycle = cycle;
                        best_economy = economy;
                    }
                }

                if (best_cycle.empty())
                    break;
                auto candidate = current;
                for (const std::uint32_t expert : best_cycle)
                    candidate[expert] = desired[expert];
                const PlacementScore candidate_score = scoreLayer(
                    input, candidate, counts, priorities);

                for (const std::uint32_t expert : best_cycle)
                {
                    const auto source = static_cast<std::uint32_t>(
                        current[expert]);
                    const auto destination = static_cast<std::uint32_t>(
                        desired[expert]);
                    plan.commands.push_back({
                        .op = static_cast<std::uint32_t>(
                            MoEOverlayDeviceMovementOp::DurableMove),
                        .layer = layer,
                        .expert = expert,
                        .source_participant = source,
                        .destination_participant = destination,
                        .payload_bytes =
                            input.payload_bytes_per_layer[layer],
                        .source_epoch = input.base_epoch,
                        .candidate_epoch = input.base_epoch + 1u,
                    });
                    const auto source_priority =
                        input.participants[source].tier_priority;
                    const auto destination_priority =
                        input.participants[destination].tier_priority;
                    if (destination_priority < source_priority)
                        ++plan.evidence.promotions;
                    else if (destination_priority > source_priority)
                        ++plan.evidence.demotions;
                    else
                        ++plan.evidence.same_priority_moves;
                    plan.candidate_owner[ownerOffset(input, layer, expert)] =
                        static_cast<std::int32_t>(destination);
                }
                current = std::move(candidate);
                working_score = candidate_score;
                plan.evidence.projected_service_gain_ns = saturatingAdd(
                    plan.evidence.projected_service_gain_ns,
                    best_economy.projected_service_gain_ns);
                plan.evidence.projected_transfer_and_repack_ns =
                    saturatingAdd(
                        plan.evidence.projected_transfer_and_repack_ns,
                        best_economy.transfer_and_repack_ns);
                plan.evidence.projected_inference_interference_ns =
                    saturatingAdd(
                        plan.evidence.projected_inference_interference_ns,
                        best_economy.inference_interference_ns);
                plan.evidence.projected_net_benefit_ns = saturatingAdd(
                    plan.evidence.projected_net_benefit_ns,
                    best_economy.projected_net_benefit_ns);
                ++plan.evidence.accepted_cycles;
                ++layer_cycles;
                layer_changed = true;
            }
            if (layer_changed)
            {
                ++plan.evidence.changed_layers;
                plan.evidence.layer_scan_next = (layer + 1u) % input.num_layers;
            }
            plan.evidence.priority_cost_after = saturatingAdd(
                plan.evidence.priority_cost_after,
                working_score.priority_cost);
            plan.evidence.same_priority_makespan_after = saturatingAdd(
                plan.evidence.same_priority_makespan_after,
                working_score.same_priority_makespan);
        }

        std::sort(
            plan.commands.begin(), plan.commands.end(),
            [](const auto &left, const auto &right)
            {
                if (left.destination_participant !=
                    right.destination_participant)
                {
                    return left.destination_participant <
                           right.destination_participant;
                }
                if (left.layer != right.layer)
                    return left.layer < right.layer;
                return left.expert < right.expert;
            });
        for (std::uint32_t ordinal = 0u;
             ordinal < plan.commands.size();
             ++ordinal)
        {
            plan.commands[ordinal].ordinal = ordinal;
            plan.commands[ordinal].payload_slot = ordinal;
        }
        return plan;
    }
} // namespace llaminar2
