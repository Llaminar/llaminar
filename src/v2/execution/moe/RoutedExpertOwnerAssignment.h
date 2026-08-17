/**
 * @file RoutedExpertOwnerAssignment.h
 * @brief Deterministic static ownership assignment for complete routed experts.
 *
 * Static expert parallelism must produce the same owner map independently on
 * every rank before any routed row executes. This file is the single authority
 * for ordering expert IDs and partitioning that order into balanced participant
 * spans. Both ordinary TP graph construction and graph-native routed-expert
 * overlays use these helpers, preventing their ownership semantics from
 * drifting apart.
 */

#pragma once

#include "execution/config/RoutedExpertPolicy.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace llaminar2::routed_expert_ownership
{
    /**
     * @brief Return the next value from a platform-stable SplitMix64 stream.
     *
     * The generator is used only to construct an immutable ownership
     * permutation. It is deliberately independent of request sampling seeds,
     * router observations, and backend state, so every participant derives the
     * same map without communication or model-specific fitting.
     *
     * @param state Mutable deterministic generator state.
     * @return Next uniformly mixed 64-bit value.
     */
    inline uint64_t nextOwnerPermutationValue(uint64_t &state) noexcept
    {
        state += UINT64_C(0x9e3779b97f4a7c15);
        uint64_t value = state;
        value = (value ^ (value >> 30u)) * UINT64_C(0xbf58476d1ce4e5b9);
        value = (value ^ (value >> 27u)) * UINT64_C(0x94d049bb133111eb);
        return value ^ (value >> 31u);
    }

    /**
     * @brief Apply one static owner-order policy to an expert-id sequence.
     *
     * `Ordinal` preserves the supplied order. `Random` performs a deterministic
     * Fisher-Yates permutation keyed by layer and routed tier. The permutation
     * changes which complete experts share one owner while preserving every
     * expert exactly once.
     *
     * @param expert_ids Complete unique expert-id sequence to reorder in place.
     * @param order Requested static ownership order.
     * @param layer_idx Model layer owning this expert set.
     * @param tier_idx Routed placement tier, or zero for ordinary TP.
     * @throws std::invalid_argument for a negative layer or tier index.
     */
    inline void applyOwnerOrder(
        std::vector<int> &expert_ids,
        RoutedExpertOwnerOrder order,
        int layer_idx,
        int tier_idx = 0)
    {
        if (layer_idx < 0 || tier_idx < 0)
        {
            throw std::invalid_argument(
                "Routed expert owner ordering requires non-negative layer and tier indices");
        }
        if (order == RoutedExpertOwnerOrder::Ordinal || expert_ids.size() < 2u)
            return;

        uint64_t state = UINT64_C(0x243f6a8885a308d3);
        state ^= (static_cast<uint64_t>(static_cast<uint32_t>(layer_idx)) + 1u) *
                 UINT64_C(0x9e3779b97f4a7c15);
        state ^= (static_cast<uint64_t>(static_cast<uint32_t>(tier_idx)) + 1u) *
                 UINT64_C(0xbf58476d1ce4e5b9);
        state ^= static_cast<uint64_t>(expert_ids.size()) *
                 UINT64_C(0x94d049bb133111eb);

        for (size_t remaining = expert_ids.size(); remaining > 1u; --remaining)
        {
            const size_t swap_index = static_cast<size_t>(
                nextOwnerPermutationValue(state) % remaining);
            std::swap(expert_ids[remaining - 1u], expert_ids[swap_index]);
        }
    }

    /**
     * @brief Compute one participant's balanced span in an ordered expert list.
     *
     * Remainder experts are assigned one each to the lowest participant IDs.
     * Consequently all participant counts differ by at most one for every
     * positive expert/participant geometry.
     *
     * @param expert_count Number of expert IDs in the ordered list.
     * @param participant_count Number of static owners.
     * @param participant_index Owner whose half-open span is requested.
     * @return `(begin, count)` in the ordered expert-id list.
     * @throws std::invalid_argument for invalid geometry or participant index.
     */
    inline std::pair<size_t, size_t> balancedOwnerSpan(
        size_t expert_count,
        int participant_count,
        int participant_index)
    {
        if (expert_count == 0u || participant_count <= 0 ||
            participant_index < 0 || participant_index >= participant_count ||
            expert_count < static_cast<size_t>(participant_count))
        {
            throw std::invalid_argument(
                "Routed expert ownership requires at least one expert per valid participant");
        }

        const size_t participants = static_cast<size_t>(participant_count);
        const size_t participant = static_cast<size_t>(participant_index);
        const size_t base = expert_count / participants;
        const size_t remainder = expert_count % participants;
        const size_t count = base + (participant < remainder ? 1u : 0u);
        const size_t begin = participant * base + std::min(participant, remainder);
        return {begin, count};
    }

    /**
     * @brief Build one participant's sorted immutable expert-id set.
     *
     * Ownership is selected from a policy-ordered sequence, but the returned
     * IDs are sorted into source-tensor order. This gives model loading,
     * prepared-weight construction, and graph execution one canonical packed
     * layout while retaining the selected ownership set exactly.
     *
     * @param num_experts Number of logical routed experts in the layer.
     * @param participant_count Number of whole-expert owners.
     * @param participant_index Participant receiving the returned IDs.
     * @param layer_idx Model layer used by deterministic random ordering.
     * @param order Static ownership order.
     * @return Sorted logical expert IDs owned by the participant.
     */
    inline std::vector<int> expertIdsForParticipant(
        int num_experts,
        int participant_count,
        int participant_index,
        int layer_idx,
        RoutedExpertOwnerOrder order)
    {
        if (num_experts <= 0)
        {
            throw std::invalid_argument(
                "Routed expert ownership requires a positive expert count");
        }

        std::vector<int> ordered_experts(static_cast<size_t>(num_experts));
        std::iota(ordered_experts.begin(), ordered_experts.end(), 0);
        applyOwnerOrder(ordered_experts, order, layer_idx);

        const auto [begin, count] = balancedOwnerSpan(
            ordered_experts.size(), participant_count, participant_index);
        std::vector<int> owned_experts(
            ordered_experts.begin() + static_cast<std::ptrdiff_t>(begin),
            ordered_experts.begin() + static_cast<std::ptrdiff_t>(begin + count));
        std::sort(owned_experts.begin(), owned_experts.end());
        return owned_experts;
    }

    /**
     * @brief Build one ordinary-TP participant's immutable expert mask.
     *
     * The result is total over every positive expert count and participant
     * count that leaves at least one complete expert per participant. Masks
     * from all participant indices are disjoint and cover every expert exactly
     * once.
     *
     * @param num_experts Number of logical routed experts in the layer.
     * @param participant_count Number of whole-expert owners.
     * @param participant_index Participant receiving the returned mask.
     * @param layer_idx Model layer used by deterministic random ordering.
     * @param order Static ownership order.
     * @return Boolean mask indexed by logical expert ID.
     */
    inline std::vector<bool> expertMaskForParticipant(
        int num_experts,
        int participant_count,
        int participant_index,
        int layer_idx,
        RoutedExpertOwnerOrder order)
    {
        const std::vector<int> owned_experts = expertIdsForParticipant(
            num_experts,
            participant_count,
            participant_index,
            layer_idx,
            order);
        std::vector<bool> mask(static_cast<size_t>(num_experts), false);
        for (const int expert_id : owned_experts)
        {
            mask[static_cast<size_t>(expert_id)] = true;
        }
        return mask;
    }

    /**
     * @brief Build the authoritative owner participant for every expert ID.
     *
     * Runtime placement tables consume a dense expert-to-participant map while
     * model loading and graph construction consume per-participant ID lists or
     * masks. Deriving all three representations here guarantees that physical
     * packing, graph scheduling, and device-resident decode publication cannot
     * disagree about a non-contiguous static assignment.
     *
     * @param num_experts Number of logical routed experts in the layer.
     * @param participant_count Number of whole-expert owners.
     * @param layer_idx Model layer used by deterministic random ordering.
     * @param order Static ownership order.
     * @return Dense vector indexed by expert ID containing its sole owner.
     */
    inline std::vector<int> ownerParticipantByExpert(
        int num_experts,
        int participant_count,
        int layer_idx,
        RoutedExpertOwnerOrder order)
    {
        if (num_experts <= 0 || participant_count <= 0 ||
            num_experts < participant_count)
        {
            throw std::invalid_argument(
                "Routed expert owner map requires at least one expert per participant");
        }

        std::vector<int> owners(static_cast<size_t>(num_experts), -1);
        for (int participant = 0; participant < participant_count; ++participant)
        {
            const std::vector<int> expert_ids = expertIdsForParticipant(
                num_experts,
                participant_count,
                participant,
                layer_idx,
                order);
            for (const int expert_id : expert_ids)
            {
                int &owner = owners[static_cast<size_t>(expert_id)];
                if (owner != -1)
                {
                    throw std::logic_error(
                        "Routed expert owner assignment produced duplicate ownership");
                }
                owner = participant;
            }
        }

        if (std::find(owners.begin(), owners.end(), -1) != owners.end())
        {
            throw std::logic_error(
                "Routed expert owner assignment did not cover every expert");
        }
        return owners;
    }
} // namespace llaminar2::routed_expert_ownership
