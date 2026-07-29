/**
 * @file SmallFloatGroupingTestOracle.h
 * @brief Shared exact oracle for compact CUDA and ROCm MoE route grouping.
 *
 * The production GPU backends use the compact float-route planner for MTP
 * verifier batches whose complete route table fits in 64 slots. This helper
 * defines one backend-independent matrix and computes the serial stable-order
 * result. CUDA and ROCm integration tests consume the same cases so backend
 * drift cannot hide behind different test data.
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

namespace llaminar2::test
{
    /**
     * @brief One compact grouped-verifier route case and its serial result.
     */
    struct SmallFloatGroupingCase
    {
        int verifier_rows = 0;
        int top_k = 0;
        int num_experts = 0;
        int max_active_experts = 0;
        std::vector<float> routing_indices;
        std::vector<float> routing_weights;
        std::vector<int> expert_counts;
        std::vector<int> expert_offsets;
        std::vector<int> grouped_token_indices;
        std::vector<int> original_to_grouped;
        std::vector<int> original_expert_ids;
        std::vector<float> grouped_weights;
        std::vector<int> active_expert_ids;
    };

    /**
     * @brief Build the exact serial grouping result for one verifier geometry.
     *
     * Expert ids intentionally repeat and leave most of the 256-entry expert
     * domain empty. This exercises stable ordering within an expert, sorted
     * active-expert publication, and long zero-count offset runs. Weights are
     * binary fractions so their byte representation survives device copies
     * exactly and can be compared without a floating-point tolerance.
     */
    inline SmallFloatGroupingCase makeSmallFloatGroupingCase(
        int verifier_rows,
        int top_k,
        int num_experts = 256)
    {
        SmallFloatGroupingCase result;
        result.verifier_rows = verifier_rows;
        result.top_k = top_k;
        result.num_experts = num_experts;

        const int total_slots = verifier_rows * top_k;
        result.max_active_experts = std::min(total_slots, num_experts);
        result.routing_indices.resize(static_cast<std::size_t>(total_slots));
        result.routing_weights.resize(static_cast<std::size_t>(total_slots));
        result.expert_counts.assign(static_cast<std::size_t>(num_experts), 0);
        result.expert_offsets.assign(static_cast<std::size_t>(num_experts), 0);
        result.grouped_token_indices.assign(static_cast<std::size_t>(total_slots), 0);
        result.original_to_grouped.assign(static_cast<std::size_t>(total_slots), -1);
        result.original_expert_ids.assign(static_cast<std::size_t>(total_slots), -1);
        result.grouped_weights.assign(static_cast<std::size_t>(total_slots), 0.0f);
        result.active_expert_ids.assign(
            static_cast<std::size_t>(result.max_active_experts),
            -1);

        /*
         * Restrict the selected domain enough to force repeated experts even
         * in small cells, while the full 256-expert output still proves that
         * offsets remain total across every unselected expert.
         */
        const int selected_expert_domain =
            std::max(1, std::min(num_experts, std::max(3, total_slots / 2)));
        for (int slot = 0; slot < total_slots; ++slot)
        {
            const int token = slot / top_k;
            const int expert =
                (slot * 37 + token * 11 + (slot % 3) * 17) %
                selected_expert_domain;
            result.routing_indices[static_cast<std::size_t>(slot)] =
                static_cast<float>(expert);
            result.routing_weights[static_cast<std::size_t>(slot)] =
                static_cast<float>(slot + 1) / 128.0f;
            ++result.expert_counts[static_cast<std::size_t>(expert)];
            result.original_expert_ids[static_cast<std::size_t>(slot)] = expert;
        }

        int running_offset = 0;
        int active_count = 0;
        for (int expert = 0; expert < num_experts; ++expert)
        {
            result.expert_offsets[static_cast<std::size_t>(expert)] =
                running_offset;
            const int count =
                result.expert_counts[static_cast<std::size_t>(expert)];
            if (count > 0)
            {
                result.active_expert_ids[static_cast<std::size_t>(active_count)] =
                    expert;
                ++active_count;
            }
            running_offset += count;
        }

        std::vector<int> write_heads(static_cast<std::size_t>(num_experts), 0);
        for (int slot = 0; slot < total_slots; ++slot)
        {
            const int expert = result.original_expert_ids[static_cast<std::size_t>(slot)];
            const int destination =
                result.expert_offsets[static_cast<std::size_t>(expert)] +
                write_heads[static_cast<std::size_t>(expert)]++;
            result.original_to_grouped[static_cast<std::size_t>(slot)] =
                destination;
            result.grouped_token_indices[static_cast<std::size_t>(destination)] =
                slot / top_k;
            result.grouped_weights[static_cast<std::size_t>(destination)] =
                result.routing_weights[static_cast<std::size_t>(slot)];
        }

        return result;
    }

    /**
     * @brief Return the complete production compact-verifier M/top-k matrix.
     */
    inline std::vector<SmallFloatGroupingCase> smallFloatGroupingCases()
    {
        constexpr int kTopKValues[] = {1, 2, 4, 8, 16};
        constexpr int kMaxRouteSlots = 64;

        std::vector<SmallFloatGroupingCase> cases;
        for (int verifier_rows = 1; verifier_rows <= 4; ++verifier_rows)
        {
            for (const int top_k : kTopKValues)
            {
                if (verifier_rows * top_k <= kMaxRouteSlots)
                {
                    cases.push_back(
                        makeSmallFloatGroupingCase(verifier_rows, top_k));
                }
            }
        }
        return cases;
    }
} // namespace llaminar2::test
