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
#include <stdexcept>

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
    /** @brief Immutable original rows, route width and active-list publication extent.
     * @details experts bounds count/offset planes; rows and top_k determine original
     * route slots; max_active bounds the sorted active-expert publication. */
    struct SmallFloatGroupingGeometry
    {
        int experts, rows, top_k, max_active;
    };

    /**
     * @brief Enumerate bounded dimension totality and every expert-count tail.
     * @param max_rows Backend's actual compact admission:4 CUDA or64 ROCm.
     * @param independent_max Independently expected policy boundary:8 CUDA or32 ROCm.
     * @return Ordered cells, including intentional repeated boundary/model cells.
     * @throws std::invalid_argument For a different unsupported backend extent.
     */
    inline std::vector<SmallFloatGroupingGeometry>
    smallFloatGroupingPublicationGeometries(int max_rows, int independent_max)
    {
        if (!((max_rows == 4 && independent_max == 8) || (max_rows == 64 && independent_max == 32)))
            throw std::invalid_argument("Unknown compact grouping backend policy");
        std::vector<SmallFloatGroupingGeometry> result;
        std::vector<int> boundaries = {1,
                                       31,
                                       32,
                                       33,
                                       63,
                                       64,
                                       65,
                                       255,
                                       256,
                                       257,
                                       511,
                                       512,
                                       independent_max - 1,
                                       independent_max,
                                       independent_max + 1};
        std::sort(boundaries.begin(), boundaries.end());
        boundaries.erase(std::unique(boundaries.begin(), boundaries.end()), boundaries.end());
        for (int e : boundaries)
            for (int k = 1; k <= 16; ++k)
                for (int m = 1; m <= std::min(max_rows, 64 / k); ++m)
                {
                    const int a = std::min(e, m * k);
                    result.push_back({e, m, k, a});
                    if (a > 1)
                        result.push_back({e, m, k, 1});
                }
        for (int e = 1; e <= 512; ++e)
            result.push_back({e, 1, 10, std::min(e, 10)});
        return result;
    }

    /**
     * @brief Author one dirty publication and independently group original slots.
     * @param g Immutable admitted geometry, never a live route-count shadow.
     * @param pattern Replay 0..7: full,invalid,partial,full,skew,dense,canonical,full.
     * @return Complete metadata and literal FP32 copied weights, including signed zero.
     * @throws std::invalid_argument For invalid geometry or replay index.
     * @details Counts include valid zero-weight routes. Integer prefixes walk expert
     * IDs once; stable scatter walks original row/top-k slots. No device scan or
     * production arithmetic is mirrored. Pattern 6 preserves the original canonical
     * fixture, while other patterns deliberately exercise high-half and empty tails.
     */
    inline SmallFloatGroupingCase
    makeSmallFloatGroupingPublicationCase(SmallFloatGroupingGeometry g, int pattern)
    {
        const int slots = g.rows * g.top_k;
        if (g.experts < 1 || g.experts > 512 || g.rows < 1 || g.rows > 64 || g.top_k < 1 ||
            g.top_k > 16 || slots > 64 || g.max_active < 1 ||
            g.max_active > std::min(g.experts, slots) || pattern < 0 || pattern > 7)
            throw std::invalid_argument("Invalid compact grouping publication case");
        auto result = makeSmallFloatGroupingCase(g.rows, g.top_k, g.experts);
        result.max_active_experts = g.max_active;
        if (pattern != 6)
            for (int slot = 0; slot < slots; ++slot)
            {
                const int choices[] = {g.experts - 1,
                                       g.experts / 2,
                                       std::min(g.experts - 1, 255),
                                       0,
                                       g.experts - 1,
                                       std::min(g.experts - 1, 300),
                                       std::min(g.experts - 1, 1),
                                       std::min(g.experts - 1, 400),
                                       g.experts / 2,
                                       std::min(g.experts - 1, 17)};
                int expert = choices[slot % 10];
                if (pattern == 1)
                    expert = slot % 2 ? -1 : g.experts;
                if (pattern == 2 && slot % 3 == 0)
                    expert = -1;
                if (pattern == 4)
                    expert = g.experts - 1;
                if (pattern == 5)
                    expert = slot % g.experts;
                result.routing_indices[slot] = float(expert);
                result.routing_weights[slot] =
                    slot % 3 == 0 ? -0.0f : (slot % 3 == 1 ? 0.0f : float(slot + 1) * 0.125f);
            }
        result.expert_counts.assign(g.experts, 0);
        result.expert_offsets.assign(g.experts, 0);
        result.grouped_token_indices.assign(slots, 0);
        result.original_to_grouped.assign(slots, -1);
        result.original_expert_ids.assign(slots, -1);
        result.grouped_weights.assign(slots, 0.0f);
        result.active_expert_ids.assign(g.max_active, -1);
        for (int slot = 0; slot < slots; ++slot)
        {
            const int expert = int(result.routing_indices[slot]);
            if (expert >= 0 && expert < g.experts)
            {
                ++result.expert_counts[expert];
                result.original_expert_ids[slot] = expert;
            }
        }
        int prefix = 0, active = 0;
        for (int e = 0; e < g.experts; ++e)
        {
            result.expert_offsets[e] = prefix;
            prefix += result.expert_counts[e];
            if (result.expert_counts[e])
            {
                if (active < g.max_active)
                    result.active_expert_ids[active] = e;
                ++active;
            }
        }
        auto next = result.expert_offsets;
        for (int slot = 0; slot < slots; ++slot)
        {
            const int e = result.original_expert_ids[slot];
            if (e < 0)
                continue;
            const int destination = next[e]++;
            result.original_to_grouped[slot] = destination;
            result.grouped_token_indices[destination] = slot / g.top_k;
            result.grouped_weights[destination] = result.routing_weights[slot];
        }
        return result;
    }

} // namespace llaminar2::test
