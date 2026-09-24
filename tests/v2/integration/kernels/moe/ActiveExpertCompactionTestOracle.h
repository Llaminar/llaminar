/**
 * @file ActiveExpertCompactionTestOracle.h
 * @brief Backend-neutral cases for stable GPU MoE expert-list compaction.
 *
 * CUDA and ROCm must publish the same ascending list consumed by grouped MoE
 * kernels.  This inventory deliberately crosses subgroup and block boundaries,
 * exercises output truncation, and includes empty and dense populations so the
 * backend suites prove one shared contract instead of drifting independently.
 */
#pragma once

#include <algorithm>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace llaminar2::test
{
    /** @brief One exact expert-count input and its serial-scan oracle. */
    struct ActiveExpertCompactionCase
    {
        std::string name;
        std::vector<int> expert_counts;
        int max_active_experts = 0;
        std::vector<int> expected_active_expert_ids;
    };

    /**
     * @brief Construct the canonical serial result for one count vector.
     *
     * The production contract retains positive-count expert ids in ascending
     * order, truncates only at output capacity, and fills every unused slot
     * with -1.  Keeping this oracle host-only prevents either GPU algorithm
     * from validating itself.
     */
    inline std::vector<int> serialActiveExpertList(
        const std::vector<int> &expert_counts,
        int max_active_experts)
    {
        std::vector<int> result;
        result.reserve(static_cast<std::size_t>(max_active_experts));
        for (int expert = 0;
             expert < static_cast<int>(expert_counts.size()) &&
             static_cast<int>(result.size()) < max_active_experts;
             ++expert)
        {
            if (expert_counts[static_cast<std::size_t>(expert)] > 0)
                result.push_back(expert);
        }
        result.resize(static_cast<std::size_t>(max_active_experts), -1);
        return result;
    }

    /**
     * @brief Return adversarial domains shared by CUDA and ROCm regressions.
     */
    inline std::vector<ActiveExpertCompactionCase>
    activeExpertCompactionCases()
    {
        std::vector<ActiveExpertCompactionCase> cases;
        auto add = [&](std::string name,
                       std::vector<int> counts,
                       int max_active_experts)
        {
            cases.push_back({
                .name = std::move(name),
                .expert_counts = std::move(counts),
                .max_active_experts = max_active_experts,
                .expected_active_expert_ids = {},
            });
            auto &test_case = cases.back();
            test_case.expected_active_expert_ids = serialActiveExpertList(
                test_case.expert_counts,
                test_case.max_active_experts);
        };

        add("single_empty", std::vector<int>(1, 0), 1);
        add("single_active", std::vector<int>(1, 7), 1);
        add("wave32_empty", std::vector<int>(32, 0), 32);

        std::vector<int> subgroup_edges(256, 0);
        for (const int expert :
             {0, 31, 32, 63, 64, 127, 128, 191, 192, 254, 255})
        {
            subgroup_edges[static_cast<std::size_t>(expert)] = expert + 1;
        }
        add("subgroup_and_block_edges", std::move(subgroup_edges), 256);

        add("dense_truncation", std::vector<int>(256, 1), 73);

        std::vector<int> block_plus_one(257, 0);
        block_plus_one[0] = 1;
        block_plus_one[255] = 2;
        block_plus_one[256] = 3;
        add("block_plus_one", std::move(block_plus_one), 257);

        std::vector<int> two_chunk_truncation(511, 0);
        for (int expert = 0; expert < 511; ++expert)
        {
            if ((expert % 3) == 0 || (expert % 17) == 1)
                two_chunk_truncation[static_cast<std::size_t>(expert)] =
                    (expert % 5) + 1;
        }
        add("two_chunk_truncation", std::move(two_chunk_truncation), 37);

        std::vector<int> four_chunks(1024, 0);
        for (int expert = 0; expert < 1024; ++expert)
        {
            if ((expert % 7) == 0 || (expert % 29) == 3 || expert == 1023)
                four_chunks[static_cast<std::size_t>(expert)] =
                    (expert % 11) + 1;
        }
        add("four_chunks_sparse", std::move(four_chunks), 1024);

        return cases;
    }
} // namespace llaminar2::test
