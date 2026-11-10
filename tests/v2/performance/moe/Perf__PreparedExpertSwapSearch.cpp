/**
 * @file Perf__PreparedExpertSwapSearch.cpp
 * @brief Isolated CPU economy-search timing with identical candidate decisions.
 *
 * The production maintenance worker prices many exchanges against one immutable
 * routing transaction. Compare the existing complete-map scorer with its exact
 * two-owner specialization. Setup is separate from search timing, all candidates
 * contribute to an authenticated checksum, and no model or device is required.
 * This performance diagnostic is intentionally not in production preflight.
 */
#include "execution/moe/MoEOverlayPreparedTransactionCost.h"
#include "execution/moe/MoEExpertOwnerMap.h"
#include "../../unit/execution/moe/ExactTwoTierAssignmentFixture.h"
#include <iomanip>
#include <limits>

#include <gtest/gtest.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

namespace llaminar2::test
{
    namespace
    {
        using namespace moe_overlay_economy;

        /**
         * @brief Median latency of complete, serial proposal-search rounds.
         * @param operation Search returning an integer sum over all candidates.
         * @param expected Independent complete-map checksum.
         * @return Median microseconds, excluding construction and assertion work.
         */
        template <typename Operation>
        double medianSearchMicroseconds(Operation &&operation, std::uint64_t expected)
        {
            std::vector<double> samples;
            for (int repetition = 0; repetition < 13; ++repetition)
            {
                const auto begin = std::chrono::steady_clock::now();
                const auto actual = operation();
                const auto end = std::chrono::steady_clock::now();
                EXPECT_EQ(actual, expected);
                if (repetition >= 3)
                    samples.push_back(std::chrono::duration<double, std::micro>(end - begin).count());
            }
            std::sort(samples.begin(), samples.end());
            return samples[samples.size() / 2];
        }

        TEST(PreparedExpertSwapSearchPerf, ExactSearchAcrossPrefillAndDecodeTransactions)
        {
            constexpr std::uint32_t experts = 128, top_k = 8;
            std::mt19937 random(42);
            for (const std::uint32_t participants : {2u, 6u, 17u})
            for (const std::uint32_t rows : {1u, 512u})
            {
                std::vector<std::int32_t> ids(rows * top_k), before(experts);
                std::vector<std::uint64_t> prices(participants);
                std::vector<ServiceCostPair> scratch(participants);
                for (auto &id : ids) id = random() % experts;
                for (auto &owner : before) owner = random() % participants;
                for (auto &price : prices) price = 1u + random() % 10000u;
                auto after = before;
                const PreparedTransactionCost transaction({ids.data(), ids.size(), rows, top_k, top_k}, experts);
                const auto setup_begin = std::chrono::steady_clock::now();
                const PreparedTransactionSwapSearch search(transaction, before, prices);
                const auto setup_end = std::chrono::steady_clock::now();
                const auto complete_map = [&]()
                {
                    std::uint64_t checksum = 0;
                    for (std::uint32_t first = 0; first < experts; ++first)
                        for (std::uint32_t second = first + 1; second < experts; ++second)
                        {
                            std::swap(after[first], after[second]);
                            const auto cost = transaction.score(
                                {before.data(), after.data(), prices.data(), experts, participants},
                                scratch.data(), participants);
                            if (!cost.complete()) throw std::logic_error("Complete-map microbench pricing failed");
                            checksum += cost.cost.after_ns;
                            std::swap(after[first], after[second]);
                        }
                    return checksum;
                };
                const auto two_owner = [&]()
                {
                    std::uint64_t checksum = 0;
                    for (std::uint32_t first = 0; first < experts; ++first)
                        for (std::uint32_t second = first + 1; second < experts; ++second)
                        {
                            const auto cost = search.scoreSwap(first, second);
                            if (!cost.complete()) throw std::logic_error("Swap microbench pricing failed");
                            checksum += cost.cost.after_ns;
                        }
                    return checksum;
                };
                const auto expected = complete_map();
                const double full_us = medianSearchMicroseconds(complete_map, expected);
                const double swap_us = medianSearchMicroseconds(two_owner, expected);
                std::cout << "[PreparedExpertSwapSearch] experts=" << experts
                          << " participants=" << participants << " rows=" << rows
                          << " setup_us=" << std::chrono::duration<double, std::micro>(setup_end - setup_begin).count()
                          << " full_map_us=" << full_us << " swap_us=" << swap_us
                          << " speedup=" << full_us / swap_us << " checksum=" << expected << '\n';
            }
        }
        /**
         * @brief Bounded authored full-planner control, explicitly opt-in.
         *
         * This exercises the public owner, including profile validation and
         * returned placement construction. The valid zero-quota third tier
         * invokes the retained general algorithm; it is not a second policy.
         * Complete vectors are checked outside clocks and raw samples are
         * printed after both reversed orders have retired.
         */
        TEST(PreparedExpertSwapSearchPerf, DISABLED_ExactTwoTierAssignment)
        {
            constexpr int experts = 512;
            std::vector<uint64_t> counts(3 * experts);
            for (std::size_t i = 0; i < counts.size(); ++i) counts[i] = 1 + (i * 17) % 101;
            for (const int layers : {1, 49})
            {
                ExactTwoTierAssignmentFixture input(experts, 168, counts,
                    {{{5, 3, 11}, {7, 13, 2}}}, false, 2, layers);
                auto oracle = input.flowOracle();
                const auto expected = oracle.solve();
                struct Sample { int order, repetition; bool general; double microseconds; };
                std::vector<Sample> samples;
                for (int order = 0; order < 2; ++order)
                    for (int repetition = 0; repetition < 4; ++repetition)
                        for (int position = 0; position < 2; ++position)
                        {
                            const bool general = (position == order);
                            const auto &chosen = general ? oracle : input;
                            const auto begin = std::chrono::steady_clock::now();
                            const auto actual = chosen.solve();
                            const auto end = std::chrono::steady_clock::now();
                            ASSERT_EQ(actual.size(), expected.size());
                            for (std::size_t layer = 0; layer < actual.size(); ++layer)
                            {
                                ASSERT_EQ(actual[layer].layer, expected[layer].layer);
                                ASSERT_EQ(actual[layer].routed_expert_tier, expected[layer].routed_expert_tier);
                            }
                            if (repetition != 0) samples.push_back({order, repetition, general,
                                std::chrono::duration<double, std::micro>(end - begin).count()});
                        }
                for (const auto &sample : samples)
                    std::cout << std::setprecision(std::numeric_limits<double>::max_digits10)
                        << "[ExactTwoTierAssignment] authored=true experts=" << experts
                        << " layers=" << layers << " quota=168 order=" << sample.order
                        << " sample=" << sample.repetition
                        << " algorithm=" << (sample.general ? "general_zero_quota_oracle" : "two_tier")
                        << " us=" << sample.microseconds << '\n';
            }
        }
        /** @brief Isolate the original sorted owner relation from candidate scoring. */
        TEST(PreparedExpertSwapSearchPerf, DISABLED_CanonicalOwnerLayerRanges)
        {
            constexpr int experts = 512, layers = 49, visits = 4257;
            std::vector<uint64_t> counts(3 * experts, 1);
            ExactTwoTierAssignmentFixture input(experts, 168, counts,
                {{{5, 3, 11}, {7, 13, 2}}}, false, 2, layers);
            // Static preparation is outside the scan clocks and does not need
            // either assignment algorithm under comparison.
            input.plan.residency_policy = RoutedExpertResidencyPolicy::StaticById;
            input.plan.placements = input.solve();
            const auto map = MoEExpertOwnerMap::build(input.plan);
            const auto scan = [&](bool range)
            {
                uint64_t checksum = 0;
                for (int visit = 0; visit < visits; ++visit)
                {
                    const int layer = 32 + visit % layers;
                    if (range)
                    {
                        for (const auto &owner : map.placementsForLayer(layer))
                            checksum += uint64_t(owner.expert_id + 1) * (owner.tier_idx + 1);
                    }
                    else
                    {
                        for (const auto &owner : map.owners())
                            if (owner.layer_idx == layer)
                                checksum += uint64_t(owner.expert_id + 1) * (owner.tier_idx + 1);
                    }
                }
                return checksum;
            };
            const auto expected = scan(false);
            struct Sample { int order, repetition; bool range; double microseconds; };
            std::vector<Sample> samples;
            for (int order = 0; order < 2; ++order)
                for (int repetition = 0; repetition < 4; ++repetition)
                    for (int position = 0; position < 2; ++position)
                    {
                        const bool range = position == order;
                        const auto begin = std::chrono::steady_clock::now();
                        const auto actual = scan(range);
                        const auto end = std::chrono::steady_clock::now();
                        ASSERT_EQ(actual, expected);
                        if (repetition != 0) samples.push_back({order, repetition, range,
                            std::chrono::duration<double, std::micro>(end - begin).count()});
                    }
            for (const auto &sample : samples)
                std::cout << std::setprecision(std::numeric_limits<double>::max_digits10)
                    << "[CanonicalOwnerLayerRanges] authored=true experts=" << experts
                    << " layers=" << layers << " visits=" << visits << " order=" << sample.order
                    << " sample=" << sample.repetition << " traversal=" << (sample.range ? "range" : "full_scan")
                    << " us=" << sample.microseconds << " checksum=" << expected << '\n';
        }
    }
}
