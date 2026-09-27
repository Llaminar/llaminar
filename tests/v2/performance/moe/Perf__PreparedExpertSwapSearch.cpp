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
    }
}
