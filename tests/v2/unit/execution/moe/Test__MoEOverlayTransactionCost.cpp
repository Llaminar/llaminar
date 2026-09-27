/**
 * @file Test__MoEOverlayTransactionCost.cpp
 * @brief Device-free exact routing-cohort economics and malformed-input regressions.
 *
 * These tests distinguish sequential token latency from window-aggregate work,
 * exercise true grouped transaction boundaries, and prove arbitrary participant
 * geometry without loading a model or touching CUDA/HIP. The implementation is
 * shared with device callers; real graph integration remains a separate gate.
 * Prepared CPU candidate scoring is checked against raw routes at every batch
 * boundary, including sparse/long prefills, padding, arbitrary owners and overflow.
 */
#include "execution/moe/MoEOverlayTransactionCost.h"
#include "execution/moe/MoEOverlayPreparedTransactionCost.h"

#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <vector>

namespace llaminar2::test
{
    namespace
    {
        using namespace moe_overlay_economy;

        /** @brief Small exact fixture, with no device-specific placement assumptions. */
        class TransactionCostFixture : public ::testing::Test
        {
        protected:
            std::array<std::int32_t, 4> before{0, 0, 1, 1};
            std::array<std::int32_t, 4> after{1, 0, 0, 1};
            std::array<std::uint64_t, 2> prices{1, 3};
            std::array<ServiceCostPair, 2> scratch{};

            /** @return A borrowed placement view; fixture arrays outlive scoring. */
            TransactionPlacementCosts placement() const
            {
                return {before.data(), after.data(), prices.data(), 4, 2};
            }

            /** @return Exact cost for the supplied logical rows, not bucket padding. */
            TransactionCostResult score(const std::int32_t *ids, std::uint64_t capacity,
                                        std::uint32_t rows = 1, std::uint32_t stride = 2)
            {
                return scoreTransaction({ids, capacity, rows, 2, stride}, placement(),
                                        scratch.data(), scratch.size());
            }

            /** @return Sum of the two separate one-row transaction maxima. */
            ServiceCostPair serial(const std::array<std::int32_t, 4> &ids)
            {
                ServiceCostPair result;
                for (std::size_t offset : {0u, 2u})
                    EXPECT_EQ(appendTransaction(result, score(ids.data() + offset, 2)),
                              TransactionCostStatus::Complete);
                return result;
            }
        };

        TEST_F(TransactionCostFixture, IdenticalMarginalsHaveOppositeMovementPayoff)
        {
            // Both windows select each expert once. Losing these boundaries
            // makes the profitable exchange indistinguishable from a regression.
            const auto gain = serial({0, 1, 2, 3});
            EXPECT_EQ(gain.before_ns, 8u);
            EXPECT_EQ(gain.after_ns, 6u);
            const auto loss = serial({0, 3, 1, 2});
            EXPECT_EQ(loss.before_ns, 6u);
            EXPECT_EQ(loss.after_ns, 8u);
        }

        TEST_F(TransactionCostFixture, GroupedRowsAreOneConcurrentTransaction)
        {
            const std::array<std::int32_t, 4> ids{0, 1, 2, 3};
            const auto batch = score(ids.data(), ids.size(), 2);
            ASSERT_TRUE(batch.complete());
            EXPECT_EQ(batch.cost.before_ns, 6u);
            EXPECT_EQ(batch.cost.after_ns, 6u);
            EXPECT_NE(batch.cost.before_ns, serial(ids).before_ns);
        }

        TEST_F(TransactionCostFixture, RareSlowRouteCannotHideBehindWholeWindowGPUWork)
        {
            // 255 fully accelerator-local tokens, then one slow remote route.
            // Expert 3 has no demand and is demoted in exchange for expert 2.
            before = {0, 0, 1, 0};
            after = {0, 0, 0, 1};
            prices = {1, 10};
            ServiceCostPair window;
            const std::array<std::int32_t, 2> local{0, 1}, remote{0, 2};
            for (int token = 0; token < 255; ++token)
                ASSERT_EQ(appendTransaction(window, score(local.data(), 2)),
                          TransactionCostStatus::Complete);
            ASSERT_EQ(appendTransaction(window, score(remote.data(), 2)),
                      TransactionCostStatus::Complete);
            EXPECT_EQ(window.before_ns, 520u);
            EXPECT_EQ(window.after_ns, 512u);
            // The broken aggregate-then-max calculation instead reports 511→512.
            EXPECT_GT(window.before_ns - window.after_ns, 0u);
        }

        TEST_F(TransactionCostFixture, LogicalPrefixIgnoresBucketAndStridePadding)
        {
            const std::array<std::int32_t, 8> ids{0, 1, -1, 2, 3, -1, -1, -1};
            const auto result = score(ids.data(), ids.size(), 2, 3);
            ASSERT_TRUE(result.complete());
            EXPECT_EQ(result.cost.before_ns, 6u);
            EXPECT_EQ(result.cost.after_ns, 6u);
            EXPECT_EQ(score(ids.data(), 4, 2, 3).status,
                      TransactionCostStatus::InvalidGeometry);
        }

        TEST_F(TransactionCostFixture, RejectsMalformedRoutesOwnersPricesAndScratch)
        {
            std::array<std::int32_t, 2> ids{0, 4};
            EXPECT_EQ(score(ids.data(), 2).status, TransactionCostStatus::InvalidExpert);
            ids[1] = -1;
            EXPECT_EQ(score(ids.data(), 2).status, TransactionCostStatus::InvalidExpert);
            ids[1] = 1;
            before[1] = -1;
            EXPECT_EQ(score(ids.data(), 2).status, TransactionCostStatus::InvalidOwner);
            before[1] = 0;
            after[1] = 2;
            EXPECT_EQ(score(ids.data(), 2).status, TransactionCostStatus::InvalidOwner);
            after[1] = 0;
            prices[1] = 0;
            EXPECT_EQ(score(ids.data(), 2).status, TransactionCostStatus::UnpricedParticipant);
            prices[1] = 3;
            EXPECT_EQ(scoreTransaction({ids.data(), 2, 1, 2, 2}, placement(),
                                      scratch.data(), 1).status,
                      TransactionCostStatus::InvalidGeometry);
            EXPECT_EQ(score(nullptr, 2).status, TransactionCostStatus::InvalidGeometry);
            EXPECT_EQ(score(ids.data(), 2, 0).status, TransactionCostStatus::InvalidGeometry);
            EXPECT_EQ(score(ids.data(), 2, 1, 1).status, TransactionCostStatus::InvalidGeometry);
        }

        TEST_F(TransactionCostFixture, OverflowAndFailedAppendNeverPublishPartialCosts)
        {
            prices[0] = std::numeric_limits<std::uint64_t>::max();
            const std::array<std::int32_t, 2> ids{0, 1};
            const auto failed = score(ids.data(), 2);
            EXPECT_EQ(failed.status, TransactionCostStatus::Overflow);
            EXPECT_EQ(failed.cost.before_ns, 0u);
            EXPECT_EQ(failed.cost.after_ns, 0u);
            ServiceCostPair window{11, 13};
            EXPECT_EQ(appendTransaction(window, failed), TransactionCostStatus::Overflow);
            EXPECT_EQ(window.before_ns, 11u);
            EXPECT_EQ(window.after_ns, 13u);
            const TransactionCostResult enormous{TransactionCostStatus::Complete,
                {1, std::numeric_limits<std::uint64_t>::max()}};
            EXPECT_EQ(appendTransaction(window, enormous), TransactionCostStatus::Overflow);
            EXPECT_EQ(window.before_ns, 11u);
            EXPECT_EQ(window.after_ns, 13u);
        }

        TEST_F(TransactionCostFixture, ExhaustiveTwoTransactionRoutesMatchIndependentOracle)
        {
            // Exhaust all distinct top-2 choices per row and every binary owner
            // map. The independent oracle gathers per-participant route counts
            // first, rather than reusing the production scatter accumulator.
            for (unsigned mask = 0; mask < 16; ++mask)
            {
                for (unsigned expert = 0; expert < 4; ++expert)
                    before[expert] = (mask >> expert) & 1u;
                after = before;
                std::swap(after[0], after[2]);
                for (int a = 0; a < 4; ++a) for (int b = a + 1; b < 4; ++b)
                for (int c = 0; c < 4; ++c) for (int d = c + 1; d < 4; ++d)
                {
                    const std::array<std::int32_t, 4> routes{a, b, c, d};
                    const auto oracle = [&](const auto &owners)
                    {
                        std::uint64_t total = 0;
                        for (unsigned row = 0; row < 2; ++row)
                        {
                            std::uint64_t maximum = 0;
                            for (unsigned p = 0; p < 2; ++p)
                            {
                                unsigned count = 0;
                                for (unsigned slot = 0; slot < 2; ++slot)
                                    count += owners[routes[row * 2 + slot]] == static_cast<int>(p);
                                maximum = std::max(maximum, count * prices[p]);
                            }
                            total += maximum;
                        }
                        return total;
                    };
                    const auto actual = serial(routes);
                    ASSERT_EQ(actual.before_ns, oracle(before));
                    ASSERT_EQ(actual.after_ns, oracle(after));
                }
            }
        }

        TEST(TransactionCost, ParticipantPermutationAndLargeTopologyDoNotChangeCost)
        {
            constexpr std::uint32_t participants = 17;
            std::vector<std::int32_t> owners(participants), ids(participants);
            std::iota(owners.begin(), owners.end(), 0);
            std::iota(ids.begin(), ids.end(), 0);
            std::vector<std::uint64_t> prices(participants);
            std::iota(prices.begin(), prices.end(), 1);
            std::vector<ServiceCostPair> scratch(participants);
            for (unsigned permutation = 0; permutation < participants; ++permutation)
            {
                const auto result = scoreTransaction(
                    {ids.data(), ids.size(), 1, participants, participants},
                    {owners.data(), owners.data(), prices.data(), participants, participants},
                    scratch.data(), scratch.size());
                ASSERT_TRUE(result.complete());
                EXPECT_EQ(result.cost.before_ns, participants);
                EXPECT_EQ(result.cost.after_ns, participants);
                std::rotate(owners.begin(), owners.begin() + 1, owners.end());
            }
        }

        TEST(PreparedTransactionCost, MatchesRawRoutesAcrossGeometryAndCandidateSwaps)
        {
            std::mt19937 random(42);
            for (const std::uint32_t experts : {6u, 32u, 256u})
            for (const std::uint32_t participants : {1u, 2u, 3u, 17u})
            for (const std::uint32_t rows : {1u, 2u, 3u, 15u, 64u, 512u, 4096u})
            {
                const auto top_k = std::min(experts, 8u);
                const auto stride = top_k + 3u;
                std::vector<std::int32_t> ids(rows * stride + 16, -1);
                std::vector<std::int32_t> before(experts), after(experts);
                std::vector<std::uint64_t> prices(participants);
                std::vector<ServiceCostPair> raw_scratch(participants), prepared_scratch(participants);
                for (auto &owner : before) owner = random() % participants;
                after = before;
                for (auto &price : prices) price = 1u + random() % 10000u;
                for (std::uint32_t row = 0; row < rows; ++row)
                    for (std::uint32_t slot = 0; slot < top_k; ++slot)
                        ids[row * stride + slot] = random() % experts;
                const TransactionRoutes routes{ids.data(), ids.size(), rows, top_k, stride};
                const moe_overlay_economy::PreparedTransactionCost prepared(routes, experts);
                for (int candidate = 0; candidate < 16; ++candidate)
                {
                    // Keep the prepared evidence while ownership changes. A
                    // cached owner map would silently price only the first swap.
                    std::swap(after[random() % experts], after[random() % experts]);
                    const TransactionPlacementCosts placement{
                        before.data(), after.data(), prices.data(), experts, participants};
                    const auto raw = scoreTransaction(routes, placement, raw_scratch.data(), participants);
                    const auto compact = prepared.score(placement, prepared_scratch.data(), participants);
                    ASSERT_EQ(raw.status, compact.status);
                    ASSERT_TRUE(compact.complete());
                    ASSERT_EQ(raw.cost.before_ns, compact.cost.before_ns);
                    ASSERT_EQ(raw.cost.after_ns, compact.cost.after_ns);
                    for (std::uint32_t participant = 0; participant < participants; ++participant)
                    {
                        ASSERT_EQ(raw_scratch[participant].before_ns, prepared_scratch[participant].before_ns);
                        ASSERT_EQ(raw_scratch[participant].after_ns, prepared_scratch[participant].after_ns);
                    }
                }
            }
        }

        TEST_F(TransactionCostFixture, PreparedCountsKeepSequentialTransactionMaximaSeparate)
        {
            const std::array<std::int32_t, 4> ids{0, 1, 2, 3};
            ServiceCostPair separate;
            for (const std::size_t offset : {0u, 2u})
            {
                const moe_overlay_economy::PreparedTransactionCost prepared(
                    {ids.data() + offset, 2, 1, 2, 2}, 4);
                ASSERT_EQ(appendTransaction(separate, prepared.score(placement(), scratch.data(), 2)),
                          TransactionCostStatus::Complete);
            }
            EXPECT_EQ(separate.before_ns, 8u);
            EXPECT_EQ(separate.after_ns, 6u);
            const moe_overlay_economy::PreparedTransactionCost grouped(
                {ids.data(), ids.size(), 2, 2, 2}, 4);
            const auto combined = grouped.score(placement(), scratch.data(), 2);
            ASSERT_TRUE(combined.complete());
            EXPECT_EQ(combined.cost.before_ns, 6u);
            EXPECT_EQ(combined.cost.after_ns, 6u);
        }

        TEST_F(TransactionCostFixture, PreparedCountsRejectMalformedInputsAndOverflow)
        {
            std::array<std::int32_t, 4> ids{0, 1, 0, 1};
            using Prepared = moe_overlay_economy::PreparedTransactionCost;
            EXPECT_THROW((Prepared{{nullptr, 4, 2, 2, 2}, 4}), std::invalid_argument);
            EXPECT_THROW((Prepared{{ids.data(), 3, 2, 2, 2}, 4}), std::invalid_argument);
            EXPECT_THROW((Prepared{{ids.data(), 4, 2, 2, 1}, 4}), std::invalid_argument);
            ids[3] = -1;
            EXPECT_THROW((Prepared{{ids.data(), 4, 2, 2, 2}, 4}), std::invalid_argument);
            ids[3] = 1;
            const Prepared prepared({ids.data(), ids.size(), 2, 2, 2}, 4);
            EXPECT_EQ(prepared.score(placement(), scratch.data(), 1).status,
                      TransactionCostStatus::InvalidGeometry);
            before[0] = -1;
            EXPECT_EQ(prepared.score(placement(), scratch.data(), 2).status,
                      TransactionCostStatus::InvalidOwner);
            before[0] = 0;
            prices[1] = 0;
            EXPECT_EQ(prepared.score(placement(), scratch.data(), 2).status,
                      TransactionCostStatus::UnpricedParticipant);
            prices[1] = 3;
            for (const auto price : {std::numeric_limits<std::uint64_t>::max(),
                                    std::numeric_limits<std::uint64_t>::max() / 3u})
            {
                // Exercise overflow both within count*price and when summing
                // individually representable experts on the same participant.
                prices[0] = price;
                const auto compact = prepared.score(placement(), scratch.data(), 2);
                EXPECT_EQ(compact.status, score(ids.data(), ids.size(), 2).status);
                EXPECT_EQ(compact.status, TransactionCostStatus::Overflow);
                EXPECT_EQ(compact.cost.before_ns, 0u);
                EXPECT_EQ(compact.cost.after_ns, 0u);
            }
        }

        TEST_F(TransactionCostFixture, PreparedEquivalentTransactionsPreserveEveryCandidateCost)
        {
            using Prepared = moe_overlay_economy::PreparedTransactionCost;
            // Equal marginal totals are deliberately not enough to deduplicate:
            // the first two route pairs and the next two have opposite payoffs.
            const std::array<std::array<std::int32_t, 2>, 6> patterns{{
                {0, 1}, {2, 3}, {0, 3}, {1, 2}, {1, 0}, {3, 2}}};
            std::map<Prepared, std::uint64_t> groups;
            for (int token = 0; token < 384; ++token)
            {
                const auto &ids = patterns[token % patterns.size()];
                ++groups[Prepared({ids.data(), 2, 1, 2, 2}, 4)];
            }
            ASSERT_EQ(groups.size(), 4u);
            for (int candidate = 0; candidate < 16; ++candidate)
            {
                for (int expert = 0; expert < 4; ++expert)
                    after[expert] = (candidate >> expert) & 1;
                ServiceCostPair raw, grouped;
                for (int token = 0; token < 384; ++token)
                    ASSERT_EQ(appendTransaction(raw,
                        score(patterns[token % patterns.size()].data(), 2)), TransactionCostStatus::Complete);
                for (const auto &[transaction, occurrences] : groups)
                    ASSERT_EQ(appendTransaction(grouped, repeatPreparedTransaction(
                        transaction.score(placement(), scratch.data(), 2), occurrences)), TransactionCostStatus::Complete);
                EXPECT_EQ(raw.before_ns, grouped.before_ns);
                EXPECT_EQ(raw.after_ns, grouped.after_ns);
            }
        }

        TEST_F(TransactionCostFixture, PreparedRepetitionChecksOverflowAndKeepsFailureAtomic)
        {
            constexpr auto maximum = std::numeric_limits<std::uint64_t>::max();
            EXPECT_EQ(repeatPreparedTransaction({TransactionCostStatus::Complete, {1, 2}}, 0).status,
                      TransactionCostStatus::InvalidGeometry);
            EXPECT_EQ(repeatPreparedTransaction({TransactionCostStatus::InvalidOwner, {}}, 3).status,
                      TransactionCostStatus::InvalidOwner);
            ServiceCostPair window{17, 19};
            const auto repeated = repeatPreparedTransaction({TransactionCostStatus::Complete, {maximum / 2, 2}}, 3);
            EXPECT_EQ(appendTransaction(window, repeated), TransactionCostStatus::Overflow);
            EXPECT_EQ(window.before_ns, 17u);
            EXPECT_EQ(window.after_ns, 19u);
            const auto boundary = repeatPreparedTransaction({TransactionCostStatus::Complete, {1, 0}}, maximum);
            ASSERT_TRUE(boundary.complete());
            EXPECT_EQ(boundary.cost.before_ns, maximum);
            EXPECT_EQ(boundary.cost.after_ns, 0u);
        }

        TEST(PreparedTransactionSwapSearch, EveryCandidateMatchesTheFullRouteScorer)
        {
            std::mt19937 random(1709);
            for (const std::uint32_t experts : {6u, 32u, 128u})
            for (const std::uint32_t participants : {1u, 2u, 3u, 17u})
            for (const std::uint32_t rows : {1u, 3u, 16u, 512u})
            {
                const auto top_k = std::min(experts, 8u);
                const auto stride = top_k + 3u;
                std::vector<std::int32_t> ids(rows * stride, -1), owners(experts);
                std::vector<std::uint64_t> prices(participants);
                std::vector<ServiceCostPair> scratch(participants);
                for (auto &owner : owners) owner = random() % participants;
                for (auto &price : prices) price = 1u + random() % 10000u;
                for (std::uint32_t row = 0; row < rows; ++row)
                    for (std::uint32_t slot = 0; slot < top_k; ++slot)
                        ids[row * stride + slot] = random() % experts;
                const TransactionRoutes routes{ids.data(), ids.size(), rows, top_k, stride};
                const moe_overlay_economy::PreparedTransactionCost prepared(routes, experts);
                for (const bool tier_only : {false, true})
                {
                    std::vector<int> selected;
                    if (tier_only)
                        for (std::uint32_t participant = 0; participant < participants; participant += 2)
                            selected.push_back(static_cast<int>(participant));
                    const moe_overlay_economy::PreparedTransactionSwapSearch search(
                        prepared, owners, prices, selected);
                    for (int candidate = 0; candidate < 64; ++candidate)
                    {
                        const auto first = random() % experts, second = random() % experts;
                        auto after = owners;
                        std::swap(after[first], after[second]);
                        const auto raw = scoreTransaction(routes,
                            {owners.data(), after.data(), prices.data(), experts, participants},
                            scratch.data(), participants);
                        ASSERT_TRUE(raw.complete());
                        const auto actual = search.scoreSwap(first, second);
                        ASSERT_TRUE(actual.complete());
                        ServiceCostPair expected;
                        auto minimum = std::numeric_limits<std::uint64_t>::max();
                        for (std::uint32_t participant = 0; participant < participants; ++participant)
                            if (!tier_only || participant % 2 == 0)
                            {
                                expected.before_ns = std::max(expected.before_ns, scratch[participant].before_ns);
                                expected.after_ns = std::max(expected.after_ns, scratch[participant].after_ns);
                                minimum = std::min(minimum, scratch[participant].before_ns);
                            }
                        ASSERT_EQ(actual.cost.before_ns, expected.before_ns);
                        ASSERT_EQ(actual.cost.after_ns, expected.after_ns);
                        ASSERT_EQ(search.minimumBefore(), minimum);
                    }
                }
            }
        }

        TEST_F(TransactionCostFixture, PreparedSwapOwnsItsSourceAndRebuildsForTheNextRound)
        {
            using Prepared = moe_overlay_economy::PreparedTransactionCost;
            using Search = moe_overlay_economy::PreparedTransactionSwapSearch;
            const std::array<std::int32_t, 2> ids{0, 1};
            const Prepared prepared({ids.data(), 2, 1, 2, 2}, 4);
            const Search original(prepared, before, prices);
            const auto first = original.scoreSwap(0, 2);
            ASSERT_TRUE(first.complete());
            EXPECT_EQ(first.cost.before_ns, 2u);
            EXPECT_EQ(first.cost.after_ns, 3u);
            std::swap(before[0], before[2]);
            prices = {7, 11};
            const Search next(prepared, before, prices);
            const auto reverse = next.scoreSwap(0, 2);
            ASSERT_TRUE(reverse.complete());
            EXPECT_EQ(reverse.cost.before_ns, 11u);
            EXPECT_EQ(reverse.cost.after_ns, 14u);
            // Mutating the caller's map and pricing cannot alter an earlier
            // immutable search, even when the caller has accepted a swap.
            EXPECT_EQ(original.scoreSwap(0, 2).cost.before_ns, first.cost.before_ns);
            EXPECT_EQ(original.scoreSwap(0, 2).cost.after_ns, first.cost.after_ns);
            EXPECT_EQ(next.scoreSwap(0, 0).cost.after_ns, reverse.cost.before_ns);
        }

        TEST_F(TransactionCostFixture, PreparedSwapRejectsInvalidSourcesAndCandidateOverflow)
        {
            using Prepared = moe_overlay_economy::PreparedTransactionCost;
            using Search = moe_overlay_economy::PreparedTransactionSwapSearch;
            const std::array<std::int32_t, 4> ids{0, 0, 0, 2};
            const Prepared prepared({ids.data(), 4, 4, 1, 1}, 4);
            EXPECT_THROW((Search{prepared, std::span(before).first(3), prices}), std::invalid_argument);
            EXPECT_THROW((Search{prepared, before, {}}), std::invalid_argument);
            for (const auto selection : {std::array<int, 2>{0, 0}, {-1, 0}, {0, 2}})
                EXPECT_THROW((Search{prepared, before, prices, selection}), std::invalid_argument);
            before[3] = -1; // Even an expert with no demand must have a valid owner.
            EXPECT_THROW((Search{prepared, before, prices}), std::invalid_argument);
            before[3] = 1;
            prices[1] = 0;
            EXPECT_THROW((Search{prepared, before, prices}), std::invalid_argument);
            prices = {std::numeric_limits<std::uint64_t>::max(), 1};
            EXPECT_THROW((Search{prepared, before, prices}), std::overflow_error);
            prices = {1, std::numeric_limits<std::uint64_t>::max() / 2};
            const std::array<int, 1> selected{0};
            const Search search(prepared, before, prices, selected);
            EXPECT_EQ(search.scoreSwap(4, 0).status, TransactionCostStatus::InvalidExpert);
            // The unselected endpoint overflows after the swap. A tier-local
            // objective may not hide invalid work elsewhere in the topology.
            const auto overflow = search.scoreSwap(0, 2);
            EXPECT_EQ(overflow.status, TransactionCostStatus::Overflow);
            EXPECT_EQ(overflow.cost.before_ns, 0u);
            EXPECT_EQ(overflow.cost.after_ns, 0u);
        }
    }
}
