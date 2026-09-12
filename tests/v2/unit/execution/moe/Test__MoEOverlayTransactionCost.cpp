/**
 * @file Test__MoEOverlayTransactionCost.cpp
 * @brief Device-free exact routing-cohort economics and malformed-input regressions.
 *
 * These tests distinguish sequential token latency from window-aggregate work,
 * exercise true grouped transaction boundaries, and prove arbitrary participant
 * geometry without loading a model or touching CUDA/HIP. The implementation is
 * shared with device callers; real graph integration remains a separate gate.
 */
#include "execution/moe/MoEOverlayTransactionCost.h"

#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <numeric>
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
    }
}
