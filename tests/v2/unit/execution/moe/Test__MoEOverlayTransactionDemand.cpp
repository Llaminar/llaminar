/**
 * @file Test__MoEOverlayTransactionDemand.cpp
 * @brief Device-free exact-demand storage, closure and publication regressions.
 *
 * Storage is allocated before appends, exactly as it will be in an admitted
 * histogram bank. Tests adversarially cross the sample bound with grouped rows,
 * validate poisoned padding and late invalid routes, and replay frozen records
 * through the shared cost arithmetic. No model, device, allocator or new thread
 * lifecycle participates in the append operation.
 */
#include "execution/moe/MoEOverlayTransactionDemand.h"
#include <gtest/gtest.h>
#include <array>
#include <limits>
#include <vector>

namespace llaminar2::test
{
    namespace
    {
        using namespace moe_overlay_economy;

        /** @brief Exactly provisioned storage with a three-row measurement target. */
        class TransactionDemandFixture : public ::testing::Test
        {
        protected:
            TransactionDemandCapacity capacity{3, 4, 2};
            TransactionDemandFrontier frontier;
            std::vector<TransactionDemandRecord> transactions{capacity.target_rows};
            std::vector<std::int32_t> ids = std::vector<std::int32_t>(capacity.routeSlots(), -7);

            /** @return Borrowed storage; allocation is outside every append. */
            TransactionDemandBank bank()
            {
                return {capacity, &frontier, transactions.data(), ids.data()};
            }

            /** @return Append status for a canonical compact top-2 row prefix. */
            TransactionDemandAppendStatus append(const std::vector<std::int32_t> &routes,
                ExpertHistogramSource phase = ExpertHistogramSource::DecodeToken)
            {
                return bank().append(phase, {routes.data(), routes.size(),
                    static_cast<std::uint32_t>(routes.size() / 2), 2, 2}, 4, capacity.target_rows);
            }
        };

        TEST_F(TransactionDemandFixture, ExactBOMCoversWholeCrossingTransaction)
        {
            ASSERT_TRUE(capacity.valid());
            EXPECT_EQ(capacity.routeSlots(), 12u);
            EXPECT_EQ(capacity.allocationBytes(), sizeof(frontier) +
                transactions.size() * sizeof(TransactionDemandRecord) + ids.size() * sizeof(std::int32_t));
            ASSERT_EQ(append({0, 1}), TransactionDemandAppendStatus::Recorded);
            ASSERT_EQ(append({1, 2}), TransactionDemandAppendStatus::Recorded);
            ASSERT_EQ(append({0, 1, 2, 3, 0, 2, 1, 3}, ExpertHistogramSource::PrefillChunk),
                      TransactionDemandAppendStatus::Recorded);
            EXPECT_EQ(frontier.logical_rows, 6u);
            EXPECT_EQ(frontier.route_slots, 12u);
            EXPECT_EQ(frontier.transactions, 3u);
            EXPECT_EQ(transactions[2].logical_rows, 4u);
            EXPECT_EQ(transactions[2].first_route_slot, 4u);
            EXPECT_EQ(transactions[2].phase, ExpertHistogramSource::PrefillChunk);
            const auto sealed = ids;
            for (int subsequent = 0; subsequent < 1000; ++subsequent)
                ASSERT_EQ(append({0, 3}), TransactionDemandAppendStatus::SampleComplete);
            EXPECT_EQ(ids, sealed);
            EXPECT_EQ(frontier.transactions, 3u);
        }

        TEST_F(TransactionDemandFixture, NoPartialMutationOnInvalidLateRouteOrGeometry)
        {
            ASSERT_EQ(append({0, 1}), TransactionDemandAppendStatus::Recorded);
            const auto unchanged = ids;
            EXPECT_EQ(append({0, 1, 2, 4}, ExpertHistogramSource::PrefillChunk),
                      TransactionDemandAppendStatus::InvalidExpert);
            EXPECT_EQ(ids, unchanged);
            EXPECT_EQ(frontier.logical_rows, 1u);
            EXPECT_EQ(frontier.transactions, 1u);
            EXPECT_EQ(append({0, 1, 2, 3, 0, 1, 2, 3, 0, 1}, ExpertHistogramSource::PrefillChunk),
                      TransactionDemandAppendStatus::InvalidGeometry);
            EXPECT_EQ(append({0, 1}, static_cast<ExpertHistogramSource>(77)),
                      TransactionDemandAppendStatus::InvalidPhase);
            EXPECT_EQ(ids, unchanged);
        }

        TEST_F(TransactionDemandFixture, PaddingIsNotDemandAndPhaseDoesNotComeFromRowCount)
        {
            const std::array<std::int32_t, 6> padded{0, 1, -1, 2, 3, -1};
            ASSERT_EQ(bank().append(ExpertHistogramSource::GroupedVerifier,
                {padded.data(), padded.size(), 2, 2, 3}, 4, capacity.target_rows), TransactionDemandAppendStatus::Recorded);
            EXPECT_EQ((std::vector<std::int32_t>(ids.begin(), ids.begin() + 4)),
                      (std::vector<std::int32_t>{0, 1, 2, 3}));
            EXPECT_EQ(transactions[0].logical_rows, 2u);
            EXPECT_EQ(transactions[0].phase, ExpertHistogramSource::GroupedVerifier);
            EXPECT_EQ(bank().append(ExpertHistogramSource::DecodeToken,
                {padded.data(), padded.size(), 2, 2, 3}, 4, capacity.target_rows), TransactionDemandAppendStatus::InvalidGeometry);
        }

        TEST_F(TransactionDemandFixture, AdaptiveBudgetUsesExistingCapacityWithoutASecondLimitState)
        {
            const std::array<std::int32_t, 2> route{0, 1};
            const TransactionRoutes rows{route.data(), route.size(), 1, 2, 2};
            ASSERT_EQ(bank().append(ExpertHistogramSource::DecodeToken, rows, 4, 1),
                      TransactionDemandAppendStatus::Recorded);
            EXPECT_EQ(bank().append(ExpertHistogramSource::DecodeToken, rows, 4, 1),
                      TransactionDemandAppendStatus::SampleComplete);
            ASSERT_EQ(bank().append(ExpertHistogramSource::DecodeToken, rows, 4, 3),
                      TransactionDemandAppendStatus::Recorded);
            EXPECT_EQ(frontier.logical_rows, 2u);
            EXPECT_EQ(bank().append(ExpertHistogramSource::DecodeToken, rows, 4, 4),
                      TransactionDemandAppendStatus::InvalidGeometry);
            EXPECT_EQ(frontier.logical_rows, 2u);
        }

        TEST_F(TransactionDemandFixture, RetiredResetReusesExactStorageWithoutRebinding)
        {
            auto *const route_address = ids.data();
            auto *const descriptor_address = transactions.data();
            for (int generation = 0; generation < 20; ++generation)
            {
                // The test stands in for the parent's retired-bank ownership.
                // There is intentionally no independent reset lifecycle API.
                frontier = {};
                ASSERT_EQ(append({0, 3}), TransactionDemandAppendStatus::Recorded);
                ASSERT_EQ(append({1, 2}), TransactionDemandAppendStatus::Recorded);
                EXPECT_EQ(frontier.transactions, 2u);
                EXPECT_EQ(ids.data(), route_address);
                EXPECT_EQ(transactions.data(), descriptor_address);
            }
        }

        TEST_F(TransactionDemandFixture, RecordedBoundariesDriveOppositeCounterfactualCosts)
        {
            const std::array<std::int32_t, 4> before{0, 0, 1, 1}, after{1, 0, 0, 1};
            const std::array<std::uint64_t, 2> prices{1, 3};
            std::array<ServiceCostPair, 2> scratch;
            for (bool profitable : {true, false})
            {
                frontier = {};
                ASSERT_EQ(append(profitable ? std::vector<std::int32_t>{0, 1} : std::vector<std::int32_t>{0, 3}),
                          TransactionDemandAppendStatus::Recorded);
                ASSERT_EQ(append(profitable ? std::vector<std::int32_t>{2, 3} : std::vector<std::int32_t>{1, 2}),
                          TransactionDemandAppendStatus::Recorded);
                ServiceCostPair total;
                std::array<unsigned, 4> marginals{};
                for (std::uint32_t t = 0; t < frontier.transactions; ++t)
                {
                    const auto &record = transactions[t];
                    const auto count = record.logical_rows * record.top_k;
                    const auto *routes = ids.data() + record.first_route_slot;
                    for (unsigned i = 0; i < count; ++i) ++marginals[routes[i]];
                    ASSERT_EQ(appendTransaction(total, scoreTransaction(
                        {routes, count, record.logical_rows, record.top_k, record.top_k},
                        {before.data(), after.data(), prices.data(), 4, 2}, scratch.data(), 2)),
                        TransactionCostStatus::Complete);
                }
                EXPECT_EQ(marginals, (std::array<unsigned, 4>{1, 1, 1, 1}));
                EXPECT_EQ(total.before_ns, profitable ? 8u : 6u);
                EXPECT_EQ(total.after_ns, profitable ? 6u : 8u);
            }
        }

        TEST_F(TransactionDemandFixture, CorruptFrontierFailsRatherThanAppearingComplete)
        {
            frontier.route_slots = capacity.routeSlots() + 1;
            EXPECT_EQ(append({0, 1}), TransactionDemandAppendStatus::InvalidFrontier);
            frontier = {.logical_rows = 1, .transactions = 0};
            EXPECT_EQ(append({0, 1}), TransactionDemandAppendStatus::InvalidFrontier);
        }

        TEST(TransactionDemandCapacity, RejectsZeroAndUnrepresentablePhysicalBOM)
        {
            EXPECT_FALSE((TransactionDemandCapacity{0, 1, 2}.valid()));
            EXPECT_FALSE((TransactionDemandCapacity{1, 0, 2}.valid()));
            EXPECT_FALSE((TransactionDemandCapacity{1, 1, 0}.valid()));
            constexpr auto maximum = std::numeric_limits<std::uint32_t>::max();
            EXPECT_FALSE((TransactionDemandCapacity{maximum, maximum, maximum}.valid()));
            EXPECT_FALSE((TransactionDemandCapacity{maximum, 1, maximum}.valid()));
            EXPECT_EQ((TransactionDemandCapacity{maximum, maximum, maximum}.allocationBytes()), 0u);
        }
    }
}
