#include "execution/mtp/MTPRejectionSampler.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

using ::testing::ElementsAre;

namespace llaminar2::test
{
    namespace
    {
        std::vector<SamplingDistributionEntry> dist(
            std::initializer_list<SamplingDistributionEntry> entries)
        {
            return std::vector<SamplingDistributionEntry>(entries);
        }
    } // namespace

    TEST(Test__MTPRejectionSampler, AcceptsDraftWhenThresholdIsBelowProbability)
    {
        const auto target = dist({{7, 0.2f}, {9, 0.8f}});
        const auto draft = dist({{7, 0.5f}, {9, 0.5f}});

        MTPRejectionSampleRowResult result =
            sampleMTPRejectionRowFromDistributions(
                target,
                draft,
                /*draft_token=*/7,
                /*accept_threshold=*/0.1f,
                /*residual_threshold=*/0.9f);

        ASSERT_TRUE(result.ok) << result.error;
        EXPECT_TRUE(result.accepted);
        EXPECT_EQ(result.token, 7);
        EXPECT_FLOAT_EQ(result.accept_probability, 0.4f);
    }

    TEST(Test__MTPRejectionSampler, SamplesResidualTokenAfterReject)
    {
        const auto target = dist({{1, 0.6f}, {2, 0.4f}});
        const auto draft = dist({{1, 0.9f}, {2, 0.1f}});

        MTPRejectionSampleRowResult result =
            sampleMTPRejectionRowFromDistributions(
                target,
                draft,
                /*draft_token=*/1,
                /*accept_threshold=*/0.9f,
                /*residual_threshold=*/0.0f);

        ASSERT_TRUE(result.ok) << result.error;
        EXPECT_FALSE(result.accepted);
        EXPECT_EQ(result.token, 2);
        EXPECT_NEAR(result.accept_probability, 0.6f / 0.9f, 1e-6f);
    }

    TEST(Test__MTPRejectionSampler, SamplesDistributionWithClampedThreshold)
    {
        const auto distribution = dist({{3, 0.25f}, {4, 0.75f}});

        EXPECT_EQ(sampleMTPDistributionWithThreshold(distribution, -1.0f), 3);
        EXPECT_EQ(sampleMTPDistributionWithThreshold(distribution, 2.0f), 4);
    }

    TEST(Test__MTPRejectionSampler, BuildsAcceptAllCatchupWithBonusReadyToken)
    {
        MTPDecodeCatchupGreedyRequest request;
        request.draft_tokens = {10, 11, 12};

        std::vector<MTPRejectionSampleRowResult> rows;
        rows.push_back({true, "", 11, 11, true, 1.0f, 0.0f});
        rows.push_back({true, "", 12, 12, true, 1.0f, 0.0f});

        MTPDecodeCatchupGreedyResult result =
            buildAllPositionMTPDecodeCatchupStochasticResult(
                request,
                rows,
                /*bonus_ready_token=*/99);

        ASSERT_TRUE(result.ok) << result.error;
        EXPECT_TRUE(result.all_speculative_accepted);
        EXPECT_THAT(result.accepted_tokens, ElementsAre(10, 11, 12));
        EXPECT_THAT(result.verifier_tokens, ElementsAre(11, 12));
        EXPECT_EQ(result.accepted_speculative_prefix, 2);
        EXPECT_EQ(result.target_verifier_state_commit_count, 3);
        EXPECT_EQ(result.ready_token, 99);
    }

    TEST(Test__MTPRejectionSampler, SummarizesAcceptAllBatchForDeviceContract)
    {
        MTPDecodeCatchupGreedyRequest request;
        request.draft_tokens = {10, 11, 12};

        std::vector<MTPRejectionSampleRowResult> rows;
        rows.push_back({true, "", 11, 11, true, 1.0f, 0.0f});
        rows.push_back({true, "", 12, 12, true, 1.0f, 0.0f});

        MTPRejectionBatchOutcome outcome =
            summarizeAllPositionMTPRejectionBatch(
                request,
                rows,
                /*bonus_ready_token=*/99);

        ASSERT_TRUE(outcome.ok) << outcome.error;
        EXPECT_THAT(outcome.output_tokens, ElementsAre(10, 11, 12));
        EXPECT_THAT(outcome.verifier_tokens, ElementsAre(11, 12));
        EXPECT_EQ(outcome.consumed_verifier_rows, 2);
        EXPECT_EQ(outcome.accepted_speculative_prefix, 2);
        EXPECT_EQ(outcome.target_verifier_state_commit_count, 3);
        EXPECT_EQ(outcome.ready_token, 99);
        EXPECT_TRUE(outcome.sampled_terminal);
    }

    TEST(Test__MTPRejectionSampler, BuildsRejectCatchupWithAcceptedStatePrefix)
    {
        MTPDecodeCatchupGreedyRequest request;
        request.draft_tokens = {10, 11, 12};

        std::vector<MTPRejectionSampleRowResult> rows;
        rows.push_back({true, "", 11, 11, true, 1.0f, 0.0f});
        rows.push_back({true, "", 12, 77, false, 0.2f, 0.9f});

        MTPDecodeCatchupGreedyResult result =
            buildAllPositionMTPDecodeCatchupStochasticResult(request, rows);

        ASSERT_TRUE(result.ok) << result.error;
        EXPECT_FALSE(result.all_speculative_accepted);
        EXPECT_THAT(result.accepted_tokens, ElementsAre(10, 11, 77));
        EXPECT_THAT(result.verifier_tokens, ElementsAre(11, 77));
        EXPECT_EQ(result.accepted_speculative_prefix, 1);
        EXPECT_EQ(result.target_verifier_state_commit_count, 2);
        EXPECT_EQ(result.rejected_verified_token, 77);
        EXPECT_EQ(result.ready_token, -1);
    }

    TEST(Test__MTPRejectionSampler, SummarizesRejectBatchForDeviceContract)
    {
        MTPDecodeCatchupGreedyRequest request;
        request.draft_tokens = {10, 11, 12};

        std::vector<MTPRejectionSampleRowResult> rows;
        rows.push_back({true, "", 11, 11, true, 1.0f, 0.0f});
        rows.push_back({true, "", 12, 77, false, 0.2f, 0.9f});

        MTPRejectionBatchOutcome outcome =
            summarizeAllPositionMTPRejectionBatch(request, rows);

        ASSERT_TRUE(outcome.ok) << outcome.error;
        EXPECT_FALSE(outcome.all_speculative_accepted);
        EXPECT_THAT(outcome.output_tokens, ElementsAre(10, 11, 77));
        EXPECT_THAT(outcome.verifier_tokens, ElementsAre(11, 77));
        EXPECT_EQ(outcome.consumed_verifier_rows, 2);
        EXPECT_EQ(outcome.accepted_speculative_prefix, 1);
        EXPECT_EQ(outcome.target_verifier_state_commit_count, 2);
        EXPECT_EQ(outcome.rejected_verified_token, 77);
        EXPECT_EQ(outcome.ready_token, -1);
        EXPECT_FALSE(outcome.sampled_terminal);
    }

    TEST(Test__MTPRejectionSampler, StopsOnFirstTokenWithoutRows)
    {
        MTPDecodeCatchupGreedyRequest request;
        request.draft_tokens = {10, 11, 12};
        request.stop_tokens = {10};

        MTPDecodeCatchupGreedyResult result =
            buildAllPositionMTPDecodeCatchupStochasticResult(request, {});

        ASSERT_TRUE(result.ok) << result.error;
        EXPECT_TRUE(result.stopped_on_output);
        EXPECT_THAT(result.accepted_tokens, ElementsAre(10));
        EXPECT_EQ(result.target_verifier_state_commit_count, 1);
    }

    TEST(Test__MTPRejectionSampler, FailsWhenAcceptedRowReturnsDifferentToken)
    {
        MTPDecodeCatchupGreedyRequest request;
        request.draft_tokens = {10, 11};

        std::vector<MTPRejectionSampleRowResult> rows;
        rows.push_back({true, "", 11, 42, true, 1.0f, 0.0f});

        MTPDecodeCatchupGreedyResult result =
            buildAllPositionMTPDecodeCatchupStochasticResult(request, rows);

        EXPECT_FALSE(result.ok);
        EXPECT_THAT(result.error, testing::HasSubstr("accepted stochastic verifier row"));
    }

    TEST(Test__MTPRejectionSampler, FailsWhenRowsEndBeforeDecision)
    {
        MTPDecodeCatchupGreedyRequest request;
        request.draft_tokens = {10, 11, 12};

        std::vector<MTPRejectionSampleRowResult> rows;
        rows.push_back({true, "", 11, 11, true, 1.0f, 0.0f});

        MTPDecodeCatchupGreedyResult result =
            buildAllPositionMTPDecodeCatchupStochasticResult(request, rows);

        EXPECT_FALSE(result.ok);
        EXPECT_THAT(result.error, testing::HasSubstr("ended before"));
    }

} // namespace llaminar2::test
