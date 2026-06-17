#include "execution/mtp/MTPRejectionSampler.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <limits>

using ::testing::ElementsAre;
using ::testing::Contains;

namespace llaminar2::test
{
    namespace
    {
        std::vector<SamplingDistributionEntry> dist(
            std::initializer_list<SamplingDistributionEntry> entries)
        {
            return std::vector<SamplingDistributionEntry>(entries);
        }

        std::vector<float> logitsFromProbabilities(
            std::initializer_list<float> probabilities)
        {
            std::vector<float> logits;
            logits.reserve(probabilities.size());
            for (float p : probabilities)
            {
                logits.push_back(
                    p > 0.0f
                        ? std::log(p)
                        : -std::numeric_limits<float>::infinity());
            }
            return logits;
        }

        void append(std::vector<float> &rows, const std::vector<float> &row)
        {
            rows.insert(rows.end(), row.begin(), row.end());
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

    TEST(Test__MTPRejectionSampler, SharedDistributionSamplerReportsSelectedProbability)
    {
        const int token_ids[] = {7, 11, 13, -1};
        const float probs[] = {0.2f, 0.3f, 0.5f, 0.0f};

        float selected_probability = -1.0f;
        const int token =
            sampling_math::sample_distribution_with_threshold_and_probability(
                token_ids,
                probs,
                4,
                0.45f,
                &selected_probability);

        EXPECT_EQ(token, 11);
        EXPECT_FLOAT_EQ(selected_probability, 0.3f);
    }

    TEST(Test__MTPRejectionSampler, SharedCompactVerifierSupportsOneHotGreedyDraft)
    {
        const int target_ids[] = {5, 7, 11, 13};
        const float target_probs[] = {0.20f, 0.30f, 0.10f, 0.40f};

        int token = -1;
        int accepted = -1;
        float accept_probability = -1.0f;
        float accept_threshold = -1.0f;

        sampling_math::speculative_verify_with_thresholds_one_hot_draft(
            target_ids,
            target_probs,
            4,
            /*draft_token=*/7,
            /*accept_threshold=*/0.30f,
            /*residual_threshold=*/0.0f,
            &token,
            &accepted,
            &accept_probability,
            &accept_threshold);

        EXPECT_EQ(token, 7);
        EXPECT_EQ(accepted, 1);
        EXPECT_FLOAT_EQ(accept_probability, 0.30f);
        EXPECT_FLOAT_EQ(accept_threshold, 0.30f)
            << "one-hot verifier follows the vLLM <= acceptance convention";

        sampling_math::speculative_verify_with_thresholds_one_hot_draft(
            target_ids,
            target_probs,
            4,
            /*draft_token=*/7,
            /*accept_threshold=*/0.95f,
            /*residual_threshold=*/0.0f,
            &token,
            &accepted,
            &accept_probability,
            &accept_threshold);

        EXPECT_EQ(token, 5)
            << "rejection residual removes the one-hot draft token and samples the first remaining mass";
        EXPECT_EQ(accepted, 0);
        EXPECT_FLOAT_EQ(accept_probability, 0.30f);
        EXPECT_FLOAT_EQ(accept_threshold, 0.95f);
    }

    TEST(Test__MTPRejectionSampler, SharedCompactVerifierSupportsVLLMRecoveredTokenSampling)
    {
        const int target_ids[] = {5, 7, 11, 13};
        const float target_probs[] = {0.20f, 0.30f, 0.10f, 0.40f};

        int token = -1;
        int accepted = -1;
        float accept_probability = -1.0f;
        float accept_threshold = -1.0f;
        sampling_math::speculative_verify_with_thresholds_one_hot_draft_vllm_recovered(
            target_ids,
            target_probs,
            4,
            /*vocab_size=*/32,
            /*draft_token=*/7,
            /*accept_threshold=*/0.95f,
            /*inverse_sample_seed=*/98765,
            /*logical_position=*/23,
            &token,
            &accepted,
            &accept_probability,
            &accept_threshold);

        EXPECT_EQ(accepted, 0);
        EXPECT_NE(token, 7)
            << "vLLM recovered sampling removes the one-hot draft proposal on reject";
        EXPECT_THAT(std::vector<int>({5, 11, 13}), ::testing::Contains(token));
        EXPECT_FLOAT_EQ(accept_probability, 0.30f);
        EXPECT_FLOAT_EQ(accept_threshold, 0.95f);
    }

    TEST(Test__MTPRejectionSampler, ComputesProcessedFullLogitStats)
    {
        const float neg_inf = -std::numeric_limits<float>::infinity();
        const std::vector<float> logits = {neg_inf, 2.0f, 2.0f, 0.0f};

        MTPFullLogitRowStats stats =
            computeMTPFullLogitRowStats(logits.data(), logits.size());

        ASSERT_TRUE(stats.ok) << stats.error;
        EXPECT_EQ(stats.argmax_token, 1)
            << "equal logits must prefer the lower token id for parity";
        EXPECT_FLOAT_EQ(stats.max_logit, 2.0f);
        EXPECT_GT(stats.exp_sum, 2.0);
        EXPECT_EQ(probabilityFromMTPFullLogits(
                      logits.data(), logits.size(), stats, 0),
                  0.0f);
    }

    TEST(Test__MTPRejectionSampler, ProcessedFullLogitsAcceptByDraftTokenProbability)
    {
        const std::vector<float> target =
            logitsFromProbabilities({0.2f, 0.8f});
        const std::vector<float> draft =
            logitsFromProbabilities({0.5f, 0.5f});

        MTPRejectionSampleRowResult accepted =
            sampleMTPRejectionRowFromProcessedLogits(
                target.data(),
                draft.data(),
                /*vocab_size=*/2,
                /*draft_token=*/0,
                /*accept_threshold=*/0.39f,
                /*residual_threshold=*/0.0f);

        ASSERT_TRUE(accepted.ok) << accepted.error;
        EXPECT_TRUE(accepted.accepted);
        EXPECT_EQ(accepted.token, 0);
        EXPECT_NEAR(accepted.accept_probability, 0.4f, 1e-6f);

        MTPRejectionSampleRowResult rejected =
            sampleMTPRejectionRowFromProcessedLogits(
                target.data(),
                draft.data(),
                /*vocab_size=*/2,
                /*draft_token=*/0,
                /*accept_threshold=*/0.41f,
                /*residual_threshold=*/0.0f);

        ASSERT_TRUE(rejected.ok) << rejected.error;
        EXPECT_FALSE(rejected.accepted);
        EXPECT_EQ(rejected.token, 1)
            << "residual max(target - draft, 0) should pick token 1";
    }

    TEST(Test__MTPRejectionSampler, ProcessedFullLogitResidualCanSampleAnyVocabToken)
    {
        const std::vector<float> target =
            logitsFromProbabilities({0.1f, 0.1f, 0.1f, 0.7f});
        const std::vector<float> draft =
            logitsFromProbabilities({0.8f, 0.1f, 0.1f, 0.0f});

        MTPRejectionSampleRowResult result =
            sampleMTPRejectionRowFromProcessedLogits(
                target.data(),
                draft.data(),
                /*vocab_size=*/4,
                /*draft_token=*/0,
                /*accept_threshold=*/0.99f,
                /*residual_threshold=*/0.0f);

        ASSERT_TRUE(result.ok) << result.error;
        EXPECT_FALSE(result.accepted);
        EXPECT_EQ(result.token, 3)
            << "full-logit residual sampling must not be limited to compact top-k rows";
    }

    TEST(Test__MTPRejectionSampler, FullProbabilitiesUseVLLMRecoveredTokenArgmax)
    {
        const std::array<float, 4> target = {0.10f, 0.20f, 0.60f, 0.10f};
        const std::array<float, 4> draft = {0.40f, 0.10f, 0.20f, 0.30f};
        const std::array<float, 4> inverse_samples = {100.0f, 1.0f, 0.5f, 20.0f};

        const int32_t recovered = sampleMTPRecoveredTokenFromProbabilities(
            target.data(),
            draft.data(),
            inverse_samples.data(),
            target.size(),
            /*draft_token=*/0);

        EXPECT_EQ(recovered, 2)
            << "vLLM recovered sampling chooses max((p-q)*inv_q), not the largest raw inv_q";

        MTPRejectionSampleRowResult rejected =
            sampleMTPRejectionRowFromProbabilities(
                target.data(),
                draft.data(),
                inverse_samples.data(),
                target.size(),
                /*draft_token=*/0,
                /*accept_threshold=*/0.99f);

        ASSERT_TRUE(rejected.ok) << rejected.error;
        EXPECT_FALSE(rejected.accepted);
        EXPECT_EQ(rejected.token, 2);
        EXPECT_NEAR(rejected.accept_probability, 0.25f, 1e-6f);
    }

    TEST(Test__MTPRejectionSampler, FullProbabilitiesAcceptAtEqualThresholdLikeVLLM)
    {
        const std::array<float, 2> target = {0.2f, 0.8f};
        const std::array<float, 2> draft = {0.5f, 0.5f};
        const std::array<float, 2> inverse_samples = {1.0f, 1.0f};

        MTPRejectionSampleRowResult result =
            sampleMTPRejectionRowFromProbabilities(
                target.data(),
                draft.data(),
                inverse_samples.data(),
                target.size(),
                /*draft_token=*/0,
                /*accept_threshold=*/0.4f);

        ASSERT_TRUE(result.ok) << result.error;
        EXPECT_TRUE(result.accepted)
            << "vLLM accepts when p/q >= uniform";
        EXPECT_EQ(result.token, 0);
    }

    TEST(Test__MTPRejectionSampler, FullProbabilitiesSupportNoDraftProbabilitiesMode)
    {
        const std::array<float, 4> target = {0.50f, 0.20f, 0.20f, 0.10f};
        const std::array<float, 4> inverse_samples = {100.0f, 1.0f, 5.0f, 1.0f};

        MTPRejectionSampleRowResult rejected =
            sampleMTPRejectionRowFromProbabilities(
                target.data(),
                /*draft_probabilities=*/nullptr,
                inverse_samples.data(),
                target.size(),
                /*draft_token=*/0,
                /*accept_threshold=*/0.99f,
                /*no_draft_probabilities=*/true);

        ASSERT_TRUE(rejected.ok) << rejected.error;
        EXPECT_FALSE(rejected.accepted);
        EXPECT_EQ(rejected.token, 2)
            << "no-draft-probs mode removes only the sampled draft token";
        EXPECT_NEAR(rejected.accept_probability, 0.5f, 1e-6f);
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

    TEST(Test__MTPRejectionSampler, SummarizesProcessedFullLogitAcceptAllBatch)
    {
        constexpr int vocab_size = 16;
        MTPDecodeCatchupGreedyRequest request;
        request.draft_tokens = {10, 11, 12};

        std::vector<float> target_rows;
        std::vector<float> draft_rows;
        append(target_rows, logitsFromProbabilities(
                                {0.01f, 0.01f, 0.01f, 0.01f,
                                 0.01f, 0.01f, 0.01f, 0.01f,
                                 0.01f, 0.01f, 0.01f, 0.90f,
                                 0.01f, 0.0f, 0.0f, 0.0f}));
        append(draft_rows, logitsFromProbabilities(
                               {0.01f, 0.01f, 0.01f, 0.01f,
                                0.01f, 0.01f, 0.01f, 0.01f,
                                0.01f, 0.01f, 0.01f, 0.90f,
                                0.01f, 0.0f, 0.0f, 0.0f}));
        append(target_rows, logitsFromProbabilities(
                                {0.01f, 0.01f, 0.01f, 0.01f,
                                 0.01f, 0.01f, 0.01f, 0.01f,
                                 0.01f, 0.01f, 0.01f, 0.01f,
                                 0.90f, 0.0f, 0.0f, 0.0f}));
        append(draft_rows, logitsFromProbabilities(
                               {0.01f, 0.01f, 0.01f, 0.01f,
                                0.01f, 0.01f, 0.01f, 0.01f,
                                0.01f, 0.01f, 0.01f, 0.01f,
                                0.90f, 0.0f, 0.0f, 0.0f}));
        const std::vector<float> bonus =
            logitsFromProbabilities({0.01f, 0.01f, 0.01f, 0.01f,
                                     0.01f, 0.01f, 0.01f, 0.90f,
                                     0.01f, 0.01f, 0.01f, 0.01f,
                                     0.0f, 0.0f, 0.0f, 0.0f});

        MTPRejectionBatchOutcome outcome =
            summarizeAllPositionMTPRejectionBatchFromProcessedLogits(
                request,
                target_rows.data(),
                draft_rows.data(),
                /*verifier_row_count=*/2,
                vocab_size,
                vocab_size,
                vocab_size,
                /*accept_thresholds=*/{0.0f, 0.0f},
                /*residual_thresholds=*/{0.0f, 0.0f},
                bonus.data(),
                /*bonus_threshold=*/0.5f);

        ASSERT_TRUE(outcome.ok) << outcome.error;
        EXPECT_TRUE(outcome.all_speculative_accepted);
        EXPECT_THAT(outcome.output_tokens, ElementsAre(10, 11, 12));
        EXPECT_EQ(outcome.accepted_speculative_prefix, 2);
        EXPECT_EQ(outcome.target_verifier_state_commit_count, 3);
        EXPECT_EQ(outcome.ready_token, 7);
        EXPECT_TRUE(outcome.sampled_terminal);
    }

    TEST(Test__MTPRejectionSampler, BuildsCatchupFromDeviceAcceptAllOutcome)
    {
        MTPDecodeCatchupGreedyRequest request;
        request.draft_tokens = {10, 11, 12};

        MTPDeviceRejectionBatchOutcome outcome;
        outcome.ok = true;
        outcome.output_tokens[0] = 10;
        outcome.output_tokens[1] = 11;
        outcome.output_tokens[2] = 12;
        outcome.output_token_count = 3;
        outcome.accepted_speculative_prefix = 2;
        outcome.target_verifier_state_commit_count = 3;
        outcome.ready_token = 99;
        outcome.all_speculative_accepted = true;
        outcome.consumed_verifier_rows = 2;
        outcome.sampled_terminal = true;

        MTPDecodeCatchupGreedyResult result =
            buildAllPositionMTPDecodeCatchupFromDeviceBatchOutcome(
                request,
                outcome);

        ASSERT_TRUE(result.ok) << result.error;
        EXPECT_TRUE(result.all_speculative_accepted);
        EXPECT_THAT(result.accepted_tokens, ElementsAre(10, 11, 12));
        EXPECT_THAT(result.verifier_tokens, ElementsAre(11, 12));
        EXPECT_EQ(result.accepted_speculative_prefix, 2);
        EXPECT_EQ(result.target_verifier_state_commit_count, 3);
        EXPECT_EQ(result.ready_token, 99);
        EXPECT_THAT(result.debug_trace,
                    testing::HasSubstr("device_stochastic_rows=2"));
    }

    TEST(Test__MTPRejectionSampler, GreedyDeviceSummaryMatchesAllPositionCatchup)
    {
        using namespace sampling_math;

        MTPDecodeCatchupGreedyRequest request;
        request.draft_tokens = {10, 11, 12};

        const int verifier_tokens[] = {
            11, 77, 99}; // accept first draft, reject second, bonus ignored
        const int draft_tokens[] = {10, 11, 12};
        int output_tokens[kSpeculativeBatchMaxOutputTokens] = {};
        int meta[kSpeculativeBatchMetaCount] = {};

        summarize_greedy_speculative_verify_batch(
            /*first_token=*/10,
            verifier_tokens,
            draft_tokens,
            /*compare_row_count=*/2,
            /*stop_tokens=*/nullptr,
            /*stop_token_count=*/0,
            output_tokens,
            meta);

        ASSERT_EQ(meta[kSpecBatchMetaOk], 1);
        MTPDeviceRejectionBatchOutcome outcome;
        outcome.ok = true;
        for (int i = 0; i < kSpeculativeBatchMaxOutputTokens; ++i)
            outcome.output_tokens[static_cast<size_t>(i)] = output_tokens[i];
        outcome.output_token_count = meta[kSpecBatchMetaOutputCount];
        outcome.accepted_speculative_prefix =
            meta[kSpecBatchMetaAcceptedSpeculativePrefix];
        outcome.target_verifier_state_commit_count =
            meta[kSpecBatchMetaTargetVerifierStateCommitCount];
        outcome.ready_token = meta[kSpecBatchMetaReadyToken];
        outcome.rejected_verified_token =
            meta[kSpecBatchMetaRejectedVerifiedToken];
        outcome.stopped_on_output = meta[kSpecBatchMetaStoppedOnOutput] != 0;
        outcome.all_speculative_accepted =
            meta[kSpecBatchMetaAllSpeculativeAccepted] != 0;
        outcome.consumed_verifier_rows =
            meta[kSpecBatchMetaConsumedVerifierRows];
        outcome.sampled_terminal = meta[kSpecBatchMetaSampledTerminal] != 0;

        MTPDecodeCatchupGreedyResult device_result =
            buildAllPositionMTPDecodeCatchupFromDeviceBatchOutcome(
                request,
                outcome);
        MTPDecodeCatchupGreedyResult row_result =
            buildAllPositionMTPDecodeCatchupGreedyResult(
                request,
                /*sampled_verifier_rows=*/{11, 77, 99});

        ASSERT_TRUE(device_result.ok) << device_result.error;
        ASSERT_TRUE(row_result.ok) << row_result.error;
        EXPECT_THAT(device_result.accepted_tokens, ElementsAre(10, 11, 77));
        EXPECT_EQ(device_result.accepted_speculative_prefix,
                  row_result.accepted_speculative_prefix);
        EXPECT_EQ(device_result.target_verifier_state_commit_count,
                  row_result.target_verifier_state_commit_count);
        EXPECT_EQ(device_result.ready_token, row_result.ready_token);
        EXPECT_EQ(device_result.rejected_verified_token,
                  row_result.rejected_verified_token);
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

    TEST(Test__MTPRejectionSampler, BuildsCatchupFromDeviceRejectOutcome)
    {
        MTPDecodeCatchupGreedyRequest request;
        request.draft_tokens = {10, 11, 12};

        MTPDeviceRejectionBatchOutcome outcome;
        outcome.ok = true;
        outcome.output_tokens[0] = 10;
        outcome.output_tokens[1] = 11;
        outcome.output_tokens[2] = 77;
        outcome.output_token_count = 3;
        outcome.accepted_speculative_prefix = 1;
        outcome.target_verifier_state_commit_count = 2;
        outcome.ready_token = -1;
        outcome.rejected_verified_token = 77;
        outcome.all_speculative_accepted = false;
        outcome.consumed_verifier_rows = 2;

        MTPDecodeCatchupGreedyResult result =
            buildAllPositionMTPDecodeCatchupFromDeviceBatchOutcome(
                request,
                outcome);

        ASSERT_TRUE(result.ok) << result.error;
        EXPECT_FALSE(result.all_speculative_accepted);
        EXPECT_THAT(result.accepted_tokens, ElementsAre(10, 11, 77));
        EXPECT_THAT(result.verifier_tokens, ElementsAre(11, 77));
        EXPECT_EQ(result.accepted_speculative_prefix, 1);
        EXPECT_EQ(result.target_verifier_state_commit_count, 2);
        EXPECT_EQ(result.rejected_verified_token, 77);
        EXPECT_EQ(result.ready_token, -1);
    }

    TEST(Test__MTPRejectionSampler, RejectsInvalidDeviceOutcome)
    {
        MTPDecodeCatchupGreedyRequest request;
        request.draft_tokens = {10, 11};

        MTPDeviceRejectionBatchOutcome outcome;
        outcome.ok = true;
        outcome.output_token_count = 0;

        MTPDecodeCatchupGreedyResult result =
            buildAllPositionMTPDecodeCatchupFromDeviceBatchOutcome(
                request,
                outcome);

        EXPECT_FALSE(result.ok);
        EXPECT_THAT(result.error, testing::HasSubstr("token count"));

        outcome.output_token_count = 3;
        result = buildAllPositionMTPDecodeCatchupFromDeviceBatchOutcome(
            request,
            outcome);

        EXPECT_FALSE(result.ok);
        EXPECT_THAT(result.error, testing::HasSubstr("more tokens"));
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
