#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "execution/mtp/MTPSpecDecodeTransaction.h"

#include <optional>

using namespace llaminar2;
using namespace testing;

TEST(Test__MTPSpecDecodeTransaction, AcceptsAllDraftsAndSelectsBonusRow)
{
    MTPSpecDecodeRequest request;
    request.request_id = 7;
    request.vocab_size = 100;
    request.draft_tokens = {11, 12, 13};
    request.sampled_tokens = {11, 12, 13, 42};

    MTPSpecDecodeTransaction tx = buildMTPSpecDecodeTransaction(request);

    ASSERT_TRUE(tx.ok) << tx.error;
    EXPECT_EQ(tx.request_id, 7);
    EXPECT_EQ(tx.draft_count, 3);
    EXPECT_EQ(tx.target_query_len, 4);
    EXPECT_EQ(tx.valid_sampled_count, 4);
    EXPECT_EQ(tx.accepted_speculative_prefix, 3);
    EXPECT_EQ(tx.rejected_token_count, 0);
    EXPECT_EQ(tx.token_index_to_sample, 3);
    EXPECT_EQ(tx.next_condition_token, 42);
    EXPECT_TRUE(tx.allDraftsAccepted());
    EXPECT_THAT(tx.committed_output_tokens, ElementsAre(11, 12, 13, 42));
    EXPECT_THAT(tx.rejected_draft_tokens, IsEmpty());
}

TEST(Test__MTPSpecDecodeTransaction, RejectsDraftSuffixAfterAcceptedPrefix)
{
    MTPSpecDecodeRequest request;
    request.vocab_size = 100;
    request.draft_tokens = {11, 12, 13};
    request.sampled_tokens = {11, 77, kMTPSpecDecodeInvalidToken, kMTPSpecDecodeInvalidToken};

    MTPSpecDecodeTransaction tx = buildMTPSpecDecodeTransaction(request);

    ASSERT_TRUE(tx.ok) << tx.error;
    EXPECT_EQ(tx.valid_sampled_count, 2);
    EXPECT_EQ(tx.accepted_speculative_prefix, 1);
    EXPECT_EQ(tx.rejected_token_count, 2);
    EXPECT_EQ(tx.token_index_to_sample, 1);
    EXPECT_EQ(tx.next_condition_token, 77);
    EXPECT_FALSE(tx.allDraftsAccepted());
    EXPECT_THAT(tx.committed_output_tokens, ElementsAre(11, 77));
    EXPECT_THAT(tx.rejected_draft_tokens, ElementsAre(12, 13));
}

TEST(Test__MTPSpecDecodeTransaction, RejectsFirstDraftButCommitsVerifierCorrection)
{
    MTPSpecDecodeRequest request;
    request.vocab_size = 100;
    request.draft_tokens = {11, 12, 13};
    request.sampled_tokens = {77, kMTPSpecDecodeInvalidToken, kMTPSpecDecodeInvalidToken};

    MTPSpecDecodeTransaction tx = buildMTPSpecDecodeTransaction(request);

    ASSERT_TRUE(tx.ok) << tx.error;
    EXPECT_EQ(tx.valid_sampled_count, 1);
    EXPECT_EQ(tx.accepted_speculative_prefix, 0);
    EXPECT_EQ(tx.rejected_token_count, 3);
    EXPECT_EQ(tx.token_index_to_sample, 0);
    EXPECT_EQ(tx.next_condition_token, 77);
    EXPECT_FALSE(tx.allDraftsAccepted());
    EXPECT_THAT(tx.committed_output_tokens, ElementsAre(77));
    EXPECT_THAT(tx.rejected_draft_tokens, ElementsAre(11, 12, 13));
}

TEST(Test__MTPSpecDecodeTransaction, HandlesDiscardedRequestWithBackupToken)
{
    MTPSpecDecodeRequest request;
    request.vocab_size = 100;
    request.draft_tokens = {11, 12, 13};
    request.sampled_tokens = {kMTPSpecDecodeInvalidToken, kMTPSpecDecodeInvalidToken};
    request.backup_next_token = 88;
    request.discarded = true;

    MTPSpecDecodeTransaction tx = buildMTPSpecDecodeTransaction(request);

    ASSERT_TRUE(tx.ok) << tx.error;
    EXPECT_EQ(tx.valid_sampled_count, 0);
    EXPECT_EQ(tx.accepted_speculative_prefix, 0);
    EXPECT_EQ(tx.rejected_token_count, 4);
    EXPECT_EQ(tx.token_index_to_sample, -1);
    EXPECT_EQ(tx.next_condition_token, 88);
    EXPECT_THAT(tx.committed_output_tokens, IsEmpty());
    EXPECT_THAT(tx.rejected_draft_tokens, ElementsAre(11, 12, 13));
}

TEST(Test__MTPSpecDecodeTransaction, RejectsNonContiguousValidSampledRows)
{
    MTPSpecDecodeRequest request;
    request.vocab_size = 100;
    request.draft_tokens = {11, 12, 13};
    request.sampled_tokens = {11, kMTPSpecDecodeInvalidToken, 13, kMTPSpecDecodeInvalidToken};

    MTPSpecDecodeTransaction tx = buildMTPSpecDecodeTransaction(request);

    EXPECT_FALSE(tx.ok);
    EXPECT_THAT(tx.error, HasSubstr("non-contiguous"));
}

TEST(Test__MTPSpecDecodeTransaction, RejectsRowsLongerThanVerifierQuery)
{
    MTPSpecDecodeRequest request;
    request.vocab_size = 100;
    request.draft_tokens = {11, 12};
    request.sampled_tokens = {11, 12, 13, 14};

    MTPSpecDecodeTransaction tx = buildMTPSpecDecodeTransaction(request);

    EXPECT_FALSE(tx.ok);
    EXPECT_THAT(tx.error, HasSubstr("longer"));
}

TEST(Test__MTPSpecDecodeTransaction, SummarizesBatchLikeVllmPaddedSpecDecode)
{
    MTPSpecDecodeRequest accept_all;
    accept_all.request_id = 1;
    accept_all.vocab_size = 100;
    accept_all.draft_tokens = {11, 12};
    accept_all.sampled_tokens = {11, 12, 90};

    MTPSpecDecodeRequest reject_after_first;
    reject_after_first.request_id = 2;
    reject_after_first.vocab_size = 100;
    reject_after_first.draft_tokens = {21, 22, 23};
    reject_after_first.sampled_tokens = {21, 91, kMTPSpecDecodeInvalidToken};

    MTPSpecDecodeBatchSummary summary =
        buildMTPSpecDecodeBatchSummary({accept_all, reject_after_first});

    ASSERT_TRUE(summary.ok) << summary.error;
    ASSERT_THAT(summary.transactions, SizeIs(2));
    EXPECT_EQ(summary.total_target_query_tokens, 7);
    EXPECT_EQ(summary.total_valid_sampled_tokens, 5);
    EXPECT_EQ(summary.total_rejected_tokens, 2);
    EXPECT_EQ(summary.transactions[0].token_index_to_sample, 2);
    EXPECT_EQ(summary.transactions[1].token_index_to_sample, 1);
    EXPECT_EQ(summary.transactions[1].next_condition_token, 91);
}

TEST(Test__MTPSpecDecodeTransaction, BuildsVerifierOutputRequestWithAcceptedBonusToken)
{
    MTPSpecDecodeRequest request = buildMTPSpecDecodeRequestFromVerifierOutput(
        /*request_id=*/9,
        /*vocab_size=*/100,
        /*draft_tokens=*/{7, 9, 8, 6},
        /*committed_output_tokens=*/{7, 9, 8, 6},
        /*bonus_ready_token=*/4,
        /*all_drafts_accepted=*/true,
        /*stopped_on_output=*/false);

    EXPECT_EQ(request.request_id, 9);
    EXPECT_THAT(request.draft_tokens, ElementsAre(7, 9, 8, 6));
    EXPECT_THAT(request.sampled_tokens, ElementsAre(7, 9, 8, 6, 4));

    MTPSpecDecodeTransaction tx =
        buildMTPSpecDecodeTransaction(request);

    ASSERT_TRUE(tx.ok) << tx.error;
    EXPECT_EQ(tx.valid_sampled_count, 5);
    EXPECT_EQ(tx.accepted_speculative_prefix, 4);
    EXPECT_EQ(tx.rejected_token_count, 0);
    EXPECT_EQ(tx.token_index_to_sample, 4);
    EXPECT_EQ(tx.next_condition_token, 4);
    EXPECT_TRUE(tx.allDraftsAccepted());
    EXPECT_THAT(tx.rejected_draft_tokens, IsEmpty());
}

TEST(Test__MTPSpecDecodeTransaction, BuildsVerifierOutputRequestWithoutBonusAfterReject)
{
    MTPSpecDecodeRequest request = buildMTPSpecDecodeRequestFromVerifierOutput(
        /*request_id=*/0,
        /*vocab_size=*/100,
        /*draft_tokens=*/{7, 9, 9},
        /*committed_output_tokens=*/{7, 9, 3},
        /*bonus_ready_token=*/5,
        /*all_drafts_accepted=*/false,
        /*stopped_on_output=*/false);

    EXPECT_THAT(request.sampled_tokens,
                ElementsAre(7, 9, 3, kMTPSpecDecodeInvalidToken));

    MTPSpecDecodeTransaction tx =
        buildMTPSpecDecodeTransaction(request);

    ASSERT_TRUE(tx.ok) << tx.error;
    EXPECT_EQ(tx.valid_sampled_count, 3);
    EXPECT_EQ(tx.accepted_speculative_prefix, 2);
    EXPECT_EQ(tx.rejected_token_count, 1);
    EXPECT_EQ(tx.token_index_to_sample, 2);
    EXPECT_EQ(tx.next_condition_token, 3);
    EXPECT_FALSE(tx.allDraftsAccepted());
    EXPECT_THAT(tx.committed_output_tokens, ElementsAre(7, 9, 3));
    EXPECT_THAT(tx.rejected_draft_tokens, ElementsAre(9));
}

TEST(Test__MTPSpecDecodeTransaction, BuildsVerifierOutputRequestWithoutBonusAfterStop)
{
    MTPSpecDecodeRequest request = buildMTPSpecDecodeRequestFromVerifierOutput(
        /*request_id=*/0,
        /*vocab_size=*/100,
        /*draft_tokens=*/{7, 9, 8},
        /*committed_output_tokens=*/{7, 9},
        /*bonus_ready_token=*/std::nullopt,
        /*all_drafts_accepted=*/true,
        /*stopped_on_output=*/true);

    EXPECT_THAT(request.sampled_tokens,
                ElementsAre(7, 9, kMTPSpecDecodeInvalidToken, kMTPSpecDecodeInvalidToken));

    MTPSpecDecodeTransaction tx =
        buildMTPSpecDecodeTransaction(request);

    ASSERT_TRUE(tx.ok) << tx.error;
    EXPECT_EQ(tx.valid_sampled_count, 2);
    EXPECT_EQ(tx.accepted_speculative_prefix, 2);
    EXPECT_EQ(tx.rejected_token_count, 2);
    EXPECT_EQ(tx.token_index_to_sample, 1);
    EXPECT_EQ(tx.next_condition_token, 9);
    EXPECT_FALSE(tx.allDraftsAccepted());
    EXPECT_THAT(tx.rejected_draft_tokens, ElementsAre(8));
}
