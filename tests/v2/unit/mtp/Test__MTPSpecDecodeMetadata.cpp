#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "execution/mtp/MTPSpecDecodeMetadata.h"

using namespace llaminar2;
using namespace testing;

namespace
{
    const WorkspaceDescriptor *findBuffer(
        const WorkspaceRequirements &reqs,
        const char *name)
    {
        return reqs.find(name);
    }
} // namespace

TEST(Test__MTPSpecDecodeMetadata, DeclaresGraphFacingWorkspaceBuffers)
{
    MTPSpecDecodeMetadataShape shape;
    shape.max_requests = 2;
    shape.max_draft_tokens = 3;

    WorkspaceRequirements reqs =
        buildMTPSpecDecodeWorkspaceRequirements(shape);

    ASSERT_THAT(reqs.buffers, SizeIs(14));
    const WorkspaceDescriptor *draft_counts =
        findBuffer(reqs, MTPSpecDecodeWorkspaceBuffers::DRAFT_COUNTS);
    ASSERT_NE(draft_counts, nullptr);
    EXPECT_EQ(draft_counts->size_bytes, 2u * sizeof(int32_t));
    EXPECT_EQ(draft_counts->alignment, 256u);
    EXPECT_TRUE(draft_counts->required);

    const WorkspaceDescriptor *query_starts =
        findBuffer(reqs, MTPSpecDecodeWorkspaceBuffers::QUERY_START_LOCS);
    ASSERT_NE(query_starts, nullptr);
    EXPECT_EQ(query_starts->size_bytes, 3u * sizeof(int32_t));

    const WorkspaceDescriptor *state_indices =
        findBuffer(reqs, MTPSpecDecodeWorkspaceBuffers::STATE_INDICES);
    ASSERT_NE(state_indices, nullptr);
    EXPECT_EQ(state_indices->size_bytes, 8u * sizeof(int32_t));

    const WorkspaceDescriptor *draft_tokens =
        findBuffer(reqs, MTPSpecDecodeWorkspaceBuffers::DRAFT_TOKENS);
    ASSERT_NE(draft_tokens, nullptr);
    EXPECT_EQ(draft_tokens->size_bytes, 6u * sizeof(int32_t));

    const WorkspaceDescriptor *sampled_tokens =
        findBuffer(reqs, MTPSpecDecodeWorkspaceBuffers::SAMPLED_TOKENS);
    ASSERT_NE(sampled_tokens, nullptr);
    EXPECT_EQ(sampled_tokens->size_bytes, 8u * sizeof(int32_t));
}

TEST(Test__MTPSpecDecodeMetadata, BuildsPaddedBatchMetadataForAcceptAndReject)
{
    MTPSpecDecodeMetadataShape shape;
    shape.max_requests = 2;
    shape.max_draft_tokens = 3;

    MTPSpecDecodeRequest accept_all;
    accept_all.vocab_size = 100;
    accept_all.draft_tokens = {7, 9, 8};
    accept_all.sampled_tokens = {7, 9, 8, 4};

    MTPSpecDecodeRequest reject_after_first;
    reject_after_first.vocab_size = 100;
    reject_after_first.draft_tokens = {11, 12, 13};
    reject_after_first.sampled_tokens = {
        11,
        77,
        kMTPSpecDecodeInvalidToken,
        kMTPSpecDecodeInvalidToken};

    MTPSpecDecodeMetadataBatch batch =
        buildMTPSpecDecodeMetadataBatch(
            shape,
            {accept_all, reject_after_first},
            /*committed_output_counts=*/{3, 2},
            /*stopped_flags=*/{0, 0});

    ASSERT_TRUE(batch.ok) << batch.error;
    EXPECT_EQ(batch.request_count, 2);
    EXPECT_EQ(batch.total_target_query_tokens, 8);
    EXPECT_THAT(batch.draft_counts, ElementsAre(3, 3));
    EXPECT_THAT(batch.target_query_lens, ElementsAre(4, 4));
    EXPECT_THAT(batch.valid_sampled_counts, ElementsAre(4, 2));
    EXPECT_THAT(batch.accepted_draft_prefixes, ElementsAre(3, 1));
    EXPECT_THAT(batch.committed_output_counts, ElementsAre(3, 2));
    EXPECT_THAT(batch.rejected_token_counts, ElementsAre(0, 2));
    EXPECT_THAT(batch.token_indices_to_sample, ElementsAre(3, 1));
    EXPECT_THAT(batch.next_condition_tokens, ElementsAre(4, 77));
    EXPECT_THAT(batch.all_drafts_accepted_flags, ElementsAre(1, 0));
    EXPECT_THAT(batch.stopped_flags, ElementsAre(0, 0));
    EXPECT_THAT(batch.query_start_locs, ElementsAre(0, 4, 8));
    EXPECT_THAT(batch.state_indices, ElementsAre(0, 1, 2, 3, 4, 5, 6, 7));
    EXPECT_THAT(batch.draft_tokens, ElementsAre(7, 9, 8, 11, 12, 13));
    EXPECT_THAT(batch.sampled_tokens,
                ElementsAre(7, 9, 8, 4,
                            11, 77,
                            kMTPSpecDecodeInvalidToken,
                            kMTPSpecDecodeInvalidToken));
    ASSERT_THAT(batch.transactions, SizeIs(2));
    EXPECT_TRUE(batch.transactions[0].allDraftsAccepted());
    EXPECT_FALSE(batch.transactions[1].allDraftsAccepted());
}

TEST(Test__MTPSpecDecodeMetadata, PadsUnusedRequestAndTokenSlots)
{
    MTPSpecDecodeMetadataShape shape;
    shape.max_requests = 2;
    shape.max_draft_tokens = 4;

    MTPSpecDecodeRequest request;
    request.vocab_size = 100;
    request.draft_tokens = {7, 9};
    request.sampled_tokens = {7, 9, 4};

    MTPSpecDecodeMetadataBatch batch =
        buildMTPSpecDecodeMetadataBatch(
            shape,
            {request},
            /*committed_output_counts=*/{2},
            /*stopped_flags=*/{0});

    ASSERT_TRUE(batch.ok) << batch.error;
    EXPECT_THAT(batch.draft_counts, ElementsAre(2, 0));
    EXPECT_THAT(batch.query_start_locs, ElementsAre(0, 3, 0));
    EXPECT_THAT(batch.draft_tokens,
                ElementsAre(7, 9,
                            kMTPSpecDecodeInvalidToken,
                            kMTPSpecDecodeInvalidToken,
                            kMTPSpecDecodeInvalidToken,
                            kMTPSpecDecodeInvalidToken,
                            kMTPSpecDecodeInvalidToken,
                            kMTPSpecDecodeInvalidToken));
    EXPECT_THAT(batch.sampled_tokens,
                ElementsAre(7, 9, 4,
                            kMTPSpecDecodeInvalidToken,
                            kMTPSpecDecodeInvalidToken,
                            kMTPSpecDecodeInvalidToken,
                            kMTPSpecDecodeInvalidToken,
                            kMTPSpecDecodeInvalidToken,
                            kMTPSpecDecodeInvalidToken,
                            kMTPSpecDecodeInvalidToken));
}

TEST(Test__MTPSpecDecodeMetadata, RejectsInvalidShapesAndOversizedRequests)
{
    MTPSpecDecodeMetadataShape invalid_shape;
    invalid_shape.max_requests = 1;
    invalid_shape.max_draft_tokens = 0;
    EXPECT_TRUE(buildMTPSpecDecodeWorkspaceRequirements(invalid_shape).buffers.empty());

    MTPSpecDecodeMetadataBatch invalid =
        buildMTPSpecDecodeMetadataBatch(invalid_shape, {}, {}, {});
    EXPECT_FALSE(invalid.ok);
    EXPECT_THAT(invalid.error, HasSubstr("invalid"));

    MTPSpecDecodeMetadataShape shape;
    shape.max_requests = 1;
    shape.max_draft_tokens = 2;

    MTPSpecDecodeRequest too_many_drafts;
    too_many_drafts.vocab_size = 100;
    too_many_drafts.draft_tokens = {1, 2, 3};
    too_many_drafts.sampled_tokens = {1, 2, 3, 4};

    MTPSpecDecodeMetadataBatch oversized =
        buildMTPSpecDecodeMetadataBatch(
            shape,
            {too_many_drafts},
            /*committed_output_counts=*/{3},
            /*stopped_flags=*/{0});
    EXPECT_FALSE(oversized.ok);
    EXPECT_THAT(oversized.error, HasSubstr("exceeds metadata shape"));
}

TEST(Test__MTPSpecDecodeMetadata, RejectsCommittedOutputCountPastValidPrefix)
{
    MTPSpecDecodeMetadataShape shape;
    shape.max_requests = 1;
    shape.max_draft_tokens = 3;

    MTPSpecDecodeRequest request;
    request.vocab_size = 100;
    request.draft_tokens = {7, 9, 8};
    request.sampled_tokens = {7, kMTPSpecDecodeInvalidToken};

    MTPSpecDecodeMetadataBatch batch =
        buildMTPSpecDecodeMetadataBatch(
            shape,
            {request},
            /*committed_output_counts=*/{2},
            /*stopped_flags=*/{0});

    EXPECT_FALSE(batch.ok);
    EXPECT_THAT(batch.error, HasSubstr("committed output count"));
}
