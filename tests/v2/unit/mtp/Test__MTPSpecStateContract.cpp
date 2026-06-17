#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "execution/compute_stages/IComputeStage.h"
#include "execution/local_execution/graph/ComputeGraph.h"
#include "execution/mtp/MTPSpecStateContract.h"
#include "execution/mtp/MTPSpecStatePublisher.h"
#include "execution/mtp/MTPSpecTransactionDriver.h"

using namespace llaminar2;
using namespace testing;

namespace
{
    class FakeVerifierStateStage : public IComputeStage
    {
    public:
        explicit FakeVerifierStateStage(
            bool captures,
            bool restore_ok = true,
            bool required_for_publication = false)
            : IComputeStage(DeviceId::cpu()),
              captures_(captures),
              restore_ok_(restore_ok),
              required_for_publication_(required_for_publication)
        {
        }

        bool execute(IDeviceContext *ctx) override
        {
            (void)ctx;
            return true;
        }

        ComputeStageType type() const override
        {
            return ComputeStageType::COPY;
        }

        bool supportsBackend(ComputeBackendType backend) const override
        {
            (void)backend;
            return true;
        }

        bool hasVerifierStateCapture() const override
        {
            return captures_;
        }

        bool requiresVerifierStateCaptureForPublication() const override
        {
            return required_for_publication_;
        }

        bool restoreVerifierStateCaptureRow(int row, void *stream) override
        {
            restored_rows.push_back(row);
            streams.push_back(stream);
            return restore_ok_;
        }

        bool restoreVerifierStateCaptureRowFromDeviceIndex(
            const int *device_row_index,
            void *stream) override
        {
            device_restored_rows.push_back(device_row_index);
            device_streams.push_back(stream);
            return restore_ok_;
        }

        bool restoreVerifierStateCaptureRowsFromDeviceIndices(
            const int *device_row_indices,
            int request_count,
            int row_index_stride,
            void *stream) override
        {
            batch_device_restored_rows.push_back(device_row_indices);
            batch_request_counts.push_back(request_count);
            batch_row_index_strides.push_back(row_index_stride);
            batch_streams.push_back(stream);
            return restore_ok_;
        }

        StageDumpInfo buildDumpInfoImpl() const override
        {
            return {};
        }

        std::vector<int> restored_rows;
        std::vector<void *> streams;
        std::vector<const int *> device_restored_rows;
        std::vector<void *> device_streams;
        std::vector<const int *> batch_device_restored_rows;
        std::vector<int> batch_request_counts;
        std::vector<int> batch_row_index_strides;
        std::vector<void *> batch_streams;

    private:
        bool captures_ = false;
        bool restore_ok_ = true;
        bool required_for_publication_ = false;
    };

    MTPSpecDecodeMetadataShape shapeFor(int requests = 1, int draft_tokens = 3)
    {
        MTPSpecDecodeMetadataShape shape;
        shape.max_requests = requests;
        shape.max_draft_tokens = draft_tokens;
        return shape;
    }

    MTPSpecStepPlan publishPlan(int accepted_count)
    {
        MTPSpecStepPlan plan;
        plan.request_id = 9;
        plan.draft_count = 3;
        plan.target_rows = 4;
        plan.accepted_count = accepted_count;
        return plan;
    }

    MTPSpecStepPlan participantPlan(int participant_id, int accepted_count)
    {
        MTPSpecStepPlan plan;
        plan.request_index = participant_id;
        plan.request_id = 42;
        plan.draft_count = 3;
        plan.target_rows = 4;
        plan.valid_sampled_count = 4;
        plan.committed_output_count = 3;
        plan.accepted_count = accepted_count;
        plan.rejected_count = 0;
        plan.base_cached_tokens = 100;
        plan.target_cached_tokens = plan.base_cached_tokens + accepted_count;
        plan.accepted_state_slot_index =
            accepted_count > 0 ? accepted_count - 1 : kMTPSpecDecodeInvalidToken;
        plan.next_condition_token = 77;
        plan.all_drafts_accepted = accepted_count == plan.draft_count;
        if (plan.all_drafts_accepted)
        {
            plan.bonus_ready_token_row = 3;
            plan.bonus_ready_token_index = 3;
            plan.bonus_ready_state_slot_index = 3;
        }
        return plan;
    }

    MTPSpecStepPlanBatch planBatch(std::initializer_list<MTPSpecStepPlan> steps)
    {
        MTPSpecStepPlanBatch batch;
        batch.ok = true;
        batch.request_count = static_cast<int>(steps.size());
        batch.steps.assign(steps.begin(), steps.end());
        return batch;
    }
} // namespace

TEST(Test__MTPSpecStateContract, BuildsAcceptAllPlanWithBonusReadyRow)
{
    MTPSpecDecodeRequest request;
    request.request_id = 17;
    request.vocab_size = 100;
    request.draft_tokens = {7, 9, 8};
    request.sampled_tokens = {7, 9, 8, 4};

    MTPSpecDecodeMetadataBatch metadata =
        buildMTPSpecDecodeMetadataBatch(
            shapeFor(),
            {request},
            /*committed_output_counts=*/{3},
            /*stopped_flags=*/{0});
    ASSERT_TRUE(metadata.ok) << metadata.error;

    MTPSpecStepPlanBatch plans =
        buildMTPSpecStepPlans(
            metadata,
            /*base_cached_tokens=*/std::vector<int32_t>{128});
    ASSERT_TRUE(plans.ok) << plans.error;
    ASSERT_THAT(plans.steps, SizeIs(1));

    const MTPSpecStepPlan &step = plans.steps.front();
    EXPECT_EQ(step.request_index, 0);
    EXPECT_EQ(step.request_id, 17);
    EXPECT_EQ(step.draft_count, 3);
    EXPECT_EQ(step.target_rows, 4);
    EXPECT_EQ(step.accepted_count, 3);
    EXPECT_EQ(step.committed_output_count, 3);
    EXPECT_EQ(step.valid_sampled_count, 4);
    EXPECT_EQ(step.rejected_count, 0);
    EXPECT_EQ(step.base_cached_tokens, 128);
    EXPECT_EQ(step.target_cached_tokens, 131);
    EXPECT_EQ(step.accepted_state_slot_index, 2);
    EXPECT_FALSE(step.requiresCorrectionReplay());
    EXPECT_TRUE(step.publishesAcceptedState());
    EXPECT_TRUE(step.all_drafts_accepted);
    EXPECT_TRUE(step.hasBonusReadyToken());
    EXPECT_EQ(step.bonus_ready_token_row, 3);
    EXPECT_EQ(step.bonus_ready_token_index, 3);
    EXPECT_EQ(step.bonus_ready_state_slot_index, 3);
}

TEST(Test__MTPSpecStateContract, CommonAcceptedPrefixLeavesMatchingParticipantsDirect)
{
    std::vector<MTPSpecStepPlan> participants = {
        participantPlan(/*participant_id=*/0, /*accepted_count=*/3),
        participantPlan(/*participant_id=*/1, /*accepted_count=*/3),
    };

    MTPSpecCommonStepPlan common =
        coordinateMTPSpecCommonAcceptedPrefix(participants);

    ASSERT_TRUE(common.ok) << common.error;
    EXPECT_EQ(common.common_accepted_count, 3);
    EXPECT_TRUE(common.all_participants_direct);
    EXPECT_FALSE(common.requires_common_fallback_replay);
    ASSERT_THAT(common.clamped_steps, SizeIs(2));
    EXPECT_EQ(common.clamped_steps[0].accepted_count, 3);
    EXPECT_EQ(common.clamped_steps[1].accepted_count, 3);
    EXPECT_TRUE(common.clamped_steps[0].hasBonusReadyToken());
}

TEST(Test__MTPSpecStateContract, CommonAcceptedPrefixClampsLongerParticipant)
{
    std::vector<MTPSpecStepPlan> participants = {
        participantPlan(/*participant_id=*/0, /*accepted_count=*/3),
        participantPlan(/*participant_id=*/1, /*accepted_count=*/1),
    };
    /*
     * Participant 1 is already at the common prefix, but it still carries a
     * participant-local correction suffix.  Once any other participant is
     * shortened, the topology owner must replay from the shared prefix, so that
     * local suffix is no longer safe to publish either.
     */
    participants[1].correction_replay_start_index = 1;
    participants[1].correction_replay_count = 1;

    MTPSpecCommonStepPlan common =
        coordinateMTPSpecCommonAcceptedPrefix(participants);

    ASSERT_TRUE(common.ok) << common.error;
    EXPECT_EQ(common.common_accepted_count, 1);
    EXPECT_FALSE(common.all_participants_direct);
    EXPECT_TRUE(common.requires_common_fallback_replay);
    ASSERT_THAT(common.clamped_steps, SizeIs(2));

    for (const MTPSpecStepPlan &step : common.clamped_steps)
    {
        EXPECT_EQ(step.accepted_count, 1);
        EXPECT_EQ(step.target_cached_tokens, step.base_cached_tokens + 1);
        EXPECT_EQ(step.accepted_state_slot_index, 0);
        EXPECT_FALSE(step.hasBonusReadyToken());
        EXPECT_FALSE(step.requiresCorrectionReplay());
        EXPECT_FALSE(step.all_drafts_accepted);
    }
}

TEST(Test__MTPSpecStateContract, CommonAcceptedPrefixRejectsDifferentTokenStreams)
{
    std::vector<MTPSpecStepPlan> participants = {
        participantPlan(/*participant_id=*/0, /*accepted_count=*/2),
        participantPlan(/*participant_id=*/1, /*accepted_count=*/2),
    };
    participants[1].next_condition_token = 88;

    MTPSpecCommonStepPlan common =
        coordinateMTPSpecCommonAcceptedPrefix(participants);

    EXPECT_FALSE(common.ok);
    EXPECT_THAT(common.error, HasSubstr("different MTP speculative step"));
}

TEST(Test__MTPSpecStateContract, BuildsRejectedSuffixPlanWithCorrectionReplay)
{
    MTPSpecDecodeRequest request;
    request.request_id = 23;
    request.vocab_size = 100;
    request.draft_tokens = {7, 9, 9};
    request.sampled_tokens = {
        7,
        9,
        3,
        kMTPSpecDecodeInvalidToken};

    MTPSpecDecodeMetadataBatch metadata =
        buildMTPSpecDecodeMetadataBatch(
            shapeFor(),
            {request},
            /*committed_output_counts=*/{3},
            /*stopped_flags=*/{0});
    ASSERT_TRUE(metadata.ok) << metadata.error;

    MTPSpecStepPlanBatch plans =
        buildMTPSpecStepPlans(
            metadata,
            /*base_cached_tokens=*/std::vector<int32_t>{64});
    ASSERT_TRUE(plans.ok) << plans.error;
    ASSERT_THAT(plans.steps, SizeIs(1));

    const MTPSpecStepPlan &step = plans.steps.front();
    EXPECT_EQ(step.request_id, 23);
    EXPECT_EQ(step.accepted_count, 2);
    EXPECT_EQ(step.target_cached_tokens, 66);
    EXPECT_EQ(step.accepted_state_slot_index, 1);
    EXPECT_EQ(step.rejected_count, 1);
    EXPECT_TRUE(step.hasRejectedSuffix());
    EXPECT_TRUE(step.requiresCorrectionReplay());
    EXPECT_EQ(step.correction_replay_start_index, 2);
    EXPECT_EQ(step.correction_replay_count, 1);
    EXPECT_FALSE(step.hasBonusReadyToken());
    EXPECT_FALSE(step.all_drafts_accepted);
}

TEST(Test__MTPSpecStateContract, StoppedRejectedTokenDoesNotReplayCorrection)
{
    MTPSpecDecodeRequest request;
    request.request_id = 31;
    request.vocab_size = 100;
    request.draft_tokens = {7, 9, 8};
    request.sampled_tokens = {
        7,
        3,
        kMTPSpecDecodeInvalidToken,
        kMTPSpecDecodeInvalidToken};

    MTPSpecDecodeMetadataBatch metadata =
        buildMTPSpecDecodeMetadataBatchWithStateCommitCounts(
            shapeFor(),
            {request},
            /*committed_output_counts=*/{2},
            /*target_verifier_state_commit_counts=*/{1},
            /*stopped_flags=*/{1});
    ASSERT_TRUE(metadata.ok) << metadata.error;

    MTPSpecStepPlanBatch plans =
        buildMTPSpecStepPlans(
            metadata,
            /*base_cached_tokens=*/std::vector<int32_t>{8});
    ASSERT_TRUE(plans.ok) << plans.error;
    ASSERT_THAT(plans.steps, SizeIs(1));

    const MTPSpecStepPlan &step = plans.steps.front();
    EXPECT_TRUE(step.stopped);
    EXPECT_EQ(step.accepted_count, 1);
    EXPECT_EQ(step.target_cached_tokens, 9);
    EXPECT_EQ(step.accepted_state_slot_index, 0);
    EXPECT_EQ(step.committed_output_count, 2);
    EXPECT_FALSE(step.requiresCorrectionReplay());
    EXPECT_FALSE(step.hasBonusReadyToken());
}

TEST(Test__MTPSpecStateContract, DiscardedRequestDoesNotPublishState)
{
    MTPSpecDecodeRequest request;
    request.request_id = 41;
    request.vocab_size = 100;
    request.draft_tokens = {7, 9};
    request.sampled_tokens = {
        kMTPSpecDecodeInvalidToken,
        kMTPSpecDecodeInvalidToken,
        kMTPSpecDecodeInvalidToken};
    request.discarded = true;
    request.backup_next_token = 42;

    MTPSpecDecodeMetadataBatch metadata =
        buildMTPSpecDecodeMetadataBatch(
            shapeFor(/*requests=*/1, /*draft_tokens=*/2),
            {request},
            /*committed_output_counts=*/{0},
            /*stopped_flags=*/{0});
    ASSERT_TRUE(metadata.ok) << metadata.error;

    MTPSpecStepPlanBatch plans =
        buildMTPSpecStepPlans(
            metadata,
            /*base_cached_tokens=*/std::vector<int32_t>{11});
    ASSERT_TRUE(plans.ok) << plans.error;
    ASSERT_THAT(plans.steps, SizeIs(1));

    const MTPSpecStepPlan &step = plans.steps.front();
    EXPECT_EQ(step.accepted_count, 0);
    EXPECT_EQ(step.target_cached_tokens, 11);
    EXPECT_EQ(step.accepted_state_slot_index, kMTPSpecDecodeInvalidToken);
    EXPECT_EQ(step.next_condition_token, 42);
    EXPECT_FALSE(step.publishesAcceptedState());
    EXPECT_FALSE(step.requiresCorrectionReplay());
    EXPECT_FALSE(step.hasBonusReadyToken());
}

TEST(Test__MTPSpecStateContract, AllowsGlobalStateSlotsAcrossMultipleRequests)
{
    MTPSpecDecodeRequest accept_all;
    accept_all.request_id = 1;
    accept_all.vocab_size = 100;
    accept_all.draft_tokens = {7, 9, 8};
    accept_all.sampled_tokens = {7, 9, 8, 4};

    MTPSpecDecodeRequest reject_after_first;
    reject_after_first.request_id = 2;
    reject_after_first.vocab_size = 100;
    reject_after_first.draft_tokens = {11, 12, 13};
    reject_after_first.sampled_tokens = {
        11,
        77,
        kMTPSpecDecodeInvalidToken,
        kMTPSpecDecodeInvalidToken};

    MTPSpecDecodeMetadataBatch metadata =
        buildMTPSpecDecodeMetadataBatch(
            shapeFor(/*requests=*/2, /*draft_tokens=*/3),
            {accept_all, reject_after_first},
            /*committed_output_counts=*/{3, 2},
            /*stopped_flags=*/{0, 0});
    ASSERT_TRUE(metadata.ok) << metadata.error;

    MTPSpecStepPlanBatch plans =
        buildMTPSpecStepPlans(
            metadata,
            /*base_cached_tokens=*/std::vector<int32_t>{100, 200});
    ASSERT_TRUE(plans.ok) << plans.error;
    ASSERT_THAT(plans.steps, SizeIs(2));

    EXPECT_EQ(plans.steps[0].accepted_state_slot_index, 2);
    EXPECT_EQ(plans.steps[0].bonus_ready_state_slot_index, 3);
    EXPECT_EQ(plans.steps[0].target_cached_tokens, 103);

    EXPECT_EQ(plans.steps[1].accepted_count, 1);
    EXPECT_EQ(plans.steps[1].accepted_state_slot_index, 4);
    EXPECT_EQ(plans.steps[1].target_cached_tokens, 201);
    EXPECT_TRUE(plans.steps[1].requiresCorrectionReplay());
}

TEST(Test__MTPSpecStateContract, TransactionDriverBuildsBatchedAcceptedOutcomePlan)
{
    MTPSpecDecodeAcceptedOutcome accept_all;
    accept_all.request_id = 10;
    accept_all.vocab_size = 100;
    accept_all.draft_count = 3;
    accept_all.committed_output_tokens = {7, 9, 8};
    accept_all.bonus_ready_token = 4;
    accept_all.accepted_verifier_input_prefix = 3;
    accept_all.target_verifier_state_commit_count = 3;
    accept_all.all_drafts_accepted = true;

    MTPSpecDecodeAcceptedOutcome reject_after_first;
    reject_after_first.request_id = 11;
    reject_after_first.vocab_size = 100;
    reject_after_first.draft_count = 2;
    reject_after_first.committed_output_tokens = {11, 77};
    reject_after_first.accepted_verifier_input_prefix = 1;
    reject_after_first.target_verifier_state_commit_count = 1;
    reject_after_first.all_drafts_accepted = false;

    MTPSpecDecodeMetadataShape shape;
    shape.max_requests = 2;
    shape.max_draft_tokens = 3;

    MTPSpecTransactionBatchPlan plan =
        buildMTPSpecTransactionBatchPlanFromAcceptedOutcomes(
            shape,
            {accept_all, reject_after_first},
            /*base_cached_tokens=*/{100, 200});

    ASSERT_TRUE(plan.ok) << plan.error;
    EXPECT_EQ(plan.request_count, 2);
    EXPECT_EQ(plan.metadata.total_target_query_tokens, 7);
    EXPECT_THAT(plan.publication_plan.base_cached_tokens,
                ElementsAre(100, 200));
    EXPECT_THAT(plan.publication_plan.target_cached_tokens,
                ElementsAre(103, 201));
    ASSERT_THAT(plan.step_plans.steps, SizeIs(2));

    const MTPSpecStepPlan &first = plan.step_plans.steps[0];
    EXPECT_EQ(first.request_id, 10);
    EXPECT_EQ(first.accepted_count, 3);
    EXPECT_EQ(first.accepted_state_slot_index, 2);
    EXPECT_EQ(first.bonus_ready_state_slot_index, 3);
    EXPECT_TRUE(first.all_drafts_accepted);
    EXPECT_FALSE(first.requiresCorrectionReplay());

    const MTPSpecStepPlan &second = plan.step_plans.steps[1];
    EXPECT_EQ(second.request_id, 11);
    EXPECT_EQ(second.accepted_count, 1);
    EXPECT_EQ(second.accepted_state_slot_index, 4)
        << "request 1's accepted slot must refer to the flattened verifier batch";
    EXPECT_EQ(second.correction_replay_start_index, 1);
    EXPECT_EQ(second.correction_replay_count, 1);
    EXPECT_TRUE(second.requiresCorrectionReplay());
}

TEST(Test__MTPSpecStateContract, TransactionDriverBuildsBatchedDeviceRejectionOutcomePlan)
{
    MTPDecodeCatchupGreedyRequest accept_all_request;
    accept_all_request.draft_tokens = {7, 9, 8};

    MTPDeviceRejectionBatchOutcome accept_all;
    accept_all.ok = true;
    accept_all.output_tokens[0] = 7;
    accept_all.output_tokens[1] = 9;
    accept_all.output_tokens[2] = 8;
    accept_all.output_token_count = 3;
    accept_all.accepted_speculative_prefix = 2;
    accept_all.target_verifier_state_commit_count = 3;
    accept_all.ready_token = 4;
    accept_all.all_speculative_accepted = true;
    accept_all.consumed_verifier_rows = 2;
    accept_all.sampled_terminal = true;

    MTPDecodeCatchupGreedyRequest reject_after_first_request;
    reject_after_first_request.draft_tokens = {11, 12, 13};

    MTPDeviceRejectionBatchOutcome reject_after_first;
    reject_after_first.ok = true;
    reject_after_first.output_tokens[0] = 11;
    reject_after_first.output_tokens[1] = 77;
    reject_after_first.output_token_count = 2;
    reject_after_first.accepted_speculative_prefix = 0;
    reject_after_first.target_verifier_state_commit_count = 1;
    reject_after_first.ready_token = -1;
    reject_after_first.rejected_verified_token = 77;
    reject_after_first.all_speculative_accepted = false;
    reject_after_first.consumed_verifier_rows = 1;

    MTPSpecDecodeMetadataShape shape;
    shape.max_requests = 2;
    shape.max_draft_tokens = 3;

    MTPSpecTransactionBatchPlan plan =
        buildMTPSpecTransactionBatchPlanFromDeviceRejectionOutcomes(
            shape,
            /*request_ids=*/{10, 11},
            /*vocab_size=*/100,
            {accept_all_request, reject_after_first_request},
            {accept_all, reject_after_first},
            /*base_cached_tokens=*/{100, 200});

    ASSERT_TRUE(plan.ok) << plan.error;
    EXPECT_EQ(plan.request_count, 2);
    EXPECT_EQ(plan.metadata.total_target_query_tokens, 8)
        << "both requests keep padded draft+bonus rows in one verifier batch";
    EXPECT_THAT(plan.metadata.valid_sampled_counts, ElementsAre(4, 2));
    EXPECT_THAT(plan.metadata.accepted_draft_prefixes, ElementsAre(3, 1));
    EXPECT_THAT(plan.metadata.committed_output_counts, ElementsAre(3, 2));
    EXPECT_THAT(plan.metadata.target_verifier_state_commit_counts,
                ElementsAre(3, 1));
    EXPECT_THAT(plan.metadata.next_condition_tokens, ElementsAre(4, 77));
    EXPECT_THAT(plan.publication_plan.target_cached_tokens,
                ElementsAre(103, 201));
    ASSERT_THAT(plan.step_plans.steps, SizeIs(2));

    const MTPSpecStepPlan &first = plan.step_plans.steps[0];
    EXPECT_EQ(first.request_id, 10);
    EXPECT_EQ(first.accepted_count, 3);
    EXPECT_EQ(first.accepted_state_slot_index, 2);
    EXPECT_EQ(first.bonus_ready_state_slot_index, 3);
    EXPECT_TRUE(first.all_drafts_accepted);
    EXPECT_FALSE(first.requiresCorrectionReplay());

    const MTPSpecStepPlan &second = plan.step_plans.steps[1];
    EXPECT_EQ(second.request_id, 11);
    EXPECT_EQ(second.accepted_count, 1);
    EXPECT_EQ(second.accepted_state_slot_index, 4);
    EXPECT_EQ(second.correction_replay_start_index, 1);
    EXPECT_EQ(second.correction_replay_count, 1);
    EXPECT_TRUE(second.requiresCorrectionReplay());
}

TEST(Test__MTPSpecStateContract, TransactionDriverBuildsDeviceRejectionPlanFromMetadata)
{
    using namespace sampling_math;

    MTPDecodeCatchupGreedyRequest accept_all_request;
    accept_all_request.draft_tokens = {5, 6, 8};

    MTPDecodeCatchupGreedyRequest reject_after_first_request;
    reject_after_first_request.draft_tokens = {11, 12, 13};

    std::vector<int> meta(
        2 * kSpeculativeBatchMetaCount,
        0);
    int *first = meta.data();
    first[kSpecBatchMetaOk] = 1;
    first[kSpecBatchMetaOutputCount] = 3;
    first[kSpecBatchMetaAcceptedSpeculativePrefix] = 2;
    first[kSpecBatchMetaTargetVerifierStateCommitCount] = 3;
    first[kSpecBatchMetaReadyToken] = 4;
    first[kSpecBatchMetaRejectedVerifiedToken] = -1;
    first[kSpecBatchMetaStoppedOnOutput] = 0;
    first[kSpecBatchMetaAllSpeculativeAccepted] = 1;
    first[kSpecBatchMetaConsumedVerifierRows] = 2;
    first[kSpecBatchMetaSampledTerminal] = 1;

    int *second = meta.data() + kSpeculativeBatchMetaCount;
    second[kSpecBatchMetaOk] = 1;
    second[kSpecBatchMetaOutputCount] = 2;
    second[kSpecBatchMetaAcceptedSpeculativePrefix] = 0;
    second[kSpecBatchMetaTargetVerifierStateCommitCount] = 1;
    second[kSpecBatchMetaReadyToken] = -1;
    second[kSpecBatchMetaRejectedVerifiedToken] = 77;
    second[kSpecBatchMetaStoppedOnOutput] = 0;
    second[kSpecBatchMetaAllSpeculativeAccepted] = 0;
    second[kSpecBatchMetaConsumedVerifierRows] = 1;
    second[kSpecBatchMetaSampledTerminal] = 0;

    MTPSpecDecodeMetadataShape shape;
    shape.max_requests = 2;
    shape.max_draft_tokens = 3;

    MTPSpecTransactionBatchPlan plan =
        buildMTPSpecTransactionBatchPlanFromDeviceRejectionMetadata(
            shape,
            /*request_ids=*/{10, 11},
            /*vocab_size=*/100,
            {accept_all_request, reject_after_first_request},
            meta,
            kSpeculativeBatchMetaCount,
            /*base_cached_tokens=*/{100, 200});

    ASSERT_TRUE(plan.ok) << plan.error;
    EXPECT_THAT(plan.metadata.committed_output_counts, ElementsAre(3, 2));
    EXPECT_THAT(plan.metadata.next_condition_tokens, ElementsAre(4, 77));
    EXPECT_THAT(plan.publication_plan.target_cached_tokens,
                ElementsAre(103, 201));
    ASSERT_THAT(plan.step_plans.steps, SizeIs(2));
    EXPECT_EQ(plan.step_plans.steps[0].bonus_ready_state_slot_index, 3);
    EXPECT_EQ(plan.step_plans.steps[1].correction_replay_start_index, 1);
    EXPECT_EQ(plan.step_plans.steps[1].correction_replay_count, 1);
}

TEST(Test__MTPSpecStateContract, TransactionDriverRejectsInvalidDeviceRejectionOutcome)
{
    MTPDecodeCatchupGreedyRequest request;
    request.draft_tokens = {7, 9};

    MTPDeviceRejectionBatchOutcome invalid;
    invalid.ok = true;
    invalid.output_token_count = 0;

    MTPSpecDecodeMetadataShape shape;
    shape.max_requests = 1;
    shape.max_draft_tokens = 2;

    MTPSpecTransactionBatchPlan plan =
        buildMTPSpecTransactionBatchPlanFromDeviceRejectionOutcomes(
            shape,
            /*request_ids=*/{10},
            /*vocab_size=*/100,
            {request},
            {invalid},
            /*base_cached_tokens=*/{100});

    EXPECT_FALSE(plan.ok);
    EXPECT_THAT(plan.error, HasSubstr("outcome 0 failed"));
    EXPECT_THAT(plan.error, HasSubstr("token count"));
}

TEST(Test__MTPSpecStateContract, TransactionDriverRejectsBaseCacheCountMismatch)
{
    MTPSpecDecodeAcceptedOutcome outcome;
    outcome.request_id = 10;
    outcome.vocab_size = 100;
    outcome.draft_count = 2;
    outcome.committed_output_tokens = {7};
    outcome.accepted_verifier_input_prefix = 1;
    outcome.target_verifier_state_commit_count = 1;

    MTPSpecDecodeMetadataShape shape;
    shape.max_requests = 2;
    shape.max_draft_tokens = 2;

    MTPSpecTransactionBatchPlan plan =
        buildMTPSpecTransactionBatchPlanFromAcceptedOutcomes(
            shape,
            {outcome},
            /*base_cached_tokens=*/{});

    EXPECT_FALSE(plan.ok);
    EXPECT_THAT(plan.error, HasSubstr("base-cache vector"));
}

TEST(Test__MTPSpecStateContract, TransactionDriverBuildsGreedyCatchupPlan)
{
    MTPDecodeCatchupGreedyRequest request;
    request.draft_tokens = {7, 9, 8};

    MTPDecodeCatchupGreedyResult result;
    result.ok = true;
    result.accepted_tokens = {7, 9, 3};
    result.verifier_tokens = {9, 3};
    result.all_speculative_accepted = false;
    result.stopped_on_output = false;
    result.accepted_speculative_prefix = 1;
    result.rejected_verified_token = 3;
    result.target_verifier_state_commit_count = 2;

    MTPSpecTransactionBatchPlan plan =
        buildMTPSpecTransactionBatchPlanFromGreedyCatchup(
            shapeFor(/*requests=*/1, /*draft_tokens=*/3),
            /*request_id=*/23,
            /*vocab_size=*/100,
            request,
            result,
            /*base_cached_tokens=*/64);

    ASSERT_TRUE(plan.ok) << plan.error;
    ASSERT_THAT(plan.step_plans.steps, SizeIs(1));
    const MTPSpecStepPlan &step = plan.step_plans.steps.front();
    EXPECT_EQ(step.request_id, 23);
    EXPECT_EQ(step.accepted_count, 2);
    EXPECT_EQ(step.base_cached_tokens, 64);
    EXPECT_EQ(step.target_cached_tokens, 66);
    EXPECT_EQ(step.accepted_state_slot_index, 1);
    EXPECT_EQ(step.correction_replay_start_index, 2);
    EXPECT_EQ(step.correction_replay_count, 1);
}

TEST(Test__MTPSpecStateContract, TransactionDriverBuildsBatchedGreedyCatchupPlan)
{
    MTPDecodeCatchupGreedyRequest accept_all_request;
    accept_all_request.draft_tokens = {7, 9, 8};
    MTPDecodeCatchupGreedyResult accept_all_result =
        buildAllPositionMTPDecodeCatchupGreedyResult(
            accept_all_request,
            /*sampled_verifier_rows=*/{9, 8, 4});
    ASSERT_TRUE(accept_all_result.ok) << accept_all_result.error;

    MTPDecodeCatchupGreedyRequest reject_request;
    reject_request.draft_tokens = {11, 12, 13};
    MTPDecodeCatchupGreedyResult reject_result =
        buildAllPositionMTPDecodeCatchupGreedyResult(
            reject_request,
            /*sampled_verifier_rows=*/{77, 123, 123});
    ASSERT_TRUE(reject_result.ok) << reject_result.error;

    MTPSpecTransactionBatchPlan plan =
        buildMTPSpecTransactionBatchPlanFromGreedyCatchups(
            shapeFor(/*requests=*/2, /*draft_tokens=*/3),
            /*request_ids=*/{10, 11},
            /*vocab_size=*/100,
            {accept_all_request, reject_request},
            {accept_all_result, reject_result},
            /*base_cached_tokens=*/{100, 200});

    ASSERT_TRUE(plan.ok) << plan.error;
    EXPECT_EQ(plan.request_count, 2);
    EXPECT_EQ(plan.metadata.total_target_query_tokens, 8);
    ASSERT_THAT(plan.step_plans.steps, SizeIs(2));

    const MTPSpecStepPlan &first = plan.step_plans.steps[0];
    EXPECT_EQ(first.request_id, 10);
    EXPECT_EQ(first.accepted_count, 3);
    EXPECT_EQ(first.target_cached_tokens, 103);
    EXPECT_EQ(first.accepted_state_slot_index, 2);
    EXPECT_EQ(first.bonus_ready_state_slot_index, 3);
    EXPECT_FALSE(first.requiresCorrectionReplay());

    const MTPSpecStepPlan &second = plan.step_plans.steps[1];
    EXPECT_EQ(second.request_id, 11);
    EXPECT_EQ(second.accepted_count, 1);
    EXPECT_EQ(second.target_cached_tokens, 201);
    EXPECT_EQ(second.accepted_state_slot_index, 4);
    EXPECT_EQ(second.correction_replay_start_index, 1);
    EXPECT_EQ(second.correction_replay_count, 1);
    EXPECT_TRUE(second.requiresCorrectionReplay());
}

TEST(Test__MTPSpecStateContract, RejectsPublicationThatDriftsFromMetadata)
{
    MTPSpecDecodeRequest request;
    request.vocab_size = 100;
    request.draft_tokens = {7, 9};
    request.sampled_tokens = {7, 9, 4};

    MTPSpecDecodeMetadataShape shape = shapeFor(/*requests=*/1, /*draft_tokens=*/2);
    MTPSpecDecodeMetadataBatch metadata =
        buildMTPSpecDecodeMetadataBatch(
            shape,
            {request},
            /*committed_output_counts=*/{2},
            /*stopped_flags=*/{0});
    ASSERT_TRUE(metadata.ok) << metadata.error;

    MTPSpecDecodeStateCommitPlan commit_plan =
        buildMTPSpecDecodeStateCommitPlan(metadata);
    ASSERT_TRUE(commit_plan.ok) << commit_plan.error;
    MTPSpecDecodeStatePublicationPlan publication =
        buildMTPSpecDecodeStatePublicationPlan(
            commit_plan,
            /*base_cached_tokens=*/{50});
    ASSERT_TRUE(publication.ok) << publication.error;

    publication.target_cached_tokens[0] = 99;
    MTPSpecStepPlanBatch plans =
        buildMTPSpecStepPlans(metadata, publication);
    EXPECT_FALSE(plans.ok);
    EXPECT_THAT(plans.error, HasSubstr("target cached-token count"));
}

TEST(Test__MTPSpecStateContract, PublisherRestoresAcceptedRowOnCapturedStages)
{
    FakeVerifierStateStage captured0(/*captures=*/true);
    FakeVerifierStateStage skipped(/*captures=*/false);
    FakeVerifierStateStage captured1(/*captures=*/true);
    int explicit_stream = 0;

    MTPSpecStatePublicationResult result =
        publishAcceptedMTPSpecState(
            publishPlan(/*accepted_count=*/2),
            {&captured0, &skipped, &captured1},
            DeviceId::cpu(),
            &explicit_stream,
            /*require_captured_stage=*/true);

    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_EQ(result.restored_stage_count, 2);
    EXPECT_EQ(result.skipped_stage_count, 1);
    EXPECT_THAT(captured0.restored_rows, ElementsAre(1));
    EXPECT_THAT(captured1.restored_rows, ElementsAre(1));
    EXPECT_TRUE(skipped.restored_rows.empty());
    EXPECT_THAT(captured0.streams, ElementsAre(&explicit_stream));
    EXPECT_THAT(captured1.streams, ElementsAre(&explicit_stream));
}

TEST(Test__MTPSpecStateContract, PublisherAllowsZeroAcceptedWithoutStageRestore)
{
    FakeVerifierStateStage captured(/*captures=*/true);

    MTPSpecStatePublicationResult result =
        publishAcceptedMTPSpecState(
            publishPlan(/*accepted_count=*/0),
            {&captured},
            DeviceId::cpu(),
            /*stream=*/nullptr);

    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_EQ(result.restored_stage_count, 0);
    EXPECT_EQ(result.skipped_stage_count, 1);
    EXPECT_TRUE(captured.restored_rows.empty());
}

TEST(Test__MTPSpecStateContract, PublisherRejectsGpuNullStream)
{
    FakeVerifierStateStage captured(/*captures=*/true);

    MTPSpecStatePublicationResult result =
        publishAcceptedMTPSpecState(
            publishPlan(/*accepted_count=*/1),
            {&captured},
            DeviceId::cuda(0),
            /*stream=*/nullptr);

    EXPECT_FALSE(result.ok);
    EXPECT_THAT(result.error, HasSubstr("explicit non-null stream"));
    EXPECT_TRUE(captured.restored_rows.empty());
}

TEST(Test__MTPSpecStateContract, PublisherCanRequireCapturedState)
{
    FakeVerifierStateStage skipped(/*captures=*/false);

    MTPSpecStatePublicationResult result =
        publishAcceptedMTPSpecState(
            publishPlan(/*accepted_count=*/1),
            {&skipped},
            DeviceId::cpu(),
            /*stream=*/nullptr,
            /*require_captured_stage=*/true);

    EXPECT_FALSE(result.ok);
    EXPECT_THAT(result.error, HasSubstr("required"));
}

TEST(Test__MTPSpecStateContract, PublisherRejectsMissingMandatoryVerifierStateStage)
{
    FakeVerifierStateStage optional_skipped(/*captures=*/false);
    FakeVerifierStateStage mandatory_skipped(
        /*captures=*/false,
        /*restore_ok=*/true,
        /*required_for_publication=*/true);

    MTPSpecStatePublicationResult result =
        publishAcceptedMTPSpecState(
            publishPlan(/*accepted_count=*/1),
            {&optional_skipped, &mandatory_skipped},
            DeviceId::cpu(),
            /*stream=*/nullptr,
            /*require_captured_stage=*/true);

    EXPECT_FALSE(result.ok);
    EXPECT_THAT(result.error, HasSubstr("required verifier capture"));
    EXPECT_TRUE(optional_skipped.restored_rows.empty());
    EXPECT_TRUE(mandatory_skipped.restored_rows.empty());
}

TEST(Test__MTPSpecStateContract, PublisherFailsAtomicallyWhenStageRestoreFails)
{
    FakeVerifierStateStage failed(/*captures=*/true, /*restore_ok=*/false);

    MTPSpecStatePublicationResult result =
        publishAcceptedMTPSpecState(
            publishPlan(/*accepted_count=*/1),
            {&failed},
            DeviceId::cpu(),
            /*stream=*/nullptr,
            /*require_captured_stage=*/true);

    EXPECT_FALSE(result.ok);
    EXPECT_THAT(result.error, HasSubstr("failed restoring verifier row 0"));
    EXPECT_THAT(failed.restored_rows, ElementsAre(0));
}

TEST(Test__MTPSpecStateContract, DeviceIndexedPublisherRestoresCapturedStagesOnExplicitGpuStream)
{
    FakeVerifierStateStage captured0(/*captures=*/true);
    FakeVerifierStateStage skipped(/*captures=*/false);
    FakeVerifierStateStage captured1(/*captures=*/true);
    int device_row = 2;
    int explicit_stream = 0;

    /*
     * The device-indexed path receives a pointer to a GPU-resident row index.
     * Unit tests use an ordinary int address as a stable sentinel; the contract
     * here is that the publisher forwards the pointer and stream untouched to
     * the stages, not that it dereferences device memory on the host.
     */
    MTPSpecStatePublicationResult result =
        publishAcceptedMTPSpecStateFromDeviceVerifierRow(
            publishPlan(/*accepted_count=*/3),
            &device_row,
            {&captured0, &skipped, &captured1},
            DeviceId::cuda(0),
            &explicit_stream,
            /*require_captured_stage=*/true);

    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_EQ(result.restored_stage_count, 2);
    EXPECT_EQ(result.skipped_stage_count, 1);
    EXPECT_TRUE(captured0.restored_rows.empty())
        << "Device-indexed publication must not fall back to host row restore.";
    EXPECT_TRUE(captured1.restored_rows.empty())
        << "Device-indexed publication must not fall back to host row restore.";
    EXPECT_TRUE(skipped.device_restored_rows.empty());
    EXPECT_THAT(captured0.device_restored_rows, ElementsAre(&device_row));
    EXPECT_THAT(captured1.device_restored_rows, ElementsAre(&device_row));
    EXPECT_THAT(captured0.device_streams, ElementsAre(&explicit_stream));
    EXPECT_THAT(captured1.device_streams, ElementsAre(&explicit_stream));
}

TEST(Test__MTPSpecStateContract, DeviceIndexedPublisherRejectsMissingMandatoryVerifierStateStage)
{
    FakeVerifierStateStage mandatory_skipped(
        /*captures=*/false,
        /*restore_ok=*/true,
        /*required_for_publication=*/true);
    int device_row = 0;
    int explicit_stream = 0;

    MTPSpecStatePublicationResult result =
        publishAcceptedMTPSpecStateFromDeviceVerifierRow(
            publishPlan(/*accepted_count=*/1),
            &device_row,
            {&mandatory_skipped},
            DeviceId::cuda(0),
            &explicit_stream,
            /*require_captured_stage=*/true);

    EXPECT_FALSE(result.ok);
    EXPECT_THAT(result.error, HasSubstr("required verifier capture"));
    EXPECT_TRUE(mandatory_skipped.device_restored_rows.empty());
}

TEST(Test__MTPSpecStateContract, DeviceIndexedPublisherRejectsCpuDevice)
{
    FakeVerifierStateStage captured(/*captures=*/true);
    int device_row = 0;
    int explicit_stream = 0;

    MTPSpecStatePublicationResult result =
        publishAcceptedMTPSpecStateFromDeviceVerifierRow(
            publishPlan(/*accepted_count=*/1),
            &device_row,
            {&captured},
            DeviceId::cpu(),
            &explicit_stream,
            /*require_captured_stage=*/true);

    EXPECT_FALSE(result.ok);
    EXPECT_THAT(result.error, HasSubstr("requires a GPU device"));
    EXPECT_TRUE(captured.device_restored_rows.empty());
}

TEST(Test__MTPSpecStateContract, DeviceIndexedPublisherRejectsNullStreamAndRow)
{
    FakeVerifierStateStage captured(/*captures=*/true);
    int device_row = 0;
    int explicit_stream = 0;

    MTPSpecStatePublicationResult null_stream =
        publishAcceptedMTPSpecStateFromDeviceVerifierRow(
            publishPlan(/*accepted_count=*/1),
            &device_row,
            {&captured},
            DeviceId::cuda(0),
            /*stream=*/nullptr,
            /*require_captured_stage=*/true);
    EXPECT_FALSE(null_stream.ok);
    EXPECT_THAT(null_stream.error, HasSubstr("explicit non-null stream"));

    MTPSpecStatePublicationResult null_row =
        publishAcceptedMTPSpecStateFromDeviceVerifierRow(
            publishPlan(/*accepted_count=*/1),
            /*device_verifier_restore_row=*/nullptr,
            {&captured},
            DeviceId::cuda(0),
            &explicit_stream,
            /*require_captured_stage=*/true);
    EXPECT_FALSE(null_row.ok);
    EXPECT_THAT(null_row.error, HasSubstr("null row pointer"));
    EXPECT_TRUE(captured.device_restored_rows.empty());
}

TEST(Test__MTPSpecStateContract, BatchedDeviceIndexedPublisherUsesBatchRestoreHook)
{
    FakeVerifierStateStage captured0(/*captures=*/true);
    FakeVerifierStateStage skipped(/*captures=*/false);
    FakeVerifierStateStage captured1(/*captures=*/true);
    int device_rows[4] = {0, -1, 2, 3};
    int explicit_stream = 0;

    MTPSpecStepPlan first = participantPlan(/*participant_id=*/0, /*accepted_count=*/2);
    MTPSpecStepPlan second = participantPlan(/*participant_id=*/1, /*accepted_count=*/0);
    second.target_cached_tokens = second.base_cached_tokens;
    second.accepted_state_slot_index = kMTPSpecDecodeInvalidToken;
    MTPSpecStepPlanBatch batch = planBatch({first, second});

    MTPSpecStatePublicationResult result =
        publishAcceptedMTPSpecStateFromDeviceVerifierRows(
            batch,
            device_rows,
            /*row_index_stride=*/2,
            {&captured0, &skipped, &captured1},
            DeviceId::cuda(0),
            &explicit_stream,
            /*require_captured_stage=*/true);

    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_EQ(result.restored_stage_count, 2);
    EXPECT_EQ(result.skipped_stage_count, 1);
    EXPECT_TRUE(captured0.device_restored_rows.empty())
        << "Batched publication must not loop the scalar device-index restore.";
    EXPECT_TRUE(captured1.device_restored_rows.empty())
        << "Batched publication must not loop the scalar device-index restore.";
    EXPECT_THAT(captured0.batch_device_restored_rows, ElementsAre(device_rows));
    EXPECT_THAT(captured1.batch_device_restored_rows, ElementsAre(device_rows));
    EXPECT_THAT(captured0.batch_request_counts, ElementsAre(2));
    EXPECT_THAT(captured1.batch_request_counts, ElementsAre(2));
    EXPECT_THAT(captured0.batch_row_index_strides, ElementsAre(2));
    EXPECT_THAT(captured1.batch_row_index_strides, ElementsAre(2));
    EXPECT_THAT(captured0.batch_streams, ElementsAre(&explicit_stream));
    EXPECT_THAT(captured1.batch_streams, ElementsAre(&explicit_stream));
    EXPECT_TRUE(skipped.batch_device_restored_rows.empty());
}

TEST(Test__MTPSpecStateContract, BatchedDeviceIndexedPublisherRejectsCpuAndNullStream)
{
    FakeVerifierStateStage captured(/*captures=*/true);
    int device_rows[2] = {0, 1};
    int explicit_stream = 0;
    MTPSpecStepPlanBatch batch =
        planBatch({participantPlan(/*participant_id=*/0, /*accepted_count=*/1)});

    MTPSpecStatePublicationResult cpu_result =
        publishAcceptedMTPSpecStateFromDeviceVerifierRows(
            batch,
            device_rows,
            /*row_index_stride=*/1,
            {&captured},
            DeviceId::cpu(),
            &explicit_stream,
            /*require_captured_stage=*/true);
    EXPECT_FALSE(cpu_result.ok);
    EXPECT_THAT(cpu_result.error, HasSubstr("requires a GPU device"));

    MTPSpecStatePublicationResult null_stream_result =
        publishAcceptedMTPSpecStateFromDeviceVerifierRows(
            batch,
            device_rows,
            /*row_index_stride=*/1,
            {&captured},
            DeviceId::cuda(0),
            /*stream=*/nullptr,
            /*require_captured_stage=*/true);
    EXPECT_FALSE(null_stream_result.ok);
    EXPECT_THAT(null_stream_result.error, HasSubstr("explicit non-null stream"));
    EXPECT_TRUE(captured.batch_device_restored_rows.empty());
}

TEST(Test__MTPSpecStateContract, BatchedDeviceIndexedPublisherFailsWhenBatchHookFails)
{
    FakeVerifierStateStage captured(/*captures=*/true, /*restore_ok=*/false);
    int device_rows[2] = {0, 1};
    int explicit_stream = 0;
    MTPSpecStepPlanBatch batch =
        planBatch({participantPlan(/*participant_id=*/0, /*accepted_count=*/1)});

    MTPSpecStatePublicationResult result =
        publishAcceptedMTPSpecStateFromDeviceVerifierRows(
            batch,
            device_rows,
            /*row_index_stride=*/1,
            {&captured},
            DeviceId::cuda(0),
            &explicit_stream,
            /*require_captured_stage=*/true);

    EXPECT_FALSE(result.ok);
    EXPECT_THAT(result.error, HasSubstr("failed restoring verifier rows"));
    EXPECT_THAT(captured.batch_device_restored_rows, ElementsAre(device_rows));
    EXPECT_TRUE(captured.device_restored_rows.empty())
        << "Failure must still come from the batch contract, not a scalar fallback.";
}

TEST(Test__MTPSpecStateContract, DeviceIndexedGraphPublisherRestoresExecutionOrder)
{
    auto captured0 = std::make_unique<FakeVerifierStateStage>(/*captures=*/true);
    auto skipped = std::make_unique<FakeVerifierStateStage>(/*captures=*/false);
    auto captured1 = std::make_unique<FakeVerifierStateStage>(/*captures=*/true);

    FakeVerifierStateStage *captured0_ptr = captured0.get();
    FakeVerifierStateStage *skipped_ptr = skipped.get();
    FakeVerifierStateStage *captured1_ptr = captured1.get();

    ComputeGraph graph;
    graph.addNode("captured0", std::move(captured0), DeviceId::cuda(0));
    graph.addNode("skipped", std::move(skipped), DeviceId::cuda(0));
    graph.addNode("captured1", std::move(captured1), DeviceId::cuda(0));
    graph.addDependency("skipped", "captured0");
    graph.addDependency("captured1", "skipped");

    int device_row = 3;
    int explicit_stream = 0;
    MTPSpecStatePublicationResult result =
        publishAcceptedMTPSpecStateFromDeviceVerifierRow(
            publishPlan(/*accepted_count=*/3),
            &device_row,
            graph,
            DeviceId::cuda(0),
            &explicit_stream,
            /*require_captured_stage=*/true);

    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_EQ(result.restored_stage_count, 2);
    EXPECT_EQ(result.skipped_stage_count, 1);
    EXPECT_THAT(captured0_ptr->device_restored_rows, ElementsAre(&device_row));
    EXPECT_TRUE(skipped_ptr->device_restored_rows.empty());
    EXPECT_THAT(captured1_ptr->device_restored_rows, ElementsAre(&device_row));
    EXPECT_THAT(captured0_ptr->device_streams, ElementsAre(&explicit_stream));
    EXPECT_THAT(captured1_ptr->device_streams, ElementsAre(&explicit_stream));
}

TEST(Test__MTPSpecStateContract, PublisherReportsFirstFailedStageAndStops)
{
    FakeVerifierStateStage captured0(/*captures=*/true);
    FakeVerifierStateStage failed1(/*captures=*/true, /*restore_ok=*/false);
    FakeVerifierStateStage failed2(/*captures=*/true, /*restore_ok=*/false);
    FakeVerifierStateStage skipped(/*captures=*/false);

    MTPSpecStatePublicationResult result =
        publishAcceptedMTPSpecState(
            publishPlan(/*accepted_count=*/3),
            {&captured0, &failed1, &skipped, &failed2},
            DeviceId::cpu(),
            /*stream=*/nullptr,
            /*require_captured_stage=*/true);

    EXPECT_FALSE(result.ok);
    EXPECT_THAT(result.error, HasSubstr("failed restoring verifier row 2"));
    EXPECT_THAT(result.error, HasSubstr("index 1"));
    EXPECT_THAT(captured0.restored_rows, ElementsAre(2));
    EXPECT_THAT(failed1.restored_rows, ElementsAre(2));
    EXPECT_TRUE(skipped.restored_rows.empty());
    /*
     * Publication is a fail-fast transaction.  Once a captured stage rejects
     * the restore row, later stages must not observe a partial live-state
     * mutation from an already-failed publication attempt.
     */
    EXPECT_TRUE(failed2.restored_rows.empty());
}

TEST(Test__MTPSpecStateContract, GraphPublisherRestoresCapturedStagesInExecutionOrder)
{
    auto captured0 = std::make_unique<FakeVerifierStateStage>(/*captures=*/true);
    auto skipped = std::make_unique<FakeVerifierStateStage>(/*captures=*/false);
    auto captured1 = std::make_unique<FakeVerifierStateStage>(/*captures=*/true);

    FakeVerifierStateStage *captured0_ptr = captured0.get();
    FakeVerifierStateStage *skipped_ptr = skipped.get();
    FakeVerifierStateStage *captured1_ptr = captured1.get();

    ComputeGraph graph;
    graph.addNode("captured0", std::move(captured0), DeviceId::cpu());
    graph.addNode("skipped", std::move(skipped), DeviceId::cpu());
    graph.addNode("captured1", std::move(captured1), DeviceId::cpu());
    graph.addDependency("skipped", "captured0");
    graph.addDependency("captured1", "skipped");

    int explicit_stream = 0;
    MTPSpecStatePublicationResult result =
        publishAcceptedMTPSpecState(
            publishPlan(/*accepted_count=*/3),
            graph,
            DeviceId::cpu(),
            &explicit_stream,
            /*require_captured_stage=*/true);

    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_EQ(result.restored_stage_count, 2);
    EXPECT_EQ(result.skipped_stage_count, 1);
    EXPECT_THAT(captured0_ptr->restored_rows, ElementsAre(2));
    EXPECT_TRUE(skipped_ptr->restored_rows.empty());
    EXPECT_THAT(captured1_ptr->restored_rows, ElementsAre(2));
    EXPECT_THAT(captured0_ptr->streams, ElementsAre(&explicit_stream));
    EXPECT_THAT(captured1_ptr->streams, ElementsAre(&explicit_stream));
}

TEST(Test__MTPSpecStateContract, GraphPublisherRejectsMissingStage)
{
    ComputeGraph graph;
    std::unique_ptr<IComputeStage> missing_stage;
    graph.addNode("missing_stage", std::move(missing_stage), DeviceId::cpu());

    MTPSpecStatePublicationResult result =
        publishAcceptedMTPSpecState(
            publishPlan(/*accepted_count=*/1),
            graph,
            DeviceId::cpu(),
            /*stream=*/nullptr,
            /*require_captured_stage=*/true);

    EXPECT_FALSE(result.ok);
    EXPECT_THAT(result.error, HasSubstr("without a stage"));
}
