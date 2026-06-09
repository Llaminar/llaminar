#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "backends/BackendManager.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/mtp/MTPDecodeCatchup.h"
#include "execution/mtp/MTPSpecDecodeMetadata.h"

#include <algorithm>

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

    ASSERT_THAT(reqs.buffers, SizeIs(23));
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

    const WorkspaceDescriptor *accepted_state_counts =
        findBuffer(reqs, MTPSpecDecodeWorkspaceBuffers::ACCEPTED_STATE_COUNTS);
    ASSERT_NE(accepted_state_counts, nullptr);
    EXPECT_EQ(accepted_state_counts->size_bytes, 2u * sizeof(int32_t));

    const WorkspaceDescriptor *speculative_state_slots =
        findBuffer(reqs, MTPSpecDecodeWorkspaceBuffers::SPECULATIVE_STATE_SLOT_INDICES);
    ASSERT_NE(speculative_state_slots, nullptr);
    EXPECT_EQ(speculative_state_slots->size_bytes, 8u * sizeof(int32_t));

    const WorkspaceDescriptor *committed_state_rows =
        findBuffer(reqs, MTPSpecDecodeWorkspaceBuffers::COMMITTED_STATE_ROWS);
    ASSERT_NE(committed_state_rows, nullptr);
    EXPECT_EQ(committed_state_rows->size_bytes, 2u * sizeof(int32_t));

    const WorkspaceDescriptor *accepted_state_slot_indices =
        findBuffer(reqs, MTPSpecDecodeWorkspaceBuffers::ACCEPTED_STATE_SLOT_INDICES);
    ASSERT_NE(accepted_state_slot_indices, nullptr);
    EXPECT_EQ(accepted_state_slot_indices->size_bytes, 2u * sizeof(int32_t));

    const WorkspaceDescriptor *bonus_ready_indices =
        findBuffer(reqs, MTPSpecDecodeWorkspaceBuffers::BONUS_READY_TOKEN_INDICES);
    ASSERT_NE(bonus_ready_indices, nullptr);
    EXPECT_EQ(bonus_ready_indices->size_bytes, 2u * sizeof(int32_t));

    const WorkspaceDescriptor *bonus_ready_state_slots =
        findBuffer(reqs, MTPSpecDecodeWorkspaceBuffers::BONUS_READY_STATE_SLOT_INDICES);
    ASSERT_NE(bonus_ready_state_slots, nullptr);
    EXPECT_EQ(bonus_ready_state_slots->size_bytes, 2u * sizeof(int32_t));

    const WorkspaceDescriptor *draft_tokens =
        findBuffer(reqs, MTPSpecDecodeWorkspaceBuffers::DRAFT_TOKENS);
    ASSERT_NE(draft_tokens, nullptr);
    EXPECT_EQ(draft_tokens->size_bytes, 6u * sizeof(int32_t));

    const WorkspaceDescriptor *sampled_tokens =
        findBuffer(reqs, MTPSpecDecodeWorkspaceBuffers::SAMPLED_TOKENS);
    ASSERT_NE(sampled_tokens, nullptr);
    EXPECT_EQ(sampled_tokens->size_bytes, 8u * sizeof(int32_t));

    const WorkspaceDescriptor *verifier_rows =
        findBuffer(reqs, MTPSpecDecodeWorkspaceBuffers::VERIFIER_LOGIT_ROWS);
    ASSERT_NE(verifier_rows, nullptr);
    EXPECT_EQ(verifier_rows->size_bytes, 8u * sizeof(int32_t));
}

TEST(Test__MTPSpecDecodeMetadata, WorkspaceBindingBindsEveryDeclaredBuffer)
{
    if (!hasCPUBackend())
    {
        initCPUBackend(-1);
    }

    MTPSpecDecodeMetadataShape shape;
    shape.max_requests = 2;
    shape.max_draft_tokens = 3;

    MTPSpecDecodeMetadataWorkspaceBinding binding(shape);
    WorkspaceRequirements reqs = binding.getWorkspaceRequirements(0, 0, 0);

    DeviceWorkspaceManager workspace(DeviceId::cpu(), 64 * 1024);
    ASSERT_TRUE(workspace.allocate(reqs));

    binding.bindWorkspace(&workspace);
    ASSERT_TRUE(binding.hasWorkspace()) << binding.bindingError();
    EXPECT_TRUE(binding.bindingError().empty());

    const auto &ptrs = binding.devicePointers();
    EXPECT_EQ(ptrs.draft_counts,
              workspace.getBuffer(MTPSpecDecodeWorkspaceBuffers::DRAFT_COUNTS));
    EXPECT_EQ(ptrs.accepted_draft_prefixes,
              workspace.getBuffer(MTPSpecDecodeWorkspaceBuffers::ACCEPTED_DRAFT_PREFIXES));
    EXPECT_EQ(ptrs.query_start_locs,
              workspace.getBuffer(MTPSpecDecodeWorkspaceBuffers::QUERY_START_LOCS));
    EXPECT_EQ(ptrs.accepted_state_counts,
              workspace.getBuffer(MTPSpecDecodeWorkspaceBuffers::ACCEPTED_STATE_COUNTS));
    EXPECT_EQ(ptrs.speculative_state_slot_indices,
              workspace.getBuffer(MTPSpecDecodeWorkspaceBuffers::SPECULATIVE_STATE_SLOT_INDICES));
    EXPECT_EQ(ptrs.committed_state_indices,
              workspace.getBuffer(MTPSpecDecodeWorkspaceBuffers::COMMITTED_STATE_INDICES));
    EXPECT_EQ(ptrs.accepted_state_slot_indices,
              workspace.getBuffer(MTPSpecDecodeWorkspaceBuffers::ACCEPTED_STATE_SLOT_INDICES));
    EXPECT_EQ(ptrs.bonus_ready_token_rows,
              workspace.getBuffer(MTPSpecDecodeWorkspaceBuffers::BONUS_READY_TOKEN_ROWS));
    EXPECT_EQ(ptrs.bonus_ready_state_slot_indices,
              workspace.getBuffer(MTPSpecDecodeWorkspaceBuffers::BONUS_READY_STATE_SLOT_INDICES));
    EXPECT_EQ(ptrs.sampled_tokens,
              workspace.getBuffer(MTPSpecDecodeWorkspaceBuffers::SAMPLED_TOKENS));
    EXPECT_EQ(ptrs.verifier_logit_rows,
              workspace.getBuffer(MTPSpecDecodeWorkspaceBuffers::VERIFIER_LOGIT_ROWS));

    ptrs.accepted_draft_prefixes[0] = 2;
    ptrs.accepted_state_counts[0] = 2;
    ptrs.speculative_state_slot_indices[0] = 4;
    ptrs.committed_state_rows[0] = 1;
    ptrs.accepted_state_slot_indices[0] = 4;
    ptrs.sampled_tokens[3] = 42;
    ptrs.verifier_logit_rows[0] = 1;
    EXPECT_EQ(ptrs.accepted_draft_prefixes[0], 2);
    EXPECT_EQ(ptrs.accepted_state_counts[0], 2);
    EXPECT_EQ(ptrs.speculative_state_slot_indices[0], 4);
    EXPECT_EQ(ptrs.committed_state_rows[0], 1);
    EXPECT_EQ(ptrs.accepted_state_slot_indices[0], 4);
    EXPECT_EQ(ptrs.sampled_tokens[3], 42);
    EXPECT_EQ(ptrs.verifier_logit_rows[0], 1);

    binding.unbindWorkspace();
    EXPECT_FALSE(binding.hasWorkspace());
    EXPECT_FALSE(binding.devicePointers().complete());
}

TEST(Test__MTPSpecDecodeMetadata, WorkspaceBindingDoesNotRequireCommittedRowCompatibilityBuffers)
{
    if (!hasCPUBackend())
    {
        initCPUBackend(-1);
    }

    MTPSpecDecodeMetadataShape shape;
    shape.max_requests = 2;
    shape.max_draft_tokens = 3;

    MTPSpecDecodeMetadataWorkspaceBinding binding(shape);
    WorkspaceRequirements reqs = binding.getWorkspaceRequirements(0, 0, 0);
    reqs.buffers.erase(
        std::remove_if(
            reqs.buffers.begin(),
            reqs.buffers.end(),
            [](const WorkspaceDescriptor &desc)
            {
                return desc.name == MTPSpecDecodeWorkspaceBuffers::COMMITTED_STATE_ROWS ||
                       desc.name == MTPSpecDecodeWorkspaceBuffers::COMMITTED_STATE_INDICES;
            }),
        reqs.buffers.end());

    DeviceWorkspaceManager workspace(DeviceId::cpu(), 64 * 1024);
    ASSERT_TRUE(workspace.allocate(reqs));

    binding.bindWorkspace(&workspace);
    ASSERT_TRUE(binding.hasWorkspace()) << binding.bindingError();
    const auto &ptrs = binding.devicePointers();
    EXPECT_EQ(ptrs.committed_state_rows, nullptr);
    EXPECT_EQ(ptrs.committed_state_indices, nullptr);
    ASSERT_NE(ptrs.accepted_state_counts, nullptr);
    ASSERT_NE(ptrs.accepted_state_slot_indices, nullptr);

    MTPSpecDecodeRequest request0;
    request0.request_id = 0;
    request0.draft_tokens = {7, 9, 8};
    request0.sampled_tokens = {7, 9, 8, 4};

    MTPSpecDecodeRequest request1;
    request1.request_id = 1;
    request1.draft_tokens = {11, 12, 13};
    request1.sampled_tokens = {11, 77};

    MTPSpecDecodeMetadataBatch batch = buildMTPSpecDecodeMetadataBatch(
        shape,
        {request0, request1},
        {3, 2},
        {0, 0});
    ASSERT_TRUE(batch.ok) << batch.error;
    ASSERT_THAT(batch.committed_state_rows, ElementsAre(2, 0));
    ASSERT_THAT(batch.accepted_state_slot_indices, ElementsAre(2, 4));

    MTPSpecDecodeMetadataUploadResult upload =
        uploadMTPSpecDecodeMetadataBatch(
            batch,
            binding,
            DeviceId::cpu(),
            /*backend=*/nullptr,
            /*stream=*/nullptr);
    ASSERT_TRUE(upload.ok) << upload.error;
    EXPECT_THAT(std::vector<int32_t>(
                    ptrs.accepted_state_slot_indices,
                    ptrs.accepted_state_slot_indices + 2),
                ElementsAre(2, 4));
    EXPECT_THAT(std::vector<int32_t>(
                    ptrs.accepted_state_counts,
                    ptrs.accepted_state_counts + 2),
                ElementsAre(3, 1));
}

TEST(Test__MTPSpecDecodeMetadata, WorkspaceBindingCanRequestLargerAllocatorShape)
{
    MTPSpecDecodeMetadataShape shape;
    shape.max_requests = 1;
    shape.max_draft_tokens = 2;

    MTPSpecDecodeMetadataWorkspaceBinding binding(shape);
    WorkspaceRequirements reqs = binding.getWorkspaceRequirements(
        /*m=*/3,
        /*n=*/4,
        /*k=*/0);

    const WorkspaceDescriptor *draft_counts =
        findBuffer(reqs, MTPSpecDecodeWorkspaceBuffers::DRAFT_COUNTS);
    ASSERT_NE(draft_counts, nullptr);
    EXPECT_EQ(draft_counts->size_bytes, 3u * sizeof(int32_t));

    const WorkspaceDescriptor *sampled_tokens =
        findBuffer(reqs, MTPSpecDecodeWorkspaceBuffers::SAMPLED_TOKENS);
    ASSERT_NE(sampled_tokens, nullptr);
    EXPECT_EQ(sampled_tokens->size_bytes, 15u * sizeof(int32_t));

    const WorkspaceDescriptor *verifier_rows =
        findBuffer(reqs, MTPSpecDecodeWorkspaceBuffers::VERIFIER_LOGIT_ROWS);
    ASSERT_NE(verifier_rows, nullptr);
    EXPECT_EQ(verifier_rows->size_bytes, 15u * sizeof(int32_t));
}

TEST(Test__MTPSpecDecodeMetadata, WorkspaceBindingReportsIncompleteBinding)
{
    if (!hasCPUBackend())
    {
        initCPUBackend(-1);
    }

    MTPSpecDecodeMetadataShape shape;
    shape.max_requests = 1;
    shape.max_draft_tokens = 2;

    WorkspaceRequirements incomplete;
    incomplete.buffers.push_back({
        MTPSpecDecodeWorkspaceBuffers::DRAFT_COUNTS,
        sizeof(int32_t),
        256,
        true});

    DeviceWorkspaceManager workspace(DeviceId::cpu(), 4096);
    ASSERT_TRUE(workspace.allocate(incomplete));

    MTPSpecDecodeMetadataWorkspaceBinding binding(shape);
    binding.bindWorkspace(&workspace);
    EXPECT_FALSE(binding.hasWorkspace());
    EXPECT_FALSE(binding.bindingError().empty());
    EXPECT_THAT(binding.bindingError(), HasSubstr("missing buffer"));
}

TEST(Test__MTPSpecDecodeMetadata, UploadBatchCopiesToBoundWorkspace)
{
    if (!hasCPUBackend())
    {
        initCPUBackend(-1);
    }

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
            /*stopped_flags=*/{0, 1});
    ASSERT_TRUE(batch.ok) << batch.error;

    MTPSpecDecodeMetadataWorkspaceBinding binding(shape);
    DeviceWorkspaceManager workspace(DeviceId::cpu(), 64 * 1024);
    ASSERT_TRUE(workspace.allocate(binding.getWorkspaceRequirements(0, 0, 0)));
    binding.bindWorkspace(&workspace);
    ASSERT_TRUE(binding.hasWorkspace()) << binding.bindingError();

    MTPSpecDecodeMetadataUploadResult upload =
        uploadMTPSpecDecodeMetadataBatch(
            batch,
            binding,
            DeviceId::cpu(),
            /*backend=*/nullptr,
            /*stream=*/nullptr);
    ASSERT_TRUE(upload.ok) << upload.error;

    size_t expected_bytes = 0;
    auto add_bytes = [&](const std::vector<int32_t> &values)
    {
        expected_bytes += values.size() * sizeof(int32_t);
    };
    add_bytes(batch.draft_counts);
    add_bytes(batch.target_query_lens);
    add_bytes(batch.valid_sampled_counts);
    add_bytes(batch.accepted_draft_prefixes);
    add_bytes(batch.committed_output_counts);
    add_bytes(batch.rejected_token_counts);
    add_bytes(batch.token_indices_to_sample);
    add_bytes(batch.next_condition_tokens);
    add_bytes(batch.all_drafts_accepted_flags);
    add_bytes(batch.stopped_flags);
    add_bytes(batch.query_start_locs);
    add_bytes(batch.state_indices);
    add_bytes(batch.accepted_state_counts);
    add_bytes(batch.speculative_state_slot_indices);
    add_bytes(batch.committed_state_rows);
    add_bytes(batch.committed_state_indices);
    add_bytes(batch.accepted_state_slot_indices);
    add_bytes(batch.bonus_ready_token_rows);
    add_bytes(batch.bonus_ready_token_indices);
    add_bytes(batch.bonus_ready_state_slot_indices);
    add_bytes(batch.draft_tokens);
    add_bytes(batch.sampled_tokens);
    EXPECT_EQ(upload.bytes_uploaded, expected_bytes);

    const auto &ptrs = binding.devicePointers();
    EXPECT_THAT(std::vector<int32_t>(ptrs.draft_counts, ptrs.draft_counts + 2),
                ElementsAre(3, 3));
    EXPECT_THAT(std::vector<int32_t>(
                    ptrs.accepted_draft_prefixes,
                    ptrs.accepted_draft_prefixes + 2),
                ElementsAre(3, 1));
    EXPECT_THAT(std::vector<int32_t>(ptrs.stopped_flags, ptrs.stopped_flags + 2),
                ElementsAre(0, 1));
    EXPECT_THAT(std::vector<int32_t>(ptrs.state_indices, ptrs.state_indices + 8),
                ElementsAre(0, 1, 2, 3, 4, 5, 6, 7));
    EXPECT_THAT(std::vector<int32_t>(
                    ptrs.accepted_state_counts,
                    ptrs.accepted_state_counts + 2),
                ElementsAre(3, 1));
    EXPECT_THAT(std::vector<int32_t>(
                    ptrs.speculative_state_slot_indices,
                    ptrs.speculative_state_slot_indices + 8),
                ElementsAre(0, 1, 2, 3, 4, 5, 6, 7));
    EXPECT_THAT(std::vector<int32_t>(
                    ptrs.committed_state_rows,
                    ptrs.committed_state_rows + 2),
                ElementsAre(2, 0));
    EXPECT_THAT(std::vector<int32_t>(
                    ptrs.committed_state_indices,
                    ptrs.committed_state_indices + 2),
                ElementsAre(2, 4));
    EXPECT_THAT(std::vector<int32_t>(
                    ptrs.accepted_state_slot_indices,
                    ptrs.accepted_state_slot_indices + 2),
                ElementsAre(2, 4));
    EXPECT_THAT(std::vector<int32_t>(
                    ptrs.bonus_ready_token_rows,
                    ptrs.bonus_ready_token_rows + 2),
                ElementsAre(3, kMTPSpecDecodeInvalidToken));
    EXPECT_THAT(std::vector<int32_t>(
                    ptrs.bonus_ready_token_indices,
                    ptrs.bonus_ready_token_indices + 2),
                ElementsAre(3, kMTPSpecDecodeInvalidToken));
    EXPECT_THAT(std::vector<int32_t>(
                    ptrs.bonus_ready_state_slot_indices,
                    ptrs.bonus_ready_state_slot_indices + 2),
                ElementsAre(3, kMTPSpecDecodeInvalidToken));
    EXPECT_THAT(std::vector<int32_t>(ptrs.sampled_tokens, ptrs.sampled_tokens + 8),
                ElementsAre(7, 9, 8, 4,
                            11, 77,
                            kMTPSpecDecodeInvalidToken,
                            kMTPSpecDecodeInvalidToken));
}

TEST(Test__MTPSpecDecodeMetadata, UploadBatchRejectsGpuNullStream)
{
    if (!hasCPUBackend())
    {
        initCPUBackend(-1);
    }

    MTPSpecDecodeMetadataShape shape;
    shape.max_requests = 1;
    shape.max_draft_tokens = 1;

    MTPSpecDecodeRequest request;
    request.vocab_size = 100;
    request.draft_tokens = {7};
    request.sampled_tokens = {7, 9};

    MTPSpecDecodeMetadataBatch batch =
        buildMTPSpecDecodeMetadataBatch(
            shape,
            {request},
            /*committed_output_counts=*/{1},
            /*stopped_flags=*/{0});
    ASSERT_TRUE(batch.ok) << batch.error;

    MTPSpecDecodeMetadataWorkspaceBinding binding(shape);
    DeviceWorkspaceManager workspace(DeviceId::cpu(), 64 * 1024);
    ASSERT_TRUE(workspace.allocate(binding.getWorkspaceRequirements(0, 0, 0)));
    binding.bindWorkspace(&workspace);
    ASSERT_TRUE(binding.hasWorkspace()) << binding.bindingError();

    MTPSpecDecodeMetadataUploadResult upload =
        uploadMTPSpecDecodeMetadataBatch(
            batch,
            binding,
            DeviceId::cuda(0),
            /*backend=*/nullptr,
            /*stream=*/nullptr);
    EXPECT_FALSE(upload.ok);
    EXPECT_THAT(upload.error, HasSubstr("explicit non-null stream"));
}

TEST(Test__MTPSpecDecodeMetadata, UploadVerifierInputPlanCopiesCompactRows)
{
    if (!hasCPUBackend())
    {
        initCPUBackend(-1);
    }

    MTPSpecDecodeMetadataShape shape;
    shape.max_requests = 1;
    shape.max_draft_tokens = 3;

    MTPSpecDecodeVerifierDraftRequest request;
    request.request_id = 0;
    request.draft_tokens = {10, 11, 12};

    MTPSpecDecodeVerifierInputPlan plan =
        buildMTPSpecDecodeVerifierInputPlan(shape, {request});
    ASSERT_TRUE(plan.ok) << plan.error;
    ASSERT_THAT(plan.verifier_logit_rows, ElementsAre(0, 1, 2));

    MTPSpecDecodeMetadataWorkspaceBinding binding(shape);
    DeviceWorkspaceManager workspace(DeviceId::cpu(), 64 * 1024);
    ASSERT_TRUE(workspace.allocate(binding.getWorkspaceRequirements(0, 0, 0)));
    binding.bindWorkspace(&workspace);
    ASSERT_TRUE(binding.hasWorkspace()) << binding.bindingError();

    MTPSpecDecodeMetadataUploadResult upload =
        uploadMTPSpecDecodeVerifierInputPlan(
            plan,
            binding,
            DeviceId::cpu(),
            /*backend=*/nullptr,
            /*stream=*/nullptr);
    ASSERT_TRUE(upload.ok) << upload.error;
    EXPECT_EQ(upload.bytes_uploaded, 3u * sizeof(int32_t));

    const auto &ptrs = binding.devicePointers();
    EXPECT_THAT(std::vector<int32_t>(
                    ptrs.verifier_logit_rows,
                    ptrs.verifier_logit_rows + 3),
                ElementsAre(0, 1, 2));
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
    EXPECT_THAT(batch.accepted_state_counts, ElementsAre(3, 1));
    EXPECT_THAT(batch.speculative_state_slot_indices,
                ElementsAre(0, 1, 2, 3, 4, 5, 6, 7));
    EXPECT_THAT(batch.committed_state_rows, ElementsAre(2, 0));
    EXPECT_THAT(batch.committed_state_indices, ElementsAre(2, 4));
    EXPECT_THAT(batch.accepted_state_slot_indices, ElementsAre(2, 4));
    EXPECT_THAT(batch.correction_replay_start_indices,
                ElementsAre(kMTPSpecDecodeInvalidToken, 1));
    EXPECT_THAT(batch.correction_replay_counts, ElementsAre(0, 1));
    EXPECT_THAT(batch.bonus_ready_token_rows,
                ElementsAre(3, kMTPSpecDecodeInvalidToken));
    EXPECT_THAT(batch.bonus_ready_token_indices,
                ElementsAre(3, kMTPSpecDecodeInvalidToken));
    EXPECT_THAT(batch.bonus_ready_state_slot_indices,
                ElementsAre(3, kMTPSpecDecodeInvalidToken));
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

TEST(Test__MTPSpecDecodeMetadata, BuildsVerifierInputPlanForCurrentRowIndexedContract)
{
    MTPSpecDecodeMetadataShape shape;
    shape.max_requests = 2;
    shape.max_draft_tokens = 3;

    MTPSpecDecodeVerifierDraftRequest request0;
    request0.request_id = 0;
    request0.draft_tokens = {7, 9, 8};

    MTPSpecDecodeVerifierDraftRequest request1;
    request1.request_id = 1;
    request1.draft_tokens = {11, 12};

    MTPSpecDecodeVerifierInputPlan plan =
        buildMTPSpecDecodeVerifierInputPlan(shape, {request0, request1});

    ASSERT_TRUE(plan.ok) << plan.error;
    EXPECT_EQ(plan.request_count, 2);
    EXPECT_EQ(plan.total_verifier_input_tokens, 5);
    EXPECT_EQ(plan.compact_logit_row_count, 5);
    EXPECT_THAT(plan.verifier_input_tokens, ElementsAre(7, 9, 8, 11, 12));
    EXPECT_THAT(plan.query_start_locs, ElementsAre(0, 3, 5));
    EXPECT_THAT(plan.verifier_logit_rows, ElementsAre(0, 1, 2, 3, 4));
    EXPECT_THAT(plan.bonus_logit_rows, ElementsAre(2, 4));
}

TEST(Test__MTPSpecDecodeMetadata, RejectsInvalidVerifierInputPlanShapes)
{
    MTPSpecDecodeMetadataShape shape;
    shape.max_requests = 1;
    shape.max_draft_tokens = 2;

    MTPSpecDecodeVerifierDraftRequest empty;
    empty.draft_tokens = {};
    EXPECT_FALSE(buildMTPSpecDecodeVerifierInputPlan(shape, {empty}).ok);

    MTPSpecDecodeVerifierDraftRequest too_deep;
    too_deep.draft_tokens = {7, 9, 8};
    MTPSpecDecodeVerifierInputPlan deep_plan =
        buildMTPSpecDecodeVerifierInputPlan(shape, {too_deep});
    EXPECT_FALSE(deep_plan.ok);
    EXPECT_THAT(deep_plan.error, HasSubstr("max_draft_tokens"));

    MTPSpecDecodeVerifierDraftRequest request0;
    request0.draft_tokens = {1};
    MTPSpecDecodeVerifierDraftRequest request1;
    request1.draft_tokens = {2};
    EXPECT_FALSE(buildMTPSpecDecodeVerifierInputPlan(
                     shape,
                     {request0, request1})
                     .ok)
        << "request_count must stay inside the metadata shape";
}

TEST(Test__MTPSpecDecodeMetadata, BuildsMetadataFromGreedyCatchupAcceptAll)
{
    MTPSpecDecodeMetadataShape shape;
    shape.max_requests = 1;
    shape.max_draft_tokens = 3;

    MTPDecodeCatchupGreedyRequest request;
    request.draft_tokens = {7, 9, 8};

    MTPDecodeCatchupGreedyResult result;
    result.ok = true;
    result.accepted_tokens = {7, 9, 8};
    result.all_speculative_accepted = true;
    result.stopped_on_output = false;
    result.accepted_speculative_prefix = 2;
    result.ready_token = 4;

    MTPSpecDecodeMetadataBatch batch =
        buildMTPSpecDecodeMetadataBatchFromGreedyCatchup(
            shape,
            /*request_id=*/0,
            /*vocab_size=*/100,
            request,
            result);

    ASSERT_TRUE(batch.ok) << batch.error;
    ASSERT_THAT(batch.transactions, SizeIs(1));
    EXPECT_TRUE(batch.transactions.front().allDraftsAccepted());
    EXPECT_THAT(batch.valid_sampled_counts, ElementsAre(4));
    EXPECT_THAT(batch.accepted_draft_prefixes, ElementsAre(3));
    EXPECT_THAT(batch.committed_output_counts, ElementsAre(3));
    EXPECT_THAT(batch.next_condition_tokens, ElementsAre(4));
    EXPECT_THAT(batch.accepted_state_counts, ElementsAre(3));
    EXPECT_THAT(batch.committed_state_rows, ElementsAre(2));
    EXPECT_THAT(batch.accepted_state_slot_indices, ElementsAre(2));
    EXPECT_THAT(batch.correction_replay_start_indices,
                ElementsAre(kMTPSpecDecodeInvalidToken));
    EXPECT_THAT(batch.correction_replay_counts, ElementsAre(0));
    EXPECT_THAT(batch.bonus_ready_token_rows, ElementsAre(3));
    EXPECT_THAT(batch.bonus_ready_state_slot_indices, ElementsAre(3));
    EXPECT_THAT(batch.sampled_tokens, ElementsAre(7, 9, 8, 4));
}

TEST(Test__MTPSpecDecodeMetadata, BuildsMetadataFromGreedyCatchupRejectAfterPrefix)
{
    MTPSpecDecodeMetadataShape shape;
    shape.max_requests = 1;
    shape.max_draft_tokens = 3;

    MTPDecodeCatchupGreedyRequest request;
    request.draft_tokens = {7, 9, 8};

    MTPDecodeCatchupGreedyResult result;
    result.ok = true;
    result.accepted_tokens = {7, 9, 3};
    result.all_speculative_accepted = false;
    result.stopped_on_output = false;
    result.accepted_speculative_prefix = 1;
    result.rejected_verified_token = 3;
    result.ready_token = 11;

    MTPSpecDecodeMetadataBatch batch =
        buildMTPSpecDecodeMetadataBatchFromGreedyCatchup(
            shape,
            /*request_id=*/0,
            /*vocab_size=*/100,
            request,
            result);

    ASSERT_TRUE(batch.ok) << batch.error;
    ASSERT_THAT(batch.transactions, SizeIs(1));
    EXPECT_FALSE(batch.transactions.front().allDraftsAccepted());
    EXPECT_THAT(batch.valid_sampled_counts, ElementsAre(3));
    EXPECT_THAT(batch.accepted_draft_prefixes, ElementsAre(2));
    EXPECT_THAT(batch.committed_output_counts, ElementsAre(3));
    EXPECT_THAT(batch.next_condition_tokens, ElementsAre(3));
    EXPECT_THAT(batch.accepted_state_counts, ElementsAre(2));
    EXPECT_THAT(batch.committed_state_rows, ElementsAre(1));
    EXPECT_THAT(batch.accepted_state_slot_indices, ElementsAre(1));
    EXPECT_THAT(batch.correction_replay_start_indices, ElementsAre(2));
    EXPECT_THAT(batch.correction_replay_counts, ElementsAre(1));
    EXPECT_THAT(batch.bonus_ready_token_rows,
                ElementsAre(kMTPSpecDecodeInvalidToken));
    EXPECT_THAT(batch.bonus_ready_state_slot_indices,
                ElementsAre(kMTPSpecDecodeInvalidToken));
    EXPECT_THAT(batch.sampled_tokens,
                ElementsAre(7, 9, 3, kMTPSpecDecodeInvalidToken));
}

TEST(Test__MTPSpecDecodeMetadata, UsesExplicitVerifierStateCommitCountForRejectedCorrection)
{
    MTPSpecDecodeMetadataShape shape;
    shape.max_requests = 1;
    shape.max_draft_tokens = 3;

    MTPDecodeCatchupGreedyRequest request;
    request.draft_tokens = {7, 9, 8};
    MTPDecodeCatchupGreedyResult result;
    result.ok = true;
    result.accepted_tokens = {7, 9, 3};
    result.verifier_tokens = {9, 3};
    result.all_speculative_accepted = false;
    result.accepted_speculative_prefix = 1;
    result.rejected_verified_token = 3;
    result.main_forward_token_count = 3;
    result.shifted_commit_count = 3;
    result.target_verifier_state_commit_count = 2;

    MTPSpecDecodeMetadataBatch batch =
        buildMTPSpecDecodeMetadataBatchFromGreedyCatchup(
            shape,
            /*request_id=*/0,
            /*vocab_size=*/100,
            request,
            result);

    ASSERT_TRUE(batch.ok) << batch.error;
    ASSERT_THAT(batch.committed_output_counts, SizeIs(1));
    ASSERT_THAT(batch.target_verifier_state_commit_counts, SizeIs(1));
    ASSERT_THAT(batch.committed_state_rows, SizeIs(1));
    EXPECT_EQ(batch.committed_output_counts[0], 3)
        << "the correction token is still emitted to the user";
    EXPECT_EQ(batch.target_verifier_state_commit_counts[0], 2)
        << "but target verifier rows only carry state through the accepted input prefix";
    EXPECT_EQ(batch.accepted_state_counts[0], 2);
    EXPECT_EQ(batch.committed_state_rows[0], 1);
    EXPECT_EQ(batch.committed_state_indices[0], 1);
    EXPECT_EQ(batch.accepted_state_slot_indices[0], 1);
    EXPECT_EQ(batch.correction_replay_start_indices[0], 2);
    EXPECT_EQ(batch.correction_replay_counts[0], 1);
    EXPECT_EQ(batch.bonus_ready_token_rows[0], kMTPSpecDecodeInvalidToken);
    EXPECT_EQ(batch.bonus_ready_state_slot_indices[0], kMTPSpecDecodeInvalidToken);
}

TEST(Test__MTPSpecDecodeMetadata, StateCommitPlanDoesNotReplayStoppedCorrectionSuffix)
{
    MTPSpecDecodeMetadataShape shape;
    shape.max_requests = 1;
    shape.max_draft_tokens = 3;

    MTPDecodeCatchupGreedyRequest request;
    request.draft_tokens = {7, 9, 8};
    request.stop_tokens = {3};
    MTPDecodeCatchupGreedyResult result;
    result.ok = true;
    result.accepted_tokens = {7, 3};
    result.verifier_tokens = {3};
    result.all_speculative_accepted = false;
    result.stopped_on_output = true;
    result.accepted_speculative_prefix = 0;
    result.rejected_verified_token = 3;
    result.main_forward_token_count = 3;
    result.shifted_commit_count = 2;
    result.target_verifier_state_commit_count = 1;

    MTPSpecDecodeMetadataBatch batch =
        buildMTPSpecDecodeMetadataBatchFromGreedyCatchup(
            shape,
            /*request_id=*/0,
            /*vocab_size=*/100,
            request,
            result);

    ASSERT_TRUE(batch.ok) << batch.error;
    EXPECT_THAT(batch.committed_output_counts, ElementsAre(2));
    EXPECT_THAT(batch.accepted_state_counts, ElementsAre(1));
    EXPECT_THAT(batch.accepted_state_slot_indices, ElementsAre(0));
    EXPECT_THAT(batch.correction_replay_start_indices,
                ElementsAre(kMTPSpecDecodeInvalidToken));
    EXPECT_THAT(batch.correction_replay_counts, ElementsAre(0));
    EXPECT_THAT(batch.bonus_ready_token_rows,
                ElementsAre(kMTPSpecDecodeInvalidToken));
}

TEST(Test__MTPSpecDecodeMetadata, RejectsStateCommitCountPastVerifierInputPrefix)
{
    MTPSpecDecodeMetadataShape shape;
    shape.max_requests = 1;
    shape.max_draft_tokens = 2;

    MTPSpecDecodeRequest request;
    request.vocab_size = 100;
    request.draft_tokens = {7, 9};
    request.sampled_tokens = {7, 9, 4};

    MTPSpecDecodeMetadataBatch batch =
        buildMTPSpecDecodeMetadataBatchWithStateCommitCounts(
            shape,
            {request},
            /*committed_output_counts=*/{2},
            /*target_verifier_state_commit_counts=*/{3},
            /*stopped_flags=*/{0});

    EXPECT_FALSE(batch.ok);
    EXPECT_THAT(batch.error, HasSubstr("state commit count"));
}

TEST(Test__MTPSpecDecodeMetadata, RejectsGreedyCatchupAcceptedPrefixDrift)
{
    MTPSpecDecodeMetadataShape shape;
    shape.max_requests = 1;
    shape.max_draft_tokens = 3;

    MTPDecodeCatchupGreedyRequest request;
    request.draft_tokens = {7, 9, 8};

    MTPDecodeCatchupGreedyResult result;
    result.ok = true;
    result.accepted_tokens = {7, 77};
    result.all_speculative_accepted = false;
    result.stopped_on_output = false;
    result.accepted_speculative_prefix = 2;
    result.target_verifier_state_commit_count = 2;
    result.rejected_verified_token = 77;
    result.ready_token = 11;

    MTPSpecDecodeMetadataBatch batch =
        buildMTPSpecDecodeMetadataBatchFromGreedyCatchup(
            shape,
            /*request_id=*/0,
            /*vocab_size=*/100,
            request,
            result);

    EXPECT_FALSE(batch.ok);
    EXPECT_THAT(batch.error, HasSubstr("accepted-prefix mismatch"));
}

TEST(Test__MTPSpecDecodeMetadata, StateCommitPlanDoesNotCommitBonusReadyTokenRow)
{
    MTPSpecDecodeMetadataShape shape;
    shape.max_requests = 1;
    shape.max_draft_tokens = 3;

    MTPSpecDecodeRequest accept_all;
    accept_all.vocab_size = 100;
    accept_all.draft_tokens = {7, 9, 8};
    accept_all.sampled_tokens = {7, 9, 8, 4};

    MTPSpecDecodeMetadataBatch batch =
        buildMTPSpecDecodeMetadataBatch(
            shape,
            {accept_all},
            /*committed_output_counts=*/{3},
            /*stopped_flags=*/{0});
    ASSERT_TRUE(batch.ok) << batch.error;

    MTPSpecDecodeStateCommitPlan plan =
        buildMTPSpecDecodeStateCommitPlan(batch);
    ASSERT_TRUE(plan.ok) << plan.error;
    EXPECT_THAT(plan.committed_state_rows, ElementsAre(2));
    EXPECT_THAT(plan.committed_state_indices, ElementsAre(2));
    EXPECT_THAT(plan.accepted_state_counts, ElementsAre(3));
    EXPECT_THAT(plan.accepted_state_slot_indices, ElementsAre(2));
    EXPECT_THAT(plan.correction_replay_start_indices,
                ElementsAre(kMTPSpecDecodeInvalidToken));
    EXPECT_THAT(plan.correction_replay_counts, ElementsAre(0));
    EXPECT_THAT(plan.bonus_ready_token_rows, ElementsAre(3));
    EXPECT_THAT(plan.bonus_ready_token_indices, ElementsAre(3));
    EXPECT_THAT(plan.bonus_ready_state_slot_indices, ElementsAre(3));
}

TEST(Test__MTPSpecDecodeMetadata, StateCommitPlanUsesAcceptedPrefixAfterReject)
{
    MTPSpecDecodeMetadataShape shape;
    shape.max_requests = 1;
    shape.max_draft_tokens = 3;

    MTPSpecDecodeRequest reject_after_prefix;
    reject_after_prefix.vocab_size = 100;
    reject_after_prefix.draft_tokens = {7, 9, 9};
    reject_after_prefix.sampled_tokens = {
        7,
        9,
        3,
        kMTPSpecDecodeInvalidToken};

    MTPSpecDecodeMetadataBatch batch =
        buildMTPSpecDecodeMetadataBatch(
            shape,
            {reject_after_prefix},
            /*committed_output_counts=*/{3},
            /*stopped_flags=*/{0});
    ASSERT_TRUE(batch.ok) << batch.error;

    MTPSpecDecodeStateCommitPlan plan =
        buildMTPSpecDecodeStateCommitPlan(batch);
    ASSERT_TRUE(plan.ok) << plan.error;
    EXPECT_THAT(plan.committed_state_rows, ElementsAre(1));
    EXPECT_THAT(plan.committed_state_indices, ElementsAre(1));
    EXPECT_THAT(plan.accepted_state_counts, ElementsAre(2));
    EXPECT_THAT(plan.accepted_state_slot_indices, ElementsAre(1));
    EXPECT_THAT(plan.correction_replay_start_indices, ElementsAre(2));
    EXPECT_THAT(plan.correction_replay_counts, ElementsAre(1));
    EXPECT_THAT(plan.bonus_ready_token_rows,
                ElementsAre(kMTPSpecDecodeInvalidToken));
    EXPECT_THAT(plan.bonus_ready_token_indices,
                ElementsAre(kMTPSpecDecodeInvalidToken));
    EXPECT_THAT(plan.bonus_ready_state_slot_indices,
                ElementsAre(kMTPSpecDecodeInvalidToken));
}

TEST(Test__MTPSpecDecodeMetadata, PublicationPlanMapsAcceptedStateCountToTargetCachedTokens)
{
    MTPSpecDecodeMetadataShape shape;
    shape.max_requests = 1;
    shape.max_draft_tokens = 3;

    MTPSpecDecodeRequest accept_all;
    accept_all.vocab_size = 100;
    accept_all.draft_tokens = {7, 9, 8};
    accept_all.sampled_tokens = {7, 9, 8, 4};

    MTPSpecDecodeMetadataBatch batch =
        buildMTPSpecDecodeMetadataBatch(
            shape,
            {accept_all},
            /*committed_output_counts=*/{3},
            /*stopped_flags=*/{0});
    ASSERT_TRUE(batch.ok) << batch.error;
    MTPSpecDecodeStateCommitPlan commit_plan =
        buildMTPSpecDecodeStateCommitPlan(batch);
    ASSERT_TRUE(commit_plan.ok) << commit_plan.error;

    MTPSpecDecodeStatePublicationPlan publication =
        buildMTPSpecDecodeStatePublicationPlan(
            commit_plan,
            /*base_cached_tokens=*/{128});
    ASSERT_TRUE(publication.ok) << publication.error;
    EXPECT_THAT(publication.base_cached_tokens, ElementsAre(128));
    EXPECT_THAT(publication.target_cached_tokens, ElementsAre(131));
    EXPECT_THAT(publication.accepted_state_counts, ElementsAre(3));
    EXPECT_THAT(publication.accepted_state_slot_indices, ElementsAre(2));
    EXPECT_THAT(publication.bonus_ready_token_rows, ElementsAre(3));
    EXPECT_THAT(publication.bonus_ready_state_slot_indices, ElementsAre(3));
    EXPECT_THAT(publication.correction_replay_counts, ElementsAre(0));
}

TEST(Test__MTPSpecDecodeMetadata, PublicationPlanCarriesCorrectionReplayAfterRejectedSuffix)
{
    MTPSpecDecodeMetadataShape shape;
    shape.max_requests = 1;
    shape.max_draft_tokens = 3;

    MTPSpecDecodeRequest reject_after_prefix;
    reject_after_prefix.vocab_size = 100;
    reject_after_prefix.draft_tokens = {7, 9, 9};
    reject_after_prefix.sampled_tokens = {
        7,
        9,
        3,
        kMTPSpecDecodeInvalidToken};

    MTPSpecDecodeMetadataBatch batch =
        buildMTPSpecDecodeMetadataBatch(
            shape,
            {reject_after_prefix},
            /*committed_output_counts=*/{3},
            /*stopped_flags=*/{0});
    ASSERT_TRUE(batch.ok) << batch.error;
    MTPSpecDecodeStateCommitPlan commit_plan =
        buildMTPSpecDecodeStateCommitPlan(batch);
    ASSERT_TRUE(commit_plan.ok) << commit_plan.error;

    MTPSpecDecodeStatePublicationPlan publication =
        buildMTPSpecDecodeStatePublicationPlan(
            commit_plan,
            /*base_cached_tokens=*/{64});
    ASSERT_TRUE(publication.ok) << publication.error;
    EXPECT_THAT(publication.target_cached_tokens, ElementsAre(66));
    EXPECT_THAT(publication.accepted_state_counts, ElementsAre(2));
    EXPECT_THAT(publication.accepted_state_slot_indices, ElementsAre(1));
    EXPECT_THAT(publication.correction_replay_start_indices, ElementsAre(2));
    EXPECT_THAT(publication.correction_replay_counts, ElementsAre(1));
    EXPECT_THAT(publication.bonus_ready_state_slot_indices,
                ElementsAre(kMTPSpecDecodeInvalidToken));
}

TEST(Test__MTPSpecDecodeMetadata, StateCommitPlanAllowsDiscardedRequestWithoutStateCommit)
{
    MTPSpecDecodeMetadataShape shape;
    shape.max_requests = 1;
    shape.max_draft_tokens = 2;

    MTPSpecDecodeRequest discarded;
    discarded.vocab_size = 100;
    discarded.draft_tokens = {7, 9};
    discarded.sampled_tokens = {
        kMTPSpecDecodeInvalidToken,
        kMTPSpecDecodeInvalidToken,
        kMTPSpecDecodeInvalidToken};
    discarded.discarded = true;
    discarded.backup_next_token = 42;

    MTPSpecDecodeMetadataBatch batch =
        buildMTPSpecDecodeMetadataBatch(
            shape,
            {discarded},
            /*committed_output_counts=*/{0},
            /*stopped_flags=*/{0});
    ASSERT_TRUE(batch.ok) << batch.error;

    MTPSpecDecodeStateCommitPlan plan =
        buildMTPSpecDecodeStateCommitPlan(batch);
    ASSERT_TRUE(plan.ok) << plan.error;
    EXPECT_THAT(plan.committed_state_rows,
                ElementsAre(kMTPSpecDecodeInvalidToken));
    EXPECT_THAT(plan.committed_state_indices,
                ElementsAre(kMTPSpecDecodeInvalidToken));
    EXPECT_THAT(plan.accepted_state_counts, ElementsAre(0));
    EXPECT_THAT(plan.accepted_state_slot_indices,
                ElementsAre(kMTPSpecDecodeInvalidToken));
    EXPECT_THAT(plan.correction_replay_start_indices,
                ElementsAre(kMTPSpecDecodeInvalidToken));
    EXPECT_THAT(plan.correction_replay_counts, ElementsAre(0));
    EXPECT_THAT(plan.bonus_ready_token_rows,
                ElementsAre(kMTPSpecDecodeInvalidToken));
    EXPECT_THAT(plan.bonus_ready_state_slot_indices,
                ElementsAre(kMTPSpecDecodeInvalidToken));

    MTPSpecDecodeStatePublicationPlan publication =
        buildMTPSpecDecodeStatePublicationPlan(
            plan,
            /*base_cached_tokens=*/{42});
    ASSERT_TRUE(publication.ok) << publication.error;
    EXPECT_THAT(publication.target_cached_tokens, ElementsAre(42));
    EXPECT_THAT(publication.accepted_state_counts, ElementsAre(0));
    EXPECT_THAT(publication.accepted_state_slot_indices,
                ElementsAre(kMTPSpecDecodeInvalidToken));
}

TEST(Test__MTPSpecDecodeMetadata, PublicationPlanRejectsInvalidBaseAndSlotState)
{
    MTPSpecDecodeMetadataShape shape;
    shape.max_requests = 1;
    shape.max_draft_tokens = 2;

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
    MTPSpecDecodeStateCommitPlan commit_plan =
        buildMTPSpecDecodeStateCommitPlan(batch);
    ASSERT_TRUE(commit_plan.ok) << commit_plan.error;

    MTPSpecDecodeStatePublicationPlan publication =
        buildMTPSpecDecodeStatePublicationPlan(commit_plan, {-1});
    ASSERT_FALSE(publication.ok);
    EXPECT_THAT(publication.error, HasSubstr("negative"));

    commit_plan.accepted_state_slot_indices[0] = 7;
    publication = buildMTPSpecDecodeStatePublicationPlan(commit_plan, {10});
    ASSERT_FALSE(publication.ok);
    EXPECT_THAT(publication.error, HasSubstr("slot index"));

    commit_plan.accepted_state_slot_indices[0] = 1;
    commit_plan.correction_replay_counts[0] = 1;
    commit_plan.correction_replay_start_indices[0] = 0;
    publication = buildMTPSpecDecodeStatePublicationPlan(commit_plan, {10});
    ASSERT_FALSE(publication.ok);
    EXPECT_THAT(publication.error, HasSubstr("correction replay"));
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
    EXPECT_THAT(batch.accepted_state_counts, ElementsAre(2, 0));
    EXPECT_THAT(batch.speculative_state_slot_indices,
                ElementsAre(0, 1, 2,
                            kMTPSpecDecodeInvalidToken,
                            kMTPSpecDecodeInvalidToken,
                            kMTPSpecDecodeInvalidToken,
                            kMTPSpecDecodeInvalidToken,
                            kMTPSpecDecodeInvalidToken,
                            kMTPSpecDecodeInvalidToken,
                            kMTPSpecDecodeInvalidToken));
    EXPECT_THAT(batch.committed_state_rows,
                ElementsAre(1, kMTPSpecDecodeInvalidToken));
    EXPECT_THAT(batch.committed_state_indices,
                ElementsAre(1, kMTPSpecDecodeInvalidToken));
    EXPECT_THAT(batch.accepted_state_slot_indices,
                ElementsAre(1, kMTPSpecDecodeInvalidToken));
    EXPECT_THAT(batch.bonus_ready_token_rows,
                ElementsAre(2, kMTPSpecDecodeInvalidToken));
    EXPECT_THAT(batch.bonus_ready_token_indices,
                ElementsAre(2, kMTPSpecDecodeInvalidToken));
    EXPECT_THAT(batch.bonus_ready_state_slot_indices,
                ElementsAre(2, kMTPSpecDecodeInvalidToken));
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

TEST(Test__MTPSpecDecodeMetadata, RejectsBonusReadyRowUnlessAllDraftsAccepted)
{
    MTPSpecDecodeMetadataShape shape;
    shape.max_requests = 1;
    shape.max_draft_tokens = 3;

    MTPSpecDecodeRequest reject_after_prefix;
    reject_after_prefix.vocab_size = 100;
    reject_after_prefix.draft_tokens = {7, 9, 8};
    reject_after_prefix.sampled_tokens = {
        7,
        99,
        kMTPSpecDecodeInvalidToken,
        kMTPSpecDecodeInvalidToken};

    MTPSpecDecodeMetadataBatch batch =
        buildMTPSpecDecodeMetadataBatch(
            shape,
            {reject_after_prefix},
            /*committed_output_counts=*/{1},
            /*stopped_flags=*/{0});

    EXPECT_FALSE(batch.ok);
    EXPECT_THAT(batch.error, HasSubstr("bonus ready token"));
}
