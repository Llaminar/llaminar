#include "execution/runner/MTPVerifierForwardExecutor.h"

#include "execution/local_execution/orchestrators/IInferenceRunner.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace llaminar2
{
namespace
{
    /**
     * @brief Small runner double that records which forward entrypoint is used.
     *
     * The production verifier path has three different entrypoints: host-token
     * single request, device-token single request, and padded request batch.
     * This fake keeps the test about routing and graph coordinates rather than
     * model math.
     */
    class RecordingInferenceRunner final : public IInferenceRunner
    {
    public:
        bool forward(const int *tokens, int seq_len) override
        {
            ++forward_count;
            last_forward_tokens.assign(tokens, tokens + seq_len);
            last_forward_seq_len = seq_len;
            return forward_success;
        }

        bool forwardWithDeviceTokenIds(
            const int *token_shadow,
            const void *token_ids_device,
            int seq_len) override
        {
            ++device_forward_count;
            last_device_token_ids = token_ids_device;
            last_forward_seq_len = seq_len;
            last_forward_tokens.assign(token_shadow, token_shadow + seq_len);
            return device_forward_success &&
                   token_shadow != nullptr &&
                   token_ids_device != nullptr;
        }

        bool forward_batch(const std::vector<std::vector<int>> &token_batches) override
        {
            ++batch_forward_count;
            last_token_batches = token_batches;
            return batch_forward_success;
        }

        bool forwardBatchWithDeviceTokenIds(
            const std::vector<std::vector<int>> &token_batches,
            const void *token_ids_device,
            int padded_seq_len) override
        {
            ++batch_device_forward_count;
            last_token_batches = token_batches;
            last_device_token_ids = token_ids_device;
            last_padded_seq_len = padded_seq_len;
            return batch_device_forward_success &&
                   token_ids_device != nullptr &&
                   padded_seq_len > 0;
        }

        bool setComputeRowIndexedAllPositionLogits(bool enabled, int row_count) override
        {
            if (enabled)
            {
                ++row_indexed_enable_count;
                last_row_indexed_count = row_count;
                row_indexed_enabled = true;
                return row_indexed_enable_success;
            }

            ++row_indexed_disable_count;
            row_indexed_enabled = false;
            return row_indexed_disable_success;
        }

        bool setComputeAllPositionLogits(bool enabled) override
        {
            if (enabled)
            {
                ++all_position_enable_count;
                all_position_enabled = true;
                return all_position_enable_success;
            }

            ++all_position_disable_count;
            all_position_enabled = false;
            return all_position_disable_success;
        }

        bool setMTPSpecVerifierInputPlan(
            const MTPSpecDecodeVerifierInputPlan &plan) override
        {
            ++set_plan_count;
            last_verifier_plan = plan;
            verifier_plan_installed = set_plan_success;
            return set_plan_success;
        }

        void clearMTPSpecVerifierInputPlan() override
        {
            ++clear_plan_count;
            verifier_plan_installed = false;
        }

        bool sampleGreedyFromAllPositionLogitsOnDeviceRows(
            int start_row,
            int row_count,
            int32_t *out_tokens) override
        {
            ++sample_rows_count;
            last_sample_start_row = start_row;
            last_sample_row_count = row_count;
            if (!sample_rows_success || !out_tokens ||
                start_row < 0 || row_count < 0 ||
                start_row + row_count >
                    static_cast<int>(scripted_verifier_samples.size()))
            {
                return false;
            }

            for (int i = 0; i < row_count; ++i)
            {
                out_tokens[i] =
                    scripted_verifier_samples[static_cast<size_t>(start_row + i)];
            }
            return true;
        }

        const float *logits() const override { return nullptr; }
        int vocab_size() const override { return 0; }
        void clear_cache() override {}
        int get_position() const override { return 0; }
        ExecutionPath executionPath() const override { return ExecutionPath::GRAPH; }
        const char *architecture() const override { return "test"; }

        int forward_count = 0;
        int device_forward_count = 0;
        int batch_forward_count = 0;
        int batch_device_forward_count = 0;
        int last_forward_seq_len = 0;
        int last_padded_seq_len = 0;
        const void *last_device_token_ids = nullptr;
        bool forward_success = true;
        bool device_forward_success = true;
        bool batch_forward_success = true;
        bool batch_device_forward_success = true;
        bool row_indexed_enable_success = true;
        bool row_indexed_disable_success = true;
        bool all_position_enable_success = true;
        bool all_position_disable_success = true;
        bool set_plan_success = true;
        bool sample_rows_success = true;
        bool row_indexed_enabled = false;
        bool all_position_enabled = false;
        bool verifier_plan_installed = false;
        int row_indexed_enable_count = 0;
        int row_indexed_disable_count = 0;
        int all_position_enable_count = 0;
        int all_position_disable_count = 0;
        int last_row_indexed_count = 0;
        int set_plan_count = 0;
        int clear_plan_count = 0;
        int sample_rows_count = 0;
        int last_sample_start_row = -1;
        int last_sample_row_count = -1;
        MTPSpecDecodeVerifierInputPlan last_verifier_plan;
        std::vector<int32_t> scripted_verifier_samples;
        std::vector<int> last_forward_tokens;
        std::vector<std::vector<int>> last_token_batches;
    };

    MTPSpecDecodeVerifierInputPlan buildVerifierPlan(
        int max_requests,
        int max_draft_tokens,
        const std::vector<std::vector<int32_t>> &draft_batches)
    {
        MTPSpecDecodeMetadataShape shape;
        shape.max_requests = max_requests;
        shape.max_draft_tokens = max_draft_tokens;

        std::vector<MTPSpecDecodeVerifierDraftRequest> requests;
        requests.reserve(draft_batches.size());
        for (size_t request = 0; request < draft_batches.size(); ++request)
        {
            MTPSpecDecodeVerifierDraftRequest draft_request;
            draft_request.request_id = static_cast<int>(request);
            draft_request.draft_tokens = draft_batches[request];
            requests.push_back(std::move(draft_request));
        }
        return buildMTPSpecDecodeVerifierInputPlan(shape, requests);
    }
} // namespace

TEST(Test__MTPVerifierForwardExecutor, SingleRequestUsesHostForward)
{
    RecordingInferenceRunner runner;
    MTPSpecDecodeVerifierInputPlan plan =
        buildVerifierPlan(/*max_requests=*/1, /*max_draft_tokens=*/3, {{5, 6, 7}});
    ASSERT_TRUE(plan.ok) << plan.error;

    MTPVerifierForwardExecutionResult result =
        executeMTPSpecVerifierForward(runner, plan);

    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_FALSE(result.used_batch_forward);
    EXPECT_FALSE(result.used_device_token_ids);
    EXPECT_EQ(runner.forward_count, 1);
    EXPECT_EQ(runner.device_forward_count, 0);
    EXPECT_EQ(runner.batch_forward_count, 0);
    EXPECT_EQ(runner.last_forward_seq_len, 3);
    EXPECT_EQ(runner.last_forward_tokens, (std::vector<int>{5, 6, 7}));
    EXPECT_EQ(result.graph_plan.verifier_logit_rows,
              (std::vector<int32_t>{0, 1, 2}));
}

TEST(Test__MTPVerifierForwardExecutor, SingleRequestCanUseDeviceTokenRow)
{
    RecordingInferenceRunner runner;
    MTPSpecDecodeVerifierInputPlan plan =
        buildVerifierPlan(/*max_requests=*/1, /*max_draft_tokens=*/2, {{17, 19}});
    ASSERT_TRUE(plan.ok) << plan.error;

    const int fake_device_tokens = 1234;
    MTPVerifierForwardExecutionOptions options;
    options.device_token_ids = &fake_device_tokens;

    MTPVerifierForwardExecutionResult result =
        executeMTPSpecVerifierForward(runner, plan, options);

    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_FALSE(result.used_batch_forward);
    EXPECT_TRUE(result.used_device_token_ids);
    EXPECT_EQ(runner.forward_count, 0);
    EXPECT_EQ(runner.device_forward_count, 1);
    EXPECT_EQ(runner.batch_forward_count, 0);
    EXPECT_EQ(runner.last_device_token_ids, &fake_device_tokens);
    EXPECT_EQ(runner.last_forward_seq_len, 2);
    EXPECT_EQ(runner.last_forward_tokens, (std::vector<int>{17, 19}));
}

TEST(Test__MTPVerifierForwardExecutor, RequestBatchUsesPaddedForwardBatch)
{
    RecordingInferenceRunner runner;
    MTPSpecDecodeVerifierInputPlan plan =
        buildVerifierPlan(
            /*max_requests=*/2,
            /*max_draft_tokens=*/3,
            {{7, 9}, {11, 12, 13}});
    ASSERT_TRUE(plan.ok) << plan.error;

    MTPVerifierForwardExecutionResult result =
        executeMTPSpecVerifierForward(runner, plan);

    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_TRUE(result.used_batch_forward);
    EXPECT_FALSE(result.used_device_token_ids);
    EXPECT_EQ(runner.forward_count, 0);
    EXPECT_EQ(runner.device_forward_count, 0);
    EXPECT_EQ(runner.batch_forward_count, 1);
    EXPECT_EQ(runner.last_token_batches,
              (std::vector<std::vector<int>>{{7, 9}, {11, 12, 13}}));
    EXPECT_EQ(result.graph_plan.padded_seq_len, 3);
    EXPECT_EQ(result.graph_plan.total_graph_tokens, 6);
    EXPECT_EQ(result.graph_plan.verifier_logit_rows,
              (std::vector<int32_t>{0, 1, 3, 4, 5}));
}

TEST(Test__MTPVerifierForwardExecutor, RequestBatchCanUsePaddedDeviceTokenRows)
{
    RecordingInferenceRunner runner;
    MTPSpecDecodeVerifierInputPlan plan =
        buildVerifierPlan(
            /*max_requests=*/2,
            /*max_draft_tokens=*/2,
            {{1, 2}, {3, 4}});
    ASSERT_TRUE(plan.ok) << plan.error;

    const int fake_device_tokens = 5678;
    MTPVerifierForwardExecutionOptions options;
    options.device_token_ids = &fake_device_tokens;

    MTPVerifierForwardExecutionResult result =
        executeMTPSpecVerifierForward(runner, plan, options);

    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_TRUE(result.used_batch_forward);
    EXPECT_TRUE(result.used_device_token_ids);
    EXPECT_EQ(runner.forward_count, 0);
    EXPECT_EQ(runner.device_forward_count, 0);
    EXPECT_EQ(runner.batch_forward_count, 0);
    EXPECT_EQ(runner.batch_device_forward_count, 1);
    EXPECT_EQ(runner.last_device_token_ids, &fake_device_tokens);
    EXPECT_EQ(runner.last_padded_seq_len, 2);
    EXPECT_EQ(runner.last_token_batches,
              (std::vector<std::vector<int>>{{1, 2}, {3, 4}}));
}

TEST(Test__MTPVerifierForwardExecutor, RequestBatchDeviceTokenFailureIsReported)
{
    RecordingInferenceRunner runner;
    runner.batch_device_forward_success = false;
    MTPSpecDecodeVerifierInputPlan plan =
        buildVerifierPlan(
            /*max_requests=*/2,
            /*max_draft_tokens=*/2,
            {{1, 2}, {3, 4}});
    ASSERT_TRUE(plan.ok) << plan.error;

    const int fake_device_tokens = 9012;
    MTPVerifierForwardExecutionOptions options;
    options.device_token_ids = &fake_device_tokens;

    MTPVerifierForwardExecutionResult result =
        executeMTPSpecVerifierForward(runner, plan, options);

    EXPECT_FALSE(result.ok);
    EXPECT_THAT(result.error, testing::HasSubstr("batched forward failed"));
    EXPECT_EQ(runner.forward_count, 0);
    EXPECT_EQ(runner.device_forward_count, 0);
    EXPECT_EQ(runner.batch_forward_count, 0);
    EXPECT_EQ(runner.batch_device_forward_count, 1);
}

TEST(Test__MTPVerifierForwardExecutor, GreedyBatchTransactionBuildsPublicationPlan)
{
    RecordingInferenceRunner runner;
    runner.scripted_verifier_samples = {9, 8, 4, 77, 123, 123};

    MTPDecodeCatchupGreedyRequest accept_all;
    accept_all.draft_tokens = {7, 9, 8};

    MTPDecodeCatchupGreedyRequest reject_after_first;
    reject_after_first.draft_tokens = {11, 12, 13};

    MTPGreedyVerifierBatchTransactionRequest request;
    request.shape.max_requests = 2;
    request.shape.max_draft_tokens = 3;
    request.request_ids = {10, 11};
    request.vocab_size = 100;
    request.requests = {accept_all, reject_after_first};
    request.base_cached_tokens = {100, 200};

    MTPGreedyVerifierBatchTransactionResult result =
        executeMTPGreedyVerifierBatchTransaction(runner, request);

    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_TRUE(result.forward.used_batch_forward);
    EXPECT_EQ(runner.batch_forward_count, 1);
    EXPECT_EQ(runner.last_token_batches,
              (std::vector<std::vector<int>>{{7, 9, 8}, {11, 12, 13}}));
    EXPECT_EQ(runner.row_indexed_enable_count, 1);
    EXPECT_EQ(runner.row_indexed_disable_count, 1);
    EXPECT_EQ(runner.all_position_enable_count, 1);
    EXPECT_EQ(runner.all_position_disable_count, 1);
    EXPECT_EQ(runner.last_row_indexed_count, 6);
    EXPECT_EQ(runner.set_plan_count, 1);
    EXPECT_EQ(runner.clear_plan_count, 1);
    EXPECT_FALSE(runner.row_indexed_enabled);
    EXPECT_FALSE(runner.all_position_enabled);
    EXPECT_FALSE(runner.verifier_plan_installed);
    EXPECT_EQ(runner.sample_rows_count, 1);
    EXPECT_EQ(runner.last_sample_start_row, 0);
    EXPECT_EQ(runner.last_sample_row_count, 6);
    EXPECT_EQ(result.sampled_verifier_rows,
              (std::vector<int32_t>{9, 8, 4, 77, 123, 123}));

    ASSERT_EQ(result.transaction_plan.step_plans.steps.size(), 2u);
    const MTPSpecStepPlan &first =
        result.transaction_plan.step_plans.steps[0];
    EXPECT_EQ(first.request_id, 10);
    EXPECT_EQ(first.accepted_count, 3);
    EXPECT_EQ(first.target_cached_tokens, 103);
    EXPECT_EQ(first.bonus_ready_state_slot_index, 3);

    const MTPSpecStepPlan &second =
        result.transaction_plan.step_plans.steps[1];
    EXPECT_EQ(second.request_id, 11);
    EXPECT_EQ(second.accepted_count, 1);
    EXPECT_EQ(second.target_cached_tokens, 201);
    EXPECT_TRUE(second.requiresCorrectionReplay());
}

TEST(Test__MTPVerifierForwardExecutor, GreedyBatchTransactionCanUsePaddedDeviceTokenRows)
{
    RecordingInferenceRunner runner;
    runner.scripted_verifier_samples = {9, 8, 4, 77};

    MTPDecodeCatchupGreedyRequest first;
    first.draft_tokens = {7, 9};
    MTPDecodeCatchupGreedyRequest second;
    second.draft_tokens = {11, 12};

    const int fake_device_tokens = 777;
    MTPGreedyVerifierBatchTransactionRequest request;
    request.shape.max_requests = 2;
    request.shape.max_draft_tokens = 2;
    request.request_ids = {10, 11};
    request.vocab_size = 100;
    request.requests = {first, second};
    request.base_cached_tokens = {100, 200};
    request.forward_options.device_token_ids = &fake_device_tokens;

    MTPGreedyVerifierBatchTransactionResult result =
        executeMTPGreedyVerifierBatchTransaction(runner, request);

    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_TRUE(result.forward.used_batch_forward);
    EXPECT_TRUE(result.forward.used_device_token_ids);
    EXPECT_EQ(runner.batch_forward_count, 0);
    EXPECT_EQ(runner.batch_device_forward_count, 1);
    EXPECT_EQ(runner.last_device_token_ids, &fake_device_tokens);
    EXPECT_EQ(runner.last_padded_seq_len, 2);
    EXPECT_EQ(runner.last_token_batches,
              (std::vector<std::vector<int>>{{7, 9}, {11, 12}}));
}

TEST(Test__MTPVerifierForwardExecutor, ScheduledGreedyBatchFeedsTransactionExecutor)
{
    RecordingInferenceRunner runner;
    runner.scripted_verifier_samples = {9, 8, 4, 77, 123, 123};

    MTPSpecRequestBatchScheduler scheduler(
        MTPSpecRequestBatchSchedulerConfig{
            /*max_request_batch=*/2,
            /*max_draft_tokens=*/3,
            MTPSpecRequestBatchMode::GREEDY});

    MTPSpecSchedulableRequest first;
    first.request_id = 10;
    first.compatibility_key = "qwen36-moe-cuda0";
    first.vocab_size = 100;
    first.base_cached_tokens = 100;
    first.greedy_request.draft_tokens = {7, 9, 8};

    MTPSpecSchedulableRequest second = first;
    second.request_id = 11;
    second.base_cached_tokens = 200;
    second.greedy_request.draft_tokens = {11, 12, 13};

    MTPSpecRequestBatch scheduled =
        scheduler.buildNextBatch({first, second});
    ASSERT_TRUE(scheduled.ok) << scheduled.error;

    MTPGreedyVerifierBatchTransactionResult result =
        executeMTPGreedyVerifierScheduledBatchTransaction(runner, scheduled);

    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_TRUE(result.forward.used_batch_forward);
    EXPECT_EQ(runner.batch_forward_count, 1);
    EXPECT_EQ(runner.last_token_batches,
              (std::vector<std::vector<int>>{{7, 9, 8}, {11, 12, 13}}));
    ASSERT_EQ(result.transaction_plan.step_plans.steps.size(), 2u);
    EXPECT_EQ(result.transaction_plan.step_plans.steps[0].request_id, 10);
    EXPECT_EQ(result.transaction_plan.step_plans.steps[1].request_id, 11);
}

TEST(Test__MTPVerifierForwardExecutor, ScheduledGreedyBatchCanUseForwardOptions)
{
    RecordingInferenceRunner runner;
    runner.scripted_verifier_samples = {9, 8, 4, 77};

    MTPSpecRequestBatchScheduler scheduler(
        MTPSpecRequestBatchSchedulerConfig{
            /*max_request_batch=*/2,
            /*max_draft_tokens=*/2,
            MTPSpecRequestBatchMode::GREEDY});

    MTPSpecSchedulableRequest first;
    first.request_id = 10;
    first.compatibility_key = "qwen36-moe-cuda0";
    first.vocab_size = 100;
    first.base_cached_tokens = 100;
    first.greedy_request.draft_tokens = {7, 9};
    first.verifier_input =
        MTPSpecVerifierInputPlacement::DEVICE_TOKEN_ROW;

    MTPSpecSchedulableRequest second = first;
    second.request_id = 11;
    second.base_cached_tokens = 200;
    second.greedy_request.draft_tokens = {11, 12};

    MTPSpecRequestBatch scheduled =
        scheduler.buildNextBatch({first, second});
    ASSERT_TRUE(scheduled.ok) << scheduled.error;

    const int fake_device_tokens = 888;
    MTPVerifierForwardExecutionOptions forward_options;
    forward_options.device_token_ids = &fake_device_tokens;

    MTPGreedyVerifierBatchTransactionResult result =
        executeMTPGreedyVerifierScheduledBatchTransaction(
            runner,
            scheduled,
            forward_options);

    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_TRUE(result.forward.used_device_token_ids);
    EXPECT_EQ(runner.batch_forward_count, 0);
    EXPECT_EQ(runner.batch_device_forward_count, 1);
    EXPECT_EQ(runner.last_device_token_ids, &fake_device_tokens);
    EXPECT_EQ(runner.last_token_batches,
              (std::vector<std::vector<int>>{{7, 9}, {11, 12}}));
}

TEST(Test__MTPVerifierForwardExecutor, ScheduledDeviceTokenBatchRequiresDevicePointer)
{
    RecordingInferenceRunner runner;
    MTPSpecRequestBatchScheduler scheduler(
        MTPSpecRequestBatchSchedulerConfig{
            /*max_request_batch=*/2,
            /*max_draft_tokens=*/2,
            MTPSpecRequestBatchMode::GREEDY});

    MTPSpecSchedulableRequest first;
    first.request_id = 10;
    first.compatibility_key = "qwen36-moe-cuda0";
    first.vocab_size = 100;
    first.base_cached_tokens = 100;
    first.greedy_request.draft_tokens = {7, 9};
    first.verifier_input =
        MTPSpecVerifierInputPlacement::DEVICE_TOKEN_ROW;

    MTPSpecSchedulableRequest second = first;
    second.request_id = 11;
    second.base_cached_tokens = 200;
    second.greedy_request.draft_tokens = {11, 12};

    MTPSpecRequestBatch scheduled =
        scheduler.buildNextBatch({first, second});
    ASSERT_TRUE(scheduled.ok) << scheduled.error;

    MTPGreedyVerifierBatchTransactionResult result =
        executeMTPGreedyVerifierScheduledBatchTransaction(runner, scheduled);

    EXPECT_FALSE(result.ok);
    EXPECT_THAT(result.error, testing::HasSubstr("device_token_ids"));
    EXPECT_EQ(runner.batch_forward_count, 0);
    EXPECT_EQ(runner.batch_device_forward_count, 0);
}

TEST(Test__MTPVerifierForwardExecutor, ScheduledHostTokenBatchRejectsDevicePointer)
{
    RecordingInferenceRunner runner;
    MTPSpecRequestBatchScheduler scheduler(
        MTPSpecRequestBatchSchedulerConfig{
            /*max_request_batch=*/2,
            /*max_draft_tokens=*/2,
            MTPSpecRequestBatchMode::GREEDY});

    MTPSpecSchedulableRequest first;
    first.request_id = 10;
    first.compatibility_key = "qwen36-moe-cuda0";
    first.vocab_size = 100;
    first.base_cached_tokens = 100;
    first.greedy_request.draft_tokens = {7, 9};

    MTPSpecSchedulableRequest second = first;
    second.request_id = 11;
    second.base_cached_tokens = 200;
    second.greedy_request.draft_tokens = {11, 12};

    MTPSpecRequestBatch scheduled =
        scheduler.buildNextBatch({first, second});
    ASSERT_TRUE(scheduled.ok) << scheduled.error;

    const int fake_device_tokens = 888;
    MTPVerifierForwardExecutionOptions forward_options;
    forward_options.device_token_ids = &fake_device_tokens;

    MTPGreedyVerifierBatchTransactionResult result =
        executeMTPGreedyVerifierScheduledBatchTransaction(
            runner,
            scheduled,
            forward_options);

    EXPECT_FALSE(result.ok);
    EXPECT_THAT(result.error, testing::HasSubstr("host-token"));
    EXPECT_EQ(runner.batch_forward_count, 0);
    EXPECT_EQ(runner.batch_device_forward_count, 0);
}

TEST(Test__MTPVerifierForwardExecutor, ScheduledTransactionRejectsNonGreedyBatch)
{
    RecordingInferenceRunner runner;

    MTPSpecRequestBatch scheduled;
    scheduled.ok = true;
    scheduled.mode = MTPSpecRequestBatchMode::STOCHASTIC;
    scheduled.request_count = 1;
    scheduled.shape.max_requests = 1;
    scheduled.shape.max_draft_tokens = 2;
    scheduled.request_ids = {12};
    scheduled.vocab_size = 100;
    scheduled.base_cached_tokens = {50};
    MTPDecodeCatchupGreedyRequest request;
    request.draft_tokens = {1, 2};
    scheduled.greedy_requests = {request};

    MTPGreedyVerifierBatchTransactionResult result =
        executeMTPGreedyVerifierScheduledBatchTransaction(runner, scheduled);

    EXPECT_FALSE(result.ok);
    EXPECT_THAT(result.error, testing::HasSubstr("not greedy"));
    EXPECT_EQ(runner.batch_forward_count, 0);
}

TEST(Test__MTPVerifierForwardExecutor, OwnedScheduledTransactionCommitsOnSuccess)
{
    RecordingInferenceRunner runner;
    runner.scripted_verifier_samples = {9, 8, 4, 77, 123, 123};

    MTPSpecRequestBatchOwner owner;
    MTPSpecSchedulableRequest first;
    first.request_id = 10;
    first.compatibility_key = "qwen36-moe-cuda0";
    first.vocab_size = 100;
    first.base_cached_tokens = 100;
    first.greedy_request.draft_tokens = {7, 9, 8};
    ASSERT_TRUE(owner.enqueueRequest(first));

    MTPSpecSchedulableRequest second = first;
    second.request_id = 11;
    second.base_cached_tokens = 200;
    second.greedy_request.draft_tokens = {11, 12, 13};
    ASSERT_TRUE(owner.enqueueRequest(second));

    MTPSpecSchedulableRequest deferred = first;
    deferred.request_id = 12;
    deferred.compatibility_key = "qwen36-moe-rocm0";
    ASSERT_TRUE(owner.enqueueRequest(deferred));

    MTPSpecRequestBatchScheduler scheduler(
        MTPSpecRequestBatchSchedulerConfig{
            /*max_request_batch=*/2,
            /*max_draft_tokens=*/3,
            MTPSpecRequestBatchMode::GREEDY});

    MTPOwnedGreedyVerifierBatchTransactionResult result =
        executeOwnedMTPGreedyVerifierScheduledBatchTransaction(
            runner,
            owner,
            scheduler);

    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_TRUE(result.committed);
    EXPECT_FALSE(result.released);
    EXPECT_FALSE(owner.hasInFlightBatch());
    ASSERT_EQ(owner.pendingCount(), 1u);
    EXPECT_EQ(owner.pendingRequests()[0].request_id, 12);
    EXPECT_THAT(result.scheduled_batch.request_ids, testing::ElementsAre(10, 11));
    ASSERT_EQ(result.transaction.transaction_plan.step_plans.steps.size(), 2u);
}

TEST(Test__MTPVerifierForwardExecutor, OwnedScheduledTransactionReleasesOnForwardFailure)
{
    RecordingInferenceRunner runner;
    runner.batch_forward_success = false;

    MTPSpecRequestBatchOwner owner;
    MTPSpecSchedulableRequest first;
    first.request_id = 20;
    first.compatibility_key = "qwen36-moe-cuda0";
    first.vocab_size = 100;
    first.base_cached_tokens = 100;
    first.greedy_request.draft_tokens = {7, 9};
    ASSERT_TRUE(owner.enqueueRequest(first));

    MTPSpecSchedulableRequest second = first;
    second.request_id = 21;
    second.base_cached_tokens = 200;
    second.greedy_request.draft_tokens = {11, 12};
    ASSERT_TRUE(owner.enqueueRequest(second));

    MTPSpecRequestBatchScheduler scheduler(
        MTPSpecRequestBatchSchedulerConfig{
            /*max_request_batch=*/2,
            /*max_draft_tokens=*/2,
            MTPSpecRequestBatchMode::GREEDY});

    MTPOwnedGreedyVerifierBatchTransactionResult result =
        executeOwnedMTPGreedyVerifierScheduledBatchTransaction(
            runner,
            owner,
            scheduler);

    EXPECT_FALSE(result.ok);
    EXPECT_TRUE(result.released);
    EXPECT_FALSE(result.committed);
    EXPECT_FALSE(owner.hasInFlightBatch());
    EXPECT_EQ(owner.pendingCount(), 2u);
    EXPECT_THAT(result.error, testing::HasSubstr("forward failed"));
    EXPECT_EQ(runner.row_indexed_enable_count, 1);
    EXPECT_EQ(runner.row_indexed_disable_count, 1);
    EXPECT_EQ(runner.all_position_enable_count, 1);
    EXPECT_EQ(runner.all_position_disable_count, 1);
    EXPECT_EQ(runner.clear_plan_count, 1);
}

TEST(Test__MTPVerifierForwardExecutor, OwnedScheduledTransactionReportsScheduleFailure)
{
    RecordingInferenceRunner runner;
    MTPSpecRequestBatchOwner owner;
    MTPSpecRequestBatchScheduler scheduler(
        MTPSpecRequestBatchSchedulerConfig{
            /*max_request_batch=*/2,
            /*max_draft_tokens=*/2,
            MTPSpecRequestBatchMode::GREEDY});

    MTPOwnedGreedyVerifierBatchTransactionResult result =
        executeOwnedMTPGreedyVerifierScheduledBatchTransaction(
            runner,
            owner,
            scheduler);

    EXPECT_FALSE(result.ok);
    EXPECT_FALSE(result.committed);
    EXPECT_FALSE(result.released);
    EXPECT_FALSE(owner.hasInFlightBatch());
    EXPECT_THAT(result.error, testing::HasSubstr("scheduling failed"));
    EXPECT_EQ(runner.batch_forward_count, 0);
}

TEST(Test__MTPVerifierForwardExecutor, GreedyBatchTransactionCleansUpAfterForwardFailure)
{
    RecordingInferenceRunner runner;
    runner.batch_forward_success = false;

    MTPDecodeCatchupGreedyRequest request0;
    request0.draft_tokens = {7, 9};
    MTPDecodeCatchupGreedyRequest request1;
    request1.draft_tokens = {11, 12};

    MTPGreedyVerifierBatchTransactionRequest request;
    request.shape.max_requests = 2;
    request.shape.max_draft_tokens = 2;
    request.request_ids = {10, 11};
    request.vocab_size = 100;
    request.requests = {request0, request1};
    request.base_cached_tokens = {100, 200};

    MTPGreedyVerifierBatchTransactionResult result =
        executeMTPGreedyVerifierBatchTransaction(runner, request);

    EXPECT_FALSE(result.ok);
    EXPECT_NE(result.error.find("forward failed"), std::string::npos);
    EXPECT_EQ(runner.batch_forward_count, 1);
    EXPECT_EQ(runner.row_indexed_enable_count, 1);
    EXPECT_EQ(runner.row_indexed_disable_count, 1);
    EXPECT_EQ(runner.all_position_enable_count, 1);
    EXPECT_EQ(runner.all_position_disable_count, 1);
    EXPECT_EQ(runner.clear_plan_count, 1);
    EXPECT_FALSE(runner.row_indexed_enabled);
    EXPECT_FALSE(runner.all_position_enabled);
    EXPECT_FALSE(runner.verifier_plan_installed);
    EXPECT_EQ(runner.sample_rows_count, 0);
}

} // namespace llaminar2
