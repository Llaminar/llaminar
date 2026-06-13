#include "execution/runner/MTPVerifierForwardExecutor.h"

#include "execution/local_execution/orchestrators/IInferenceRunner.h"

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

        const float *logits() const override { return nullptr; }
        int vocab_size() const override { return 0; }
        void clear_cache() override {}
        int get_position() const override { return 0; }
        ExecutionPath executionPath() const override { return ExecutionPath::GRAPH; }
        const char *architecture() const override { return "test"; }

        int forward_count = 0;
        int device_forward_count = 0;
        int batch_forward_count = 0;
        int last_forward_seq_len = 0;
        const void *last_device_token_ids = nullptr;
        bool forward_success = true;
        bool device_forward_success = true;
        bool batch_forward_success = true;
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

TEST(Test__MTPVerifierForwardExecutor, RequestBatchRejectsSingleRowDeviceTokenHook)
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

    EXPECT_FALSE(result.ok);
    EXPECT_NE(result.error.find("single-row device token input hook"),
              std::string::npos);
    EXPECT_EQ(runner.forward_count, 0);
    EXPECT_EQ(runner.device_forward_count, 0);
    EXPECT_EQ(runner.batch_forward_count, 0);
}

} // namespace llaminar2
