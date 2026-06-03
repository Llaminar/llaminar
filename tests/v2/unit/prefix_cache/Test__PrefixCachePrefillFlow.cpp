#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <algorithm>

#include "backends/GlobalDeviceAddress.h"
#include "config/OrchestrationConfig.h"
#include "execution/local_execution/engine/PrefillBucketUtils.h"
#include "execution/local_execution/orchestrators/IInferenceRunner.h"
#include "execution/mpi_orchestration/RankExecutionPlan.h"
#include "execution/runner/OrchestrationRunner.h"
#include "utils/DebugEnv.h"

using namespace llaminar2;
using namespace testing;

namespace
{
    class PrefixFlowMockRunner : public IInferenceRunner
    {
    public:
        bool forward(const int *tokens, int seq_len) override
        {
            ++forward_calls;
            last_forward_tokens.assign(tokens, tokens + seq_len);
            position += seq_len;
            if (all_position_logits_enabled)
            {
                all_position_logits.assign(static_cast<size_t>(seq_len) * vocab_size(), -1.0f);
                for (int row = 0; row < seq_len; ++row)
                {
                    const int token =
                        row < static_cast<int>(verify_argmax_tokens.size())
                            ? verify_argmax_tokens[static_cast<size_t>(row)]
                            : verify_argmax_token;
                    all_position_logits[static_cast<size_t>(row) * vocab_size() +
                                        static_cast<size_t>(token)] = 10.0f;
                }
                return true;
            }
            logits_buffer.assign(vocab_size(), -1.0f);
            logits_buffer[prefill_argmax_token] = 10.0f;
            return true;
        }

        bool supportsPrefillChunkSchedule(int seq_len) const override
        {
            return supports_chunk_schedule && seq_len > 0;
        }

        bool forwardPrefillChunkSchedule(
            const int *tokens,
            int seq_len,
            const PrefillChunkSchedulerPolicy &policy,
            int pad_token_id,
            bool allow_padded_execution) override
        {
            ++chunk_schedule_calls;
            last_chunk_schedule_tokens.assign(tokens, tokens + seq_len);
            last_chunk_schedule_policy = policy;
            last_chunk_schedule_pad_token_id = pad_token_id;
            last_chunk_schedule_allow_padded = allow_padded_execution;
            if (!chunk_schedule_ok)
                return false;

            position += seq_len;
            logits_buffer.assign(vocab_size(), -1.0f);
            logits_buffer[prefill_argmax_token] = 10.0f;
            return true;
        }

        const float *logits() const override { return logits_buffer.data(); }
        int vocab_size() const override { return 16; }
        void clear_cache() override
        {
            ++clear_calls;
            position = 0;
        }
        int get_position() const override { return position; }
        ExecutionPath executionPath() const override { return ExecutionPath::GRAPH; }
        const char *architecture() const override { return "mock"; }
        int sampleGreedyOnDevice() override { return -1; }

        PrefixLookupResult lookupPrefix(const std::vector<int32_t> &tokens) override
        {
            ++lookup_calls;
            lookup_tokens = tokens;
            return lookup_result;
        }

        bool populatePrefix(const PrefixLookupResult &hit, int seq_idx = 0) override
        {
            (void)seq_idx;
            ++populate_calls;
            populated_tokens.push_back(hit.cached_tokens);
            position = hit.cached_tokens;
            return populate_ok;
        }

        bool harvestPrefix(const std::vector<int32_t> &tokens, int prompt_token_count) override
        {
            ++harvest_calls;
            harvested_tokens = tokens;
            harvested_prompt_token_count = prompt_token_count;
            return true;
        }

        bool restorePrefixTerminalState(const PrefixLookupResult &hit) override
        {
            ++restore_terminal_calls;
            restored_tokens = hit.cached_tokens;
            logits_buffer.assign(vocab_size(), -1.0f);
            logits_buffer[prefill_argmax_token] = 10.0f;
            return restore_terminal_ok;
        }

        bool forwardMTP(int32_t draft_condition_token) override
        {
            if (!mtp_enabled)
                return false;
            return forwardMTPCommon(draft_condition_token);
        }

        bool supportsChainedMTPDrafts() const override
        {
            return supports_chained_mtp;
        }

        bool forwardMTPFromLastDraft(int32_t draft_condition_token, int position_id) override
        {
            if (!mtp_enabled || !supports_chained_mtp)
                return false;
            ++chained_mtp_calls;
            chained_mtp_positions.push_back(position_id);
            return forwardMTPCommon(draft_condition_token);
        }

        bool forwardMTPCommon(int32_t draft_condition_token)
        {
            ++forward_mtp_calls;
            last_mtp_condition_token = draft_condition_token;
            mtp_logits.assign(vocab_size(), -1.0f);
            const int token_index = forward_mtp_calls - 1;
            const int token =
                token_index < static_cast<int>(mtp_argmax_tokens.size())
                    ? mtp_argmax_tokens[static_cast<size_t>(token_index)]
                    : mtp_argmax_token;
            mtp_logits[token] = 10.0f;
            ++shifted_mtp_rows;
            return true;
        }

        const float *mtpLogits() const override
        {
            return mtp_logits.empty() ? nullptr : mtp_logits.data();
        }

        bool commitMTPShiftedRowsFromLastForward(
            const int32_t *tokens,
            int token_count,
            int already_appended_tokens) override
        {
            ++commit_mtp_calls;
            last_commit_already_appended = already_appended_tokens;
            last_commit_tokens.assign(tokens, tokens + token_count);
            shifted_mtp_rows += std::max(0, token_count - already_appended_tokens);
            return true;
        }

        bool setComputeAllPositionLogits(bool enabled) override
        {
            if (!mtp_enabled)
                return false;
            all_position_logits_enabled = enabled;
            return true;
        }

        const float *getAllPositionLogits() const override
        {
            return all_position_logits.empty() ? nullptr : all_position_logits.data();
        }

        PrefixStateSnapshot captureLivePrefixState(int seq_idx = 0) const override
        {
            (void)seq_idx;
            PrefixStateSnapshot snapshot;
            snapshot.valid = mtp_enabled;
            snapshot.cached_tokens = position;
            snapshot.mtp_cached_tokens = {shifted_mtp_rows};
            return snapshot;
        }

        PrefixStateSnapshot captureLivePrefixCheckpoint(int seq_idx = 0) const override
        {
            return captureLivePrefixState(seq_idx);
        }

        bool restoreLivePrefixState(const PrefixStateSnapshot &snapshot, int seq_idx = 0) override
        {
            (void)seq_idx;
            if (!snapshot.valid)
                return false;
            ++restore_live_calls;
            position = snapshot.cached_tokens;
            if (!snapshot.mtp_cached_tokens.empty())
            {
                shifted_mtp_rows = snapshot.mtp_cached_tokens.front();
                restored_mtp_rows.push_back(shifted_mtp_rows);
            }
            all_position_logits_enabled = false;
            return true;
        }

        PrefixLookupResult lookup_result;
        bool populate_ok = true;
        bool restore_terminal_ok = true;
        bool mtp_enabled = false;
        bool supports_chained_mtp = true;
        bool all_position_logits_enabled = false;
        bool supports_chunk_schedule = false;
        bool chunk_schedule_ok = true;
        std::vector<float> logits_buffer = std::vector<float>(16, -1.0f);
        std::vector<float> mtp_logits;
        std::vector<float> all_position_logits;
        std::vector<int> mtp_argmax_tokens;
        std::vector<int> verify_argmax_tokens;
        int prefill_argmax_token = 9;
        int mtp_argmax_token = 11;
        int verify_argmax_token = 11;
        int forward_calls = 0;
        int chunk_schedule_calls = 0;
        int forward_mtp_calls = 0;
        int chained_mtp_calls = 0;
        int commit_mtp_calls = 0;
        int clear_calls = 0;
        int lookup_calls = 0;
        int populate_calls = 0;
        int harvest_calls = 0;
        int restore_terminal_calls = 0;
        int restore_live_calls = 0;
        int last_mtp_condition_token = -1;
        int restored_tokens = 0;
        int harvested_prompt_token_count = 0;
        int position = 0;
        int shifted_mtp_rows = 0;
        int last_commit_already_appended = 0;
        std::vector<int> last_forward_tokens;
        std::vector<int> last_commit_tokens;
        std::vector<int> chained_mtp_positions;
        std::vector<int> restored_mtp_rows;
        std::vector<int> last_chunk_schedule_tokens;
        PrefillChunkSchedulerPolicy last_chunk_schedule_policy;
        int last_chunk_schedule_pad_token_id = -1;
        bool last_chunk_schedule_allow_padded = false;
        std::vector<int32_t> lookup_tokens;
        std::vector<int32_t> harvested_tokens;
        std::vector<int> populated_tokens;
    };

    class ScopedPrefillChunkScheduleEnv
    {
    public:
        ScopedPrefillChunkScheduleEnv()
            : old_gpu_graphs_(mutableDebugEnv().execution.gpu_graphs),
              old_buckets_(mutableDebugEnv().execution.prefill_graph_buckets),
              old_min_seq_(mutableDebugEnv().execution.prefill_graph_min_seq),
              old_bucket_sizes_(mutableDebugEnv().execution.prefill_graph_bucket_sizes),
              old_pad_token_(mutableDebugEnv().execution.prefill_graph_pad_token_id)
        {
            mutableDebugEnv().execution.gpu_graphs = true;
            mutableDebugEnv().execution.prefill_graph_buckets = true;
            mutableDebugEnv().execution.prefill_graph_min_seq = 1;
            mutableDebugEnv().execution.prefill_graph_bucket_sizes = {2};
            mutableDebugEnv().execution.prefill_graph_pad_token_id = 99;
        }

        ~ScopedPrefillChunkScheduleEnv()
        {
            mutableDebugEnv().execution.gpu_graphs = old_gpu_graphs_;
            mutableDebugEnv().execution.prefill_graph_buckets = old_buckets_;
            mutableDebugEnv().execution.prefill_graph_min_seq = old_min_seq_;
            mutableDebugEnv().execution.prefill_graph_bucket_sizes = old_bucket_sizes_;
            mutableDebugEnv().execution.prefill_graph_pad_token_id = old_pad_token_;
        }

    private:
        bool old_gpu_graphs_;
        bool old_buckets_;
        int old_min_seq_;
        std::vector<int> old_bucket_sizes_;
        int old_pad_token_;
    };

    RankExecutionPlan makePlan(bool mtp_enabled = false, int mtp_draft_tokens = 1)
    {
        RankExecutionPlan plan;
        plan.rank = 0;
        plan.hostname = "localhost";
        plan.numa_node = 0;
        plan.pp_stage_id = 0;
        plan.first_layer = 0;
        plan.last_layer = 1;
        plan.has_embedding = true;
        plan.has_lm_head = true;
        plan.primary_device = GlobalDeviceAddress::cpu();
        plan.runtime.prefix_cache.enabled = true;
        plan.runtime.prefix_cache.storage_mode = PrefixCacheStorageMode::Ram;
        plan.runtime.prefix_cache.block_size = 2;
        plan.runtime.mtp.enabled = mtp_enabled;
        plan.runtime.mtp.draft_tokens = mtp_draft_tokens;
        plan.runtime.mtp.verify_mode = MTPVerifyMode::Greedy;
        return plan;
    }

    OrchestrationConfig makeConfig(bool mtp_enabled = false, int mtp_draft_tokens = 1)
    {
        OrchestrationConfig config;
        config.device_for_this_rank = GlobalDeviceAddress::cpu();
        config.prefix_cache.enabled = true;
        config.prefix_cache.storage_mode = PrefixCacheStorageMode::Ram;
        config.prefix_cache.block_size = 2;
        config.mtp.enabled = mtp_enabled;
        config.mtp.draft_tokens = mtp_draft_tokens;
        config.mtp.verify_mode = MTPVerifyMode::Greedy;
        return config;
    }

    std::unique_ptr<OrchestrationRunner> makeRunner(std::unique_ptr<PrefixFlowMockRunner> mock,
                                                    bool mtp_enabled = false,
                                                    int mtp_draft_tokens = 1)
    {
        mock->mtp_enabled = mtp_enabled;
        auto runner = std::make_unique<OrchestrationRunner>(
            makeConfig(mtp_enabled, mtp_draft_tokens),
            makePlan(mtp_enabled, mtp_draft_tokens),
            std::move(mock));
        SamplingParams greedy;
        greedy.temperature = 0.0f;
        runner->setSamplingParams(greedy);
        return runner;
    }
} // namespace

TEST(Test__PrefixCachePrefillFlow, SharedPrefixRunsOnlySuffixAndHarvestsPrompt)
{
    auto mock = std::make_unique<PrefixFlowMockRunner>();
    auto *mock_ptr = mock.get();
    mock_ptr->lookup_result.supported = true;
    mock_ptr->lookup_result.cache_enabled = true;
    mock_ptr->lookup_result.block_size = 2;
    mock_ptr->lookup_result.cached_tokens = 2;

    auto runner = makeRunner(std::move(mock));
    const std::vector<int32_t> prompt = {1, 2, 3, 4, 5};
    ASSERT_TRUE(runner->prefill(prompt)) << runner->lastError();

    EXPECT_EQ(mock_ptr->lookup_calls, 1);
    EXPECT_EQ(mock_ptr->clear_calls, 1);
    EXPECT_EQ(mock_ptr->populate_calls, 1);
    EXPECT_THAT(mock_ptr->populated_tokens, ElementsAre(2));
    EXPECT_EQ(mock_ptr->forward_calls, 1);
    EXPECT_THAT(mock_ptr->last_forward_tokens, ElementsAre(3, 4, 5));
    EXPECT_EQ(mock_ptr->harvest_calls, 1);
    EXPECT_EQ(mock_ptr->harvested_prompt_token_count, 5);
    EXPECT_THAT(mock_ptr->harvested_tokens, ElementsAre(1, 2, 3, 4, 5));

    const auto probe = runner->prefixStateProbe();
    EXPECT_TRUE(probe.prefix_request.enabled);
    EXPECT_FALSE(probe.prefix_request.bypassed);
    EXPECT_FALSE(probe.prefix_request.hit);
    EXPECT_TRUE(probe.prefix_request.partial_hit);
    EXPECT_EQ(probe.prefix_request.requested_tokens, 5);
    EXPECT_EQ(probe.prefix_request.matched_tokens, 2);
    EXPECT_EQ(probe.prefix_request.matched_blocks, 1);
    EXPECT_FALSE(probe.prefix_request.terminal_logits_restored);
    EXPECT_EQ(probe.prefix_request.storage_tier, "none");
}

TEST(Test__PrefixCachePrefillFlow, CoordinatedPrefixHitPopulatesOnlyCompleteBlocks)
{
    auto mock = std::make_unique<PrefixFlowMockRunner>();
    auto *mock_ptr = mock.get();
    mock_ptr->lookup_result.supported = true;
    mock_ptr->lookup_result.cache_enabled = true;
    mock_ptr->lookup_result.block_size = 2;
    mock_ptr->lookup_result.cached_tokens = 3;

    auto runner = makeRunner(std::move(mock));
    const std::vector<int32_t> prompt = {1, 2, 3, 4, 5};
    ASSERT_TRUE(runner->prefill(prompt)) << runner->lastError();

    EXPECT_EQ(mock_ptr->populate_calls, 1);
    EXPECT_THAT(mock_ptr->populated_tokens, ElementsAre(2));
    EXPECT_EQ(mock_ptr->forward_calls, 1);
    EXPECT_THAT(mock_ptr->last_forward_tokens, ElementsAre(3, 4, 5));
}

TEST(Test__PrefixCachePrefillFlow, LongPrefixSuffixUsesChunkScheduleWhenRunnerSupportsIt)
{
    ScopedPrefillChunkScheduleEnv env;
    auto mock = std::make_unique<PrefixFlowMockRunner>();
    auto *mock_ptr = mock.get();
    mock_ptr->supports_chunk_schedule = true;
    mock_ptr->lookup_result.supported = true;
    mock_ptr->lookup_result.cache_enabled = true;
    mock_ptr->lookup_result.block_size = 2;
    mock_ptr->lookup_result.cached_tokens = 2;

    auto runner = makeRunner(std::move(mock));
    const std::vector<int32_t> prompt = {1, 2, 3, 4, 5};
    ASSERT_TRUE(runner->prefill(prompt)) << runner->lastError();

    EXPECT_EQ(mock_ptr->forward_calls, 0);
    EXPECT_EQ(mock_ptr->chunk_schedule_calls, 1);
    EXPECT_THAT(mock_ptr->last_chunk_schedule_tokens, ElementsAre(3, 4, 5));
    EXPECT_EQ(mock_ptr->last_chunk_schedule_policy.real_token_start, 2);
    EXPECT_EQ(mock_ptr->last_chunk_schedule_policy.real_token_count, 3);
    EXPECT_EQ(mock_ptr->last_chunk_schedule_policy.fixed_chunk_real_tokens, 2);
    EXPECT_EQ(mock_ptr->last_chunk_schedule_policy.min_rebalance_interval_tokens, 2);
    EXPECT_EQ(mock_ptr->last_chunk_schedule_policy.max_rebalance_interval_tokens, 0);
    EXPECT_THAT(mock_ptr->last_chunk_schedule_policy.bucket_sizes, ElementsAre(2));
    EXPECT_EQ(mock_ptr->last_chunk_schedule_pad_token_id, 99);
    EXPECT_TRUE(mock_ptr->last_chunk_schedule_allow_padded);
    EXPECT_EQ(mock_ptr->harvest_calls, 1);

    const auto probe = runner->prefixStateProbe();
    EXPECT_EQ(probe.prefill_chunk_schedules, 1u);
    EXPECT_EQ(probe.prefill_chunk_successful_schedules, 1u);
    EXPECT_EQ(probe.prefill_chunks, 2u);
    EXPECT_EQ(probe.prefill_chunk_real_tokens, 3u);
    EXPECT_EQ(probe.prefill_chunk_padded_tokens, 1u);
    EXPECT_EQ(probe.prefill_chunk_failures, 0u);
}

TEST(Test__PrefixCachePrefillFlow, LongPrefixSuffixFallsBackWhenChunkScheduleUnsupported)
{
    ScopedPrefillChunkScheduleEnv env;
    auto mock = std::make_unique<PrefixFlowMockRunner>();
    auto *mock_ptr = mock.get();
    mock_ptr->lookup_result.supported = true;
    mock_ptr->lookup_result.cache_enabled = true;
    mock_ptr->lookup_result.block_size = 2;
    mock_ptr->lookup_result.cached_tokens = 2;

    auto runner = makeRunner(std::move(mock));
    const std::vector<int32_t> prompt = {1, 2, 3, 4, 5};
    ASSERT_TRUE(runner->prefill(prompt)) << runner->lastError();

    EXPECT_EQ(mock_ptr->chunk_schedule_calls, 0);
    EXPECT_EQ(mock_ptr->forward_calls, 1);
    EXPECT_THAT(mock_ptr->last_forward_tokens, ElementsAre(3, 4, 5));

    const auto probe = runner->prefixStateProbe();
    EXPECT_EQ(probe.prefill_chunk_schedules, 0u);
    EXPECT_EQ(probe.prefill_chunks, 0u);
    EXPECT_EQ(probe.prefill_chunk_failures, 0u);
}

TEST(Test__PrefixCachePrefillFlow, LongPrefixSuffixReportsChunkScheduleFailure)
{
    ScopedPrefillChunkScheduleEnv env;
    auto mock = std::make_unique<PrefixFlowMockRunner>();
    auto *mock_ptr = mock.get();
    mock_ptr->supports_chunk_schedule = true;
    mock_ptr->chunk_schedule_ok = false;
    mock_ptr->lookup_result.supported = true;
    mock_ptr->lookup_result.cache_enabled = true;
    mock_ptr->lookup_result.block_size = 2;
    mock_ptr->lookup_result.cached_tokens = 2;

    auto runner = makeRunner(std::move(mock));
    const std::vector<int32_t> prompt = {1, 2, 3, 4, 5};
    ASSERT_FALSE(runner->prefill(prompt));
    EXPECT_THAT(runner->lastError(), HasSubstr("chunked prefill failed"));

    EXPECT_EQ(mock_ptr->chunk_schedule_calls, 1);
    EXPECT_EQ(mock_ptr->forward_calls, 0);
    EXPECT_EQ(mock_ptr->harvest_calls, 0);

    const auto probe = runner->prefixStateProbe();
    EXPECT_EQ(probe.prefill_chunk_schedules, 1u);
    EXPECT_EQ(probe.prefill_chunk_successful_schedules, 0u);
    EXPECT_EQ(probe.prefill_chunks, 0u);
    EXPECT_EQ(probe.prefill_chunk_real_tokens, 0u);
    EXPECT_EQ(probe.prefill_chunk_padded_tokens, 0u);
    EXPECT_EQ(probe.prefill_chunk_failures, 1u);
}

TEST(Test__PrefixCachePrefillFlow, MTPPartialHitWithoutTerminalHiddenRecomputesBoundaryBlock)
{
    auto mock = std::make_unique<PrefixFlowMockRunner>();
    auto *mock_ptr = mock.get();
    mock_ptr->lookup_result.supported = true;
    mock_ptr->lookup_result.cache_enabled = true;
    mock_ptr->lookup_result.block_size = 2;
    mock_ptr->lookup_result.cached_tokens = 4;
    mock_ptr->lookup_result.has_terminal_hidden = false;

    auto runner = makeRunner(std::move(mock), /*mtp_enabled=*/true);
    const std::vector<int32_t> prompt = {1, 2, 3, 4, 5};
    ASSERT_TRUE(runner->prefill(prompt)) << runner->lastError();

    EXPECT_EQ(mock_ptr->clear_calls, 1);
    EXPECT_EQ(mock_ptr->populate_calls, 1);
    EXPECT_THAT(mock_ptr->populated_tokens, ElementsAre(2));
    EXPECT_EQ(mock_ptr->forward_calls, 1);
    EXPECT_THAT(mock_ptr->last_forward_tokens, ElementsAre(3, 4, 5));
}

TEST(Test__PrefixCachePrefillFlow, FullHitWithTerminalLogitsSkipsForward)
{
    auto mock = std::make_unique<PrefixFlowMockRunner>();
    auto *mock_ptr = mock.get();
    mock_ptr->lookup_result.supported = true;
    mock_ptr->lookup_result.cache_enabled = true;
    mock_ptr->lookup_result.block_size = 2;
    mock_ptr->lookup_result.cached_tokens = 4;
    mock_ptr->lookup_result.has_terminal_logits = true;

    auto runner = makeRunner(std::move(mock));
    ASSERT_TRUE(runner->prefill({1, 2, 3, 4})) << runner->lastError();

    EXPECT_EQ(mock_ptr->forward_calls, 0);
    EXPECT_EQ(mock_ptr->populate_calls, 1);
    EXPECT_EQ(mock_ptr->restore_terminal_calls, 1);
    EXPECT_EQ(mock_ptr->restored_tokens, 4);

    auto probe = runner->prefixStateProbe();
    EXPECT_TRUE(probe.prefix_request.hit);
    EXPECT_FALSE(probe.prefix_request.partial_hit);
    EXPECT_EQ(probe.prefix_request.matched_tokens, 4);
    EXPECT_TRUE(probe.prefix_request.terminal_logits_restored);
    EXPECT_FALSE(probe.prefix_request.terminal_hidden_restored);

    auto step = runner->decodeStep();
    ASSERT_TRUE(step.success()) << step.error;
    ASSERT_EQ(step.tokens.size(), 1u);
    EXPECT_EQ(step.tokens[0], mock_ptr->prefill_argmax_token);
}

TEST(Test__PrefixCachePrefillFlow, FullHitWithMTPCommitsAcceptedVerifierStateWithoutPromptDuplication)
{
    auto mock = std::make_unique<PrefixFlowMockRunner>();
    auto *mock_ptr = mock.get();
    mock_ptr->lookup_result.supported = true;
    mock_ptr->lookup_result.cache_enabled = true;
    mock_ptr->lookup_result.block_size = 2;
    mock_ptr->lookup_result.cached_tokens = 4;
    mock_ptr->lookup_result.has_terminal_logits = true;
    mock_ptr->lookup_result.has_terminal_hidden = true;

    auto runner = makeRunner(std::move(mock), /*mtp_enabled=*/true);
    ASSERT_TRUE(runner->prefill({1, 2, 3, 4})) << runner->lastError();

    EXPECT_EQ(mock_ptr->forward_calls, 0);
    EXPECT_EQ(mock_ptr->restore_terminal_calls, 1);

    auto step = runner->decodeStep();
    ASSERT_TRUE(step.success()) << step.error;
    EXPECT_THAT(step.tokens, ElementsAre(mock_ptr->prefill_argmax_token,
                                         mock_ptr->mtp_argmax_token));
    EXPECT_EQ(mock_ptr->forward_mtp_calls, 1);
    EXPECT_EQ(mock_ptr->last_mtp_condition_token, mock_ptr->prefill_argmax_token);
    EXPECT_EQ(mock_ptr->restore_live_calls, 0);
    EXPECT_EQ(mock_ptr->commit_mtp_calls, 1);
    EXPECT_EQ(mock_ptr->last_commit_already_appended, 1);
    EXPECT_THAT(mock_ptr->last_commit_tokens,
                ElementsAre(mock_ptr->prefill_argmax_token,
                            mock_ptr->mtp_argmax_token));

    const auto probe = runner->prefixStateProbe();
    EXPECT_TRUE(probe.prefix_request.hit);
    EXPECT_TRUE(probe.prefix_request.terminal_logits_restored);
    EXPECT_TRUE(probe.prefix_request.terminal_hidden_restored);
    EXPECT_TRUE(probe.mtp_request.enabled);
    EXPECT_FALSE(probe.mtp_request.bypassed);
    EXPECT_EQ(probe.mtp_request.draft_steps, 1u);
    EXPECT_EQ(probe.mtp_request.accepted_tokens, 1u);
    EXPECT_EQ(probe.mtp_request.rejected_tokens, 0u);
    EXPECT_DOUBLE_EQ(probe.mtp_request.acceptance_rate, 1.0);
    EXPECT_EQ(probe.mtp_draft_steps, 1u);
    EXPECT_EQ(probe.mtp_accepted_tokens, 1u);
    EXPECT_EQ(probe.mtp_rejected_tokens, 0u);
    EXPECT_EQ(probe.mtp_rollbacks, 0u);
    EXPECT_EQ(probe.mtp_verifier_runs, 1u);
    EXPECT_EQ(probe.mtp_verifier_token_count, 2u);
}

TEST(Test__PrefixCachePrefillFlow, FullMTPHitWithoutTerminalHiddenRecomputesFinalBlock)
{
    auto mock = std::make_unique<PrefixFlowMockRunner>();
    auto *mock_ptr = mock.get();
    mock_ptr->lookup_result.supported = true;
    mock_ptr->lookup_result.cache_enabled = true;
    mock_ptr->lookup_result.block_size = 2;
    mock_ptr->lookup_result.cached_tokens = 4;
    mock_ptr->lookup_result.has_terminal_logits = true;
    mock_ptr->lookup_result.has_terminal_hidden = false;

    auto runner = makeRunner(std::move(mock), /*mtp_enabled=*/true);
    ASSERT_TRUE(runner->prefill({1, 2, 3, 4})) << runner->lastError();

    EXPECT_EQ(mock_ptr->restore_terminal_calls, 0);
    EXPECT_EQ(mock_ptr->clear_calls, 2);
    EXPECT_EQ(mock_ptr->populate_calls, 2);
    EXPECT_THAT(mock_ptr->populated_tokens, ElementsAre(4, 2));
    EXPECT_EQ(mock_ptr->forward_calls, 1);
    EXPECT_THAT(mock_ptr->last_forward_tokens, ElementsAre(3, 4));
}

TEST(Test__PrefixCachePrefillFlow, MTPStatsRecordRejectedDraftToken)
{
    auto mock = std::make_unique<PrefixFlowMockRunner>();
    auto *mock_ptr = mock.get();
    mock_ptr->lookup_result.supported = true;
    mock_ptr->lookup_result.cache_enabled = true;
    mock_ptr->lookup_result.block_size = 2;
    mock_ptr->lookup_result.cached_tokens = 4;
    mock_ptr->lookup_result.has_terminal_logits = true;
    mock_ptr->lookup_result.has_terminal_hidden = true;
    mock_ptr->verify_argmax_token = 12;

    auto runner = makeRunner(std::move(mock), /*mtp_enabled=*/true);
    ASSERT_TRUE(runner->prefill({1, 2, 3, 4})) << runner->lastError();

    auto step = runner->decodeStep();
    ASSERT_TRUE(step.success()) << step.error;
    EXPECT_THAT(step.tokens, ElementsAre(mock_ptr->prefill_argmax_token,
                                         mock_ptr->verify_argmax_token));

    const auto probe = runner->prefixStateProbe();
    EXPECT_EQ(probe.mtp_draft_steps, 1u);
    EXPECT_EQ(probe.mtp_accepted_tokens, 0u);
    EXPECT_EQ(probe.mtp_rejected_tokens, 1u);
    EXPECT_EQ(probe.mtp_rollbacks, 1u);
    EXPECT_EQ(probe.mtp_verifier_runs, 1u);
    EXPECT_EQ(probe.mtp_verifier_token_count, 2u);
}

TEST(Test__PrefixCachePrefillFlow, PartialPrefixHitChainedMTPDraftDepthThreeCommitsAcceptedVerifierState)
{
    auto mock = std::make_unique<PrefixFlowMockRunner>();
    auto *mock_ptr = mock.get();
    mock_ptr->lookup_result.supported = true;
    mock_ptr->lookup_result.cache_enabled = true;
    mock_ptr->lookup_result.block_size = 2;
    mock_ptr->lookup_result.cached_tokens = 2;
    mock_ptr->mtp_argmax_tokens = {11, 12, 13};
    mock_ptr->verify_argmax_tokens = {11, 12, 13, 14};

    auto runner = makeRunner(std::move(mock), /*mtp_enabled=*/true, /*mtp_draft_tokens=*/3);
    ASSERT_TRUE(runner->prefill({1, 2, 3, 4})) << runner->lastError();

    auto step = runner->decodeStep();
    ASSERT_TRUE(step.success()) << step.error;
    EXPECT_THAT(step.tokens, ElementsAre(9, 11, 12, 13));

    EXPECT_EQ(mock_ptr->forward_mtp_calls, 3);
    EXPECT_EQ(mock_ptr->chained_mtp_calls, 2);
    EXPECT_THAT(mock_ptr->chained_mtp_positions, ElementsAre(5, 6));
    EXPECT_EQ(mock_ptr->restore_live_calls, 0);
    EXPECT_EQ(mock_ptr->commit_mtp_calls, 1);
    EXPECT_EQ(mock_ptr->last_commit_already_appended, 1);
    EXPECT_THAT(mock_ptr->last_commit_tokens, ElementsAre(9, 11, 12, 13));

    const auto probe = runner->prefixStateProbe();
    EXPECT_EQ(probe.mtp_draft_steps, 3u);
    EXPECT_EQ(probe.mtp_accepted_tokens, 3u);
    EXPECT_EQ(probe.mtp_rejected_tokens, 0u);
    EXPECT_EQ(probe.mtp_rollbacks, 0u);
    EXPECT_EQ(probe.mtp_verifier_runs, 1u);
    EXPECT_EQ(probe.mtp_verifier_token_count, 4u);
}

TEST(Test__PrefixCachePrefillFlow, FullPrefixTerminalRestoreHardFailsForChainedMTPDrafts)
{
    auto mock = std::make_unique<PrefixFlowMockRunner>();
    auto *mock_ptr = mock.get();
    mock_ptr->lookup_result.supported = true;
    mock_ptr->lookup_result.cache_enabled = true;
    mock_ptr->lookup_result.block_size = 2;
    mock_ptr->lookup_result.cached_tokens = 4;
    mock_ptr->lookup_result.has_terminal_logits = true;
    mock_ptr->lookup_result.has_terminal_hidden = true;

    auto runner = makeRunner(std::move(mock), /*mtp_enabled=*/true, /*mtp_draft_tokens=*/2);
    ASSERT_FALSE(runner->prefill({1, 2, 3, 4}));
    EXPECT_THAT(runner->lastError(),
                HasSubstr("Prefix cache terminal restore with chained MTP drafts is not supported"));
    EXPECT_EQ(mock_ptr->restore_terminal_calls, 0);
    EXPECT_EQ(mock_ptr->forward_mtp_calls, 0);
    EXPECT_EQ(mock_ptr->harvest_calls, 0);
}

TEST(Test__PrefixCachePrefillFlow, PartialPrefixHitChainedMTPDraftDepthThreeRestoresMatchingSidecarCheckpointOnReject)
{
    auto mock = std::make_unique<PrefixFlowMockRunner>();
    auto *mock_ptr = mock.get();
    mock_ptr->lookup_result.supported = true;
    mock_ptr->lookup_result.cache_enabled = true;
    mock_ptr->lookup_result.block_size = 2;
    mock_ptr->lookup_result.cached_tokens = 2;
    mock_ptr->mtp_argmax_tokens = {11, 12, 13};
    mock_ptr->verify_argmax_tokens = {11, 15, 13, 14};

    auto runner = makeRunner(std::move(mock), /*mtp_enabled=*/true, /*mtp_draft_tokens=*/3);
    ASSERT_TRUE(runner->prefill({1, 2, 3, 4})) << runner->lastError();

    auto step = runner->decodeStep();
    ASSERT_TRUE(step.success()) << step.error;
    EXPECT_THAT(step.tokens, ElementsAre(9, 11, 15));

    EXPECT_EQ(mock_ptr->forward_mtp_calls, 3);
    EXPECT_EQ(mock_ptr->chained_mtp_calls, 2);
    EXPECT_THAT(mock_ptr->chained_mtp_positions, ElementsAre(5, 6));
    ASSERT_THAT(mock_ptr->restored_mtp_rows, Not(IsEmpty()));
    EXPECT_EQ(mock_ptr->restored_mtp_rows.back(), 1);
    EXPECT_EQ(mock_ptr->restore_live_calls, 1);
    EXPECT_THAT(mock_ptr->last_forward_tokens, ElementsAre(9, 11));
    EXPECT_EQ(mock_ptr->commit_mtp_calls, 1);
    EXPECT_EQ(mock_ptr->last_commit_already_appended, 1);
    EXPECT_THAT(mock_ptr->last_commit_tokens, ElementsAre(9, 11, 15));

    const auto probe = runner->prefixStateProbe();
    EXPECT_EQ(probe.mtp_draft_steps, 3u);
    EXPECT_EQ(probe.mtp_accepted_tokens, 1u);
    EXPECT_EQ(probe.mtp_rejected_tokens, 1u);
    EXPECT_EQ(probe.mtp_rollbacks, 1u);
    EXPECT_EQ(probe.mtp_verifier_runs, 1u);
    EXPECT_EQ(probe.mtp_verifier_token_count, 4u);
}

TEST(Test__PrefixCachePrefillFlow, ChainedMTPDraftDepthHardFailsWhenRunnerDoesNotSupportIt)
{
    auto mock = std::make_unique<PrefixFlowMockRunner>();
    mock->supports_chained_mtp = false;

    auto runner = makeRunner(std::move(mock), /*mtp_enabled=*/true, /*mtp_draft_tokens=*/2);
    ASSERT_FALSE(runner->prefill({1, 2, 3, 4}));
    EXPECT_THAT(runner->lastError(), HasSubstr("requires runner support for chained MTP sidecars"));
}

TEST(Test__PrefixCachePrefillFlow, FullHitWithoutTerminalLogitsRecomputesFinalBlock)
{
    auto mock = std::make_unique<PrefixFlowMockRunner>();
    auto *mock_ptr = mock.get();
    mock_ptr->lookup_result.supported = true;
    mock_ptr->lookup_result.cache_enabled = true;
    mock_ptr->lookup_result.block_size = 2;
    mock_ptr->lookup_result.cached_tokens = 4;
    mock_ptr->lookup_result.has_terminal_logits = false;

    auto runner = makeRunner(std::move(mock));
    ASSERT_TRUE(runner->prefill({1, 2, 3, 4})) << runner->lastError();

    EXPECT_EQ(mock_ptr->clear_calls, 2);
    EXPECT_EQ(mock_ptr->populate_calls, 2);
    EXPECT_THAT(mock_ptr->populated_tokens, ElementsAre(4, 2));
    EXPECT_EQ(mock_ptr->forward_calls, 1);
    EXPECT_THAT(mock_ptr->last_forward_tokens, ElementsAre(3, 4));
}
