/**
 * @file Test__PrefixCachePrefillFlow.cpp
 * @brief Regression coverage for request-boundary prefix-cache prefill semantics.
 *
 * These tests exercise the OrchestrationRunner layer with a lightweight
 * IInferenceRunner mock.  They pin the visible lifetime boundaries for KV,
 * GDN, MTP, and MoE model-runtime state when prefix-cache hits, misses, and
 * terminal restores cross a request boundary.
 */

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <algorithm>

#include "backends/GlobalDeviceAddress.h"
#include "config/OrchestrationConfig.h"
#include "execution/local_execution/engine/PrefillBucketUtils.h"
#include "execution/local_execution/orchestrators/IInferenceRunner.h"
#include "execution/mtp/MTPSpecStateContract.h"
#include "execution/mpi_orchestration/RankExecutionPlan.h"
#include "execution/prefix_cache/PrefixCacheCoordinator.h"
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
            forward_token_batches.emplace_back(tokens, tokens + seq_len);
            position += seq_len;
            if (all_position_logits_enabled)
            {
                /*
                 * MTP grouped verification enables row-indexed all-position
                 * logits before the verifier forward.  The graph still consumes
                 * every verifier input token, but the logits surface contains
                 * only the compact rows selected by MTPSpecDecode metadata.
                 * Modeling that distinction here keeps prefix-cache tests on
                 * the same grouped publication contract as production.
                 */
                const int row_count =
                    row_indexed_all_position_logits_enabled
                        ? row_indexed_all_position_logits_row_count
                        : seq_len;
                all_position_logits.assign(
                    static_cast<size_t>(row_count) *
                        static_cast<size_t>(vocab_size()),
                    -1.0f);
                for (int row = 0; row < row_count; ++row)
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
            int token = prefill_argmax_token;
            if (seq_len == 1)
            {
                if (decode_argmax_index < decode_argmax_tokens.size())
                {
                    token = decode_argmax_tokens[decode_argmax_index++];
                }
                else if (mtp_enabled)
                {
                    const size_t token_index = decode_argmax_index++;
                    token = token_index < verify_argmax_tokens.size()
                                ? verify_argmax_tokens[token_index]
                                : verify_argmax_token;
                }
            }
            logits_buffer[token] = 10.0f;
            syncShiftedRowsToPosition();
            return true;
        }

        bool forwardGroupedMTPVerifierWithHostTokenIds(
            const std::vector<std::vector<int>> &token_batches) override
        {
            if (token_batches.size() != 1 || token_batches.front().empty())
                return false;
            const auto &tokens = token_batches.front();
            return forward(tokens.data(), static_cast<int>(tokens.size()));
        }

        bool forwardRestoredPrefixMTPDecodeBridge(
            const RestoredPrefixMTPDecodeBridgeRequest &request) override
        {
            if (!mtp_enabled || !request.valid() ||
                position != request.restored_prefix_tokens)
            {
                return false;
            }
            ++restored_prefix_bridge_calls;
            restored_prefix_bridge_requests.push_back(request);
            const int token = request.token_id;
            return forward(&token, 1);
        }

        bool supportsPrefillChunkSchedule(int seq_len) const override
        {
            return supports_chunk_schedule && seq_len > 0;
        }

        ServingGraphPreparationKind
        servingGraphPreparationKind() const noexcept override
        {
            return serving_graph_preparation_kind;
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
            syncShiftedRowsToPosition();
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
            shifted_mtp_rows = 0;
        }
        int get_position() const override { return position; }
        ExecutionPath executionPath() const override { return ExecutionPath::GRAPH; }
        const char *architecture() const override { return "mock"; }
        int sampleGreedyOnDevice() override { return -1; }
        uint64_t moeRuntimeMovementEpoch() const override
        {
            return runtime_movement_epoch;
        }

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
            syncShiftedRowsToPosition();
            return populate_ok;
        }

        bool harvestPrefix(
            const PrefixLookupResult &admission,
            const std::vector<int32_t> &tokens,
            int prompt_token_count) override
        {
            ++harvest_calls;
            harvested_fingerprint = admission.fingerprint_key;
            harvested_tokens = tokens;
            harvested_prompt_token_count = prompt_token_count;
            if (movement_epoch_after_harvest)
                runtime_movement_epoch = *movement_epoch_after_harvest;
            return harvest_ok;
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

        bool supportsMTPSidecarPreservesMainState() const override
        {
            return true;
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

        bool commitMTPShiftedRowFromCurrentTerminalHidden(
            int32_t token,
            int already_appended_tokens,
            bool allow_speculative_discard = false,
            int position_offset_override = -1) override
        {
            ++commit_mtp_calls;
            last_commit_already_appended = already_appended_tokens;
            last_commit_allow_speculative_discard = allow_speculative_discard;
            last_commit_position_offset_override = position_offset_override;
            last_commit_tokens.assign(1, token);
            ++shifted_mtp_rows;
            return already_appended_tokens >= 0;
        }

        /**
         * @brief Append the first shifted MTP row from a verifier-base checkpoint.
         *
         * Grouped publication verifies all target rows before publishing live
         * state.  When the sidecar's first shifted row is not reusable, the
         * serial-equivalent source for the accepted first token is the terminal
         * hidden state captured before draft work.  The mock records the same
         * logical repair that production runners perform without pretending to
         * own tensor payloads.
         */
        bool commitMTPShiftedRowFromCheckpointTerminalHidden(
            const PrefixStateSnapshot &checkpoint,
            int32_t token,
            int already_appended_tokens,
            bool allow_speculative_discard = false,
            int position_offset_override = -1) override
        {
            ++checkpoint_terminal_hidden_commit_calls;
            if (!checkpoint.valid || already_appended_tokens < 0)
                return false;
            if (position_offset_override >= 0 &&
                position_offset_override != checkpoint.cached_tokens)
            {
                return false;
            }
            return commitMTPShiftedRowFromCurrentTerminalHidden(
                token,
                already_appended_tokens,
                allow_speculative_discard,
                checkpoint.cached_tokens);
        }

        bool setComputeAllPositionLogits(bool enabled) override
        {
            if (!mtp_enabled)
                return false;
            all_position_logits_enabled = enabled;
            if (!enabled)
                row_indexed_all_position_logits_enabled = false;
            return true;
        }

        bool setComputeRowIndexedAllPositionLogits(bool enabled, int row_count) override
        {
            if (!mtp_enabled)
                return false;
            if (enabled && row_count <= 0)
                return false;
            row_indexed_all_position_logits_enabled = enabled;
            row_indexed_all_position_logits_row_count = enabled ? row_count : 0;
            all_position_logits_enabled = enabled || all_position_logits_enabled;
            return true;
        }

        bool setMTPSpecVerifierInputPlan(
            const MTPSpecDecodeVerifierInputPlan &plan) override
        {
            if (!mtp_enabled || !plan.ok)
                return false;
            ++set_mtp_verifier_plan_calls;
            mtp_verifier_plan_installed = true;
            last_mtp_verifier_plan = plan;
            return true;
        }

        void clearMTPSpecVerifierInputPlan() override
        {
            ++clear_mtp_verifier_plan_calls;
            mtp_verifier_plan_installed = false;
            last_mtp_verifier_plan = MTPSpecDecodeVerifierInputPlan{};
        }

        const float *getAllPositionLogits() const override
        {
            return all_position_logits.empty() ? nullptr : all_position_logits.data();
        }

        /**
         * @brief Publish a grouped verifier transaction into the mock live state.
         *
         * The mock does not own KV/GDN tensors, so this method validates the
         * externally visible step-plan shape and updates the same logical
         * counters that production publication updates after copying accepted
         * verifier rows.  Tests can then prove that prefix-cache MTP exercised
         * grouped publication without smuggling the old serial row-replay path
         * back into the unit double.
         */
        bool publishGroupedDecodeEquivalentMTPSpecStateBatch(
            const MTPSpecStepPlanBatch &plans,
            std::string *error = nullptr) override
        {
            if (!mtp_enabled)
            {
                if (error)
                    *error = "MTP is disabled";
                return false;
            }
            if (!plans.ok)
            {
                if (error)
                    *error = "invalid grouped MTP plan: " + plans.error;
                return false;
            }
            if (plans.request_count != 1 || plans.steps.size() != 1)
            {
                if (error)
                    *error = "prefix-flow mock supports exactly one grouped MTP request";
                return false;
            }

            const MTPSpecStepPlan &step = plans.steps.front();
            if (step.request_index != 0 ||
                step.target_cached_tokens !=
                    step.base_cached_tokens + step.accepted_count)
            {
                if (error)
                    *error = "grouped MTP plan has inconsistent single-request metadata";
                return false;
            }

            ++grouped_mtp_publication_calls;
            last_grouped_mtp_publication_plan = plans;
            last_grouped_mtp_step = step;
            position = step.target_cached_tokens;
            syncShiftedRowsToPosition();
            return true;
        }

        PrefixStateSnapshot captureLivePrefixState(int seq_idx = 0) const override
        {
            (void)seq_idx;
            PrefixStateSnapshot snapshot;
            snapshot.valid = mtp_enabled;
            snapshot.provenance = PrefixStateProvenance::DecodeEquivalent;
            snapshot.cached_tokens = position;
            snapshot.mtp_cached_tokens = {shifted_mtp_rows};
            return snapshot;
        }

        PrefixStateSnapshot captureLivePrefixCheckpoint(
            const PrefixCheckpointCaptureRequest &request) const override
        {
            PrefixStateSnapshot snapshot =
                captureLivePrefixState(request.sequence_index);
            if (!request.valid())
                return {};
            snapshot.cached_tokens = request.logical_cached_tokens;
            return snapshot;
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
            row_indexed_all_position_logits_enabled = false;
            row_indexed_all_position_logits_row_count = 0;
            return true;
        }

        PrefixLookupResult lookup_result;
        bool populate_ok = true;
        bool harvest_ok = true;
        bool restore_terminal_ok = true;
        bool mtp_enabled = false;
        bool supports_chained_mtp = true;
        bool all_position_logits_enabled = false;
        bool row_indexed_all_position_logits_enabled = false;
        int row_indexed_all_position_logits_row_count = 0;
        bool mtp_verifier_plan_installed = false;
        bool supports_chunk_schedule = false;
        bool chunk_schedule_ok = true;
        ServingGraphPreparationKind serving_graph_preparation_kind =
            ServingGraphPreparationKind::Unresolved;
        std::vector<float> logits_buffer = std::vector<float>(16, -1.0f);
        std::vector<float> mtp_logits;
        std::vector<float> all_position_logits;
        std::vector<int> mtp_argmax_tokens;
        std::vector<int> verify_argmax_tokens;
        std::vector<int> decode_argmax_tokens;
        int prefill_argmax_token = 9;
        int mtp_argmax_token = 11;
        int verify_argmax_token = 11;
        size_t decode_argmax_index = 0;
        int forward_calls = 0;
        int restored_prefix_bridge_calls = 0;
        int chunk_schedule_calls = 0;
        int forward_mtp_calls = 0;
        int chained_mtp_calls = 0;
        int commit_mtp_calls = 0;
        int checkpoint_terminal_hidden_commit_calls = 0;
        int set_mtp_verifier_plan_calls = 0;
        int clear_mtp_verifier_plan_calls = 0;
        int grouped_mtp_publication_calls = 0;
        int clear_calls = 0;
        int lookup_calls = 0;
        int populate_calls = 0;
        int harvest_calls = 0;
        int restore_terminal_calls = 0;
        int restore_live_calls = 0;
        int last_mtp_condition_token = -1;
        int restored_tokens = 0;
        int harvested_prompt_token_count = 0;
        uint64_t harvested_fingerprint = 0;
        uint64_t runtime_movement_epoch = 0;
        std::optional<uint64_t> movement_epoch_after_harvest;
        int position = 0;
        int shifted_mtp_rows = 0;
        int last_commit_already_appended = 0;
        bool last_commit_allow_speculative_discard = false;
        int last_commit_position_offset_override = -1;
        std::vector<int> last_forward_tokens;
        std::vector<std::vector<int>> forward_token_batches;
        std::vector<int> last_commit_tokens;
        std::vector<int> chained_mtp_positions;
        std::vector<int> restored_mtp_rows;
        std::vector<int> last_chunk_schedule_tokens;
        PrefillChunkSchedulerPolicy last_chunk_schedule_policy;
        MTPSpecDecodeVerifierInputPlan last_mtp_verifier_plan;
        MTPSpecStepPlanBatch last_grouped_mtp_publication_plan;
        MTPSpecStepPlan last_grouped_mtp_step;
        int last_chunk_schedule_pad_token_id = -1;
        bool last_chunk_schedule_allow_padded = false;
        std::vector<int32_t> lookup_tokens;
        std::vector<int32_t> harvested_tokens;
        std::vector<int> populated_tokens;
        std::vector<RestoredPrefixMTPDecodeBridgeRequest>
            restored_prefix_bridge_requests;

    private:
        void syncShiftedRowsToPosition()
        {
            if (mtp_enabled)
            {
                shifted_mtp_rows = std::max(0, position - 1);
            }
        }
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

    RankExecutionPlan makePlan(
        bool mtp_enabled = false,
        int mtp_draft_tokens = 1,
        RoutedExpertAssignmentPolicy prefill_assignment_policy =
            RoutedExpertAssignmentPolicy::StaticOwner,
        int prefix_block_size = 2,
        int prefill_window_tokens = 0,
        bool prefix_cache_enabled = true)
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
        plan.runtime.prefix_cache.enabled = prefix_cache_enabled;
        plan.runtime.prefix_cache.storage_mode =
            prefix_cache_enabled ? PrefixCacheStorageMode::Ram : PrefixCacheStorageMode::Disabled;
        plan.runtime.prefix_cache.block_size = prefix_block_size;
        plan.runtime.moe_rebalance.mode = MoERebalanceRuntimeMode::Off;
        plan.runtime.moe_routed_prefill.assignment_window_tokens =
            prefill_window_tokens;
        plan.runtime.mtp.enabled = mtp_enabled;
        plan.runtime.mtp.draft_tokens = mtp_draft_tokens;
        plan.runtime.mtp.verify_mode = MTPVerifyMode::Greedy;
        return plan;
    }

    OrchestrationConfig makeConfig(
        bool mtp_enabled = false,
        int mtp_draft_tokens = 1,
        RoutedExpertAssignmentPolicy prefill_assignment_policy =
            RoutedExpertAssignmentPolicy::StaticOwner,
        int prefix_block_size = 2,
        int prefill_window_tokens = 0,
        bool prefix_cache_enabled = true)
    {
        OrchestrationConfig config;
        config.device_for_this_rank = GlobalDeviceAddress::cpu();
        config.prefix_cache.enabled = prefix_cache_enabled;
        config.prefix_cache.storage_mode =
            prefix_cache_enabled ? PrefixCacheStorageMode::Ram : PrefixCacheStorageMode::Disabled;
        config.prefix_cache.block_size = prefix_block_size;
        config.moe_rebalance.mode = MoERebalanceRuntimeMode::Off;
        config.moe_routed_prefill.assignment_window_tokens =
            prefill_window_tokens;
        if (prefill_assignment_policy ==
            RoutedExpertAssignmentPolicy::LeastLoadedResident)
        {
            auto placement = std::make_shared<MoERoutedExpertPlacementPlan>();
            placement->enabled = true;
            RoutedExpertDomain domain;
            domain.name = "least_loaded_prefill_test";
            domain.routed_prefill_assignment_policy =
                RoutedExpertAssignmentPolicy::LeastLoadedResident;
            placement->domains.push_back(std::move(domain));
            config.moe_routed_expert_plan = std::move(placement);
        }
        config.mtp.enabled = mtp_enabled;
        config.mtp.draft_tokens = mtp_draft_tokens;
        config.mtp.verify_mode = MTPVerifyMode::Greedy;
        return config;
    }

    std::unique_ptr<OrchestrationRunner> makeRunner(std::unique_ptr<PrefixFlowMockRunner> mock,
                                                    bool mtp_enabled = false,
                                                    int mtp_draft_tokens = 1,
                                                    RoutedExpertAssignmentPolicy prefill_assignment_policy =
                                                        RoutedExpertAssignmentPolicy::StaticOwner,
                                                    int prefix_block_size = 2,
                                                    int prefill_window_tokens = 0,
                                                    bool prefix_cache_enabled = true)
    {
        mock->mtp_enabled = mtp_enabled;
        auto runner = std::make_unique<OrchestrationRunner>(
            makeConfig(mtp_enabled,
                       mtp_draft_tokens,
                       prefill_assignment_policy,
                       prefix_block_size,
                       prefill_window_tokens,
                       prefix_cache_enabled),
            makePlan(mtp_enabled,
                     mtp_draft_tokens,
                     prefill_assignment_policy,
                     prefix_block_size,
                     prefill_window_tokens,
                     prefix_cache_enabled),
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
    EXPECT_EQ(mock_ptr->clear_calls, 0);
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

TEST(Test__PrefixCachePrefillFlow, FreshPrefixMissDoesNotManufactureRequestReset)
{
    auto mock = std::make_unique<PrefixFlowMockRunner>();
    auto *mock_ptr = mock.get();
    mock_ptr->lookup_result.supported = true;
    mock_ptr->lookup_result.cache_enabled = true;
    mock_ptr->lookup_result.block_size = 2;
    mock_ptr->lookup_result.cached_tokens = 0;

    auto runner = makeRunner(std::move(mock));
    const std::vector<int32_t> prompt = {1, 2, 3, 4};
    ASSERT_TRUE(runner->prefill(prompt)) << runner->lastError();

    EXPECT_EQ(mock_ptr->clear_calls, 0);
    EXPECT_EQ(mock_ptr->populate_calls, 0);
    EXPECT_EQ(mock_ptr->forward_calls, 1);
    EXPECT_THAT(mock_ptr->last_forward_tokens, ElementsAre(1, 2, 3, 4));
    EXPECT_EQ(mock_ptr->harvest_calls, 1);
}

TEST(Test__PrefixCachePrefillFlow,
     RequestSummaryOwnsExactAsynchronousMovementInterval)
{
    auto mock = std::make_unique<PrefixFlowMockRunner>();
    auto *mock_ptr = mock.get();
    mock_ptr->lookup_result.supported = true;
    mock_ptr->lookup_result.cache_enabled = true;
    mock_ptr->lookup_result.block_size = 2;
    mock_ptr->lookup_result.cached_tokens = 0;
    mock_ptr->lookup_result.placement_epochs = PrefixPlacementEpochSpan::at(7);
    mock_ptr->runtime_movement_epoch = 7;
    mock_ptr->movement_epoch_after_harvest = 8;

    auto runner = makeRunner(std::move(mock));
    ASSERT_TRUE(runner->prefill({1, 2, 3, 4})) << runner->lastError();

    const auto probe = runner->prefixStateProbe();
    EXPECT_EQ(probe.prefix_request.admission_placement_epochs.earliest(), 7u);
    EXPECT_EQ(probe.prefix_request.completion_movement_epoch, 8u);
    EXPECT_TRUE(probe.prefix_request.crossedMovementEpoch());
}

/**
 * @brief Publication after lookup must not rewrite that lookup's admission.
 *
 * The cache selected epoch seven before the maintenance thread published
 * eight. Model output may still complete correctly, but harvest can discard
 * the old archive. Resampling eight during coordination would conceal that
 * interval and falsely require the following request to restore the archive.
 */
TEST(Test__PrefixCachePrefillFlow,
     MovementAfterLookupCannotRelabelItsAdmission)
{
    auto mock = std::make_unique<PrefixFlowMockRunner>();
    auto *mock_ptr = mock.get();
    mock_ptr->lookup_result.supported = true;
    mock_ptr->lookup_result.cache_enabled = true;
    mock_ptr->lookup_result.block_size = 2;
    mock_ptr->lookup_result.placement_epochs = PrefixPlacementEpochSpan::at(7);
    mock_ptr->runtime_movement_epoch = 8;
    mock_ptr->movement_epoch_after_harvest = 8;

    auto runner = makeRunner(std::move(mock));
    ASSERT_TRUE(runner->prefill({1, 2, 3, 4})) << runner->lastError();

    const auto summary = runner->prefixStateProbe().prefix_request;
    EXPECT_EQ(summary.admission_placement_epochs.earliest(), 7u);
    EXPECT_EQ(summary.completion_movement_epoch, 8u);
    EXPECT_TRUE(summary.crossedMovementEpoch());
    EXPECT_TRUE(summary.movementPrecededAdmissionOf(
        PrefixCacheRequestSummary{
            .admission_placement_epochs = PrefixPlacementEpochSpan::at(8),
            .completion_movement_epoch = 8}));
}

TEST(Test__PrefixCachePrefillFlow,
     MovementAfterLaterAdmissionCannotExplainThatRequestsLookup)
{
    const PrefixCacheRequestSummary archived{
        .admission_placement_epochs = PrefixPlacementEpochSpan::at(7),
        .completion_movement_epoch = 7,
    };
    const PrefixCacheRequestSummary restored_then_moved{
        .hit = true,
        .matched_tokens = 17,
        .admission_placement_epochs = PrefixPlacementEpochSpan::at(7),
        .completion_movement_epoch = 8,
    };

    EXPECT_FALSE(
        archived.movementPrecededAdmissionOf(restored_then_moved));
    EXPECT_TRUE(restored_then_moved.crossedMovementEpoch());
}

/**
 * @brief A nested TP lookup must retain movement across its child admissions.
 *
 * One child obtains epoch seven before publication and its peer obtains eight
 * afterwards. Harvest at eight correctly discards the first child's stale
 * archive. Reducing the admission to MAX alone conceals that interval and
 * incorrectly claims that an immediate cold replay has no movement cause.
 */
TEST(Test__PrefixCachePrefillFlow,
     MovementBetweenParticipantLookupsCannotRelabelOlderAdmission)
{
    PrefixLookupResult before_publication;
    before_publication.supported = true;
    before_publication.cache_enabled = true;
    before_publication.block_size = 2;
    before_publication.placement_epochs = PrefixPlacementEpochSpan::at(7);
    PrefixLookupResult after_publication = before_publication;
    after_publication.placement_epochs = PrefixPlacementEpochSpan::at(8);
    auto mock = std::make_unique<PrefixFlowMockRunner>();
    mock->lookup_result = makePrefixLookupResult(
        coordinatePrefixLookups({
            makePrefixParticipantLookup(
                0, DeviceId::cpu(), before_publication, {},
                PrefixFingerprintCoordinationPolicy::ValidateParticipantLocally),
            makePrefixParticipantLookup(
                1, DeviceId::cpu(), after_publication, {},
                PrefixFingerprintCoordinationPolicy::ValidateParticipantLocally),
        }),
        2);
    mock->runtime_movement_epoch = 8;
    mock->movement_epoch_after_harvest = 8;

    auto runner = makeRunner(std::move(mock));
    ASSERT_TRUE(runner->prefill({1, 2, 3, 4})) << runner->lastError();
    const auto summary = runner->prefixStateProbe().prefix_request;
    EXPECT_EQ(summary.admission_placement_epochs, PrefixPlacementEpochSpan::covering(7, 8));
    EXPECT_TRUE(summary.crossedMovementEpoch());
    EXPECT_TRUE(summary.movementPrecededAdmissionOf(
        PrefixCacheRequestSummary{
            .admission_placement_epochs = PrefixPlacementEpochSpan::at(8),
            .completion_movement_epoch = 8}));
}

TEST(Test__PrefixCachePrefillFlow,
     MovementBeforeLaterAdmissionCanInvalidateAnArchive)
{
    const PrefixCacheRequestSummary movement_crossed_harvest{
        .admission_placement_epochs = PrefixPlacementEpochSpan::at(7),
        .completion_movement_epoch = 8,
    };
    const PrefixCacheRequestSummary next_after_crossed_harvest{
        .admission_placement_epochs = PrefixPlacementEpochSpan::at(8),
        .completion_movement_epoch = 8,
    };
    EXPECT_TRUE(
        movement_crossed_harvest.movementPrecededAdmissionOf(
            next_after_crossed_harvest));

    const PrefixCacheRequestSummary archived_before_gap{
        .admission_placement_epochs = PrefixPlacementEpochSpan::at(7),
        .completion_movement_epoch = 7,
    };
    const PrefixCacheRequestSummary next_after_gap{
        .admission_placement_epochs = PrefixPlacementEpochSpan::at(8),
        .completion_movement_epoch = 8,
    };
    EXPECT_TRUE(
        archived_before_gap.movementPrecededAdmissionOf(next_after_gap));
}

/** @brief Publication between the later request's child lookups precedes its aggregate miss. */
TEST(Test__PrefixCachePrefillFlow, MovementWithinLaterLookupCanInvalidateAnArchive)
{
    const PrefixCacheRequestSummary archived{
        .admission_placement_epochs = PrefixPlacementEpochSpan::at(7),
        .completion_movement_epoch = 7,
    };
    const PrefixCacheRequestSummary next{
        .admission_placement_epochs = PrefixPlacementEpochSpan::covering(7, 8),
        .completion_movement_epoch = 8,
    };
    EXPECT_TRUE(archived.movementPrecededAdmissionOf(next));
}

TEST(Test__PrefixCachePrefillFlow, PrefixHarvestFailureIsFatalToRequest)
{
    auto mock = std::make_unique<PrefixFlowMockRunner>();
    auto *mock_ptr = mock.get();
    mock_ptr->lookup_result.supported = true;
    mock_ptr->lookup_result.cache_enabled = true;
    mock_ptr->lookup_result.block_size = 2;
    mock_ptr->lookup_result.cached_tokens = 0;
    mock_ptr->harvest_ok = false;

    auto runner = makeRunner(std::move(mock));
    EXPECT_FALSE(runner->prefill({1, 2, 3, 4}));
    EXPECT_THAT(
        runner->lastError(),
        HasSubstr("Prefix cache harvest failed after successful request execution"));
}

TEST(Test__PrefixCachePrefillFlow, LivePrefixMissResetsBeforeFullPrefill)
{
    auto mock = std::make_unique<PrefixFlowMockRunner>();
    auto *mock_ptr = mock.get();
    mock_ptr->lookup_result.supported = true;
    mock_ptr->lookup_result.cache_enabled = true;
    mock_ptr->lookup_result.block_size = 2;
    mock_ptr->lookup_result.cached_tokens = 0;
    mock_ptr->position = 3;

    auto runner = makeRunner(std::move(mock));
    const std::vector<int32_t> prompt = {1, 2, 3, 4};
    ASSERT_TRUE(runner->prefill(prompt)) << runner->lastError();

    EXPECT_EQ(mock_ptr->clear_calls, 1);
    EXPECT_EQ(mock_ptr->populate_calls, 0);
    EXPECT_EQ(mock_ptr->forward_calls, 1);
    EXPECT_THAT(mock_ptr->last_forward_tokens, ElementsAre(1, 2, 3, 4));
    EXPECT_EQ(mock_ptr->harvest_calls, 1);
}

TEST(Test__PrefixCachePrefillFlow, LLEPPrefixMissWithoutConfiguredWindowUsesSinglePrefill)
{
    auto mock = std::make_unique<PrefixFlowMockRunner>();
    auto *mock_ptr = mock.get();
    mock_ptr->lookup_result.supported = true;
    mock_ptr->lookup_result.cache_enabled = true;
    mock_ptr->lookup_result.block_size = 2;
    mock_ptr->lookup_result.cached_tokens = 0;

    auto runner = makeRunner(std::move(mock),
                             /*mtp_enabled=*/false,
                             /*mtp_draft_tokens=*/1,
                             RoutedExpertAssignmentPolicy::LeastLoadedResident);
    const std::vector<int32_t> prompt = {1, 2, 3, 4, 5};
    ASSERT_TRUE(runner->prefill(prompt)) << runner->lastError();

    EXPECT_EQ(mock_ptr->clear_calls, 0);
    EXPECT_EQ(mock_ptr->populate_calls, 0);
    EXPECT_EQ(mock_ptr->forward_calls, 1);
    EXPECT_THAT(mock_ptr->forward_token_batches,
                ElementsAre(ElementsAre(1, 2, 3, 4, 5)));
    EXPECT_EQ(mock_ptr->harvest_calls, 1);
}

TEST(Test__PrefixCachePrefillFlow, LLEPPartialHitUsesStableCacheBlockPrefillBoundaries)
{
    auto mock = std::make_unique<PrefixFlowMockRunner>();
    auto *mock_ptr = mock.get();
    mock_ptr->lookup_result.supported = true;
    mock_ptr->lookup_result.cache_enabled = true;
    mock_ptr->lookup_result.block_size = 2;
    mock_ptr->lookup_result.cached_tokens = 2;

    auto runner = makeRunner(std::move(mock),
                             /*mtp_enabled=*/false,
                             /*mtp_draft_tokens=*/1,
                             RoutedExpertAssignmentPolicy::LeastLoadedResident);
    const std::vector<int32_t> prompt = {1, 2, 3, 4, 5};
    ASSERT_TRUE(runner->prefill(prompt)) << runner->lastError();

    EXPECT_EQ(mock_ptr->clear_calls, 0);
    EXPECT_EQ(mock_ptr->populate_calls, 1);
    EXPECT_THAT(mock_ptr->populated_tokens, ElementsAre(2));
    EXPECT_EQ(mock_ptr->forward_calls, 2);
    EXPECT_THAT(mock_ptr->forward_token_batches,
                            ElementsAre(ElementsAre(3, 4),
                                        ElementsAre(5)));
    EXPECT_EQ(mock_ptr->harvest_calls, 1);
}

TEST(Test__PrefixCachePrefillFlow, LLEPFullHitAtUnalignedBoundaryRecomputesRemainder)
{
    auto mock = std::make_unique<PrefixFlowMockRunner>();
    auto *mock_ptr = mock.get();
    mock_ptr->lookup_result.supported = true;
    mock_ptr->lookup_result.cache_enabled = true;
    mock_ptr->lookup_result.block_size = 2;
    mock_ptr->lookup_result.cached_tokens = 5;
    mock_ptr->lookup_result.has_terminal_logits = true;

    auto runner = makeRunner(std::move(mock),
                             /*mtp_enabled=*/false,
                             /*mtp_draft_tokens=*/1,
                             RoutedExpertAssignmentPolicy::LeastLoadedResident);
    const std::vector<int32_t> prompt = {1, 2, 3, 4, 5};
    ASSERT_TRUE(runner->prefill(prompt)) << runner->lastError();

    EXPECT_EQ(mock_ptr->clear_calls, 0);
    EXPECT_EQ(mock_ptr->populate_calls, 1);
    EXPECT_THAT(mock_ptr->populated_tokens, ElementsAre(4));
    EXPECT_EQ(mock_ptr->restore_terminal_calls, 0);
    EXPECT_EQ(mock_ptr->forward_calls, 1);
    EXPECT_THAT(mock_ptr->last_forward_tokens, ElementsAre(5));
    EXPECT_EQ(mock_ptr->harvest_calls, 1);

    const auto probe = runner->prefixStateProbe();
    EXPECT_FALSE(probe.prefix_request.hit);
    EXPECT_TRUE(probe.prefix_request.partial_hit);
    EXPECT_EQ(probe.prefix_request.matched_tokens, 4);
    EXPECT_EQ(probe.prefix_request.matched_blocks, 2);
    EXPECT_FALSE(probe.prefix_request.terminal_logits_restored);
}

TEST(Test__PrefixCachePrefillFlow, LLEPFullHitWithTerminalRuntimeSnapshotRestoresUnalignedMTPState)
{
    auto make_block = [](int block_index,
                         int token_start,
                         int token_count,
                         bool terminal) {
        PrefixBlockHandle block;
        block.key.fingerprint = 0x1234;
        block.key.block_index = block_index;
        block.key.token_start = token_start;
        block.key.token_count = token_count;
        block.total_bytes = 1;
        block.layout.block_size = 2;
        block.layout.includes_mtp_state = true;
        block.layout.hybrid_state_bytes = 16;
        block.has_hybrid_state = terminal;
        block.has_terminal_hidden = terminal;
        block.has_terminal_logits = terminal;
        block.has_model_runtime_state = terminal;
        if (terminal)
        {
            block.mtp_storage =
                std::make_shared<std::vector<uint8_t>>(4, 0x42);
            block.mtp_payload = block.mtp_storage->data();
            block.model_runtime_state_storage =
                std::make_shared<std::vector<uint8_t>>(4, 0x7f);
        }
        return block;
    };

    auto mock = std::make_unique<PrefixFlowMockRunner>();
    auto *mock_ptr = mock.get();
    mock_ptr->lookup_result.supported = true;
    mock_ptr->lookup_result.cache_enabled = true;
    mock_ptr->lookup_result.block_size = 2;
    mock_ptr->lookup_result.cached_tokens = 5;
    mock_ptr->lookup_result.has_terminal_logits = true;
    mock_ptr->lookup_result.has_terminal_hidden = true;
    mock_ptr->lookup_result.blocks = {
        make_block(0, 0, 2, false),
        make_block(1, 2, 2, false),
        make_block(2, 4, 1, true),
    };

    auto runner = makeRunner(std::move(mock),
                             /*mtp_enabled=*/true,
                             /*mtp_draft_tokens=*/1,
                             RoutedExpertAssignmentPolicy::LeastLoadedResident);
    const std::vector<int32_t> prompt = {1, 2, 3, 4, 5};
    ASSERT_TRUE(runner->prefill(prompt)) << runner->lastError();

    EXPECT_EQ(mock_ptr->clear_calls, 0);
    EXPECT_EQ(mock_ptr->populate_calls, 1);
    EXPECT_THAT(mock_ptr->populated_tokens, ElementsAre(5));
    EXPECT_EQ(mock_ptr->restore_terminal_calls, 1);
    EXPECT_EQ(mock_ptr->forward_calls, 0);
    EXPECT_EQ(mock_ptr->harvest_calls, 1);

    const auto probe = runner->prefixStateProbe();
    EXPECT_TRUE(probe.prefix_request.hit);
    EXPECT_FALSE(probe.prefix_request.partial_hit);
    EXPECT_EQ(probe.prefix_request.matched_tokens, 5);
    EXPECT_EQ(probe.prefix_request.matched_blocks, 3);
    EXPECT_TRUE(probe.prefix_request.terminal_logits_restored);
    EXPECT_TRUE(probe.prefix_request.terminal_hidden_restored);
    EXPECT_TRUE(probe.prefix_request.mtp_state_restored);
}

TEST(Test__PrefixCachePrefillFlow, LLEPConfiguredPrefillWindowSegmentsUncachedPrefill)
{
    auto mock = std::make_unique<PrefixFlowMockRunner>();
    auto *mock_ptr = mock.get();

    auto runner = makeRunner(std::move(mock),
                             /*mtp_enabled=*/false,
                             /*mtp_draft_tokens=*/1,
                             RoutedExpertAssignmentPolicy::LeastLoadedResident,
                             /*prefix_block_size=*/2,
                             /*prefill_window_tokens=*/2,
                             /*prefix_cache_enabled=*/false);
    const std::vector<int32_t> prompt = {1, 2, 3, 4, 5};
    ASSERT_TRUE(runner->prefill(prompt)) << runner->lastError();

    EXPECT_EQ(mock_ptr->lookup_calls, 0);
    EXPECT_EQ(mock_ptr->clear_calls, 0);
    EXPECT_EQ(mock_ptr->populate_calls, 0);
    EXPECT_EQ(mock_ptr->forward_calls, 3);
    EXPECT_THAT(mock_ptr->forward_token_batches,
                ElementsAre(ElementsAre(1, 2),
                            ElementsAre(3, 4),
                            ElementsAre(5)));
    EXPECT_EQ(mock_ptr->harvest_calls, 0);
}

TEST(Test__PrefixCachePrefillFlow, LLEPPrefixMissWithoutPrefixBlockSizeUsesSinglePrefill)
{
    auto mock = std::make_unique<PrefixFlowMockRunner>();
    auto *mock_ptr = mock.get();
    mock_ptr->lookup_result.supported = true;
    mock_ptr->lookup_result.cache_enabled = true;
    mock_ptr->lookup_result.block_size = 0;
    mock_ptr->lookup_result.cached_tokens = 0;

    auto runner = makeRunner(std::move(mock),
                             /*mtp_enabled=*/false,
                             /*mtp_draft_tokens=*/1,
                             RoutedExpertAssignmentPolicy::LeastLoadedResident,
                             /*prefix_block_size=*/0);
    ASSERT_TRUE(runner->prefill({1, 2, 3})) << runner->lastError();

    EXPECT_EQ(mock_ptr->clear_calls, 0);
    EXPECT_EQ(mock_ptr->populate_calls, 0);
    EXPECT_EQ(mock_ptr->forward_calls, 1);
    EXPECT_THAT(mock_ptr->last_forward_tokens, ElementsAre(1, 2, 3));
    EXPECT_EQ(mock_ptr->harvest_calls, 1);
}

TEST(Test__PrefixCachePrefillFlow, PopulateFailureHardFailsWithoutMissFallback)
{
    auto mock = std::make_unique<PrefixFlowMockRunner>();
    auto *mock_ptr = mock.get();
    mock_ptr->lookup_result.supported = true;
    mock_ptr->lookup_result.cache_enabled = true;
    mock_ptr->lookup_result.block_size = 2;
    mock_ptr->lookup_result.cached_tokens = 2;
    mock_ptr->populate_ok = false;

    auto runner = makeRunner(std::move(mock));
    ASSERT_FALSE(runner->prefill({1, 2, 3, 4}));

    EXPECT_THAT(runner->lastError(), HasSubstr("Prefix cache populate failed"));
    EXPECT_THAT(runner->lastError(), HasSubstr("refusing to downgrade the hit to a miss"));
    EXPECT_EQ(mock_ptr->clear_calls, 0);
    EXPECT_EQ(mock_ptr->populate_calls, 1);
    EXPECT_EQ(mock_ptr->forward_calls, 0);
    EXPECT_EQ(mock_ptr->harvest_calls, 0);
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
    // This lightweight host runner consumes only the three live rows. The
    // retained bucket still authenticates the schedule, but padding is charged
    // only when a native device executable actually launches physical rows.
    EXPECT_EQ(probe.prefill_chunk_padded_tokens, 0u);
    EXPECT_EQ(probe.prefill_chunk_failures, 0u);
}

TEST(Test__PrefixCachePrefillFlow,
     NativeCapturedOneTokenPrefixSuffixUsesSerialDecodeTransaction)
{
    ScopedPrefillChunkScheduleEnv env;
    auto mock = std::make_unique<PrefixFlowMockRunner>();
    auto *mock_ptr = mock.get();
    mock_ptr->supports_chunk_schedule = true;
    mock_ptr->serving_graph_preparation_kind =
        ServingGraphPreparationKind::NativeDeviceExecutableFamily;
    mock_ptr->lookup_result.supported = true;
    mock_ptr->lookup_result.cache_enabled = true;
    mock_ptr->lookup_result.block_size = 2;
    mock_ptr->lookup_result.cached_tokens = 2;

    auto runner = makeRunner(std::move(mock));
    ASSERT_TRUE(runner->prefill({1, 2, 3})) << runner->lastError();

    EXPECT_EQ(mock_ptr->forward_calls, 1);
    EXPECT_THAT(mock_ptr->last_forward_tokens, ElementsAre(3));
    EXPECT_EQ(mock_ptr->restored_prefix_bridge_calls, 0)
        << "Non-MTP restoration owns only main-model state and must not enter the MTP bridge.";
    EXPECT_EQ(mock_ptr->chunk_schedule_calls, 0)
        << "A one-row restored suffix is mathematically decode and must not acquire padded prefill arithmetic.";
    EXPECT_EQ(mock_ptr->position, 3);

    const auto probe = runner->prefixStateProbe();
    EXPECT_EQ(probe.prefill_chunk_schedules, 0u);
    EXPECT_EQ(probe.prefill_chunk_successful_schedules, 0u);
    EXPECT_EQ(probe.prefill_chunks, 0u);
    EXPECT_EQ(probe.prefill_chunk_real_tokens, 0u);
    EXPECT_EQ(probe.prefill_chunk_padded_tokens, 0u);
    EXPECT_EQ(probe.prefill_chunk_failures, 0u);
}

TEST(Test__PrefixCachePrefillFlow,
     NativeCapturedShortPrefillFailsWithoutRetainedGraphScheduleSupport)
{
    ScopedPrefillChunkScheduleEnv env;
    auto mock = std::make_unique<PrefixFlowMockRunner>();
    auto *mock_ptr = mock.get();
    mock_ptr->serving_graph_preparation_kind =
        ServingGraphPreparationKind::NativeDeviceExecutableFamily;
    mock_ptr->lookup_result.supported = true;
    mock_ptr->lookup_result.cache_enabled = true;
    mock_ptr->lookup_result.block_size = 2;
    mock_ptr->lookup_result.cached_tokens = 0;

    auto runner = makeRunner(std::move(mock));
    ASSERT_FALSE(runner->prefill({1, 2}));
    EXPECT_THAT(
        runner->lastError(),
        HasSubstr("runner does not support prefill chunk scheduling"));
    EXPECT_EQ(mock_ptr->forward_calls, 0);
    EXPECT_EQ(mock_ptr->chunk_schedule_calls, 0);
    EXPECT_EQ(mock_ptr->harvest_calls, 0);
}

TEST(Test__PrefixCachePrefillFlow, LongPrefixSuffixFailsWhenChunkScheduleUnsupported)
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
    ASSERT_FALSE(runner->prefill(prompt));
    EXPECT_THAT(runner->lastError(), HasSubstr("runner does not support prefill chunk scheduling"));

    EXPECT_EQ(mock_ptr->chunk_schedule_calls, 0);
    EXPECT_EQ(mock_ptr->forward_calls, 0);
    EXPECT_EQ(mock_ptr->harvest_calls, 0);

    const auto probe = runner->prefixStateProbe();
    EXPECT_EQ(probe.prefill_chunk_schedules, 0u);
    EXPECT_EQ(probe.prefill_chunks, 0u);
    EXPECT_EQ(probe.prefill_chunk_failures, 1u);
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

    EXPECT_EQ(mock_ptr->clear_calls, 0);
    EXPECT_EQ(mock_ptr->populate_calls, 1);
    EXPECT_THAT(mock_ptr->populated_tokens, ElementsAre(2));
    EXPECT_EQ(mock_ptr->forward_calls, 1);
    EXPECT_THAT(mock_ptr->last_forward_tokens, ElementsAre(3, 4, 5));
}

TEST(Test__PrefixCachePrefillFlow,
     OneTokenMTPPartialHitUsesTypedRestoredPrefixDecodeBridge)
{
    auto mock = std::make_unique<PrefixFlowMockRunner>();
    auto *mock_ptr = mock.get();
    mock_ptr->lookup_result.supported = true;
    mock_ptr->lookup_result.cache_enabled = true;
    mock_ptr->lookup_result.block_size = 2;
    mock_ptr->lookup_result.cached_tokens = 4;
    mock_ptr->lookup_result.has_terminal_hidden = true;

    auto runner = makeRunner(std::move(mock), /*mtp_enabled=*/true);
    ASSERT_TRUE(runner->prefill({1, 2, 3, 4, 5}))
        << runner->lastError();

    ASSERT_EQ(mock_ptr->restored_prefix_bridge_calls, 1);
    ASSERT_EQ(mock_ptr->restored_prefix_bridge_requests.size(), 1u);
    EXPECT_EQ(mock_ptr->restored_prefix_bridge_requests.front().token_id, 5);
    EXPECT_EQ(
        mock_ptr->restored_prefix_bridge_requests.front()
            .restored_prefix_tokens,
        4);
    EXPECT_EQ(mock_ptr->forward_calls, 1);
    EXPECT_THAT(mock_ptr->last_forward_tokens, ElementsAre(5));
    EXPECT_EQ(mock_ptr->position, 5);
    EXPECT_EQ(mock_ptr->shifted_mtp_rows, 4);
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
    EXPECT_EQ(mock_ptr->commit_mtp_calls, 2);
    EXPECT_EQ(mock_ptr->checkpoint_terminal_hidden_commit_calls, 1);
    EXPECT_EQ(mock_ptr->last_commit_already_appended, 1);
    EXPECT_THAT(mock_ptr->last_commit_tokens,
                ElementsAre(mock_ptr->prefill_argmax_token,
                            mock_ptr->mtp_argmax_token));
    EXPECT_EQ(mock_ptr->grouped_mtp_publication_calls, 1);
    EXPECT_EQ(mock_ptr->last_grouped_mtp_step.accepted_count, 2);
    EXPECT_EQ(mock_ptr->last_grouped_mtp_step.rejected_count, 0);

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
    EXPECT_EQ(mock_ptr->clear_calls, 0);
    EXPECT_EQ(mock_ptr->populate_calls, 1);
    EXPECT_THAT(mock_ptr->populated_tokens, ElementsAre(2));
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
    EXPECT_EQ(mock_ptr->grouped_mtp_publication_calls, 1);
    EXPECT_EQ(mock_ptr->last_grouped_mtp_step.accepted_count, 1);
    EXPECT_EQ(mock_ptr->last_grouped_mtp_step.rejected_count, 1);

    const auto probe = runner->prefixStateProbe();
    EXPECT_EQ(probe.mtp_draft_steps, 1u);
    EXPECT_EQ(probe.mtp_accepted_tokens, 0u);
    EXPECT_EQ(probe.mtp_rejected_tokens, 1u);
    EXPECT_EQ(probe.mtp_rollbacks, 0u);
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
    EXPECT_EQ(mock_ptr->commit_mtp_calls, 2);
    EXPECT_EQ(mock_ptr->checkpoint_terminal_hidden_commit_calls, 1);
    EXPECT_EQ(mock_ptr->last_commit_already_appended, 1);
    EXPECT_THAT(mock_ptr->last_commit_tokens, ElementsAre(9, 11, 12, 13));
    EXPECT_EQ(mock_ptr->grouped_mtp_publication_calls, 1);
    EXPECT_EQ(mock_ptr->last_grouped_mtp_step.accepted_count, 4);
    EXPECT_EQ(mock_ptr->last_grouped_mtp_step.rejected_count, 0);

    const auto probe = runner->prefixStateProbe();
    EXPECT_EQ(probe.mtp_draft_steps, 3u);
    EXPECT_EQ(probe.mtp_accepted_tokens, 3u);
    EXPECT_EQ(probe.mtp_rejected_tokens, 0u);
    EXPECT_EQ(probe.mtp_rollbacks, 0u);
    EXPECT_EQ(probe.mtp_verifier_runs, 1u);
    EXPECT_EQ(probe.mtp_verifier_token_count, 4u);
}

TEST(Test__PrefixCachePrefillFlow, FullPrefixTerminalRestoreSupportsChainedMTPDraftDepthThree)
{
    auto mock = std::make_unique<PrefixFlowMockRunner>();
    auto *mock_ptr = mock.get();
    mock_ptr->lookup_result.supported = true;
    mock_ptr->lookup_result.cache_enabled = true;
    mock_ptr->lookup_result.block_size = 2;
    mock_ptr->lookup_result.cached_tokens = 4;
    mock_ptr->lookup_result.has_terminal_logits = true;
    mock_ptr->lookup_result.has_terminal_hidden = true;
    mock_ptr->mtp_argmax_tokens = {11, 12, 13};
    mock_ptr->verify_argmax_tokens = {11, 12, 13, 14};

    auto runner = makeRunner(std::move(mock), /*mtp_enabled=*/true, /*mtp_draft_tokens=*/3);
    ASSERT_TRUE(runner->prefill({1, 2, 3, 4})) << runner->lastError();

    EXPECT_EQ(mock_ptr->restore_terminal_calls, 1);
    EXPECT_EQ(mock_ptr->forward_calls, 0);
    EXPECT_EQ(mock_ptr->populate_calls, 1);
    EXPECT_THAT(mock_ptr->populated_tokens, ElementsAre(4));
    EXPECT_EQ(mock_ptr->harvest_calls, 1);

    auto step = runner->decodeStep();
    ASSERT_TRUE(step.success()) << step.error;
    EXPECT_THAT(step.tokens, ElementsAre(9, 11, 12, 13));

    EXPECT_EQ(mock_ptr->forward_mtp_calls, 3);
    EXPECT_EQ(mock_ptr->chained_mtp_calls, 2);
    EXPECT_THAT(mock_ptr->chained_mtp_positions, ElementsAre(5, 6));
    EXPECT_EQ(mock_ptr->restore_live_calls, 0);
    EXPECT_EQ(mock_ptr->commit_mtp_calls, 2);
    EXPECT_EQ(mock_ptr->checkpoint_terminal_hidden_commit_calls, 1);
    EXPECT_EQ(mock_ptr->last_commit_already_appended, 1);
    EXPECT_THAT(mock_ptr->last_commit_tokens, ElementsAre(9, 11, 12, 13));
    EXPECT_EQ(mock_ptr->grouped_mtp_publication_calls, 1);
    EXPECT_EQ(mock_ptr->last_grouped_mtp_step.accepted_count, 4);
    EXPECT_EQ(mock_ptr->last_grouped_mtp_step.rejected_count, 0);

    const auto probe = runner->prefixStateProbe();
    EXPECT_TRUE(probe.prefix_request.hit);
    EXPECT_TRUE(probe.prefix_request.terminal_logits_restored);
    EXPECT_TRUE(probe.prefix_request.terminal_hidden_restored);
    EXPECT_EQ(probe.mtp_draft_steps, 3u);
    EXPECT_EQ(probe.mtp_accepted_tokens, 3u);
    EXPECT_EQ(probe.mtp_rejected_tokens, 0u);
    EXPECT_EQ(probe.mtp_rollbacks, 0u);
    EXPECT_EQ(probe.mtp_verifier_runs, 1u);
    EXPECT_EQ(probe.mtp_verifier_token_count, 4u);
}

TEST(Test__PrefixCachePrefillFlow, PartialPrefixHitChainedMTPDraftDepthThreePublishesGroupedCorrectionOnReject)
{
    auto mock = std::make_unique<PrefixFlowMockRunner>();
    auto *mock_ptr = mock.get();
    mock_ptr->lookup_result.supported = true;
    mock_ptr->lookup_result.cache_enabled = true;
    mock_ptr->lookup_result.block_size = 2;
    mock_ptr->lookup_result.cached_tokens = 2;
    mock_ptr->mtp_argmax_tokens = {11, 12, 13};
    mock_ptr->verify_argmax_tokens = {11, 15, 14};

    auto runner = makeRunner(std::move(mock), /*mtp_enabled=*/true, /*mtp_draft_tokens=*/3);
    ASSERT_TRUE(runner->prefill({1, 2, 3, 4})) << runner->lastError();

    auto step = runner->decodeStep();
    ASSERT_TRUE(step.success()) << step.error;
    EXPECT_THAT(step.tokens, ElementsAre(9, 11, 15));

    EXPECT_EQ(mock_ptr->forward_mtp_calls, 3);
    EXPECT_EQ(mock_ptr->chained_mtp_calls, 2);
    EXPECT_THAT(mock_ptr->chained_mtp_positions, ElementsAre(5, 6));
    EXPECT_EQ(mock_ptr->restore_live_calls, 0);
    EXPECT_THAT(mock_ptr->last_forward_tokens, ElementsAre(9, 11, 12, 13));
    EXPECT_EQ(mock_ptr->commit_mtp_calls, 2);
    EXPECT_EQ(mock_ptr->checkpoint_terminal_hidden_commit_calls, 1);
    EXPECT_EQ(mock_ptr->last_commit_already_appended, 1);
    EXPECT_TRUE(mock_ptr->last_commit_allow_speculative_discard);
    EXPECT_EQ(mock_ptr->last_commit_position_offset_override, 4);
    EXPECT_THAT(mock_ptr->last_commit_tokens, ElementsAre(9, 11));
    EXPECT_EQ(mock_ptr->grouped_mtp_publication_calls, 1);
    EXPECT_EQ(mock_ptr->last_grouped_mtp_step.accepted_count, 2);
    EXPECT_EQ(mock_ptr->last_grouped_mtp_step.rejected_count, 2);

    const auto probe = runner->prefixStateProbe();
    EXPECT_EQ(probe.mtp_draft_steps, 3u);
    EXPECT_EQ(probe.mtp_accepted_tokens, 1u);
    EXPECT_EQ(probe.mtp_rejected_tokens, 1u);
    EXPECT_EQ(probe.mtp_rollbacks, 0u);
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

    EXPECT_EQ(mock_ptr->clear_calls, 0);
    EXPECT_EQ(mock_ptr->populate_calls, 1);
    EXPECT_THAT(mock_ptr->populated_tokens, ElementsAre(2));
    EXPECT_EQ(mock_ptr->forward_calls, 1);
    EXPECT_THAT(mock_ptr->last_forward_tokens, ElementsAre(3, 4));
}

TEST(Test__PrefixCachePrefillFlow, AdvertisedTerminalRestoreFailureHardFails)
{
    auto mock = std::make_unique<PrefixFlowMockRunner>();
    auto *mock_ptr = mock.get();
    mock_ptr->lookup_result.supported = true;
    mock_ptr->lookup_result.cache_enabled = true;
    mock_ptr->lookup_result.block_size = 2;
    mock_ptr->lookup_result.cached_tokens = 4;
    mock_ptr->lookup_result.has_terminal_logits = true;
    mock_ptr->restore_terminal_ok = false;

    auto runner = makeRunner(std::move(mock));
    ASSERT_FALSE(runner->prefill({1, 2, 3, 4}));

    EXPECT_THAT(runner->lastError(), HasSubstr("Prefix cache terminal restore failed"));
    EXPECT_EQ(mock_ptr->clear_calls, 0);
    EXPECT_EQ(mock_ptr->populate_calls, 1);
    EXPECT_EQ(mock_ptr->restore_terminal_calls, 1);
    EXPECT_EQ(mock_ptr->forward_calls, 0);
    EXPECT_EQ(mock_ptr->harvest_calls, 0);
}
