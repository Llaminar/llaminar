/**
 * @file Test__RankOrchestrator.cpp
 * @brief Unit tests for RankOrchestrator mocks and interface contract
 * @author David Sanftenberg
 * @date January 2026
 *
 * Tests the mock implementations used for testing RankOrchestrator
 * coordination logic. The mocks enable testing LOCAL tensor parallelism
 * coordination without real devices.
 *
 * Note: Tests for the actual RankOrchestrator class will be enabled
 * once the implementation is added to the build.
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "execution/local_execution/orchestrators/IInferenceRunner.h"
#include "execution/local_execution/orchestrators/RankOrchestrator.h"
#include "execution/debug/TPSnapshot.h"
#include "execution/mtp/MTPSpecStateContract.h"
#include "collective/ILocalTPContext.h"
#include "backends/GlobalDeviceAddress.h"
#include "config/OrchestrationConfig.h"
#include "tensors/Tensors.h"
#include "utils/DebugEnv.h"
#include "mocks/MockModelContext.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <vector>
#include <memory>
#include <stdexcept>
#include <utility>

using namespace llaminar2;

struct ForwardMTPRendezvous
{
    explicit ForwardMTPRendezvous(int expected_) : expected(expected_) {}

    int expected = 0;
    std::atomic<int> arrivals{0};
    std::mutex mutex;
    std::condition_variable cv;
};

struct MTPPublicationRendezvous
{
    explicit MTPPublicationRendezvous(int expected_) : expected(expected_) {}

    int expected = 0;
    std::atomic<int> arrivals{0};
    std::mutex mutex;
    std::condition_variable cv;
};

struct ChainedMTPRendezvous
{
    explicit ChainedMTPRendezvous(int expected_) : expected(expected_) {}

    int expected = 0;
    std::atomic<int> arrivals{0};
    std::mutex mutex;
    std::condition_variable cv;
};

// =============================================================================
// MockDeviceGraphOrchestrator - Mock for per-device runners
// =============================================================================

/**
 * @brief Mock inference runner for per-device testing
 *
 * Tracks method calls and provides configurable return values for testing
 * the RankOrchestrator coordination logic.
 */
class MockDeviceGraphOrchestrator : public IInferenceRunner
{
public:
    struct Config
    {
        int vocab_size = 32000;
        bool forward_should_fail = false;
        std::string architecture = "mock_qwen2";
        int forward_sleep_ms = 0;
    };

    MockDeviceGraphOrchestrator() : MockDeviceGraphOrchestrator(Config{}) {}

    explicit MockDeviceGraphOrchestrator(const Config &config)
        : config_(config), position_(0)
    {
        logits_.resize(static_cast<size_t>(config_.vocab_size), 0.0f);
    }

    // =====================================================================
    // IInferenceRunner Implementation
    // =====================================================================

    bool forward(const int *tokens, int seq_len) override
    {
        (void)tokens;
        forward_calls_.fetch_add(1, std::memory_order_relaxed);
        if (config_.forward_sleep_ms > 0)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(config_.forward_sleep_ms));
        }
        if (config_.forward_should_fail)
        {
            return false;
        }
        position_ += seq_len;
        return true;
    }

    const float *logits() const override
    {
        return logits_.data();
    }

    bool forwardMTP(int32_t draft_condition_token) override
    {
        forward_mtp_calls_.fetch_add(1, std::memory_order_relaxed);
        last_mtp_condition_token_ = draft_condition_token;
        if (forward_mtp_rendezvous_)
        {
            std::unique_lock<std::mutex> lock(forward_mtp_rendezvous_->mutex);
            forward_mtp_rendezvous_->arrivals.fetch_add(1, std::memory_order_acq_rel);
            forward_mtp_rendezvous_->cv.notify_all();
            const bool all_arrived = forward_mtp_rendezvous_->cv.wait_for(
                lock,
                std::chrono::milliseconds(500),
                [barrier = forward_mtp_rendezvous_]()
                {
                    return barrier->arrivals.load(std::memory_order_acquire) >= barrier->expected;
                });
            if (!all_arrived)
                return false;
        }
        return forward_mtp_ok_;
    }

    bool supportsChainedMTPDrafts() const override
    {
        return supports_chained_mtp_drafts_;
    }

    bool forwardMTPFromLastDraft(
        int32_t draft_condition_token,
        int position_id) override
    {
        forward_mtp_from_last_draft_calls_.fetch_add(1, std::memory_order_relaxed);
        last_chained_mtp_condition_token_ = draft_condition_token;
        last_chained_mtp_position_id_ = position_id;
        if (chained_mtp_rendezvous_)
        {
            std::unique_lock<std::mutex> lock(chained_mtp_rendezvous_->mutex);
            chained_mtp_rendezvous_->arrivals.fetch_add(1, std::memory_order_acq_rel);
            chained_mtp_rendezvous_->cv.notify_all();
            const bool all_arrived = chained_mtp_rendezvous_->cv.wait_for(
                lock,
                std::chrono::milliseconds(500),
                [barrier = chained_mtp_rendezvous_]()
                {
                    return barrier->arrivals.load(std::memory_order_acquire) >= barrier->expected;
                });
            if (!all_arrived)
                return false;
        }
        return forward_mtp_from_last_draft_ok_;
    }

    bool supportsMTPSpecStatePublication() const override
    {
        return supports_mtp_spec_state_publication_;
    }

    MTPVerifierRowCapability mtpVerifierRowCapability() const override
    {
        return mtp_verifier_row_capability_;
    }

    MTPVerifierEconomyCapability mtpVerifierEconomyCapability() const override
    {
        return mtp_verifier_economy_capability_;
    }

    bool publishAcceptedMTPSpecState(
        const MTPSpecStepPlan &plan,
        std::string *error = nullptr) override
    {
        publish_mtp_spec_state_calls_.fetch_add(1, std::memory_order_relaxed);
        last_mtp_spec_state_plan_ = plan;
        if (mtp_publication_rendezvous_)
        {
            std::unique_lock<std::mutex> lock(mtp_publication_rendezvous_->mutex);
            mtp_publication_rendezvous_->arrivals.fetch_add(1, std::memory_order_acq_rel);
            mtp_publication_rendezvous_->cv.notify_all();
            const bool all_arrived = mtp_publication_rendezvous_->cv.wait_for(
                lock,
                std::chrono::milliseconds(500),
                [barrier = mtp_publication_rendezvous_]()
                {
                    return barrier->arrivals.load(std::memory_order_acquire) >= barrier->expected;
                });
            if (!all_arrived)
            {
                if (error)
                    *error = "mock MTP spec-state publication rendezvous timed out";
                return false;
            }
        }
        if (!supports_mtp_spec_state_publication_)
        {
            if (error)
                *error = "mock MTP spec-state publication disabled";
            return false;
        }
        if (!publish_mtp_spec_state_ok_)
        {
            if (error)
                *error = "mock MTP spec-state publication failed";
            return false;
        }
        position_ = plan.target_cached_tokens;
        return true;
    }

    bool publishAcceptedMTPSpecStateBatch(
        const MTPSpecStepPlanBatch &plans,
        std::string *error = nullptr) override
    {
        publish_mtp_spec_state_batch_calls_.fetch_add(
            1,
            std::memory_order_relaxed);
        last_mtp_spec_state_batch_ = plans;
        if (mtp_publication_rendezvous_)
        {
            std::unique_lock<std::mutex> lock(mtp_publication_rendezvous_->mutex);
            mtp_publication_rendezvous_->arrivals.fetch_add(1, std::memory_order_acq_rel);
            mtp_publication_rendezvous_->cv.notify_all();
            const bool all_arrived = mtp_publication_rendezvous_->cv.wait_for(
                lock,
                std::chrono::milliseconds(500),
                [barrier = mtp_publication_rendezvous_]()
                {
                    return barrier->arrivals.load(std::memory_order_acquire) >= barrier->expected;
                });
            if (!all_arrived)
            {
                if (error)
                    *error = "mock MTP spec-state batch publication rendezvous timed out";
                return false;
            }
        }
        if (!supports_mtp_spec_state_publication_)
        {
            if (error)
                *error = "mock MTP spec-state batch publication disabled";
            return false;
        }
        if (!publish_mtp_spec_state_ok_)
        {
            if (error)
                *error = "mock MTP spec-state batch publication failed";
            return false;
        }
        if (!plans.ok ||
            plans.request_count <= 0 ||
            static_cast<int>(plans.steps.size()) != plans.request_count)
        {
            if (error)
                *error = "mock MTP spec-state batch publication received invalid plans";
            return false;
        }

        /*
         * The mock owns one scalar position rather than per-request state.
         * Store the highest target count so tests can still tell that the
         * published batch, not the stale single-step path, mutated state.
         */
        int max_target_cached_tokens = 0;
        for (const MTPSpecStepPlan &step : plans.steps)
            max_target_cached_tokens =
                std::max(max_target_cached_tokens, step.target_cached_tokens);
        position_ = max_target_cached_tokens;
        return true;
    }

    const float *mtpLogits() const override
    {
        return mtp_logits_.empty() ? logits_.data() : mtp_logits_.data();
    }

    int sampleGreedyFromMTPLogitsOnDevice() override
    {
        ++sample_mtp_logits_calls_;
        const float *values = mtpLogits();
        if (!values || config_.vocab_size <= 0)
            return -1;

        int best = 0;
        float best_value = values[0];
        for (int i = 1; i < config_.vocab_size; ++i)
        {
            if (values[i] > best_value)
            {
                best = i;
                best_value = values[i];
            }
        }
        return best;
    }

    bool commitMTPShiftedRowsFromPartialForward(
        const int32_t *tokens,
        int token_count,
        int already_appended_tokens,
        int main_forward_token_count,
        bool allow_speculative_discard = false,
        int position_offset_override = -1,
        int already_appended_shifted_kv_tokens = -1) override
    {
        (void)already_appended_shifted_kv_tokens;
        ++commit_mtp_shifted_rows_calls_;
        last_commit_mtp_already_appended_ = already_appended_tokens;
        last_commit_mtp_main_forward_token_count_ = main_forward_token_count;
        last_commit_mtp_allow_speculative_discard_ = allow_speculative_discard;
        last_commit_mtp_position_offset_override_ = position_offset_override;
        last_commit_mtp_tokens_.clear();
        if (tokens && token_count > 0)
            last_commit_mtp_tokens_.assign(tokens, tokens + token_count);
        return commit_mtp_shifted_rows_ok_;
    }

    bool commitMTPShiftedRowFromCurrentTerminalHidden(
        int32_t token,
        int already_appended_tokens,
        bool allow_speculative_discard = false,
        int position_offset_override = -1) override
    {
        const int32_t one_token = token;
        return commitMTPShiftedRowsFromPartialForward(
            &one_token,
            1,
            already_appended_tokens,
            /*main_forward_token_count=*/0,
            allow_speculative_discard,
            position_offset_override);
    }

    bool ensureMTPCheckpointTerminalHidden() override
    {
        ++ensure_mtp_checkpoint_terminal_hidden_calls_;
        return ensure_mtp_checkpoint_terminal_hidden_ok_;
    }

    bool hasMTPLogitsLocal() const override
    {
        return mtp_logits_local_ != nullptr;
    }

    LogitsLocalInfo getMTPLogitsLocalInfo() const override
    {
        get_mtp_logits_local_info_calls_.fetch_add(
            1,
            std::memory_order_relaxed);
        if (!mtp_logits_local_)
            return {};
        const auto &shape = mtp_logits_local_->shape();
        return LogitsLocalInfo{
            nullptr,
            std::nullopt,
            shape.size() >= 2 ? shape[1] : 0,
            mtp_logits_local_.get(),
            nullptr,
            nullptr,
            nullptr,
            0};
    }

    LogitsLocalInfo consumeMTPLogitsLocalInfoForSampling() override
    {
        consume_mtp_logits_local_info_calls_.fetch_add(
            1,
            std::memory_order_relaxed);
        return makeMTPLocalInfo();
    }

    LogitsLocalInfo makeMTPLocalInfo() const
    {
        if (!mtp_logits_local_)
            return {};
        const auto &shape = mtp_logits_local_->shape();
        return LogitsLocalInfo{
            nullptr,
            std::nullopt,
            shape.size() >= 2 ? shape[1] : 0,
            mtp_logits_local_.get(),
            nullptr,
            nullptr,
            nullptr,
            0};
    }

    bool setComputeAllPositionLogits(bool enabled) override
    {
        set_all_position_logits_calls_.fetch_add(1, std::memory_order_relaxed);
        compute_all_position_logits_ = enabled;
        return set_all_position_logits_ok_;
    }

    bool setComputeRowIndexedAllPositionLogits(bool enabled, int row_count) override
    {
        set_row_indexed_all_position_logits_calls_.fetch_add(
            1,
            std::memory_order_relaxed);
        compute_row_indexed_all_position_logits_ = enabled;
        row_indexed_all_position_logit_rows_ = enabled ? row_count : 0;
        return set_row_indexed_all_position_logits_ok_;
    }

    bool setMTPSpecVerifierInputPlan(
        const MTPSpecDecodeVerifierInputPlan &plan) override
    {
        set_mtp_spec_verifier_input_plan_calls_.fetch_add(
            1,
            std::memory_order_relaxed);
        last_mtp_spec_verifier_input_plan_ = plan;
        return set_mtp_spec_verifier_input_plan_ok_;
    }

    void clearMTPSpecVerifierInputPlan() override
    {
        clear_mtp_spec_verifier_input_plan_calls_.fetch_add(
            1,
            std::memory_order_relaxed);
    }

    const float *getAllPositionLogits() const override
    {
        return all_position_logits_.empty() ? logits_.data() : all_position_logits_.data();
    }

    bool hasAllPositionLogitsLocal() const override
    {
        return all_position_logits_local_ != nullptr;
    }

    LogitsLocalInfo getAllPositionLogitsLocalInfo() const override
    {
        get_all_position_logits_local_info_calls_.fetch_add(
            1,
            std::memory_order_relaxed);
        if (!all_position_logits_local_)
            return {};
        return makeAllPositionLocalInfo();
    }

    LogitsLocalInfo consumeAllPositionLogitsLocalInfoForSampling() override
    {
        consume_all_position_logits_local_info_calls_.fetch_add(
            1,
            std::memory_order_relaxed);
        if (!all_position_logits_local_)
            return {};
        return makeAllPositionLocalInfo();
    }

    LogitsLocalInfo makeAllPositionLocalInfo() const
    {
        const auto &shape = all_position_logits_local_->shape();
        return LogitsLocalInfo{
            nullptr,
            std::nullopt,
            shape.size() >= 2 ? shape[1] : 0,
            all_position_logits_local_.get(),
            nullptr,
            nullptr,
            nullptr,
            0};
    }

    std::string mtpDecodeUnsupportedReason() const override
    {
        return mtp_unsupported_reason_;
    }

    DeviceId primaryDeviceId() const override
    {
        return device_id_;
    }

    int vocab_size() const override
    {
        return config_.vocab_size;
    }

    bool supportsMTPSidecarLogitsStreamHandoff() const override
    {
        return supports_mtp_sidecar_logits_stream_handoff_;
    }

    bool supportsMTPDeviceDraftTokenInput() const override
    {
        return supports_mtp_device_draft_token_input_;
    }

    bool supportsMTPSidecarPreservesMainState() const override
    {
        return supports_mtp_sidecar_preserves_main_state_;
    }

    bool supportsMTPShiftedRowReuseFromSidecar() const override
    {
        return supports_mtp_shifted_row_reuse_from_sidecar_;
    }

    bool applyPenaltiesOnDevice(
        const std::vector<LogitPenalty> &penalties,
        int vocab_size) override
    {
        ++apply_penalties_on_device_calls_;
        last_penalty_count_ = penalties.size();
        last_penalty_vocab_size_ = vocab_size;
        return apply_penalties_on_device_ok_;
    }

    bool applyPenaltiesToMTPLogitsOnDevice(
        const std::vector<LogitPenalty> &penalties,
        int vocab_size) override
    {
        ++apply_penalties_to_mtp_logits_calls_;
        last_penalty_count_ = penalties.size();
        last_penalty_vocab_size_ = vocab_size;
        return apply_penalties_to_mtp_logits_ok_;
    }

    bool applyPenaltiesToAllPositionLogitsOnDeviceRow(
        int row,
        const std::vector<LogitPenalty> &penalties,
        int vocab_size) override
    {
        ++apply_penalties_to_all_position_row_calls_;
        last_all_position_penalty_row_ = row;
        last_penalty_count_ = penalties.size();
        last_penalty_vocab_size_ = vocab_size;
        return apply_penalties_to_all_position_row_ok_;
    }

    bool verifyGreedyAllPositionBatchOutcomeOnDevice(
        const int32_t *draft_tokens,
        int draft_token_count,
        const int32_t *stop_tokens,
        int stop_token_count,
        DeviceSpeculativeVerifyBatchOutcome *out) override
    {
        (void)draft_tokens;
        (void)stop_tokens;
        ++verify_greedy_all_position_batch_outcome_calls_;
        last_stochastic_row_count_ = draft_token_count;
        last_stop_token_count_ = stop_token_count;
        if (out)
        {
            *out = DeviceSpeculativeVerifyBatchOutcome{};
            out->accepted_speculative_prefix = draft_token_count;
            out->consumed_verifier_rows = draft_token_count;
        }
        return verify_greedy_all_position_batch_outcome_ok_;
    }

    bool supportsGreedyAllPositionBatchOutcomeOnDevice() const override
    {
        return supports_greedy_all_position_batch_outcome_;
    }

    bool supportsDeviceStochasticMTPVerification() const override
    {
        return supports_device_stochastic_mtp_verification_;
    }

    bool buildStochasticDistributionOnDevice(
        DeviceLogitsSource source,
        int row,
        DeviceDistributionBuffer buffer,
        int slot,
        const SamplingParams &params,
        int vocab_size) override
    {
        (void)source;
        (void)buffer;
        (void)params;
        ++build_stochastic_distribution_calls_;
        last_stochastic_row_ = row;
        last_stochastic_slot_ = slot;
        last_stochastic_vocab_size_ = vocab_size;
        return stochastic_device_ops_ok_;
    }

    bool buildStochasticDistributionsOnDevice(
        DeviceLogitsSource source,
        int first_row,
        DeviceDistributionBuffer buffer,
        int first_slot,
        int row_count,
        const SamplingParams &params,
        int vocab_size) override
    {
        (void)source;
        (void)buffer;
        (void)params;
        ++build_stochastic_distributions_calls_;
        last_stochastic_row_ = first_row;
        last_stochastic_slot_ = first_slot;
        last_stochastic_row_count_ = row_count;
        last_stochastic_vocab_size_ = vocab_size;
        return stochastic_device_ops_ok_;
    }

    bool buildStochasticProcessedLogitRowsOnDevice(
        DeviceLogitsSource source,
        int first_row,
        DeviceDistributionBuffer buffer,
        int first_slot,
        int row_count,
        const SamplingParams &params,
        int vocab_size) override
    {
        (void)source;
        (void)buffer;
        (void)params;
        ++build_stochastic_processed_rows_calls_;
        last_stochastic_row_ = first_row;
        last_stochastic_slot_ = first_slot;
        last_stochastic_row_count_ = row_count;
        last_stochastic_vocab_size_ = vocab_size;
        return stochastic_device_ops_ok_;
    }

    int sampleStochasticDraftProposalOnDevice(
        DeviceLogitsSource source,
        int row,
        int slot,
        const SamplingParams &params,
        int vocab_size,
        float threshold) override
    {
        (void)source;
        (void)params;
        ++sample_stochastic_draft_proposal_calls_;
        last_stochastic_row_ = row;
        last_stochastic_slot_ = slot;
        last_stochastic_vocab_size_ = vocab_size;
        last_stochastic_threshold_ = threshold;
        return stochastic_sample_token_;
    }

    bool sampleStochasticDraftProposalOnDeviceDeferred(
        DeviceLogitsSource source,
        int row,
        int slot,
        const SamplingParams &params,
        int vocab_size,
        float threshold) override
    {
        return sampleStochasticDraftProposalOnDevice(
                   source,
                   row,
                   slot,
                   params,
                   vocab_size,
                   threshold) >= 0;
    }

    int sampleStochasticDistributionOnDevice(
        DeviceDistributionBuffer buffer,
        int slot,
        float threshold) override
    {
        (void)buffer;
        ++sample_stochastic_distribution_calls_;
        last_stochastic_slot_ = slot;
        last_stochastic_threshold_ = threshold;
        return stochastic_sample_token_;
    }

    bool sampleStochasticDistributionOnDeviceDeferred(
        DeviceDistributionBuffer buffer,
        int slot,
        float threshold) override
    {
        return sampleStochasticDistributionOnDevice(buffer, slot, threshold) >= 0;
    }

    const void *prepareMTPVerifierInputTokensOnDevice(
        int32_t first_token,
        int first_draft_slot,
        int draft_token_count,
        int total_verifier_input_tokens) override
    {
        ++prepare_mtp_verifier_input_tokens_calls_;
        last_verifier_first_token_ = first_token;
        last_verifier_first_draft_slot_ = first_draft_slot;
        last_verifier_draft_token_count_ = draft_token_count;
        last_verifier_total_input_tokens_ = total_verifier_input_tokens;
        if (!stochastic_device_ops_ok_)
            return nullptr;

        verifier_device_tokens_[0] = first_token;
        return verifier_device_tokens_.data();
    }

    const void *prepareMTPVerifierInputTokensOnDeviceFromDeviceFirstToken(
        int first_target_sample_slot,
        int first_draft_slot,
        int draft_token_count,
        int total_verifier_input_tokens) override
    {
        ++prepare_mtp_verifier_input_tokens_from_device_calls_;
        last_verifier_first_target_sample_slot_ = first_target_sample_slot;
        last_verifier_first_draft_slot_ = first_draft_slot;
        last_verifier_draft_token_count_ = draft_token_count;
        last_verifier_total_input_tokens_ = total_verifier_input_tokens;
        if (!stochastic_device_ops_ok_)
            return nullptr;

        return verifier_device_tokens_from_device_first_.data();
    }

    bool stageStochasticDraftTokensForDeviceVerification(
        const int32_t *draft_tokens,
        int draft_token_count,
        int first_draft_slot = 0) override
    {
        ++stage_stochastic_draft_tokens_calls_;
        last_staged_first_draft_slot_ = first_draft_slot;
        last_staged_draft_tokens_.clear();
        if (!stochastic_device_ops_ok_ ||
            !draft_tokens ||
            draft_token_count <= 0 ||
            first_draft_slot < 0)
        {
            return false;
        }
        last_staged_draft_tokens_.assign(
            draft_tokens,
            draft_tokens + draft_token_count);
        return true;
    }

    bool verifyStochasticDistributionsBatchOutcomeOnDevice(
        int first_target_slot,
        int first_draft_slot,
        const int32_t *draft_tokens,
        const float *accept_thresholds,
        const float *residual_thresholds,
        int row_count,
        int32_t first_token,
        const int32_t *stop_tokens,
        int stop_token_count,
        int bonus_target_slot,
        float bonus_threshold,
        DeviceSpeculativeVerifyBatchOutcome *out,
        uint64_t inverse_sample_seed = 0,
        int inverse_sample_first_logical_position = 0,
        bool use_vllm_probability_rejection = false) override
    {
        (void)first_target_slot;
        (void)first_draft_slot;
        (void)draft_tokens;
        (void)accept_thresholds;
        (void)residual_thresholds;
        (void)first_token;
        (void)stop_tokens;
        (void)bonus_target_slot;
        (void)bonus_threshold;
        (void)inverse_sample_seed;
        (void)inverse_sample_first_logical_position;
        ++verify_stochastic_batch_outcome_calls_;
        last_stochastic_row_count_ = row_count;
        last_stop_token_count_ = stop_token_count;
        last_use_vllm_probability_rejection_ = use_vllm_probability_rejection;
        if (out)
        {
            *out = DeviceSpeculativeVerifyBatchOutcome{};
            out->accepted_speculative_prefix = row_count;
            out->consumed_verifier_rows = row_count;
        }
        return stochastic_device_ops_ok_;
    }

    void clear_cache() override
    {
        clear_cache_calls_.fetch_add(1, std::memory_order_relaxed);
        position_ = 0;
    }

    int get_position() const override
    {
        return position_;
    }

    ExecutionPath executionPath() const override
    {
        return ExecutionPath::GRAPH;
    }

    const char *architecture() const override
    {
        return config_.architecture.c_str();
    }

    uint64_t moePlacementEpoch() const override
    {
        return moe_placement_epoch_;
    }

    std::vector<MoERebalanceController *> moeRebalanceControllers() const override
    {
        if (!moe_rebalance_controller_)
            return {};
        return {moe_rebalance_controller_.get()};
    }

    MoERebalanceController *moeRebalanceControllerForDomain(
        const std::string &domain_id) const override
    {
        if (!moe_rebalance_controller_)
            return nullptr;
        return moe_rebalance_controller_->domainId() == domain_id
                   ? moe_rebalance_controller_.get()
                   : nullptr;
    }

    PrefixLookupResult lookupPrefix(const std::vector<int32_t> &tokens) override
    {
        ++prefix_lookup_calls_;
        prefix_lookup_tokens_ = tokens;
        return prefix_lookup_result_;
    }

    bool populatePrefix(const PrefixLookupResult &hit, int seq_idx = 0) override
    {
        (void)seq_idx;
        ++prefix_populate_calls_;
        populated_prefix_tokens_.push_back(hit.cached_tokens);
        return prefix_populate_ok_;
    }

    bool harvestPrefix(const std::vector<int32_t> &tokens, int prompt_token_count) override
    {
        ++prefix_harvest_calls_;
        harvested_prefix_tokens_ = tokens;
        harvested_prompt_token_count_ = prompt_token_count;
        return prefix_harvest_ok_;
    }

    bool restorePrefixTerminalState(const PrefixLookupResult &hit) override
    {
        ++prefix_terminal_restore_calls_;
        terminal_restored_tokens_.push_back(hit.cached_tokens);
        if (prefix_terminal_restore_requires_blocks_ && hit.blocks.empty())
            return false;
        return prefix_terminal_restore_ok_;
    }

    PrefixStateSnapshot captureLivePrefixState(int seq_idx = 0) const override
    {
        (void)seq_idx;
        prefix_live_capture_calls_.fetch_add(1, std::memory_order_relaxed);
        PrefixStateSnapshot snapshot;
        if (!prefix_live_capture_ok_)
            return snapshot;
        snapshot.valid = true;
        snapshot.cached_tokens = position_;
        return snapshot;
    }

    PrefixStateSnapshot captureLivePrefixCheckpoint(int seq_idx = 0) const override
    {
        (void)seq_idx;
        prefix_live_capture_calls_.fetch_add(1, std::memory_order_relaxed);
        PrefixStateSnapshot snapshot;
        if (!prefix_live_capture_ok_)
            return snapshot;
        snapshot.valid = true;
        snapshot.logical_checkpoint = true;
        snapshot.cached_tokens = position_;
        return snapshot;
    }

    bool restoreLivePrefixState(const PrefixStateSnapshot &snapshot, int seq_idx = 0) override
    {
        (void)seq_idx;
        prefix_live_restore_calls_.fetch_add(1, std::memory_order_relaxed);
        if (!prefix_live_restore_ok_ || !snapshot.valid)
            return false;
        position_ = snapshot.cached_tokens;
        return true;
    }

    bool truncateLivePrefixState(int cached_tokens, int seq_idx = 0) override
    {
        (void)seq_idx;
        prefix_live_truncate_calls_.fetch_add(1, std::memory_order_relaxed);
        if (!prefix_live_truncate_ok_ || cached_tokens < 0)
            return false;
        position_ = cached_tokens;
        return true;
    }

    // =====================================================================
    // Test Utilities
    // =====================================================================

    size_t forward_call_count() const
    {
        return forward_calls_.load(std::memory_order_relaxed);
    }

    size_t clear_cache_call_count() const
    {
        return clear_cache_calls_.load(std::memory_order_relaxed);
    }

    void set_forward_fails(bool fails) { config_.forward_should_fail = fails; }
    void set_forward_sleep_ms(int ms) { config_.forward_sleep_ms = ms; }

    void set_mock_logits(const std::vector<float> &logits)
    {
        logits_ = logits;
        config_.vocab_size = static_cast<int>(logits.size());
    }

    void set_mock_mtp_logits(const std::vector<float> &logits)
    {
        mtp_logits_ = logits;
    }

    void set_mock_mtp_logits_local(int local_vocab, const std::vector<float> &logits)
    {
        mtp_logits_local_ = std::make_shared<FP32Tensor>(
            std::vector<size_t>{1, static_cast<size_t>(local_vocab)},
            DeviceId::cpu());
        std::memcpy(mtp_logits_local_->mutable_data(), logits.data(),
                    logits.size() * sizeof(float));
    }

    void set_mock_all_position_logits(const std::vector<float> &logits)
    {
        all_position_logits_ = logits;
    }

    void set_mock_all_position_logits_local(int rows, int local_vocab, const std::vector<float> &logits)
    {
        all_position_logits_local_ = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(local_vocab)},
            DeviceId::cpu());
        std::memcpy(all_position_logits_local_->mutable_data(), logits.data(),
                    logits.size() * sizeof(float));
    }

    void set_vocab_size(int size)
    {
        config_.vocab_size = size;
        logits_.resize(static_cast<size_t>(size), 0.0f);
    }

    void set_prefix_lookup_result(PrefixLookupResult result)
    {
        prefix_lookup_result_ = std::move(result);
    }

    void set_prefix_populate_ok(bool ok) { prefix_populate_ok_ = ok; }
    void set_prefix_terminal_restore_requires_blocks(bool required) { prefix_terminal_restore_requires_blocks_ = required; }
    void set_forward_mtp_ok(bool ok) { forward_mtp_ok_ = ok; }
    void set_supports_chained_mtp_drafts(bool supported) { supports_chained_mtp_drafts_ = supported; }
    void set_forward_mtp_from_last_draft_ok(bool ok) { forward_mtp_from_last_draft_ok_ = ok; }
    void set_commit_mtp_shifted_rows_ok(bool ok) { commit_mtp_shifted_rows_ok_ = ok; }
    void set_ensure_mtp_checkpoint_terminal_hidden_ok(bool ok)
    {
        ensure_mtp_checkpoint_terminal_hidden_ok_ = ok;
    }
    void set_supports_mtp_spec_state_publication(bool supported) { supports_mtp_spec_state_publication_ = supported; }
    void set_mtp_verifier_row_capability(MTPVerifierRowCapability capability)
    {
        mtp_verifier_row_capability_ = capability;
    }

    void set_mtp_verifier_economy_capability(MTPVerifierEconomyCapability capability)
    {
        mtp_verifier_economy_capability_ = capability;
    }
    void set_publish_mtp_spec_state_ok(bool ok) { publish_mtp_spec_state_ok_ = ok; }
    void set_all_position_logits_ok(bool ok) { set_all_position_logits_ok_ = ok; }
    void set_mtp_unsupported_reason(std::string reason) { mtp_unsupported_reason_ = std::move(reason); }
    void set_primary_device_id(DeviceId device_id) { device_id_ = device_id; }
    void set_supports_mtp_sidecar_preserves_main_state(bool supported) { supports_mtp_sidecar_preserves_main_state_ = supported; }
    void set_supports_mtp_shifted_row_reuse_from_sidecar(bool supported) { supports_mtp_shifted_row_reuse_from_sidecar_ = supported; }
    void set_supports_device_stochastic_mtp_verification(bool supported) { supports_device_stochastic_mtp_verification_ = supported; }
    void set_stochastic_sample_token(int token) { stochastic_sample_token_ = token; }
    void set_prefix_live_capture_ok(bool ok) { prefix_live_capture_ok_ = ok; }
    void set_prefix_live_restore_ok(bool ok) { prefix_live_restore_ok_ = ok; }
    void set_prefix_live_truncate_ok(bool ok) { prefix_live_truncate_ok_ = ok; }
    void set_moe_placement_epoch(uint64_t epoch) { moe_placement_epoch_ = epoch; }
    void set_moe_rebalance_controller(std::unique_ptr<MoERebalanceController> controller)
    {
        moe_rebalance_controller_ = std::move(controller);
    }
    void set_forward_mtp_rendezvous(std::shared_ptr<ForwardMTPRendezvous> rendezvous)
    {
        forward_mtp_rendezvous_ = std::move(rendezvous);
    }
    void set_mtp_publication_rendezvous(std::shared_ptr<MTPPublicationRendezvous> rendezvous)
    {
        mtp_publication_rendezvous_ = std::move(rendezvous);
    }
    void set_chained_mtp_rendezvous(std::shared_ptr<ChainedMTPRendezvous> rendezvous)
    {
        chained_mtp_rendezvous_ = std::move(rendezvous);
    }

    size_t prefix_lookup_call_count() const { return prefix_lookup_calls_; }
    size_t prefix_populate_call_count() const { return prefix_populate_calls_; }
    size_t prefix_harvest_call_count() const { return prefix_harvest_calls_; }
    size_t prefix_terminal_restore_call_count() const { return prefix_terminal_restore_calls_; }
    const std::vector<int> &populated_prefix_tokens() const { return populated_prefix_tokens_; }
    const std::vector<int> &terminal_restored_tokens() const { return terminal_restored_tokens_; }
    const std::vector<int32_t> &prefix_lookup_tokens() const { return prefix_lookup_tokens_; }
    const std::vector<int32_t> &harvested_prefix_tokens() const { return harvested_prefix_tokens_; }
    int harvested_prompt_token_count() const { return harvested_prompt_token_count_; }
    size_t forward_mtp_call_count() const { return forward_mtp_calls_.load(std::memory_order_relaxed); }
    size_t forward_mtp_from_last_draft_call_count() const { return forward_mtp_from_last_draft_calls_.load(std::memory_order_relaxed); }
    size_t sample_mtp_logits_call_count() const { return sample_mtp_logits_calls_; }
    size_t get_mtp_logits_local_info_call_count() const { return get_mtp_logits_local_info_calls_.load(std::memory_order_relaxed); }
    size_t consume_mtp_logits_local_info_call_count() const { return consume_mtp_logits_local_info_calls_.load(std::memory_order_relaxed); }
    size_t commit_mtp_shifted_rows_call_count() const { return commit_mtp_shifted_rows_calls_; }
    size_t ensure_mtp_checkpoint_terminal_hidden_call_count() const
    {
        return ensure_mtp_checkpoint_terminal_hidden_calls_;
    }
    size_t publish_mtp_spec_state_call_count() const { return publish_mtp_spec_state_calls_.load(std::memory_order_relaxed); }
    size_t publish_mtp_spec_state_batch_call_count() const { return publish_mtp_spec_state_batch_calls_.load(std::memory_order_relaxed); }
    int32_t last_mtp_condition_token() const { return last_mtp_condition_token_; }
    int32_t last_chained_mtp_condition_token() const { return last_chained_mtp_condition_token_; }
    int last_chained_mtp_position_id() const { return last_chained_mtp_position_id_; }
    int last_commit_mtp_already_appended() const { return last_commit_mtp_already_appended_; }
    int last_commit_mtp_main_forward_token_count() const { return last_commit_mtp_main_forward_token_count_; }
    bool last_commit_mtp_allow_speculative_discard() const { return last_commit_mtp_allow_speculative_discard_; }
    int last_commit_mtp_position_offset_override() const { return last_commit_mtp_position_offset_override_; }
    const std::vector<int32_t> &last_commit_mtp_tokens() const { return last_commit_mtp_tokens_; }
    const MTPSpecStepPlan &last_mtp_spec_state_plan() const { return last_mtp_spec_state_plan_; }
    const MTPSpecStepPlanBatch &last_mtp_spec_state_batch() const { return last_mtp_spec_state_batch_; }
    size_t set_all_position_logits_call_count() const { return set_all_position_logits_calls_.load(std::memory_order_relaxed); }
    size_t set_row_indexed_all_position_logits_call_count() const { return set_row_indexed_all_position_logits_calls_.load(std::memory_order_relaxed); }
    size_t set_mtp_spec_verifier_input_plan_call_count() const { return set_mtp_spec_verifier_input_plan_calls_.load(std::memory_order_relaxed); }
    size_t clear_mtp_spec_verifier_input_plan_call_count() const { return clear_mtp_spec_verifier_input_plan_calls_.load(std::memory_order_relaxed); }
    size_t apply_penalties_on_device_call_count() const { return apply_penalties_on_device_calls_; }
    size_t apply_penalties_to_mtp_logits_call_count() const { return apply_penalties_to_mtp_logits_calls_; }
    size_t get_all_position_logits_local_info_call_count() const { return get_all_position_logits_local_info_calls_.load(std::memory_order_relaxed); }
    size_t consume_all_position_logits_local_info_call_count() const { return consume_all_position_logits_local_info_calls_.load(std::memory_order_relaxed); }
    bool compute_all_position_logits() const { return compute_all_position_logits_; }
    bool compute_row_indexed_all_position_logits() const { return compute_row_indexed_all_position_logits_; }
    int row_indexed_all_position_logit_rows() const { return row_indexed_all_position_logit_rows_; }
    size_t apply_penalties_to_all_position_row_call_count() const { return apply_penalties_to_all_position_row_calls_; }
    size_t build_stochastic_processed_rows_call_count() const { return build_stochastic_processed_rows_calls_; }
    size_t sample_stochastic_draft_proposal_call_count() const { return sample_stochastic_draft_proposal_calls_; }
    size_t sample_stochastic_distribution_call_count() const { return sample_stochastic_distribution_calls_; }
    size_t prepare_mtp_verifier_input_tokens_call_count() const
    {
        return prepare_mtp_verifier_input_tokens_calls_;
    }
    size_t prepare_mtp_verifier_input_tokens_from_device_call_count() const
    {
        return prepare_mtp_verifier_input_tokens_from_device_calls_;
    }
    size_t stage_stochastic_draft_tokens_call_count() const
    {
        return stage_stochastic_draft_tokens_calls_;
    }
    size_t verify_stochastic_batch_outcome_call_count() const { return verify_stochastic_batch_outcome_calls_; }
    size_t verify_greedy_all_position_batch_outcome_call_count() const { return verify_greedy_all_position_batch_outcome_calls_; }
    int32_t last_verifier_first_token() const { return last_verifier_first_token_; }
    int last_verifier_first_target_sample_slot() const { return last_verifier_first_target_sample_slot_; }
    int last_verifier_first_draft_slot() const { return last_verifier_first_draft_slot_; }
    int last_verifier_draft_token_count() const { return last_verifier_draft_token_count_; }
    int last_verifier_total_input_tokens() const { return last_verifier_total_input_tokens_; }
    int last_staged_first_draft_slot() const { return last_staged_first_draft_slot_; }
    const std::vector<int32_t> &last_staged_draft_tokens() const
    {
        return last_staged_draft_tokens_;
    }
    int last_stochastic_row_count() const { return last_stochastic_row_count_; }
    bool last_use_vllm_probability_rejection() const { return last_use_vllm_probability_rejection_; }
    size_t prefix_live_capture_call_count() const { return prefix_live_capture_calls_.load(std::memory_order_relaxed); }
    size_t prefix_live_restore_call_count() const { return prefix_live_restore_calls_.load(std::memory_order_relaxed); }
    size_t prefix_live_truncate_call_count() const { return prefix_live_truncate_calls_.load(std::memory_order_relaxed); }

    void reset_call_counts()
    {
        forward_calls_.store(0, std::memory_order_relaxed);
        clear_cache_calls_.store(0, std::memory_order_relaxed);
        forward_mtp_calls_.store(0, std::memory_order_relaxed);
        forward_mtp_from_last_draft_calls_.store(0, std::memory_order_relaxed);
        sample_mtp_logits_calls_ = 0;
        commit_mtp_shifted_rows_calls_ = 0;
        ensure_mtp_checkpoint_terminal_hidden_calls_ = 0;
        publish_mtp_spec_state_calls_.store(0, std::memory_order_relaxed);
        set_all_position_logits_calls_.store(0, std::memory_order_relaxed);
        set_row_indexed_all_position_logits_calls_.store(0, std::memory_order_relaxed);
        set_mtp_spec_verifier_input_plan_calls_.store(0, std::memory_order_relaxed);
        clear_mtp_spec_verifier_input_plan_calls_.store(0, std::memory_order_relaxed);
        get_mtp_logits_local_info_calls_.store(0, std::memory_order_relaxed);
        consume_mtp_logits_local_info_calls_.store(0, std::memory_order_relaxed);
        get_all_position_logits_local_info_calls_.store(0, std::memory_order_relaxed);
        consume_all_position_logits_local_info_calls_.store(0, std::memory_order_relaxed);
        prefix_live_capture_calls_.store(0, std::memory_order_relaxed);
        prefix_live_restore_calls_.store(0, std::memory_order_relaxed);
        prefix_live_truncate_calls_.store(0, std::memory_order_relaxed);
        prepare_mtp_verifier_input_tokens_calls_ = 0;
        prepare_mtp_verifier_input_tokens_from_device_calls_ = 0;
        stage_stochastic_draft_tokens_calls_ = 0;
        last_staged_draft_tokens_.clear();
    }

private:
    Config config_;
    int position_;
    std::vector<float> logits_;
    std::vector<float> mtp_logits_;
    std::vector<float> all_position_logits_;
    std::shared_ptr<FP32Tensor> mtp_logits_local_;
    std::shared_ptr<FP32Tensor> all_position_logits_local_;
    std::shared_ptr<ForwardMTPRendezvous> forward_mtp_rendezvous_;
    std::shared_ptr<MTPPublicationRendezvous> mtp_publication_rendezvous_;
    std::shared_ptr<ChainedMTPRendezvous> chained_mtp_rendezvous_;
    PrefixLookupResult prefix_lookup_result_;
    DeviceId device_id_ = DeviceId::cpu();
    bool prefix_populate_ok_ = true;
    bool prefix_harvest_ok_ = true;
    bool prefix_terminal_restore_ok_ = true;
    bool prefix_terminal_restore_requires_blocks_ = false;
    bool forward_mtp_ok_ = true;
    bool supports_chained_mtp_drafts_ = false;
    bool forward_mtp_from_last_draft_ok_ = true;
    bool commit_mtp_shifted_rows_ok_ = true;
    bool ensure_mtp_checkpoint_terminal_hidden_ok_ = true;
    bool supports_mtp_spec_state_publication_ = false;
    MTPVerifierRowCapability mtp_verifier_row_capability_;
    MTPVerifierEconomyCapability mtp_verifier_economy_capability_;
    bool supports_greedy_all_position_batch_outcome_ = true;
    bool publish_mtp_spec_state_ok_ = true;
    bool set_all_position_logits_ok_ = true;
    bool set_row_indexed_all_position_logits_ok_ = true;
    bool set_mtp_spec_verifier_input_plan_ok_ = true;
    bool compute_all_position_logits_ = false;
    bool compute_row_indexed_all_position_logits_ = false;
    int row_indexed_all_position_logit_rows_ = 0;
    std::string mtp_unsupported_reason_;
    std::unique_ptr<MoERebalanceController> moe_rebalance_controller_;
    bool supports_mtp_sidecar_logits_stream_handoff_ = false;
    bool supports_mtp_device_draft_token_input_ = false;
    bool supports_mtp_sidecar_preserves_main_state_ = false;
    bool supports_mtp_shifted_row_reuse_from_sidecar_ = false;
    bool supports_device_stochastic_mtp_verification_ = false;
    bool apply_penalties_on_device_ok_ = true;
    bool apply_penalties_to_mtp_logits_ok_ = true;
    bool apply_penalties_to_all_position_row_ok_ = true;
    bool verify_greedy_all_position_batch_outcome_ok_ = true;
    bool stochastic_device_ops_ok_ = true;
    int stochastic_sample_token_ = 17;
    bool prefix_live_capture_ok_ = true;
    bool prefix_live_restore_ok_ = true;
    bool prefix_live_truncate_ok_ = true;
    uint64_t moe_placement_epoch_ = 0;
    int32_t last_mtp_condition_token_ = -1;
    int32_t last_chained_mtp_condition_token_ = -1;
    int last_chained_mtp_position_id_ = -1;
    int last_commit_mtp_already_appended_ = 0;
    int last_commit_mtp_main_forward_token_count_ = 0;
    int last_commit_mtp_position_offset_override_ = -1;
    size_t ensure_mtp_checkpoint_terminal_hidden_calls_ = 0;
    bool last_commit_mtp_allow_speculative_discard_ = false;
    MTPSpecStepPlan last_mtp_spec_state_plan_;
    MTPSpecStepPlanBatch last_mtp_spec_state_batch_;
    MTPSpecDecodeVerifierInputPlan last_mtp_spec_verifier_input_plan_;
    size_t sample_mtp_logits_calls_ = 0;
    size_t commit_mtp_shifted_rows_calls_ = 0;
    size_t apply_penalties_on_device_calls_ = 0;
    size_t apply_penalties_to_mtp_logits_calls_ = 0;
    size_t apply_penalties_to_all_position_row_calls_ = 0;
    size_t build_stochastic_distribution_calls_ = 0;
    size_t build_stochastic_distributions_calls_ = 0;
    size_t build_stochastic_processed_rows_calls_ = 0;
    size_t sample_stochastic_draft_proposal_calls_ = 0;
    size_t sample_stochastic_distribution_calls_ = 0;
    size_t prepare_mtp_verifier_input_tokens_calls_ = 0;
    size_t prepare_mtp_verifier_input_tokens_from_device_calls_ = 0;
    size_t stage_stochastic_draft_tokens_calls_ = 0;
    size_t verify_stochastic_batch_outcome_calls_ = 0;
    size_t verify_greedy_all_position_batch_outcome_calls_ = 0;
    size_t last_penalty_count_ = 0;
    int last_penalty_vocab_size_ = 0;
    int last_all_position_penalty_row_ = -1;
    int last_stochastic_row_ = -1;
    int last_stochastic_slot_ = -1;
    int last_stochastic_row_count_ = 0;
    int last_stochastic_vocab_size_ = 0;
    int last_stop_token_count_ = 0;
    int32_t last_verifier_first_token_ = -1;
    int last_verifier_first_target_sample_slot_ = -1;
    int last_verifier_first_draft_slot_ = -1;
    int last_verifier_draft_token_count_ = 0;
    int last_verifier_total_input_tokens_ = 0;
    int last_staged_first_draft_slot_ = -1;
    float last_stochastic_threshold_ = 0.0f;
    bool last_use_vllm_probability_rejection_ = false;
    std::array<int32_t, 8> verifier_device_tokens_{};
    std::array<int32_t, 8> verifier_device_tokens_from_device_first_{};
    std::vector<int32_t> last_staged_draft_tokens_;
    size_t prefix_lookup_calls_ = 0;
    size_t prefix_populate_calls_ = 0;
    size_t prefix_harvest_calls_ = 0;
    size_t prefix_terminal_restore_calls_ = 0;
    int harvested_prompt_token_count_ = 0;
    std::vector<int> populated_prefix_tokens_;
    std::vector<int> terminal_restored_tokens_;
    std::vector<int32_t> prefix_lookup_tokens_;
    std::vector<int32_t> harvested_prefix_tokens_;
    std::vector<int32_t> last_commit_mtp_tokens_;
    mutable std::atomic<size_t> forward_calls_{0};
    mutable std::atomic<size_t> clear_cache_calls_{0};
    mutable std::atomic<size_t> forward_mtp_calls_{0};
    mutable std::atomic<size_t> forward_mtp_from_last_draft_calls_{0};
    mutable std::atomic<size_t> publish_mtp_spec_state_calls_{0};
    mutable std::atomic<size_t> publish_mtp_spec_state_batch_calls_{0};
    mutable std::atomic<size_t> set_all_position_logits_calls_{0};
    mutable std::atomic<size_t> set_row_indexed_all_position_logits_calls_{0};
    mutable std::atomic<size_t> set_mtp_spec_verifier_input_plan_calls_{0};
    mutable std::atomic<size_t> clear_mtp_spec_verifier_input_plan_calls_{0};
    mutable std::atomic<size_t> get_mtp_logits_local_info_calls_{0};
    mutable std::atomic<size_t> consume_mtp_logits_local_info_calls_{0};
    mutable std::atomic<size_t> get_all_position_logits_local_info_calls_{0};
    mutable std::atomic<size_t> consume_all_position_logits_local_info_calls_{0};
    mutable std::atomic<size_t> prefix_live_capture_calls_{0};
    mutable std::atomic<size_t> prefix_live_restore_calls_{0};
    mutable std::atomic<size_t> prefix_live_truncate_calls_{0};
};

// =============================================================================
// MockLocalTPContext - Mock for LOCAL TP context
// =============================================================================

/**
 * @brief Mock LOCAL TP context for testing collective operations
 *
 * Tracks synchronization calls and provides configurable devices/weights.
 */
class MockLocalTPContext : public ILocalTPContext
{
public:
    struct Config
    {
        std::vector<GlobalDeviceAddress> devices;
        std::vector<float> weights;
        CollectiveBackendType backend = CollectiveBackendType::HOST;
        bool allreduce_should_fail = false;
        bool allgather_should_fail = false;
    };

    MockLocalTPContext() : MockLocalTPContext(Config{}) {}

    explicit MockLocalTPContext(const Config &config)
        : config_(config)
    {
        // Default to 2 CPU devices if none specified
        if (config_.devices.empty())
        {
            config_.devices.push_back(GlobalDeviceAddress::cpu());
            config_.devices.push_back(GlobalDeviceAddress::cpu());
        }
        // Default to equal weights
        if (config_.weights.empty())
        {
            float equal = 1.0f / static_cast<float>(config_.devices.size());
            config_.weights.resize(config_.devices.size(), equal);
        }
    }

    // =====================================================================
    // ILocalTPContext Configuration API
    // =====================================================================

    const std::vector<GlobalDeviceAddress> &devices() const override
    {
        return config_.devices;
    }

    const std::vector<float> &weights() const override
    {
        return config_.weights;
    }

    CollectiveBackendType backend() const override
    {
        return config_.backend;
    }

    int degree() const override
    {
        return static_cast<int>(config_.devices.size());
    }

    int myIndex() const override { return 0; }

    // =====================================================================
    // ILocalTPContext Collective Operations
    // =====================================================================

    bool allreduce(TensorBase * /*tensor*/) override
    {
        allreduce_calls_.fetch_add(1, std::memory_order_relaxed);
        return !config_.allreduce_should_fail;
    }

    bool allreduce(TensorBase *tensor, const std::string & /*stage_name*/, size_t /*count*/ = 0) override
    {
        return allreduce(tensor);
    }

    bool allreduce(const TensorBase * /*input*/, TensorBase * /*output*/) override
    {
        allreduce_calls_.fetch_add(1, std::memory_order_relaxed);
        return !config_.allreduce_should_fail;
    }

    bool allgather(const TensorBase * /*local_shard*/, TensorBase * /*global_tensor*/) override
    {
        allgather_calls_.fetch_add(1, std::memory_order_relaxed);
        return !config_.allgather_should_fail;
    }

    bool gatherFromDevices(
        const std::vector<const TensorBase *> &shards,
        TensorBase *output) override
    {
        gather_from_devices_calls_.fetch_add(1, std::memory_order_relaxed);

        // Simple mock implementation: copy data from shards to output
        if (shards.empty() || !output)
        {
            return false;
        }

        float *dst = output->mutable_data();
        size_t offset = 0;
        for (const auto *shard : shards)
        {
            if (shard)
            {
                const float *src = shard->data();
                size_t count = shard->numel();
                std::memcpy(dst + offset, src, count * sizeof(float));
                offset += count;
            }
        }
        return !config_.allgather_should_fail;
    }

    bool reduceScatter(const TensorBase * /*input*/, TensorBase * /*output_shard*/) override
    {
        reduce_scatter_calls_.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    // =====================================================================
    // ILocalTPContext Synchronization
    // =====================================================================

    void synchronize() override
    {
        synchronize_calls_.fetch_add(1, std::memory_order_relaxed);
    }

    // =====================================================================
    // ILocalTPContext Device Management
    // =====================================================================

    int indexForDevice(const GlobalDeviceAddress &device) const override
    {
        for (size_t i = 0; i < config_.devices.size(); ++i)
        {
            if (config_.devices[i] == device)
            {
                return static_cast<int>(i);
            }
        }
        return -1;
    }

    const GlobalDeviceAddress &deviceAt(int index) const override
    {
        if (index < 0 || index >= static_cast<int>(config_.devices.size()))
        {
            throw std::out_of_range("MockLocalTPContext::deviceAt: index out of range");
        }
        return config_.devices[static_cast<size_t>(index)];
    }

    float weightForDevice(const GlobalDeviceAddress &device) const override
    {
        int idx = indexForDevice(device);
        return (idx >= 0) ? config_.weights[static_cast<size_t>(idx)] : 0.0f;
    }

    // =====================================================================
    // ILocalTPContext Sharding Utilities
    // =====================================================================

    int headsForDevice(const GlobalDeviceAddress &device, int total_heads) const override
    {
        float w = weightForDevice(device);
        return static_cast<int>(w * static_cast<float>(total_heads) + 0.5f);
    }

    std::pair<int, int> rowRangeForDevice(
        const GlobalDeviceAddress &device, int total_rows) const override
    {
        int idx = indexForDevice(device);
        if (idx < 0)
            return {0, 0};

        float cumulative = 0.0f;
        for (int i = 0; i < idx; ++i)
        {
            cumulative += config_.weights[static_cast<size_t>(i)];
        }
        int start = static_cast<int>(cumulative * static_cast<float>(total_rows));
        int end = static_cast<int>((cumulative + config_.weights[static_cast<size_t>(idx)]) * static_cast<float>(total_rows));
        return {start, end};
    }

    std::pair<int, int> colRangeForDevice(
        const GlobalDeviceAddress &device, int total_cols) const override
    {
        return rowRangeForDevice(device, total_cols);
    }

    // =====================================================================
    // ILocalTPContext BAR Registry (no-ops for tests)
    // =====================================================================

    void registerBARBackedOutput(
        const std::string & /*stage_name*/,
        const GlobalDeviceAddress & /*device*/,
        TensorBase * /*tensor*/) override
    {
        // No-op for unit tests
    }

    bool hasBARBackedOutputs(const std::string & /*stage_name*/) const override { return false; }
    void clearBARBackedOutputs() override {}
    bool reserveTempBufferBytes(size_t /*bytes*/) override { return true; }

    // =====================================================================
    // ILocalTPContext Broadcast (no-op)
    // =====================================================================
    bool broadcast(TensorBase * /*tensor*/, int /*source_device_index*/ = 0) override
    {
        broadcast_calls_.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    void requestAbort() override {}
    bool isAbortRequested() const override { return false; }

    // =====================================================================
    // Test Utilities
    // =====================================================================

    size_t synchronize_call_count() const
    {
        return synchronize_calls_.load(std::memory_order_relaxed);
    }

    size_t allreduce_call_count() const
    {
        return allreduce_calls_.load(std::memory_order_relaxed);
    }

    size_t allgather_call_count() const
    {
        return allgather_calls_.load(std::memory_order_relaxed);
    }

    size_t broadcast_call_count() const
    {
        return broadcast_calls_.load(std::memory_order_relaxed);
    }

    size_t gather_from_devices_call_count() const
    {
        return gather_from_devices_calls_.load(std::memory_order_relaxed);
    }

    void reset_call_counts()
    {
        synchronize_calls_.store(0, std::memory_order_relaxed);
        allreduce_calls_.store(0, std::memory_order_relaxed);
        allgather_calls_.store(0, std::memory_order_relaxed);
        gather_from_devices_calls_.store(0, std::memory_order_relaxed);
        reduce_scatter_calls_.store(0, std::memory_order_relaxed);
    }

    void set_allreduce_fails(bool fails) { config_.allreduce_should_fail = fails; }
    void set_allgather_fails(bool fails) { config_.allgather_should_fail = fails; }

private:
    Config config_;
    mutable std::atomic<size_t> synchronize_calls_{0};
    mutable std::atomic<size_t> allreduce_calls_{0};
    mutable std::atomic<size_t> allgather_calls_{0};
    mutable std::atomic<size_t> broadcast_calls_{0};
    mutable std::atomic<size_t> gather_from_devices_calls_{0};
    mutable std::atomic<size_t> reduce_scatter_calls_{0};
};

// =============================================================================
// Test Fixture
// =============================================================================

/**
 * @brief Test fixture for RankOrchestrator tests
 *
 * Provides helper methods to create mock orchestrators with different
 * configurations.
 */
class Test__RankOrchestrator : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Default setup: 2 device runners
        mock_runners_.clear();
        mock_runners_.push_back(std::make_unique<MockDeviceGraphOrchestrator>());
        mock_runners_.push_back(std::make_unique<MockDeviceGraphOrchestrator>());

        // Create mock TP context with 2 devices
        MockLocalTPContext::Config tp_config;
        tp_config.devices = {GlobalDeviceAddress::cpu(), GlobalDeviceAddress::cpu()};
        tp_config.weights = {0.5f, 0.5f};
        mock_tp_ctx_ = std::make_unique<MockLocalTPContext>(tp_config);
    }

    // Store raw pointers to mock runners for call verification
    std::vector<MockDeviceGraphOrchestrator *> getRawMockRunners()
    {
        std::vector<MockDeviceGraphOrchestrator *> result;
        for (auto &runner : mock_runners_)
        {
            result.push_back(runner.get());
        }
        return result;
    }

    std::vector<std::unique_ptr<MockDeviceGraphOrchestrator>> mock_runners_;
    std::unique_ptr<MockLocalTPContext> mock_tp_ctx_;
};

static PrefixLookupResult makePrefixHit(int cached_tokens,
                                        bool terminal_logits = false,
                                        bool supported = true,
                                        bool include_blocks = false,
                                        bool include_mtp_state = false,
                                        bool requires_terminal_logits = true,
                                        bool requires_terminal_hidden = true)
{
    PrefixLookupResult hit;
    hit.cache_enabled = true;
    hit.supported = supported;
    hit.block_size = 2;
    hit.cached_tokens = cached_tokens;
    hit.requires_terminal_hidden = requires_terminal_hidden;
    hit.requires_terminal_logits = requires_terminal_logits;
    hit.has_terminal_hidden = terminal_logits;
    hit.has_terminal_logits = terminal_logits;
    if (include_blocks)
    {
        for (int start = 0; start < cached_tokens; start += hit.block_size)
        {
            PrefixBlockHandle handle;
            handle.key.token_start = start;
            handle.key.token_count = std::min(hit.block_size, cached_tokens - start);
            const bool terminal_block = start + handle.key.token_count == cached_tokens;
            handle.has_terminal_hidden = terminal_logits && terminal_block;
            handle.has_terminal_logits = terminal_logits && terminal_block;
            if (include_mtp_state)
            {
                handle.layout.includes_mtp_state = true;
                handle.mtp_storage = std::make_shared<std::vector<uint8_t>>(8, 0x5a);
                handle.mtp_payload = handle.mtp_storage->data();
            }
            hit.blocks.push_back(handle);
        }
    }
    return hit;
}

static std::unique_ptr<MoERebalanceController> makeDomainController(
    const std::string &domain_id)
{
    MoERebalanceController::Config cfg;
    cfg.domain_id = domain_id;
    cfg.mode = MoERebalanceMode::OBSERVE;
    cfg.num_layers = 1;
    cfg.num_experts = 2;
    cfg.top_k = 1;
    cfg.window_size = 4;
    cfg.sockets = {DeviceId(DeviceType::CPU, 0), DeviceId(DeviceType::CPU, 1)};
    cfg.initial_expert_to_socket = {0, 1};
    return std::make_unique<MoERebalanceController>(std::move(cfg));
}

static std::unique_ptr<MockLocalTPContext> makeTPContextForRunnerCount(int count)
{
    MockLocalTPContext::Config tp_config;
    for (int i = 0; i < count; ++i)
    {
        tp_config.devices.push_back(GlobalDeviceAddress::cpu());
        tp_config.weights.push_back(1.0f / static_cast<float>(count));
    }
    return std::make_unique<MockLocalTPContext>(tp_config);
}

static RankOrchestrator::Config makeRankConfigForRunnerCount(int count)
{
    RankOrchestrator::Config config;
    config.mode = RankOrchestrator::ParallelismMode::TP;
    for (int i = 0; i < count; ++i)
    {
        config.devices.push_back(GlobalDeviceAddress::cpu());
        config.weights.push_back(1.0f / static_cast<float>(count));
    }
    config.prefix_cache.enabled = true;
    config.prefix_cache.storage_mode = PrefixCacheStorageMode::Ram;
    config.prefix_cache.block_size = 2;
    return config;
}

static MTPSpecStepPlan makeMTPSpecPublicationPlan(
    int accepted_count,
    int draft_count = 3)
{
    MTPSpecStepPlan plan;
    plan.request_id = 11;
    plan.draft_count = draft_count;
    plan.target_rows = draft_count + 1;
    plan.valid_sampled_count = draft_count + 1;
    plan.committed_output_count = draft_count;
    plan.accepted_count = accepted_count;
    plan.base_cached_tokens = 64;
    plan.target_cached_tokens = plan.base_cached_tokens + accepted_count;
    plan.accepted_state_slot_index =
        accepted_count > 0 ? accepted_count - 1 : kMTPSpecDecodeInvalidToken;
    plan.next_condition_token = 123;
    plan.all_drafts_accepted = accepted_count == draft_count;
    if (plan.all_drafts_accepted)
    {
        plan.bonus_ready_token_row = draft_count;
        plan.bonus_ready_token_index = draft_count;
        plan.bonus_ready_state_slot_index = draft_count;
    }
    return plan;
}

static MTPSpecStepPlanBatch makeMTPSpecPublicationBatch()
{
    MTPSpecStepPlanBatch batch;
    batch.ok = true;
    batch.shape.max_requests = 2;
    batch.shape.max_draft_tokens = 3;
    batch.request_count = 2;

    MTPSpecStepPlan first =
        makeMTPSpecPublicationPlan(/*accepted_count=*/2, /*draft_count=*/3);
    first.request_index = 0;
    first.request_id = 101;
    first.base_cached_tokens = 64;
    first.target_cached_tokens = 66;
    first.accepted_state_slot_index = 1;

    MTPSpecStepPlan second =
        makeMTPSpecPublicationPlan(/*accepted_count=*/1, /*draft_count=*/2);
    second.request_index = 1;
    second.request_id = 102;
    second.base_cached_tokens = 80;
    second.target_cached_tokens = 81;
    second.accepted_state_slot_index = 4;

    batch.steps = {first, second};
    return batch;
}

// =============================================================================
// Construction Tests
// =============================================================================

TEST_F(Test__RankOrchestrator, ConstructsWithValidConfig)
{
    // Verify mock setup is valid
    ASSERT_EQ(mock_runners_.size(), 2u);
    ASSERT_NE(mock_tp_ctx_, nullptr);
    EXPECT_EQ(mock_tp_ctx_->degree(), 2);
}

TEST_F(Test__RankOrchestrator, MockTPContextDegreeMatchesDevices)
{
    // Create TP context with 3 devices
    MockLocalTPContext::Config tp_config;
    tp_config.devices = {
        GlobalDeviceAddress::cpu(),
        GlobalDeviceAddress::cpu(),
        GlobalDeviceAddress::cpu()};
    auto tp_ctx = std::make_unique<MockLocalTPContext>(tp_config);

    EXPECT_EQ(tp_ctx->degree(), 3);
    EXPECT_EQ(tp_ctx->devices().size(), 3u);
}

// =============================================================================
// Mock Device Runner Tests
// =============================================================================

TEST_F(Test__RankOrchestrator, MockDeviceRunnerForwardTracksCallCount)
{
    auto runner = std::make_unique<MockDeviceGraphOrchestrator>();
    int tokens[] = {1, 2, 3};

    EXPECT_EQ(runner->forward_call_count(), 0u);

    bool result = runner->forward(tokens, 3);
    EXPECT_TRUE(result);
    EXPECT_EQ(runner->forward_call_count(), 1u);

    result = runner->forward(tokens, 3);
    EXPECT_TRUE(result);
    EXPECT_EQ(runner->forward_call_count(), 2u);
}

TEST_F(Test__RankOrchestrator, MockDeviceRunnerClearCacheTracksCallCount)
{
    auto runner = std::make_unique<MockDeviceGraphOrchestrator>();

    EXPECT_EQ(runner->clear_cache_call_count(), 0u);

    runner->clear_cache();
    EXPECT_EQ(runner->clear_cache_call_count(), 1u);

    runner->clear_cache();
    EXPECT_EQ(runner->clear_cache_call_count(), 2u);
}

TEST_F(Test__RankOrchestrator, MockDeviceRunnerReturnsConfiguredVocabSize)
{
    MockDeviceGraphOrchestrator::Config config;
    config.vocab_size = 50000;
    auto runner = std::make_unique<MockDeviceGraphOrchestrator>(config);

    EXPECT_EQ(runner->vocab_size(), 50000);
}

TEST_F(Test__RankOrchestrator, MockDeviceRunnerForwardCanFail)
{
    auto runner = std::make_unique<MockDeviceGraphOrchestrator>();
    runner->set_forward_fails(true);

    int tokens[] = {1, 2, 3};
    bool result = runner->forward(tokens, 3);

    EXPECT_FALSE(result);
    EXPECT_EQ(runner->forward_call_count(), 1u); // Still tracked
}

TEST_F(Test__RankOrchestrator, MockDeviceRunnerReturnsLogits)
{
    auto runner = std::make_unique<MockDeviceGraphOrchestrator>();
    runner->set_mock_logits({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});

    const float *logits = runner->logits();
    ASSERT_NE(logits, nullptr);
    EXPECT_FLOAT_EQ(logits[0], 1.0f);
    EXPECT_FLOAT_EQ(logits[4], 5.0f);
}

TEST_F(Test__RankOrchestrator, MockDeviceRunnerReturnsArchitecture)
{
    MockDeviceGraphOrchestrator::Config config;
    config.architecture = "test_arch";
    auto runner = std::make_unique<MockDeviceGraphOrchestrator>(config);

    EXPECT_STREQ(runner->architecture(), "test_arch");
}

TEST_F(Test__RankOrchestrator, MockDeviceRunnerReturnsGraphExecutionPath)
{
    auto runner = std::make_unique<MockDeviceGraphOrchestrator>();
    EXPECT_EQ(runner->executionPath(), ExecutionPath::GRAPH);
}

// =============================================================================
// Mock TP Context Tests
// =============================================================================

TEST_F(Test__RankOrchestrator, MockTPContextReturnsConfiguredDevices)
{
    MockLocalTPContext::Config config;
    config.devices = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)};
    auto ctx = std::make_unique<MockLocalTPContext>(config);

    EXPECT_EQ(ctx->devices().size(), 2u);
    EXPECT_EQ(ctx->devices()[0], GlobalDeviceAddress::cuda(0));
    EXPECT_EQ(ctx->devices()[1], GlobalDeviceAddress::cuda(1));
}

TEST_F(Test__RankOrchestrator, MockTPContextReturnsConfiguredWeights)
{
    MockLocalTPContext::Config config;
    config.devices = {GlobalDeviceAddress::cpu(), GlobalDeviceAddress::cpu()};
    config.weights = {0.7f, 0.3f};
    auto ctx = std::make_unique<MockLocalTPContext>(config);

    EXPECT_EQ(ctx->weights().size(), 2u);
    EXPECT_FLOAT_EQ(ctx->weights()[0], 0.7f);
    EXPECT_FLOAT_EQ(ctx->weights()[1], 0.3f);
}

TEST_F(Test__RankOrchestrator, MockTPContextSynchronizeTracksCallCount)
{
    auto ctx = std::make_unique<MockLocalTPContext>();

    EXPECT_EQ(ctx->synchronize_call_count(), 0u);

    ctx->synchronize();
    EXPECT_EQ(ctx->synchronize_call_count(), 1u);

    ctx->synchronize();
    ctx->synchronize();
    EXPECT_EQ(ctx->synchronize_call_count(), 3u);
}

TEST_F(Test__RankOrchestrator, MockTPContextAllreduceReturnsTrue)
{
    auto ctx = std::make_unique<MockLocalTPContext>();

    bool result = ctx->allreduce(static_cast<TensorBase *>(nullptr));
    EXPECT_TRUE(result);
    EXPECT_EQ(ctx->allreduce_call_count(), 1u);
}

TEST_F(Test__RankOrchestrator, MockTPContextAllreduceCanFail)
{
    auto ctx = std::make_unique<MockLocalTPContext>();
    ctx->set_allreduce_fails(true);

    bool result = ctx->allreduce(static_cast<TensorBase *>(nullptr));
    EXPECT_FALSE(result);
    EXPECT_EQ(ctx->allreduce_call_count(), 1u); // Still tracked
}

TEST_F(Test__RankOrchestrator, MockTPContextAllgatherReturnsTrue)
{
    auto ctx = std::make_unique<MockLocalTPContext>();

    bool result = ctx->allgather(nullptr, nullptr);
    EXPECT_TRUE(result);
    EXPECT_EQ(ctx->allgather_call_count(), 1u);
}

TEST_F(Test__RankOrchestrator, MockTPContextDeviceIndexing)
{
    MockLocalTPContext::Config config;
    config.devices = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::rocm(1)};
    auto ctx = std::make_unique<MockLocalTPContext>(config);

    EXPECT_EQ(ctx->indexForDevice(GlobalDeviceAddress::cuda(0)), 0);
    EXPECT_EQ(ctx->indexForDevice(GlobalDeviceAddress::rocm(1)), 1);
    EXPECT_EQ(ctx->indexForDevice(GlobalDeviceAddress::cpu()), -1); // Not found
}

TEST_F(Test__RankOrchestrator, MockTPContextDeviceAtReturnsCorrectDevice)
{
    MockLocalTPContext::Config config;
    config.devices = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)};
    auto ctx = std::make_unique<MockLocalTPContext>(config);

    EXPECT_EQ(ctx->deviceAt(0), GlobalDeviceAddress::cuda(0));
    EXPECT_EQ(ctx->deviceAt(1), GlobalDeviceAddress::cuda(1));
}

TEST_F(Test__RankOrchestrator, MockTPContextDeviceAtThrowsForInvalidIndex)
{
    auto ctx = std::make_unique<MockLocalTPContext>();

    EXPECT_THROW(ctx->deviceAt(-1), std::out_of_range);
    EXPECT_THROW(ctx->deviceAt(99), std::out_of_range);
}

TEST_F(Test__RankOrchestrator, MockTPContextWeightForDevice)
{
    MockLocalTPContext::Config config;
    config.devices = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)};
    config.weights = {0.6f, 0.4f};
    auto ctx = std::make_unique<MockLocalTPContext>(config);

    EXPECT_FLOAT_EQ(ctx->weightForDevice(GlobalDeviceAddress::cuda(0)), 0.6f);
    EXPECT_FLOAT_EQ(ctx->weightForDevice(GlobalDeviceAddress::cuda(1)), 0.4f);
    EXPECT_FLOAT_EQ(ctx->weightForDevice(GlobalDeviceAddress::cpu()), 0.0f); // Not found
}

TEST_F(Test__RankOrchestrator, MockTPContextHeadsForDevice)
{
    MockLocalTPContext::Config config;
    config.devices = {GlobalDeviceAddress::cpu(), GlobalDeviceAddress::cpu()};
    config.weights = {0.5f, 0.5f};
    auto ctx = std::make_unique<MockLocalTPContext>(config);

    // 16 total heads with 50/50 weights = 8 each
    EXPECT_EQ(ctx->headsForDevice(config.devices[0], 16), 8);
    EXPECT_EQ(ctx->headsForDevice(config.devices[1], 16), 8);
}

// =============================================================================
// Integration: Multi-Runner Coordination Tests
// =============================================================================

TEST_F(Test__RankOrchestrator, AllDevicesReadyWhenAllHaveVocabSize)
{
    // Verify that when all mock runners have valid vocab_size, allDevicesReady would return true
    // (Testing the mock behavior that the real orchestrator uses)
    for (auto &runner : mock_runners_)
    {
        EXPECT_GT(runner->vocab_size(), 0);
    }
    EXPECT_EQ(mock_runners_.size(), 2u);
}

TEST_F(Test__RankOrchestrator, MultipleRunnersCanBeCalledInParallel)
{
    // Simulate what RankOrchestrator does: call forward on all runners
    int tokens[] = {1, 2, 3};

    // Call forward on all mock runners
    bool all_success = true;
    for (auto &runner : mock_runners_)
    {
        if (!runner->forward(tokens, 3))
        {
            all_success = false;
        }
    }

    EXPECT_TRUE(all_success);
    for (auto &runner : mock_runners_)
    {
        EXPECT_EQ(runner->forward_call_count(), 1u);
    }
}

TEST_F(Test__RankOrchestrator, ForwardFailsIfAnyDeviceFails)
{
    // Set one runner to fail
    mock_runners_[1]->set_forward_fails(true);

    int tokens[] = {1, 2, 3};

    // Simulate RankOrchestrator::forward
    bool all_success = true;
    for (auto &runner : mock_runners_)
    {
        if (!runner->forward(tokens, 3))
        {
            all_success = false;
        }
    }

    EXPECT_FALSE(all_success);
    // Both should have been called
    EXPECT_EQ(mock_runners_[0]->forward_call_count(), 1u);
    EXPECT_EQ(mock_runners_[1]->forward_call_count(), 1u);
}

TEST_F(Test__RankOrchestrator, ForwardTPWorkerTimeoutAbortsInsteadOfHanging)
{
    std::vector<std::unique_ptr<IInferenceRunner>> runners;

    auto slow_runner = std::make_unique<MockDeviceGraphOrchestrator>();
    slow_runner->set_forward_sleep_ms(250);
    runners.push_back(std::move(slow_runner));
    runners.push_back(std::make_unique<MockDeviceGraphOrchestrator>());

    EXPECT_DEATH(
        {
            setenv("LLAMINAR_TP_COLLECT_TIMEOUT_MS", "10", 1);
            mutableDebugEnv().reload();

            auto orchestrator = RankOrchestrator::createForTest(
                llaminar2::test::MockModelContext::createMinimal(),
                std::move(runners),
                makeTPContextForRunnerCount(2),
                makeRankConfigForRunnerCount(2));

            int tokens[] = {1};
            (void)orchestrator->forward(tokens, 1);
        },
        "");
}

TEST_F(Test__RankOrchestrator, ClearCacheClearsAllDevices)
{
    // Simulate RankOrchestrator::clear_cache
    for (auto &runner : mock_runners_)
    {
        runner->clear_cache();
    }

    for (auto &runner : mock_runners_)
    {
        EXPECT_EQ(runner->clear_cache_call_count(), 1u);
    }
}

TEST_F(Test__RankOrchestrator, LogitsReturnsFromPrimaryDevice)
{
    // Set different logits on each runner
    mock_runners_[0]->set_mock_logits({10.0f, 20.0f, 30.0f});
    mock_runners_[1]->set_mock_logits({1.0f, 2.0f, 3.0f});

    // Simulate RankOrchestrator::logits (returns from primary device)
    const float *logits = mock_runners_[0]->logits();

    ASSERT_NE(logits, nullptr);
    EXPECT_FLOAT_EQ(logits[0], 10.0f);
    EXPECT_FLOAT_EQ(logits[1], 20.0f);
    EXPECT_FLOAT_EQ(logits[2], 30.0f);
}

TEST_F(Test__RankOrchestrator, VocabSizeFromPrimaryDevice)
{
    mock_runners_[0]->set_vocab_size(50000);
    mock_runners_[1]->set_vocab_size(32000);

    // Simulate RankOrchestrator::vocab_size (returns from primary device)
    int vocab = mock_runners_[0]->vocab_size();

    EXPECT_EQ(vocab, 50000);
}

TEST_F(Test__RankOrchestrator, SynchronizeDevicesCallsTPContext)
{
    // Simulate RankOrchestrator::synchronizeDevices
    mock_tp_ctx_->synchronize();

    EXPECT_EQ(mock_tp_ctx_->synchronize_call_count(), 1u);
}

TEST_F(Test__RankOrchestrator, DeviceCountMatchesTPDegree)
{
    // Verify TP context degree matches expected device count
    EXPECT_EQ(mock_tp_ctx_->degree(), static_cast<int>(mock_runners_.size()));
}

TEST_F(Test__RankOrchestrator, LocalTPContextReturnsContext)
{
    // Verify mock TP context is valid
    ASSERT_NE(mock_tp_ctx_, nullptr);
    EXPECT_EQ(mock_tp_ctx_->degree(), 2);
    EXPECT_EQ(mock_tp_ctx_->backend(), CollectiveBackendType::HOST);
}

TEST_F(Test__RankOrchestrator, MoERebalanceControllersAreLookupByDomain)
{
    auto runner0 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner0_ptr = runner0.get();
    runner0_ptr->set_moe_rebalance_controller(makeDomainController("hot_rocm"));

    auto runner1 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner1_ptr = runner1.get();
    runner1_ptr->set_moe_rebalance_controller(makeDomainController("cold_cpu"));

    std::vector<std::unique_ptr<IInferenceRunner>> runners;
    runners.push_back(std::move(runner0));
    runners.push_back(std::move(runner1));

    auto orchestrator = RankOrchestrator::createForTest(
        llaminar2::test::MockModelContext::createMinimal(),
        std::move(runners),
        makeTPContextForRunnerCount(2),
        makeRankConfigForRunnerCount(2));

    auto controllers = orchestrator->moeRebalanceControllers();
    ASSERT_EQ(controllers.size(), 2u);
    EXPECT_EQ(controllers[0]->domainId(), "hot_rocm");
    EXPECT_EQ(controllers[1]->domainId(), "cold_cpu");

    EXPECT_EQ(orchestrator->moeRebalanceController(), controllers[0])
        << "Compatibility lookup remains first-controller only";
    EXPECT_EQ(orchestrator->moeRebalanceControllerForDomain("hot_rocm"), controllers[0]);
    EXPECT_EQ(orchestrator->moeRebalanceControllerForDomain("cold_cpu"), controllers[1]);
    EXPECT_EQ(orchestrator->moeRebalanceControllerForDomain("missing"), nullptr);
}

TEST_F(Test__RankOrchestrator, PrefixLookupClampsToCommonLocalTPMinimum)
{
    auto runner0 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner0_ptr = runner0.get();
    runner0_ptr->set_prefix_lookup_result(makePrefixHit(/*cached_tokens=*/4, /*terminal_logits=*/true));
    runner0_ptr->set_moe_placement_epoch(7);

    auto runner1 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner1_ptr = runner1.get();
    runner1_ptr->set_prefix_lookup_result(makePrefixHit(/*cached_tokens=*/2, /*terminal_logits=*/false));
    runner1_ptr->set_moe_placement_epoch(19);

    std::vector<std::unique_ptr<IInferenceRunner>> runners;
    runners.push_back(std::move(runner0));
    runners.push_back(std::move(runner1));

    auto orchestrator = RankOrchestrator::createForTest(
        llaminar2::test::MockModelContext::createMinimal(),
        std::move(runners),
        makeTPContextForRunnerCount(2),
        makeRankConfigForRunnerCount(2));

    const std::vector<int32_t> prompt = {1, 2, 3, 4};
    PrefixLookupResult hit = orchestrator->lookupPrefix(prompt);
    EXPECT_TRUE(hit.cache_enabled);
    EXPECT_TRUE(hit.supported);
    EXPECT_EQ(hit.cached_tokens, 2);
    EXPECT_EQ(hit.placement_epoch, 19u);
    EXPECT_EQ(orchestrator->moePlacementEpoch(), 19u);
    EXPECT_FALSE(hit.has_terminal_logits)
        << "Rank-level terminal state is usable only when all children have it";
    EXPECT_EQ(runner0_ptr->prefix_lookup_tokens(), prompt);
    EXPECT_EQ(runner1_ptr->prefix_lookup_tokens(), prompt);

    ASSERT_TRUE(orchestrator->populatePrefix(hit));
    EXPECT_EQ(runner0_ptr->populated_prefix_tokens(), std::vector<int>({2}));
    EXPECT_EQ(runner1_ptr->populated_prefix_tokens(), std::vector<int>({2}));
}

TEST_F(Test__RankOrchestrator, PrefixLookupAllowsParticipantLocalFingerprintsForLocalTPSlices)
{
    PrefixLookupResult shard0_hit = makePrefixHit(/*cached_tokens=*/4,
                                                  /*terminal_logits=*/true,
                                                  /*supported=*/true,
                                                  /*include_blocks=*/true,
                                                  /*include_mtp_state=*/true);
    shard0_hit.fingerprint_key = 0x1000;

    PrefixLookupResult shard1_hit = makePrefixHit(/*cached_tokens=*/4,
                                                  /*terminal_logits=*/true,
                                                  /*supported=*/true,
                                                  /*include_blocks=*/true,
                                                  /*include_mtp_state=*/true);
    shard1_hit.fingerprint_key = 0x2000;

    auto runner0 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner0_ptr = runner0.get();
    runner0_ptr->set_prefix_lookup_result(std::move(shard0_hit));

    auto runner1 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner1_ptr = runner1.get();
    runner1_ptr->set_prefix_lookup_result(std::move(shard1_hit));

    std::vector<std::unique_ptr<IInferenceRunner>> runners;
    runners.push_back(std::move(runner0));
    runners.push_back(std::move(runner1));

    auto orchestrator = RankOrchestrator::createForTest(
        llaminar2::test::MockModelContext::createMinimal(),
        std::move(runners),
        makeTPContextForRunnerCount(2),
        makeRankConfigForRunnerCount(2));

    PrefixLookupResult hit = orchestrator->lookupPrefix({1, 2, 3, 4});
    EXPECT_TRUE(hit.cache_enabled);
    EXPECT_TRUE(hit.supported);
    EXPECT_EQ(hit.cached_tokens, 4);
    EXPECT_EQ(hit.fingerprint_key, 0u)
        << "Rank-level fingerprints remain participant-local for sharded TP payloads";
    EXPECT_TRUE(hit.has_terminal_logits);
    EXPECT_TRUE(hit.has_terminal_hidden);
    ASSERT_EQ(hit.blocks.size(), 2u);
    EXPECT_NE(hit.blocks.back().mtp_payload, nullptr);

    ASSERT_TRUE(orchestrator->populatePrefix(hit));
    EXPECT_EQ(runner0_ptr->populated_prefix_tokens(), std::vector<int>({4}));
    EXPECT_EQ(runner1_ptr->populated_prefix_tokens(), std::vector<int>({4}));

    ASSERT_TRUE(orchestrator->restorePrefixTerminalState(hit));
    EXPECT_EQ(runner0_ptr->terminal_restored_tokens(), std::vector<int>({4}));
    EXPECT_EQ(runner1_ptr->terminal_restored_tokens(), std::vector<int>({4}));
}

TEST_F(Test__RankOrchestrator, PrefixLookupAllowsTerminalLogitsOnOnlyOwningPPStage)
{
    auto runner0 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner0_ptr = runner0.get();
    runner0_ptr->set_prefix_lookup_result(makePrefixHit(/*cached_tokens=*/4,
                                                        /*terminal_logits=*/false,
                                                        /*supported=*/true,
                                                        /*include_blocks=*/false,
                                                        /*include_mtp_state=*/false,
                                                        /*requires_terminal_logits=*/false,
                                                        /*requires_terminal_hidden=*/false));

    auto runner1 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner1_ptr = runner1.get();
    runner1_ptr->set_prefix_lookup_result(makePrefixHit(/*cached_tokens=*/4,
                                                        /*terminal_logits=*/true,
                                                        /*supported=*/true,
                                                        /*include_blocks=*/false,
                                                        /*include_mtp_state=*/false,
                                                        /*requires_terminal_logits=*/true,
                                                        /*requires_terminal_hidden=*/true));

    std::vector<std::unique_ptr<IInferenceRunner>> runners;
    runners.push_back(std::move(runner0));
    runners.push_back(std::move(runner1));

    auto orchestrator = RankOrchestrator::createForTest(
        llaminar2::test::MockModelContext::createMinimal(),
        std::move(runners),
        makeTPContextForRunnerCount(2),
        makeRankConfigForRunnerCount(2));

    PrefixLookupResult hit = orchestrator->lookupPrefix({1, 2, 3, 4});
    EXPECT_TRUE(hit.has_terminal_logits);
    EXPECT_TRUE(hit.has_terminal_hidden);
    ASSERT_TRUE(orchestrator->restorePrefixTerminalState(hit));
    EXPECT_EQ(runner0_ptr->terminal_restored_tokens(), std::vector<int>({4}));
    EXPECT_EQ(runner1_ptr->terminal_restored_tokens(), std::vector<int>({4}));
}

TEST_F(Test__RankOrchestrator, PrefixLookupChildMissClampsAllChildrenToZero)
{
    auto runner0 = std::make_unique<MockDeviceGraphOrchestrator>();
    runner0->set_prefix_lookup_result(makePrefixHit(/*cached_tokens=*/4, /*terminal_logits=*/true));

    auto runner1 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner1_ptr = runner1.get();
    runner1->set_prefix_lookup_result(makePrefixHit(/*cached_tokens=*/0,
                                                    /*terminal_logits=*/false,
                                                    /*supported=*/false));

    std::vector<std::unique_ptr<IInferenceRunner>> runners;
    runners.push_back(std::move(runner0));
    runners.push_back(std::move(runner1));

    auto orchestrator = RankOrchestrator::createForTest(
        llaminar2::test::MockModelContext::createMinimal(),
        std::move(runners),
        makeTPContextForRunnerCount(2),
        makeRankConfigForRunnerCount(2));

    PrefixLookupResult hit = orchestrator->lookupPrefix({1, 2, 3, 4});
    EXPECT_FALSE(hit.supported);
    EXPECT_EQ(hit.cached_tokens, 0);
    EXPECT_EQ(runner1_ptr->prefix_lookup_call_count(), 1u);
}

TEST_F(Test__RankOrchestrator, PrefixTerminalRestoreRunsOnAllChildrenAtCommonLength)
{
    auto runner0 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner0_ptr = runner0.get();
    runner0_ptr->set_prefix_lookup_result(makePrefixHit(/*cached_tokens=*/4, /*terminal_logits=*/true));

    auto runner1 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner1_ptr = runner1.get();
    runner1_ptr->set_prefix_lookup_result(makePrefixHit(/*cached_tokens=*/4, /*terminal_logits=*/true));

    std::vector<std::unique_ptr<IInferenceRunner>> runners;
    runners.push_back(std::move(runner0));
    runners.push_back(std::move(runner1));

    auto orchestrator = RankOrchestrator::createForTest(
        llaminar2::test::MockModelContext::createMinimal(),
        std::move(runners),
        makeTPContextForRunnerCount(2),
        makeRankConfigForRunnerCount(2));

    PrefixLookupResult hit = orchestrator->lookupPrefix({1, 2, 3, 4});
    ASSERT_TRUE(hit.has_terminal_logits);
    ASSERT_TRUE(orchestrator->restorePrefixTerminalState(hit));

    EXPECT_EQ(runner0_ptr->terminal_restored_tokens(), std::vector<int>({4}));
    EXPECT_EQ(runner1_ptr->terminal_restored_tokens(), std::vector<int>({4}));
}

TEST_F(Test__RankOrchestrator, PrefixTerminalRestoreSurvivesLiveCacheClearAfterLookup)
{
    auto runner0 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner0_ptr = runner0.get();
    runner0_ptr->set_prefix_lookup_result(makePrefixHit(/*cached_tokens=*/4,
                                                        /*terminal_logits=*/true,
                                                        /*supported=*/true,
                                                        /*include_blocks=*/true));
    runner0_ptr->set_prefix_terminal_restore_requires_blocks(true);

    auto runner1 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner1_ptr = runner1.get();
    runner1_ptr->set_prefix_lookup_result(makePrefixHit(/*cached_tokens=*/4,
                                                        /*terminal_logits=*/true,
                                                        /*supported=*/true,
                                                        /*include_blocks=*/true));
    runner1_ptr->set_prefix_terminal_restore_requires_blocks(true);

    std::vector<std::unique_ptr<IInferenceRunner>> runners;
    runners.push_back(std::move(runner0));
    runners.push_back(std::move(runner1));

    auto orchestrator = RankOrchestrator::createForTest(
        llaminar2::test::MockModelContext::createMinimal(),
        std::move(runners),
        makeTPContextForRunnerCount(2),
        makeRankConfigForRunnerCount(2));

    PrefixLookupResult hit = orchestrator->lookupPrefix({1, 2, 3, 4});
    ASSERT_TRUE(hit.has_terminal_logits);

    orchestrator->clear_cache();
    ASSERT_TRUE(orchestrator->restorePrefixTerminalState(hit))
        << "OrchestrationRunner clears live KV between lookup and restore; "
           "RankOrchestrator must keep pending child prefix handles for that request.";

    EXPECT_EQ(runner0_ptr->clear_cache_call_count(), 1u);
    EXPECT_EQ(runner1_ptr->clear_cache_call_count(), 1u);
    EXPECT_EQ(runner0_ptr->terminal_restored_tokens(), std::vector<int>({4}));
    EXPECT_EQ(runner1_ptr->terminal_restored_tokens(), std::vector<int>({4}));
}

TEST_F(Test__RankOrchestrator, PrefixLookupCarriesRepresentativeMTPBlocksForSummaries)
{
    auto runner0 = std::make_unique<MockDeviceGraphOrchestrator>();
    runner0->set_prefix_lookup_result(makePrefixHit(/*cached_tokens=*/4,
                                                    /*terminal_logits=*/true,
                                                    /*supported=*/true,
                                                    /*include_blocks=*/true,
                                                    /*include_mtp_state=*/true));

    auto runner1 = std::make_unique<MockDeviceGraphOrchestrator>();
    runner1->set_prefix_lookup_result(makePrefixHit(/*cached_tokens=*/4,
                                                    /*terminal_logits=*/true,
                                                    /*supported=*/true,
                                                    /*include_blocks=*/true,
                                                    /*include_mtp_state=*/true));

    std::vector<std::unique_ptr<IInferenceRunner>> runners;
    runners.push_back(std::move(runner0));
    runners.push_back(std::move(runner1));

    auto orchestrator = RankOrchestrator::createForTest(
        llaminar2::test::MockModelContext::createMinimal(),
        std::move(runners),
        makeTPContextForRunnerCount(2),
        makeRankConfigForRunnerCount(2));

    PrefixLookupResult hit = orchestrator->lookupPrefix({1, 2, 3, 4});
    ASSERT_TRUE(hit.supported);
    ASSERT_EQ(hit.cached_tokens, 4);
    ASSERT_EQ(hit.blocks.size(), 2u);
    EXPECT_NE(hit.blocks.front().mtp_payload, nullptr);
}

TEST_F(Test__RankOrchestrator, PrefixTerminalRestoreSkipsWhenAggregateTerminalUnavailable)
{
    auto runner0 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner0_ptr = runner0.get();
    runner0_ptr->set_prefix_lookup_result(makePrefixHit(/*cached_tokens=*/4, /*terminal_logits=*/true));

    auto runner1 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner1_ptr = runner1.get();
    runner1_ptr->set_prefix_lookup_result(makePrefixHit(/*cached_tokens=*/4, /*terminal_logits=*/false));

    std::vector<std::unique_ptr<IInferenceRunner>> runners;
    runners.push_back(std::move(runner0));
    runners.push_back(std::move(runner1));

    auto orchestrator = RankOrchestrator::createForTest(
        llaminar2::test::MockModelContext::createMinimal(),
        std::move(runners),
        makeTPContextForRunnerCount(2),
        makeRankConfigForRunnerCount(2));

    PrefixLookupResult hit = orchestrator->lookupPrefix({1, 2, 3, 4});
    ASSERT_FALSE(hit.has_terminal_logits);
    EXPECT_FALSE(orchestrator->restorePrefixTerminalState(hit));
    EXPECT_EQ(runner0_ptr->prefix_terminal_restore_call_count(), 0u);
    EXPECT_EQ(runner1_ptr->prefix_terminal_restore_call_count(), 0u);
}

TEST_F(Test__RankOrchestrator, PrefixPopulateFailureClearsAllChildren)
{
    auto runner0 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner0_ptr = runner0.get();
    runner0_ptr->set_prefix_lookup_result(makePrefixHit(/*cached_tokens=*/4, /*terminal_logits=*/true));

    auto runner1 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner1_ptr = runner1.get();
    runner1_ptr->set_prefix_lookup_result(makePrefixHit(/*cached_tokens=*/4, /*terminal_logits=*/true));
    runner1_ptr->set_prefix_populate_ok(false);

    std::vector<std::unique_ptr<IInferenceRunner>> runners;
    runners.push_back(std::move(runner0));
    runners.push_back(std::move(runner1));

    auto orchestrator = RankOrchestrator::createForTest(
        llaminar2::test::MockModelContext::createMinimal(),
        std::move(runners),
        makeTPContextForRunnerCount(2),
        makeRankConfigForRunnerCount(2));

    PrefixLookupResult hit = orchestrator->lookupPrefix({1, 2, 3, 4});
    ASSERT_FALSE(orchestrator->populatePrefix(hit));

    EXPECT_EQ(runner0_ptr->populated_prefix_tokens(), std::vector<int>({4}));
    EXPECT_EQ(runner1_ptr->populated_prefix_tokens(), std::vector<int>({4}));
    EXPECT_EQ(runner0_ptr->clear_cache_call_count(), 1u);
    EXPECT_EQ(runner1_ptr->clear_cache_call_count(), 1u);
}

TEST_F(Test__RankOrchestrator, PrefixLookupPipelineStageMissClampsWholePipeline)
{
    auto stage0 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *stage0_ptr = stage0.get();
    stage0_ptr->set_prefix_lookup_result(makePrefixHit(/*cached_tokens=*/6, /*terminal_logits=*/true));

    auto stage1 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *stage1_ptr = stage1.get();
    stage1_ptr->set_prefix_lookup_result(makePrefixHit(/*cached_tokens=*/2, /*terminal_logits=*/false));

    std::vector<std::unique_ptr<IInferenceRunner>> stages;
    stages.push_back(std::move(stage0));
    stages.push_back(std::move(stage1));

    auto orchestrator = RankOrchestrator::createForTestWithPipelineStages(
        llaminar2::test::MockModelContext::createMinimal(),
        std::move(stages),
        makeRankConfigForRunnerCount(2));

    const std::vector<int32_t> prompt = {1, 2, 3, 4, 5, 6};
    PrefixLookupResult hit = orchestrator->lookupPrefix(prompt);

    EXPECT_TRUE(hit.cache_enabled);
    EXPECT_TRUE(hit.supported);
    EXPECT_EQ(hit.cached_tokens, 2);
    EXPECT_FALSE(hit.has_terminal_logits)
        << "Pipeline terminal state is usable only when every stage has it";
    EXPECT_EQ(stage0_ptr->prefix_lookup_tokens(), prompt);
    EXPECT_EQ(stage1_ptr->prefix_lookup_tokens(), prompt);

    ASSERT_TRUE(orchestrator->populatePrefix(hit));
    EXPECT_EQ(stage0_ptr->populated_prefix_tokens(), std::vector<int>({2}));
    EXPECT_EQ(stage1_ptr->populated_prefix_tokens(), std::vector<int>({2}));
}

TEST_F(Test__RankOrchestrator, LocalPPSidecarMethodsDelegateOnlyToFinalStage)
{
    auto stage0 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *stage0_ptr = stage0.get();
    stage0_ptr->set_forward_mtp_ok(false);
    stage0_ptr->set_mock_mtp_logits({10.0f, 0.0f, 0.0f});

    auto stage1 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *stage1_ptr = stage1.get();
    stage1_ptr->set_vocab_size(3);
    stage1_ptr->set_mock_mtp_logits({0.0f, 1.0f, 9.0f});
    stage1_ptr->set_supports_chained_mtp_drafts(true);

    std::vector<std::unique_ptr<IInferenceRunner>> stages;
    stages.push_back(std::move(stage0));
    stages.push_back(std::move(stage1));

    auto orchestrator = RankOrchestrator::createForTestWithPipelineStages(
        llaminar2::test::MockModelContext::createMinimal(),
        std::move(stages),
        makeRankConfigForRunnerCount(2));

    /*
     * LocalPP MTP does not treat stages as TP participants.  The normal
     * verifier still runs through forwardPP(), but sidecar-only operations are
     * owned by the pipeline tail because that is where terminal hidden, final
     * norm, LM head, and MTP logits live.
     */
    EXPECT_TRUE(orchestrator->mtpDecodeUnsupportedReason().empty());
    EXPECT_TRUE(orchestrator->forwardMTP(42));
    EXPECT_EQ(stage0_ptr->forward_mtp_call_count(), 0u);
    EXPECT_EQ(stage1_ptr->forward_mtp_call_count(), 1u);
    EXPECT_EQ(stage1_ptr->last_mtp_condition_token(), 42);

    EXPECT_TRUE(orchestrator->supportsChainedMTPDrafts());
    EXPECT_TRUE(orchestrator->forwardMTPFromLastDraft(77, 13));
    EXPECT_EQ(stage0_ptr->forward_mtp_from_last_draft_call_count(), 0u);
    EXPECT_EQ(stage1_ptr->forward_mtp_from_last_draft_call_count(), 1u);
    EXPECT_EQ(stage1_ptr->last_chained_mtp_condition_token(), 77);
    EXPECT_EQ(stage1_ptr->last_chained_mtp_position_id(), 13);

    ASSERT_NE(orchestrator->mtpLogits(), nullptr);
    EXPECT_FLOAT_EQ(orchestrator->mtpLogits()[2], 9.0f);
    EXPECT_EQ(orchestrator->sampleGreedyFromMTPLogitsOnDevice(), 2);
    EXPECT_EQ(stage0_ptr->sample_mtp_logits_call_count(), 0u);
    EXPECT_EQ(stage1_ptr->sample_mtp_logits_call_count(), 1u);

    const int32_t accepted_tokens[] = {5, 6};
    EXPECT_TRUE(orchestrator->commitMTPShiftedRowsFromPartialForward(
        accepted_tokens,
        2,
        1,
        2,
        /*allow_speculative_discard=*/true,
        /*position_offset_override=*/99));
    EXPECT_EQ(stage0_ptr->commit_mtp_shifted_rows_call_count(), 0u);
    EXPECT_EQ(stage1_ptr->commit_mtp_shifted_rows_call_count(), 1u);
    EXPECT_EQ(stage1_ptr->last_commit_mtp_already_appended(), 1);
    EXPECT_EQ(stage1_ptr->last_commit_mtp_main_forward_token_count(), 2);
    EXPECT_TRUE(stage1_ptr->last_commit_mtp_allow_speculative_discard());
    EXPECT_EQ(stage1_ptr->last_commit_mtp_position_offset_override(), 99);
    EXPECT_EQ(stage1_ptr->last_commit_mtp_tokens(),
              std::vector<int32_t>({5, 6}));

    EXPECT_FALSE(orchestrator->supportsMTPSpecStatePublication())
        << "PP all-position publication must remain disabled until every "
           "stage can publish its own accepted verifier row state.";
}

TEST_F(Test__RankOrchestrator, LocalPPCheckpointTerminalHiddenDelegatesOnlyToFinalStage)
{
    auto stage0 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *stage0_ptr = stage0.get();
    stage0_ptr->set_ensure_mtp_checkpoint_terminal_hidden_ok(false);

    auto stage1 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *stage1_ptr = stage1.get();
    stage1_ptr->set_ensure_mtp_checkpoint_terminal_hidden_ok(true);

    std::vector<std::unique_ptr<IInferenceRunner>> stages;
    stages.push_back(std::move(stage0));
    stages.push_back(std::move(stage1));

    auto orchestrator = RankOrchestrator::createForTestWithPipelineStages(
        llaminar2::test::MockModelContext::createMinimal(),
        std::move(stages),
        makeRankConfigForRunnerCount(2));

    /*
     * ensureMTPCheckpointTerminalHidden() materializes the stable terminal row
     * used by MTP sidecar replay.  In LocalPP that row exists only on the final
     * pipeline stage.  Earlier stages still checkpoint their KV/GDN state through
     * captureLivePrefixState(), but they must not row-select from activation
     * tensors that have already been handed to the next stage.
     */
    EXPECT_TRUE(orchestrator->ensureMTPCheckpointTerminalHidden());
    EXPECT_EQ(stage0_ptr->ensure_mtp_checkpoint_terminal_hidden_call_count(), 0u);
    EXPECT_EQ(stage1_ptr->ensure_mtp_checkpoint_terminal_hidden_call_count(), 1u);
}

TEST_F(Test__RankOrchestrator, LocalPPAllPositionPublicationRunsOnEveryStage)
{
    auto stage0 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *stage0_ptr = stage0.get();
    stage0_ptr->set_supports_mtp_spec_state_publication(true);

    auto stage1 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *stage1_ptr = stage1.get();
    stage1_ptr->set_supports_mtp_spec_state_publication(true);
    stage1_ptr->set_mock_all_position_logits({1.0f, 3.0f, 2.0f});

    std::vector<std::unique_ptr<IInferenceRunner>> stages;
    stages.push_back(std::move(stage0));
    stages.push_back(std::move(stage1));

    auto orchestrator = RankOrchestrator::createForTestWithPipelineStages(
        llaminar2::test::MockModelContext::createMinimal(),
        std::move(stages),
        makeRankConfigForRunnerCount(2));

    EXPECT_TRUE(orchestrator->supportsMTPSpecStatePublication());
    EXPECT_TRUE(orchestrator->setComputeAllPositionLogits(true));
    EXPECT_TRUE(orchestrator->setComputeRowIndexedAllPositionLogits(true, 2));

    MTPSpecDecodeVerifierInputPlan verifier_plan;
    verifier_plan.ok = true;
    verifier_plan.compact_logit_row_count = 2;
    verifier_plan.verifier_logit_rows = {0, 1};
    EXPECT_TRUE(orchestrator->setMTPSpecVerifierInputPlan(verifier_plan));
    orchestrator->clearMTPSpecVerifierInputPlan();

    EXPECT_EQ(stage0_ptr->set_all_position_logits_call_count(), 1u);
    EXPECT_EQ(stage1_ptr->set_all_position_logits_call_count(), 1u);
    EXPECT_EQ(stage0_ptr->set_row_indexed_all_position_logits_call_count(), 1u);
    EXPECT_EQ(stage1_ptr->set_row_indexed_all_position_logits_call_count(), 1u);
    EXPECT_TRUE(stage0_ptr->compute_row_indexed_all_position_logits());
    EXPECT_TRUE(stage1_ptr->compute_row_indexed_all_position_logits());
    EXPECT_EQ(stage0_ptr->row_indexed_all_position_logit_rows(), 2);
    EXPECT_EQ(stage1_ptr->row_indexed_all_position_logit_rows(), 2);
    EXPECT_EQ(stage0_ptr->set_mtp_spec_verifier_input_plan_call_count(), 1u);
    EXPECT_EQ(stage1_ptr->set_mtp_spec_verifier_input_plan_call_count(), 1u);
    EXPECT_EQ(stage0_ptr->clear_mtp_spec_verifier_input_plan_call_count(), 1u);
    EXPECT_EQ(stage1_ptr->clear_mtp_spec_verifier_input_plan_call_count(), 1u);

    std::string error;
    EXPECT_TRUE(orchestrator->publishAcceptedMTPSpecState(
        makeMTPSpecPublicationPlan(/*accepted_count=*/2),
        &error))
        << error;
    EXPECT_EQ(stage0_ptr->publish_mtp_spec_state_call_count(), 1u);
    EXPECT_EQ(stage1_ptr->publish_mtp_spec_state_call_count(), 1u);
    EXPECT_EQ(stage0_ptr->last_mtp_spec_state_plan().accepted_count, 2);
    EXPECT_EQ(stage1_ptr->last_mtp_spec_state_plan().accepted_count, 2);
    EXPECT_FALSE(stage0_ptr->last_mtp_spec_state_plan().publish_mtp_shifted_kv);
    EXPECT_TRUE(stage1_ptr->last_mtp_spec_state_plan().publish_mtp_shifted_kv);
    ASSERT_NE(orchestrator->getAllPositionLogits(), nullptr);
    EXPECT_FLOAT_EQ(orchestrator->getAllPositionLogits()[1], 3.0f);
}

TEST_F(Test__RankOrchestrator, LocalPPBatchPublicationRunsOnEveryStageWithFinalShiftedKV)
{
    auto stage0 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *stage0_ptr = stage0.get();
    stage0_ptr->set_supports_mtp_spec_state_publication(true);

    auto stage1 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *stage1_ptr = stage1.get();
    stage1_ptr->set_supports_mtp_spec_state_publication(true);

    std::vector<std::unique_ptr<IInferenceRunner>> stages;
    stages.push_back(std::move(stage0));
    stages.push_back(std::move(stage1));

    auto orchestrator = RankOrchestrator::createForTestWithPipelineStages(
        llaminar2::test::MockModelContext::createMinimal(),
        std::move(stages),
        makeRankConfigForRunnerCount(2));

    MTPSpecStepPlanBatch batch = makeMTPSpecPublicationBatch();
    std::string error;
    EXPECT_TRUE(orchestrator->publishAcceptedMTPSpecStateBatch(batch, &error))
        << error;

    ASSERT_THAT(stage0_ptr->last_mtp_spec_state_batch().steps, ::testing::SizeIs(2));
    ASSERT_THAT(stage1_ptr->last_mtp_spec_state_batch().steps, ::testing::SizeIs(2));
    EXPECT_EQ(stage0_ptr->publish_mtp_spec_state_batch_call_count(), 1u);
    EXPECT_EQ(stage1_ptr->publish_mtp_spec_state_batch_call_count(), 1u);

    /*
     * Pipeline stages all publish their local verifier-captured state, but
     * only the final stage owns the shifted MTP sidecar KV cache.
     */
    EXPECT_FALSE(stage0_ptr->last_mtp_spec_state_batch().steps[0].publish_mtp_shifted_kv);
    EXPECT_FALSE(stage0_ptr->last_mtp_spec_state_batch().steps[1].publish_mtp_shifted_kv);
    EXPECT_TRUE(stage1_ptr->last_mtp_spec_state_batch().steps[0].publish_mtp_shifted_kv);
    EXPECT_TRUE(stage1_ptr->last_mtp_spec_state_batch().steps[1].publish_mtp_shifted_kv);
}

TEST_F(Test__RankOrchestrator, LocalPPStochasticDeviceHooksDelegateOnlyToFinalStage)
{
    auto stage0 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *stage0_ptr = stage0.get();
    stage0_ptr->set_primary_device_id(DeviceId::rocm(0));

    auto stage1 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *stage1_ptr = stage1.get();
    stage1_ptr->set_primary_device_id(DeviceId::rocm(1));
    stage1_ptr->set_supports_device_stochastic_mtp_verification(true);
    stage1_ptr->set_supports_mtp_sidecar_preserves_main_state(true);
    stage1_ptr->set_stochastic_sample_token(23);

    std::vector<std::unique_ptr<IInferenceRunner>> stages;
    stages.push_back(std::move(stage0));
    stages.push_back(std::move(stage1));

    auto orchestrator = RankOrchestrator::createForTestWithPipelineStages(
        llaminar2::test::MockModelContext::createMinimal(),
        std::move(stages),
        makeRankConfigForRunnerCount(2));

    EXPECT_EQ(orchestrator->primaryDeviceId(), DeviceId::rocm(1))
        << "LocalPP MTP device policy follows the logits-owning final stage";
    EXPECT_TRUE(orchestrator->supportsDeviceStochasticMTPVerification());
    EXPECT_TRUE(orchestrator->supportsMTPSidecarPreservesMainState());
    EXPECT_FALSE(orchestrator->supportsMTPSidecarLogitsStreamHandoff())
        << "PP device-token handoff remains gated until token slots have an "
           "explicit pipeline-head ownership contract";
    EXPECT_FALSE(orchestrator->supportsMTPDeviceDraftTokenInput());

    const void *host_first_tokens =
        orchestrator->prepareMTPVerifierInputTokensOnDevice(
            /*first_token=*/11,
            /*first_draft_slot=*/2,
            /*draft_token_count=*/3,
            /*total_verifier_input_tokens=*/4);
    ASSERT_NE(host_first_tokens, nullptr);
    EXPECT_EQ(stage0_ptr->prepare_mtp_verifier_input_tokens_call_count(), 0u);
    EXPECT_EQ(stage1_ptr->prepare_mtp_verifier_input_tokens_call_count(), 1u);
    EXPECT_EQ(stage1_ptr->last_verifier_first_token(), 11);
    EXPECT_EQ(stage1_ptr->last_verifier_first_draft_slot(), 2);
    EXPECT_EQ(stage1_ptr->last_verifier_draft_token_count(), 3);
    EXPECT_EQ(stage1_ptr->last_verifier_total_input_tokens(), 4);

    const void *device_first_tokens =
        orchestrator->prepareMTPVerifierInputTokensOnDeviceFromDeviceFirstToken(
            /*first_target_sample_slot=*/1,
            /*first_draft_slot=*/0,
            /*draft_token_count=*/2,
            /*total_verifier_input_tokens=*/3);
    ASSERT_NE(device_first_tokens, nullptr);
    EXPECT_NE(device_first_tokens, host_first_tokens)
        << "The mock exposes separate stable buffers for host-first and "
           "device-first verifier token rows.";
    EXPECT_EQ(stage0_ptr->prepare_mtp_verifier_input_tokens_from_device_call_count(), 0u);
    EXPECT_EQ(stage1_ptr->prepare_mtp_verifier_input_tokens_from_device_call_count(), 1u);
    EXPECT_EQ(stage1_ptr->last_verifier_first_target_sample_slot(), 1);
    EXPECT_EQ(stage1_ptr->last_verifier_first_draft_slot(), 0);
    EXPECT_EQ(stage1_ptr->last_verifier_draft_token_count(), 2);
    EXPECT_EQ(stage1_ptr->last_verifier_total_input_tokens(), 3);

    const int32_t staged_drafts[] = {19, 23};
    EXPECT_TRUE(orchestrator->stageStochasticDraftTokensForDeviceVerification(
        staged_drafts,
        /*draft_token_count=*/2,
        /*first_draft_slot=*/1));
    EXPECT_EQ(stage0_ptr->stage_stochastic_draft_tokens_call_count(), 0u);
    EXPECT_EQ(stage1_ptr->stage_stochastic_draft_tokens_call_count(), 1u);
    EXPECT_EQ(stage1_ptr->last_staged_first_draft_slot(), 1);
    EXPECT_THAT(
        stage1_ptr->last_staged_draft_tokens(),
        ::testing::ElementsAre(19, 23));

    std::vector<LogitPenalty> penalties;
    penalties.push_back(LogitPenalty{7, -1.0f});
    EXPECT_TRUE(orchestrator->applyPenaltiesToAllPositionLogitsOnDeviceRow(
        /*row=*/1,
        penalties,
        /*vocab_size=*/32));
    EXPECT_EQ(stage0_ptr->apply_penalties_to_all_position_row_call_count(), 0u);
    EXPECT_EQ(stage1_ptr->apply_penalties_to_all_position_row_call_count(), 1u);

    SamplingParams params;
    params.temperature = 0.7f;
    params.top_k = 8;
    EXPECT_TRUE(orchestrator->buildStochasticProcessedLogitRowsOnDevice(
        DeviceLogitsSource::AllPosition,
        /*first_row=*/0,
        DeviceDistributionBuffer::Target,
        /*first_slot=*/0,
        /*row_count=*/2,
        params,
        /*vocab_size=*/32));
    EXPECT_EQ(stage0_ptr->build_stochastic_processed_rows_call_count(), 0u);
    EXPECT_EQ(stage1_ptr->build_stochastic_processed_rows_call_count(), 1u);

    EXPECT_EQ(orchestrator->sampleStochasticDraftProposalOnDevice(
                  DeviceLogitsSource::MTP,
                  /*row=*/0,
                  /*slot=*/1,
                  params,
                  /*vocab_size=*/32,
                  /*threshold=*/0.25f),
              23);
    EXPECT_EQ(stage0_ptr->sample_stochastic_draft_proposal_call_count(), 0u);
    EXPECT_EQ(stage1_ptr->sample_stochastic_draft_proposal_call_count(), 1u);

    DeviceSpeculativeVerifyBatchOutcome outcome;
    EXPECT_TRUE(orchestrator->verifyStochasticDistributionsBatchOutcomeOnDevice(
        /*first_target_slot=*/0,
        /*first_draft_slot=*/0,
        /*draft_tokens=*/nullptr,
        /*accept_thresholds=*/nullptr,
        /*residual_thresholds=*/nullptr,
        /*row_count=*/2,
        /*first_token=*/11,
        /*stop_tokens=*/nullptr,
        /*stop_token_count=*/0,
        /*bonus_target_slot=*/2,
        /*bonus_threshold=*/0.5f,
        &outcome,
        /*inverse_sample_seed=*/123,
        /*inverse_sample_first_logical_position=*/9,
        /*use_vllm_probability_rejection=*/true));
    EXPECT_EQ(stage0_ptr->verify_stochastic_batch_outcome_call_count(), 0u);
    EXPECT_EQ(stage1_ptr->verify_stochastic_batch_outcome_call_count(), 1u);
    EXPECT_TRUE(stage1_ptr->last_use_vllm_probability_rejection());
    EXPECT_EQ(stage1_ptr->last_stochastic_row_count(), 2);
}

TEST_F(Test__RankOrchestrator, ForwardMTPRunsOnEveryLocalTPChild)
{
    auto runner0 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner0_ptr = runner0.get();

    auto runner1 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner1_ptr = runner1.get();

    std::vector<std::unique_ptr<IInferenceRunner>> runners;
    runners.push_back(std::move(runner0));
    runners.push_back(std::move(runner1));

    auto orchestrator = RankOrchestrator::createForTest(
        llaminar2::test::MockModelContext::createMinimal(),
        std::move(runners),
        makeTPContextForRunnerCount(2),
        makeRankConfigForRunnerCount(2));

    EXPECT_TRUE(orchestrator->forwardMTP(42));
    EXPECT_EQ(runner0_ptr->forward_mtp_call_count(), 1u);
    EXPECT_EQ(runner1_ptr->forward_mtp_call_count(), 1u);
    EXPECT_EQ(runner0_ptr->last_mtp_condition_token(), 42);
    EXPECT_EQ(runner1_ptr->last_mtp_condition_token(), 42);
}

TEST_F(Test__RankOrchestrator, LocalTPSidecarStateReuseRequiresEveryChildSupport)
{
    auto runner0 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner0_ptr = runner0.get();
    runner0_ptr->set_supports_mtp_sidecar_preserves_main_state(true);
    runner0_ptr->set_supports_mtp_shifted_row_reuse_from_sidecar(true);

    auto runner1 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner1_ptr = runner1.get();
    runner1_ptr->set_supports_mtp_sidecar_preserves_main_state(true);
    runner1_ptr->set_supports_mtp_shifted_row_reuse_from_sidecar(true);

    std::vector<std::unique_ptr<IInferenceRunner>> runners;
    runners.push_back(std::move(runner0));
    runners.push_back(std::move(runner1));

    auto orchestrator = RankOrchestrator::createForTest(
        llaminar2::test::MockModelContext::createMinimal(),
        std::move(runners),
        makeTPContextForRunnerCount(2),
        makeRankConfigForRunnerCount(2));

    EXPECT_TRUE(orchestrator->supportsMTPSidecarPreservesMainState())
        << "LocalTP sidecar preservation is a domain-wide all-child capability.";
    EXPECT_TRUE(orchestrator->supportsMTPShiftedRowReuseFromSidecar())
        << "Dense LocalTP may reuse the first shifted row only when every shard can.";

    runner1_ptr->set_supports_mtp_sidecar_preserves_main_state(false);
    EXPECT_FALSE(orchestrator->supportsMTPSidecarPreservesMainState());
    EXPECT_TRUE(orchestrator->supportsMTPShiftedRowReuseFromSidecar())
        << "The reuse predicate is independent so tests can catch either "
           "capability drifting on a child runner.";

    runner1_ptr->set_supports_mtp_sidecar_preserves_main_state(true);
    runner0_ptr->set_supports_mtp_shifted_row_reuse_from_sidecar(false);
    EXPECT_TRUE(orchestrator->supportsMTPSidecarPreservesMainState());
    EXPECT_FALSE(orchestrator->supportsMTPShiftedRowReuseFromSidecar());
}

TEST_F(Test__RankOrchestrator, ForwardMTPEntersLocalTPChildrenConcurrently)
{
    auto rendezvous = std::make_shared<ForwardMTPRendezvous>(2);

    auto runner0 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner0_ptr = runner0.get();
    runner0_ptr->set_forward_mtp_rendezvous(rendezvous);

    auto runner1 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner1_ptr = runner1.get();
    runner1_ptr->set_forward_mtp_rendezvous(rendezvous);

    std::vector<std::unique_ptr<IInferenceRunner>> runners;
    runners.push_back(std::move(runner0));
    runners.push_back(std::move(runner1));

    auto orchestrator = RankOrchestrator::createForTest(
        llaminar2::test::MockModelContext::createMinimal(),
        std::move(runners),
        makeTPContextForRunnerCount(2),
        makeRankConfigForRunnerCount(2));

    EXPECT_TRUE(orchestrator->forwardMTP(99));
    EXPECT_EQ(rendezvous->arrivals.load(std::memory_order_acquire), 2);
    EXPECT_EQ(runner0_ptr->forward_mtp_call_count(), 1u);
    EXPECT_EQ(runner1_ptr->forward_mtp_call_count(), 1u);
    EXPECT_EQ(runner0_ptr->last_mtp_condition_token(), 99);
    EXPECT_EQ(runner1_ptr->last_mtp_condition_token(), 99);
}

TEST_F(Test__RankOrchestrator, ForwardMTPFailureStillAttemptsEveryLocalTPChild)
{
    auto runner0 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner0_ptr = runner0.get();
    runner0_ptr->set_forward_mtp_ok(false);

    auto runner1 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner1_ptr = runner1.get();

    std::vector<std::unique_ptr<IInferenceRunner>> runners;
    runners.push_back(std::move(runner0));
    runners.push_back(std::move(runner1));

    auto orchestrator = RankOrchestrator::createForTest(
        llaminar2::test::MockModelContext::createMinimal(),
        std::move(runners),
        makeTPContextForRunnerCount(2),
        makeRankConfigForRunnerCount(2));

    EXPECT_FALSE(orchestrator->forwardMTP(7));
    EXPECT_EQ(runner0_ptr->forward_mtp_call_count(), 1u);
    EXPECT_EQ(runner1_ptr->forward_mtp_call_count(), 1u);
    EXPECT_EQ(runner0_ptr->last_mtp_condition_token(), 7);
    EXPECT_EQ(runner1_ptr->last_mtp_condition_token(), 7);
}

TEST_F(Test__RankOrchestrator, ChainedMTPRequiresEveryLocalTPChildSupport)
{
    auto runner0 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner0_ptr = runner0.get();
    runner0_ptr->set_supports_chained_mtp_drafts(true);

    auto runner1 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner1_ptr = runner1.get();

    std::vector<std::unique_ptr<IInferenceRunner>> runners;
    runners.push_back(std::move(runner0));
    runners.push_back(std::move(runner1));

    auto orchestrator = RankOrchestrator::createForTest(
        llaminar2::test::MockModelContext::createMinimal(),
        std::move(runners),
        makeTPContextForRunnerCount(2),
        makeRankConfigForRunnerCount(2));

    EXPECT_FALSE(orchestrator->supportsChainedMTPDrafts());
    EXPECT_FALSE(orchestrator->forwardMTPFromLastDraft(55, 123));
    EXPECT_EQ(runner0_ptr->forward_mtp_from_last_draft_call_count(), 0u);
    EXPECT_EQ(runner1_ptr->forward_mtp_from_last_draft_call_count(), 0u);
}

TEST_F(Test__RankOrchestrator, ChainedMTPRunsOnEveryLocalTPChild)
{
    auto rendezvous = std::make_shared<ChainedMTPRendezvous>(2);

    auto runner0 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner0_ptr = runner0.get();
    runner0_ptr->set_supports_chained_mtp_drafts(true);
    runner0_ptr->set_chained_mtp_rendezvous(rendezvous);

    auto runner1 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner1_ptr = runner1.get();
    runner1_ptr->set_supports_chained_mtp_drafts(true);
    runner1_ptr->set_chained_mtp_rendezvous(rendezvous);

    std::vector<std::unique_ptr<IInferenceRunner>> runners;
    runners.push_back(std::move(runner0));
    runners.push_back(std::move(runner1));

    auto orchestrator = RankOrchestrator::createForTest(
        llaminar2::test::MockModelContext::createMinimal(),
        std::move(runners),
        makeTPContextForRunnerCount(2),
        makeRankConfigForRunnerCount(2));

    EXPECT_TRUE(orchestrator->supportsChainedMTPDrafts());
    EXPECT_TRUE(orchestrator->forwardMTPFromLastDraft(56, 124));
    EXPECT_EQ(rendezvous->arrivals.load(std::memory_order_acquire), 2);
    EXPECT_EQ(runner0_ptr->forward_mtp_from_last_draft_call_count(), 1u);
    EXPECT_EQ(runner1_ptr->forward_mtp_from_last_draft_call_count(), 1u);
    EXPECT_EQ(runner0_ptr->last_chained_mtp_condition_token(), 56);
    EXPECT_EQ(runner1_ptr->last_chained_mtp_condition_token(), 56);
    EXPECT_EQ(runner0_ptr->last_chained_mtp_position_id(), 124);
    EXPECT_EQ(runner1_ptr->last_chained_mtp_position_id(), 124);
}

TEST_F(Test__RankOrchestrator, ChainedMTPFailureStillAttemptsEveryLocalTPChild)
{
    auto runner0 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner0_ptr = runner0.get();
    runner0_ptr->set_supports_chained_mtp_drafts(true);
    runner0_ptr->set_forward_mtp_from_last_draft_ok(false);

    auto runner1 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner1_ptr = runner1.get();
    runner1_ptr->set_supports_chained_mtp_drafts(true);

    std::vector<std::unique_ptr<IInferenceRunner>> runners;
    runners.push_back(std::move(runner0));
    runners.push_back(std::move(runner1));

    auto orchestrator = RankOrchestrator::createForTest(
        llaminar2::test::MockModelContext::createMinimal(),
        std::move(runners),
        makeTPContextForRunnerCount(2),
        makeRankConfigForRunnerCount(2));

    EXPECT_FALSE(orchestrator->forwardMTPFromLastDraft(57, 125));
    EXPECT_EQ(runner0_ptr->forward_mtp_from_last_draft_call_count(), 1u);
    EXPECT_EQ(runner1_ptr->forward_mtp_from_last_draft_call_count(), 1u);
    EXPECT_EQ(runner0_ptr->last_chained_mtp_condition_token(), 57);
    EXPECT_EQ(runner1_ptr->last_chained_mtp_condition_token(), 57);
    EXPECT_EQ(runner0_ptr->last_chained_mtp_position_id(), 125);
    EXPECT_EQ(runner1_ptr->last_chained_mtp_position_id(), 125);
}

TEST_F(Test__RankOrchestrator, LocalTPAllPositionRowBatchSamplingConsumesVerifierStreamsOnce)
{
    auto runner0 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner0_ptr = runner0.get();
    runner0_ptr->set_mock_all_position_logits_local(
        /*rows=*/2,
        /*local_vocab=*/3,
        {
            0.0f, 5.0f, 1.0f,
            1.0f, 2.0f, 3.0f,
        });

    auto runner1 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner1_ptr = runner1.get();
    runner1_ptr->set_mock_all_position_logits_local(
        /*rows=*/2,
        /*local_vocab=*/3,
        {
            2.0f, 3.0f, 4.0f,
            9.0f, 0.0f, 0.0f,
        });

    std::vector<std::unique_ptr<IInferenceRunner>> runners;
    runners.push_back(std::move(runner0));
    runners.push_back(std::move(runner1));

    auto orchestrator = RankOrchestrator::createForTest(
        llaminar2::test::MockModelContext::createMinimal(),
        std::move(runners),
        makeTPContextForRunnerCount(2),
        makeRankConfigForRunnerCount(2));

    std::array<int32_t, 2> sampled = {-1, -1};
    ASSERT_TRUE(orchestrator->sampleGreedyFromAllPositionLogitsOnDeviceRows(
        /*start_row=*/0,
        /*row_count=*/static_cast<int>(sampled.size()),
        sampled.data()));

    EXPECT_EQ(sampled[0], 1)
        << "row 0 should choose shard 0 local column 1";
    EXPECT_EQ(sampled[1], 3)
        << "row 1 should choose shard 1 local column 0 with shard offset";
    EXPECT_EQ(runner0_ptr->consume_all_position_logits_local_info_call_count(), 1u);
    EXPECT_EQ(runner1_ptr->consume_all_position_logits_local_info_call_count(), 1u);
    EXPECT_EQ(runner0_ptr->get_all_position_logits_local_info_call_count(), 0u);
    EXPECT_EQ(runner1_ptr->get_all_position_logits_local_info_call_count(), 0u);
}

TEST_F(Test__RankOrchestrator, LocalTPDoesNotAdvertiseCompactGreedyVerifierOutcome)
{
    auto runner0 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto runner1 = std::make_unique<MockDeviceGraphOrchestrator>();

    std::vector<std::unique_ptr<IInferenceRunner>> runners;
    runners.push_back(std::move(runner0));
    runners.push_back(std::move(runner1));

    auto orchestrator = RankOrchestrator::createForTest(
        llaminar2::test::MockModelContext::createMinimal(),
        std::move(runners),
        makeTPContextForRunnerCount(2),
        makeRankConfigForRunnerCount(2));

    EXPECT_FALSE(orchestrator->supportsGreedyAllPositionBatchOutcomeOnDevice())
        << "LocalTP verifier logits are sharded. The rank may sample rows "
           "from child-local shards, but it must not advertise one compact "
           "device reducer until a true cross-participant reducer exists.";
}

TEST_F(Test__RankOrchestrator, SpecStatePublicationRequiresEveryLocalTPChildSupport)
{
    auto runner0 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner0_ptr = runner0.get();
    runner0_ptr->set_supports_mtp_spec_state_publication(true);

    auto runner1 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner1_ptr = runner1.get();

    std::vector<std::unique_ptr<IInferenceRunner>> runners;
    runners.push_back(std::move(runner0));
    runners.push_back(std::move(runner1));

    auto orchestrator = RankOrchestrator::createForTest(
        llaminar2::test::MockModelContext::createMinimal(),
        std::move(runners),
        makeTPContextForRunnerCount(2),
        makeRankConfigForRunnerCount(2));

    EXPECT_FALSE(orchestrator->supportsMTPSpecStatePublication());

    std::string error;
    EXPECT_FALSE(orchestrator->publishAcceptedMTPSpecState(
        makeMTPSpecPublicationPlan(/*accepted_count=*/2),
        &error));
    EXPECT_NE(error.find("participant 1"), std::string::npos);
    EXPECT_EQ(runner0_ptr->publish_mtp_spec_state_call_count(), 0u);
    EXPECT_EQ(runner1_ptr->publish_mtp_spec_state_call_count(), 0u);
}

TEST_F(Test__RankOrchestrator, VerifierRowCapabilityClampsToWeakestParticipant)
{
    MTPVerifierRowCapability strong;
    strong.dense_decode_equivalent =
        MTPVerifierRowEquivalenceSpec::proven(4);
    strong.dense_direct_all_position =
        MTPVerifierRowEquivalenceSpec::proven(4);
    strong.moe_decode_equivalent =
        MTPVerifierRowEquivalenceSpec::proven(4);
    strong.device_resident_direct_publication = true;

    MTPVerifierRowCapability weak = strong;
    weak.dense_direct_all_position =
        MTPVerifierRowEquivalenceSpec::proven(2);
    weak.moe_decode_equivalent =
        MTPVerifierRowEquivalenceSpec::proven(3);
    weak.device_resident_direct_publication = false;

    auto runner0 = std::make_unique<MockDeviceGraphOrchestrator>();
    runner0->set_mtp_verifier_row_capability(strong);

    auto runner1 = std::make_unique<MockDeviceGraphOrchestrator>();
    runner1->set_mtp_verifier_row_capability(weak);

    std::vector<std::unique_ptr<IInferenceRunner>> runners;
    runners.push_back(std::move(runner0));
    runners.push_back(std::move(runner1));

    auto orchestrator = RankOrchestrator::createForTest(
        llaminar2::test::MockModelContext::createMinimal(),
        std::move(runners),
        makeTPContextForRunnerCount(2),
        makeRankConfigForRunnerCount(2));

    const MTPVerifierRowCapability capability =
        orchestrator->mtpVerifierRowCapability();
    EXPECT_TRUE(capability.supportsDenseDecodeEquivalentRows(4, true));
    EXPECT_TRUE(capability.supportsDenseDirectAllPositionRows(2, true));
    EXPECT_FALSE(capability.supportsDenseDirectAllPositionRows(3, false))
        << "Rank-level direct publication must clamp to the weakest child.";
    EXPECT_TRUE(capability.supportsMoEDecodeEquivalentRows(3, true));
    EXPECT_FALSE(capability.supportsMoEDecodeEquivalentRows(4, false));
    EXPECT_FALSE(capability.device_resident_direct_publication)
        << "Device-resident publication is an all-participant contract.";
}

TEST_F(Test__RankOrchestrator, VerifierEconomyCapabilityClampsToWeakestParticipant)
{
    MTPVerifierEconomyCapability strong;
    strong.dense = MTPVerifierEconomyLane::groupedPromoted(4);
    strong.moe = MTPVerifierEconomyLane::groupedPromoted(4);

    MTPVerifierEconomyCapability weak;
    weak.dense = MTPVerifierEconomyLane::serialFallbackCorrect(4);
    weak.moe = MTPVerifierEconomyLane::groupedPromoted(2);
    weak.moe.host_bridge_free_hot_path = false;
    weak.moe.perf_gate_status = "device_bridge_pending";

    auto runner0 = std::make_unique<MockDeviceGraphOrchestrator>();
    runner0->set_mtp_verifier_economy_capability(strong);

    auto runner1 = std::make_unique<MockDeviceGraphOrchestrator>();
    runner1->set_mtp_verifier_economy_capability(weak);

    std::vector<std::unique_ptr<IInferenceRunner>> runners;
    runners.push_back(std::move(runner0));
    runners.push_back(std::move(runner1));

    auto orchestrator = RankOrchestrator::createForTest(
        llaminar2::test::MockModelContext::createMinimal(),
        std::move(runners),
        makeTPContextForRunnerCount(2),
        makeRankConfigForRunnerCount(2));

    const MTPVerifierEconomyCapability capability =
        orchestrator->mtpVerifierEconomyCapability();
    EXPECT_TRUE(capability.supportsDenseRows(4, true));
    EXPECT_FALSE(capability.hasEconomicalDensePath(4, false))
        << "A correct serial fallback must not be advertised as an "
           "economical grouped verifier.";
    EXPECT_TRUE(capability.supportsMoERows(2, true));
    EXPECT_FALSE(capability.supportsMoERows(3, false))
        << "Rank-level MoE economy must clamp row count to the weakest child.";
    EXPECT_FALSE(capability.hasEconomicalMoEPath(2, true))
        << "Every participant must be host-bridge free before the rank path is "
           "economical.";
    EXPECT_EQ(capability.moe.perf_gate_status, "mixed_capability");
}

TEST_F(Test__RankOrchestrator, SpecStatePublicationRunsOnEveryLocalTPChild)
{
    auto rendezvous = std::make_shared<MTPPublicationRendezvous>(2);

    auto runner0 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner0_ptr = runner0.get();
    runner0_ptr->set_supports_mtp_spec_state_publication(true);
    runner0_ptr->set_mtp_publication_rendezvous(rendezvous);

    auto runner1 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner1_ptr = runner1.get();
    runner1_ptr->set_supports_mtp_spec_state_publication(true);
    runner1_ptr->set_mtp_publication_rendezvous(rendezvous);

    std::vector<std::unique_ptr<IInferenceRunner>> runners;
    runners.push_back(std::move(runner0));
    runners.push_back(std::move(runner1));

    auto orchestrator = RankOrchestrator::createForTest(
        llaminar2::test::MockModelContext::createMinimal(),
        std::move(runners),
        makeTPContextForRunnerCount(2),
        makeRankConfigForRunnerCount(2));

    MTPSpecStepPlan plan = makeMTPSpecPublicationPlan(/*accepted_count=*/2);
    std::string error;
    EXPECT_TRUE(orchestrator->supportsMTPSpecStatePublication());
    EXPECT_TRUE(orchestrator->publishAcceptedMTPSpecState(plan, &error))
        << error;

    EXPECT_EQ(rendezvous->arrivals.load(std::memory_order_acquire), 2);
    EXPECT_EQ(runner0_ptr->publish_mtp_spec_state_call_count(), 1u);
    EXPECT_EQ(runner1_ptr->publish_mtp_spec_state_call_count(), 1u);
    EXPECT_EQ(runner0_ptr->last_mtp_spec_state_plan().accepted_count, 2);
    EXPECT_EQ(runner1_ptr->last_mtp_spec_state_plan().accepted_count, 2);
    EXPECT_EQ(runner0_ptr->get_position(), 66);
    EXPECT_EQ(runner1_ptr->get_position(), 66);
}

TEST_F(Test__RankOrchestrator, SpecStatePublicationFailureStillAttemptsEveryLocalTPChild)
{
    auto runner0 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner0_ptr = runner0.get();
    runner0_ptr->set_supports_mtp_spec_state_publication(true);
    runner0_ptr->set_publish_mtp_spec_state_ok(false);

    auto runner1 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner1_ptr = runner1.get();
    runner1_ptr->set_supports_mtp_spec_state_publication(true);

    std::vector<std::unique_ptr<IInferenceRunner>> runners;
    runners.push_back(std::move(runner0));
    runners.push_back(std::move(runner1));

    auto orchestrator = RankOrchestrator::createForTest(
        llaminar2::test::MockModelContext::createMinimal(),
        std::move(runners),
        makeTPContextForRunnerCount(2),
        makeRankConfigForRunnerCount(2));

    std::string error;
    EXPECT_FALSE(orchestrator->publishAcceptedMTPSpecState(
        makeMTPSpecPublicationPlan(/*accepted_count=*/1),
        &error));
    EXPECT_NE(error.find("participant 0"), std::string::npos);
    EXPECT_EQ(runner0_ptr->publish_mtp_spec_state_call_count(), 1u);
    EXPECT_EQ(runner1_ptr->publish_mtp_spec_state_call_count(), 1u);
}

TEST_F(Test__RankOrchestrator, SpecStateBatchPublicationRequiresEveryLocalTPChildSupport)
{
    auto runner0 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner0_ptr = runner0.get();
    runner0_ptr->set_supports_mtp_spec_state_publication(true);

    auto runner1 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner1_ptr = runner1.get();

    std::vector<std::unique_ptr<IInferenceRunner>> runners;
    runners.push_back(std::move(runner0));
    runners.push_back(std::move(runner1));

    auto orchestrator = RankOrchestrator::createForTest(
        llaminar2::test::MockModelContext::createMinimal(),
        std::move(runners),
        makeTPContextForRunnerCount(2),
        makeRankConfigForRunnerCount(2));

    std::string error;
    EXPECT_FALSE(orchestrator->publishAcceptedMTPSpecStateBatch(
        makeMTPSpecPublicationBatch(),
        &error));
    EXPECT_NE(error.find("participant 1"), std::string::npos);
    EXPECT_EQ(runner0_ptr->publish_mtp_spec_state_batch_call_count(), 0u);
    EXPECT_EQ(runner1_ptr->publish_mtp_spec_state_batch_call_count(), 0u);
}

TEST_F(Test__RankOrchestrator, SpecStateBatchPublicationRunsOnEveryLocalTPChild)
{
    auto rendezvous = std::make_shared<MTPPublicationRendezvous>(2);

    auto runner0 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner0_ptr = runner0.get();
    runner0_ptr->set_supports_mtp_spec_state_publication(true);
    runner0_ptr->set_mtp_publication_rendezvous(rendezvous);

    auto runner1 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner1_ptr = runner1.get();
    runner1_ptr->set_supports_mtp_spec_state_publication(true);
    runner1_ptr->set_mtp_publication_rendezvous(rendezvous);

    std::vector<std::unique_ptr<IInferenceRunner>> runners;
    runners.push_back(std::move(runner0));
    runners.push_back(std::move(runner1));

    auto orchestrator = RankOrchestrator::createForTest(
        llaminar2::test::MockModelContext::createMinimal(),
        std::move(runners),
        makeTPContextForRunnerCount(2),
        makeRankConfigForRunnerCount(2));

    MTPSpecStepPlanBatch batch = makeMTPSpecPublicationBatch();
    std::string error;
    EXPECT_TRUE(orchestrator->publishAcceptedMTPSpecStateBatch(batch, &error))
        << error;

    EXPECT_EQ(rendezvous->arrivals.load(std::memory_order_acquire), 2);
    EXPECT_EQ(runner0_ptr->publish_mtp_spec_state_call_count(), 0u);
    EXPECT_EQ(runner1_ptr->publish_mtp_spec_state_call_count(), 0u);
    EXPECT_EQ(runner0_ptr->publish_mtp_spec_state_batch_call_count(), 1u);
    EXPECT_EQ(runner1_ptr->publish_mtp_spec_state_batch_call_count(), 1u);
    ASSERT_THAT(runner0_ptr->last_mtp_spec_state_batch().steps, ::testing::SizeIs(2));
    ASSERT_THAT(runner1_ptr->last_mtp_spec_state_batch().steps, ::testing::SizeIs(2));
    EXPECT_EQ(runner0_ptr->last_mtp_spec_state_batch().request_count, 2);
    EXPECT_EQ(runner1_ptr->last_mtp_spec_state_batch().request_count, 2);
    EXPECT_EQ(runner0_ptr->last_mtp_spec_state_batch().steps[0].request_id, 101);
    EXPECT_EQ(runner0_ptr->last_mtp_spec_state_batch().steps[1].request_id, 102);
    EXPECT_EQ(runner1_ptr->last_mtp_spec_state_batch().steps[0].accepted_count, 2);
    EXPECT_EQ(runner1_ptr->last_mtp_spec_state_batch().steps[1].accepted_count, 1);
    EXPECT_EQ(runner0_ptr->get_position(), 81);
    EXPECT_EQ(runner1_ptr->get_position(), 81);
}

TEST_F(Test__RankOrchestrator, SpecStateBatchPublicationFailureStillAttemptsEveryLocalTPChild)
{
    auto runner0 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner0_ptr = runner0.get();
    runner0_ptr->set_supports_mtp_spec_state_publication(true);
    runner0_ptr->set_publish_mtp_spec_state_ok(false);

    auto runner1 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner1_ptr = runner1.get();
    runner1_ptr->set_supports_mtp_spec_state_publication(true);

    std::vector<std::unique_ptr<IInferenceRunner>> runners;
    runners.push_back(std::move(runner0));
    runners.push_back(std::move(runner1));

    auto orchestrator = RankOrchestrator::createForTest(
        llaminar2::test::MockModelContext::createMinimal(),
        std::move(runners),
        makeTPContextForRunnerCount(2),
        makeRankConfigForRunnerCount(2));

    std::string error;
    EXPECT_FALSE(orchestrator->publishAcceptedMTPSpecStateBatch(
        makeMTPSpecPublicationBatch(),
        &error));
    EXPECT_NE(error.find("participant 0"), std::string::npos);
    EXPECT_EQ(runner0_ptr->publish_mtp_spec_state_batch_call_count(), 1u);
    EXPECT_EQ(runner1_ptr->publish_mtp_spec_state_batch_call_count(), 1u);
}

TEST_F(Test__RankOrchestrator, AllPositionLogitToggleRunsOnEveryLocalTPChild)
{
    auto runner0 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner0_ptr = runner0.get();

    auto runner1 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner1_ptr = runner1.get();

    std::vector<std::unique_ptr<IInferenceRunner>> runners;
    runners.push_back(std::move(runner0));
    runners.push_back(std::move(runner1));

    auto orchestrator = RankOrchestrator::createForTest(
        llaminar2::test::MockModelContext::createMinimal(),
        std::move(runners),
        makeTPContextForRunnerCount(2),
        makeRankConfigForRunnerCount(2));

    EXPECT_TRUE(orchestrator->setComputeAllPositionLogits(true));
    EXPECT_TRUE(runner0_ptr->compute_all_position_logits());
    EXPECT_TRUE(runner1_ptr->compute_all_position_logits());
    EXPECT_EQ(runner0_ptr->set_all_position_logits_call_count(), 1u);
    EXPECT_EQ(runner1_ptr->set_all_position_logits_call_count(), 1u);

    EXPECT_TRUE(orchestrator->setComputeAllPositionLogits(false));
    EXPECT_FALSE(runner0_ptr->compute_all_position_logits());
    EXPECT_FALSE(runner1_ptr->compute_all_position_logits());
    EXPECT_EQ(runner0_ptr->set_all_position_logits_call_count(), 2u);
    EXPECT_EQ(runner1_ptr->set_all_position_logits_call_count(), 2u);
}

TEST_F(Test__RankOrchestrator, MultiChildMTPDecodePropagatesChildTopologyBypass)
{
    auto runner0 = std::make_unique<MockDeviceGraphOrchestrator>();
    runner0->set_mtp_unsupported_reason("MTP decode all-position logits are not enabled for column-parallel LM head");

    std::vector<std::unique_ptr<IInferenceRunner>> runners;
    runners.push_back(std::move(runner0));
    runners.push_back(std::make_unique<MockDeviceGraphOrchestrator>());

    auto orchestrator = RankOrchestrator::createForTest(
        llaminar2::test::MockModelContext::createMinimal(),
        std::move(runners),
        makeTPContextForRunnerCount(2),
        makeRankConfigForRunnerCount(2));

    const std::string reason = orchestrator->mtpDecodeUnsupportedReason();
    EXPECT_NE(reason.find("column-parallel"), std::string::npos);
}

TEST_F(Test__RankOrchestrator, MultiChildMTPDecodeAllowsReplicatedLogitTopology)
{
    std::vector<std::unique_ptr<IInferenceRunner>> runners;
    runners.push_back(std::make_unique<MockDeviceGraphOrchestrator>());
    runners.push_back(std::make_unique<MockDeviceGraphOrchestrator>());

    auto orchestrator = RankOrchestrator::createForTest(
        llaminar2::test::MockModelContext::createMinimal(),
        std::move(runners),
        makeTPContextForRunnerCount(2),
        makeRankConfigForRunnerCount(2));

    EXPECT_TRUE(orchestrator->mtpDecodeUnsupportedReason().empty());
}

TEST_F(Test__RankOrchestrator, MultiChildMTPLogitsRequireMatchingReplicas)
{
    auto runner0 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner0_ptr = runner0.get();
    runner0_ptr->set_vocab_size(3);
    runner0_ptr->set_mock_mtp_logits({0.0f, 5.0f, 1.0f});

    auto runner1 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner1_ptr = runner1.get();
    runner1_ptr->set_vocab_size(3);
    runner1_ptr->set_mock_mtp_logits({0.0f, 5.0f, 1.0f});

    std::vector<std::unique_ptr<IInferenceRunner>> runners;
    runners.push_back(std::move(runner0));
    runners.push_back(std::move(runner1));

    auto orchestrator = RankOrchestrator::createForTest(
        llaminar2::test::MockModelContext::createMinimal(),
        std::move(runners),
        makeTPContextForRunnerCount(2),
        makeRankConfigForRunnerCount(2));

    ASSERT_NE(orchestrator->mtpLogits(), nullptr);
    EXPECT_FLOAT_EQ(orchestrator->mtpLogits()[1], 5.0f);

    runner1_ptr->set_mock_mtp_logits({0.0f, 4.0f, 1.0f});
    EXPECT_EQ(orchestrator->mtpLogits(), nullptr);
}

TEST_F(Test__RankOrchestrator, MultiChildMTPLogitsGatherColumnParallelShards)
{
    auto runner0 = std::make_unique<MockDeviceGraphOrchestrator>();
    runner0->set_mock_mtp_logits_local(2, {0.0f, 5.0f});

    auto runner1 = std::make_unique<MockDeviceGraphOrchestrator>();
    runner1->set_mock_mtp_logits_local(3, {1.0f, 9.0f, 2.0f});

    std::vector<std::unique_ptr<IInferenceRunner>> runners;
    runners.push_back(std::move(runner0));
    runners.push_back(std::move(runner1));

    auto model_ctx = llaminar2::test::MockModelContext::createMinimal();
    model_ctx->setVocabSize(5);

    auto orchestrator = RankOrchestrator::createForTest(
        std::move(model_ctx),
        std::move(runners),
        makeTPContextForRunnerCount(2),
        makeRankConfigForRunnerCount(2));

    const float *logits = orchestrator->mtpLogits();
    ASSERT_NE(logits, nullptr);
    EXPECT_FLOAT_EQ(logits[0], 0.0f);
    EXPECT_FLOAT_EQ(logits[1], 5.0f);
    EXPECT_FLOAT_EQ(logits[2], 1.0f);
    EXPECT_FLOAT_EQ(logits[3], 9.0f);
    EXPECT_FLOAT_EQ(logits[4], 2.0f);
}

TEST_F(Test__RankOrchestrator, MultiChildMTPSamplingUsesColumnParallelShardInfos)
{
    auto runner0 = std::make_unique<MockDeviceGraphOrchestrator>();
    runner0->set_mock_mtp_logits_local(2, {0.1f, 0.4f});
    auto *runner0_ptr = runner0.get();

    auto runner1 = std::make_unique<MockDeviceGraphOrchestrator>();
    runner1->set_mock_mtp_logits_local(3, {0.8f, 0.7f, 0.6f});
    auto *runner1_ptr = runner1.get();

    std::vector<std::unique_ptr<IInferenceRunner>> runners;
    runners.push_back(std::move(runner0));
    runners.push_back(std::move(runner1));

    auto model_ctx = llaminar2::test::MockModelContext::createMinimal();
    model_ctx->setVocabSize(5);

    auto orchestrator = RankOrchestrator::createForTest(
        std::move(model_ctx),
        std::move(runners),
        makeTPContextForRunnerCount(2),
        makeRankConfigForRunnerCount(2));

    EXPECT_EQ(orchestrator->sampleGreedyFromMTPLogitsOnDevice(), 2);
    EXPECT_EQ(runner0_ptr->consume_mtp_logits_local_info_call_count(), 1u);
    EXPECT_EQ(runner1_ptr->consume_mtp_logits_local_info_call_count(), 1u);
    EXPECT_EQ(runner0_ptr->get_mtp_logits_local_info_call_count(), 0u);
    EXPECT_EQ(runner1_ptr->get_mtp_logits_local_info_call_count(), 0u);
}

TEST_F(Test__RankOrchestrator, MultiChildMTPPenaltyApplicationFansOutToEveryShard)
{
    auto runner0 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner0_ptr = runner0.get();

    auto runner1 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner1_ptr = runner1.get();

    std::vector<std::unique_ptr<IInferenceRunner>> runners;
    runners.push_back(std::move(runner0));
    runners.push_back(std::move(runner1));

    auto model_ctx = llaminar2::test::MockModelContext::createMinimal();
    model_ctx->setVocabSize(5);

    auto orchestrator = RankOrchestrator::createForTest(
        std::move(model_ctx),
        std::move(runners),
        makeTPContextForRunnerCount(2),
        makeRankConfigForRunnerCount(2));

    const std::vector<LogitPenalty> penalties = {{2, 1.25f}};
    EXPECT_TRUE(orchestrator->applyPenaltiesToMTPLogitsOnDevice(penalties, 5))
        << "LocalTP MTP logits are sharded; rank orchestration must ask every "
           "participant to apply its local slice of the global sparse map.";
    EXPECT_EQ(runner0_ptr->apply_penalties_to_mtp_logits_call_count(), 1u);
    EXPECT_EQ(runner1_ptr->apply_penalties_to_mtp_logits_call_count(), 1u);

    EXPECT_TRUE(orchestrator->applyPenaltiesOnDevice(penalties, 5))
        << "The main LM-head penalty path should mirror MTP so temperature-zero "
           "requests with model-default penalties remain TP-capable.";
    EXPECT_EQ(runner0_ptr->apply_penalties_on_device_call_count(), 1u);
    EXPECT_EQ(runner1_ptr->apply_penalties_on_device_call_count(), 1u);
}

TEST_F(Test__RankOrchestrator, MultiChildMTPLogitsRejectMixedLocalAndReplicatedShards)
{
    auto runner0 = std::make_unique<MockDeviceGraphOrchestrator>();
    runner0->set_mock_mtp_logits_local(2, {0.0f, 5.0f});

    auto runner1 = std::make_unique<MockDeviceGraphOrchestrator>();
    runner1->set_vocab_size(3);
    runner1->set_mock_mtp_logits({1.0f, 9.0f, 2.0f});

    std::vector<std::unique_ptr<IInferenceRunner>> runners;
    runners.push_back(std::move(runner0));
    runners.push_back(std::move(runner1));

    auto model_ctx = llaminar2::test::MockModelContext::createMinimal();
    model_ctx->setVocabSize(5);

    auto orchestrator = RankOrchestrator::createForTest(
        std::move(model_ctx),
        std::move(runners),
        makeTPContextForRunnerCount(2),
        makeRankConfigForRunnerCount(2));

    EXPECT_EQ(orchestrator->mtpLogits(), nullptr);
}

TEST_F(Test__RankOrchestrator, MultiChildAllPositionLogitsRequireMatchingReplicas)
{
    auto runner0 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner0_ptr = runner0.get();
    runner0_ptr->set_vocab_size(3);
    runner0_ptr->set_mock_all_position_logits({1.0f, 2.0f, 3.0f});

    auto runner1 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner1_ptr = runner1.get();
    runner1_ptr->set_vocab_size(3);
    runner1_ptr->set_mock_all_position_logits({1.0f, 2.0f, 3.0f});

    std::vector<std::unique_ptr<IInferenceRunner>> runners;
    runners.push_back(std::move(runner0));
    runners.push_back(std::move(runner1));

    auto orchestrator = RankOrchestrator::createForTest(
        llaminar2::test::MockModelContext::createMinimal(),
        std::move(runners),
        makeTPContextForRunnerCount(2),
        makeRankConfigForRunnerCount(2));

    ASSERT_NE(orchestrator->getAllPositionLogits(), nullptr);
    EXPECT_FLOAT_EQ(orchestrator->getAllPositionLogits()[2], 3.0f);

    runner1_ptr->set_mock_all_position_logits({1.0f, 2.5f, 3.0f});
    EXPECT_EQ(orchestrator->getAllPositionLogits(), nullptr);
}

TEST_F(Test__RankOrchestrator, MultiChildAllPositionLogitsGatherColumnParallelShards)
{
    auto runner0 = std::make_unique<MockDeviceGraphOrchestrator>();
    runner0->set_mock_all_position_logits_local(
        /*rows=*/2,
        /*local_vocab=*/2,
        {10.0f, 20.0f,
         30.0f, 40.0f});

    auto runner1 = std::make_unique<MockDeviceGraphOrchestrator>();
    runner1->set_mock_all_position_logits_local(
        /*rows=*/2,
        /*local_vocab=*/3,
        {1.0f, 2.0f, 3.0f,
         4.0f, 5.0f, 6.0f});

    std::vector<std::unique_ptr<IInferenceRunner>> runners;
    runners.push_back(std::move(runner0));
    runners.push_back(std::move(runner1));

    auto model_ctx = llaminar2::test::MockModelContext::createMinimal();
    model_ctx->setVocabSize(5);

    auto orchestrator = RankOrchestrator::createForTest(
        std::move(model_ctx),
        std::move(runners),
        makeTPContextForRunnerCount(2),
        makeRankConfigForRunnerCount(2));

    const int tokens[] = {11, 22};
    ASSERT_TRUE(orchestrator->forward(tokens, 2));

    const float *logits = orchestrator->getAllPositionLogits();
    ASSERT_NE(logits, nullptr);
    EXPECT_FLOAT_EQ(logits[0], 10.0f);
    EXPECT_FLOAT_EQ(logits[1], 20.0f);
    EXPECT_FLOAT_EQ(logits[2], 1.0f);
    EXPECT_FLOAT_EQ(logits[3], 2.0f);
    EXPECT_FLOAT_EQ(logits[4], 3.0f);
    EXPECT_FLOAT_EQ(logits[5], 30.0f);
    EXPECT_FLOAT_EQ(logits[6], 40.0f);
    EXPECT_FLOAT_EQ(logits[7], 4.0f);
    EXPECT_FLOAT_EQ(logits[8], 5.0f);
    EXPECT_FLOAT_EQ(logits[9], 6.0f);
}

TEST_F(Test__RankOrchestrator, MultiChildAllPositionSamplingUsesRequestedColumnParallelShardRow)
{
    auto runner0 = std::make_unique<MockDeviceGraphOrchestrator>();
    runner0->set_mock_all_position_logits_local(
        /*rows=*/2,
        /*local_vocab=*/2,
        {0.1f, 0.2f,
         0.3f, 6.0f});

    auto runner1 = std::make_unique<MockDeviceGraphOrchestrator>();
    runner1->set_mock_all_position_logits_local(
        /*rows=*/2,
        /*local_vocab=*/3,
        {0.5f, 0.4f, 0.1f,
         5.0f, 0.1f, 0.2f});

    std::vector<std::unique_ptr<IInferenceRunner>> runners;
    runners.push_back(std::move(runner0));
    runners.push_back(std::move(runner1));

    auto model_ctx = llaminar2::test::MockModelContext::createMinimal();
    model_ctx->setVocabSize(5);

    auto orchestrator = RankOrchestrator::createForTest(
        std::move(model_ctx),
        std::move(runners),
        makeTPContextForRunnerCount(2),
        makeRankConfigForRunnerCount(2));

    EXPECT_EQ(orchestrator->sampleGreedyFromAllPositionLogitsOnDevice(0), 2);
    EXPECT_EQ(orchestrator->sampleGreedyFromAllPositionLogitsOnDevice(1), 1);
    EXPECT_EQ(orchestrator->sampleGreedyFromAllPositionLogitsOnDevice(2), -1);
}

TEST_F(Test__RankOrchestrator, MultiChildAllPositionLogitsRejectMixedLocalAndReplicatedShards)
{
    auto runner0 = std::make_unique<MockDeviceGraphOrchestrator>();
    runner0->set_mock_all_position_logits_local(/*rows=*/1, /*local_vocab=*/2, {10.0f, 20.0f});

    auto runner1 = std::make_unique<MockDeviceGraphOrchestrator>();
    runner1->set_vocab_size(3);
    runner1->set_mock_all_position_logits({1.0f, 2.0f, 3.0f});

    std::vector<std::unique_ptr<IInferenceRunner>> runners;
    runners.push_back(std::move(runner0));
    runners.push_back(std::move(runner1));

    auto model_ctx = llaminar2::test::MockModelContext::createMinimal();
    model_ctx->setVocabSize(5);

    auto orchestrator = RankOrchestrator::createForTest(
        std::move(model_ctx),
        std::move(runners),
        makeTPContextForRunnerCount(2),
        makeRankConfigForRunnerCount(2));

    EXPECT_EQ(orchestrator->getAllPositionLogits(), nullptr);
}

TEST_F(Test__RankOrchestrator, SingleChildMTPDelegatesWithoutTopologyBypass)
{
    auto runner0 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner0_ptr = runner0.get();
    runner0_ptr->set_mock_mtp_logits({0.0f, 1.0f});
    runner0_ptr->set_mock_all_position_logits({2.0f, 3.0f});

    std::vector<std::unique_ptr<IInferenceRunner>> runners;
    runners.push_back(std::move(runner0));

    auto orchestrator = RankOrchestrator::createForTest(
        llaminar2::test::MockModelContext::createMinimal(),
        std::move(runners),
        makeTPContextForRunnerCount(1),
        makeRankConfigForRunnerCount(1));

    EXPECT_TRUE(orchestrator->mtpDecodeUnsupportedReason().empty());
    EXPECT_TRUE(orchestrator->forwardMTP(5));
    EXPECT_EQ(runner0_ptr->forward_mtp_call_count(), 1u);
    ASSERT_NE(orchestrator->mtpLogits(), nullptr);
    EXPECT_FLOAT_EQ(orchestrator->mtpLogits()[1], 1.0f);
    EXPECT_TRUE(orchestrator->setComputeAllPositionLogits(true));
    ASSERT_NE(orchestrator->getAllPositionLogits(), nullptr);
    EXPECT_FLOAT_EQ(orchestrator->getAllPositionLogits()[0], 2.0f);
}

TEST_F(Test__RankOrchestrator, LivePrefixCheckpointCapturesTruncatesAndRestoresEveryLocalTPChild)
{
    auto runner0 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner0_ptr = runner0.get();
    auto runner1 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner1_ptr = runner1.get();

    int tokens[] = {1, 2, 3};
    ASSERT_TRUE(runner0_ptr->forward(tokens, 3));
    ASSERT_TRUE(runner1_ptr->forward(tokens, 3));

    std::vector<std::unique_ptr<IInferenceRunner>> runners;
    runners.push_back(std::move(runner0));
    runners.push_back(std::move(runner1));

    auto orchestrator = RankOrchestrator::createForTest(
        llaminar2::test::MockModelContext::createMinimal(),
        std::move(runners),
        makeTPContextForRunnerCount(2),
        makeRankConfigForRunnerCount(2));

    PrefixStateSnapshot snapshot = orchestrator->captureLivePrefixState();
    ASSERT_TRUE(snapshot.valid);
    EXPECT_EQ(snapshot.cached_tokens, 3);
    ASSERT_EQ(snapshot.participant_snapshots.size(), 2u);
    EXPECT_EQ(runner0_ptr->prefix_live_capture_call_count(), 1u);
    EXPECT_EQ(runner1_ptr->prefix_live_capture_call_count(), 1u);

    ASSERT_TRUE(orchestrator->truncateLivePrefixState(1));
    EXPECT_EQ(runner0_ptr->get_position(), 1);
    EXPECT_EQ(runner1_ptr->get_position(), 1);
    EXPECT_EQ(runner0_ptr->prefix_live_truncate_call_count(), 1u);
    EXPECT_EQ(runner1_ptr->prefix_live_truncate_call_count(), 1u);

    ASSERT_TRUE(orchestrator->restoreLivePrefixState(snapshot));
    EXPECT_EQ(runner0_ptr->get_position(), 3);
    EXPECT_EQ(runner1_ptr->get_position(), 3);
    EXPECT_EQ(runner0_ptr->prefix_live_restore_call_count(), 1u);
    EXPECT_EQ(runner1_ptr->prefix_live_restore_call_count(), 1u);
}

TEST_F(Test__RankOrchestrator, LivePrefixCheckpointRejectsDivergentChildPositions)
{
    auto runner0 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner0_ptr = runner0.get();
    auto runner1 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner1_ptr = runner1.get();

    int tokens0[] = {1, 2, 3};
    int tokens1[] = {1, 2};
    ASSERT_TRUE(runner0_ptr->forward(tokens0, 3));
    ASSERT_TRUE(runner1_ptr->forward(tokens1, 2));

    std::vector<std::unique_ptr<IInferenceRunner>> runners;
    runners.push_back(std::move(runner0));
    runners.push_back(std::move(runner1));

    auto orchestrator = RankOrchestrator::createForTest(
        llaminar2::test::MockModelContext::createMinimal(),
        std::move(runners),
        makeTPContextForRunnerCount(2),
        makeRankConfigForRunnerCount(2));

    PrefixStateSnapshot snapshot = orchestrator->captureLivePrefixState();
    EXPECT_FALSE(snapshot.valid);
    EXPECT_TRUE(snapshot.participant_snapshots.empty());
    EXPECT_EQ(runner0_ptr->prefix_live_capture_call_count(), 1u);
    EXPECT_EQ(runner1_ptr->prefix_live_capture_call_count(), 1u);
}

TEST_F(Test__RankOrchestrator, RestoreLivePrefixStateAttemptsEveryChildAndReportsFailure)
{
    auto runner0 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner0_ptr = runner0.get();
    auto runner1 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner1_ptr = runner1.get();

    int tokens[] = {1, 2, 3};
    ASSERT_TRUE(runner0_ptr->forward(tokens, 3));
    ASSERT_TRUE(runner1_ptr->forward(tokens, 3));

    std::vector<std::unique_ptr<IInferenceRunner>> runners;
    runners.push_back(std::move(runner0));
    runners.push_back(std::move(runner1));

    auto orchestrator = RankOrchestrator::createForTest(
        llaminar2::test::MockModelContext::createMinimal(),
        std::move(runners),
        makeTPContextForRunnerCount(2),
        makeRankConfigForRunnerCount(2));

    PrefixStateSnapshot snapshot = orchestrator->captureLivePrefixState();
    ASSERT_TRUE(snapshot.valid);
    runner1_ptr->set_prefix_live_restore_ok(false);

    EXPECT_FALSE(orchestrator->restoreLivePrefixState(snapshot));
    EXPECT_EQ(runner0_ptr->prefix_live_restore_call_count(), 1u);
    EXPECT_EQ(runner1_ptr->prefix_live_restore_call_count(), 1u);
}

// =============================================================================
// Edge Cases and Error Handling
// =============================================================================

TEST_F(Test__RankOrchestrator, EmptyDeviceRunnersReturnsSafeDefaults)
{
    std::vector<std::unique_ptr<MockDeviceGraphOrchestrator>> empty_runners;

    // With no runners, we should handle gracefully
    EXPECT_TRUE(empty_runners.empty());
}

TEST_F(Test__RankOrchestrator, SingleTPContextWithOneDevice)
{
    // Test TP context with single device - should work but is unusual
    MockLocalTPContext::Config config;
    config.devices = {GlobalDeviceAddress::cpu()};
    config.weights = {1.0f};
    auto ctx = std::make_unique<MockLocalTPContext>(config);

    EXPECT_EQ(ctx->degree(), 1);
    EXPECT_FLOAT_EQ(ctx->weights()[0], 1.0f);
}

TEST_F(Test__RankOrchestrator, MockRunnerPositionUpdatesOnForward)
{
    auto runner = std::make_unique<MockDeviceGraphOrchestrator>();
    int tokens[] = {1, 2, 3, 4, 5};

    EXPECT_EQ(runner->get_position(), 0);

    runner->forward(tokens, 5);
    EXPECT_EQ(runner->get_position(), 5);

    runner->forward(tokens, 3);
    EXPECT_EQ(runner->get_position(), 8);
}

TEST_F(Test__RankOrchestrator, MockRunnerClearCacheResetsPosition)
{
    auto runner = std::make_unique<MockDeviceGraphOrchestrator>();
    int tokens[] = {1, 2, 3};

    runner->forward(tokens, 3);
    EXPECT_EQ(runner->get_position(), 3);

    runner->clear_cache();
    EXPECT_EQ(runner->get_position(), 0);
}

TEST_F(Test__RankOrchestrator, MockTPContextRowRangeCalculation)
{
    MockLocalTPContext::Config config;
    // Use distinct devices so indexForDevice can differentiate them
    config.devices = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)};
    config.weights = {0.75f, 0.25f};
    auto ctx = std::make_unique<MockLocalTPContext>(config);

    // For 100 total rows with 75/25 split:
    // Device 0: rows 0-74 (75 rows)
    // Device 1: rows 75-99 (25 rows)
    auto [start0, end0] = ctx->rowRangeForDevice(config.devices[0], 100);
    EXPECT_EQ(start0, 0);
    EXPECT_EQ(end0, 75);

    auto [start1, end1] = ctx->rowRangeForDevice(config.devices[1], 100);
    EXPECT_EQ(start1, 75);
    EXPECT_EQ(end1, 100);
}

TEST_F(Test__RankOrchestrator, MockTPContextColRangeMatchesRowRange)
{
    MockLocalTPContext::Config config;
    // Use distinct devices so indexForDevice can differentiate them
    config.devices = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)};
    config.weights = {0.6f, 0.4f};
    auto ctx = std::make_unique<MockLocalTPContext>(config);

    auto row_range = ctx->rowRangeForDevice(config.devices[0], 50);
    auto col_range = ctx->colRangeForDevice(config.devices[0], 50);

    EXPECT_EQ(row_range, col_range);
}

TEST_F(Test__RankOrchestrator, ResetCallCountsWorks)
{
    auto runner = std::make_unique<MockDeviceGraphOrchestrator>();
    int tokens[] = {1, 2, 3};

    runner->forward(tokens, 3);
    runner->clear_cache();

    EXPECT_EQ(runner->forward_call_count(), 1u);
    EXPECT_EQ(runner->clear_cache_call_count(), 1u);

    runner->reset_call_counts();

    EXPECT_EQ(runner->forward_call_count(), 0u);
    EXPECT_EQ(runner->clear_cache_call_count(), 0u);
}

TEST_F(Test__RankOrchestrator, MockTPContextResetCallCountsWorks)
{
    mock_tp_ctx_->synchronize();
    mock_tp_ctx_->allreduce(static_cast<TensorBase *>(nullptr));

    EXPECT_EQ(mock_tp_ctx_->synchronize_call_count(), 1u);
    EXPECT_EQ(mock_tp_ctx_->allreduce_call_count(), 1u);

    mock_tp_ctx_->reset_call_counts();

    EXPECT_EQ(mock_tp_ctx_->synchronize_call_count(), 0u);
    EXPECT_EQ(mock_tp_ctx_->allreduce_call_count(), 0u);
}

// =============================================================================
// TPSnapshot Row/Cols Inference Tests
// =============================================================================
// These tests verify that TPSnapshot correctly handles row/cols metadata
// for column-parallel stages. The fix ensures local_cols is properly computed
// from hidden_size/tp_degree rather than using flattened size.

TEST_F(Test__RankOrchestrator, TPSnapshot_ColumnParallel_CorrectRowsColsForSingleRowData)
{
    // Test case: Single-row column-parallel data (typical decode case)
    // hidden_size=896, tp_degree=2, local_cols=448, seq_len=1
    // Each device should have [1, 448] shape

    TPSnapshot snapshot;
    snapshot.key = "layer0_QKV_PROJ";
    snapshot.mode = SnapshotShardingMode::COLUMN_PARALLEL;
    snapshot.tp_degree = 2;

    // Device 0: 448 elements -> should be [1, 448]
    DeviceSnapshotData dev0;
    dev0.device_index = 0;
    dev0.rows = 1;   // Correctly set rows
    dev0.cols = 448; // Correctly set cols (local_cols = 896/2)
    dev0.global_start_col = 0;
    dev0.global_total_cols = 896;
    dev0.data.resize(448, 1.0f);

    // Device 1: 448 elements -> should be [1, 448]
    DeviceSnapshotData dev1;
    dev1.device_index = 1;
    dev1.rows = 1;
    dev1.cols = 448;
    dev1.global_start_col = 448;
    dev1.global_total_cols = 896;
    dev1.data.resize(448, 2.0f);

    snapshot.device_data.push_back(std::move(dev0));
    snapshot.device_data.push_back(std::move(dev1));

    // computeCombined should correctly concatenate columns
    ASSERT_TRUE(snapshot.computeCombined());
    EXPECT_EQ(snapshot.combined_rows, 1);
    EXPECT_EQ(snapshot.combined_cols, 896);
    EXPECT_EQ(snapshot.combined_data.size(), 896);

    // Verify data is correctly concatenated: [1,1,1,...448...,2,2,2,...448...]
    EXPECT_FLOAT_EQ(snapshot.combined_data[0], 1.0f);   // First from device 0
    EXPECT_FLOAT_EQ(snapshot.combined_data[447], 1.0f); // Last from device 0
    EXPECT_FLOAT_EQ(snapshot.combined_data[448], 2.0f); // First from device 1
    EXPECT_FLOAT_EQ(snapshot.combined_data[895], 2.0f); // Last from device 1
}

TEST_F(Test__RankOrchestrator, TPSnapshot_ColumnParallel_CorrectRowsColsForMultiRowData)
{
    // Test case: Multi-row column-parallel data (typical prefill case)
    // hidden_size=896, tp_degree=2, local_cols=448, seq_len=9
    // Each device should have [9, 448] shape (total 4032 elements)

    TPSnapshot snapshot;
    snapshot.key = "layer0_ATTENTION_CONTEXT";
    snapshot.mode = SnapshotShardingMode::COLUMN_PARALLEL;
    snapshot.tp_degree = 2;

    const size_t seq_len = 9;
    const size_t local_cols = 448;
    const size_t total_elements = seq_len * local_cols;

    // Device 0: [9, 448] = 4032 elements
    DeviceSnapshotData dev0;
    dev0.device_index = 0;
    dev0.rows = seq_len;
    dev0.cols = local_cols;
    dev0.global_start_col = 0;
    dev0.global_total_cols = 896;
    dev0.data.resize(total_elements);
    // Fill with row index for verification
    for (size_t r = 0; r < seq_len; ++r)
    {
        for (size_t c = 0; c < local_cols; ++c)
        {
            dev0.data[r * local_cols + c] = static_cast<float>(r * 10 + 0); // Row * 10 + device
        }
    }

    // Device 1: [9, 448] = 4032 elements
    DeviceSnapshotData dev1;
    dev1.device_index = 1;
    dev1.rows = seq_len;
    dev1.cols = local_cols;
    dev1.global_start_col = local_cols;
    dev1.global_total_cols = 896;
    dev1.data.resize(total_elements);
    for (size_t r = 0; r < seq_len; ++r)
    {
        for (size_t c = 0; c < local_cols; ++c)
        {
            dev1.data[r * local_cols + c] = static_cast<float>(r * 10 + 1);
        }
    }

    snapshot.device_data.push_back(std::move(dev0));
    snapshot.device_data.push_back(std::move(dev1));

    // computeCombined should correctly concatenate columns for each row
    ASSERT_TRUE(snapshot.computeCombined());
    EXPECT_EQ(snapshot.combined_rows, seq_len);
    EXPECT_EQ(snapshot.combined_cols, 896);
    EXPECT_EQ(snapshot.combined_data.size(), seq_len * 896);

    // Verify row-by-row concatenation
    // Row 0: [dev0 cols...] [dev1 cols...]
    // Combined row 0, col 0 = dev0 data
    EXPECT_FLOAT_EQ(snapshot.combined_data[0], 0.0f); // Row 0, dev0
    // Combined row 0, col 448 = dev1 data
    EXPECT_FLOAT_EQ(snapshot.combined_data[448], 1.0f); // Row 0, dev1

    // Row 5: should have value 50 (5*10+0) from dev0, 51 (5*10+1) from dev1
    size_t row5_offset = 5 * 896;
    EXPECT_FLOAT_EQ(snapshot.combined_data[row5_offset], 50.0f);       // Row 5, dev0
    EXPECT_FLOAT_EQ(snapshot.combined_data[row5_offset + 448], 51.0f); // Row 5, dev1
}

TEST_F(Test__RankOrchestrator, TPSnapshot_ColumnParallel_WrongColsBreaksCombine)
{
    // Test case: What happens if cols is incorrectly set to flattened size?
    // This was the BUG: cols=4032 instead of cols=448
    // With incorrect cols, row stride calculation breaks

    TPSnapshot snapshot;
    snapshot.key = "layer0_BUG_CASE";
    snapshot.mode = SnapshotShardingMode::COLUMN_PARALLEL;
    snapshot.tp_degree = 2;

    const size_t seq_len = 9;
    const size_t local_cols = 448;
    const size_t total_elements = seq_len * local_cols;

    // Device 0 with CORRECT metadata
    DeviceSnapshotData dev0_correct;
    dev0_correct.device_index = 0;
    dev0_correct.rows = seq_len;
    dev0_correct.cols = local_cols; // CORRECT: 448
    dev0_correct.global_start_col = 0;
    dev0_correct.global_total_cols = 896;
    dev0_correct.data.resize(total_elements, 1.0f);

    // Device 1 with CORRECT metadata
    DeviceSnapshotData dev1_correct;
    dev1_correct.device_index = 1;
    dev1_correct.rows = seq_len;
    dev1_correct.cols = local_cols; // CORRECT: 448
    dev1_correct.global_start_col = local_cols;
    dev1_correct.global_total_cols = 896;
    dev1_correct.data.resize(total_elements, 2.0f);

    snapshot.device_data.push_back(std::move(dev0_correct));
    snapshot.device_data.push_back(std::move(dev1_correct));

    ASSERT_TRUE(snapshot.computeCombined());

    // With correct metadata, combined should have proper dimensions
    EXPECT_EQ(snapshot.combined_rows, seq_len);
    EXPECT_EQ(snapshot.combined_cols, 896);

    // Now test with INCORRECT metadata (the bug case)
    TPSnapshot buggy_snapshot;
    buggy_snapshot.key = "layer0_BUG_CASE";
    buggy_snapshot.mode = SnapshotShardingMode::COLUMN_PARALLEL;
    buggy_snapshot.tp_degree = 2;

    // Device 0 with BUG: cols = flattened size instead of actual cols
    DeviceSnapshotData dev0_buggy;
    dev0_buggy.device_index = 0;
    dev0_buggy.rows = 1;              // BUG: rows=1 because size/cols gives 1 when cols=4032
    dev0_buggy.cols = total_elements; // BUG: cols=4032 (flattened size)
    dev0_buggy.global_start_col = 0;
    dev0_buggy.global_total_cols = total_elements * 2;
    dev0_buggy.data.resize(total_elements, 1.0f);

    DeviceSnapshotData dev1_buggy;
    dev1_buggy.device_index = 1;
    dev1_buggy.rows = 1;
    dev1_buggy.cols = total_elements; // BUG: cols=4032
    dev1_buggy.global_start_col = total_elements;
    dev1_buggy.global_total_cols = total_elements * 2;
    dev1_buggy.data.resize(total_elements, 2.0f);

    buggy_snapshot.device_data.push_back(std::move(dev0_buggy));
    buggy_snapshot.device_data.push_back(std::move(dev1_buggy));

    ASSERT_TRUE(buggy_snapshot.computeCombined());

    // With buggy metadata, combined has WRONG dimensions
    // This would cause comparison against PyTorch (which has [9, 896]) to fail
    EXPECT_EQ(buggy_snapshot.combined_rows, 1);                  // WRONG: should be 9
    EXPECT_EQ(buggy_snapshot.combined_cols, total_elements * 2); // WRONG: should be 896

    // The total data size is the same, but the row/col interpretation is wrong
    // This demonstrates why correct rows/cols metadata is critical
    EXPECT_NE(buggy_snapshot.combined_rows, snapshot.combined_rows);
    EXPECT_NE(buggy_snapshot.combined_cols, snapshot.combined_cols);
}

TEST_F(Test__RankOrchestrator, TPSnapshot_ColumnParallel_ProportionalWeights)
{
    // Test case: Proportional TP with 73%/27% split (heterogeneous GPUs)
    // hidden_size=896, device0 gets 73% = 654 cols, device1 gets 27% = 242 cols

    TPSnapshot snapshot;
    snapshot.key = "layer0_PROPORTIONAL";
    snapshot.mode = SnapshotShardingMode::COLUMN_PARALLEL;
    snapshot.tp_degree = 2;

    const size_t seq_len = 4;
    const size_t dev0_cols = 654; // 73% of 896
    const size_t dev1_cols = 242; // 27% of 896
    const size_t total_cols = dev0_cols + dev1_cols;

    DeviceSnapshotData dev0;
    dev0.device_index = 0;
    dev0.rows = seq_len;
    dev0.cols = dev0_cols;
    dev0.global_start_col = 0;
    dev0.global_total_cols = total_cols;
    dev0.data.resize(seq_len * dev0_cols, 1.0f);

    DeviceSnapshotData dev1;
    dev1.device_index = 1;
    dev1.rows = seq_len;
    dev1.cols = dev1_cols;
    dev1.global_start_col = dev0_cols;
    dev1.global_total_cols = total_cols;
    dev1.data.resize(seq_len * dev1_cols, 2.0f);

    snapshot.device_data.push_back(std::move(dev0));
    snapshot.device_data.push_back(std::move(dev1));

    ASSERT_TRUE(snapshot.computeCombined());
    EXPECT_EQ(snapshot.combined_rows, seq_len);
    EXPECT_EQ(snapshot.combined_cols, total_cols);

    // Verify uneven column concatenation works
    // Row 0: [654 cols from dev0] [242 cols from dev1]
    EXPECT_FLOAT_EQ(snapshot.combined_data[0], 1.0f);              // First col from dev0
    EXPECT_FLOAT_EQ(snapshot.combined_data[653], 1.0f);            // Last col from dev0
    EXPECT_FLOAT_EQ(snapshot.combined_data[654], 2.0f);            // First col from dev1
    EXPECT_FLOAT_EQ(snapshot.combined_data[total_cols - 1], 2.0f); // Last col from dev1
}

TEST_F(Test__RankOrchestrator, TPSnapshot_Replicated_SingleRowMultipleDevices)
{
    // Test case: Replicated stage (same output on all devices)
    // Each device has full [1, 896] output

    TPSnapshot snapshot;
    snapshot.key = "layer0_ATTN_NORM";
    snapshot.mode = SnapshotShardingMode::REPLICATED;
    snapshot.tp_degree = 2;

    const size_t total_cols = 896;

    DeviceSnapshotData dev0;
    dev0.device_index = 0;
    dev0.rows = 1;
    dev0.cols = total_cols;
    dev0.global_start_col = 0;
    dev0.global_total_cols = total_cols;
    dev0.data.resize(total_cols, 3.14f);

    DeviceSnapshotData dev1;
    dev1.device_index = 1;
    dev1.rows = 1;
    dev1.cols = total_cols;
    dev1.global_start_col = 0;
    dev1.global_total_cols = total_cols;
    dev1.data.resize(total_cols, 3.14f); // Same data as dev0

    snapshot.device_data.push_back(std::move(dev0));
    snapshot.device_data.push_back(std::move(dev1));

    ASSERT_TRUE(snapshot.computeCombined());
    EXPECT_EQ(snapshot.combined_rows, 1);
    EXPECT_EQ(snapshot.combined_cols, total_cols);

    // For replicated, should just use first device's data
    EXPECT_FLOAT_EQ(snapshot.combined_data[0], 3.14f);
}
