/**
 * @file Test__PrefillDecodeTransition.cpp
 * @brief Regression tests for the prefill→decode transition bug fix
 *
 * Verifies that after prefill(), the first decodeStep() samples from the
 * already-computed prefill logits instead of re-feeding the last prompt token.
 *
 * Bug: Prior to the fix, prefill() stored last_token_ = prompt_tokens.back(),
 * and decodeStep() called forward(&last_token_, 1), reprocessing the last
 * prompt token at position N+1. This corrupted GDN recurrence state and
 * created duplicate KV cache entries, causing degenerate output on Qwen3.5.
 *
 * Fix: prefill() sets prefill_logits_ready_ = true. The first decodeStep()
 * detects this flag, skips the forward() call, and samples from the existing
 * prefill logits directly.
 *
 * @author David Sanftenberg
 * @date April 2026
 */

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "execution/runner/OrchestrationRunner.h"
#include "execution/global/GlobalOrchestrator.h"
#include "execution/global_pp/GlobalPPTopology.h"
#include "execution/local_execution/orchestrators/IInferenceRunner.h"
#include "execution/local_execution/orchestrators/RankOrchestrator.h"
#include "execution/mtp/MTPSpecStateContract.h"
#include "config/OrchestrationConfig.h"
#include "execution/mpi_orchestration/RankExecutionPlan.h"
#include "backends/GlobalDeviceAddress.h"
#include "tensors/Tensors.h"
#include "utils/DebugEnv.h"
#include "utils/PerfStatsCollector.h"
#include "../../../mocks/MockModelContext.h"
#include "../../../mocks/MockMPIContext.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using namespace llaminar2;
using namespace testing;

namespace
{
    class ScopedEnv
    {
    public:
        ScopedEnv(const char *name, const char *value)
            : name_(name)
        {
            const char *old = std::getenv(name);
            if (old)
            {
                had_old_ = true;
                old_value_ = old;
            }
            setenv(name, value, 1);
            mutableDebugEnv().reload();
        }

        ~ScopedEnv()
        {
            if (had_old_)
                setenv(name_.c_str(), old_value_.c_str(), 1);
            else
                unsetenv(name_.c_str());
            mutableDebugEnv().reload();
        }

    private:
        std::string name_;
        bool had_old_ = false;
        std::string old_value_;
    };

    const PerfStatRecord *findPerfRecord(
        const std::vector<PerfStatRecord> &records,
        PerfStatRecord::Kind kind,
        const std::string &name)
    {
        auto it = std::find_if(records.begin(), records.end(), [&](const auto &record)
                               {
                                   return record.kind == kind &&
                                          record.domain == "mtp" &&
                                          record.name == name;
                               });
        return it == records.end() ? nullptr : &*it;
    }

    const PerfStatRecord *findPerfRecordWithTags(
        const std::vector<PerfStatRecord> &records,
        PerfStatRecord::Kind kind,
        const std::string &name,
        const std::map<std::string, std::string> &expected_tags)
    {
        auto it = std::find_if(records.begin(), records.end(), [&](const auto &record)
                               {
                                   if (record.kind != kind ||
                                       record.domain != "mtp" ||
                                       record.name != name)
                                   {
                                       return false;
                                   }
                                   for (const auto &[key, value] : expected_tags)
                                   {
                                       auto tag_it = record.tags.find(key);
                                       if (tag_it == record.tags.end() || tag_it->second != value)
                                           return false;
                                   }
                                   return true;
                               });
        return it == records.end() ? nullptr : &*it;
    }

    // =========================================================================
    // Mock IInferenceRunner
    // =========================================================================

    /**
     * @brief Mock runner that tracks forward() calls and provides fake logits.
     *
     * The mock provides a small fake vocabulary (10 tokens) with deterministic
     * logits so we can verify which token gets sampled.
     */
    class MockInferenceRunner : public IInferenceRunner
    {
    public:
        static constexpr int VOCAB_SIZE = 10;
        static constexpr int PREFILL_ARGMAX_TOKEN = 7; // Token with highest logit after prefill
        static constexpr int DECODE_ARGMAX_TOKEN = 3;  // Token with highest logit after decode forward
        static constexpr int MTP_ARGMAX_TOKEN = 9;
        static constexpr int VERIFY_REJECT_TOKEN = 4;

        MockInferenceRunner()
        {
            // Set up prefill logits: token 7 has highest value
            setupPrefillLogits();
        }

        // Core API
        bool forward(const int *tokens, int seq_len) override
        {
            forward_call_count_++;
            last_forward_tokens_.assign(tokens, tokens + seq_len);
            forward_history_.push_back(last_forward_tokens_);
            last_forward_seq_len_ = seq_len;
            position_ += seq_len;

            if (all_position_logits_enabled_)
            {
                setupAllPositionLogits(seq_len);
                return true;
            }

            // After prefill (first forward in a cycle), set prefill logits.
            // After decode forwards, set decode logits.
            // clear_cache() resets the cycle so the next forward is "prefill" again.
            if (is_first_forward_in_cycle_)
            {
                setupPrefillLogits();
                is_first_forward_in_cycle_ = false;
            }
            else
            {
                setupDecodeLogits();
            }

            return true;
        }

        const float *logits() const override
        {
            if (hide_local_logits_)
                return nullptr;
            if (column_parallel_logits_)
                return nullptr;
            return logits_.data();
        }

        bool forwardMTP(int32_t draft_condition_token) override
        {
            if (!mtp_enabled_)
                return false;
            forward_mtp_count_++;
            last_mtp_condition_token_ = draft_condition_token;
            mtp_logits_.assign(VOCAB_SIZE, -10.0f);
            mtp_logits_[MTP_ARGMAX_TOKEN] = 10.0f;
            if (column_parallel_logits_)
            {
                resetLocalTensor(mtp_logits_local_, 1);
                setLocalToken(mtp_logits_local_, 0, MTP_ARGMAX_TOKEN, 10.0f);
            }
            return true;
        }

        bool supportsChainedMTPDrafts() const override
        {
            return supports_chained_mtp_drafts_;
        }

        bool forwardMTPFromLastDraft(int32_t draft_condition_token, int position_id) override
        {
            if (!mtp_enabled_ || !supports_chained_mtp_drafts_ || position_id < 0)
                return false;
            forward_mtp_from_last_draft_count_++;
            last_chained_mtp_condition_token_ = draft_condition_token;
            last_chained_mtp_position_id_ = position_id;
            mtp_logits_.assign(VOCAB_SIZE, -10.0f);
            mtp_logits_[MTP_ARGMAX_TOKEN] = 10.0f;
            if (column_parallel_logits_)
            {
                resetLocalTensor(mtp_logits_local_, 1);
                setLocalToken(mtp_logits_local_, 0, MTP_ARGMAX_TOKEN, 10.0f);
            }
            return true;
        }

        const float *mtpLogits() const override
        {
            if (hide_local_logits_)
                return nullptr;
            if (column_parallel_logits_)
                return nullptr;
            return mtp_logits_.empty() ? nullptr : mtp_logits_.data();
        }

        bool commitMTPShiftedRowsFromLastForward(
            const int32_t *tokens,
            int token_count,
            int already_appended_tokens) override
        {
            return commitMTPShiftedRowsFromPartialForward(
                tokens,
                token_count,
                already_appended_tokens,
                token_count);
        }

        bool commitMTPShiftedRowsFromPartialForward(
            const int32_t *tokens,
            int token_count,
            int already_appended_tokens,
            int main_forward_token_count,
            bool allow_speculative_discard = false,
            int position_offset_override = -1) override
        {
            commit_mtp_shifted_count_++;
            last_commit_mtp_already_appended_ = already_appended_tokens;
            last_commit_mtp_main_forward_token_count_ = main_forward_token_count;
            last_commit_mtp_allow_speculative_discard_ = allow_speculative_discard;
            last_commit_mtp_position_offset_override_ = position_offset_override;
            last_commit_mtp_tokens_.clear();
            if (tokens && token_count > 0)
            {
                last_commit_mtp_tokens_.assign(tokens, tokens + token_count);
            }
            const int catchup_token_count = token_count - already_appended_tokens;
            const int hidden_source_row_start = already_appended_tokens - 1;
            const int hidden_source_row_end = hidden_source_row_start + catchup_token_count;
            return token_count <= already_appended_tokens ||
                   (tokens != nullptr &&
                    main_forward_token_count > 0 &&
                    hidden_source_row_start >= 0 &&
                    hidden_source_row_end <= main_forward_token_count);
        }

        bool commitMTPShiftedRowFromCurrentTerminalHidden(
            int32_t token,
            int already_appended_tokens,
            bool allow_speculative_discard = false,
            int position_offset_override = -1) override
        {
            commit_mtp_shifted_count_++;
            sequential_commit_mtp_shifted_count_++;
            last_commit_mtp_already_appended_ = already_appended_tokens;
            last_commit_mtp_main_forward_token_count_ = 0;
            last_commit_mtp_allow_speculative_discard_ = allow_speculative_discard;
            last_commit_mtp_position_offset_override_ = position_offset_override;
            last_commit_mtp_tokens_.assign(1, token);
            return already_appended_tokens >= 0;
        }

        bool hasMTPLogitsLocal() const override
        {
            return column_parallel_logits_ && mtp_logits_local_ != nullptr;
        }

        LogitsLocalInfo getMTPLogitsLocalInfo() const override
        {
            return makeLocalInfo(mtp_logits_local_.get());
        }

        bool setComputeAllPositionLogits(bool enabled) override
        {
            if (!mtp_enabled_)
                return false;
            all_position_logits_enabled_ = enabled;
            set_all_position_count_++;
            return true;
        }

        const float *getAllPositionLogits() const override
        {
            if (hide_local_logits_)
                return nullptr;
            if (column_parallel_logits_)
                return nullptr;
            return all_position_logits_.empty() ? nullptr : all_position_logits_.data();
        }

        bool hasAllPositionLogitsLocal() const override
        {
            return column_parallel_logits_ && all_position_logits_local_ != nullptr;
        }

        LogitsLocalInfo getAllPositionLogitsLocalInfo() const override
        {
            return makeLocalInfo(all_position_logits_local_.get());
        }

        std::string mtpDecodeUnsupportedReason() const override
        {
            return mtp_unsupported_reason_;
        }

        DeviceId primaryDeviceId() const override
        {
            return primary_device_;
        }

        bool supportsMTPTokenCoordination() const override
        {
            return supports_mtp_token_coordination_;
        }

        bool supportsMTPSidecarSampleFusion() const override
        {
            return supports_mtp_sidecar_sample_fusion_;
        }

        bool supportsMTPSidecarPreservesMainState() const override
        {
            return supports_mtp_sidecar_preserves_main_state_;
        }

        bool requiresMTPDecodeEquivalentVerifierReplay() const override
        {
            return requires_mtp_decode_equivalent_replay_;
        }

        bool supportsMTPSpecStatePublication() const override
        {
            return supports_mtp_spec_state_publication_;
        }

        bool publishAcceptedMTPSpecState(
            const MTPSpecStepPlan &plan,
            std::string *error = nullptr) override
        {
            ++publish_mtp_spec_state_count_;
            last_published_mtp_spec_step_ = plan;
            if (!supports_mtp_spec_state_publication_)
            {
                if (error)
                    *error = "mock MTP spec-state publication is disabled";
                return false;
            }
            if (!publish_mtp_spec_state_ok_)
            {
                if (error)
                    *error = "mock MTP spec-state publication failed";
                return false;
            }
            position_ = plan.target_cached_tokens;
            all_position_logits_enabled_ = false;
            return true;
        }

        bool forwardMTPAndSampleGreedy(int32_t draft_condition_token, int32_t *out_token) override
        {
            ++forward_mtp_and_sample_count_;
            if (!out_token)
                return false;
            if (!forwardMTP(draft_condition_token) || mtp_logits_.empty())
                return false;
            const int token = greedyArgmax(mtp_logits_.data(), VOCAB_SIZE);
            if (token < 0)
                return false;
            *out_token = token;
            return true;
        }

        bool forwardMTPFromLastDraftAndSampleGreedy(
            int32_t draft_condition_token,
            int position_id,
            int32_t *out_token) override
        {
            ++forward_mtp_from_last_draft_and_sample_count_;
            if (!out_token)
                return false;
            if (!forwardMTPFromLastDraft(draft_condition_token, position_id) || mtp_logits_.empty())
                return false;
            const int token = greedyArgmax(mtp_logits_.data(), VOCAB_SIZE);
            if (token < 0)
                return false;
            *out_token = token;
            return true;
        }

        int sampleGreedyFromMTPLogitsOnDevice() override
        {
            ++sample_mtp_logits_count_;
            if (!supports_mtp_token_coordination_ || mtp_logits_.empty())
                return -1;
            return greedyArgmax(mtp_logits_.data(), VOCAB_SIZE);
        }

        int sampleGreedyFromAllPositionLogitsOnDevice(int row) override
        {
            ++sample_all_position_logits_count_;
            if (!supports_mtp_token_coordination_ || row < 0 || all_position_logits_.empty())
                return -1;
            const size_t offset = static_cast<size_t>(row) * static_cast<size_t>(VOCAB_SIZE);
            if (offset + static_cast<size_t>(VOCAB_SIZE) > all_position_logits_.size())
                return -1;
            return greedyArgmax(all_position_logits_.data() + offset, VOCAB_SIZE);
        }

        bool sampleGreedyFromAllPositionLogitsOnDeviceRows(
            int start_row,
            int row_count,
            int32_t *out_tokens) override
        {
            ++sample_all_position_logits_batched_count_;
            last_sample_all_position_start_row_ = start_row;
            last_sample_all_position_row_count_ = row_count;
            if (!supports_mtp_token_coordination_)
            {
                return IInferenceRunner::sampleGreedyFromAllPositionLogitsOnDeviceRows(
                    start_row, row_count, out_tokens);
            }
            if (start_row < 0 || row_count <= 0 || !out_tokens || all_position_logits_.empty())
            {
                return false;
            }
            for (int i = 0; i < row_count; ++i)
            {
                const int row = start_row + i;
                const size_t offset = static_cast<size_t>(row) * static_cast<size_t>(VOCAB_SIZE);
                if (offset + static_cast<size_t>(VOCAB_SIZE) > all_position_logits_.size())
                    return false;
                const int token = greedyArgmax(all_position_logits_.data() + offset, VOCAB_SIZE);
                if (token < 0)
                    return false;
                out_tokens[i] = static_cast<int32_t>(token);
            }
            return true;
        }

        int vocab_size() const override { return VOCAB_SIZE; }

        void clear_cache() override
        {
            clear_cache_count_++;
            is_first_forward_in_cycle_ = true; // Reset cycle
            setupPrefillLogits();              // Reset logits state
            position_ = 0;
        }

        int get_position() const override { return position_; }

        ExecutionPath executionPath() const override { return ExecutionPath::GRAPH; }
        const char *architecture() const override { return "mock"; }

        bool hasLogitsLocal() const override
        {
            return column_parallel_logits_ && logits_local_ != nullptr;
        }

        LogitsLocalInfo getLogitsLocalInfo() const override
        {
            return makeLocalInfo(logits_local_.get());
        }

        // GPU sampling returns -1 by default to force CPU fallback.
        int sampleGreedyOnDevice() override
        {
            ++sample_main_logits_count_;
            if (!supports_mtp_token_coordination_)
                return -1;
            return greedyArgmax(logits_.data(), VOCAB_SIZE);
        }

        int sampleOnDevice(const SamplingParams &params) override
        {
            ++sample_device_count_;
            if (!supports_stochastic_device_sampling_ || params.is_greedy())
                return -1;
            return greedyArgmax(logits_.data(), VOCAB_SIZE);
        }

        bool applyPenaltiesOnDevice(const std::vector<LogitPenalty> &penalties,
                                    int vocab_size) override
        {
            ++apply_main_penalties_count_;
            return applyPenaltiesToRow(logits_, 0, penalties, vocab_size);
        }

        bool applyPenaltiesToMTPLogitsOnDevice(const std::vector<LogitPenalty> &penalties,
                                               int vocab_size) override
        {
            ++apply_mtp_penalties_count_;
            return applyPenaltiesToRow(mtp_logits_, 0, penalties, vocab_size);
        }

        bool applyPenaltiesToAllPositionLogitsOnDeviceRow(
            int row,
            const std::vector<LogitPenalty> &penalties,
            int vocab_size) override
        {
            ++apply_all_position_penalties_count_;
            return applyPenaltiesToRow(all_position_logits_, row, penalties, vocab_size);
        }

        bool supportsDeviceStochasticMTPVerification() const override
        {
            return supports_stochastic_device_sampling_;
        }

        bool buildStochasticDistributionOnDevice(
            DeviceLogitsSource source,
            int row,
            DeviceDistributionBuffer buffer,
            int slot,
            const SamplingParams &params,
            int vocab_size) override
        {
            ++device_distribution_build_count_;
            if (!supports_stochastic_device_sampling_ || row < 0 ||
                vocab_size != VOCAB_SIZE || params.top_k <= 0)
            {
                return false;
            }

            const float *row_logits = nullptr;
            switch (source)
            {
            case DeviceLogitsSource::Main:
                if (row != 0 || logits_.size() != VOCAB_SIZE)
                    return false;
                row_logits = logits_.data();
                break;
            case DeviceLogitsSource::MTP:
                if (row != 0 || mtp_logits_.size() != VOCAB_SIZE)
                    return false;
                row_logits = mtp_logits_.data();
                break;
            case DeviceLogitsSource::AllPosition:
            {
                const size_t offset = static_cast<size_t>(row) * VOCAB_SIZE;
                if (offset + VOCAB_SIZE > all_position_logits_.size())
                    return false;
                row_logits = all_position_logits_.data() + offset;
                break;
            }
            }

            if (!row_logits)
                return false;

            auto &target = deviceDistribution(buffer, slot);
            SamplingParams distribution_params = params;
            distribution_params.presence_penalty = 0.0f;
            distribution_params.frequency_penalty = 0.0f;
            distribution_params.dry_multiplier = 0.0f;
            distribution_params.dry_penalty_last_n = 0;
            Sampler distribution_sampler(params.seed);
            target = distribution_sampler.compute_distribution(
                row_logits,
                VOCAB_SIZE,
                distribution_params);
            return !target.empty();
        }

        int sampleStochasticDistributionOnDevice(
            DeviceDistributionBuffer buffer,
            int slot,
            float threshold) override
        {
            ++device_distribution_sample_count_;
            if (!supports_stochastic_device_sampling_)
                return -1;
            const auto &distribution = deviceDistribution(buffer, slot);
            return sampleWithThreshold(distribution, threshold);
        }

        bool verifyStochasticDistributionsOnDevice(
            int target_slot,
            int draft_slot,
            int draft_token,
            float accept_threshold,
            float residual_threshold,
            DeviceSpeculativeVerifyResult *out) override
        {
            ++device_distribution_verify_count_;
            if (!supports_stochastic_device_sampling_ || !out)
                return false;

            const auto &target = deviceDistribution(DeviceDistributionBuffer::Target, target_slot);
            const auto &draft = deviceDistribution(DeviceDistributionBuffer::Draft, draft_slot);
            if (target.empty() || draft.empty())
                return false;

            const float p = Sampler::probability_of_token(target, draft_token);
            const float q = Sampler::probability_of_token(draft, draft_token);
            const float accept_probability =
                Sampler::speculative_accept_probability(p, q);
            out->accepted = accept_threshold < accept_probability;
            out->accept_probability = accept_probability;
            out->accept_threshold = accept_threshold;
            out->token = out->accepted
                             ? draft_token
                             : sampleResidualWithThreshold(target, draft, residual_threshold);
            return out->token >= 0;
        }

        bool verifyStochasticDistributionsBatchOnDevice(
            int first_target_slot,
            int first_draft_slot,
            const int32_t *draft_tokens,
            const float *accept_thresholds,
            const float *residual_thresholds,
            int row_count,
            DeviceSpeculativeVerifyResult *out) override
        {
            ++device_distribution_verify_batch_count_;
            if (!supports_stochastic_device_sampling_ ||
                !draft_tokens || !accept_thresholds || !residual_thresholds || !out ||
                first_target_slot < 0 || first_draft_slot < 0 || row_count <= 0)
            {
                return false;
            }

            for (int row = 0; row < row_count; ++row)
            {
                const auto &target =
                    deviceDistribution(DeviceDistributionBuffer::Target,
                                       first_target_slot + row);
                const auto &draft =
                    deviceDistribution(DeviceDistributionBuffer::Draft,
                                       first_draft_slot + row);
                if (target.empty() || draft.empty())
                    return false;

                const int32_t draft_token = draft_tokens[row];
                const float p = Sampler::probability_of_token(target, draft_token);
                const float q = Sampler::probability_of_token(draft, draft_token);
                const float accept_probability =
                    Sampler::speculative_accept_probability(p, q);
                out[row].accepted =
                    accept_thresholds[row] < accept_probability;
                out[row].accept_probability = accept_probability;
                out[row].accept_threshold = accept_thresholds[row];
                out[row].token = out[row].accepted ? draft_token : -1;
                if (!out[row].accepted)
                    out[row].token =
                        sampleResidualWithThreshold(target, draft, residual_thresholds[row]);
            }
            return true;
        }

        // =====================================================================
        // Test inspection methods
        // =====================================================================

        int forwardCallCount() const { return forward_call_count_; }
        int clearCacheCount() const { return clear_cache_count_; }
        int forwardMTPCount() const { return forward_mtp_count_; }
        int forwardMTPFromLastDraftCount() const { return forward_mtp_from_last_draft_count_; }
        int forwardMTPAndSampleCount() const { return forward_mtp_and_sample_count_; }
        int forwardMTPFromLastDraftAndSampleCount() const { return forward_mtp_from_last_draft_and_sample_count_; }
        int restoreCount() const { return restore_count_; }
        int captureCheckpointCount() const { return capture_checkpoint_count_; }
        int setAllPositionCount() const { return set_all_position_count_; }
        int lastMTPConditionToken() const { return last_mtp_condition_token_; }
        int lastChainedMTPConditionToken() const { return last_chained_mtp_condition_token_; }
        int lastChainedMTPPositionId() const { return last_chained_mtp_position_id_; }
        int commitMTPShiftedCount() const { return commit_mtp_shifted_count_; }
        int lastCommitMTPAlreadyAppended() const { return last_commit_mtp_already_appended_; }
        int lastCommitMTPMainForwardTokenCount() const { return last_commit_mtp_main_forward_token_count_; }
        bool lastCommitMTPAllowSpeculativeDiscard() const { return last_commit_mtp_allow_speculative_discard_; }
        int lastCommitMTPPositionOffsetOverride() const { return last_commit_mtp_position_offset_override_; }
        const std::vector<int> &lastCommitMTPTokens() const { return last_commit_mtp_tokens_; }
        int sequentialCommitMTPShiftedCount() const { return sequential_commit_mtp_shifted_count_; }
        int sampleMainLogitsCount() const { return sample_main_logits_count_; }
        int sampleDeviceCount() const { return sample_device_count_; }
        int sampleMTPLogitsCount() const { return sample_mtp_logits_count_; }
        int sampleAllPositionLogitsCount() const { return sample_all_position_logits_count_; }
        int sampleAllPositionLogitsBatchedCount() const { return sample_all_position_logits_batched_count_; }
        int applyMainPenaltiesCount() const { return apply_main_penalties_count_; }
        int applyMTPPenaltiesCount() const { return apply_mtp_penalties_count_; }
        int applyAllPositionPenaltiesCount() const { return apply_all_position_penalties_count_; }
        int deviceDistributionBuildCount() const { return device_distribution_build_count_; }
        int deviceDistributionSampleCount() const { return device_distribution_sample_count_; }
        int deviceDistributionVerifyCount() const { return device_distribution_verify_count_; }
        int deviceDistributionVerifyBatchCount() const { return device_distribution_verify_batch_count_; }
        int lastSampleAllPositionStartRow() const { return last_sample_all_position_start_row_; }
        int lastSampleAllPositionRowCount() const { return last_sample_all_position_row_count_; }
        const PrefixStateSnapshot &lastRestoredSnapshot() const { return last_restored_snapshot_; }
        const std::vector<int> &lastForwardTokens() const { return last_forward_tokens_; }
        const std::vector<std::vector<int>> &forwardHistory() const { return forward_history_; }
        int lastForwardSeqLen() const { return last_forward_seq_len_; }
        void enableMTP(bool accept_mtp_token)
        {
            mtp_enabled_ = true;
            accept_mtp_token_ = accept_mtp_token;
        }
        void enableChainedMTPDrafts()
        {
            supports_chained_mtp_drafts_ = true;
        }
        void setMTPUnsupportedReason(std::string reason)
        {
            mtp_unsupported_reason_ = std::move(reason);
        }
        void setPrimaryDevice(DeviceId device)
        {
            primary_device_ = device;
        }
        void enableColumnParallelShard(int vocab_start, int vocab_local)
        {
            column_parallel_logits_ = true;
            vocab_start_ = vocab_start;
            vocab_local_ = vocab_local;
            setupPrefillLogits();
        }
        void enableMTPTokenCoordination(bool hide_local_logits)
        {
            supports_mtp_token_coordination_ = true;
            hide_local_logits_ = hide_local_logits;
        }
        void enableMTPSidecarSampleFusion()
        {
            supports_mtp_sidecar_sample_fusion_ = true;
        }
        void enableMTPSidecarPreservesMainState()
        {
            supports_mtp_sidecar_preserves_main_state_ = true;
        }
        void enableStochasticDeviceSampling()
        {
            supports_stochastic_device_sampling_ = true;
        }
        void setCapturedSnapshot(PrefixStateSnapshot snapshot)
        {
            captured_snapshot_ = std::move(snapshot);
            use_captured_snapshot_ = true;
        }
        void setCapturedCheckpointScript(std::vector<PrefixStateSnapshot> snapshots)
        {
            captured_checkpoint_script_ = std::move(snapshots);
            captured_checkpoint_script_index_ = 0;
        }
        void setVerifierAcceptedPrefixScript(std::vector<int> script)
        {
            verifier_accepted_prefix_script_ = std::move(script);
            verifier_accepted_prefix_script_index_ = 0;
        }
        void setDecodeArgmaxScript(std::vector<int> script)
        {
            decode_argmax_script_ = std::move(script);
            decode_argmax_script_index_ = 0;
        }
        PrefixStateSnapshot captureLivePrefixState(int seq_idx = 0) const override
        {
            (void)seq_idx;
            if (use_captured_snapshot_)
            {
                PrefixStateSnapshot snapshot = captured_snapshot_;
                snapshot.cached_tokens = position_;
                if (snapshot.provenance == PrefixStateProvenance::Unknown)
                    snapshot.provenance = snapshot.logical_checkpoint
                                              ? PrefixStateProvenance::LogicalCheckpoint
                                              : PrefixStateProvenance::PayloadCheckpoint;
                return snapshot;
            }
            PrefixStateSnapshot snapshot;
            snapshot.valid = mtp_enabled_;
            snapshot.cached_tokens = position_;
            snapshot.provenance = PrefixStateProvenance::PayloadCheckpoint;
            return snapshot;
        }

        PrefixStateSnapshot captureLivePrefixCheckpoint(int seq_idx = 0) const override
        {
            (void)seq_idx;
            capture_checkpoint_count_++;
            if (captured_checkpoint_script_index_ < captured_checkpoint_script_.size())
            {
                PrefixStateSnapshot snapshot =
                    captured_checkpoint_script_[captured_checkpoint_script_index_++];
                snapshot.cached_tokens = position_;
                if (snapshot.provenance == PrefixStateProvenance::Unknown)
                    snapshot.provenance = snapshot.logical_checkpoint
                                              ? PrefixStateProvenance::LogicalCheckpoint
                                              : PrefixStateProvenance::PayloadCheckpoint;
                return snapshot;
            }
            if (use_captured_snapshot_)
            {
                PrefixStateSnapshot snapshot = captured_snapshot_;
                snapshot.cached_tokens = position_;
                if (snapshot.provenance == PrefixStateProvenance::Unknown)
                    snapshot.provenance = snapshot.logical_checkpoint
                                              ? PrefixStateProvenance::LogicalCheckpoint
                                              : PrefixStateProvenance::PayloadCheckpoint;
                return snapshot;
            }
            PrefixStateSnapshot snapshot;
            snapshot.valid = mtp_enabled_;
            snapshot.logical_checkpoint = true;
            snapshot.provenance = PrefixStateProvenance::LogicalCheckpoint;
            snapshot.cached_tokens = position_;
            snapshot.mtp_cached_tokens = {std::max(0, position_ - 1)};
            return snapshot;
        }

        bool restoreLivePrefixState(const PrefixStateSnapshot &snapshot, int seq_idx = 0) override
        {
            (void)seq_idx;
            if (!snapshot.valid)
                return false;
            restore_count_++;
            last_restored_snapshot_ = snapshot;
            position_ = snapshot.cached_tokens;
            all_position_logits_enabled_ = false;
            return true;
        }

        void requireMTPDecodeEquivalentReplay()
        {
            requires_mtp_decode_equivalent_replay_ = true;
        }
        void enableMTPSpecStatePublication()
        {
            supports_mtp_spec_state_publication_ = true;
        }
        void setMTPSpecStatePublicationOk(bool ok)
        {
            publish_mtp_spec_state_ok_ = ok;
        }
        int publishMTPSpecStateCount() const
        {
            return publish_mtp_spec_state_count_;
        }
        const MTPSpecStepPlan &lastPublishedMTPSpecStep() const
        {
            return last_published_mtp_spec_step_;
        }

    private:
        static int greedyArgmax(const float *logits, int vocab)
        {
            if (!logits || vocab <= 0)
                return -1;
            int token = 0;
            float best = logits[0];
            for (int i = 1; i < vocab; ++i)
            {
                if (logits[i] > best)
                {
                    best = logits[i];
                    token = i;
                }
            }
            return token;
        }

        static bool applyPenaltiesToRow(std::vector<float> &logits,
                                        int row,
                                        const std::vector<LogitPenalty> &penalties,
                                        int vocab_size)
        {
            if (vocab_size != VOCAB_SIZE || row < 0)
                return false;
            const size_t offset = static_cast<size_t>(row) * VOCAB_SIZE;
            if (logits.size() < offset + VOCAB_SIZE)
                return false;
            for (const auto &penalty : penalties)
            {
                if (penalty.token_id < 0 || penalty.token_id >= VOCAB_SIZE)
                    continue;
                logits[offset + static_cast<size_t>(penalty.token_id)] -= penalty.penalty;
            }
            return true;
        }

        std::vector<SamplingDistributionEntry> &deviceDistribution(
            DeviceDistributionBuffer buffer,
            int slot)
        {
            if (slot < 0)
                return invalid_distribution_;
            if (buffer == DeviceDistributionBuffer::Target)
            {
                if (slot >= static_cast<int>(target_device_distributions_.size()))
                    return invalid_distribution_;
                return target_device_distributions_[static_cast<size_t>(slot)];
            }
            if (slot >= static_cast<int>(draft_device_distributions_.size()))
                return invalid_distribution_;
            return draft_device_distributions_[static_cast<size_t>(slot)];
        }

        const std::vector<SamplingDistributionEntry> &deviceDistribution(
            DeviceDistributionBuffer buffer,
            int slot) const
        {
            if (slot < 0)
                return invalid_distribution_;
            if (buffer == DeviceDistributionBuffer::Target)
            {
                if (slot >= static_cast<int>(target_device_distributions_.size()))
                    return invalid_distribution_;
                return target_device_distributions_[static_cast<size_t>(slot)];
            }
            if (slot >= static_cast<int>(draft_device_distributions_.size()))
                return invalid_distribution_;
            return draft_device_distributions_[static_cast<size_t>(slot)];
        }

        static int sampleWithThreshold(
            const std::vector<SamplingDistributionEntry> &distribution,
            float threshold)
        {
            if (distribution.empty())
                return -1;
            float cumulative = 0.0f;
            for (const auto &entry : distribution)
            {
                cumulative += entry.probability;
                if (threshold < cumulative)
                    return entry.token_id;
            }
            return distribution.back().token_id;
        }

        static int sampleResidualWithThreshold(
            const std::vector<SamplingDistributionEntry> &target,
            const std::vector<SamplingDistributionEntry> &draft,
            float threshold)
        {
            if (target.empty())
                return -1;

            std::vector<SamplingDistributionEntry> residual;
            residual.reserve(target.size());
            float total = 0.0f;
            for (const auto &entry : target)
            {
                const float q = Sampler::probability_of_token(draft, entry.token_id);
                const float p = std::max(0.0f, entry.probability - q);
                if (p > 0.0f)
                {
                    residual.push_back({entry.token_id, p});
                    total += p;
                }
            }

            if (!(total > 0.0f))
                return sampleWithThreshold(target, threshold);
            for (auto &entry : residual)
                entry.probability /= total;
            return sampleWithThreshold(residual, threshold);
        }

        void setupPrefillLogits()
        {
            logits_.assign(VOCAB_SIZE, -10.0f);
            logits_[PREFILL_ARGMAX_TOKEN] = 10.0f; // Token 7 is argmax
            if (column_parallel_logits_)
            {
                resetLocalTensor(logits_local_, 1);
                setLocalToken(logits_local_, 0, PREFILL_ARGMAX_TOKEN, 10.0f);
            }
        }

        void setupDecodeLogits()
        {
            logits_.assign(VOCAB_SIZE, -10.0f);
            int token = DECODE_ARGMAX_TOKEN;
            if (decode_argmax_script_index_ < decode_argmax_script_.size())
            {
                token = decode_argmax_script_[decode_argmax_script_index_++];
                token = std::clamp(token, 0, VOCAB_SIZE - 1);
            }
            else if (mtp_enabled_)
            {
                const size_t verifier_row = decode_argmax_script_index_++;
                token = accept_mtp_token_
                            ? MTP_ARGMAX_TOKEN
                            : (verifier_row == 0 ? VERIFY_REJECT_TOKEN
                                                 : DECODE_ARGMAX_TOKEN);
            }
            logits_[token] = 10.0f;
            if (column_parallel_logits_)
            {
                resetLocalTensor(logits_local_, 1);
                setLocalToken(logits_local_, 0, token, 10.0f);
            }
        }

        void setupAllPositionLogits(int seq_len)
        {
            all_position_logits_.assign(static_cast<size_t>(seq_len) * VOCAB_SIZE, -10.0f);
            int accepted_prefix = accept_mtp_token_ ? 1 : 0;
            if (verifier_accepted_prefix_script_index_ < verifier_accepted_prefix_script_.size())
            {
                accepted_prefix =
                    verifier_accepted_prefix_script_[verifier_accepted_prefix_script_index_++];
            }
            const int speculative_depth = std::max(0, seq_len - 1);
            accepted_prefix = std::clamp(accepted_prefix, 0, speculative_depth);

            for (int row = 0; row < seq_len; ++row)
            {
                int token = DECODE_ARGMAX_TOKEN;
                if (row < speculative_depth)
                {
                    if (row < accepted_prefix)
                        token = MTP_ARGMAX_TOKEN;
                    else
                        token = row == 0 ? VERIFY_REJECT_TOKEN : DECODE_ARGMAX_TOKEN;
                }
                all_position_logits_[static_cast<size_t>(row) * VOCAB_SIZE +
                                     static_cast<size_t>(token)] = 10.0f;
            }
            if (column_parallel_logits_)
            {
                resetLocalTensor(all_position_logits_local_, seq_len);
                for (int row = 0; row < seq_len; ++row)
                {
                    int token = DECODE_ARGMAX_TOKEN;
                    if (row < speculative_depth)
                    {
                        if (row < accepted_prefix)
                            token = MTP_ARGMAX_TOKEN;
                        else
                            token = row == 0 ? VERIFY_REJECT_TOKEN : DECODE_ARGMAX_TOKEN;
                    }
                    setLocalToken(all_position_logits_local_, row, token, 10.0f);
                }
            }
        }

        void resetLocalTensor(std::shared_ptr<FP32Tensor> &tensor, int rows)
        {
            tensor = std::make_shared<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(vocab_local_)},
                DeviceId::cpu());
            std::fill(tensor->mutable_data(), tensor->mutable_data() + tensor->numel(), -10.0f);
        }

        void setLocalToken(const std::shared_ptr<FP32Tensor> &tensor, int row, int global_token, float value)
        {
            if (!tensor)
                return;
            if (global_token < vocab_start_ || global_token >= vocab_start_ + vocab_local_)
                return;
            tensor->mutable_data()[static_cast<size_t>(row) * static_cast<size_t>(vocab_local_) +
                                   static_cast<size_t>(global_token - vocab_start_)] = value;
        }

        LogitsLocalInfo makeLocalInfo(FP32Tensor *tensor) const
        {
            if (!tensor)
                return {};
            LogitsLocalInfo info;
            info.gpu_ptr = nullptr;
            info.device = std::nullopt;
            info.vocab_local = static_cast<size_t>(vocab_local_);
            info.tensor = tensor;
            info.stream = nullptr;
            return info;
        }

        std::vector<float> logits_;
        std::vector<float> mtp_logits_;
        std::vector<float> all_position_logits_;
        std::shared_ptr<FP32Tensor> logits_local_;
        std::shared_ptr<FP32Tensor> mtp_logits_local_;
        std::shared_ptr<FP32Tensor> all_position_logits_local_;
        int forward_call_count_{0};
        int forward_mtp_count_{0};
        int forward_mtp_from_last_draft_count_{0};
        int forward_mtp_and_sample_count_{0};
        int forward_mtp_from_last_draft_and_sample_count_{0};
        int clear_cache_count_{0};
        int restore_count_{0};
        mutable int capture_checkpoint_count_{0};
        int set_all_position_count_{0};
        int commit_mtp_shifted_count_{0};
        int sequential_commit_mtp_shifted_count_{0};
        int last_commit_mtp_already_appended_{0};
        int last_commit_mtp_main_forward_token_count_{0};
        int last_commit_mtp_position_offset_override_{-1};
        int sample_main_logits_count_{0};
        int sample_device_count_{0};
        int sample_mtp_logits_count_{0};
        int sample_all_position_logits_count_{0};
        int sample_all_position_logits_batched_count_{0};
        int apply_main_penalties_count_{0};
        int apply_mtp_penalties_count_{0};
        int apply_all_position_penalties_count_{0};
        int device_distribution_build_count_{0};
        int device_distribution_sample_count_{0};
        int device_distribution_verify_count_{0};
        int device_distribution_verify_batch_count_{0};
        int last_sample_all_position_start_row_{-1};
        int last_sample_all_position_row_count_{0};
        int last_mtp_condition_token_{-1};
        int last_chained_mtp_condition_token_{-1};
        int last_chained_mtp_position_id_{-1};
        bool is_first_forward_in_cycle_{true};
        bool mtp_enabled_{false};
        bool accept_mtp_token_{true};
        bool all_position_logits_enabled_{false};
        bool column_parallel_logits_{false};
        bool supports_mtp_token_coordination_{false};
        bool supports_chained_mtp_drafts_{false};
        bool supports_mtp_sidecar_sample_fusion_{false};
        bool supports_mtp_sidecar_preserves_main_state_{false};
        bool supports_mtp_spec_state_publication_{false};
        bool publish_mtp_spec_state_ok_{true};
        bool supports_stochastic_device_sampling_{false};
        bool requires_mtp_decode_equivalent_replay_{false};
        bool hide_local_logits_{false};
        bool use_captured_snapshot_{false};
        bool last_commit_mtp_allow_speculative_discard_{false};
        DeviceId primary_device_{DeviceId::cpu()};
        int vocab_start_{0};
        int vocab_local_{VOCAB_SIZE};
        std::string mtp_unsupported_reason_;
        PrefixStateSnapshot captured_snapshot_;
        PrefixStateSnapshot last_restored_snapshot_;
        MTPSpecStepPlan last_published_mtp_spec_step_;
        std::vector<int> last_forward_tokens_;
        std::vector<std::vector<int>> forward_history_;
        std::vector<int> last_commit_mtp_tokens_;
        std::vector<int> verifier_accepted_prefix_script_;
        std::vector<int> decode_argmax_script_;
        mutable std::vector<PrefixStateSnapshot> captured_checkpoint_script_;
        std::array<std::vector<SamplingDistributionEntry>, 4> target_device_distributions_;
        std::array<std::vector<SamplingDistributionEntry>, 3> draft_device_distributions_;
        std::vector<SamplingDistributionEntry> invalid_distribution_;
        size_t verifier_accepted_prefix_script_index_{0};
        size_t decode_argmax_script_index_{0};
        mutable size_t captured_checkpoint_script_index_{0};
        int last_forward_seq_len_{0};
        int publish_mtp_spec_state_count_{0};
        int position_{0};
    };

    // =========================================================================
    // Test Fixture
    // =========================================================================

    class Test__PrefillDecodeTransition : public ::testing::Test
    {
    protected:
        struct LocalTPRunnerHarness
        {
            OrchestrationRunner *runner = nullptr;
            MockInferenceRunner *child0 = nullptr;
            MockInferenceRunner *child1 = nullptr;
        };

        void SetUp() override
        {
            // Create a minimal execution plan (single device, full pipeline)
            plan_.rank = 0;
            plan_.hostname = "localhost";
            plan_.numa_node = 0;
            plan_.pp_stage_id = 0;
            plan_.first_layer = 0;
            plan_.last_layer = 23;
            plan_.has_embedding = true;
            plan_.has_lm_head = true;
            plan_.primary_device = GlobalDeviceAddress::cpu();
            // No next_rank/prev_rank → isPipelineTail() = true, isPipelineHead() = true
        }

        /**
         * @brief Create an OrchestrationRunner with the mock runner injected
         */
        std::pair<OrchestrationRunner *, MockInferenceRunner *> createRunner(bool mtp_enabled = false,
                                                                             bool mtp_accept = true,
                                                                             std::string mtp_unsupported_reason = {},
                                                                             std::shared_ptr<IMPIContext> mpi_ctx = nullptr,
                                                                             bool mtp_token_coordination = false,
                                                                             bool hide_local_logits = false,
                                                                             DeviceId primary_device = DeviceId::cpu(),
                                                                             int mtp_draft_tokens = 1,
                                                                             bool chained_mtp_support = false,
                                                                             bool sidecar_sample_fusion = false,
                                                                             MTPDepthPolicyConfig depth_policy = {},
                                                                             MTPVerifyMode verify_mode = MTPVerifyMode::Greedy)
        {
            auto mock = std::make_unique<MockInferenceRunner>();
            auto *mock_ptr = mock.get(); // Keep raw pointer for inspection
            if (mtp_enabled)
            {
                mock_ptr->enableMTP(mtp_accept);
            }
            if (chained_mtp_support)
            {
                mock_ptr->enableChainedMTPDrafts();
            }
            if (sidecar_sample_fusion)
            {
                mock_ptr->enableMTPSidecarSampleFusion();
            }
            mock_ptr->setMTPUnsupportedReason(std::move(mtp_unsupported_reason));
            mock_ptr->setPrimaryDevice(primary_device);
            if (mtp_token_coordination)
            {
                mock_ptr->enableMTPTokenCoordination(hide_local_logits);
            }

            OrchestrationConfig config;
            if (primary_device.is_rocm())
                config.device_for_this_rank = GlobalDeviceAddress::rocm(primary_device.ordinal);
            else if (primary_device.is_cuda())
                config.device_for_this_rank = GlobalDeviceAddress::cuda(primary_device.ordinal);
            else
                config.device_for_this_rank = GlobalDeviceAddress::cpu();
            config.mtp.enabled = mtp_enabled;
            config.mtp.draft_tokens = mtp_draft_tokens;
            config.mtp.verify_mode = verify_mode;
            config.mtp.depth_policy = depth_policy;

            std::unique_ptr<OrchestrationRunner> runner;
            if (mpi_ctx)
            {
                runner = std::make_unique<OrchestrationRunner>(
                    std::move(config), plan_, std::move(mock), std::move(mpi_ctx));
            }
            else
            {
                runner = std::make_unique<OrchestrationRunner>(
                    std::move(config), plan_, std::move(mock));
            }

            // Set greedy sampling (temperature=0)
            SamplingParams greedy;
            greedy.temperature = 0.0f;
            runner->setSamplingParams(greedy);

            runners_.push_back(std::move(runner));
            return {runners_.back().get(), mock_ptr};
        }

        LocalTPRunnerHarness createLocalTPRunner(bool mtp_accept = true,
                                                 bool column_parallel_logits = false,
                                                 std::vector<GlobalDeviceAddress> devices = {})
        {
            if (devices.empty())
            {
                devices = {GlobalDeviceAddress::cpu(), GlobalDeviceAddress::cpu()};
            }
            auto child0 = std::make_unique<MockInferenceRunner>();
            auto child1 = std::make_unique<MockInferenceRunner>();
            child0->enableMTP(mtp_accept);
            child1->enableMTP(mtp_accept);
            child0->setPrimaryDevice(devices[0].toLocalDeviceId());
            child1->setPrimaryDevice(devices[1].toLocalDeviceId());
            if (column_parallel_logits)
            {
                child0->enableColumnParallelShard(0, MockInferenceRunner::VOCAB_SIZE / 2);
                child1->enableColumnParallelShard(MockInferenceRunner::VOCAB_SIZE / 2,
                                                  MockInferenceRunner::VOCAB_SIZE / 2);
            }

            auto *child0_ptr = child0.get();
            auto *child1_ptr = child1.get();

            std::vector<std::unique_ptr<IInferenceRunner>> device_runners;
            device_runners.push_back(std::move(child0));
            device_runners.push_back(std::move(child1));

            RankOrchestrator::Config rank_config;
            rank_config.mode = RankOrchestrator::ParallelismMode::TP;
            rank_config.devices = devices;
            rank_config.mtp.enabled = true;
            rank_config.mtp.draft_tokens = 1;
            rank_config.mtp.verify_mode = MTPVerifyMode::Greedy;

            auto model_ctx = test::MockModelContext::createMinimal();
            model_ctx->setVocabSize(MockInferenceRunner::VOCAB_SIZE);

            auto rank_runner = RankOrchestrator::createForTest(
                std::move(model_ctx),
                std::move(device_runners),
                nullptr,
                rank_config);

            OrchestrationConfig config;
            config.device_for_this_rank = devices.front();
            config.mtp.enabled = true;
            config.mtp.draft_tokens = 1;
            config.mtp.verify_mode = MTPVerifyMode::Greedy;

            RankExecutionPlan runner_plan = plan_;
            runner_plan.primary_device = devices.front();
            runner_plan.local_tp_devices = devices;
            runner_plan.local_tp_backend = devices.front().isROCm()
                                               ? CollectiveBackendType::RCCL
                                               : CollectiveBackendType::HOST;

            auto runner = std::make_unique<OrchestrationRunner>(
                std::move(config), std::move(runner_plan), std::move(rank_runner));

            SamplingParams greedy;
            greedy.temperature = 0.0f;
            runner->setSamplingParams(greedy);

            runners_.push_back(std::move(runner));
            return {runners_.back().get(), child0_ptr, child1_ptr};
        }

        static GlobalPPTopology buildSingleStageGlobalTPTopo(int world_size)
        {
            GlobalPPStageSpec stage;
            stage.stage_id = 0;
            stage.first_layer = 0;
            stage.last_layer = 23;
            stage.has_embedding = true;
            stage.has_lm_head = true;
            stage.is_global_tp = true;
            for (int rank = 0; rank < world_size; ++rank)
            {
                stage.participating_ranks.push_back(rank);
            }
            stage.per_rank_device = GlobalDeviceAddress::cpu();
            return GlobalPPTopology::build({stage}, 24, world_size);
        }

        RankExecutionPlan plan_;
        std::vector<std::unique_ptr<OrchestrationRunner>> runners_; // Prevent dangling
    };

    // =========================================================================
    // Core Regression Tests
    // =========================================================================

    /**
     * @brief Verify that prefill calls forward with full prompt tokens
     */
    TEST_F(Test__PrefillDecodeTransition, PrefillCallsForwardWithFullPrompt)
    {
        auto [runner, mock] = createRunner();

        std::vector<int32_t> prompt = {1, 2, 3, 4, 5};
        ASSERT_TRUE(runner->prefill(prompt));

        EXPECT_EQ(mock->forwardCallCount(), 1);
        EXPECT_EQ(mock->lastForwardSeqLen(), 5);
        EXPECT_THAT(mock->lastForwardTokens(), ElementsAre(1, 2, 3, 4, 5));
    }

    /**
     * @brief REGRESSION: First decodeStep after prefill must NOT call forward()
     *
     * This is the core regression test. Before the fix, the first decodeStep()
     * called forward(&last_token_, 1) which re-processed the last prompt token
     * at position N+1, corrupting GDN recurrence state.
     */
    TEST_F(Test__PrefillDecodeTransition, FirstDecodeStepSkipsForward)
    {
        auto [runner, mock] = createRunner();

        std::vector<int32_t> prompt = {1, 2, 3, 4, 5};
        ASSERT_TRUE(runner->prefill(prompt));
        EXPECT_EQ(mock->forwardCallCount(), 1); // Only prefill forward

        // First decode step should NOT call forward
        GenerationResult step1 = runner->decodeStep();
        ASSERT_TRUE(step1.success());
        EXPECT_EQ(mock->forwardCallCount(), 1); // Still 1 — no additional forward
    }

    /**
     * @brief First decodeStep samples from prefill logits (argmax = token 7)
     */
    TEST_F(Test__PrefillDecodeTransition, FirstDecodeStepSamplesFromPrefillLogits)
    {
        auto [runner, mock] = createRunner();

        std::vector<int32_t> prompt = {1, 2, 3, 4, 5};
        ASSERT_TRUE(runner->prefill(prompt));

        GenerationResult step1 = runner->decodeStep();
        ASSERT_TRUE(step1.success());
        ASSERT_EQ(step1.tokens.size(), 1u);
        EXPECT_EQ(step1.tokens[0], MockInferenceRunner::PREFILL_ARGMAX_TOKEN);
    }

    /**
     * @brief Second decodeStep DOES call forward with the token from step 1
     */
    TEST_F(Test__PrefillDecodeTransition, SecondDecodeStepCallsForward)
    {
        auto [runner, mock] = createRunner();

        std::vector<int32_t> prompt = {1, 2, 3, 4, 5};
        ASSERT_TRUE(runner->prefill(prompt));

        // First decode: samples from prefill logits (token 7), no forward
        GenerationResult step1 = runner->decodeStep();
        ASSERT_TRUE(step1.success());
        EXPECT_EQ(mock->forwardCallCount(), 1);

        // Second decode: MUST call forward with token 7
        GenerationResult step2 = runner->decodeStep();
        ASSERT_TRUE(step2.success());
        EXPECT_EQ(mock->forwardCallCount(), 2); // Now 2 — prefill + decode
        EXPECT_EQ(mock->lastForwardSeqLen(), 1);
        EXPECT_THAT(mock->lastForwardTokens(),
                    ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN));
    }

    /**
     * @brief Second decodeStep samples from decode logits (argmax = token 3)
     */
    TEST_F(Test__PrefillDecodeTransition, SecondDecodeStepSamplesFromDecodeLogits)
    {
        auto [runner, mock] = createRunner();

        std::vector<int32_t> prompt = {1, 2, 3, 4, 5};
        ASSERT_TRUE(runner->prefill(prompt));

        // First decode: token 7 from prefill logits
        GenerationResult step1 = runner->decodeStep();
        ASSERT_TRUE(step1.success());
        EXPECT_EQ(step1.tokens[0], MockInferenceRunner::PREFILL_ARGMAX_TOKEN);

        // Second decode: token 3 from decode logits (after forward with token 7)
        GenerationResult step2 = runner->decodeStep();
        ASSERT_TRUE(step2.success());
        ASSERT_EQ(step2.tokens.size(), 1u);
        EXPECT_EQ(step2.tokens[0], MockInferenceRunner::DECODE_ARGMAX_TOKEN);
    }

    TEST_F(Test__PrefillDecodeTransition, MTPFirstDecodeAcceptsGreedyDraftAndCommitsVerifierState)
    {
        auto [runner, mock] = createRunner(/*mtp_enabled=*/true, /*mtp_accept=*/true);

        std::vector<int32_t> prompt = {1, 2, 3, 4, 5};
        ASSERT_TRUE(runner->prefill(prompt));

        GenerationResult step1 = runner->decodeStep();
        ASSERT_TRUE(step1.success());
        EXPECT_THAT(step1.tokens,
                    ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN,
                                MockInferenceRunner::MTP_ARGMAX_TOKEN));
        EXPECT_EQ(mock->forwardMTPCount(), 1);
        EXPECT_EQ(mock->lastMTPConditionToken(), MockInferenceRunner::PREFILL_ARGMAX_TOKEN);
        EXPECT_EQ(mock->commitMTPShiftedCount(), 2);
        EXPECT_EQ(mock->lastCommitMTPAlreadyAppended(), 1);
        EXPECT_THAT(mock->lastCommitMTPTokens(),
                    ElementsAre(MockInferenceRunner::MTP_ARGMAX_TOKEN));
        EXPECT_EQ(mock->restoreCount(), 1);
        EXPECT_EQ(mock->setAllPositionCount(), 0);
        EXPECT_THAT(mock->lastForwardTokens(),
                    ElementsAre(MockInferenceRunner::MTP_ARGMAX_TOKEN));
        EXPECT_EQ(mock->captureCheckpointCount(), 2);

        const auto probe = runner->prefixStateProbe();
        EXPECT_EQ(probe.mtp_draft_steps, 1u);
        EXPECT_EQ(probe.mtp_verifier_runs, 1u);
        EXPECT_EQ(probe.mtp_verifier_token_count, 2u);
        EXPECT_EQ(probe.mtp_accepted_tokens, 1u);
        EXPECT_EQ(probe.mtp_rejected_tokens, 0u);
        EXPECT_EQ(probe.mtp_rollbacks, 0u);
        EXPECT_EQ(probe.mtp_transaction_commits, 1u);
        EXPECT_EQ(probe.mtp_transaction_rollbacks, 0u);
        EXPECT_EQ(probe.mtp_transaction_validation_failures, 0u);
    }

    TEST_F(Test__PrefillDecodeTransition, MTPGreedyPenaltiesHardFailBeforeSidecar)
    {
        auto [runner, mock] = createRunner(
            /*mtp_enabled=*/true,
            /*mtp_accept=*/true,
            /*mtp_unsupported_reason=*/{},
            /*mpi_ctx=*/nullptr,
            /*mtp_token_coordination=*/true);

        SamplingParams sampling;
        sampling.temperature = 0.0f;
        sampling.presence_penalty = 1.0f;
        runner->setSamplingParams(sampling);

        ASSERT_TRUE(runner->prefill({1, 2, 3, 4, 5}));

        GenerationResult step1 = runner->decodeStep();
        ASSERT_FALSE(step1.success());
        EXPECT_THAT(step1.error, HasSubstr("greedy_penalty_mtp_requires_new_transaction_path"));
        EXPECT_EQ(mock->forwardMTPCount(), 0);
        EXPECT_EQ(mock->applyMainPenaltiesCount(), 0);
        EXPECT_EQ(mock->applyMTPPenaltiesCount(), 0);
        EXPECT_EQ(mock->applyAllPositionPenaltiesCount(), 0);

        const auto probe = runner->prefixStateProbe();
        EXPECT_FALSE(probe.mtp_bypassed);
        EXPECT_EQ(probe.mtp_bypasses, 0u);
        EXPECT_EQ(probe.mtp_draft_steps, 0u);
        EXPECT_EQ(probe.mtp_accepted_tokens, 0u);
    }

    TEST_F(Test__PrefillDecodeTransition, MTPDraftDepthGreaterThanOneHardFailsBeforePrefillForward)
    {
        auto [runner, mock] = createRunner(
            /*mtp_enabled=*/true,
            /*mtp_accept=*/true,
            /*mtp_unsupported_reason=*/{},
            /*mpi_ctx=*/nullptr,
            /*mtp_token_coordination=*/false,
            /*hide_local_logits=*/false,
            DeviceId::cpu(),
            /*mtp_draft_tokens=*/2);

        EXPECT_FALSE(runner->prefill({1, 2, 3, 4, 5}));
        EXPECT_NE(runner->lastError().find("requires runner support for chained MTP sidecars"), std::string::npos)
            << runner->lastError();
        EXPECT_EQ(mock->forwardMTPCount(), 0);
        EXPECT_EQ(mock->forwardCallCount(), 0);
    }

    TEST_F(Test__PrefillDecodeTransition, MTPChainedDraftCapturesOnlyFirstPostSidecarCheckpoint)
    {
        const std::filesystem::path export_path =
            std::filesystem::temp_directory_path() / "llaminar_mtp_chained_checkpoint_unit.json";
        {
            ScopedEnv enable("LLAMINAR_PERF_STATS_JSON", export_path.string().c_str());
            PerfStatsCollector::reset();

            auto [runner, mock] = createRunner(
                /*mtp_enabled=*/true,
                /*mtp_accept=*/true,
                /*mtp_unsupported_reason=*/{},
                /*mpi_ctx=*/nullptr,
                /*mtp_token_coordination=*/false,
                /*hide_local_logits=*/false,
                DeviceId::cpu(),
                /*mtp_draft_tokens=*/3,
                /*chained_mtp_support=*/true);

            ASSERT_TRUE(runner->prefill({1, 2, 3, 4, 5}));

            GenerationResult step1 = runner->decodeStep();
            ASSERT_TRUE(step1.success()) << step1.error;

            EXPECT_EQ(mock->forwardMTPCount(), 1);
            EXPECT_EQ(mock->forwardMTPFromLastDraftCount(), 2);
            EXPECT_EQ(mock->captureCheckpointCount(), 2)
                << "only the live checkpoint plus the first post-sidecar checkpoint are restorable";

            const auto records = PerfStatsCollector::snapshot({"mtp"});
            const PerfStatRecord *post_sidecar_capture =
                findPerfRecord(records, PerfStatRecord::Kind::Timer, "capture_post_sidecar_prefix_state");
            ASSERT_NE(post_sidecar_capture, nullptr);
            EXPECT_EQ(post_sidecar_capture->count, 1u);

            const PerfStatRecord *skipped_speculative =
                findPerfRecord(records, PerfStatRecord::Kind::Counter, "post_sidecar_checkpoint_skipped_speculative");
            ASSERT_NE(skipped_speculative, nullptr);
            EXPECT_DOUBLE_EQ(skipped_speculative->value, 2.0);
        }
        std::filesystem::remove(export_path);
        PerfStatsCollector::reset();
    }

    TEST_F(Test__PrefillDecodeTransition, MTPSidecarSampleFusionUsesCombinedFirstAndChainedDraftCalls)
    {
        const std::filesystem::path export_path =
            std::filesystem::temp_directory_path() / "llaminar_mtp_sidecar_sample_fusion_unit.json";
        {
            ScopedEnv enable("LLAMINAR_PERF_STATS_JSON", export_path.string().c_str());
            PerfStatsCollector::reset();

            auto [runner, mock] = createRunner(
                /*mtp_enabled=*/true,
                /*mtp_accept=*/true,
                /*mtp_unsupported_reason=*/{},
                /*mpi_ctx=*/nullptr,
                /*mtp_token_coordination=*/true,
                /*hide_local_logits=*/true,
                DeviceId::cpu(),
                /*mtp_draft_tokens=*/3,
                /*chained_mtp_support=*/true,
                /*sidecar_sample_fusion=*/true);

            ASSERT_TRUE(runner->prefill({1, 2, 3, 4, 5}));

            GenerationResult step1 = runner->decodeStep();
            ASSERT_TRUE(step1.success()) << step1.error;

            EXPECT_EQ(mock->forwardMTPAndSampleCount(), 1);
            EXPECT_EQ(mock->forwardMTPFromLastDraftAndSampleCount(), 2);
            EXPECT_EQ(mock->forwardMTPCount(), 1);
            EXPECT_EQ(mock->forwardMTPFromLastDraftCount(), 2);
            EXPECT_EQ(mock->sampleMTPLogitsCount(), 0)
                << "the orchestrator should not do a separate MTP logits sample after fused calls";

            const auto records = PerfStatsCollector::snapshot({"mtp"});
            const PerfStatRecord *device_samples =
                findPerfRecord(records, PerfStatRecord::Kind::Counter, "mtp_token_device_samples");
            ASSERT_NE(device_samples, nullptr);
            EXPECT_DOUBLE_EQ(device_samples->value, 3.0);

            const PerfStatRecord *sidecar_timer =
                findPerfRecord(records, PerfStatRecord::Kind::Timer, "sidecar_forward");
            ASSERT_NE(sidecar_timer, nullptr);
            EXPECT_EQ(sidecar_timer->count, 3u);

            const PerfStatRecord *host_sample =
                findPerfRecord(records, PerfStatRecord::Kind::Timer, "sample_mtp_token_host");
            EXPECT_EQ(host_sample, nullptr);
        }
        std::filesystem::remove(export_path);
        PerfStatsCollector::reset();
    }

    TEST_F(Test__PrefillDecodeTransition, CUDAMTPUsesSequentialGreedyVerifierInsteadOfAllPositionReplay)
    {
        const std::filesystem::path export_path =
            std::filesystem::temp_directory_path() / "llaminar_mtp_cuda_sequential_verifier_unit.json";
        {
            ScopedEnv enable("LLAMINAR_PERF_STATS_JSON", export_path.string().c_str());
            PerfStatsCollector::reset();

            auto [runner, mock] = createRunner(
                /*mtp_enabled=*/true,
                /*mtp_accept=*/true,
                /*mtp_unsupported_reason=*/{},
                /*mpi_ctx=*/nullptr,
                /*mtp_token_coordination=*/true,
                /*hide_local_logits=*/false,
                DeviceId::cuda(0),
                /*mtp_draft_tokens=*/2,
                /*chained_mtp_support=*/true,
                /*sidecar_sample_fusion=*/true);
            mock->requireMTPDecodeEquivalentReplay();
            mock->setDecodeArgmaxScript({
                MockInferenceRunner::MTP_ARGMAX_TOKEN,
                MockInferenceRunner::MTP_ARGMAX_TOKEN,
                MockInferenceRunner::DECODE_ARGMAX_TOKEN,
            });

            ASSERT_TRUE(runner->prefill({1, 2, 3, 4, 5}));

            GenerationResult step1 = runner->decodeStep();
            ASSERT_TRUE(step1.success()) << step1.error;
            EXPECT_THAT(step1.tokens,
                        ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN,
                                    MockInferenceRunner::MTP_ARGMAX_TOKEN,
                                    MockInferenceRunner::MTP_ARGMAX_TOKEN));

            EXPECT_EQ(mock->setAllPositionCount(), 0)
                << "CUDA dense must not enter the unsafe multi-row verifier shortcut path";
            EXPECT_EQ(mock->sampleAllPositionLogitsCount(), 0);
            EXPECT_EQ(mock->sampleAllPositionLogitsBatchedCount(), 0);
            EXPECT_EQ(mock->restoreCount(), 1)
                << "depth>1 speculative sidecar rows are discarded back to the first sidecar checkpoint";
            EXPECT_EQ(mock->sequentialCommitMTPShiftedCount(), 3);
            EXPECT_TRUE(mock->lastCommitMTPAllowSpeculativeDiscard());
            EXPECT_EQ(mock->lastCommitMTPPositionOffsetOverride(), 5);

            ASSERT_GE(mock->forwardHistory().size(), 4u);
            EXPECT_THAT(mock->forwardHistory()[0], ElementsAre(1, 2, 3, 4, 5));
            EXPECT_THAT(mock->forwardHistory()[1], ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN));
            EXPECT_THAT(mock->forwardHistory()[2], ElementsAre(MockInferenceRunner::MTP_ARGMAX_TOKEN));
            EXPECT_THAT(mock->forwardHistory()[3], ElementsAre(MockInferenceRunner::MTP_ARGMAX_TOKEN));

            const auto probe = runner->prefixStateProbe();
            EXPECT_EQ(probe.mtp_draft_steps, 2u);
            EXPECT_EQ(probe.mtp_verifier_runs, 1u);
            EXPECT_EQ(probe.mtp_verifier_token_count, 3u);
            EXPECT_EQ(probe.mtp_accepted_tokens, 2u);
            EXPECT_EQ(probe.mtp_rejected_tokens, 0u);
            EXPECT_EQ(probe.mtp_rollbacks, 0u);

            const auto records = PerfStatsCollector::snapshot({"mtp"});
            const PerfStatRecord *catchup =
                findPerfRecordWithTags(records,
                                       PerfStatRecord::Kind::Counter,
                                       "decode_equivalent_catchup_runs",
                                       {{"implementation", "shared_stepwise"},
                                        {"draft_tokens", "7,9,9"},
                                        {"accepted_tokens", "7,9,9"},
                                        {"verifier_tokens", "9,9"}});
            ASSERT_NE(catchup, nullptr);
            const PerfStatRecord *verifier_forward =
                findPerfRecordWithTags(records,
                                       PerfStatRecord::Kind::Timer,
                                       "verifier_forward",
                                       {{"implementation", "shared_stepwise"},
                                        {"verifier_path", "decode_equivalent_catchup"}});
            ASSERT_NE(verifier_forward, nullptr);
        }
        std::filesystem::remove(export_path);
        PerfStatsCollector::reset();
    }

    TEST_F(Test__PrefillDecodeTransition, CUDAMTPSequentialVerifierSkipsBaseRestoreWhenSidecarPreservesMainState)
    {
        const std::filesystem::path export_path =
            std::filesystem::temp_directory_path() / "llaminar_mtp_cuda_sidecar_preserved_unit.json";
        {
            ScopedEnv enable("LLAMINAR_PERF_STATS_JSON", export_path.string().c_str());
            PerfStatsCollector::reset();

            auto [runner, mock] = createRunner(
                /*mtp_enabled=*/true,
                /*mtp_accept=*/true,
                /*mtp_unsupported_reason=*/{},
                /*mpi_ctx=*/nullptr,
                /*mtp_token_coordination=*/true,
                /*hide_local_logits=*/false,
                DeviceId::cuda(0),
                /*mtp_draft_tokens=*/1,
                /*chained_mtp_support=*/false,
                /*sidecar_sample_fusion=*/true);
            mock->enableMTPSidecarPreservesMainState();
            mock->requireMTPDecodeEquivalentReplay();
            mock->setDecodeArgmaxScript({
                MockInferenceRunner::MTP_ARGMAX_TOKEN,
                MockInferenceRunner::DECODE_ARGMAX_TOKEN,
            });

            ASSERT_TRUE(runner->prefill({1, 2, 3, 4, 5}));

            GenerationResult step1 = runner->decodeStep();
            ASSERT_TRUE(step1.success()) << step1.error;
            EXPECT_THAT(step1.tokens,
                        ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN,
                                    MockInferenceRunner::MTP_ARGMAX_TOKEN));

            EXPECT_EQ(mock->restoreCount(), 0)
                << "graph-native sidecar execution preserves main verifier state, so "
                   "the CUDA sequential verifier should not restore the base checkpoint";
            EXPECT_EQ(mock->captureCheckpointCount(), 2)
                << "the post-sidecar checkpoint is still captured for non-row-restore "
                   "runners; only the redundant base restore is skipped";
            EXPECT_EQ(mock->sequentialCommitMTPShiftedCount(), 2);
            EXPECT_EQ(mock->forwardCallCount(), 3);
            EXPECT_THAT(mock->forwardHistory()[1], ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN));
            EXPECT_THAT(mock->forwardHistory()[2], ElementsAre(MockInferenceRunner::MTP_ARGMAX_TOKEN));

            const auto records = PerfStatsCollector::snapshot({"mtp"});
            const PerfStatRecord *catchup =
                findPerfRecordWithTags(records,
                                       PerfStatRecord::Kind::Counter,
                                       "decode_equivalent_catchup_runs",
                                       {{"implementation", "shared_stepwise"},
                                        {"draft_tokens", "7,9"},
                                        {"accepted_tokens", "7,9"},
                                        {"verifier_tokens", "9"}});
            ASSERT_NE(catchup, nullptr);

            const PerfStatRecord *restore_counter =
                findPerfRecord(records, PerfStatRecord::Kind::Counter, "decode_equivalent_sequential_verifier_base_restores");
            EXPECT_EQ(restore_counter, nullptr);
            const PerfStatRecord *restore_timer =
                findPerfRecord(records, PerfStatRecord::Kind::Timer, "decode_equivalent_sequential_verifier_restore_base_checkpoint");
            EXPECT_EQ(restore_timer, nullptr);
            const PerfStatRecord *skipped_restore =
                findPerfRecord(records,
                               PerfStatRecord::Kind::Counter,
                               "decode_equivalent_sequential_verifier_base_restore_skipped_sidecar_preserved");
            ASSERT_NE(skipped_restore, nullptr);
            EXPECT_DOUBLE_EQ(skipped_restore->value, 1.0);
        }
        std::filesystem::remove(export_path);
        PerfStatsCollector::reset();
    }

    TEST_F(Test__PrefillDecodeTransition, DynamicMTPDepthDemotesAfterZeroAcceptWindow)
    {
        MTPDepthPolicyConfig depth_policy;
        depth_policy.mode = MTPDepthPolicyMode::Dynamic;
        depth_policy.min_depth = 1;
        depth_policy.max_depth = 3;
        depth_policy.initial_depth = 3;
        depth_policy.window_size = 1;
        depth_policy.min_samples = 1;
        depth_policy.cooldown_steps = 0;
        depth_policy.demote_zero_accept_rate = 0.30;

        auto [runner, mock] = createRunner(
            /*mtp_enabled=*/true,
            /*mtp_accept=*/true,
            /*mtp_unsupported_reason=*/{},
            /*mpi_ctx=*/nullptr,
            /*mtp_token_coordination=*/false,
            /*hide_local_logits=*/false,
            DeviceId::cpu(),
            /*mtp_draft_tokens=*/3,
            /*chained_mtp_support=*/true,
            /*sidecar_sample_fusion=*/false,
            depth_policy);
        mock->setDecodeArgmaxScript({
            MockInferenceRunner::VERIFY_REJECT_TOKEN,
            MockInferenceRunner::DECODE_ARGMAX_TOKEN,
            MockInferenceRunner::VERIFY_REJECT_TOKEN,
            MockInferenceRunner::DECODE_ARGMAX_TOKEN});

        ASSERT_TRUE(runner->prefill({1, 2, 3, 4, 5}));

        GenerationResult step1 = runner->decodeStep();
        ASSERT_TRUE(step1.success()) << step1.error;
        EXPECT_EQ(mock->forwardMTPCount(), 1);
        EXPECT_EQ(mock->forwardMTPFromLastDraftCount(), 2);
        auto probe = runner->prefixStateProbe();
        EXPECT_EQ(probe.mtp_current_depth, 2);
        EXPECT_EQ(probe.mtp_depth_policy_demotions, 1u);
        EXPECT_EQ(probe.mtp_depth_policy_updates, 1u);

        GenerationResult step2 = runner->decodeStep();
        ASSERT_TRUE(step2.success()) << step2.error;
        EXPECT_EQ(mock->forwardMTPCount(), 2);
        EXPECT_EQ(mock->forwardMTPFromLastDraftCount(), 3)
            << "second step should use depth 2 after the first demotion";
        probe = runner->prefixStateProbe();
        EXPECT_EQ(probe.mtp_current_depth, 1);
        EXPECT_EQ(probe.mtp_depth_policy_demotions, 2u);
        EXPECT_EQ(probe.mtp_request.depth_policy_mode, "dynamic");
        EXPECT_TRUE(probe.mtp_request.adaptive_depth_enabled);
        EXPECT_EQ(probe.mtp_request.last_depth_policy_reason, "demote_zero_accept_rate");
    }

    TEST_F(Test__PrefillDecodeTransition, DynamicMTPDepthPersistsAcrossClearCachePrefillCycles)
    {
        MTPDepthPolicyConfig depth_policy;
        depth_policy.mode = MTPDepthPolicyMode::Dynamic;
        depth_policy.min_depth = 1;
        depth_policy.max_depth = 3;
        depth_policy.initial_depth = 3;
        depth_policy.window_size = 1;
        depth_policy.min_samples = 1;
        depth_policy.cooldown_steps = 0;
        depth_policy.demote_zero_accept_rate = 0.30;

        auto [runner, mock] = createRunner(
            /*mtp_enabled=*/true,
            /*mtp_accept=*/true,
            /*mtp_unsupported_reason=*/{},
            /*mpi_ctx=*/nullptr,
            /*mtp_token_coordination=*/false,
            /*hide_local_logits=*/false,
            DeviceId::cpu(),
            /*mtp_draft_tokens=*/3,
            /*chained_mtp_support=*/true,
            /*sidecar_sample_fusion=*/false,
            depth_policy);
        mock->setDecodeArgmaxScript({
            MockInferenceRunner::VERIFY_REJECT_TOKEN,
            MockInferenceRunner::DECODE_ARGMAX_TOKEN,
            MockInferenceRunner::VERIFY_REJECT_TOKEN,
            MockInferenceRunner::DECODE_ARGMAX_TOKEN});

        ASSERT_TRUE(runner->prefill({1, 2, 3, 4, 5}));
        GenerationResult step1 = runner->decodeStep();
        ASSERT_TRUE(step1.success()) << step1.error;
        EXPECT_EQ(mock->forwardMTPCount(), 1);
        EXPECT_EQ(mock->forwardMTPFromLastDraftCount(), 2);

        auto probe = runner->prefixStateProbe();
        EXPECT_EQ(probe.mtp_current_depth, 2);
        EXPECT_EQ(probe.mtp_depth_policy_demotions, 1u);

        runner->clearCache();
        ASSERT_TRUE(runner->prefill({6, 7, 8, 9, 10}));
        GenerationResult step2 = runner->decodeStep();
        ASSERT_TRUE(step2.success()) << step2.error;

        EXPECT_EQ(mock->forwardMTPCount(), 2);
        EXPECT_EQ(mock->forwardMTPFromLastDraftCount(), 3)
            << "the second request should start from the learned depth 2, not reset to depth 3";

        probe = runner->prefixStateProbe();
        EXPECT_EQ(probe.mtp_current_depth, 1);
        EXPECT_EQ(probe.mtp_depth_policy_demotions, 1u);
        EXPECT_EQ(probe.mtp_depth_policy_updates, 1u);
    }

    TEST_F(Test__PrefillDecodeTransition, DynamicMTPDepthPromotesAfterFullAcceptWindows)
    {
        MTPDepthPolicyConfig depth_policy;
        depth_policy.mode = MTPDepthPolicyMode::Dynamic;
        depth_policy.min_depth = 1;
        depth_policy.max_depth = 3;
        depth_policy.initial_depth = 1;
        depth_policy.window_size = 1;
        depth_policy.min_samples = 1;
        depth_policy.cooldown_steps = 0;
        depth_policy.promote_consecutive_windows = 1;
        depth_policy.promote_full_accept_rate = 0.75;

        auto [runner, mock] = createRunner(
            /*mtp_enabled=*/true,
            /*mtp_accept=*/true,
            /*mtp_unsupported_reason=*/{},
            /*mpi_ctx=*/nullptr,
            /*mtp_token_coordination=*/false,
            /*hide_local_logits=*/false,
            DeviceId::cpu(),
            /*mtp_draft_tokens=*/3,
            /*chained_mtp_support=*/true,
            /*sidecar_sample_fusion=*/false,
            depth_policy);
        mock->setVerifierAcceptedPrefixScript({1, 2});

        ASSERT_TRUE(runner->prefill({1, 2, 3, 4, 5}));

        GenerationResult step1 = runner->decodeStep();
        ASSERT_TRUE(step1.success()) << step1.error;
        EXPECT_EQ(mock->forwardMTPCount(), 1);
        EXPECT_EQ(mock->forwardMTPFromLastDraftCount(), 0);
        auto probe = runner->prefixStateProbe();
        EXPECT_EQ(probe.mtp_current_depth, 2);
        EXPECT_EQ(probe.mtp_depth_policy_promotions, 1u);

        GenerationResult step2 = runner->decodeStep();
        ASSERT_TRUE(step2.success()) << step2.error;
        EXPECT_EQ(mock->forwardMTPCount(), 2);
        EXPECT_EQ(mock->forwardMTPFromLastDraftCount(), 1)
            << "second step should use depth 2 before promoting to depth 3";
        probe = runner->prefixStateProbe();
        EXPECT_EQ(probe.mtp_current_depth, 3);
        EXPECT_EQ(probe.mtp_depth_policy_promotions, 2u);
        EXPECT_EQ(probe.mtp_depth_policy_updates, 2u);
        EXPECT_EQ(probe.mtp_request.last_depth_policy_reason, "promote_full_accept_rate");
    }

    TEST_F(Test__PrefillDecodeTransition, FixedMTPDepthRemainsHardPinned)
    {
        MTPDepthPolicyConfig depth_policy;
        depth_policy.mode = MTPDepthPolicyMode::Fixed;
        depth_policy.min_depth = 1;
        depth_policy.max_depth = 1;
        depth_policy.initial_depth = 1;
        depth_policy.window_size = 1;
        depth_policy.min_samples = 1;
        depth_policy.cooldown_steps = 0;

        auto [runner, mock] = createRunner(
            /*mtp_enabled=*/true,
            /*mtp_accept=*/true,
            /*mtp_unsupported_reason=*/{},
            /*mpi_ctx=*/nullptr,
            /*mtp_token_coordination=*/false,
            /*hide_local_logits=*/false,
            DeviceId::cpu(),
            /*mtp_draft_tokens=*/3,
            /*chained_mtp_support=*/true,
            /*sidecar_sample_fusion=*/false,
            depth_policy);
        mock->setVerifierAcceptedPrefixScript({0});

        ASSERT_TRUE(runner->prefill({1, 2, 3, 4, 5}));
        GenerationResult step1 = runner->decodeStep();
        ASSERT_TRUE(step1.success()) << step1.error;

        EXPECT_EQ(mock->forwardMTPCount(), 1);
        EXPECT_EQ(mock->forwardMTPFromLastDraftCount(), 2)
            << "fixed depth must use the hard-pinned --mtp-draft-tokens value";
        const auto probe = runner->prefixStateProbe();
        EXPECT_EQ(probe.mtp_current_depth, 3);
        EXPECT_EQ(probe.mtp_min_depth, 3);
        EXPECT_EQ(probe.mtp_max_depth, 3);
        EXPECT_EQ(probe.mtp_depth_policy_windows, 0u);
        EXPECT_EQ(probe.mtp_depth_policy_updates, 0u);
        EXPECT_FALSE(probe.mtp_request.adaptive_depth_enabled);
    }

    TEST_F(Test__PrefillDecodeTransition, MTPChainedFirstSpecRejectReplaysReturnedCorrection)
    {
        const std::filesystem::path export_path =
            std::filesystem::temp_directory_path() / "llaminar_mtp_chained_reject_lag_unit.json";
        {
            ScopedEnv enable("LLAMINAR_PERF_STATS_JSON", export_path.string().c_str());
            PerfStatsCollector::reset();

            auto [runner, mock] = createRunner(
                /*mtp_enabled=*/true,
                /*mtp_accept=*/false,
                /*mtp_unsupported_reason=*/{},
                /*mpi_ctx=*/nullptr,
                /*mtp_token_coordination=*/false,
                /*hide_local_logits=*/false,
                DeviceId::cpu(),
                /*mtp_draft_tokens=*/2,
                /*chained_mtp_support=*/true);

            ASSERT_TRUE(runner->prefill({1, 2, 3, 4, 5}));

            GenerationResult step1 = runner->decodeStep();
            ASSERT_TRUE(step1.success()) << step1.error;
            EXPECT_THAT(step1.tokens,
                        ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN,
                                    MockInferenceRunner::VERIFY_REJECT_TOKEN));

            EXPECT_EQ(mock->forwardMTPCount(), 1);
            EXPECT_EQ(mock->forwardMTPFromLastDraftCount(), 1);
            EXPECT_EQ(mock->restoreCount(), 1);
            EXPECT_EQ(mock->captureCheckpointCount(), 2);
            EXPECT_EQ(mock->commitMTPShiftedCount(), 2);
            EXPECT_EQ(mock->lastCommitMTPAlreadyAppended(), 1);
            EXPECT_EQ(mock->lastCommitMTPMainForwardTokenCount(), 0);
            EXPECT_THAT(mock->lastCommitMTPTokens(),
                        ElementsAre(MockInferenceRunner::VERIFY_REJECT_TOKEN));
            EXPECT_THAT(mock->lastForwardTokens(),
                        ElementsAre(MockInferenceRunner::VERIFY_REJECT_TOKEN));

            const auto records = PerfStatsCollector::snapshot({"mtp"});
            const PerfStatRecord *reject_trace =
                findPerfRecordWithTags(records, PerfStatRecord::Kind::Counter, "acceptance_trace",
                                       {
                                           {"first_token", std::to_string(MockInferenceRunner::PREFILL_ARGMAX_TOKEN)},
                                           {"draft_tokens", "7,9,9"},
                                           {"verifier_tokens", "4"},
                                           {"accepted_speculative_prefix", "0"},
                                           {"all_speculative_accepted", "false"},
                                           {"verifier_state_matches_output", "true"},
                                       });
            ASSERT_NE(reject_trace, nullptr);

            const PerfStatRecord *catchup =
                findPerfRecordWithTags(records,
                                       PerfStatRecord::Kind::Counter,
                                       "decode_equivalent_catchup_runs",
                                       {{"accepted_tokens", "7,4"},
                                        {"verifier_tokens", "4"},
                                        {"accepted_speculative_prefix", "0"},
                                        {"all_speculative_accepted", "false"}});
            ASSERT_NE(catchup, nullptr);

            const PerfStatRecord *skipped_speculative =
                findPerfRecord(records, PerfStatRecord::Kind::Counter, "post_sidecar_checkpoint_skipped_speculative");
            ASSERT_NE(skipped_speculative, nullptr);
            EXPECT_DOUBLE_EQ(skipped_speculative->value, 1.0);

            GenerationResult step2 = runner->decodeStep();
            ASSERT_TRUE(step2.success()) << step2.error;
            ASSERT_FALSE(step2.tokens.empty());
            EXPECT_NE(step2.tokens.front(), MockInferenceRunner::VERIFY_REJECT_TOKEN)
                << "a returned rejected correction must not be emitted again as the next first token";
        }
        std::filesystem::remove(export_path);
        PerfStatsCollector::reset();
    }

    TEST_F(Test__PrefillDecodeTransition, MTPDecodeRecordsStructuredPerfStats)
    {
        const std::filesystem::path export_path =
            std::filesystem::temp_directory_path() / "llaminar_mtp_decode_perf_stats_unit.json";
        {
            ScopedEnv enable("LLAMINAR_PERF_STATS_JSON", export_path.string().c_str());
            PerfStatsCollector::reset();

            auto [runner, mock] = createRunner(/*mtp_enabled=*/true, /*mtp_accept=*/true);

            std::vector<int32_t> prompt = {1, 2, 3, 4, 5};
            ASSERT_TRUE(runner->prefill(prompt));

            GenerationResult step1 = runner->decodeStep();
            ASSERT_TRUE(step1.success());
            EXPECT_EQ(mock->forwardMTPCount(), 1);
            EXPECT_EQ(mock->commitMTPShiftedCount(), 2);
            EXPECT_EQ(mock->lastCommitMTPAlreadyAppended(), 1);
            EXPECT_THAT(mock->lastCommitMTPTokens(),
                        ElementsAre(MockInferenceRunner::MTP_ARGMAX_TOKEN));

            const auto records = PerfStatsCollector::snapshot({"mtp"});
            const PerfStatRecord *step_calls =
                findPerfRecord(records, PerfStatRecord::Kind::Counter, "decode_step_calls");
            ASSERT_NE(step_calls, nullptr);
            EXPECT_DOUBLE_EQ(step_calls->value, 1.0);

            const PerfStatRecord *capture_timer =
                findPerfRecord(records, PerfStatRecord::Kind::Timer, "capture_live_prefix_state");
            ASSERT_NE(capture_timer, nullptr);
            EXPECT_EQ(capture_timer->count, 1u);

            const PerfStatRecord *post_sidecar_capture =
                findPerfRecord(records, PerfStatRecord::Kind::Timer, "capture_post_sidecar_prefix_state");
            ASSERT_NE(post_sidecar_capture, nullptr);
            EXPECT_EQ(post_sidecar_capture->count, 1u);

            const PerfStatRecord *logical_checkpoint =
                findPerfRecord(records, PerfStatRecord::Kind::Counter, "live_prefix_checkpoint_logical");
            ASSERT_NE(logical_checkpoint, nullptr);
            EXPECT_DOUBLE_EQ(logical_checkpoint->value, 1.0);

            const PerfStatRecord *sidecar_timer =
                findPerfRecord(records, PerfStatRecord::Kind::Timer, "sidecar_forward");
            ASSERT_NE(sidecar_timer, nullptr);
            EXPECT_EQ(sidecar_timer->count, 1u);

            const PerfStatRecord *verifier_timer =
                findPerfRecord(records, PerfStatRecord::Kind::Timer, "verifier_forward");
            ASSERT_NE(verifier_timer, nullptr);
            EXPECT_EQ(verifier_timer->count, 1u);

            const PerfStatRecord *accepted =
                findPerfRecord(records, PerfStatRecord::Kind::Counter, "accepted_tokens");
            ASSERT_NE(accepted, nullptr);
            EXPECT_DOUBLE_EQ(accepted->value, 1.0);

            const PerfStatRecord *accept_trace =
                findPerfRecordWithTags(records, PerfStatRecord::Kind::Counter, "acceptance_trace",
                                       {
                                           {"first_token", std::to_string(MockInferenceRunner::PREFILL_ARGMAX_TOKEN)},
                                           {"draft_tokens", "7,9"},
                                           {"verifier_tokens", "9"},
                                           {"accepted_speculative_prefix", "1"},
                                           {"all_speculative_accepted", "true"},
                                           {"verifier_state_matches_output", "true"},
                                       });
            ASSERT_NE(accept_trace, nullptr);
            EXPECT_DOUBLE_EQ(accept_trace->value, 1.0);
            EXPECT_EQ(accept_trace->tags.at("draft_step"), "1");
            EXPECT_EQ(accept_trace->tags.at("output_tokens"), "2");
            EXPECT_EQ(accept_trace->tags.at("used_ready_logits"), "true");

            const PerfStatRecord *commits =
                findPerfRecord(records, PerfStatRecord::Kind::Counter, "transaction_commits");
            ASSERT_NE(commits, nullptr);
            EXPECT_DOUBLE_EQ(commits->value, 1.0);
        }
        std::filesystem::remove(export_path);
        PerfStatsCollector::reset();
    }

    TEST_F(Test__PrefillDecodeTransition, MTPDecodeRecordsRejectedAcceptanceTrace)
    {
        const std::filesystem::path export_path =
            std::filesystem::temp_directory_path() / "llaminar_mtp_decode_reject_trace_unit.json";
        {
            ScopedEnv enable("LLAMINAR_PERF_STATS_JSON", export_path.string().c_str());
            PerfStatsCollector::reset();

            auto [runner, mock] = createRunner(/*mtp_enabled=*/true, /*mtp_accept=*/false);

            ASSERT_TRUE(runner->prefill({1, 2, 3, 4, 5}));

            GenerationResult step1 = runner->decodeStep();
            ASSERT_TRUE(step1.success());
            EXPECT_EQ(mock->forwardMTPCount(), 1);
            EXPECT_EQ(mock->restoreCount(), 1);
            EXPECT_EQ(mock->captureCheckpointCount(), 2);
            EXPECT_EQ(mock->commitMTPShiftedCount(), 2);
            EXPECT_EQ(mock->lastCommitMTPAlreadyAppended(), 1);
            EXPECT_EQ(mock->lastCommitMTPMainForwardTokenCount(), 0);
            EXPECT_THAT(mock->lastCommitMTPTokens(),
                        ElementsAre(MockInferenceRunner::VERIFY_REJECT_TOKEN));

            const auto records = PerfStatsCollector::snapshot({"mtp"});
            const PerfStatRecord *reject_trace =
                findPerfRecordWithTags(records, PerfStatRecord::Kind::Counter, "acceptance_trace",
                                       {
                                           {"first_token", std::to_string(MockInferenceRunner::PREFILL_ARGMAX_TOKEN)},
                                           {"draft_tokens", "7,9"},
                                           {"verifier_tokens", "4"},
                                           {"rejected_verified_token", std::to_string(MockInferenceRunner::VERIFY_REJECT_TOKEN)},
                                           {"accepted_speculative_prefix", "0"},
                                           {"all_speculative_accepted", "false"},
                                           {"verifier_state_matches_output", "true"},
                                       });
            ASSERT_NE(reject_trace, nullptr);
            EXPECT_DOUBLE_EQ(reject_trace->value, 1.0);
            EXPECT_EQ(reject_trace->tags.at("draft_step"), "1");
            EXPECT_EQ(reject_trace->tags.at("output_tokens"), "2");
            EXPECT_EQ(reject_trace->tags.at("used_ready_logits"), "true");

            const PerfStatRecord *rejected =
                findPerfRecord(records, PerfStatRecord::Kind::Counter, "rejected_tokens");
            ASSERT_NE(rejected, nullptr);
            EXPECT_DOUBLE_EQ(rejected->value, 1.0);

            const PerfStatRecord *rollbacks =
                findPerfRecord(records, PerfStatRecord::Kind::Counter, "rollbacks");
            EXPECT_EQ(rollbacks, nullptr);

            const PerfStatRecord *post_sidecar_capture =
                findPerfRecord(records, PerfStatRecord::Kind::Timer, "capture_post_sidecar_prefix_state");
            ASSERT_NE(post_sidecar_capture, nullptr);
            EXPECT_EQ(post_sidecar_capture->count, 1u);

            const PerfStatRecord *replay_sidecar =
                findPerfRecord(records, PerfStatRecord::Kind::Timer, "replay_sidecar_forward");
            EXPECT_EQ(replay_sidecar, nullptr);
        }
        std::filesystem::remove(export_path);
        PerfStatsCollector::reset();
    }

    TEST_F(Test__PrefillDecodeTransition, AllPositionSpecPublicationAcceptsDepthTwoWithoutSequentialReplay)
    {
        const std::filesystem::path export_path =
            std::filesystem::temp_directory_path() / "llaminar_mtp_all_position_accept_unit.json";
        {
            ScopedEnv enable("LLAMINAR_PERF_STATS_JSON", export_path.string().c_str());
            PerfStatsCollector::reset();

            auto [runner, mock] = createRunner(
                /*mtp_enabled=*/true,
                /*mtp_accept=*/true,
                /*mtp_unsupported_reason=*/{},
                /*mpi_ctx=*/nullptr,
                /*mtp_token_coordination=*/true,
                /*hide_local_logits=*/false,
                DeviceId::cpu(),
                /*mtp_draft_tokens=*/2,
                /*chained_mtp_support=*/true);
            mock->enableMTPSidecarPreservesMainState();
            mock->requireMTPDecodeEquivalentReplay();
            mock->enableMTPSpecStatePublication();
            mock->setVerifierAcceptedPrefixScript({2});

            ASSERT_TRUE(runner->prefill({1, 2, 3, 4, 5}));
            const int forward_count_after_prefill = mock->forwardCallCount();

            GenerationResult step1 = runner->decodeStep();
            ASSERT_TRUE(step1.success()) << step1.error;
            EXPECT_THAT(step1.tokens,
                        ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN,
                                    MockInferenceRunner::MTP_ARGMAX_TOKEN,
                                    MockInferenceRunner::MTP_ARGMAX_TOKEN));

            EXPECT_EQ(mock->setAllPositionCount(), 2);
            EXPECT_EQ(mock->sampleAllPositionLogitsBatchedCount(), 1);
            EXPECT_EQ(mock->lastSampleAllPositionStartRow(), 0);
            EXPECT_EQ(mock->lastSampleAllPositionRowCount(), 3);
            EXPECT_EQ(mock->publishMTPSpecStateCount(), 1);
            EXPECT_EQ(mock->sequentialCommitMTPShiftedCount(), 1)
                << "the first shifted MTP row is committed from terminal hidden before the all-position verifier";
            EXPECT_EQ(mock->commitMTPShiftedCount(), 2)
                << "accepted verifier hidden rows should fill the shifted prefix without sequential verifier replay";
            EXPECT_EQ(mock->lastCommitMTPAlreadyAppended(), 1);
            EXPECT_EQ(mock->lastCommitMTPMainForwardTokenCount(), 3);
            EXPECT_THAT(mock->lastCommitMTPTokens(),
                        ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN,
                                    MockInferenceRunner::MTP_ARGMAX_TOKEN,
                                    MockInferenceRunner::MTP_ARGMAX_TOKEN));
            EXPECT_EQ(mock->restoreCount(), 0);
            EXPECT_EQ(mock->forwardCallCount(), forward_count_after_prefill + 1)
                << "only the all-position verifier graph should run on the main path";

            const MTPSpecStepPlan &published = mock->lastPublishedMTPSpecStep();
            EXPECT_EQ(published.draft_count, 3);
            EXPECT_EQ(published.target_rows, 4);
            EXPECT_EQ(published.accepted_count, 3);
            EXPECT_EQ(published.target_cached_tokens, 8);
            EXPECT_FALSE(published.requiresCorrectionReplay());
            EXPECT_TRUE(published.hasBonusReadyToken());

            const auto records = PerfStatsCollector::snapshot({"mtp"});
            const PerfStatRecord *trace =
                findPerfRecordWithTags(records,
                                       PerfStatRecord::Kind::Counter,
                                       "acceptance_trace",
                                       {{"draft_tokens", "7,9,9"},
                                        {"verifier_tokens", "9,9"},
                                        {"all_position_rows", "9,9,3"},
                                        {"accepted_speculative_prefix", "2"},
                                        {"all_speculative_accepted", "true"},
                                        {"verifier_path", "all_position_state_publication"},
                                        {"decode_equivalent_replay_required", "false"},
                                        {"correction_replay_tokens", "0"},
                                        {"ready_token", "3"}});
            ASSERT_NE(trace, nullptr);

            const PerfStatRecord *publication_runs =
                findPerfRecordWithTags(records,
                                       PerfStatRecord::Kind::Counter,
                                       "all_position_state_publication_verifier_runs",
                                       {{"verifier_rows", "3"},
                                        {"correction_replay_tokens", "0"},
                                        {"accepted_state_count", "3"},
                                        {"target_cached_tokens", "8"}});
            ASSERT_NE(publication_runs, nullptr);
        }
        std::filesystem::remove(export_path);
        PerfStatsCollector::reset();
    }

    TEST_F(Test__PrefillDecodeTransition, AllPositionSpecPublicationReplaysOnlyRejectedCorrection)
    {
        const std::filesystem::path export_path =
            std::filesystem::temp_directory_path() / "llaminar_mtp_all_position_reject_unit.json";
        {
            ScopedEnv enable("LLAMINAR_PERF_STATS_JSON", export_path.string().c_str());
            PerfStatsCollector::reset();

            auto [runner, mock] = createRunner(
                /*mtp_enabled=*/true,
                /*mtp_accept=*/false,
                /*mtp_unsupported_reason=*/{},
                /*mpi_ctx=*/nullptr,
                /*mtp_token_coordination=*/true,
                /*hide_local_logits=*/false,
                DeviceId::cpu(),
                /*mtp_draft_tokens=*/2,
                /*chained_mtp_support=*/true);
            mock->enableMTPSidecarPreservesMainState();
            mock->enableMTPSpecStatePublication();
            mock->setVerifierAcceptedPrefixScript({0});
            mock->setDecodeArgmaxScript({MockInferenceRunner::DECODE_ARGMAX_TOKEN});

            ASSERT_TRUE(runner->prefill({1, 2, 3, 4, 5}));
            const int forward_count_after_prefill = mock->forwardCallCount();

            GenerationResult step1 = runner->decodeStep();
            ASSERT_TRUE(step1.success()) << step1.error;
            EXPECT_THAT(step1.tokens,
                        ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN,
                                    MockInferenceRunner::VERIFY_REJECT_TOKEN));

            EXPECT_EQ(mock->setAllPositionCount(), 2);
            EXPECT_EQ(mock->sampleAllPositionLogitsBatchedCount(), 1);
            EXPECT_EQ(mock->publishMTPSpecStateCount(), 1);
            EXPECT_EQ(mock->sequentialCommitMTPShiftedCount(), 2)
                << "the first shifted row is committed before the verifier, then only the rejected correction suffix is replayed";
            EXPECT_EQ(mock->lastCommitMTPAlreadyAppended(), 1);
            EXPECT_THAT(mock->lastCommitMTPTokens(),
                        ElementsAre(MockInferenceRunner::VERIFY_REJECT_TOKEN));
            EXPECT_EQ(mock->forwardCallCount(), forward_count_after_prefill + 2)
                << "all-position verifier plus one correction replay";

            const MTPSpecStepPlan &published = mock->lastPublishedMTPSpecStep();
            EXPECT_EQ(published.accepted_count, 1);
            EXPECT_EQ(published.target_cached_tokens, 6);
            EXPECT_TRUE(published.requiresCorrectionReplay());
            EXPECT_EQ(published.correction_replay_start_index, 1);
            EXPECT_EQ(published.correction_replay_count, 1);
            EXPECT_FALSE(published.hasBonusReadyToken());

            const auto records = PerfStatsCollector::snapshot({"mtp"});
            const PerfStatRecord *trace =
                findPerfRecordWithTags(records,
                                       PerfStatRecord::Kind::Counter,
                                       "acceptance_trace",
                                       {{"draft_tokens", "7,9,9"},
                                        {"verifier_tokens", "4"},
                                        {"all_position_rows", "4,3,3"},
                                        {"accepted_speculative_prefix", "0"},
                                        {"all_speculative_accepted", "false"},
                                        {"verifier_path", "all_position_state_publication"},
                                        {"decode_equivalent_replay_required", "false"},
                                        {"correction_replay_tokens", "1"},
                                        {"ready_token", "3"}});
            ASSERT_NE(trace, nullptr);

            const PerfStatRecord *publication_runs =
                findPerfRecordWithTags(records,
                                       PerfStatRecord::Kind::Counter,
                                       "all_position_state_publication_verifier_runs",
                                       {{"verifier_rows", "3"},
                                        {"correction_replay_tokens", "1"},
                                        {"accepted_state_count", "1"},
                                        {"target_cached_tokens", "6"}});
            ASSERT_NE(publication_runs, nullptr);
        }
        std::filesystem::remove(export_path);
        PerfStatsCollector::reset();
    }

    TEST_F(Test__PrefillDecodeTransition, AllPositionSpecPublicationCommitsAcceptedPrefixWithBonusVerifierRow)
    {
        const std::filesystem::path export_path =
            std::filesystem::temp_directory_path() / "llaminar_mtp_all_position_prefix_reject_unit.json";
        {
            ScopedEnv enable("LLAMINAR_PERF_STATS_JSON", export_path.string().c_str());
            PerfStatsCollector::reset();

            auto [runner, mock] = createRunner(
                /*mtp_enabled=*/true,
                /*mtp_accept=*/false,
                /*mtp_unsupported_reason=*/{},
                /*mpi_ctx=*/nullptr,
                /*mtp_token_coordination=*/true,
                /*hide_local_logits=*/false,
                DeviceId::cpu(),
                /*mtp_draft_tokens=*/2,
                /*chained_mtp_support=*/true);
            mock->enableMTPSidecarPreservesMainState();
            mock->enableMTPSpecStatePublication();
            mock->setVerifierAcceptedPrefixScript({1});
            mock->setDecodeArgmaxScript({MockInferenceRunner::DECODE_ARGMAX_TOKEN});

            ASSERT_TRUE(runner->prefill({1, 2, 3, 4, 5}));
            const int forward_count_after_prefill = mock->forwardCallCount();

            GenerationResult step1 = runner->decodeStep();
            ASSERT_TRUE(step1.success()) << step1.error;
            EXPECT_THAT(step1.tokens,
                        ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN,
                                    MockInferenceRunner::MTP_ARGMAX_TOKEN,
                                    MockInferenceRunner::DECODE_ARGMAX_TOKEN));

            EXPECT_EQ(mock->setAllPositionCount(), 2);
            EXPECT_EQ(mock->sampleAllPositionLogitsBatchedCount(), 1);
            EXPECT_EQ(mock->lastSampleAllPositionRowCount(), 3)
                << "depth-2 all-position verification includes a bonus-ready row";
            EXPECT_EQ(mock->publishMTPSpecStateCount(), 1);
            EXPECT_EQ(mock->sequentialCommitMTPShiftedCount(), 2)
                << "first shifted row plus rejected correction replay are sequential";
            EXPECT_EQ(mock->commitMTPShiftedCount(), 3)
                << "accepted verifier prefix must be committed from verifier hidden rows";
            EXPECT_EQ(mock->lastCommitMTPAlreadyAppended(), 2)
                << "the final shifted commit is the rejected correction replay";
            EXPECT_EQ(mock->lastCommitMTPMainForwardTokenCount(), 0);
            EXPECT_EQ(mock->forwardCallCount(), forward_count_after_prefill + 2)
                << "all-position verifier plus one correction replay";

            const auto records = PerfStatsCollector::snapshot({"mtp"});
            const PerfStatRecord *publication_runs =
                findPerfRecordWithTags(records,
                                       PerfStatRecord::Kind::Counter,
                                       "all_position_state_publication_verifier_runs",
                                       {{"forward_tokens", "4"},
                                        {"verifier_rows", "3"},
                                        {"correction_replay_tokens", "1"},
                                        {"accepted_state_count", "2"},
                                        {"target_cached_tokens", "7"}});
            ASSERT_NE(publication_runs, nullptr);
        }
        std::filesystem::remove(export_path);
        PerfStatsCollector::reset();
    }

    TEST_F(Test__PrefillDecodeTransition, AllPositionSpecPublicationStochasticAcceptsOnDevice)
    {
        const std::filesystem::path export_path =
            std::filesystem::temp_directory_path() / "llaminar_mtp_all_position_stochastic_accept_unit.json";
        {
            ScopedEnv enable("LLAMINAR_PERF_STATS_JSON", export_path.string().c_str());
            PerfStatsCollector::reset();

            auto [runner, mock] = createRunner(
                /*mtp_enabled=*/true,
                /*mtp_accept=*/true,
                /*mtp_unsupported_reason=*/{},
                /*mpi_ctx=*/nullptr,
                /*mtp_token_coordination=*/true,
                /*hide_local_logits=*/false,
                DeviceId::rocm(0),
                /*mtp_draft_tokens=*/1,
                /*chained_mtp_support=*/false,
                /*sidecar_sample_fusion=*/false,
                {},
                MTPVerifyMode::SpeculativeSampling);
            mock->enableStochasticDeviceSampling();
            mock->enableMTPSidecarPreservesMainState();
            mock->requireMTPDecodeEquivalentReplay();
            mock->enableMTPSpecStatePublication();
            mock->setVerifierAcceptedPrefixScript({1});

            SamplingParams sampling;
            sampling.temperature = 0.8f;
            sampling.top_k = 2;
            sampling.top_p = 0.95f;
            sampling.presence_penalty = 0.25f;
            sampling.seed = 123;
            runner->setSamplingParams(sampling);

            ASSERT_TRUE(runner->prefill({1, 2, 3, 4, 5}));
            const int forward_count_after_prefill = mock->forwardCallCount();

            GenerationResult step1 = runner->decodeStep();
            ASSERT_TRUE(step1.success()) << step1.error;
            EXPECT_THAT(step1.tokens,
                        ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN,
                                    MockInferenceRunner::MTP_ARGMAX_TOKEN));

            EXPECT_EQ(mock->setAllPositionCount(), 2);
            EXPECT_EQ(mock->sampleAllPositionLogitsBatchedCount(), 0);
            EXPECT_EQ(mock->publishMTPSpecStateCount(), 1);
            EXPECT_EQ(mock->sequentialCommitMTPShiftedCount(), 1)
                << "the first shifted MTP row is committed before the stochastic all-position verifier";
            EXPECT_EQ(mock->commitMTPShiftedCount(), 2)
                << "the accepted stochastic verifier row should fill the shifted prefix without replay";
            EXPECT_EQ(mock->lastCommitMTPAlreadyAppended(), 1);
            EXPECT_EQ(mock->lastCommitMTPMainForwardTokenCount(), 2);
            EXPECT_EQ(mock->restoreCount(), 0);
            EXPECT_EQ(mock->forwardCallCount(), forward_count_after_prefill + 1);
            EXPECT_EQ(mock->deviceDistributionBuildCount(), 4)
                << "first token, draft, all-position verifier row, and bonus ready distribution";
            EXPECT_EQ(mock->deviceDistributionSampleCount(), 3);
            EXPECT_EQ(mock->deviceDistributionVerifyBatchCount(), 1);
            EXPECT_EQ(mock->applyMainPenaltiesCount(), 1);
            EXPECT_EQ(mock->applyMTPPenaltiesCount(), 1);
            EXPECT_EQ(mock->applyAllPositionPenaltiesCount(), 2);

            const MTPSpecStepPlan &published = mock->lastPublishedMTPSpecStep();
            EXPECT_EQ(published.accepted_count, 2);
            EXPECT_EQ(published.target_cached_tokens, 7);
            EXPECT_FALSE(published.requiresCorrectionReplay());
            EXPECT_TRUE(published.hasBonusReadyToken());

            const auto probe = runner->prefixStateProbe();
            EXPECT_EQ(probe.mtp_accepted_tokens, 1u);
            EXPECT_EQ(probe.mtp_stochastic_accept_tests, 1u);
            EXPECT_EQ(probe.mtp_stochastic_accepts, 1u);
            EXPECT_EQ(probe.mtp_stochastic_terminal_samples, 1u);
            EXPECT_EQ(probe.mtp_transaction_commits, 1u);

            const auto records = PerfStatsCollector::snapshot({"mtp"});
            const PerfStatRecord *trace =
                findPerfRecordWithTags(records,
                                       PerfStatRecord::Kind::Counter,
                                       "acceptance_trace",
                                       {{"draft_tokens", "7,9"},
                                        {"verifier_tokens", "9"},
                                        {"all_position_rows", "9,3"},
                                        {"accepted_speculative_prefix", "1"},
                                        {"all_speculative_accepted", "true"},
                                        {"verifier_path", "all_position_state_publication"},
                                        {"decode_equivalent_replay_required", "false"},
                                        {"correction_replay_tokens", "0"}});
            ASSERT_NE(trace, nullptr);
        }
        std::filesystem::remove(export_path);
        PerfStatsCollector::reset();
    }

    TEST_F(Test__PrefillDecodeTransition, AllPositionSpecPublicationStochasticReplaysResidualCorrection)
    {
        const std::filesystem::path export_path =
            std::filesystem::temp_directory_path() / "llaminar_mtp_all_position_stochastic_reject_unit.json";
        {
            ScopedEnv enable("LLAMINAR_PERF_STATS_JSON", export_path.string().c_str());
            PerfStatsCollector::reset();

            auto [runner, mock] = createRunner(
                /*mtp_enabled=*/true,
                /*mtp_accept=*/false,
                /*mtp_unsupported_reason=*/{},
                /*mpi_ctx=*/nullptr,
                /*mtp_token_coordination=*/true,
                /*hide_local_logits=*/false,
                DeviceId::rocm(0),
                /*mtp_draft_tokens=*/1,
                /*chained_mtp_support=*/false,
                /*sidecar_sample_fusion=*/false,
                {},
                MTPVerifyMode::SpeculativeSampling);
            mock->enableStochasticDeviceSampling();
            mock->enableMTPSidecarPreservesMainState();
            mock->enableMTPSpecStatePublication();
            mock->setVerifierAcceptedPrefixScript({0});
            mock->setDecodeArgmaxScript({MockInferenceRunner::DECODE_ARGMAX_TOKEN});

            SamplingParams sampling;
            sampling.temperature = 0.8f;
            sampling.top_k = 2;
            sampling.top_p = 0.95f;
            sampling.presence_penalty = 0.25f;
            sampling.seed = 123;
            runner->setSamplingParams(sampling);

            ASSERT_TRUE(runner->prefill({1, 2, 3, 4, 5}));
            const int forward_count_after_prefill = mock->forwardCallCount();

            GenerationResult step1 = runner->decodeStep();
            ASSERT_TRUE(step1.success()) << step1.error;
            EXPECT_THAT(step1.tokens,
                        ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN,
                                    MockInferenceRunner::VERIFY_REJECT_TOKEN));

            EXPECT_EQ(mock->setAllPositionCount(), 2);
            EXPECT_EQ(mock->publishMTPSpecStateCount(), 1);
            EXPECT_EQ(mock->sequentialCommitMTPShiftedCount(), 2);
            EXPECT_EQ(mock->lastCommitMTPAlreadyAppended(), 1);
            EXPECT_EQ(mock->forwardCallCount(), forward_count_after_prefill + 2)
                << "all-position verifier plus one residual correction replay";
            EXPECT_EQ(mock->deviceDistributionBuildCount(), 4)
                << "first token, draft, all-position verifier row, and correction ready distribution";
            EXPECT_EQ(mock->deviceDistributionSampleCount(), 3);
            EXPECT_EQ(mock->deviceDistributionVerifyBatchCount(), 1);
            EXPECT_EQ(mock->applyMainPenaltiesCount(), 2);
            EXPECT_EQ(mock->applyMTPPenaltiesCount(), 1);
            EXPECT_EQ(mock->applyAllPositionPenaltiesCount(), 1);

            const MTPSpecStepPlan &published = mock->lastPublishedMTPSpecStep();
            EXPECT_EQ(published.accepted_count, 1);
            EXPECT_TRUE(published.requiresCorrectionReplay());
            EXPECT_EQ(published.correction_replay_count, 1);
            EXPECT_FALSE(published.hasBonusReadyToken());

            const auto probe = runner->prefixStateProbe();
            EXPECT_EQ(probe.mtp_accepted_tokens, 0u);
            EXPECT_EQ(probe.mtp_rejected_tokens, 1u);
            EXPECT_EQ(probe.mtp_stochastic_accept_tests, 1u);
            EXPECT_EQ(probe.mtp_stochastic_accepts, 0u);
            EXPECT_EQ(probe.mtp_stochastic_residual_samples, 1u);
            EXPECT_EQ(probe.mtp_transaction_commits, 1u);

            const auto records = PerfStatsCollector::snapshot({"mtp"});
            const PerfStatRecord *trace =
                findPerfRecordWithTags(records,
                                       PerfStatRecord::Kind::Counter,
                                       "acceptance_trace",
                                       {{"draft_tokens", "7,9"},
                                        {"verifier_tokens", "4"},
                                        {"all_position_rows", "4,-1"},
                                        {"accepted_speculative_prefix", "0"},
                                        {"all_speculative_accepted", "false"},
                                        {"verifier_path", "all_position_state_publication"},
                                        {"decode_equivalent_replay_required", "false"},
                                        {"correction_replay_tokens", "1"}});
            ASSERT_NE(trace, nullptr);
        }
        std::filesystem::remove(export_path);
        PerfStatsCollector::reset();
    }

    TEST_F(Test__PrefillDecodeTransition, StatefulMTPVerifierUsesSharedDecodeEquivalentCatchup)
    {
        const std::filesystem::path export_path =
            std::filesystem::temp_directory_path() / "llaminar_mtp_shared_catchup_unit.json";
        {
            ScopedEnv enable("LLAMINAR_PERF_STATS_JSON", export_path.string().c_str());
            PerfStatsCollector::reset();

            auto [runner, mock] = createRunner(
                /*mtp_enabled=*/true,
                /*mtp_accept=*/true,
                /*mtp_unsupported_reason=*/{},
                /*mpi_ctx=*/nullptr,
                /*mtp_token_coordination=*/true,
                /*hide_local_logits=*/false,
                DeviceId::cuda(0),
                /*mtp_draft_tokens=*/3,
                /*chained_mtp_support=*/true);
            mock->requireMTPDecodeEquivalentReplay();
            mock->setDecodeArgmaxScript({
                MockInferenceRunner::MTP_ARGMAX_TOKEN,
                MockInferenceRunner::DECODE_ARGMAX_TOKEN,
                MockInferenceRunner::DECODE_ARGMAX_TOKEN});

            ASSERT_TRUE(runner->prefill({1, 2, 3, 4, 5}));
            const int forward_count_after_prefill = mock->forwardCallCount();

            GenerationResult step1 = runner->decodeStep();
            ASSERT_TRUE(step1.success()) << step1.error;
            EXPECT_THAT(step1.tokens,
                        ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN,
                                    MockInferenceRunner::MTP_ARGMAX_TOKEN,
                                    MockInferenceRunner::DECODE_ARGMAX_TOKEN));

            EXPECT_EQ(mock->setAllPositionCount(), 0)
                << "stateful catch-up must not use all-position verifier rows";
            EXPECT_EQ(mock->forwardCallCount(), forward_count_after_prefill + 3);
            EXPECT_EQ(mock->sequentialCommitMTPShiftedCount(), 3);
            EXPECT_EQ(mock->lastCommitMTPAlreadyAppended(), 2);
            EXPECT_TRUE(mock->lastCommitMTPAllowSpeculativeDiscard());
            EXPECT_EQ(mock->lastCommitMTPPositionOffsetOverride(), 5);
            EXPECT_THAT(mock->lastCommitMTPTokens(),
                        ElementsAre(MockInferenceRunner::DECODE_ARGMAX_TOKEN));
            EXPECT_THAT(mock->forwardHistory()[static_cast<size_t>(forward_count_after_prefill)],
                        ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN));
            EXPECT_THAT(mock->forwardHistory()[static_cast<size_t>(forward_count_after_prefill + 1)],
                        ElementsAre(MockInferenceRunner::MTP_ARGMAX_TOKEN));
            EXPECT_THAT(mock->forwardHistory()[static_cast<size_t>(forward_count_after_prefill + 2)],
                        ElementsAre(MockInferenceRunner::DECODE_ARGMAX_TOKEN));

            const auto records = PerfStatsCollector::snapshot({"mtp"});
            const PerfStatRecord *catchup =
                findPerfRecordWithTags(records,
                                       PerfStatRecord::Kind::Counter,
                                       "decode_equivalent_catchup_runs",
                                       {{"implementation", "shared_stepwise"},
                                        {"draft_tokens", "7,9,9,9"},
                                        {"accepted_tokens", "7,9,3"},
                                        {"verifier_tokens", "9,3"},
                                        {"accepted_speculative_prefix", "1"},
                                        {"all_speculative_accepted", "false"}});
            ASSERT_NE(catchup, nullptr);

            const PerfStatRecord *legacy_run_counter =
                findPerfRecordWithTags(records,
                                       PerfStatRecord::Kind::Counter,
                                       "decode_equivalent_sequential_verifier_runs",
                                       {{"forward_tokens", "3"},
                                        {"draft_tokens", "4"},
                                        {"catchup_implementation", "shared_stepwise"}});
            ASSERT_NE(legacy_run_counter, nullptr);

            const PerfStatRecord *accept_trace =
                findPerfRecordWithTags(records,
                                       PerfStatRecord::Kind::Counter,
                                       "acceptance_trace",
                                       {{"first_token", std::to_string(MockInferenceRunner::PREFILL_ARGMAX_TOKEN)},
                                        {"accepted_speculative_prefix", "1"},
                                        {"all_speculative_accepted", "false"},
                                        {"verifier_path", "decode_equivalent_catchup"},
                                        {"catchup_implementation", "shared_stepwise"},
                                        {"decode_equivalent_replay_required", "true"},
                                        {"output_tokens", "3"}});
            ASSERT_NE(accept_trace, nullptr);

            const PerfStatRecord *spec_tx =
                findPerfRecordWithTags(records,
                                       PerfStatRecord::Kind::Counter,
                                       "spec_decode_transaction_metadata",
                                       {{"path", "decode_equivalent_sequential_verifier"},
                                        {"implementation", "shared_stepwise"},
                                        {"target_query_len", "5"},
                                        {"valid_sampled_count", "3"},
                                        {"committed_output_count", "3"},
                                        {"accepted_state_count", "2"},
                                        {"committed_state_row", "1"},
                                        {"committed_state_index", "1"},
                                        {"accepted_state_slot_index", "1"},
                                        {"bonus_ready_token_row", "-1"},
                                        {"bonus_ready_token_index", "-1"},
                                        {"bonus_ready_state_slot_index", "-1"},
                                        {"accepted_verifier_input_prefix", "2"},
                                        {"accepted_mtp_draft_prefix", "1"},
                                        {"rejected_token_count", "2"},
                                        {"token_index_to_sample", "2"},
                                        {"next_condition_token", std::to_string(MockInferenceRunner::DECODE_ARGMAX_TOKEN)},
                                        {"all_drafts_accepted", "false"},
                                        {"stopped_on_output", "false"},
                                        {"draft_tokens", "7,9,9,9"},
                                        {"committed_output_tokens", "7,9,3"}});
            ASSERT_NE(spec_tx, nullptr);
        }
        std::filesystem::remove(export_path);
        PerfStatsCollector::reset();
    }

    TEST_F(Test__PrefillDecodeTransition, ROCmMTPHardFailsWithBroadConcurrentDecodeFlag)
    {
        ScopedEnv broad_concurrent_decode("LLAMINAR_ROCM_CONCURRENT_DECODE", "1");

        auto [runner, mock] = createRunner(
            /*mtp_enabled=*/true,
            /*mtp_accept=*/true,
            /*mtp_unsupported_reason=*/{},
            /*mpi_ctx=*/nullptr,
            /*mtp_token_coordination=*/false,
            /*hide_local_logits=*/false,
            DeviceId::rocm(0));

        ASSERT_TRUE(runner->prefill({1, 2, 3, 4, 5}));

        GenerationResult step1 = runner->decodeStep();
        EXPECT_FALSE(step1.success());
        EXPECT_NE(step1.error.find("LLAMINAR_ROCM_CONCURRENT_DECODE"), std::string::npos);
        EXPECT_NE(step1.error.find("LLAMINAR_ROCM_CONCURRENT_M2_ROWS"), std::string::npos);
        EXPECT_EQ(mock->forwardMTPCount(), 0);
        EXPECT_EQ(mock->forwardCallCount(), 1);
    }

    TEST_F(Test__PrefillDecodeTransition, ROCmMTPAllowsNarrowM2RowOverlapFlag)
    {
        ScopedEnv gpu_graphs("LLAMINAR_GPU_GRAPHS", "0");
        ScopedEnv broad_concurrent_decode("LLAMINAR_ROCM_CONCURRENT_DECODE", "0");
        ScopedEnv m2_rows("LLAMINAR_ROCM_CONCURRENT_M2_ROWS", "1");

        auto [runner, mock] = createRunner(
            /*mtp_enabled=*/true,
            /*mtp_accept=*/true,
            /*mtp_unsupported_reason=*/{},
            /*mpi_ctx=*/nullptr,
            /*mtp_token_coordination=*/false,
            /*hide_local_logits=*/false,
            DeviceId::rocm(0));

        ASSERT_TRUE(runner->prefill({1, 2, 3, 4, 5}));

        GenerationResult step1 = runner->decodeStep();
        ASSERT_TRUE(step1.success()) << step1.error;
        EXPECT_THAT(step1.tokens,
                    ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN,
                                MockInferenceRunner::MTP_ARGMAX_TOKEN));
        EXPECT_EQ(mock->forwardMTPCount(), 1);
    }

    TEST_F(Test__PrefillDecodeTransition, ROCmMTPAllowsGpuGraphsWithoutM2RowOverlap)
    {
        ScopedEnv gpu_graphs("LLAMINAR_GPU_GRAPHS", "1");
        ScopedEnv broad_concurrent_decode("LLAMINAR_ROCM_CONCURRENT_DECODE", "0");
        ScopedEnv m2_rows("LLAMINAR_ROCM_CONCURRENT_M2_ROWS", "0");

        auto [runner, mock] = createRunner(
            /*mtp_enabled=*/true,
            /*mtp_accept=*/true,
            /*mtp_unsupported_reason=*/{},
            /*mpi_ctx=*/nullptr,
            /*mtp_token_coordination=*/false,
            /*hide_local_logits=*/false,
            DeviceId::rocm(0));

        ASSERT_TRUE(runner->prefill({1, 2, 3, 4, 5}));

        GenerationResult step1 = runner->decodeStep();
        ASSERT_TRUE(step1.success()) << step1.error;
        EXPECT_THAT(step1.tokens,
                    ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN,
                                MockInferenceRunner::MTP_ARGMAX_TOKEN));
        EXPECT_EQ(mock->forwardMTPCount(), 1);
    }

    TEST_F(Test__PrefillDecodeTransition, ROCmMTPHardFailsWithM2RowOverlapUnderGpuGraphs)
    {
        ScopedEnv gpu_graphs("LLAMINAR_GPU_GRAPHS", "1");
        ScopedEnv broad_concurrent_decode("LLAMINAR_ROCM_CONCURRENT_DECODE", "0");
        ScopedEnv m2_rows("LLAMINAR_ROCM_CONCURRENT_M2_ROWS", "1");

        auto [runner, mock] = createRunner(
            /*mtp_enabled=*/true,
            /*mtp_accept=*/true,
            /*mtp_unsupported_reason=*/{},
            /*mpi_ctx=*/nullptr,
            /*mtp_token_coordination=*/false,
            /*hide_local_logits=*/false,
            DeviceId::rocm(0));

        ASSERT_TRUE(runner->prefill({1, 2, 3, 4, 5}));

        GenerationResult step1 = runner->decodeStep();
        EXPECT_FALSE(step1.success());
        EXPECT_NE(step1.error.find("LLAMINAR_ROCM_CONCURRENT_M2_ROWS"), std::string::npos)
            << step1.error;
        EXPECT_NE(step1.error.find("LLAMINAR_GPU_GRAPHS=1"), std::string::npos)
            << step1.error;
        EXPECT_EQ(mock->forwardMTPCount(), 0);
        EXPECT_EQ(mock->forwardCallCount(), 1);
    }

    TEST_F(Test__PrefillDecodeTransition, MTPSecondDecodeUsesVerifierTerminalTokenWithoutRefeedingPreviousToken)
    {
        auto [runner, mock] = createRunner(/*mtp_enabled=*/true, /*mtp_accept=*/true);
        mock->setDecodeArgmaxScript({
            MockInferenceRunner::MTP_ARGMAX_TOKEN,
            MockInferenceRunner::DECODE_ARGMAX_TOKEN,
            MockInferenceRunner::MTP_ARGMAX_TOKEN,
            MockInferenceRunner::DECODE_ARGMAX_TOKEN});

        std::vector<int32_t> prompt = {1, 2, 3, 4, 5};
        ASSERT_TRUE(runner->prefill(prompt));

        GenerationResult step1 = runner->decodeStep();
        ASSERT_TRUE(step1.success());
        ASSERT_THAT(step1.tokens,
                    ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN,
                                MockInferenceRunner::MTP_ARGMAX_TOKEN));

        GenerationResult step2 = runner->decodeStep();
        ASSERT_TRUE(step2.success());
        EXPECT_THAT(step2.tokens,
                    ElementsAre(MockInferenceRunner::DECODE_ARGMAX_TOKEN,
                                MockInferenceRunner::MTP_ARGMAX_TOKEN));
        EXPECT_EQ(mock->forwardMTPCount(), 2);
        EXPECT_EQ(mock->lastMTPConditionToken(), MockInferenceRunner::DECODE_ARGMAX_TOKEN);
        EXPECT_EQ(mock->restoreCount(), 2);
        EXPECT_THAT(mock->lastForwardTokens(),
                    ElementsAre(MockInferenceRunner::MTP_ARGMAX_TOKEN));
    }

    TEST_F(Test__PrefillDecodeTransition, MTPReadyVerifierTokenCanBeConsumedByGreedyBypass)
    {
        auto [runner, mock] = createRunner(/*mtp_enabled=*/true, /*mtp_accept=*/true);
        mock->setDecodeArgmaxScript({
            MockInferenceRunner::MTP_ARGMAX_TOKEN,
            MockInferenceRunner::DECODE_ARGMAX_TOKEN});

        ASSERT_TRUE(runner->prefill({1, 2, 3, 4, 5}));

        GenerationResult step1 = runner->decodeStep();
        ASSERT_TRUE(step1.success());
        ASSERT_THAT(step1.tokens,
                    ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN,
                                MockInferenceRunner::MTP_ARGMAX_TOKEN));

        mock->setMTPUnsupportedReason("temporary topology bypass");

        GenerationResult step2 = runner->decodeStep();
        ASSERT_TRUE(step2.success()) << step2.error;
        EXPECT_THAT(step2.tokens, ElementsAre(MockInferenceRunner::DECODE_ARGMAX_TOKEN));
        EXPECT_EQ(mock->forwardCallCount(), 3);
        EXPECT_EQ(mock->forwardMTPCount(), 1);
    }

    TEST_F(Test__PrefillDecodeTransition, MTPReadyVerifierTokenHardFailsIfSamplingChangesBeforeConsume)
    {
        auto [runner, mock] = createRunner(/*mtp_enabled=*/true, /*mtp_accept=*/true);
        mock->setDecodeArgmaxScript({
            MockInferenceRunner::MTP_ARGMAX_TOKEN,
            MockInferenceRunner::DECODE_ARGMAX_TOKEN});

        ASSERT_TRUE(runner->prefill({1, 2, 3, 4, 5}));

        GenerationResult step1 = runner->decodeStep();
        ASSERT_TRUE(step1.success());

        SamplingParams sampling;
        sampling.temperature = 0.8f;
        runner->setSamplingParams(sampling);

        GenerationResult step2 = runner->decodeStep();
        EXPECT_FALSE(step2.success());
        EXPECT_NE(step2.error.find("Ready MTP verifier token"), std::string::npos);
        EXPECT_EQ(mock->restoreCount(), 1);
    }

    TEST_F(Test__PrefillDecodeTransition, MTPFirstDecodeForcedRejectReplaysReturnedCorrection)
    {
        auto [runner, mock] = createRunner(/*mtp_enabled=*/true, /*mtp_accept=*/false);

        std::vector<int32_t> prompt = {1, 2, 3, 4, 5};
        ASSERT_TRUE(runner->prefill(prompt));

        GenerationResult step1 = runner->decodeStep();
        ASSERT_TRUE(step1.success());
        EXPECT_THAT(step1.tokens,
                    ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN,
                                MockInferenceRunner::VERIFY_REJECT_TOKEN));
        EXPECT_EQ(mock->forwardMTPCount(), 1);
        EXPECT_EQ(mock->restoreCount(), 1);
        EXPECT_EQ(mock->captureCheckpointCount(), 2);
        EXPECT_EQ(mock->commitMTPShiftedCount(), 2);
        EXPECT_EQ(mock->lastCommitMTPAlreadyAppended(), 1);
        EXPECT_EQ(mock->lastCommitMTPMainForwardTokenCount(), 0);
        EXPECT_THAT(mock->lastCommitMTPTokens(),
                    ElementsAre(MockInferenceRunner::VERIFY_REJECT_TOKEN));
        EXPECT_THAT(mock->lastForwardTokens(),
                    ElementsAre(MockInferenceRunner::VERIFY_REJECT_TOKEN));

        const auto probe = runner->prefixStateProbe();
        EXPECT_EQ(probe.mtp_draft_steps, 1u);
        EXPECT_EQ(probe.mtp_verifier_runs, 1u);
        EXPECT_EQ(probe.mtp_verifier_token_count, 2u);
        EXPECT_EQ(probe.mtp_accepted_tokens, 0u);
        EXPECT_EQ(probe.mtp_rejected_tokens, 1u);
        EXPECT_EQ(probe.mtp_rollbacks, 0u);
        EXPECT_EQ(probe.mtp_transaction_commits, 1u);
        EXPECT_EQ(probe.mtp_transaction_rollbacks, 0u);
        EXPECT_EQ(probe.mtp_transaction_validation_failures, 0u);

        GenerationResult step2 = runner->decodeStep();
        ASSERT_TRUE(step2.success()) << step2.error;
        ASSERT_FALSE(step2.tokens.empty());
        EXPECT_NE(step2.tokens.front(), MockInferenceRunner::VERIFY_REJECT_TOKEN);
    }

    TEST_F(Test__PrefillDecodeTransition, CUDAMTPForcedRejectUsesSequentialVerifierAndShiftedCommit)
    {
        const std::filesystem::path export_path =
            std::filesystem::temp_directory_path() / "llaminar_cuda_mtp_shared_catchup_unit.json";
        {
            ScopedEnv enable("LLAMINAR_PERF_STATS_JSON", export_path.string().c_str());
            PerfStatsCollector::reset();

            auto [runner, mock] = createRunner(
                /*mtp_enabled=*/true,
                /*mtp_accept=*/false,
                /*mtp_unsupported_reason=*/{},
                /*mpi_ctx=*/nullptr,
                /*mtp_token_coordination=*/false,
                /*hide_local_logits=*/false,
                DeviceId::cuda(0));
            mock->requireMTPDecodeEquivalentReplay();
            mock->setDecodeArgmaxScript({
                MockInferenceRunner::VERIFY_REJECT_TOKEN,
                MockInferenceRunner::DECODE_ARGMAX_TOKEN,
            });

            std::vector<int32_t> prompt = {1, 2, 3, 4, 5};
            ASSERT_TRUE(runner->prefill(prompt));
            const int forward_count_after_prefill = mock->forwardCallCount();

            GenerationResult step1 = runner->decodeStep();
            ASSERT_TRUE(step1.success()) << step1.error;
            EXPECT_THAT(step1.tokens,
                        ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN,
                                    MockInferenceRunner::VERIFY_REJECT_TOKEN));
            EXPECT_EQ(mock->restoreCount(), 1);
            EXPECT_EQ(mock->captureCheckpointCount(), 2)
                << "shared catch-up keeps the post-sidecar checkpoint until a "
                   "backend-optimized multi-row path is explicitly promoted.";
            EXPECT_EQ(mock->commitMTPShiftedCount(), 2);
            EXPECT_EQ(mock->lastCommitMTPAlreadyAppended(), 1);
            EXPECT_EQ(mock->lastCommitMTPMainForwardTokenCount(), 0);
            EXPECT_EQ(mock->sequentialCommitMTPShiftedCount(), 2);
            EXPECT_TRUE(mock->lastCommitMTPAllowSpeculativeDiscard());
            EXPECT_EQ(mock->lastCommitMTPPositionOffsetOverride(), 5);
            EXPECT_EQ(mock->forwardCallCount(), forward_count_after_prefill + 2)
                << "CUDA verifies the first row, commits the correction, and forwards it exactly once";
            EXPECT_THAT(mock->lastForwardTokens(),
                        ElementsAre(MockInferenceRunner::VERIFY_REJECT_TOKEN));
            EXPECT_EQ(mock->setAllPositionCount(), 0);
            EXPECT_EQ(mock->sampleAllPositionLogitsBatchedCount(), 0);

            const auto records = PerfStatsCollector::snapshot({"mtp"});
            const PerfStatRecord *post_sidecar_capture =
                findPerfRecord(records,
                               PerfStatRecord::Kind::Timer,
                               "capture_post_sidecar_prefix_state");
            ASSERT_NE(post_sidecar_capture, nullptr);

            const PerfStatRecord *replay_tokens =
                findPerfRecord(records, PerfStatRecord::Kind::Counter, "replay_tokens");
            EXPECT_EQ(replay_tokens, nullptr);

            const PerfStatRecord *replay_forward =
                findPerfRecord(records, PerfStatRecord::Kind::Timer, "replay_forward_sequential_shifted_commit");
            EXPECT_EQ(replay_forward, nullptr);

            const PerfStatRecord *catchup =
                findPerfRecordWithTags(records,
                                       PerfStatRecord::Kind::Counter,
                                       "decode_equivalent_catchup_runs",
                                       {{"implementation", "shared_stepwise"},
                                        {"draft_tokens", "7,9"},
                                        {"accepted_tokens", "7,4"},
                                        {"verifier_tokens", "4"}});
            ASSERT_NE(catchup, nullptr);
        }
        std::filesystem::remove(export_path);
        PerfStatsCollector::reset();
    }

    TEST_F(Test__PrefillDecodeTransition, MTPForcedRejectRestoresHybridPayloadSnapshot)
    {
        auto [runner, mock] = createRunner(/*mtp_enabled=*/true, /*mtp_accept=*/false);

        PrefixPayloadLayout hybrid_layout;
        hybrid_layout.block_size = 5;
        hybrid_layout.total_layers = 2;
        hybrid_layout.gdn_layers = 1;
        hybrid_layout.hybrid_host_state_bytes = 16;
        hybrid_layout.hybrid_state_bytes = 16;
        hybrid_layout.includes_hybrid_state = true;

        auto hybrid_storage = std::make_shared<std::vector<uint8_t>>(
            std::initializer_list<uint8_t>{1, 3, 5, 7, 9, 11, 13, 15});

        PrefixBlockHandle hybrid_block;
        hybrid_block.key.fingerprint = 0x1234;
        hybrid_block.key.block_index = 0;
        hybrid_block.key.token_start = 0;
        hybrid_block.key.token_count = 5;
        hybrid_block.layout = hybrid_layout;
        hybrid_block.total_bytes = hybrid_layout.totalBytes();
        hybrid_block.hybrid_storage = hybrid_storage;
        hybrid_block.hybrid_payload = hybrid_storage->data();
        hybrid_block.has_hybrid_state = true;

        PrefixStateSnapshot checkpoint;
        checkpoint.valid = true;
        checkpoint.cached_tokens = 5;
        checkpoint.blocks.push_back(hybrid_block);
        mock->setCapturedSnapshot(checkpoint);

        std::vector<int32_t> prompt = {1, 2, 3, 4, 5};
        ASSERT_TRUE(runner->prefill(prompt));

        GenerationResult step1 = runner->decodeStep();
        ASSERT_TRUE(step1.success());
        EXPECT_THAT(step1.tokens,
                    ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN,
                                MockInferenceRunner::VERIFY_REJECT_TOKEN));
        ASSERT_EQ(mock->restoreCount(), 1);
        EXPECT_EQ(mock->forwardMTPCount(), 1);
        EXPECT_EQ(mock->captureCheckpointCount(), 2);

        const PrefixStateSnapshot &restored = mock->lastRestoredSnapshot();
        ASSERT_TRUE(restored.valid);
        ASSERT_EQ(restored.blocks.size(), 1u);
        EXPECT_TRUE(restored.blocks[0].has_hybrid_state);
        EXPECT_TRUE(restored.blocks[0].layout.includes_hybrid_state);
        EXPECT_EQ(restored.blocks[0].layout.hybrid_state_bytes, 16u);
        ASSERT_NE(restored.blocks[0].hybrid_storage, nullptr);
        EXPECT_EQ(*restored.blocks[0].hybrid_storage, *hybrid_storage);
        EXPECT_EQ(restored.cached_tokens, static_cast<int>(prompt.size()));
    }

    TEST_F(Test__PrefillDecodeTransition, MTPBypassForNonGreedySamplingIsRecordedOncePerRequest)
    {
        auto [runner, mock] = createRunner(/*mtp_enabled=*/true, /*mtp_accept=*/true);

        SamplingParams sampling;
        sampling.temperature = 0.8f;
        runner->setSamplingParams(sampling);

        std::vector<int32_t> prompt = {1, 2, 3, 4, 5};
        ASSERT_TRUE(runner->prefill(prompt));

        GenerationResult step1 = runner->decodeStep();
        ASSERT_TRUE(step1.success());
        EXPECT_EQ(mock->forwardMTPCount(), 0);

        auto probe = runner->prefixStateProbe();
        EXPECT_TRUE(probe.mtp_config_enabled);
        EXPECT_TRUE(probe.mtp_bypassed);
        EXPECT_NE(probe.mtp_bypass_reason.find("sampling is not greedy"), std::string::npos);
        EXPECT_EQ(probe.mtp_bypasses, 1u);
        EXPECT_EQ(probe.mtp_draft_steps, 0u);

        GenerationResult step2 = runner->decodeStep();
        ASSERT_TRUE(step2.success());
        EXPECT_EQ(mock->forwardMTPCount(), 0);
        EXPECT_EQ(runner->prefixStateProbe().mtp_bypasses, 1u);
    }

    TEST_F(Test__PrefillDecodeTransition, MTPSpeculativeSamplingUsesHostVerifierForCPU)
    {
        auto [runner, mock] = createRunner(
            /*mtp_enabled=*/true,
            /*mtp_accept=*/true,
            /*mtp_unsupported_reason=*/{},
            /*mpi_ctx=*/nullptr,
            /*mtp_token_coordination=*/false,
            /*hide_local_logits=*/false,
            DeviceId::cpu(),
            /*mtp_draft_tokens=*/1,
            /*chained_mtp_support=*/false,
            /*sidecar_sample_fusion=*/false,
            {},
            MTPVerifyMode::SpeculativeSampling);

        SamplingParams sampling;
        sampling.temperature = 0.8f;
        sampling.top_k = 2;
        sampling.top_p = 0.95f;
        sampling.presence_penalty = 0.25f;
        sampling.seed = 123;
        runner->setSamplingParams(sampling);

        ASSERT_TRUE(runner->prefill({1, 2, 3, 4, 5}));

        GenerationResult step1 = runner->decodeStep();
        ASSERT_TRUE(step1.success()) << step1.error;
        EXPECT_THAT(step1.tokens,
                    ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN,
                                MockInferenceRunner::MTP_ARGMAX_TOKEN));
        EXPECT_EQ(mock->forwardMTPCount(), 1);
        EXPECT_EQ(mock->sampleMainLogitsCount(), 0);
        EXPECT_EQ(mock->sampleMTPLogitsCount(), 0);
        EXPECT_EQ(mock->sampleAllPositionLogitsBatchedCount(), 0);
        EXPECT_EQ(mock->deviceDistributionBuildCount(), 0);
        EXPECT_EQ(mock->deviceDistributionSampleCount(), 0);
        EXPECT_EQ(mock->deviceDistributionVerifyCount(), 0);
        EXPECT_EQ(mock->deviceDistributionVerifyBatchCount(), 0);

        const auto probe = runner->prefixStateProbe();
        EXPECT_FALSE(probe.mtp_bypassed);
        EXPECT_EQ(probe.mtp_bypasses, 0u);
        EXPECT_EQ(probe.mtp_draft_steps, 1u);
        EXPECT_EQ(probe.mtp_verifier_runs, 1u);
        EXPECT_EQ(probe.mtp_accepted_tokens, 1u);
        EXPECT_EQ(probe.mtp_rejected_tokens, 0u);
        EXPECT_EQ(probe.mtp_stochastic_accept_tests, 1u);
        EXPECT_EQ(probe.mtp_stochastic_accepts, 1u);
        EXPECT_EQ(probe.mtp_stochastic_residual_samples, 0u);
        EXPECT_EQ(probe.mtp_stochastic_terminal_samples, 1u);
        EXPECT_EQ(probe.mtp_request.verify_mode, "speculative-sampling");
        EXPECT_TRUE(probe.mtp_request.stochastic_verify);
        EXPECT_EQ(probe.mtp_request.stochastic_accept_tests, 1u);
        EXPECT_EQ(probe.mtp_request.stochastic_accepts, 1u);
        EXPECT_EQ(probe.mtp_request.stochastic_residual_samples, 0u);
        EXPECT_EQ(probe.mtp_request.stochastic_terminal_samples, 1u);
        EXPECT_DOUBLE_EQ(probe.mtp_request.stochastic_acceptance_rate, 1.0);
        EXPECT_EQ(probe.mtp_transaction_commits, 1u);
        EXPECT_EQ(probe.mtp_transaction_validation_failures, 0u);
    }

    TEST_F(Test__PrefillDecodeTransition, MTPSpeculativeSamplingGPURequiresDeviceVerifier)
    {
        auto [runner, mock] = createRunner(
            /*mtp_enabled=*/true,
            /*mtp_accept=*/true,
            /*mtp_unsupported_reason=*/{},
            /*mpi_ctx=*/nullptr,
            /*mtp_token_coordination=*/false,
            /*hide_local_logits=*/false,
            DeviceId::rocm(0),
            /*mtp_draft_tokens=*/1,
            /*chained_mtp_support=*/false,
            /*sidecar_sample_fusion=*/false,
            {},
            MTPVerifyMode::SpeculativeSampling);

        SamplingParams sampling;
        sampling.temperature = 0.8f;
        sampling.top_k = 2;
        sampling.top_p = 0.95f;
        sampling.presence_penalty = 0.25f;
        sampling.seed = 123;
        runner->setSamplingParams(sampling);

        ASSERT_TRUE(runner->prefill({1, 2, 3, 4, 5}));

        GenerationResult step1 = runner->decodeStep();
        ASSERT_FALSE(step1.success());
        EXPECT_THAT(step1.error, HasSubstr("device-resident distribution verification"));
        EXPECT_EQ(mock->sampleDeviceCount(), 0);
        EXPECT_EQ(mock->deviceDistributionBuildCount(), 0);
        EXPECT_EQ(mock->deviceDistributionSampleCount(), 0);
        EXPECT_EQ(mock->deviceDistributionVerifyCount(), 0);
        EXPECT_EQ(mock->deviceDistributionVerifyBatchCount(), 0);
    }

    TEST_F(Test__PrefillDecodeTransition, MTPSpeculativeSamplingUsesDeviceResidentVerifierForGPU)
    {
        auto [runner, mock] = createRunner(
            /*mtp_enabled=*/true,
            /*mtp_accept=*/true,
            /*mtp_unsupported_reason=*/{},
            /*mpi_ctx=*/nullptr,
            /*mtp_token_coordination=*/false,
            /*hide_local_logits=*/false,
            DeviceId::rocm(0),
            /*mtp_draft_tokens=*/1,
            /*chained_mtp_support=*/false,
            /*sidecar_sample_fusion=*/false,
            {},
            MTPVerifyMode::SpeculativeSampling);
        mock->enableStochasticDeviceSampling();

        SamplingParams sampling;
        sampling.temperature = 0.8f;
        sampling.top_k = 2;
        sampling.top_p = 0.95f;
        sampling.presence_penalty = 0.25f;
        sampling.seed = 123;
        runner->setSamplingParams(sampling);

        ASSERT_TRUE(runner->prefill({1, 2, 3, 4, 5}));

        GenerationResult step1 = runner->decodeStep();
        ASSERT_TRUE(step1.success()) << step1.error;
        EXPECT_THAT(step1.tokens,
                    ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN,
                                MockInferenceRunner::MTP_ARGMAX_TOKEN));
        EXPECT_EQ(mock->sampleDeviceCount(), 0);
        EXPECT_EQ(mock->sampleMainLogitsCount(), 0);
        EXPECT_EQ(mock->sampleMTPLogitsCount(), 0);
        EXPECT_EQ(mock->sampleAllPositionLogitsBatchedCount(), 0);
        EXPECT_EQ(mock->deviceDistributionBuildCount(), 4)
            << "first token, MTP draft, verifier row, and terminal ready-token distributions should stay compact/device-resident";
        EXPECT_EQ(mock->deviceDistributionSampleCount(), 3)
            << "first token, MTP draft, and terminal ready-token sampling should avoid host full-logit sampling";
        EXPECT_EQ(mock->deviceDistributionVerifyCount(), 0);
        EXPECT_EQ(mock->deviceDistributionVerifyBatchCount(), 1);
        EXPECT_EQ(mock->applyMainPenaltiesCount(), 3);
        EXPECT_EQ(mock->applyMTPPenaltiesCount(), 1);
        EXPECT_EQ(mock->applyAllPositionPenaltiesCount(), 0);
        EXPECT_EQ(mock->forwardMTPCount(), 1);

        const auto probe = runner->prefixStateProbe();
        EXPECT_FALSE(probe.mtp_bypassed);
        EXPECT_EQ(probe.mtp_draft_steps, 1u);
        EXPECT_EQ(probe.mtp_accepted_tokens, 1u);
        EXPECT_EQ(probe.mtp_stochastic_accept_tests, 1u);
        EXPECT_EQ(probe.mtp_stochastic_accepts, 1u);
        EXPECT_EQ(probe.mtp_stochastic_residual_samples, 0u);
        EXPECT_EQ(probe.mtp_stochastic_terminal_samples, 1u);
        EXPECT_EQ(probe.mtp_request.verify_mode, "speculative-sampling");
        EXPECT_TRUE(probe.mtp_request.stochastic_verify);
        EXPECT_EQ(probe.mtp_transaction_commits, 1u);
        EXPECT_EQ(probe.mtp_transaction_validation_failures, 0u);
    }

    TEST_F(Test__PrefillDecodeTransition, StatefulMTPSpeculativeSamplingUsesDecodeEquivalentDeviceVerifier)
    {
        auto [runner, mock] = createRunner(
            /*mtp_enabled=*/true,
            /*mtp_accept=*/true,
            /*mtp_unsupported_reason=*/{},
            /*mpi_ctx=*/nullptr,
            /*mtp_token_coordination=*/false,
            /*hide_local_logits=*/false,
            DeviceId::rocm(0),
            /*mtp_draft_tokens=*/1,
            /*chained_mtp_support=*/false,
            /*sidecar_sample_fusion=*/false,
            {},
            MTPVerifyMode::SpeculativeSampling);
        mock->enableStochasticDeviceSampling();
        mock->requireMTPDecodeEquivalentReplay();
        mock->setDecodeArgmaxScript({
            MockInferenceRunner::MTP_ARGMAX_TOKEN,
            MockInferenceRunner::DECODE_ARGMAX_TOKEN,
        });

        SamplingParams sampling;
        sampling.temperature = 0.8f;
        sampling.top_k = 2;
        sampling.top_p = 0.95f;
        sampling.presence_penalty = 0.25f;
        sampling.seed = 123;
        runner->setSamplingParams(sampling);

        ASSERT_TRUE(runner->prefill({1, 2, 3, 4, 5}));

        GenerationResult step1 = runner->decodeStep();
        ASSERT_TRUE(step1.success()) << step1.error;
        EXPECT_THAT(step1.tokens,
                    ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN,
                                MockInferenceRunner::MTP_ARGMAX_TOKEN));
        EXPECT_EQ(mock->restoreCount(), 1)
            << "stateful stochastic verification must restore the verifier base";
        EXPECT_EQ(mock->setAllPositionCount(), 0)
            << "stateful stochastic verification must not use all-position verifier rows";
        EXPECT_EQ(mock->applyAllPositionPenaltiesCount(), 0);
        EXPECT_EQ(mock->deviceDistributionBuildCount(), 4)
            << "first token, MTP draft, sequential target row, and ready token distributions stay compact/device-resident";
        EXPECT_EQ(mock->deviceDistributionSampleCount(), 3);
        EXPECT_EQ(mock->deviceDistributionVerifyBatchCount(), 1);
        EXPECT_EQ(mock->sequentialCommitMTPShiftedCount(), 2)
            << "first token and accepted draft must publish shifted MTP rows from sequential terminal hidden";

        const auto probe = runner->prefixStateProbe();
        EXPECT_FALSE(probe.mtp_bypassed);
        EXPECT_EQ(probe.mtp_draft_steps, 1u);
        EXPECT_EQ(probe.mtp_accepted_tokens, 1u);
        EXPECT_EQ(probe.mtp_stochastic_accept_tests, 1u);
        EXPECT_EQ(probe.mtp_stochastic_accepts, 1u);
        EXPECT_EQ(probe.mtp_stochastic_residual_samples, 0u);
        EXPECT_EQ(probe.mtp_stochastic_terminal_samples, 1u);
        EXPECT_EQ(probe.mtp_transaction_commits, 1u);
        EXPECT_EQ(probe.mtp_transaction_validation_failures, 0u);
    }

    TEST_F(Test__PrefillDecodeTransition, StatefulMTPSpeculativeSamplingSkipsBaseRestoreWhenSidecarPreservesMainState)
    {
        const std::filesystem::path export_path =
            std::filesystem::temp_directory_path() / "llaminar_mtp_stochastic_sidecar_preserved_unit.json";
        {
            ScopedEnv enable("LLAMINAR_PERF_STATS_JSON", export_path.string().c_str());
            PerfStatsCollector::reset();

            auto [runner, mock] = createRunner(
                /*mtp_enabled=*/true,
                /*mtp_accept=*/true,
                /*mtp_unsupported_reason=*/{},
                /*mpi_ctx=*/nullptr,
                /*mtp_token_coordination=*/false,
                /*hide_local_logits=*/false,
                DeviceId::rocm(0),
                /*mtp_draft_tokens=*/1,
                /*chained_mtp_support=*/false,
                /*sidecar_sample_fusion=*/false,
                {},
                MTPVerifyMode::SpeculativeSampling);
            mock->enableStochasticDeviceSampling();
            mock->enableMTPSidecarPreservesMainState();
            mock->requireMTPDecodeEquivalentReplay();
            mock->setDecodeArgmaxScript({
                MockInferenceRunner::MTP_ARGMAX_TOKEN,
                MockInferenceRunner::DECODE_ARGMAX_TOKEN,
            });

            SamplingParams sampling;
            sampling.temperature = 0.8f;
            sampling.top_k = 2;
            sampling.top_p = 0.95f;
            sampling.seed = 123;
            runner->setSamplingParams(sampling);

            ASSERT_TRUE(runner->prefill({1, 2, 3, 4, 5}));

            GenerationResult step1 = runner->decodeStep();
            ASSERT_TRUE(step1.success()) << step1.error;
            EXPECT_THAT(step1.tokens,
                        ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN,
                                    MockInferenceRunner::MTP_ARGMAX_TOKEN));
            EXPECT_EQ(mock->restoreCount(), 0)
                << "graph-native sidecar execution preserves main verifier state, "
                   "so stochastic decode-equivalent verification should not restore "
                   "the same base checkpoint after sidecar draft";
            EXPECT_EQ(mock->setAllPositionCount(), 0);
            EXPECT_EQ(mock->deviceDistributionVerifyBatchCount(), 1);
            EXPECT_EQ(mock->sequentialCommitMTPShiftedCount(), 2);

            const auto records = PerfStatsCollector::snapshot({"mtp"});
            const PerfStatRecord *restore_counter =
                findPerfRecord(records,
                               PerfStatRecord::Kind::Counter,
                               "decode_equivalent_sequential_verifier_base_restores");
            EXPECT_EQ(restore_counter, nullptr);
            const PerfStatRecord *restore_timer =
                findPerfRecord(records,
                               PerfStatRecord::Kind::Timer,
                               "decode_equivalent_sequential_verifier_restore_base_checkpoint");
            EXPECT_EQ(restore_timer, nullptr);
            const PerfStatRecord *skipped_restore =
                findPerfRecord(records,
                               PerfStatRecord::Kind::Counter,
                               "decode_equivalent_sequential_verifier_base_restore_skipped_sidecar_preserved");
            ASSERT_NE(skipped_restore, nullptr);
            EXPECT_DOUBLE_EQ(skipped_restore->value, 1.0);

            const PerfStatRecord *verifier_runs =
                findPerfRecordWithTags(records,
                                       PerfStatRecord::Kind::Counter,
                                       "decode_equivalent_stochastic_verifier_runs",
                                       {{"restored_verifier_base", "true"}});
            ASSERT_NE(verifier_runs, nullptr);
        }
        std::filesystem::remove(export_path);
        PerfStatsCollector::reset();
    }

    TEST_F(Test__PrefillDecodeTransition, MTPSpeculativeSamplingHostVerifierSamplesResidualForCPU)
    {
        auto [runner, mock] = createRunner(
            /*mtp_enabled=*/true,
            /*mtp_accept=*/false,
            /*mtp_unsupported_reason=*/{},
            /*mpi_ctx=*/nullptr,
            /*mtp_token_coordination=*/false,
            /*hide_local_logits=*/false,
            DeviceId::cpu(),
            /*mtp_draft_tokens=*/1,
            /*chained_mtp_support=*/false,
            /*sidecar_sample_fusion=*/false,
            {},
            MTPVerifyMode::SpeculativeSampling);

        SamplingParams sampling;
        sampling.temperature = 0.8f;
        sampling.top_k = 2;
        sampling.top_p = 0.95f;
        sampling.presence_penalty = 0.25f;
        sampling.seed = 456;
        runner->setSamplingParams(sampling);

        ASSERT_TRUE(runner->prefill({1, 2, 3, 4, 5}));

        GenerationResult step1 = runner->decodeStep();
        ASSERT_TRUE(step1.success()) << step1.error;
        EXPECT_THAT(step1.tokens,
                    ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN,
                                MockInferenceRunner::VERIFY_REJECT_TOKEN));
        EXPECT_EQ(mock->forwardMTPCount(), 1);
        EXPECT_EQ(mock->commitMTPShiftedCount(), 2)
            << "CPU stochastic verification still commits shifted rows for the first token and residual correction";
        EXPECT_EQ(mock->sequentialCommitMTPShiftedCount(), 2);
        EXPECT_EQ(mock->deviceDistributionBuildCount(), 0);
        EXPECT_EQ(mock->deviceDistributionSampleCount(), 0);
        EXPECT_EQ(mock->deviceDistributionVerifyCount(), 0);
        EXPECT_EQ(mock->deviceDistributionVerifyBatchCount(), 0);

        const auto probe = runner->prefixStateProbe();
        EXPECT_FALSE(probe.mtp_bypassed);
        EXPECT_EQ(probe.mtp_draft_steps, 1u);
        EXPECT_EQ(probe.mtp_accepted_tokens, 0u);
        EXPECT_EQ(probe.mtp_rejected_tokens, 1u);
        EXPECT_EQ(probe.mtp_stochastic_accept_tests, 1u);
        EXPECT_EQ(probe.mtp_stochastic_accepts, 0u);
        EXPECT_EQ(probe.mtp_stochastic_residual_samples, 1u);
        EXPECT_EQ(probe.mtp_stochastic_terminal_samples, 0u);
        EXPECT_EQ(probe.mtp_request.verify_mode, "speculative-sampling");
        EXPECT_TRUE(probe.mtp_request.stochastic_verify);
        EXPECT_EQ(probe.mtp_request.stochastic_accept_tests, 1u);
        EXPECT_EQ(probe.mtp_request.stochastic_accepts, 0u);
        EXPECT_EQ(probe.mtp_request.stochastic_residual_samples, 1u);
        EXPECT_EQ(probe.mtp_request.stochastic_terminal_samples, 0u);
        EXPECT_DOUBLE_EQ(probe.mtp_request.stochastic_acceptance_rate, 0.0);
        EXPECT_EQ(probe.mtp_transaction_commits, 1u);
        EXPECT_EQ(probe.mtp_transaction_rollbacks, 1u);
        EXPECT_EQ(probe.mtp_transaction_validation_failures, 0u);
    }

    TEST_F(Test__PrefillDecodeTransition, MTPSpeculativeSamplingDeviceVerifierBatchesAcceptThenSamplesResidual)
    {
        auto [runner, mock] = createRunner(
            /*mtp_enabled=*/true,
            /*mtp_accept=*/false,
            /*mtp_unsupported_reason=*/{},
            /*mpi_ctx=*/nullptr,
            /*mtp_token_coordination=*/false,
            /*hide_local_logits=*/false,
            DeviceId::rocm(0),
            /*mtp_draft_tokens=*/1,
            /*chained_mtp_support=*/false,
            /*sidecar_sample_fusion=*/false,
            {},
            MTPVerifyMode::SpeculativeSampling);
        mock->enableStochasticDeviceSampling();

        SamplingParams sampling;
        sampling.temperature = 0.8f;
        sampling.top_k = 2;
        sampling.top_p = 0.95f;
        sampling.presence_penalty = 0.25f;
        sampling.seed = 456;
        runner->setSamplingParams(sampling);

        ASSERT_TRUE(runner->prefill({1, 2, 3, 4, 5}));

        GenerationResult step1 = runner->decodeStep();
        ASSERT_TRUE(step1.success()) << step1.error;
        EXPECT_THAT(step1.tokens,
                    ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN,
                                MockInferenceRunner::VERIFY_REJECT_TOKEN));
        EXPECT_EQ(mock->deviceDistributionBuildCount(), 4)
            << "first token, MTP draft, verifier row, and terminal row distributions should stay compact/device-resident";
        EXPECT_EQ(mock->deviceDistributionSampleCount(), 3)
            << "first token, MTP draft, and post-correction ready-token sampling should avoid full-logit host sampling";
        EXPECT_EQ(mock->deviceDistributionVerifyBatchCount(), 1);
        EXPECT_EQ(mock->deviceDistributionVerifyCount(), 0)
            << "the first rejected row should use the batched residual-capable verifier";
        EXPECT_EQ(mock->applyAllPositionPenaltiesCount(), 0);

        const auto probe = runner->prefixStateProbe();
        EXPECT_EQ(probe.mtp_stochastic_accept_tests, 1u);
        EXPECT_EQ(probe.mtp_stochastic_accepts, 0u);
        EXPECT_EQ(probe.mtp_stochastic_residual_samples, 1u);
        EXPECT_EQ(probe.mtp_transaction_rollbacks, 1u);
    }

    TEST_F(Test__PrefillDecodeTransition, MTPPPTopologyFailsBeforePrefillForward)
    {
        auto [runner, mock] = createRunner(
            /*mtp_enabled=*/true,
            /*mtp_accept=*/true,
            "MTP decode is not enabled for PP topologies");

        std::vector<int32_t> prompt = {1, 2, 3, 4, 5};
        EXPECT_FALSE(runner->prefill(prompt));
        EXPECT_NE(runner->lastError().find("MTP is not enabled for PP topologies"),
                  std::string::npos);
        EXPECT_EQ(mock->forwardCallCount(), 0)
            << "Unsupported PP MTP must fail before shifted-prefill sidecar state is populated";
        EXPECT_EQ(mock->forwardMTPCount(), 0);

        auto probe = runner->prefixStateProbe();
        EXPECT_TRUE(probe.mtp_config_enabled);
        EXPECT_FALSE(probe.mtp_bypassed);
        EXPECT_EQ(probe.mtp_bypasses, 0u);
    }

    TEST_F(Test__PrefillDecodeTransition, MTPBypassForRunnerTopologyReasonPreservesGreedyDecode)
    {
        auto [runner, mock] = createRunner(
            /*mtp_enabled=*/true,
            /*mtp_accept=*/true,
            "MTP decode requires TP logits and checkpoint coordination");

        std::vector<int32_t> prompt = {1, 2, 3, 4, 5};
        ASSERT_TRUE(runner->prefill(prompt));

        GenerationResult step1 = runner->decodeStep();
        ASSERT_TRUE(step1.success());
        ASSERT_EQ(step1.tokens.size(), 1u);
        EXPECT_EQ(step1.tokens[0], MockInferenceRunner::PREFILL_ARGMAX_TOKEN);
        EXPECT_EQ(mock->forwardMTPCount(), 0);

        auto probe = runner->prefixStateProbe();
        EXPECT_TRUE(probe.mtp_config_enabled);
        EXPECT_TRUE(probe.mtp_bypassed);
        EXPECT_NE(probe.mtp_bypass_reason.find("TP logits"), std::string::npos);
        EXPECT_EQ(probe.mtp_bypasses, 1u);
        EXPECT_EQ(probe.mtp_draft_steps, 0u);
    }

    TEST_F(Test__PrefillDecodeTransition, MTPBypassForWorldSizeWithoutTokenCoordination)
    {
        auto mpi = std::make_shared<llaminar2::test::MockMPIContext>(0, 2);
        auto [runner, mock] = createRunner(
            /*mtp_enabled=*/true,
            /*mtp_accept=*/true,
            /*mtp_unsupported_reason=*/std::string{},
            mpi);

        std::vector<int32_t> prompt = {1, 2, 3, 4, 5};
        ASSERT_TRUE(runner->prefill(prompt));

        GenerationResult step1 = runner->decodeStep();
        ASSERT_TRUE(step1.success());
        ASSERT_EQ(step1.tokens.size(), 1u);
        EXPECT_EQ(step1.tokens[0], MockInferenceRunner::PREFILL_ARGMAX_TOKEN);
        EXPECT_EQ(mock->forwardMTPCount(), 0);

        auto probe = runner->prefixStateProbe();
        EXPECT_TRUE(probe.mtp_config_enabled);
        EXPECT_TRUE(probe.mtp_bypassed);
        EXPECT_NE(probe.mtp_bypass_reason.find("world_size > 1"), std::string::npos);
        EXPECT_EQ(probe.mtp_bypasses, 1u);
        EXPECT_EQ(probe.mtp_draft_steps, 0u);
    }

    TEST_F(Test__PrefillDecodeTransition, MTPWorldSizeUsesCoordinatedSamplingWithoutLocalLogits)
    {
        auto mpi = std::make_shared<llaminar2::test::MockMPIContext>(0, 2);
        auto [runner, mock] = createRunner(
            /*mtp_enabled=*/true,
            /*mtp_accept=*/true,
            /*mtp_unsupported_reason=*/std::string{},
            mpi,
            /*mtp_token_coordination=*/true,
            /*hide_local_logits=*/true);

        std::vector<int32_t> prompt = {1, 2, 3, 4, 5};
        ASSERT_TRUE(runner->prefill(prompt));

        GenerationResult step1 = runner->decodeStep();
        ASSERT_TRUE(step1.success()) << step1.error;
        EXPECT_THAT(step1.tokens,
                    ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN,
                                MockInferenceRunner::MTP_ARGMAX_TOKEN));
        EXPECT_EQ(mock->forwardMTPCount(), 1);
        EXPECT_EQ(mock->lastMTPConditionToken(), MockInferenceRunner::PREFILL_ARGMAX_TOKEN);
        EXPECT_GE(mock->sampleMainLogitsCount(), 1);
        EXPECT_EQ(mock->sampleMTPLogitsCount(), 1);
        EXPECT_EQ(mock->sampleAllPositionLogitsCount(), 0);
        EXPECT_EQ(mock->sampleAllPositionLogitsBatchedCount(), 0);
        EXPECT_EQ(mock->lastSampleAllPositionStartRow(), -1);
        EXPECT_EQ(mock->lastSampleAllPositionRowCount(), 0);

        auto probe = runner->prefixStateProbe();
        EXPECT_FALSE(probe.mtp_bypassed);
        EXPECT_EQ(probe.mtp_draft_steps, 1u);
        EXPECT_EQ(probe.mtp_accepted_tokens, 1u);
        EXPECT_EQ(probe.mtp_rollbacks, 0u);
    }

    TEST_F(Test__PrefillDecodeTransition, GlobalTPMTPDecodeRunsThroughGlobalOrchestratorCoordination)
    {
        auto mpi = std::make_shared<llaminar2::test::MockMPIContext>(0, 2);
        auto child = std::make_unique<MockInferenceRunner>();
        auto *child_ptr = child.get();
        child_ptr->enableMTP(/*accept_mtp_token=*/true);

        GlobalOrchestrator::Config global_config;
        global_config.topology = buildSingleStageGlobalTPTopo(2);
        global_config.rank = 0;
        global_config.world_size = 2;
        global_config.mpi_ctx = mpi.get();
        global_config.rank_runner = std::move(child);
        global_config.vocab_size = MockInferenceRunner::VOCAB_SIZE;
        global_config.d_model = 16;
        global_config.architecture_name = "mock";

        auto global_runner = std::make_unique<GlobalOrchestrator>(std::move(global_config));

        OrchestrationConfig config;
        config.device_for_this_rank = GlobalDeviceAddress::cpu();
        config.mtp.enabled = true;
        config.mtp.draft_tokens = 1;
        config.mtp.verify_mode = MTPVerifyMode::Greedy;

        auto runner = std::make_unique<OrchestrationRunner>(
            std::move(config), plan_, std::move(global_runner), mpi);
        SamplingParams greedy;
        greedy.temperature = 0.0f;
        runner->setSamplingParams(greedy);

        std::vector<int32_t> prompt = {1, 2, 3, 4, 5};
        ASSERT_TRUE(runner->prefill(prompt));

        GenerationResult step1 = runner->decodeStep();
        ASSERT_TRUE(step1.success()) << step1.error;
        EXPECT_THAT(step1.tokens,
                    ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN,
                                MockInferenceRunner::MTP_ARGMAX_TOKEN));
        EXPECT_EQ(child_ptr->forwardMTPCount(), 1);
        EXPECT_EQ(child_ptr->lastMTPConditionToken(), MockInferenceRunner::PREFILL_ARGMAX_TOKEN);
        EXPECT_EQ(child_ptr->commitMTPShiftedCount(), 2);
        EXPECT_EQ(child_ptr->lastCommitMTPAlreadyAppended(), 1);
        EXPECT_THAT(child_ptr->lastCommitMTPTokens(),
                    ElementsAre(MockInferenceRunner::MTP_ARGMAX_TOKEN));
        EXPECT_EQ(child_ptr->restoreCount(), 1)
            << "GlobalTP restores the verifier base and commits through shared decode-equivalent replay";

        auto probe = runner->prefixStateProbe();
        EXPECT_FALSE(probe.mtp_bypassed);
        EXPECT_EQ(probe.mtp_draft_steps, 1u);
        EXPECT_EQ(probe.mtp_accepted_tokens, 1u);
        EXPECT_EQ(probe.mtp_rollbacks, 0u);
    }

    TEST_F(Test__PrefillDecodeTransition, LocalTPMTPDecodeRunsEveryParticipantAndCommitsVerifierState)
    {
        auto harness = createLocalTPRunner(/*mtp_accept=*/true);

        std::vector<int32_t> prompt = {1, 2, 3, 4, 5};
        ASSERT_TRUE(harness.runner->prefill(prompt));

        GenerationResult step1 = harness.runner->decodeStep();
        ASSERT_TRUE(step1.success());
        EXPECT_THAT(step1.tokens,
                    ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN,
                                MockInferenceRunner::MTP_ARGMAX_TOKEN));

        EXPECT_EQ(harness.child0->forwardMTPCount(), 1);
        EXPECT_EQ(harness.child1->forwardMTPCount(), 1);
        EXPECT_EQ(harness.child0->lastMTPConditionToken(), MockInferenceRunner::PREFILL_ARGMAX_TOKEN);
        EXPECT_EQ(harness.child1->lastMTPConditionToken(), MockInferenceRunner::PREFILL_ARGMAX_TOKEN);
        EXPECT_EQ(harness.child0->restoreCount(), 1);
        EXPECT_EQ(harness.child1->restoreCount(), 1);
        EXPECT_EQ(harness.child0->commitMTPShiftedCount(), 2);
        EXPECT_EQ(harness.child1->commitMTPShiftedCount(), 2);
        EXPECT_EQ(harness.child0->lastCommitMTPAlreadyAppended(), 1);
        EXPECT_EQ(harness.child1->lastCommitMTPAlreadyAppended(), 1);
        EXPECT_THAT(harness.child0->lastCommitMTPTokens(),
                    ElementsAre(MockInferenceRunner::MTP_ARGMAX_TOKEN));
        EXPECT_THAT(harness.child1->lastCommitMTPTokens(),
                    ElementsAre(MockInferenceRunner::MTP_ARGMAX_TOKEN));
        EXPECT_EQ(harness.child0->setAllPositionCount(), 0);
        EXPECT_EQ(harness.child1->setAllPositionCount(), 0);
        EXPECT_THAT(harness.child0->lastForwardTokens(),
                    ElementsAre(MockInferenceRunner::MTP_ARGMAX_TOKEN));
        EXPECT_THAT(harness.child1->lastForwardTokens(),
                    ElementsAre(MockInferenceRunner::MTP_ARGMAX_TOKEN));

        const auto probe = harness.runner->prefixStateProbe();
        EXPECT_EQ(probe.mtp_draft_steps, 1u);
        EXPECT_EQ(probe.mtp_verifier_runs, 1u);
        EXPECT_EQ(probe.mtp_verifier_token_count, 2u);
        EXPECT_EQ(probe.mtp_accepted_tokens, 1u);
        EXPECT_EQ(probe.mtp_rejected_tokens, 0u);
        EXPECT_EQ(probe.mtp_rollbacks, 0u);
    }

    TEST_F(Test__PrefillDecodeTransition, ROCmLocalTPMTPSegmentedCollectivesFailBeforeSidecarLaunch)
    {
        ScopedEnv gpu_graphs("LLAMINAR_GPU_GRAPHS", "1");
        ScopedEnv segmented_collectives("LLAMINAR_GPU_GRAPH_COLLECTIVE_SEGMENTED", "1");

        auto harness = createLocalTPRunner(
            /*mtp_accept=*/true,
            /*column_parallel_logits=*/false,
            {GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)});

        std::vector<int32_t> prompt = {1, 2, 3, 4, 5};
        ASSERT_TRUE(harness.runner->prefill(prompt));

        GenerationResult step = harness.runner->decodeStep();
        EXPECT_FALSE(step.success());
        EXPECT_NE(step.error.find("ROCm LocalTP MTP decode is incompatible"), std::string::npos)
            << step.error;
        EXPECT_NE(step.error.find("LLAMINAR_GPU_GRAPH_COLLECTIVE_SEGMENTED"), std::string::npos)
            << step.error;
        EXPECT_EQ(harness.child0->forwardMTPCount(), 0);
        EXPECT_EQ(harness.child1->forwardMTPCount(), 0);
    }

    TEST_F(Test__PrefillDecodeTransition, LocalTPMTPForcedRejectCountsOnceAcrossParticipants)
    {
        auto harness = createLocalTPRunner(/*mtp_accept=*/false);

        std::vector<int32_t> prompt = {1, 2, 3, 4, 5};
        ASSERT_TRUE(harness.runner->prefill(prompt));

        GenerationResult step1 = harness.runner->decodeStep();
        ASSERT_TRUE(step1.success());
        EXPECT_THAT(step1.tokens,
                    ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN,
                                MockInferenceRunner::VERIFY_REJECT_TOKEN));

        EXPECT_EQ(harness.child0->forwardMTPCount(), 1);
        EXPECT_EQ(harness.child1->forwardMTPCount(), 1);
        EXPECT_EQ(harness.child0->restoreCount(), 1);
        EXPECT_EQ(harness.child1->restoreCount(), 1);
        EXPECT_EQ(harness.child0->captureCheckpointCount(), 2);
        EXPECT_EQ(harness.child1->captureCheckpointCount(), 2);
        EXPECT_EQ(harness.child0->commitMTPShiftedCount(), 2);
        EXPECT_EQ(harness.child1->commitMTPShiftedCount(), 2);
        EXPECT_EQ(harness.child0->lastCommitMTPAlreadyAppended(), 1);
        EXPECT_EQ(harness.child1->lastCommitMTPAlreadyAppended(), 1);
        EXPECT_EQ(harness.child0->lastCommitMTPMainForwardTokenCount(), 0);
        EXPECT_EQ(harness.child1->lastCommitMTPMainForwardTokenCount(), 0);
        EXPECT_THAT(harness.child0->lastCommitMTPTokens(),
                    ElementsAre(MockInferenceRunner::VERIFY_REJECT_TOKEN));
        EXPECT_THAT(harness.child1->lastCommitMTPTokens(),
                    ElementsAre(MockInferenceRunner::VERIFY_REJECT_TOKEN));
        EXPECT_THAT(harness.child0->lastForwardTokens(),
                    ElementsAre(MockInferenceRunner::VERIFY_REJECT_TOKEN));
        EXPECT_THAT(harness.child1->lastForwardTokens(),
                    ElementsAre(MockInferenceRunner::VERIFY_REJECT_TOKEN));

        const auto probe = harness.runner->prefixStateProbe();
        EXPECT_EQ(probe.mtp_draft_steps, 1u);
        EXPECT_EQ(probe.mtp_verifier_runs, 1u);
        EXPECT_EQ(probe.mtp_verifier_token_count, 2u);
        EXPECT_EQ(probe.mtp_accepted_tokens, 0u);
        EXPECT_EQ(probe.mtp_rejected_tokens, 1u);
        EXPECT_EQ(probe.mtp_rollbacks, 0u);
    }

    TEST_F(Test__PrefillDecodeTransition, LocalTPMTPColumnParallelAcceptsGatheredDraftAndVerifierLogits)
    {
        auto harness = createLocalTPRunner(/*mtp_accept=*/true, /*column_parallel_logits=*/true);

        std::vector<int32_t> prompt = {1, 2, 3, 4, 5};
        ASSERT_TRUE(harness.runner->prefill(prompt));

        GenerationResult step1 = harness.runner->decodeStep();
        ASSERT_TRUE(step1.success()) << step1.error;
        EXPECT_THAT(step1.tokens,
                    ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN,
                                MockInferenceRunner::MTP_ARGMAX_TOKEN));

        EXPECT_EQ(harness.child0->forwardMTPCount(), 1);
        EXPECT_EQ(harness.child1->forwardMTPCount(), 1);
        EXPECT_EQ(harness.child0->lastMTPConditionToken(), MockInferenceRunner::PREFILL_ARGMAX_TOKEN);
        EXPECT_EQ(harness.child1->lastMTPConditionToken(), MockInferenceRunner::PREFILL_ARGMAX_TOKEN);
        EXPECT_THAT(harness.child0->lastForwardTokens(),
                    ElementsAre(MockInferenceRunner::MTP_ARGMAX_TOKEN));
        EXPECT_THAT(harness.child1->lastForwardTokens(),
                    ElementsAre(MockInferenceRunner::MTP_ARGMAX_TOKEN));

        const auto probe = harness.runner->prefixStateProbe();
        EXPECT_EQ(probe.mtp_draft_steps, 1u);
        EXPECT_EQ(probe.mtp_verifier_runs, 1u);
        EXPECT_EQ(probe.mtp_verifier_token_count, 2u);
        EXPECT_EQ(probe.mtp_accepted_tokens, 1u);
        EXPECT_EQ(probe.mtp_rejected_tokens, 0u);
        EXPECT_EQ(probe.mtp_rollbacks, 0u);
    }

    TEST_F(Test__PrefillDecodeTransition, LocalTPMTPColumnParallelRejectsUsingGatheredVerifierLogits)
    {
        auto harness = createLocalTPRunner(/*mtp_accept=*/false, /*column_parallel_logits=*/true);

        std::vector<int32_t> prompt = {1, 2, 3, 4, 5};
        ASSERT_TRUE(harness.runner->prefill(prompt));

        GenerationResult step1 = harness.runner->decodeStep();
        ASSERT_TRUE(step1.success()) << step1.error;
        EXPECT_THAT(step1.tokens,
                    ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN,
                                MockInferenceRunner::VERIFY_REJECT_TOKEN));

        EXPECT_EQ(harness.child0->forwardMTPCount(), 1);
        EXPECT_EQ(harness.child1->forwardMTPCount(), 1);
        EXPECT_EQ(harness.child0->commitMTPShiftedCount(), 2);
        EXPECT_EQ(harness.child1->commitMTPShiftedCount(), 2);
        EXPECT_EQ(harness.child0->lastCommitMTPMainForwardTokenCount(), 0);
        EXPECT_EQ(harness.child1->lastCommitMTPMainForwardTokenCount(), 0);
        EXPECT_THAT(harness.child0->lastForwardTokens(),
                    ElementsAre(MockInferenceRunner::VERIFY_REJECT_TOKEN));
        EXPECT_THAT(harness.child1->lastForwardTokens(),
                    ElementsAre(MockInferenceRunner::VERIFY_REJECT_TOKEN));

        const auto probe = harness.runner->prefixStateProbe();
        EXPECT_EQ(probe.mtp_draft_steps, 1u);
        EXPECT_EQ(probe.mtp_verifier_runs, 1u);
        EXPECT_EQ(probe.mtp_verifier_token_count, 2u);
        EXPECT_EQ(probe.mtp_accepted_tokens, 0u);
        EXPECT_EQ(probe.mtp_rejected_tokens, 1u);
        EXPECT_EQ(probe.mtp_rollbacks, 0u);
    }

    /**
     * @brief Third and subsequent decode steps continue calling forward normally
     */
    TEST_F(Test__PrefillDecodeTransition, SubsequentDecodeStepsCallForward)
    {
        auto [runner, mock] = createRunner();

        std::vector<int32_t> prompt = {1, 2, 3};
        ASSERT_TRUE(runner->prefill(prompt));

        // Step 1: skip forward, sample from prefill
        runner->decodeStep();
        EXPECT_EQ(mock->forwardCallCount(), 1);

        // Step 2: forward with step 1's token
        runner->decodeStep();
        EXPECT_EQ(mock->forwardCallCount(), 2);

        // Step 3: forward with step 2's token
        runner->decodeStep();
        EXPECT_EQ(mock->forwardCallCount(), 3);

        // Step 4: forward with step 3's token
        runner->decodeStep();
        EXPECT_EQ(mock->forwardCallCount(), 4);
    }

    // =========================================================================
    // clearCache() Reset Tests
    // =========================================================================

    /**
     * @brief clearCache resets the prefill_logits_ready flag
     *
     * After clearCache(), a new prefill/decode cycle should work correctly.
     */
    TEST_F(Test__PrefillDecodeTransition, ClearCacheResetsPrefillLogitsReady)
    {
        auto [runner, mock] = createRunner();

        // First generation cycle
        std::vector<int32_t> prompt = {1, 2, 3};
        ASSERT_TRUE(runner->prefill(prompt));
        EXPECT_EQ(mock->forwardCallCount(), 1);

        // First decode: no forward (samples from prefill logits)
        runner->decodeStep();
        EXPECT_EQ(mock->forwardCallCount(), 1);

        // Clear and start fresh
        runner->clearCache();

        // Second generation cycle
        ASSERT_TRUE(runner->prefill(prompt));
        EXPECT_EQ(mock->forwardCallCount(), 2);

        // First decode of new cycle: should again skip forward
        GenerationResult step = runner->decodeStep();
        EXPECT_EQ(mock->forwardCallCount(), 2);
        EXPECT_EQ(step.tokens[0], MockInferenceRunner::PREFILL_ARGMAX_TOKEN);
    }

    /**
     * @brief clearCache between prefill and first decode resets the flag,
     *        so next decodeStep would need a new prefill first
     */
    TEST_F(Test__PrefillDecodeTransition, ClearCacheBetweenPrefillAndDecode)
    {
        auto [runner, mock] = createRunner();

        std::vector<int32_t> prompt = {1, 2, 3};
        ASSERT_TRUE(runner->prefill(prompt));

        // Clear before first decode
        runner->clearCache();

        // New prefill
        ASSERT_TRUE(runner->prefill(prompt));
        EXPECT_EQ(mock->forwardCallCount(), 2);

        // First decode after second prefill: skips forward
        GenerationResult step = runner->decodeStep();
        EXPECT_EQ(mock->forwardCallCount(), 2);
        ASSERT_TRUE(step.success());
    }

    // =========================================================================
    // generate() Integration Tests
    // =========================================================================

    /**
     * @brief generate() uses the prefill-logits-ready flow correctly
     *
     * Verifies that generate(prompt, N) results in exactly:
     *   1 prefill forward + (N-1) decode forwards = N total forwards
     * NOT 1 prefill forward + N decode forwards (the old bug).
     */
    TEST_F(Test__PrefillDecodeTransition, GenerateUsesCorrectForwardCount)
    {
        auto [runner, mock] = createRunner();

        std::vector<int32_t> prompt = {1, 2, 3, 4, 5};
        GenerationResult result = runner->generate(prompt, 5, SamplingParams{});

        ASSERT_TRUE(result.success());
        EXPECT_EQ(result.tokens.size(), 5u);

        // 1 prefill forward + 4 decode forwards = 5 total
        // (first decode samples from prefill logits, no forward)
        EXPECT_EQ(mock->forwardCallCount(), 5);
    }

    /**
     * @brief generate() with 1 token should do 1 prefill forward and 0 decode forwards
     */
    TEST_F(Test__PrefillDecodeTransition, GenerateSingleTokenOnlyPrefillForward)
    {
        auto [runner, mock] = createRunner();

        std::vector<int32_t> prompt = {1, 2, 3};
        GenerationResult result = runner->generate(prompt, 1, SamplingParams{});

        ASSERT_TRUE(result.success());
        EXPECT_EQ(result.tokens.size(), 1u);
        EXPECT_EQ(result.tokens[0], MockInferenceRunner::PREFILL_ARGMAX_TOKEN);

        // Only prefill forward, no decode forwards
        EXPECT_EQ(mock->forwardCallCount(), 1);
    }

    TEST_F(Test__PrefillDecodeTransition, MTPGenerateHonorsMaxNewTokenBudget)
    {
        const std::filesystem::path export_path =
            std::filesystem::temp_directory_path() / "llaminar_mtp_budget_clamp_generate_unit.json";
        ScopedEnv enable("LLAMINAR_PERF_STATS_JSON", export_path.string().c_str());
        PerfStatsCollector::reset();

        auto [runner, mock] = createRunner(/*mtp_enabled=*/true, /*mtp_accept=*/true);

        SamplingParams greedy;
        greedy.temperature = 0.0f;
        std::vector<int32_t> prompt = {1, 2, 3};
        GenerationResult result = runner->generate(prompt, 1, greedy);

        ASSERT_TRUE(result.success()) << result.error;
        EXPECT_THAT(result.tokens,
                    ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN));
        EXPECT_EQ(mock->forwardMTPCount(), 0)
            << "budget-one MTP should emit the first greedy token without sidecar/verifier work";
        EXPECT_EQ(mock->captureCheckpointCount(), 1);
        EXPECT_THAT(mock->lastForwardTokens(),
                    ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN));

        const auto probe = runner->prefixStateProbe();
        EXPECT_EQ(probe.mtp_draft_steps, 0u);
        EXPECT_EQ(probe.mtp_verifier_runs, 0u);
        EXPECT_EQ(probe.mtp_verifier_token_count, 0u);
        EXPECT_EQ(probe.mtp_accepted_tokens, 0u);
        EXPECT_EQ(probe.mtp_rejected_tokens, 0u);
        EXPECT_EQ(probe.mtp_rollbacks, 0u);
        EXPECT_EQ(probe.mtp_transaction_commits, 1u);
        EXPECT_EQ(probe.mtp_transaction_rollbacks, 0u);
        EXPECT_EQ(probe.mtp_transaction_validation_failures, 0u);

        const auto records = PerfStatsCollector::snapshot({"mtp"});
        const PerfStatRecord *direct_emit =
            findPerfRecord(records, PerfStatRecord::Kind::Counter, "budget_limited_direct_emits");
        ASSERT_NE(direct_emit, nullptr);
        EXPECT_DOUBLE_EQ(direct_emit->value, 1.0);
        const PerfStatRecord *clamped =
            findPerfRecordWithTags(records,
                                   PerfStatRecord::Kind::Counter,
                                   "draft_steps_budget_clamped",
                                   {{"configured", "1"}, {"effective", "0"}, {"token_budget", "1"}});
        ASSERT_NE(clamped, nullptr);

        std::filesystem::remove(export_path);
        PerfStatsCollector::reset();
    }

    TEST_F(Test__PrefillDecodeTransition, MTPDecodeStepHonorsExplicitTokenBudget)
    {
        auto [runner, mock] = createRunner(/*mtp_enabled=*/true, /*mtp_accept=*/true);

        ASSERT_TRUE(runner->prefill({1, 2, 3}));
        runner->setDecodeStepTokenBudget(1);
        GenerationResult step = runner->decodeStep();
        runner->setDecodeStepTokenBudget(0);

        ASSERT_TRUE(step.success()) << step.error;
        EXPECT_THAT(step.tokens,
                    ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN));
        EXPECT_EQ(mock->forwardMTPCount(), 0);
        EXPECT_EQ(mock->captureCheckpointCount(), 1);
        EXPECT_THAT(mock->lastForwardTokens(),
                    ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN));

        const auto probe = runner->prefixStateProbe();
        EXPECT_EQ(probe.mtp_draft_steps, 0u);
        EXPECT_EQ(probe.mtp_verifier_runs, 0u);
        EXPECT_EQ(probe.mtp_verifier_token_count, 0u);
        EXPECT_EQ(probe.mtp_accepted_tokens, 0u);
        EXPECT_EQ(probe.mtp_rejected_tokens, 0u);
        EXPECT_EQ(probe.mtp_rollbacks, 0u);
        EXPECT_EQ(probe.mtp_transaction_commits, 1u);
        EXPECT_EQ(probe.mtp_transaction_rollbacks, 0u);
        EXPECT_EQ(probe.mtp_transaction_validation_failures, 0u);
    }

    TEST_F(Test__PrefillDecodeTransition, MTPTransactionRejectsUnsafeVerifierPrefillSnapshot)
    {
        auto [runner, mock] = createRunner(/*mtp_enabled=*/true, /*mtp_accept=*/true);

        ASSERT_TRUE(runner->prefill({1, 2, 3}));
        PrefixStateSnapshot unsafe_snapshot;
        unsafe_snapshot.valid = true;
        unsafe_snapshot.logical_checkpoint = true;
        unsafe_snapshot.provenance = PrefixStateProvenance::VerifierPrefillRows;
        unsafe_snapshot.cached_tokens = 3;
        unsafe_snapshot.mtp_cached_tokens = {2};
        mock->setCapturedSnapshot(unsafe_snapshot);

        runner->setDecodeStepTokenBudget(1);
        GenerationResult step = runner->decodeStep();
        runner->setDecodeStepTokenBudget(0);

        ASSERT_FALSE(step.success());
        EXPECT_THAT(step.error, HasSubstr("MTP transaction validation failed"));
        EXPECT_EQ(mock->forwardMTPCount(), 0);

        const auto probe = runner->prefixStateProbe();
        EXPECT_EQ(probe.mtp_transaction_commits, 0u);
        EXPECT_EQ(probe.mtp_transaction_validation_failures, 1u);
        EXPECT_EQ(probe.mtp_unsafe_verifier_state_rejections, 1u);
    }

    TEST_F(Test__PrefillDecodeTransition, MTPBudgetClampReducesChainedDraftDepthBeforeVerifier)
    {
        const std::filesystem::path export_path =
            std::filesystem::temp_directory_path() / "llaminar_mtp_budget_depth_clamp_unit.json";
        {
            ScopedEnv enable("LLAMINAR_PERF_STATS_JSON", export_path.string().c_str());
            PerfStatsCollector::reset();

            auto [runner, mock] = createRunner(
                /*mtp_enabled=*/true,
                /*mtp_accept=*/true,
                /*mtp_unsupported_reason=*/{},
                /*mpi_ctx=*/nullptr,
                /*mtp_token_coordination=*/false,
                /*hide_local_logits=*/false,
                DeviceId::cpu(),
                /*mtp_draft_tokens=*/3,
                /*chained_mtp_support=*/true);

            ASSERT_TRUE(runner->prefill({1, 2, 3}));
            runner->setDecodeStepTokenBudget(2);
            GenerationResult step = runner->decodeStep();
            runner->setDecodeStepTokenBudget(0);

            ASSERT_TRUE(step.success()) << step.error;
            EXPECT_THAT(step.tokens,
                        ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN,
                                    MockInferenceRunner::MTP_ARGMAX_TOKEN));
            EXPECT_EQ(mock->forwardMTPCount(), 1);
            EXPECT_EQ(mock->forwardMTPFromLastDraftCount(), 0)
                << "token budget leaves room for only one speculative output";
            EXPECT_THAT(mock->lastForwardTokens(),
                        ElementsAre(MockInferenceRunner::MTP_ARGMAX_TOKEN));

            const auto probe = runner->prefixStateProbe();
            EXPECT_EQ(probe.mtp_draft_steps, 1u);
            EXPECT_EQ(probe.mtp_verifier_runs, 1u);
            EXPECT_EQ(probe.mtp_verifier_token_count, 2u);
            EXPECT_EQ(probe.mtp_accepted_tokens, 1u);
            EXPECT_EQ(probe.mtp_rollbacks, 0u);

            const auto records = PerfStatsCollector::snapshot({"mtp"});
            const PerfStatRecord *clamped =
                findPerfRecordWithTags(records,
                                       PerfStatRecord::Kind::Counter,
                                       "draft_steps_budget_clamped",
                                       {{"configured", "3"},
                                        {"effective", "1"},
                                        {"token_budget", "2"}});
            ASSERT_NE(clamped, nullptr);
            const PerfStatRecord *skipped =
                findPerfRecord(records, PerfStatRecord::Kind::Counter, "draft_steps_budget_skipped");
            ASSERT_NE(skipped, nullptr);
            EXPECT_DOUBLE_EQ(skipped->value, 2.0);
        }
        std::filesystem::remove(export_path);
        PerfStatsCollector::reset();
    }

    TEST_F(Test__PrefillDecodeTransition, MTPBudgetClampDirectEmitDoesNotRunVerifier)
    {
        const std::filesystem::path export_path =
            std::filesystem::temp_directory_path() / "llaminar_mtp_budget_direct_emit_unit.json";
        {
            ScopedEnv enable("LLAMINAR_PERF_STATS_JSON", export_path.string().c_str());
            PerfStatsCollector::reset();

            auto [runner, mock] = createRunner(/*mtp_enabled=*/true, /*mtp_accept=*/true);

            ASSERT_TRUE(runner->prefill({1, 2, 3}));
            const int forward_count_after_prefill = mock->forwardCallCount();

            runner->setDecodeStepTokenBudget(1);
            GenerationResult step = runner->decodeStep();
            runner->setDecodeStepTokenBudget(0);

            ASSERT_TRUE(step.success()) << step.error;
            EXPECT_THAT(step.tokens,
                        ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN));
            EXPECT_EQ(mock->restoreCount(), 0);
            EXPECT_EQ(mock->commitMTPShiftedCount(), 1);
            EXPECT_EQ(mock->forwardMTPCount(), 0);
            EXPECT_EQ(mock->forwardCallCount(), forward_count_after_prefill + 1)
                << "budget-one decode should advance state once but not run verifier or replay work";
            EXPECT_THAT(mock->lastForwardTokens(),
                        ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN));

            const auto records = PerfStatsCollector::snapshot({"mtp"});
            const PerfStatRecord *direct_emit =
                findPerfRecord(records, PerfStatRecord::Kind::Counter, "budget_limited_direct_emits");
            ASSERT_NE(direct_emit, nullptr);
            EXPECT_DOUBLE_EQ(direct_emit->value, 1.0);

            const PerfStatRecord *clamped =
                findPerfRecordWithTags(records,
                                       PerfStatRecord::Kind::Counter,
                                       "draft_steps_budget_clamped",
                                       {{"configured", "1"},
                                        {"effective", "0"},
                                        {"token_budget", "1"}});
            ASSERT_NE(clamped, nullptr);

            const PerfStatRecord *replay_tokens =
                findPerfRecord(records, PerfStatRecord::Kind::Counter, "replay_tokens");
            EXPECT_EQ(replay_tokens, nullptr);

            const PerfStatRecord *replay_forward =
                findPerfRecord(records, PerfStatRecord::Kind::Timer, "replay_forward");
            EXPECT_EQ(replay_forward, nullptr);
        }
        std::filesystem::remove(export_path);
        PerfStatsCollector::reset();
    }

    TEST_F(Test__PrefillDecodeTransition, MTPGenerateCountsAcceptedDraftsTowardMaxNewTokens)
    {
        auto [runner, mock] = createRunner(/*mtp_enabled=*/true, /*mtp_accept=*/true);

        SamplingParams greedy;
        greedy.temperature = 0.0f;

        std::vector<int32_t> prompt = {1, 2, 3};
        GenerationResult result = runner->generate(prompt, 2, greedy);

        ASSERT_TRUE(result.success()) << result.error;
        EXPECT_THAT(result.tokens,
                    ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN,
                                MockInferenceRunner::MTP_ARGMAX_TOKEN));
        EXPECT_EQ(mock->forwardMTPCount(), 1);
        EXPECT_THAT(mock->lastForwardTokens(),
                    ElementsAre(MockInferenceRunner::MTP_ARGMAX_TOKEN));
    }

    /**
     * @brief generate() first token should be the prefill argmax, not a re-forwarded token
     */
    TEST_F(Test__PrefillDecodeTransition, GenerateFirstTokenIsPrefillArgmax)
    {
        auto [runner, mock] = createRunner();

        std::vector<int32_t> prompt = {1, 2};
        GenerationResult result = runner->generate(prompt, 3, SamplingParams{});

        ASSERT_TRUE(result.success());
        ASSERT_GE(result.tokens.size(), 1u);

        // First generated token comes from prefill logits
        EXPECT_EQ(result.tokens[0], MockInferenceRunner::PREFILL_ARGMAX_TOKEN);
    }

    // =========================================================================
    // Edge Cases
    // =========================================================================

    /**
     * @brief Single-token prompt still works correctly
     */
    TEST_F(Test__PrefillDecodeTransition, SingleTokenPrompt)
    {
        auto [runner, mock] = createRunner();

        std::vector<int32_t> prompt = {42};
        ASSERT_TRUE(runner->prefill(prompt));
        EXPECT_EQ(mock->forwardCallCount(), 1);
        EXPECT_THAT(mock->lastForwardTokens(), ElementsAre(42));

        // First decode: samples from prefill logits, no forward
        GenerationResult step1 = runner->decodeStep();
        ASSERT_TRUE(step1.success());
        EXPECT_EQ(mock->forwardCallCount(), 1);
        EXPECT_EQ(step1.tokens[0], MockInferenceRunner::PREFILL_ARGMAX_TOKEN);
    }

    /**
     * @brief Multiple prefill/decode cycles work correctly
     */
    TEST_F(Test__PrefillDecodeTransition, MultipleCycles)
    {
        auto [runner, mock] = createRunner();

        for (int cycle = 0; cycle < 3; ++cycle)
        {
            std::vector<int32_t> prompt = {10, 20, 30};
            ASSERT_TRUE(runner->prefill(prompt));

            // First decode skips forward
            int forwards_before = mock->forwardCallCount();
            GenerationResult step = runner->decodeStep();
            ASSERT_TRUE(step.success()) << "Cycle " << cycle;
            EXPECT_EQ(mock->forwardCallCount(), forwards_before)
                << "First decode in cycle " << cycle << " should not call forward";

            // Second decode calls forward
            runner->decodeStep();
            EXPECT_EQ(mock->forwardCallCount(), forwards_before + 1)
                << "Second decode in cycle " << cycle << " should call forward";

            runner->clearCache();
        }
    }

} // namespace
