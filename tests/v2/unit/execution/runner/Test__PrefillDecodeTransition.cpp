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
#include "execution/mtp/MTPSpecDecodeMetadata.h"
#include "execution/mtp/MTPSpecStateContract.h"
#include "config/OrchestrationConfig.h"
#include "execution/mpi_orchestration/RankExecutionPlan.h"
#include "backends/GlobalDeviceAddress.h"
#include "kernels/common/SamplingMath.h"
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
    float mtpSeededVerifierThreshold(uint64_t seed,
                                     int logical_position,
                                     int purpose)
    {
        return sampling_math::mtp_spec_threshold_from_seed(
            seed,
            logical_position,
            purpose);
    }

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
            const bool was_first_forward_in_cycle = is_first_forward_in_cycle_;
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
            if (mtp_enabled_ && was_first_forward_in_cycle)
            {
                mtp_shifted_cached_tokens_ =
                    shiftedTargetForMainTokens(position_);
            }

            return true;
        }

        bool forward_batch(const std::vector<std::vector<int>> &token_batches) override
        {
            ++forward_batch_call_count_;
            last_forward_batch_ = token_batches;
            sequence_lengths_.clear();
            sequence_lengths_.reserve(token_batches.size());
            padded_seq_len_ = 0;
            for (const auto &tokens : token_batches)
            {
                sequence_lengths_.push_back(static_cast<int>(tokens.size()));
                padded_seq_len_ = std::max(
                    padded_seq_len_,
                    static_cast<int>(tokens.size()));
            }
            position_ += padded_seq_len_;
            if (all_position_logits_enabled_)
            {
                setupAllPositionLogitsForBatch(token_batches);
                return !token_batches.empty();
            }
            setupPrefillLogits();
            setupBatchPrefillLogits();
            is_first_forward_in_cycle_ = false;
            if (mtp_enabled_)
            {
                mtp_shifted_cached_tokens_ =
                    shiftedTargetForMainTokens(position_);
            }
            return !token_batches.empty();
        }

        bool forwardWithDeviceTokenIds(
            const int *token_shadow,
            const void *token_ids_device,
            int seq_len) override
        {
            ++forward_with_device_token_ids_count_;
            last_forward_device_token_ids_ = token_ids_device;
            last_forward_device_token_seq_len_ = seq_len;
            return token_shadow && token_ids_device && forward(token_shadow, seq_len);
        }

        const void *prepareMTPVerifierInputTokensOnDevice(
            int32_t first_token,
            int first_draft_slot,
            int draft_token_count,
            int total_verifier_input_tokens) override
        {
            ++prepare_mtp_verifier_input_tokens_on_device_count_;
            last_prepare_mtp_verifier_first_token_ = first_token;
            last_prepare_mtp_verifier_first_draft_slot_ = first_draft_slot;
            last_prepare_mtp_verifier_draft_token_count_ = draft_token_count;
            last_prepare_mtp_verifier_total_tokens_ = total_verifier_input_tokens;
            if (!supports_stochastic_device_sampling_ ||
                first_draft_slot < 0 ||
                draft_token_count < 0 ||
                total_verifier_input_tokens != draft_token_count + 1 ||
                total_verifier_input_tokens >
                    static_cast<int>(device_verifier_input_tokens_.size()) ||
                first_draft_slot + draft_token_count >
                    static_cast<int>(device_draft_sample_tokens_.size()))
            {
                return nullptr;
            }

            device_verifier_input_tokens_[0] = first_token;
            for (int i = 0; i < draft_token_count; ++i)
            {
                device_verifier_input_tokens_[static_cast<size_t>(i + 1)] =
                    device_draft_sample_tokens_[static_cast<size_t>(
                        first_draft_slot + i)];
            }
            return device_verifier_input_tokens_.data();
        }

        const void *prepareMTPVerifierInputTokensOnDeviceFromDeviceFirstToken(
            int first_target_sample_slot,
            int first_draft_slot,
            int draft_token_count,
            int total_verifier_input_tokens) override
        {
            ++prepare_mtp_verifier_input_tokens_on_device_count_;
            ++prepare_mtp_verifier_input_tokens_device_first_count_;
            last_prepare_mtp_verifier_first_target_sample_slot_ =
                first_target_sample_slot;
            last_prepare_mtp_verifier_first_draft_slot_ = first_draft_slot;
            last_prepare_mtp_verifier_draft_token_count_ = draft_token_count;
            last_prepare_mtp_verifier_total_tokens_ = total_verifier_input_tokens;
            if (!supports_stochastic_device_sampling_ ||
                first_target_sample_slot < 0 ||
                first_target_sample_slot >=
                    static_cast<int>(device_target_sample_tokens_.size()) ||
                first_draft_slot < 0 ||
                draft_token_count < 0 ||
                total_verifier_input_tokens != draft_token_count + 1 ||
                total_verifier_input_tokens >
                    static_cast<int>(device_verifier_input_tokens_.size()) ||
                first_draft_slot + draft_token_count >
                    static_cast<int>(device_draft_sample_tokens_.size()))
            {
                return nullptr;
            }

            device_verifier_input_tokens_[0] =
                device_target_sample_tokens_[
                    static_cast<size_t>(first_target_sample_slot)];
            for (int i = 0; i < draft_token_count; ++i)
            {
                device_verifier_input_tokens_[static_cast<size_t>(i + 1)] =
                    device_draft_sample_tokens_[static_cast<size_t>(
                        first_draft_slot + i)];
            }
            return device_verifier_input_tokens_.data();
        }

        const float *logits() const override
        {
            if (hide_local_logits_)
                return nullptr;
            if (column_parallel_logits_)
                return nullptr;
            return logits_.data();
        }

        const float *getLogits(int seq_idx = 0) const override
        {
            ++get_logits_count_;
            if (!batch_logits_.empty())
            {
                if (seq_idx < 0 ||
                    seq_idx >= static_cast<int>(sequence_lengths_.size()) ||
                    padded_seq_len_ <= 0)
                {
                    return nullptr;
                }
                return batch_logits_.data() +
                       static_cast<size_t>(seq_idx) *
                           static_cast<size_t>(padded_seq_len_) *
                           static_cast<size_t>(VOCAB_SIZE);
            }
            return logits();
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
            appendSpeculativeSidecarShiftedRow();
            return true;
        }

        /**
         * @brief Mock the sidecar path that leaves logits ordered for device sampling.
         *
         * The production runner may leave a GPU graph stream pending here. The
         * unit mock only records that OrchestrationRunner chose this path, then
         * delegates to the normal sidecar implementation so existing fake logits
         * stay identical.
         */
        bool forwardMTPForDeviceSampling(int32_t draft_condition_token) override
        {
            ++forward_mtp_for_device_sampling_count_;
            return forwardMTP(draft_condition_token);
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
            appendSpeculativeSidecarShiftedRow();
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
            int position_offset_override = -1,
            int already_appended_shifted_kv_tokens = -1) override
        {
            commit_mtp_shifted_count_++;
            last_commit_mtp_already_appended_ = already_appended_tokens;
            last_commit_mtp_main_forward_token_count_ = main_forward_token_count;
            last_commit_mtp_allow_speculative_discard_ = allow_speculative_discard;
            last_commit_mtp_position_offset_override_ = position_offset_override;
            last_commit_mtp_already_appended_shifted_kv_ =
                already_appended_shifted_kv_tokens;
            last_commit_mtp_tokens_.clear();
            if (tokens && token_count > 0)
            {
                last_commit_mtp_tokens_.assign(tokens, tokens + token_count);
            }
            const int catchup_token_count = token_count - already_appended_tokens;
            const int hidden_source_row_start = already_appended_tokens - 1;
            const int hidden_source_row_end = hidden_source_row_start + catchup_token_count;
            const bool rows_valid =
                token_count <= already_appended_tokens ||
                (tokens != nullptr &&
                 main_forward_token_count > 0 &&
                 hidden_source_row_start >= 0 &&
                 hidden_source_row_end <= main_forward_token_count);
            if (!rows_valid)
                return false;

            const int position_offset =
                position_offset_override >= 0
                    ? position_offset_override
                    : position_;
            const int resident_shifted_kv_tokens =
                already_appended_shifted_kv_tokens >= 0
                    ? already_appended_shifted_kv_tokens
                    : already_appended_tokens;
            const int expected_before =
                std::max(0, position_offset - 1 + resident_shifted_kv_tokens);
            if (mtp_shifted_cached_tokens_ > expected_before)
            {
                if (mtp_draft_tokens_ <= 1 && !allow_speculative_discard)
                    return false;
                mtp_shifted_cached_tokens_ = expected_before;
            }
            if (mtp_shifted_cached_tokens_ < expected_before)
                return false;
            mtp_shifted_cached_tokens_ =
                expected_before +
                std::max(0, token_count - already_appended_tokens);
            return true;
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
            return appendOneShiftedMTPRow(
                already_appended_tokens,
                position_offset_override);
        }

        bool commitMTPShiftedRowFromDeviceTargetSample(
            int target_sample_slot,
            int already_appended_tokens,
            bool allow_speculative_discard = false,
            int position_offset_override = -1) override
        {
            ++device_target_shifted_commit_count_;
            if (!supports_mtp_device_draft_token_input_ ||
                target_sample_slot < 0 ||
                target_sample_slot >= static_cast<int>(device_target_sample_tokens_.size()))
            {
                return false;
            }
            const int32_t token =
                device_target_sample_tokens_[static_cast<size_t>(target_sample_slot)];
            if (token < 0)
                return false;
            last_device_target_shifted_commit_token_ = token;
            return commitMTPShiftedRowFromCurrentTerminalHidden(
                token,
                already_appended_tokens,
                allow_speculative_discard,
                position_offset_override);
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

        /**
         * @brief Mock the production compact verifier-logits capability.
         *
         * Production graph builders treat row_count as a fixed graph shape. The
         * mock follows the same 1..4 contract so runner tests fail if
         * OrchestrationRunner forgets to enable row-indexed verifier logits
         * around the all-position verifier forward.
         */
        bool setComputeRowIndexedAllPositionLogits(bool enabled, int row_count) override
        {
            if (!mtp_enabled_)
                return false;
            if (enabled && (row_count <= 0 || row_count > 16))
                return false;
            row_indexed_all_position_logits_enabled_ = enabled;
            row_indexed_all_position_logits_row_count_ = enabled ? row_count : 0;
            ++set_row_indexed_all_position_count_;
            return true;
        }

        bool setMTPSpecVerifierInputPlan(
            const MTPSpecDecodeVerifierInputPlan &plan) override
        {
            if (!mtp_enabled_ || !plan.ok)
                return false;
            ++set_mtp_spec_verifier_plan_count_;
            last_mtp_spec_verifier_plan_ = plan;
            last_mtp_spec_verifier_rows_ = plan.verifier_logit_rows;
            last_mtp_spec_verifier_tokens_ = plan.verifier_input_tokens;
            /*
             * The real GPU runner snapshots pre-verifier base cache counts
             * into device metadata before verifier graph replay mutates live
             * KV/GDN state.  Mirror that timing in the mock: verifier
             * forward_batch() rewrites sequence_lengths_ to verifier-row
             * lengths, so publication must not read it after replay.
             */
            for (int request_index = 0;
                 request_index < plan.request_count &&
                 request_index < kMockResidentOutcomeRequestCapacity;
                 ++request_index)
            {
                resident_base_cached_tokens_[static_cast<size_t>(request_index)] =
                    request_index < static_cast<int>(sequence_lengths_.size())
                        ? sequence_lengths_[static_cast<size_t>(request_index)]
                        : position_;
            }
            mtp_spec_verifier_plan_installed_ = true;
            return true;
        }

        void clearMTPSpecVerifierInputPlan() override
        {
            ++clear_mtp_spec_verifier_plan_count_;
            mtp_spec_verifier_plan_installed_ = false;
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

        bool supportsGreedyAllPositionBatchOutcomeOnDevice() const override
        {
            return primary_device_.is_gpu() && supports_mtp_token_coordination_;
        }

        bool supportsRowLocalAllPositionPenaltyApplication() const override
        {
            return primary_device_.is_gpu() && supports_mtp_token_coordination_;
        }

        bool supportsMTPSidecarSampleFusion() const override
        {
            return supports_mtp_sidecar_sample_fusion_;
        }

        bool supportsMTPSidecarLogitsStreamHandoff() const override
        {
            return supports_mtp_sidecar_stream_handoff_;
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

        bool supportsLogicalMTPVerifierBaseCheckpoint() const override
        {
            return supports_logical_mtp_verifier_base_checkpoint_;
        }

        bool requiresMTPDecodeEquivalentVerifierReplay() const override
        {
            return requires_mtp_decode_equivalent_replay_;
        }

        bool supportsMTPSpecStatePublication() const override
        {
            return supports_mtp_spec_state_publication_ &&
                   !hide_mtp_spec_state_publication_from_policy_;
        }

        bool supportsDeviceResidentMTPSpecStatePublication() const override
        {
            return supports_device_resident_mtp_spec_state_publication_;
        }

        MTPVerifierEconomyCapability mtpVerifierEconomyCapability() const override
        {
            return mtp_verifier_economy_capability_;
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
            adoptPublishedMainTokens(plan.target_cached_tokens);
            all_position_logits_enabled_ = false;
            row_indexed_all_position_logits_enabled_ = false;
            row_indexed_all_position_logits_row_count_ = 0;
            return true;
        }

        bool publishAcceptedMTPSpecStateBatch(
            const MTPSpecStepPlanBatch &plans,
            std::string *error = nullptr) override
        {
            ++publish_mtp_spec_state_batch_count_;
            publication_events_.push_back("host_plan_publish");
            last_published_mtp_spec_batch_ = plans;
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
            if (!plans.ok || plans.steps.empty())
            {
                if (error)
                    *error = plans.ok
                                 ? "mock MTP spec-state batch has no steps"
                                 : plans.error;
                return false;
            }

            ++publish_mtp_spec_state_count_;
            last_published_mtp_spec_step_ = plans.steps.front();
            adoptPublishedMainTokens(plans.steps.front().target_cached_tokens);
            for (const MTPSpecStepPlan &step : plans.steps)
            {
                if (step.request_index >= 0 &&
                    step.request_index < static_cast<int>(sequence_lengths_.size()))
                {
                    sequence_lengths_[static_cast<size_t>(step.request_index)] =
                        step.target_cached_tokens;
                    if (step.request_index == 0)
                    {
                        mtp_shifted_cached_tokens_ =
                            shiftedTargetForMainTokens(step.target_cached_tokens);
                    }
                }
            }
            all_position_logits_enabled_ = false;
            row_indexed_all_position_logits_enabled_ = false;
            row_indexed_all_position_logits_row_count_ = 0;
            return true;
        }

        bool publishAcceptedMTPSpecStateBatchFromDeviceOutcome(
            const DeviceSpeculativePublicationRequest &request,
            std::string *error = nullptr) override
        {
            using namespace sampling_math;
            ++publish_device_resident_mtp_spec_state_count_;
            publication_events_.push_back("device_outcome_publish");
            last_device_resident_publication_request_ = request;
            if (!supports_device_resident_mtp_spec_state_publication_)
            {
                if (error)
                    *error = "mock device-resident MTP spec-state publication is disabled";
                return false;
            }
            if (!supports_mtp_spec_state_publication_)
            {
                if (error)
                    *error = "mock host-plan MTP spec-state publication is disabled";
                return false;
            }
            if (!publish_mtp_spec_state_ok_)
            {
                if (error)
                    *error = "mock MTP spec-state publication failed";
                return false;
            }
            if (!request.valid())
            {
                if (error)
                    *error = "mock device-resident MTP publication request is invalid";
                return false;
            }

            const int *meta = request.outcome.meta_device;
            for (int request_index = 0;
                 request_index < request.request_count;
                 ++request_index)
            {
                const size_t meta_base =
                    static_cast<size_t>(request_index) *
                    static_cast<size_t>(request.outcome.meta_stride);
                const int *row_meta = meta + meta_base;
                if (row_meta[kSpecBatchMetaOk] == 0)
                {
                    if (error)
                        *error = "mock device-resident MTP publication row is invalid";
                    return false;
                }
                const int base_cached_tokens =
                    !request.base_cached_tokens.empty()
                        ? request.base_cached_tokens[static_cast<size_t>(request_index)]
                        : (resident_base_cached_tokens_[static_cast<size_t>(request_index)] >= 0
                               ? resident_base_cached_tokens_[static_cast<size_t>(request_index)]
                               : (request_index < static_cast<int>(sequence_lengths_.size())
                                      ? sequence_lengths_[static_cast<size_t>(request_index)]
                                      : position_));
                const int target_cached_tokens =
                    base_cached_tokens +
                    row_meta[kSpecBatchMetaTargetVerifierStateCommitCount];
                resident_target_positions_[static_cast<size_t>(request_index)] =
                    target_cached_tokens;
                resident_target_sequence_lengths_[static_cast<size_t>(request_index)] =
                    target_cached_tokens;
                resident_accepted_state_counts_[static_cast<size_t>(request_index)] =
                    row_meta[kSpecBatchMetaTargetVerifierStateCommitCount];
                int publication_ok = 0;
                int32_t next_condition_token = -1;
                int all_drafts_accepted = 0;
                int stopped = 0;
                derive_speculative_publication_metadata(
                    request.outcome.meta_device,
                    request.outcome.meta_stride,
                    request_index,
                    /*padded_state_rows_per_request=*/request.max_draft_tokens,
                    base_cached_tokens,
                    request.max_draft_tokens,
                    nullptr,
                    nullptr,
                    nullptr,
                    &publication_ok,
                    request.outcome.output_tokens_device,
                    request.outcome.output_token_stride,
                    &next_condition_token,
                    &all_drafts_accepted,
                    &stopped);
                if (publication_ok == 0)
                {
                    if (error)
                        *error = "mock device-resident MTP publication metadata derivation failed";
                    return false;
                }
                resident_next_condition_tokens_[static_cast<size_t>(request_index)] =
                    next_condition_token;
                resident_all_drafts_accepted_flags_[static_cast<size_t>(request_index)] =
                    all_drafts_accepted != 0 ? 1 : 0;
                resident_stopped_flags_[static_cast<size_t>(request_index)] =
                    stopped != 0 ? 1 : 0;
                resident_publication_ok_flags_[static_cast<size_t>(request_index)] = 1;
            }
            resident_logical_state_request_count_ = request.request_count;
            resident_logical_state_valid_ = true;
            all_position_logits_enabled_ = false;
            row_indexed_all_position_logits_enabled_ = false;
            row_indexed_all_position_logits_row_count_ = 0;
            return true;
        }

        bool adoptDeviceResidentMTPSpecPublishedHostState(
            const MTPSpecStepPlanBatch &plans,
            std::string *error = nullptr) override
        {
            ++adopt_device_resident_host_state_count_;
            publication_events_.push_back("host_state_adopt");
            last_adopted_device_resident_host_state_ = plans;
            if (!supports_device_resident_mtp_spec_state_publication_)
            {
                if (error)
                    *error = "mock device-resident host-state adoption is disabled";
                return false;
            }
            if (!plans.ok || plans.steps.empty())
            {
                if (error)
                    *error = plans.ok
                                 ? "mock device-resident host-state adoption has no steps"
                                 : plans.error;
                return false;
            }
            if (sequence_lengths_.size() <
                static_cast<size_t>(std::max(1, batch_capacity_)))
            {
                sequence_lengths_.resize(
                    static_cast<size_t>(std::max(1, batch_capacity_)),
                    position_);
            }
            for (const MTPSpecStepPlan &step : plans.steps)
            {
                if (step.request_index < 0 ||
                    step.request_index >= static_cast<int>(sequence_lengths_.size()))
                {
                    if (error)
                        *error = "mock device-resident host-state adoption request index is out of range";
                    return false;
                }
                sequence_lengths_[static_cast<size_t>(step.request_index)] =
                    step.target_cached_tokens;
                if (step.request_index == 0)
                    adoptPublishedMainTokens(step.target_cached_tokens);
            }
            return true;
        }

        bool adoptDeviceResidentMTPSpecPublishedHostStateFromDeviceMetadata(
            const DeviceResidentHostStateAdoptionRequest &request,
            std::string *error = nullptr) override
        {
            ++adopt_device_resident_host_state_count_;
            publication_events_.push_back("host_state_adopt");
            if (!supports_device_resident_mtp_spec_state_publication_)
            {
                if (error)
                    *error = "mock device-resident metadata host-state adoption is disabled";
                return false;
            }
            if (!request.valid() ||
                !request.logical_state.sameMailboxAs(deviceResidentLogicalSequenceState()))
            {
                if (error)
                    *error = "mock device-resident metadata host-state adoption request is invalid";
                return false;
            }
            if (sequence_lengths_.size() <
                static_cast<size_t>(std::max(1, batch_capacity_)))
            {
                sequence_lengths_.resize(
                    static_cast<size_t>(std::max(1, batch_capacity_)),
                    position_);
            }
            for (int request_index = 0;
                 request_index < request.logical_state.request_count;
                 ++request_index)
            {
                const int32_t base =
                    request.base_cached_tokens[static_cast<size_t>(request_index)];
                const int32_t target =
                    request.logical_state
                        .targetSequenceLengthDeviceForRequest(request_index)[0];
                const int32_t accepted =
                    request.logical_state
                        .acceptedStateCountDeviceForRequest(request_index)[0];
                const int32_t ok =
                    request.logical_state
                        .publicationOkFlagDeviceForRequest(request_index)[0];
                if (ok == 0 || target != base + accepted)
                {
                    if (error)
                    {
                        std::ostringstream msg;
                        msg << "mock device-resident metadata host-state adoption saw invalid counts"
                            << " request=" << request_index
                            << " base=" << base
                            << " accepted=" << accepted
                            << " target=" << target
                            << " ok=" << ok;
                        *error = msg.str();
                    }
                    return false;
                }
                sequence_lengths_[static_cast<size_t>(request_index)] = target;
                if (request_index == 0)
                    adoptPublishedMainTokens(target);
            }
            return true;
        }

        DeviceResidentLogicalSequenceStateHandle deviceResidentLogicalSequenceState() const override
        {
            if (!resident_logical_state_valid_)
                return {};

            DeviceResidentLogicalSequenceStateHandle handle;
            handle.target_positions_device = resident_target_positions_.data();
            handle.target_sequence_lengths_device =
                resident_target_sequence_lengths_.data();
            handle.accepted_state_counts_device =
                resident_accepted_state_counts_.data();
            handle.next_condition_tokens_device =
                resident_next_condition_tokens_.data();
            handle.all_drafts_accepted_flags_device =
                resident_all_drafts_accepted_flags_.data();
            handle.stopped_flags_device =
                resident_stopped_flags_.data();
            handle.publication_ok_flags_device =
                resident_publication_ok_flags_.data();
            handle.request_count = resident_logical_state_request_count_;
            handle.device = primary_device_;
            handle.stream = const_cast<int *>(&resident_stream_token_);
            handle.ready_event = const_cast<int *>(&resident_ready_event_token_);
            handle.live_state_epoch = 1;
            return handle;
        }

        bool commitMTPShiftedRowFromDeviceResidentLogicalState(
            const DeviceResidentLogicalSequenceStateHandle &logical_state,
            int request_index,
            int already_appended_tokens,
            bool allow_speculative_discard = false,
            int position_offset_override = -1) override
        {
            ++resident_logical_state_shifted_commit_count_;
            if (!logical_state.coversRequest(request_index) ||
                logical_state.accepted_state_counts_device !=
                    resident_accepted_state_counts_.data() ||
                logical_state.next_condition_tokens_device !=
                    resident_next_condition_tokens_.data() ||
                logical_state.all_drafts_accepted_flags_device !=
                    resident_all_drafts_accepted_flags_.data() ||
                logical_state.stopped_flags_device !=
                    resident_stopped_flags_.data())
            {
                return false;
            }
            const int32_t token =
                logical_state.next_condition_tokens_device[request_index];
            if (token < 0)
                return false;

            ++commit_mtp_shifted_count_;
            last_commit_mtp_already_appended_ = already_appended_tokens;
            last_commit_mtp_main_forward_token_count_ = 0;
            last_commit_mtp_allow_speculative_discard_ =
                allow_speculative_discard;
            last_commit_mtp_position_offset_override_ =
                position_offset_override;
            last_commit_mtp_tokens_.assign(1, token);
            last_resident_logical_state_shifted_commit_token_ = token;
            return appendOneShiftedMTPRow(
                already_appended_tokens,
                position_offset_override);
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

        bool forwardMTPBatchAndSampleGreedy(
            const int32_t *draft_condition_tokens,
            const int *position_ids,
            int request_batch,
            int32_t *out_tokens) override
        {
            ++forward_mtp_batch_and_sample_count_;
            last_mtp_batch_condition_tokens_.clear();
            last_mtp_batch_position_ids_.clear();
            if (!mtp_enabled_ ||
                !draft_condition_tokens ||
                !position_ids ||
                !out_tokens ||
                request_batch <= 0 ||
                request_batch > batch_capacity_)
            {
                return false;
            }

            last_mtp_batch_condition_tokens_.assign(
                draft_condition_tokens,
                draft_condition_tokens + request_batch);
            last_mtp_batch_position_ids_.assign(
                position_ids,
                position_ids + request_batch);
            for (int i = 0; i < request_batch; ++i)
            {
                if (position_ids[i] < 0)
                    return false;
                out_tokens[i] = MTP_ARGMAX_TOKEN;
            }
            mtp_logits_.assign(VOCAB_SIZE, -10.0f);
            mtp_logits_[MTP_ARGMAX_TOKEN] = 10.0f;
            return true;
        }

        bool forwardMTPBatchAndSampleGreedyToDeviceDraftSlots(
            const int32_t *draft_condition_tokens,
            const int *position_ids,
            int request_batch,
            int first_draft_slot,
            int slot_stride,
            int32_t *out_tokens) override
        {
            ++forward_mtp_batch_and_sample_to_device_draft_slots_count_;
            last_mtp_batch_device_draft_first_slot_ = first_draft_slot;
            last_mtp_batch_device_draft_slot_stride_ = slot_stride;
            const int last_slot =
                first_draft_slot +
                (request_batch > 0 ? (request_batch - 1) * slot_stride : 0);
            if (first_draft_slot < 0 ||
                slot_stride <= 0 ||
                last_slot >=
                    static_cast<int>(device_draft_sample_tokens_.size()))
            {
                return false;
            }

            std::array<int32_t, 8> shadow{};
            shadow.fill(-1);
            if (!forwardMTPBatchAndSampleGreedy(
                    draft_condition_tokens,
                    position_ids,
                    request_batch,
                    shadow.data()))
            {
                return false;
            }

            for (int i = 0; i < request_batch; ++i)
            {
                const int32_t token = shadow[static_cast<size_t>(i)];
                if (token < 0)
                    return false;
                device_draft_sample_tokens_[
                    static_cast<size_t>(first_draft_slot + i * slot_stride)] = token;
                if (out_tokens)
                    out_tokens[i] = token;
            }
            return true;
        }

        bool forwardMTPBatchFromLastDraftAndSampleGreedy(
            const int32_t *draft_condition_tokens,
            const int *position_ids,
            int request_batch,
            int32_t *out_tokens) override
        {
            ++forward_mtp_batch_from_last_draft_and_sample_count_;
            last_chained_mtp_batch_condition_tokens_.clear();
            last_chained_mtp_batch_position_ids_.clear();
            if (!mtp_enabled_ ||
                !supports_chained_mtp_drafts_ ||
                !draft_condition_tokens ||
                !position_ids ||
                !out_tokens ||
                request_batch <= 0 ||
                request_batch > batch_capacity_)
            {
                return false;
            }

            last_chained_mtp_batch_condition_tokens_.assign(
                draft_condition_tokens,
                draft_condition_tokens + request_batch);
            last_chained_mtp_batch_position_ids_.assign(
                position_ids,
                position_ids + request_batch);
            for (int i = 0; i < request_batch; ++i)
            {
                if (position_ids[i] < 0)
                    return false;
                out_tokens[i] = MTP_ARGMAX_TOKEN;
            }
            mtp_logits_.assign(VOCAB_SIZE, -10.0f);
            mtp_logits_[MTP_ARGMAX_TOKEN] = 10.0f;
            return true;
        }

        bool forwardMTPBatchFromLastDraftAndSampleGreedyToDeviceDraftSlots(
            const int32_t *draft_condition_tokens,
            const int *position_ids,
            int request_batch,
            int first_draft_slot,
            int slot_stride,
            int32_t *out_tokens) override
        {
            ++forward_mtp_batch_from_last_draft_and_sample_to_device_draft_slots_count_;
            last_chained_mtp_batch_device_draft_first_slots_.push_back(first_draft_slot);
            last_chained_mtp_batch_device_draft_slot_strides_.push_back(slot_stride);
            const int last_slot =
                first_draft_slot +
                (request_batch > 0 ? (request_batch - 1) * slot_stride : 0);
            if (first_draft_slot < 0 ||
                slot_stride <= 0 ||
                last_slot >= static_cast<int>(device_draft_sample_tokens_.size()))
            {
                return false;
            }

            std::array<int32_t, 8> shadow{};
            shadow.fill(-1);
            if (!forwardMTPBatchFromLastDraftAndSampleGreedy(
                    draft_condition_tokens,
                    position_ids,
                    request_batch,
                    shadow.data()))
            {
                return false;
            }

            for (int i = 0; i < request_batch; ++i)
            {
                const int32_t token = shadow[static_cast<size_t>(i)];
                if (token < 0)
                    return false;
                device_draft_sample_tokens_[
                    static_cast<size_t>(first_draft_slot + i * slot_stride)] = token;
                if (out_tokens)
                    out_tokens[i] = token;
            }
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

        /**
         * @brief Mock the chained sidecar stream handoff entrypoint.
         *
         * This mirrors forwardMTPForDeviceSampling(), but starts from the hidden
         * row produced by the previous fake sidecar step.
         */
        bool forwardMTPFromLastDraftForDeviceSampling(
            int32_t draft_condition_token,
            int position_id) override
        {
            ++forward_mtp_from_last_draft_for_device_sampling_count_;
            return forwardMTPFromLastDraft(draft_condition_token, position_id);
        }

        bool forwardMTPFromDeviceDraftForDeviceSampling(
            int draft_sample_slot,
            int position_id) override
        {
            ++forward_mtp_from_device_draft_for_device_sampling_count_;
            if (!supports_mtp_device_draft_token_input_ ||
                draft_sample_slot < 0 ||
                draft_sample_slot >= static_cast<int>(device_draft_sample_tokens_.size()))
            {
                return false;
            }
            const int32_t token =
                device_draft_sample_tokens_[static_cast<size_t>(draft_sample_slot)];
            if (token < 0)
                return false;
            return forwardMTPFromLastDraft(token, position_id);
        }

        bool forwardMTPFromDeviceTargetForDeviceSampling(
            int target_sample_slot,
            int position_id) override
        {
            (void)position_id;
            ++forward_mtp_from_device_target_for_device_sampling_count_;
            if (!supports_mtp_device_draft_token_input_ ||
                target_sample_slot < 0 ||
                target_sample_slot >= static_cast<int>(device_target_sample_tokens_.size()))
            {
                return false;
            }
            const int32_t token =
                device_target_sample_tokens_[static_cast<size_t>(target_sample_slot)];
            if (token < 0)
                return false;
            return forwardMTP(token);
        }

        bool forwardMTPFromDeviceResidentLogicalStateForDeviceSampling(
            const DeviceResidentLogicalSequenceStateHandle &logical_state,
            int request_index = 0) override
        {
            ++forward_mtp_from_resident_logical_state_for_device_sampling_count_;
            if (!supports_mtp_device_draft_token_input_ ||
                !logical_state.coversRequest(request_index) ||
                logical_state.next_condition_tokens_device !=
                    resident_next_condition_tokens_.data() ||
                logical_state.all_drafts_accepted_flags_device !=
                    resident_all_drafts_accepted_flags_.data() ||
                logical_state.stopped_flags_device !=
                    resident_stopped_flags_.data() ||
                logical_state.target_positions_device !=
                    resident_target_positions_.data())
            {
                return false;
            }

            const int32_t token =
                logical_state.next_condition_tokens_device[request_index];
            if (token < 0)
                return false;
            last_resident_logical_state_sidecar_token_ = token;
            return forwardMTP(token);
        }

        bool flushPendingMTPWork() override
        {
            ++flush_pending_mtp_work_count_;
            return flush_pending_mtp_work_ok_;
        }

        void setMTPAllPositionVerifierSyncDeferralEnabled(bool enabled) override
        {
            ++all_position_verifier_sync_deferral_set_count_;
            all_position_verifier_sync_deferral_enabled_ = enabled;
            if (enabled)
                ++all_position_verifier_sync_deferral_enable_count_;
            else
                ++all_position_verifier_sync_deferral_disable_count_;
        }

        int sampleGreedyFromMTPLogitsOnDevice() override
        {
            ++sample_mtp_logits_count_;
            if (!supports_mtp_token_coordination_ || mtp_logits_.empty())
                return -1;
            return greedyArgmax(mtp_logits_.data(), VOCAB_SIZE);
        }

        bool sampleGreedyFromMTPLogitsToDeviceDraftSlot(
            int draft_sample_slot,
            int32_t *out_token) override
        {
            ++sample_mtp_logits_to_device_draft_slot_count_;
            last_sample_mtp_logits_device_draft_slot_ = draft_sample_slot;
            if (!supports_mtp_token_coordination_ ||
                !supports_mtp_device_draft_token_input_ ||
                mtp_logits_.empty() ||
                draft_sample_slot < 0 ||
                draft_sample_slot >=
                    static_cast<int>(device_draft_sample_tokens_.size()))
            {
                return false;
            }
            const int token = greedyArgmax(mtp_logits_.data(), VOCAB_SIZE);
            if (token < 0)
                return false;
            device_draft_sample_tokens_[static_cast<size_t>(draft_sample_slot)] =
                token;
            if (out_token)
                *out_token = token;
            return true;
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

        bool verifyGreedyAllPositionBatchOutcomeOnDevice(
            const int32_t *draft_tokens,
            int draft_token_count,
            const int32_t *stop_tokens,
            int stop_token_count,
            DeviceSpeculativeVerifyBatchOutcome *out) override
        {
            using namespace sampling_math;
            ++verify_greedy_all_position_batch_outcome_count_;
            if (!out || !draft_tokens || draft_token_count <= 0 ||
                draft_token_count > kSpeculativeBatchMaxOutputTokens ||
                stop_token_count < 0 ||
                stop_token_count > kSpeculativeBatchMaxStopTokens ||
                (stop_token_count > 0 && !stop_tokens))
            {
                return false;
            }

            /*
             * Production compares verifier row i against draft token i+1 and
             * treats the final verifier row as the already-ready bonus token.
             * The mock follows that shape so unit tests cover the compact
             * device outcome contract instead of the old row-by-row fallback.
             */
            std::array<int32_t, kSpeculativeBatchMaxOutputTokens> verify_tokens =
                {-1, -1, -1, -1, -1};
            if (!sampleGreedyFromAllPositionLogitsOnDeviceRows(
                    0, draft_token_count, verify_tokens.data()))
            {
                return false;
            }

            const int compare_rows = draft_token_count - 1;
            std::array<int, kSpeculativeBatchMaxRows> tokens =
                {-1, -1, -1, -1};
            std::array<int, kSpeculativeBatchMaxRows> accepted =
                {0, 0, 0, 0};
            for (int row = 0; row < compare_rows; ++row)
            {
                const int32_t expected_draft =
                    draft_tokens[static_cast<size_t>(row + 1)] >= 0
                        ? draft_tokens[static_cast<size_t>(row + 1)]
                        : device_verifier_input_tokens_[
                              static_cast<size_t>(row + 1)];
                tokens[static_cast<size_t>(row)] =
                    verify_tokens[static_cast<size_t>(row)];
                accepted[static_cast<size_t>(row)] =
                    verify_tokens[static_cast<size_t>(row)] == expected_draft
                        ? 1
                        : 0;
            }

            std::array<int, kSpeculativeBatchMaxStopTokens> packed_stop_tokens =
                {-1, -1, -1, -1, -1, -1, -1, -1};
            for (int i = 0; i < stop_token_count; ++i)
                packed_stop_tokens[static_cast<size_t>(i)] = stop_tokens[i];

            std::array<int, kSpeculativeBatchMaxOutputTokens> output_tokens =
                {-1, -1, -1, -1, -1};
            std::array<int, kSpeculativeBatchMetaCount> meta = {};
            const int ready_token =
                verify_tokens[static_cast<size_t>(compare_rows)];
            const int first_token =
                draft_tokens[0] >= 0
                    ? draft_tokens[0]
                    : device_verifier_input_tokens_[0];
            summarize_speculative_verify_batch(
                first_token,
                tokens.data(),
                accepted.data(),
                compare_rows,
                packed_stop_tokens.data(),
                stop_token_count,
                ready_token,
                1,
                output_tokens.data(),
                meta.data());
            if (meta[kSpecBatchMetaOk] == 0)
                return false;

            *out = DeviceSpeculativeVerifyBatchOutcome{};
            out->ok = true;
            for (size_t i = 0; i < out->output_tokens.size(); ++i)
                out->output_tokens[i] = output_tokens[i];
            out->output_token_count = meta[kSpecBatchMetaOutputCount];
            out->accepted_speculative_prefix =
                meta[kSpecBatchMetaAcceptedSpeculativePrefix];
            out->target_verifier_state_commit_count =
                meta[kSpecBatchMetaTargetVerifierStateCommitCount];
            out->ready_token = meta[kSpecBatchMetaReadyToken];
            out->rejected_verified_token =
                meta[kSpecBatchMetaRejectedVerifiedToken];
            out->stopped_on_output = meta[kSpecBatchMetaStoppedOnOutput] != 0;
            out->all_speculative_accepted =
                meta[kSpecBatchMetaAllSpeculativeAccepted] != 0;
            out->consumed_verifier_rows =
                meta[kSpecBatchMetaConsumedVerifierRows];
            out->sampled_terminal =
                meta[kSpecBatchMetaSampledTerminal] != 0;
            return true;
        }

        bool verifyGreedyAllPositionBatchOutcomeOnDeviceResident(
            const int32_t *draft_tokens,
            int draft_token_count,
            const int32_t *stop_tokens,
            int stop_token_count,
            DeviceSpeculativeOutcomeHandle *out_handle) override
        {
            using namespace sampling_math;
            if (!out_handle)
                return false;
            *out_handle = DeviceSpeculativeOutcomeHandle{};

            DeviceSpeculativeVerifyBatchOutcome outcome;
            if (!verifyGreedyAllPositionBatchOutcomeOnDevice(
                    draft_tokens,
                    draft_token_count,
                    stop_tokens,
                    stop_token_count,
                    &outcome))
            {
                return false;
            }

            resident_output_tokens_.fill(-1);
            resident_meta_.fill(0);
            for (int i = 0; i < outcome.output_token_count; ++i)
            {
                resident_output_tokens_[static_cast<size_t>(i)] =
                    outcome.output_tokens[static_cast<size_t>(i)];
            }
            resident_meta_[kSpecBatchMetaOk] = outcome.ok ? 1 : 0;
            resident_meta_[kSpecBatchMetaOutputCount] =
                outcome.output_token_count;
            resident_meta_[kSpecBatchMetaAcceptedSpeculativePrefix] =
                outcome.accepted_speculative_prefix;
            resident_meta_[kSpecBatchMetaTargetVerifierStateCommitCount] =
                outcome.target_verifier_state_commit_count;
            resident_meta_[kSpecBatchMetaReadyToken] = outcome.ready_token;
            resident_meta_[kSpecBatchMetaRejectedVerifiedToken] =
                outcome.rejected_verified_token;
            resident_meta_[kSpecBatchMetaStoppedOnOutput] =
                outcome.stopped_on_output ? 1 : 0;
            resident_meta_[kSpecBatchMetaAllSpeculativeAccepted] =
                outcome.all_speculative_accepted ? 1 : 0;
            resident_meta_[kSpecBatchMetaConsumedVerifierRows] =
                outcome.consumed_verifier_rows;
            resident_meta_[kSpecBatchMetaSampledTerminal] =
                outcome.sampled_terminal ? 1 : 0;

            out_handle->output_tokens_device = resident_output_tokens_.data();
            out_handle->meta_device = resident_meta_.data();
            out_handle->request_count = 1;
            out_handle->output_token_stride =
                kSpeculativeBatchMaxOutputTokens;
            out_handle->meta_stride = kSpeculativeBatchMetaCount;
            out_handle->device = primary_device_;
            out_handle->stream = &resident_stream_token_;
            out_handle->response_ready_event =
                std::shared_ptr<void>(
                    &resident_outcome_response_ready_event_token_,
                    [](void *) {});
            return out_handle->valid();
        }

        int vocab_size() const override { return VOCAB_SIZE; }

        void clear_cache() override
        {
            clear_cache_count_++;
            is_first_forward_in_cycle_ = true; // Reset cycle
            setupPrefillLogits();              // Reset logits state
            position_ = 0;
            mtp_shifted_cached_tokens_ = 0;
            sequence_lengths_.clear();
            batch_logits_.clear();
            padded_seq_len_ = 0;
            row_indexed_all_position_logits_enabled_ = false;
            row_indexed_all_position_logits_row_count_ = 0;
        }

        int get_position() const override { return position_; }
        int batch_size() const override { return batch_capacity_; }
        int padded_seq_len() const override { return padded_seq_len_; }
        const std::vector<int> &sequence_lengths() const override
        {
            return sequence_lengths_;
        }

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

        bool sampleGreedyFromMainLogitsToDeviceTargetSlot(
            int target_sample_slot,
            int32_t *out_token) override
        {
            ++sample_main_logits_to_device_target_slot_count_;
            last_sample_main_logits_device_target_slot_ = target_sample_slot;
            if (!supports_mtp_token_coordination_ ||
                !supports_mtp_device_draft_token_input_ ||
                target_sample_slot < 0 ||
                target_sample_slot >=
                    static_cast<int>(device_target_sample_tokens_.size()))
            {
                return false;
            }
            const int token = greedyArgmax(logits_.data(), VOCAB_SIZE);
            if (token < 0)
                return false;
            device_target_sample_tokens_[static_cast<size_t>(target_sample_slot)] =
                token;
            if (out_token)
                *out_token = token;
            return true;
        }

        int sampleOnDevice(const SamplingParams &params) override
        {
            ++sample_device_count_;
            if (!supports_stochastic_device_sampling_ || params.is_greedy())
                return -1;
            return greedyArgmax(logits_.data(), VOCAB_SIZE);
        }

        bool sampleMainLogitsBatchRowsOnDevice(
            int request_count,
            const SamplingParams &params,
            int32_t *out_tokens,
            const float *stochastic_thresholds = nullptr) override
        {
            ++sample_main_logits_batch_rows_count_;
            last_main_logits_batch_sampling_params_ = params;
            last_main_logits_batch_request_count_ = request_count;
            last_main_logits_batch_thresholds_.clear();
            if (stochastic_thresholds && request_count > 0)
            {
                last_main_logits_batch_thresholds_.assign(
                    stochastic_thresholds,
                    stochastic_thresholds + request_count);
            }
            if (!supports_main_logits_batch_rows_on_device_ ||
                !primary_device_.is_gpu() ||
                request_count <= 0 ||
                !out_tokens ||
                padded_seq_len_ <= 0 ||
                static_cast<int>(sequence_lengths_.size()) < request_count ||
                batch_logits_.empty())
            {
                return false;
            }

            for (int request = 0; request < request_count; ++request)
            {
                const int logical_length =
                    sequence_lengths_[static_cast<size_t>(request)];
                if (logical_length <= 0 || logical_length > padded_seq_len_)
                    return false;
                const size_t offset =
                    (static_cast<size_t>(request) *
                         static_cast<size_t>(padded_seq_len_) +
                     static_cast<size_t>(logical_length - 1)) *
                    static_cast<size_t>(VOCAB_SIZE);
                if (offset + VOCAB_SIZE > batch_logits_.size())
                    return false;
                out_tokens[request] = static_cast<int32_t>(
                    greedyArgmax(batch_logits_.data() + offset, VOCAB_SIZE));
            }
            return true;
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

        bool buildStochasticDistributionsOnDevice(
            DeviceLogitsSource source,
            int first_row,
            DeviceDistributionBuffer buffer,
            int first_slot,
            int row_count,
            const SamplingParams &params,
            int vocab_size) override
        {
            if (row_count <= 0)
                return false;

            /*
             * The mock does not need a separate batched math kernel. It does
             * need to preserve production semantics: contiguous rows map to
             * contiguous compact distribution slots, and failures abort the
             * whole batch.
             */
            for (int row = 0; row < row_count; ++row)
            {
                if (!buildStochasticDistributionOnDevice(
                        source,
                        first_row + row,
                        buffer,
                        first_slot + row,
                        params,
                        vocab_size))
                {
                    return false;
                }
            }
            return true;
        }

        bool buildStochasticProbabilityRowsOnDevice(
            DeviceLogitsSource source,
            int first_row,
            DeviceDistributionBuffer buffer,
            int first_slot,
            int row_count,
            const SamplingParams &params,
            int vocab_size) override
        {
            ++device_probability_rows_build_count_;
            if (row_count <= 0)
                return false;

            for (int row = 0; row < row_count; ++row)
            {
                const int source_row = first_row + row;
                const int slot = first_slot + row;
                const float *row_logits = nullptr;
                switch (source)
                {
                case DeviceLogitsSource::Main:
                    if (source_row != 0 || logits_.size() != VOCAB_SIZE)
                        return false;
                    row_logits = logits_.data();
                    break;
                case DeviceLogitsSource::MTP:
                    if (source_row != 0 || mtp_logits_.size() != VOCAB_SIZE)
                        return false;
                    row_logits = mtp_logits_.data();
                    break;
                case DeviceLogitsSource::AllPosition:
                {
                    const size_t offset =
                        static_cast<size_t>(source_row) * VOCAB_SIZE;
                    if (offset + VOCAB_SIZE > all_position_logits_.size())
                        return false;
                    row_logits = all_position_logits_.data() + offset;
                    break;
                }
                }

                if (!row_logits || vocab_size != VOCAB_SIZE || params.top_k <= 0)
                    return false;

                SamplingParams distribution_params = params;
                distribution_params.presence_penalty = 0.0f;
                distribution_params.frequency_penalty = 0.0f;
                distribution_params.dry_multiplier = 0.0f;
                distribution_params.dry_penalty_last_n = 0;
                Sampler distribution_sampler(params.seed);
                auto &target = deviceDistribution(buffer, slot);
                target = distribution_sampler.compute_distribution(
                    row_logits,
                    VOCAB_SIZE,
                    distribution_params);
                if (target.empty())
                    return false;
            }
            return true;
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
            ++device_processed_rows_build_count_;
            if (row_count <= 0)
                return false;

            for (int row = 0; row < row_count; ++row)
            {
                const int source_row = first_row + row;
                const int slot = first_slot + row;
                const float *row_logits = nullptr;
                switch (source)
                {
                case DeviceLogitsSource::Main:
                    if (source_row != 0 || logits_.size() != VOCAB_SIZE)
                        return false;
                    row_logits = logits_.data();
                    break;
                case DeviceLogitsSource::MTP:
                    if (source_row != 0 || mtp_logits_.size() != VOCAB_SIZE)
                        return false;
                    row_logits = mtp_logits_.data();
                    break;
                case DeviceLogitsSource::AllPosition:
                {
                    const size_t offset =
                        static_cast<size_t>(source_row) * VOCAB_SIZE;
                    if (offset + VOCAB_SIZE > all_position_logits_.size())
                        return false;
                    row_logits = all_position_logits_.data() + offset;
                    break;
                }
                }

                if (!row_logits || vocab_size != VOCAB_SIZE || params.top_k <= 0)
                    return false;

                SamplingParams distribution_params = params;
                distribution_params.presence_penalty = 0.0f;
                distribution_params.frequency_penalty = 0.0f;
                distribution_params.dry_multiplier = 0.0f;
                distribution_params.dry_penalty_last_n = 0;
                Sampler distribution_sampler(params.seed);
                auto &target = deviceDistribution(buffer, slot);
                target = distribution_sampler.compute_distribution(
                    row_logits,
                    VOCAB_SIZE,
                    distribution_params);
                if (target.empty())
                    return false;
            }
            return true;
        }

        int sampleStochasticDraftProposalOnDevice(
            DeviceLogitsSource source,
            int row,
            int slot,
            const SamplingParams &params,
            int vocab_size,
            float threshold) override
        {
            ++device_draft_temperature_proposal_count_;
            if (!buildTemperatureOnlyDraftProposal(source, row, slot, params, vocab_size))
                return -1;
            const auto &distribution =
                deviceDistribution(DeviceDistributionBuffer::Draft, slot);
            const int token = sampleWithThreshold(distribution, threshold);
            if (token >= 0 &&
                slot >= 0 &&
                slot < static_cast<int>(device_draft_sample_tokens_.size()))
            {
                device_draft_sample_tokens_[static_cast<size_t>(slot)] = token;
            }
            return token;
        }

        bool sampleStochasticDraftProposalOnDeviceDeferred(
            DeviceLogitsSource source,
            int row,
            int slot,
            const SamplingParams &params,
            int vocab_size,
            float threshold) override
        {
            ++device_draft_temperature_proposal_deferred_count_;
            const int token =
                sampleStochasticDraftProposalOnDevice(
                    source,
                    row,
                    slot,
                    params,
                    vocab_size,
                    threshold);
            return token >= 0;
        }

        bool stageStochasticDraftTokensForDeviceVerification(
            const int32_t *draft_tokens,
            int draft_token_count,
            int first_draft_slot = 0) override
        {
            ++stage_stochastic_draft_tokens_count_;
            last_staged_stochastic_draft_tokens_.clear();
            last_staged_stochastic_draft_first_slots_.push_back(first_draft_slot);
            if (!supports_stochastic_device_sampling_ ||
                !draft_tokens ||
                first_draft_slot < 0 ||
                draft_token_count <= 0 ||
                first_draft_slot + draft_token_count >
                    static_cast<int>(device_draft_sample_tokens_.size()))
            {
                return false;
            }

            for (int slot = 0; slot < draft_token_count; ++slot)
            {
                const int32_t token = draft_tokens[slot];
                if (token < 0)
                    return false;
                device_draft_sample_tokens_[
                    static_cast<size_t>(first_draft_slot + slot)] = token;
                last_staged_stochastic_draft_tokens_.push_back(token);
            }
            return true;
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
            const int token = sampleWithThreshold(distribution, threshold);
            if (buffer == DeviceDistributionBuffer::Draft &&
                slot >= 0 &&
                slot < static_cast<int>(device_draft_sample_tokens_.size()))
            {
                device_draft_sample_tokens_[static_cast<size_t>(slot)] = token;
            }
            if (buffer == DeviceDistributionBuffer::Target &&
                slot >= 0 &&
                slot < static_cast<int>(device_target_sample_tokens_.size()))
            {
                device_target_sample_tokens_[static_cast<size_t>(slot)] = token;
            }
            return token;
        }

        bool sampleStochasticDistributionOnDeviceDeferred(
            DeviceDistributionBuffer buffer,
            int slot,
            float threshold) override
        {
            ++device_distribution_sample_deferred_count_;
            if (!supports_stochastic_device_sampling_)
                return false;
            const auto &distribution = deviceDistribution(buffer, slot);
            const int token = sampleWithThreshold(distribution, threshold);
            if (token < 0)
                return false;
            if (buffer == DeviceDistributionBuffer::Draft &&
                slot >= 0 &&
                slot < static_cast<int>(device_draft_sample_tokens_.size()))
            {
                device_draft_sample_tokens_[static_cast<size_t>(slot)] = token;
            }
            if (buffer == DeviceDistributionBuffer::Target &&
                slot >= 0 &&
                slot < static_cast<int>(device_target_sample_tokens_.size()))
            {
                device_target_sample_tokens_[static_cast<size_t>(slot)] = token;
            }
            return true;
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

        bool verifyStochasticProbabilityRowOnDevice(
            int target_slot,
            int draft_slot,
            int draft_token,
            float accept_threshold,
            uint64_t inverse_sample_seed,
            int inverse_sample_logical_position,
            DeviceSpeculativeVerifyResult *out) override
        {
            ++device_probability_verify_row_count_;
            last_probability_row_inverse_sample_seed_ = inverse_sample_seed;
            last_probability_row_inverse_sample_logical_position_ =
                inverse_sample_logical_position;
            if (!supports_stochastic_device_sampling_ || !out ||
                target_slot < 0 || draft_slot < 0 || draft_token < 0)
            {
                return false;
            }

            const auto &target =
                deviceDistribution(DeviceDistributionBuffer::Target, target_slot);
            const auto &draft =
                deviceDistribution(DeviceDistributionBuffer::Draft, draft_slot);
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
                             : sampleResidualWithThreshold(target, draft, 0.5f);
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
                !accept_thresholds || !residual_thresholds || !out ||
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

                int32_t draft_token =
                    draft_tokens ? draft_tokens[row] : -1;
                if (draft_token < 0)
                {
                    const int device_slot = first_draft_slot + row;
                    if (device_slot < 0 ||
                        device_slot >= static_cast<int>(device_draft_sample_tokens_.size()))
                    {
                        return false;
                    }
                    draft_token =
                        device_draft_sample_tokens_[static_cast<size_t>(device_slot)];
                }
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
            using namespace sampling_math;
            batch_outcome_used_host_draft_tokens_ = draft_tokens != nullptr;
            last_batch_outcome_first_target_slots_.push_back(first_target_slot);
            last_batch_outcome_first_draft_slots_.push_back(first_draft_slot);
            last_batch_outcome_bonus_target_slots_.push_back(bonus_target_slot);
            last_batch_outcome_inverse_sample_seed_ = inverse_sample_seed;
            last_batch_outcome_inverse_sample_first_logical_position_ =
                inverse_sample_first_logical_position;
            last_batch_outcome_used_vllm_probability_rejection_ =
                use_vllm_probability_rejection;
            if (!out ||
                row_count <= 0 ||
                row_count > kSpeculativeBatchMaxRows ||
                stop_token_count < 0 ||
                stop_token_count > kSpeculativeBatchMaxStopTokens ||
                (stop_token_count > 0 && !stop_tokens))
            {
                return false;
            }
            *out = DeviceSpeculativeVerifyBatchOutcome{};

            std::array<DeviceSpeculativeVerifyResult, kSpeculativeBatchMaxRows> rows;
            if (use_vllm_probability_rejection)
            {
                ++device_distribution_verify_batch_count_;
                for (int row = 0; row < row_count; ++row)
                {
                    const int target_slot = first_target_slot + row;
                    const auto &target =
                        deviceDistribution(DeviceDistributionBuffer::Target,
                                           target_slot);
                    if (target.empty())
                        return false;

                    int32_t draft_token =
                        draft_tokens ? draft_tokens[row] : -1;
                    if (draft_token < 0)
                    {
                        const int device_slot = first_draft_slot + row;
                        if (device_slot < 0 ||
                            device_slot >= static_cast<int>(device_draft_sample_tokens_.size()))
                        {
                            return false;
                        }
                        draft_token =
                            device_draft_sample_tokens_[static_cast<size_t>(device_slot)];
                    }
                    if (draft_token < 0)
                        return false;

                    const float p = Sampler::probability_of_token(target, draft_token);
                    rows[static_cast<size_t>(row)].accepted =
                        accept_thresholds[row] < p;
                    rows[static_cast<size_t>(row)].accept_probability = p;
                    rows[static_cast<size_t>(row)].accept_threshold =
                        accept_thresholds[row];
                    rows[static_cast<size_t>(row)].token =
                        rows[static_cast<size_t>(row)].accepted
                            ? draft_token
                            : sampleResidualWithThreshold(
                                  target,
                                  {{draft_token, 1.0f}},
                                  residual_thresholds[row]);
                    if (rows[static_cast<size_t>(row)].token < 0)
                        return false;
                }
            }
            else if (!verifyStochasticDistributionsBatchOnDevice(
                         first_target_slot,
                         first_draft_slot,
                         draft_tokens,
                         accept_thresholds,
                         residual_thresholds,
                         row_count,
                         rows.data()))
            {
                return false;
            }

            std::array<int, kSpeculativeBatchMaxRows> tokens = {-1, -1, -1, -1};
            std::array<int, kSpeculativeBatchMaxRows> accepted = {0, 0, 0, 0};
            for (int row = 0; row < row_count; ++row)
            {
                tokens[static_cast<size_t>(row)] = rows[static_cast<size_t>(row)].token;
                accepted[static_cast<size_t>(row)] =
                    rows[static_cast<size_t>(row)].accepted ? 1 : 0;
            }

            std::array<int, kSpeculativeBatchMaxStopTokens> packed_stop_tokens =
                {-1, -1, -1, -1, -1, -1, -1, -1};
            for (int i = 0; i < stop_token_count; ++i)
                packed_stop_tokens[static_cast<size_t>(i)] = stop_tokens[i];

            const int32_t ready_token =
                bonus_target_slot >= 0
                    ? sampleStochasticDistributionOnDevice(
                          DeviceDistributionBuffer::Target,
                          bonus_target_slot,
                          bonus_threshold)
                    : -1;

            std::array<int, kSpeculativeBatchMaxOutputTokens> output_tokens =
                {-1, -1, -1, -1, -1};
            std::array<int, kSpeculativeBatchMetaCount> meta = {};
            summarize_speculative_verify_batch(
                first_token,
                tokens.data(),
                accepted.data(),
                row_count,
                packed_stop_tokens.data(),
                stop_token_count,
                ready_token,
                bonus_target_slot >= 0 ? 1 : 0,
                output_tokens.data(),
                meta.data());

            if (meta[kSpecBatchMetaOk] == 0)
                return false;

            out->ok = true;
            for (size_t i = 0; i < out->output_tokens.size(); ++i)
                out->output_tokens[i] = output_tokens[i];
            out->output_token_count = meta[kSpecBatchMetaOutputCount];
            out->accepted_speculative_prefix =
                meta[kSpecBatchMetaAcceptedSpeculativePrefix];
            out->target_verifier_state_commit_count =
                meta[kSpecBatchMetaTargetVerifierStateCommitCount];
            out->ready_token = meta[kSpecBatchMetaReadyToken];
            out->rejected_verified_token = meta[kSpecBatchMetaRejectedVerifiedToken];
            out->stopped_on_output = meta[kSpecBatchMetaStoppedOnOutput] != 0;
            out->all_speculative_accepted =
                meta[kSpecBatchMetaAllSpeculativeAccepted] != 0;
            out->consumed_verifier_rows = meta[kSpecBatchMetaConsumedVerifierRows];
            out->sampled_terminal = meta[kSpecBatchMetaSampledTerminal] != 0;
            return true;
        }

        bool verifyStochasticDistributionsBatchOutcomeOnDeviceFirstToken(
            int first_target_slot,
            int first_draft_slot,
            const int32_t *draft_tokens,
            const float *accept_thresholds,
            const float *residual_thresholds,
            int row_count,
            int first_target_sample_slot,
            const int32_t *stop_tokens,
            int stop_token_count,
            int bonus_target_slot,
            float bonus_threshold,
            DeviceSpeculativeVerifyBatchOutcome *out,
            uint64_t inverse_sample_seed = 0,
            int inverse_sample_first_logical_position = 0,
            bool use_vllm_probability_rejection = false) override
        {
            ++device_distribution_batch_outcome_device_first_count_;
            if (first_target_sample_slot < 0 ||
                first_target_sample_slot >=
                    static_cast<int>(device_target_sample_tokens_.size()))
            {
                return false;
            }
            const int32_t first_token =
                device_target_sample_tokens_[
                    static_cast<size_t>(first_target_sample_slot)];
            if (first_token < 0)
                return false;
            return verifyStochasticDistributionsBatchOutcomeOnDevice(
                first_target_slot,
                first_draft_slot,
                draft_tokens,
                accept_thresholds,
                residual_thresholds,
                row_count,
                first_token,
                stop_tokens,
                stop_token_count,
                bonus_target_slot,
                bonus_threshold,
                out,
                inverse_sample_seed,
                inverse_sample_first_logical_position,
                use_vllm_probability_rejection);
        }

        bool verifyStochasticDistributionsRequestBatchOutcomesOnDevice(
            const DeviceStochasticBatchOutcomeRequest *requests,
            int request_count,
            DeviceSpeculativeVerifyBatchOutcome *outcomes) override
        {
            ++device_distribution_request_batch_outcome_count_;
            last_request_batch_outcome_request_ids_.clear();
            last_request_batch_outcome_row_counts_.clear();
            last_request_batch_outcome_first_target_slots_.clear();
            last_request_batch_outcome_first_draft_slots_.clear();
            last_request_batch_outcome_bonus_target_slots_.clear();
            last_request_batch_outcome_first_tokens_.clear();
            last_request_batch_outcome_draft_tokens_.clear();
            last_request_batch_outcome_accept_thresholds_.clear();
            last_request_batch_outcome_residual_thresholds_.clear();
            last_request_batch_outcome_bonus_thresholds_.clear();
            last_request_batch_outcome_inverse_sample_seeds_.clear();
            last_request_batch_outcome_inverse_sample_first_positions_.clear();
            last_request_batch_outcome_derived_thresholds_.clear();
            if (!requests || request_count <= 0 || !outcomes)
                return false;

            for (int i = 0; i < request_count; ++i)
            {
                const DeviceStochasticBatchOutcomeRequest &request = requests[i];
                last_request_batch_outcome_request_ids_.push_back(
                    request.request_id);
                last_request_batch_outcome_row_counts_.push_back(
                    request.row_count);
                last_request_batch_outcome_first_target_slots_.push_back(
                    request.first_target_slot);
                last_request_batch_outcome_first_draft_slots_.push_back(
                    request.first_draft_slot);
                last_request_batch_outcome_bonus_target_slots_.push_back(
                    request.bonus_target_slot);
                last_request_batch_outcome_first_tokens_.push_back(
                    request.first_token);
                last_request_batch_outcome_bonus_thresholds_.push_back(
                    request.bonus_threshold);
                last_request_batch_outcome_inverse_sample_seeds_.push_back(
                    request.inverse_sample_seed);
                last_request_batch_outcome_inverse_sample_first_positions_.push_back(
                    request.inverse_sample_first_logical_position);
                last_request_batch_outcome_derived_thresholds_.push_back(
                    request.derive_thresholds_from_seed);
                std::vector<int32_t> draft_tokens;
                std::vector<float> accept_thresholds;
                std::vector<float> residual_thresholds;
                draft_tokens.reserve(static_cast<size_t>(request.row_count));
                accept_thresholds.reserve(static_cast<size_t>(request.row_count));
                residual_thresholds.reserve(static_cast<size_t>(request.row_count));
                for (int row = 0; row < request.row_count; ++row)
                {
                    draft_tokens.push_back(
                        request.draft_tokens[static_cast<size_t>(row)]);
                    if (request.derive_thresholds_from_seed)
                    {
                        const int logical_position =
                            request.inverse_sample_first_logical_position + row;
                        /*
                         * Seeded vLLM-style verification derives these
                         * thresholds in the backend kernel.  The mock records
                         * the derived values so existing logical-position
                         * assertions keep testing the same RNG contract without
                         * reintroducing host arrays into the production path.
                         */
                        accept_thresholds.push_back(
                            mtpSeededVerifierThreshold(
                                request.inverse_sample_seed,
                                logical_position,
                                1 /* MTPSpecStochasticDrawPurpose::Accept */));
                        residual_thresholds.push_back(
                            mtpSeededVerifierThreshold(
                                request.inverse_sample_seed,
                                logical_position,
                                2 /* MTPSpecStochasticDrawPurpose::Residual */));
                    }
                    else
                    {
                        accept_thresholds.push_back(
                            request.accept_thresholds[static_cast<size_t>(row)]);
                        residual_thresholds.push_back(
                            request.residual_thresholds[static_cast<size_t>(row)]);
                    }
                }
                last_request_batch_outcome_draft_tokens_.push_back(
                    std::move(draft_tokens));
                last_request_batch_outcome_accept_thresholds_.push_back(
                    std::move(accept_thresholds));
                last_request_batch_outcome_residual_thresholds_.push_back(
                    std::move(residual_thresholds));
            }

            if (!IInferenceRunner::
                    verifyStochasticDistributionsRequestBatchOutcomesOnDevice(
                        requests,
                        request_count,
                        outcomes))
            {
                return false;
            }

            for (int i = 0; i < request_count; ++i)
            {
                const std::optional<int> forced_rejection_token =
                    forcedRequestBatchRejectionToken(requests[i].request_id);
                if (!forced_rejection_token.has_value())
                    continue;

                std::array<int, sampling_math::kSpeculativeBatchMaxRows>
                    row_tokens = {-1, -1, -1, -1};
                std::array<int, sampling_math::kSpeculativeBatchMaxRows>
                    row_accepted = {0, 0, 0, 0};
                std::array<int, sampling_math::kSpeculativeBatchMaxOutputTokens>
                    output_tokens = {-1, -1, -1, -1, -1};
                std::array<int, sampling_math::kSpeculativeBatchMetaCount> meta = {};
                row_tokens[0] = *forced_rejection_token;

                sampling_math::summarize_speculative_verify_batch(
                    requests[i].first_token,
                    row_tokens.data(),
                    row_accepted.data(),
                    requests[i].row_count,
                    requests[i].stop_token_count > 0
                        ? requests[i].stop_tokens.data()
                        : nullptr,
                    requests[i].stop_token_count,
                    /*bonus_ready_token=*/-1,
                    /*has_bonus_ready_token=*/0,
                    output_tokens.data(),
                    meta.data());
                if (meta[sampling_math::kSpecBatchMetaOk] == 0)
                    return false;

                DeviceSpeculativeVerifyBatchOutcome forced;
                forced.ok = true;
                for (size_t token_index = 0;
                     token_index < forced.output_tokens.size();
                     ++token_index)
                {
                    forced.output_tokens[token_index] =
                        output_tokens[token_index];
                }
                forced.output_token_count =
                    meta[sampling_math::kSpecBatchMetaOutputCount];
                forced.accepted_speculative_prefix =
                    meta[sampling_math::kSpecBatchMetaAcceptedSpeculativePrefix];
                forced.target_verifier_state_commit_count =
                    meta[sampling_math::kSpecBatchMetaTargetVerifierStateCommitCount];
                forced.ready_token =
                    meta[sampling_math::kSpecBatchMetaReadyToken];
                forced.rejected_verified_token =
                    meta[sampling_math::kSpecBatchMetaRejectedVerifiedToken];
                forced.stopped_on_output =
                    meta[sampling_math::kSpecBatchMetaStoppedOnOutput] != 0;
                forced.all_speculative_accepted =
                    meta[sampling_math::kSpecBatchMetaAllSpeculativeAccepted] != 0;
                forced.consumed_verifier_rows =
                    meta[sampling_math::kSpecBatchMetaConsumedVerifierRows];
                forced.sampled_terminal =
                    meta[sampling_math::kSpecBatchMetaSampledTerminal] != 0;
                outcomes[i] = forced;
            }

            return true;
        }

        bool verifyStochasticDistributionsRequestBatchOutcomesOnDeviceResident(
            const DeviceStochasticBatchOutcomeRequest *requests,
            int request_count,
            DeviceSpeculativeOutcomeHandle *out_handle) override
        {
            ++device_distribution_request_batch_outcome_count_;
            last_request_batch_outcome_request_ids_.clear();
            last_request_batch_outcome_row_counts_.clear();
            last_request_batch_outcome_first_target_slots_.clear();
            last_request_batch_outcome_first_draft_slots_.clear();
            last_request_batch_outcome_bonus_target_slots_.clear();
            last_request_batch_outcome_first_tokens_.clear();
            last_request_batch_outcome_draft_tokens_.clear();
            last_request_batch_outcome_accept_thresholds_.clear();
            last_request_batch_outcome_residual_thresholds_.clear();
            last_request_batch_outcome_bonus_thresholds_.clear();
            last_request_batch_outcome_inverse_sample_seeds_.clear();
            last_request_batch_outcome_inverse_sample_first_positions_.clear();
            last_request_batch_outcome_derived_thresholds_.clear();
            if (!requests || request_count <= 0 || !out_handle)
                return false;
            if (request_count > kMockResidentOutcomeRequestCapacity)
                return false;

            for (int i = 0; i < request_count; ++i)
            {
                const DeviceStochasticBatchOutcomeRequest &request = requests[i];
                last_request_batch_outcome_request_ids_.push_back(
                    request.request_id);
                last_request_batch_outcome_row_counts_.push_back(
                    request.row_count);
                last_request_batch_outcome_first_target_slots_.push_back(
                    request.first_target_slot);
                last_request_batch_outcome_first_draft_slots_.push_back(
                    request.first_draft_slot);
                last_request_batch_outcome_bonus_target_slots_.push_back(
                    request.bonus_target_slot);
                last_request_batch_outcome_first_tokens_.push_back(
                    request.first_token);
                last_request_batch_outcome_bonus_thresholds_.push_back(
                    request.bonus_threshold);
                last_request_batch_outcome_inverse_sample_seeds_.push_back(
                    request.inverse_sample_seed);
                last_request_batch_outcome_inverse_sample_first_positions_.push_back(
                    request.inverse_sample_first_logical_position);
                last_request_batch_outcome_derived_thresholds_.push_back(
                    request.derive_thresholds_from_seed);

                std::vector<int32_t> draft_tokens;
                std::vector<float> accept_thresholds;
                std::vector<float> residual_thresholds;
                draft_tokens.reserve(static_cast<size_t>(request.row_count));
                accept_thresholds.reserve(static_cast<size_t>(request.row_count));
                residual_thresholds.reserve(static_cast<size_t>(request.row_count));
                for (int row = 0; row < request.row_count; ++row)
                {
                    draft_tokens.push_back(
                        request.draft_tokens[static_cast<size_t>(row)]);
                    if (request.derive_thresholds_from_seed)
                    {
                        const int logical_position =
                            request.inverse_sample_first_logical_position + row;
                        accept_thresholds.push_back(
                            mtpSeededVerifierThreshold(
                                request.inverse_sample_seed,
                                logical_position,
                                1 /* MTPSpecStochasticDrawPurpose::Accept */));
                        residual_thresholds.push_back(
                            mtpSeededVerifierThreshold(
                                request.inverse_sample_seed,
                                logical_position,
                                2 /* MTPSpecStochasticDrawPurpose::Residual */));
                    }
                    else
                    {
                        accept_thresholds.push_back(
                            request.accept_thresholds[static_cast<size_t>(row)]);
                        residual_thresholds.push_back(
                            request.residual_thresholds[static_cast<size_t>(row)]);
                    }
                }
                last_request_batch_outcome_draft_tokens_.push_back(
                    std::move(draft_tokens));
                last_request_batch_outcome_accept_thresholds_.push_back(
                    std::move(accept_thresholds));
                last_request_batch_outcome_residual_thresholds_.push_back(
                    std::move(residual_thresholds));
            }

            std::array<DeviceSpeculativeVerifyBatchOutcome,
                       kMockResidentOutcomeRequestCapacity>
                outcomes{};
            for (int i = 0; i < request_count; ++i)
            {
                const DeviceStochasticBatchOutcomeRequest &request = requests[i];
                const bool descriptor_only =
                    deviceDistribution(
                        DeviceDistributionBuffer::Target,
                        request.first_target_slot).empty();
                if (descriptor_only)
                {
                    /*
                     * Some focused descriptor tests only assert that the
                     * scalar wrapper builds the right request metadata.  They
                     * do not install fake target distributions.  Emit a tiny
                     * reject-first compact row so those tests can stay about
                     * the descriptor contract without inventing a bonus-ready
                     * terminal token. Decode-style tests still use the
                     * distribution-backed branch below.
                     */
                    std::array<int, sampling_math::kSpeculativeBatchMaxRows>
                        row_tokens = {-1, -1, -1, -1};
                    std::array<int, sampling_math::kSpeculativeBatchMaxRows>
                        row_accepted = {0, 0, 0, 0};
                    std::array<int, sampling_math::kSpeculativeBatchMaxOutputTokens>
                        output_tokens = {-1, -1, -1, -1, -1};
                    std::array<int, sampling_math::kSpeculativeBatchMetaCount>
                        meta = {};
                    for (int row = 0; row < request.row_count; ++row)
                    {
                        row_tokens[static_cast<size_t>(row)] =
                            request.draft_tokens[static_cast<size_t>(row)] >= 0
                                ? request.draft_tokens[static_cast<size_t>(row)]
                                : (100 + row);
                    }
                    const int32_t first_token =
                        request.first_token_from_device ? 99 : request.first_token;
                    sampling_math::summarize_speculative_verify_batch(
                        first_token,
                        row_tokens.data(),
                        row_accepted.data(),
                        request.row_count,
                        request.stop_token_count > 0
                            ? request.stop_tokens.data()
                            : nullptr,
                        request.stop_token_count,
                        /*bonus_ready_token=*/-1,
                        /*has_bonus_ready_token=*/0,
                        output_tokens.data(),
                        meta.data());
                    if (meta[sampling_math::kSpecBatchMetaOk] == 0)
                        return false;
                    DeviceSpeculativeVerifyBatchOutcome fallback;
                    fallback.ok = true;
                    for (size_t token_index = 0;
                         token_index < fallback.output_tokens.size();
                         ++token_index)
                    {
                        fallback.output_tokens[token_index] =
                            output_tokens[token_index];
                    }
                    fallback.output_token_count =
                        meta[sampling_math::kSpecBatchMetaOutputCount];
                    fallback.accepted_speculative_prefix =
                        meta[sampling_math::kSpecBatchMetaAcceptedSpeculativePrefix];
                    fallback.target_verifier_state_commit_count =
                        meta[sampling_math::kSpecBatchMetaTargetVerifierStateCommitCount];
                    fallback.ready_token =
                        meta[sampling_math::kSpecBatchMetaReadyToken];
                    fallback.rejected_verified_token =
                        meta[sampling_math::kSpecBatchMetaRejectedVerifiedToken];
                    fallback.stopped_on_output =
                        meta[sampling_math::kSpecBatchMetaStoppedOnOutput] != 0;
                    fallback.all_speculative_accepted =
                        meta[sampling_math::kSpecBatchMetaAllSpeculativeAccepted] != 0;
                    fallback.consumed_verifier_rows =
                        meta[sampling_math::kSpecBatchMetaConsumedVerifierRows];
                    fallback.sampled_terminal =
                        meta[sampling_math::kSpecBatchMetaSampledTerminal] != 0;
                    outcomes[static_cast<size_t>(i)] = fallback;
                }
                else
                {
                    const int32_t *stop_tokens =
                        request.stop_token_count > 0
                            ? request.stop_tokens.data()
                            : nullptr;
                    const int32_t *draft_tokens = request.hostDraftTokensOrNull();
                    std::array<float, sampling_math::kSpeculativeBatchMaxRows>
                        derived_accept_thresholds = {};
                    std::array<float, sampling_math::kSpeculativeBatchMaxRows>
                        derived_residual_thresholds = {};
                    const float *accept_thresholds =
                        request.accept_thresholds.data();
                    const float *residual_thresholds =
                        request.residual_thresholds.data();
                    if (request.derive_thresholds_from_seed)
                    {
                        for (int row = 0; row < request.row_count; ++row)
                        {
                            const int logical_position =
                                request.inverse_sample_first_logical_position +
                                row;
                            derived_accept_thresholds[static_cast<size_t>(row)] =
                                mtpSeededVerifierThreshold(
                                    request.inverse_sample_seed,
                                    logical_position,
                                    1 /* MTPSpecStochasticDrawPurpose::Accept */);
                            derived_residual_thresholds[static_cast<size_t>(row)] =
                                mtpSeededVerifierThreshold(
                                    request.inverse_sample_seed,
                                    logical_position,
                                    2 /* MTPSpecStochasticDrawPurpose::Residual */);
                        }
                        accept_thresholds = derived_accept_thresholds.data();
                        residual_thresholds =
                            derived_residual_thresholds.data();
                    }
                    bool ok = false;
                    if (request.first_token_from_device)
                    {
                        ok = verifyStochasticDistributionsBatchOutcomeOnDeviceFirstToken(
                            request.first_target_slot,
                            request.first_draft_slot,
                            draft_tokens,
                            accept_thresholds,
                            residual_thresholds,
                            request.row_count,
                            request.first_target_sample_slot,
                            stop_tokens,
                            request.stop_token_count,
                            request.bonus_target_slot,
                            request.bonus_threshold,
                            &outcomes[static_cast<size_t>(i)],
                            request.inverse_sample_seed,
                            request.inverse_sample_first_logical_position,
                            request.use_vllm_probability_rejection);
                    }
                    else
                    {
                        ok = verifyStochasticDistributionsBatchOutcomeOnDevice(
                            request.first_target_slot,
                            request.first_draft_slot,
                            draft_tokens,
                            accept_thresholds,
                            residual_thresholds,
                            request.row_count,
                            request.first_token,
                            stop_tokens,
                            request.stop_token_count,
                            request.bonus_target_slot,
                            request.bonus_threshold,
                            &outcomes[static_cast<size_t>(i)],
                            request.inverse_sample_seed,
                            request.inverse_sample_first_logical_position,
                            request.use_vllm_probability_rejection);
                    }
                    if (!ok)
                        return false;
                }

                const std::optional<int> forced_rejection_token =
                    forcedRequestBatchRejectionToken(request.request_id);
                if (forced_rejection_token.has_value())
                {
                    std::array<int, sampling_math::kSpeculativeBatchMaxRows>
                        row_tokens = {-1, -1, -1, -1};
                    std::array<int, sampling_math::kSpeculativeBatchMaxRows>
                        row_accepted = {0, 0, 0, 0};
                    std::array<int, sampling_math::kSpeculativeBatchMaxOutputTokens>
                        output_tokens = {-1, -1, -1, -1, -1};
                    std::array<int, sampling_math::kSpeculativeBatchMetaCount>
                        meta = {};
                    row_tokens[0] = *forced_rejection_token;

                    sampling_math::summarize_speculative_verify_batch(
                        request.first_token,
                        row_tokens.data(),
                        row_accepted.data(),
                        request.row_count,
                        request.stop_token_count > 0
                            ? request.stop_tokens.data()
                            : nullptr,
                        request.stop_token_count,
                        /*bonus_ready_token=*/-1,
                        /*has_bonus_ready_token=*/0,
                        output_tokens.data(),
                        meta.data());
                    if (meta[sampling_math::kSpecBatchMetaOk] == 0)
                        return false;

                    DeviceSpeculativeVerifyBatchOutcome forced;
                    forced.ok = true;
                    for (size_t token_index = 0;
                         token_index < forced.output_tokens.size();
                         ++token_index)
                    {
                        forced.output_tokens[token_index] =
                            output_tokens[token_index];
                    }
                    forced.output_token_count =
                        meta[sampling_math::kSpecBatchMetaOutputCount];
                    forced.accepted_speculative_prefix =
                        meta[sampling_math::kSpecBatchMetaAcceptedSpeculativePrefix];
                    forced.target_verifier_state_commit_count =
                        meta[sampling_math::kSpecBatchMetaTargetVerifierStateCommitCount];
                    forced.ready_token =
                        meta[sampling_math::kSpecBatchMetaReadyToken];
                    forced.rejected_verified_token =
                        meta[sampling_math::kSpecBatchMetaRejectedVerifiedToken];
                    forced.stopped_on_output =
                        meta[sampling_math::kSpecBatchMetaStoppedOnOutput] != 0;
                    forced.all_speculative_accepted =
                        meta[sampling_math::kSpecBatchMetaAllSpeculativeAccepted] != 0;
                    forced.consumed_verifier_rows =
                        meta[sampling_math::kSpecBatchMetaConsumedVerifierRows];
                    forced.sampled_terminal =
                        meta[sampling_math::kSpecBatchMetaSampledTerminal] != 0;
                    outcomes[static_cast<size_t>(i)] = forced;
                }

                writeResidentOutcomeRow(
                    i,
                    outcomes[static_cast<size_t>(i)]);
            }

            out_handle->output_tokens_device =
                resident_output_tokens_.data();
            out_handle->meta_device = resident_meta_.data();
            out_handle->request_count = request_count;
            out_handle->output_token_stride =
                sampling_math::kSpeculativeBatchMaxOutputTokens;
            out_handle->meta_stride =
                sampling_math::kSpeculativeBatchMetaCount;
            out_handle->device = primary_device_;
            out_handle->stream = &resident_stream_token_;
            out_handle->response_ready_event =
                std::shared_ptr<void>(
                    &resident_outcome_response_ready_event_token_,
                    [](void *) {});
            return out_handle->valid();
        }

        bool copyDeviceSpeculativeOutcomesToHost(
            const DeviceSpeculativeOutcomeHandle &handle,
            DeviceSpeculativeVerifyBatchOutcome *outcomes) override
        {
            using namespace sampling_math;
            resident_sidecar_count_at_last_host_bridge_ =
                forward_mtp_from_resident_logical_state_for_device_sampling_count_;
            publication_events_.push_back("host_outcome_bridge");
            if (!handle.valid() ||
                !outcomes ||
                handle.device != primary_device_ ||
                handle.output_token_stride != kSpeculativeBatchMaxOutputTokens ||
                handle.meta_stride != kSpeculativeBatchMetaCount ||
                handle.request_count <= 0 ||
                handle.request_count > kMockResidentOutcomeRequestCapacity)
            {
                return false;
            }

            const auto *output_tokens =
                static_cast<const int32_t *>(handle.output_tokens_device);
            const auto *meta = static_cast<const int *>(handle.meta_device);
            for (int request = 0; request < handle.request_count; ++request)
            {
                const size_t token_base =
                    static_cast<size_t>(request) *
                    static_cast<size_t>(handle.output_token_stride);
                const size_t meta_base =
                    static_cast<size_t>(request) *
                    static_cast<size_t>(handle.meta_stride);
                const int *row_meta = meta + meta_base;
                if (row_meta[kSpecBatchMetaOk] == 0)
                    return false;

                DeviceSpeculativeVerifyBatchOutcome &out = outcomes[request];
                out = DeviceSpeculativeVerifyBatchOutcome{};
                out.ok = true;
                for (int i = 0; i < row_meta[kSpecBatchMetaOutputCount]; ++i)
                {
                    out.output_tokens[static_cast<size_t>(i)] =
                        output_tokens[token_base + static_cast<size_t>(i)];
                }
                out.output_token_count = row_meta[kSpecBatchMetaOutputCount];
                out.accepted_speculative_prefix =
                    row_meta[kSpecBatchMetaAcceptedSpeculativePrefix];
                out.target_verifier_state_commit_count =
                    row_meta[kSpecBatchMetaTargetVerifierStateCommitCount];
                out.ready_token = row_meta[kSpecBatchMetaReadyToken];
                out.rejected_verified_token =
                    row_meta[kSpecBatchMetaRejectedVerifiedToken];
                out.stopped_on_output =
                    row_meta[kSpecBatchMetaStoppedOnOutput] != 0;
                out.all_speculative_accepted =
                    row_meta[kSpecBatchMetaAllSpeculativeAccepted] != 0;
                out.consumed_verifier_rows =
                    row_meta[kSpecBatchMetaConsumedVerifierRows];
                out.sampled_terminal =
                    row_meta[kSpecBatchMetaSampledTerminal] != 0;
            }
            return true;
        }

        // =====================================================================
        // Test inspection methods
        // =====================================================================

        int forwardCallCount() const { return forward_call_count_; }
        int forwardWithDeviceTokenIdsCount() const { return forward_with_device_token_ids_count_; }
        int prepareMTPVerifierInputTokensOnDeviceCount() const
        {
            return prepare_mtp_verifier_input_tokens_on_device_count_;
        }
        int lastPrepareMTPVerifierFirstToken() const { return last_prepare_mtp_verifier_first_token_; }
        int lastPrepareMTPVerifierFirstDraftSlot() const
        {
            return last_prepare_mtp_verifier_first_draft_slot_;
        }
        int lastPrepareMTPVerifierDraftTokenCount() const
        {
            return last_prepare_mtp_verifier_draft_token_count_;
        }
        int lastPrepareMTPVerifierTotalTokens() const
        {
            return last_prepare_mtp_verifier_total_tokens_;
        }
        const void *lastForwardDeviceTokenIds() const { return last_forward_device_token_ids_; }
        int lastForwardDeviceTokenSeqLen() const { return last_forward_device_token_seq_len_; }
        const std::array<int32_t, 8> &deviceVerifierInputTokens() const
        {
            return device_verifier_input_tokens_;
        }
        int clearCacheCount() const { return clear_cache_count_; }
        int forwardMTPCount() const { return forward_mtp_count_; }
        int forwardMTPFromLastDraftCount() const { return forward_mtp_from_last_draft_count_; }
        int forwardMTPForDeviceSamplingCount() const { return forward_mtp_for_device_sampling_count_; }
        int forwardMTPFromLastDraftForDeviceSamplingCount() const
        {
            return forward_mtp_from_last_draft_for_device_sampling_count_;
        }
        int forwardMTPFromDeviceDraftForDeviceSamplingCount() const
        {
            return forward_mtp_from_device_draft_for_device_sampling_count_;
        }
        int forwardMTPFromDeviceTargetForDeviceSamplingCount() const
        {
            return forward_mtp_from_device_target_for_device_sampling_count_;
        }
        int forwardMTPFromResidentLogicalStateForDeviceSamplingCount() const
        {
            return forward_mtp_from_resident_logical_state_for_device_sampling_count_;
        }
        int deviceTargetShiftedCommitCount() const
        {
            return device_target_shifted_commit_count_;
        }
        int residentLogicalStateShiftedCommitCount() const
        {
            return resident_logical_state_shifted_commit_count_;
        }
        int lastDeviceTargetShiftedCommitToken() const
        {
            return last_device_target_shifted_commit_token_;
        }
        int lastResidentLogicalStateShiftedCommitToken() const
        {
            return last_resident_logical_state_shifted_commit_token_;
        }
        int lastResidentLogicalStateSidecarToken() const
        {
            return last_resident_logical_state_sidecar_token_;
        }
        int residentAcceptedStateCount(int request_index) const
        {
            if (request_index < 0 ||
                request_index >= resident_logical_state_request_count_)
            {
                return -1;
            }
            return resident_accepted_state_counts_[static_cast<size_t>(request_index)];
        }
        int prepareMTPVerifierInputTokensDeviceFirstCount() const
        {
            return prepare_mtp_verifier_input_tokens_device_first_count_;
        }
        int verifyStochasticDistributionsBatchOutcomeDeviceFirstCount() const
        {
            return device_distribution_batch_outcome_device_first_count_;
        }
        int stageStochasticDraftTokensCount() const
        {
            return stage_stochastic_draft_tokens_count_;
        }
        const std::vector<int32_t> &lastStagedStochasticDraftTokens() const
        {
            return last_staged_stochastic_draft_tokens_;
        }
        const std::vector<int> &lastStagedStochasticDraftFirstSlots() const
        {
            return last_staged_stochastic_draft_first_slots_;
        }
        const std::vector<int> &lastBatchOutcomeFirstTargetSlots() const
        {
            return last_batch_outcome_first_target_slots_;
        }
        const std::vector<int> &lastBatchOutcomeFirstDraftSlots() const
        {
            return last_batch_outcome_first_draft_slots_;
        }
        const std::vector<int> &lastBatchOutcomeBonusTargetSlots() const
        {
            return last_batch_outcome_bonus_target_slots_;
        }
        int verifyStochasticRequestBatchOutcomeCount() const
        {
            return device_distribution_request_batch_outcome_count_;
        }
        const std::vector<int> &lastRequestBatchOutcomeRequestIds() const
        {
            return last_request_batch_outcome_request_ids_;
        }
        const std::vector<int> &lastRequestBatchOutcomeRowCounts() const
        {
            return last_request_batch_outcome_row_counts_;
        }
        const std::vector<int> &lastRequestBatchOutcomeFirstTargetSlots() const
        {
            return last_request_batch_outcome_first_target_slots_;
        }
        const std::vector<int> &lastRequestBatchOutcomeFirstDraftSlots() const
        {
            return last_request_batch_outcome_first_draft_slots_;
        }
        const std::vector<int> &lastRequestBatchOutcomeBonusTargetSlots() const
        {
            return last_request_batch_outcome_bonus_target_slots_;
        }
        const std::vector<int32_t> &lastRequestBatchOutcomeFirstTokens() const
        {
            return last_request_batch_outcome_first_tokens_;
        }
        const std::vector<std::vector<int32_t>> &lastRequestBatchOutcomeDraftTokens() const
        {
            return last_request_batch_outcome_draft_tokens_;
        }
        const std::vector<std::vector<float>> &lastRequestBatchOutcomeAcceptThresholds() const
        {
            return last_request_batch_outcome_accept_thresholds_;
        }
        const std::vector<std::vector<float>> &lastRequestBatchOutcomeResidualThresholds() const
        {
            return last_request_batch_outcome_residual_thresholds_;
        }
        const std::vector<float> &lastRequestBatchOutcomeBonusThresholds() const
        {
            return last_request_batch_outcome_bonus_thresholds_;
        }
        const std::vector<uint64_t> &lastRequestBatchOutcomeInverseSampleSeeds() const
        {
            return last_request_batch_outcome_inverse_sample_seeds_;
        }
        const std::vector<int> &lastRequestBatchOutcomeInverseSampleFirstPositions() const
        {
            return last_request_batch_outcome_inverse_sample_first_positions_;
        }
        const std::vector<bool> &lastRequestBatchOutcomeDerivedThresholds() const
        {
            return last_request_batch_outcome_derived_thresholds_;
        }
        int forwardMTPAndSampleCount() const { return forward_mtp_and_sample_count_; }
        int forwardMTPBatchAndSampleCount() const
        {
            return forward_mtp_batch_and_sample_count_;
        }
        int forwardMTPBatchAndSampleToDeviceDraftSlotsCount() const
        {
            return forward_mtp_batch_and_sample_to_device_draft_slots_count_;
        }
        int lastMTPBatchDeviceDraftFirstSlot() const
        {
            return last_mtp_batch_device_draft_first_slot_;
        }
        int lastMTPBatchDeviceDraftSlotStride() const
        {
            return last_mtp_batch_device_draft_slot_stride_;
        }
        int forwardMTPBatchFromLastDraftAndSampleCount() const
        {
            return forward_mtp_batch_from_last_draft_and_sample_count_;
        }
        int forwardMTPBatchFromLastDraftAndSampleToDeviceDraftSlotsCount() const
        {
            return forward_mtp_batch_from_last_draft_and_sample_to_device_draft_slots_count_;
        }
        const std::vector<int> &lastChainedMTPBatchDeviceDraftFirstSlots() const
        {
            return last_chained_mtp_batch_device_draft_first_slots_;
        }
        const std::vector<int> &lastChainedMTPBatchDeviceDraftSlotStrides() const
        {
            return last_chained_mtp_batch_device_draft_slot_strides_;
        }
        int forwardMTPFromLastDraftAndSampleCount() const { return forward_mtp_from_last_draft_and_sample_count_; }
        int flushPendingMTPWorkCount() const { return flush_pending_mtp_work_count_; }
        int restoreCount() const { return restore_count_; }
        int captureCheckpointCount() const { return capture_checkpoint_count_; }
        int setAllPositionCount() const { return set_all_position_count_; }
        int setRowIndexedAllPositionCount() const { return set_row_indexed_all_position_count_; }
        int setMTPSpecVerifierPlanCount() const { return set_mtp_spec_verifier_plan_count_; }
        int clearMTPSpecVerifierPlanCount() const { return clear_mtp_spec_verifier_plan_count_; }
        bool mtpSpecVerifierPlanInstalled() const { return mtp_spec_verifier_plan_installed_; }
        const std::vector<int32_t> &lastMTPSpecVerifierRows() const { return last_mtp_spec_verifier_rows_; }
        const std::vector<int32_t> &lastMTPSpecVerifierTokens() const { return last_mtp_spec_verifier_tokens_; }
        const std::vector<int32_t> &lastMTPBatchConditionTokens() const
        {
            return last_mtp_batch_condition_tokens_;
        }
        const std::vector<int> &lastMTPBatchPositionIds() const
        {
            return last_mtp_batch_position_ids_;
        }
        const std::vector<int32_t> &lastChainedMTPBatchConditionTokens() const
        {
            return last_chained_mtp_batch_condition_tokens_;
        }
        const std::vector<int> &lastChainedMTPBatchPositionIds() const
        {
            return last_chained_mtp_batch_position_ids_;
        }
        int lastMTPConditionToken() const { return last_mtp_condition_token_; }
        int lastChainedMTPConditionToken() const { return last_chained_mtp_condition_token_; }
        int lastChainedMTPPositionId() const { return last_chained_mtp_position_id_; }
        int commitMTPShiftedCount() const { return commit_mtp_shifted_count_; }
        int lastCommitMTPAlreadyAppended() const { return last_commit_mtp_already_appended_; }
        int lastCommitMTPMainForwardTokenCount() const { return last_commit_mtp_main_forward_token_count_; }
        bool lastCommitMTPAllowSpeculativeDiscard() const { return last_commit_mtp_allow_speculative_discard_; }
        int lastCommitMTPPositionOffsetOverride() const { return last_commit_mtp_position_offset_override_; }
        int lastCommitMTPAlreadyAppendedShiftedKV() const
        {
            return last_commit_mtp_already_appended_shifted_kv_;
        }
        const std::vector<int> &lastCommitMTPTokens() const { return last_commit_mtp_tokens_; }
        int sequentialCommitMTPShiftedCount() const { return sequential_commit_mtp_shifted_count_; }
        int getLogitsCallCount() const { return get_logits_count_; }
        int sampleMainLogitsCount() const { return sample_main_logits_count_; }
        int sampleMainLogitsToDeviceTargetSlotCount() const
        {
            return sample_main_logits_to_device_target_slot_count_;
        }
        int lastSampleMainLogitsDeviceTargetSlot() const
        {
            return last_sample_main_logits_device_target_slot_;
        }
        int sampleDeviceCount() const { return sample_device_count_; }
        int sampleMainLogitsBatchRowsCount() const
        {
            return sample_main_logits_batch_rows_count_;
        }
        int lastMainLogitsBatchRequestCount() const
        {
            return last_main_logits_batch_request_count_;
        }
        const std::vector<float> &lastMainLogitsBatchThresholds() const
        {
            return last_main_logits_batch_thresholds_;
        }
        int sampleMTPLogitsCount() const { return sample_mtp_logits_count_; }
        int sampleMTPLogitsToDeviceDraftSlotCount() const
        {
            return sample_mtp_logits_to_device_draft_slot_count_;
        }
        int lastSampleMTPLogitsDeviceDraftSlot() const
        {
            return last_sample_mtp_logits_device_draft_slot_;
        }
        int sampleAllPositionLogitsCount() const { return sample_all_position_logits_count_; }
        int sampleAllPositionLogitsBatchedCount() const { return sample_all_position_logits_batched_count_; }
        int verifyGreedyAllPositionBatchOutcomeCount() const
        {
            return verify_greedy_all_position_batch_outcome_count_;
        }
        int applyMainPenaltiesCount() const { return apply_main_penalties_count_; }
        int applyMTPPenaltiesCount() const { return apply_mtp_penalties_count_; }
        int applyAllPositionPenaltiesCount() const { return apply_all_position_penalties_count_; }
        int deviceDistributionBuildCount() const { return device_distribution_build_count_; }
        int deviceProbabilityRowsBuildCount() const { return device_probability_rows_build_count_; }
        int deviceProcessedRowsBuildCount() const { return device_processed_rows_build_count_; }
        int deviceDistributionSampleCount() const { return device_distribution_sample_count_; }
        int deviceDistributionSampleDeferredCount() const
        {
            return device_distribution_sample_deferred_count_;
	        }
        int deviceDraftTemperatureProposalCount() const
        {
            return device_draft_temperature_proposal_count_;
        }
        int deviceDraftTemperatureProposalDeferredCount() const
        {
            return device_draft_temperature_proposal_deferred_count_;
        }
	        int deviceDistributionVerifyCount() const { return device_distribution_verify_count_; }
	        int deviceDistributionVerifyBatchCount() const { return device_distribution_verify_batch_count_; }
        int deviceProbabilityVerifyRowCount() const { return device_probability_verify_row_count_; }
        uint64_t lastProbabilityRowInverseSampleSeed() const
        {
            return last_probability_row_inverse_sample_seed_;
        }
        int lastProbabilityRowInverseSampleLogicalPosition() const
        {
            return last_probability_row_inverse_sample_logical_position_;
        }
	        bool batchOutcomeUsedHostDraftTokens() const
	        {
	            return batch_outcome_used_host_draft_tokens_;
        }
        bool lastBatchOutcomeUsedVLLMProbabilityRejection() const
        {
            return last_batch_outcome_used_vllm_probability_rejection_;
        }
        uint64_t lastBatchOutcomeInverseSampleSeed() const
        {
            return last_batch_outcome_inverse_sample_seed_;
        }
        int lastBatchOutcomeInverseSampleFirstLogicalPosition() const
        {
            return last_batch_outcome_inverse_sample_first_logical_position_;
        }
        int allPositionVerifierSyncDeferralSetCount() const { return all_position_verifier_sync_deferral_set_count_; }
        int allPositionVerifierSyncDeferralEnableCount() const { return all_position_verifier_sync_deferral_enable_count_; }
        int allPositionVerifierSyncDeferralDisableCount() const { return all_position_verifier_sync_deferral_disable_count_; }
        bool allPositionVerifierSyncDeferralEnabled() const { return all_position_verifier_sync_deferral_enabled_; }
        int lastSampleAllPositionStartRow() const { return last_sample_all_position_start_row_; }
        int lastSampleAllPositionRowCount() const { return last_sample_all_position_row_count_; }
        const PrefixStateSnapshot &lastRestoredSnapshot() const { return last_restored_snapshot_; }
        const std::vector<int> &lastForwardTokens() const { return last_forward_tokens_; }
        const std::vector<std::vector<int>> &forwardHistory() const { return forward_history_; }
        int lastForwardSeqLen() const { return last_forward_seq_len_; }
        int forwardBatchCallCount() const { return forward_batch_call_count_; }
        const std::vector<std::vector<int>> &lastForwardBatch() const
        {
            return last_forward_batch_;
        }
        void setBatchCapacity(int capacity)
        {
            batch_capacity_ = capacity;
        }
        void enableMTP(bool accept_mtp_token)
        {
            mtp_enabled_ = true;
            accept_mtp_token_ = accept_mtp_token;
            mtp_shifted_cached_tokens_ = shiftedTargetForMainTokens(position_);
        }
        void setMTPDraftTokens(int draft_tokens)
        {
            mtp_draft_tokens_ = std::max(1, draft_tokens);
            mtp_shifted_cached_tokens_ = shiftedTargetForMainTokens(position_);
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
        void enableMTPSidecarLogitsStreamHandoff()
        {
            supports_mtp_sidecar_stream_handoff_ = true;
        }
        void enableMTPDeviceDraftTokenInput()
        {
            supports_mtp_device_draft_token_input_ = true;
        }
        void enableMTPSidecarPreservesMainState()
        {
            supports_mtp_sidecar_preserves_main_state_ = true;
        }
        void enableMTPShiftedRowReuseFromSidecar()
        {
            supports_mtp_shifted_row_reuse_from_sidecar_ = true;
        }
        void enableLogicalMTPVerifierBaseCheckpoint()
        {
            supports_logical_mtp_verifier_base_checkpoint_ = true;
        }
        void enableStochasticDeviceSampling()
        {
            supports_stochastic_device_sampling_ = true;
        }
        /**
         * @brief Force one request-batched stochastic verifier lane to reject.
         *
         * Production GPU verification can naturally produce a mixed request
         * batch where one request accepts all drafts and another rejects early.
         * Tests use this hook to exercise that orchestration state handoff
         * deterministically without relying on fragile probability thresholds.
         */
        void forceStochasticRequestBatchReject(int request_id,
                                               int correction_token = 4)
        {
            forced_request_batch_rejections_.push_back(
                {request_id, correction_token});
        }
        void enableMainLogitsBatchRowsOnDevice()
        {
            supports_main_logits_batch_rows_on_device_ = true;
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
        /**
         * @brief Override the token produced by rejecting all-position rows.
         *
         * Pending-condition verifier rows model a serial continuation after the
         * previously emitted correction token.  Tests can use this hook to make
         * that row match the mock decode-logit script instead of the default
         * first-step reject token.
         */
        void setVerifierRejectTokenScript(std::vector<int> script)
        {
            verifier_reject_token_script_ = std::move(script);
            verifier_reject_token_script_index_ = 0;
            last_verifier_reject_token_ = VERIFY_REJECT_TOKEN;
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
            snapshot.mtp_cached_tokens = {mtp_shifted_cached_tokens_};
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
            snapshot.mtp_cached_tokens = {mtp_shifted_cached_tokens_};
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
            mtp_shifted_cached_tokens_ =
                snapshot.mtp_cached_tokens.empty()
                    ? shiftedTargetForMainTokens(position_)
                    : snapshot.mtp_cached_tokens.front();
            all_position_logits_enabled_ = false;
            row_indexed_all_position_logits_enabled_ = false;
            row_indexed_all_position_logits_row_count_ = 0;
            return true;
        }

        PrefixRuntimeStateSnapshot prefixStateProbe() const override
        {
            PrefixRuntimeStateSnapshot probe;
            probe.initialized = true;
            probe.architecture = architecture();
            probe.execution_path = "graph";
            probe.primary_device = primary_device_;
            probe.current_position = position_;
            probe.positions = {position_};
            probe.sequence_lengths = sequence_lengths_.empty()
                                         ? std::vector<int>{position_}
                                         : sequence_lengths_;
            if (mtp_enabled_)
            {
                PrefixKVCacheProbe mtp_cache;
                mtp_cache.owner = "mock_mtp";
                mtp_cache.device = primary_device_;
                mtp_cache.n_layers = 1;
                PrefixKVLayerProbe layer;
                layer.cache_layer = 0;
                layer.global_layer = 0;
                layer.seq_idx = 0;
                layer.cached_tokens = mtp_shifted_cached_tokens_;
                mtp_cache.layers.push_back(layer);
                probe.mtp_kv_caches.push_back(std::move(mtp_cache));
            }
            return probe;
        }

        void requireMTPDecodeEquivalentReplay()
        {
            requires_mtp_decode_equivalent_replay_ = true;
        }
        void enableMTPSpecStatePublication()
        {
            supports_mtp_spec_state_publication_ = true;
        }
        void enableDeviceResidentMTPSpecStatePublication()
        {
            supports_mtp_spec_state_publication_ = true;
            supports_device_resident_mtp_spec_state_publication_ = true;
        }
        void hideMTPSpecStatePublicationFromPolicy()
        {
            hide_mtp_spec_state_publication_from_policy_ = true;
        }
        void enableGroupedOutcomeDeviceResidentPublication(int rows)
        {
            const MTPVerifierEconomyLane lane =
                MTPVerifierEconomyLane::groupedOutcomeDevicePublicationEconomicsPending(rows);
            mtp_verifier_economy_capability_.dense = lane;
            mtp_verifier_economy_capability_.moe = lane;
        }
        void setMTPSpecStatePublicationOk(bool ok)
        {
            publish_mtp_spec_state_ok_ = ok;
        }
        int publishMTPSpecStateCount() const
        {
            return publish_mtp_spec_state_count_;
        }
        int publishMTPSpecStateBatchCount() const
        {
            return publish_mtp_spec_state_batch_count_;
        }
        int publishDeviceResidentMTPSpecStateCount() const
        {
            return publish_device_resident_mtp_spec_state_count_;
        }
        int adoptDeviceResidentHostStateCount() const
        {
            return adopt_device_resident_host_state_count_;
        }
        const MTPSpecStepPlan &lastPublishedMTPSpecStep() const
        {
            return last_published_mtp_spec_step_;
        }
        const MTPSpecStepPlanBatch &lastPublishedMTPSpecBatch() const
        {
            return last_published_mtp_spec_batch_;
        }
        const DeviceSpeculativePublicationRequest &
        lastDeviceResidentPublicationRequest() const
        {
            return last_device_resident_publication_request_;
        }
        const MTPSpecStepPlanBatch &
        lastAdoptedDeviceResidentHostState() const
        {
            return last_adopted_device_resident_host_state_;
        }
        const std::vector<std::string> &publicationEvents() const
        {
            return publication_events_;
        }
        int residentSidecarCountAtLastHostBridge() const
        {
            return resident_sidecar_count_at_last_host_bridge_;
        }

    private:
        std::optional<int> forcedRequestBatchRejectionToken(int request_id) const
        {
            for (const auto &entry : forced_request_batch_rejections_)
            {
                if (entry.first == request_id)
                    return entry.second;
            }
            return std::nullopt;
        }

        /**
         * @brief Serialize one compact resident verifier outcome row.
         *
         * The production GPU path writes `[request, token]` and
         * `[request, metadata-field]` rows in arena memory.  The mock keeps the
         * same padded layout so resident enqueue tests exercise the same stride
         * and host-bridge contract without needing CUDA or ROCm hardware.
         */
        void writeResidentOutcomeRow(
            int request_index,
            const DeviceSpeculativeVerifyBatchOutcome &outcome)
        {
            using namespace sampling_math;
            const size_t token_base =
                static_cast<size_t>(request_index) *
                static_cast<size_t>(kSpeculativeBatchMaxOutputTokens);
            const size_t meta_base =
                static_cast<size_t>(request_index) *
                static_cast<size_t>(kSpeculativeBatchMetaCount);

            for (int i = 0; i < kSpeculativeBatchMaxOutputTokens; ++i)
            {
                resident_output_tokens_[token_base + static_cast<size_t>(i)] =
                    outcome.output_tokens[static_cast<size_t>(i)];
            }

            int *meta = resident_meta_.data() + meta_base;
            std::fill(meta, meta + kSpeculativeBatchMetaCount, 0);
            meta[kSpecBatchMetaOk] = outcome.ok ? 1 : 0;
            meta[kSpecBatchMetaOutputCount] = outcome.output_token_count;
            meta[kSpecBatchMetaAcceptedSpeculativePrefix] =
                outcome.accepted_speculative_prefix;
            meta[kSpecBatchMetaTargetVerifierStateCommitCount] =
                outcome.target_verifier_state_commit_count;
            meta[kSpecBatchMetaReadyToken] = outcome.ready_token;
            meta[kSpecBatchMetaRejectedVerifiedToken] =
                outcome.rejected_verified_token;
            meta[kSpecBatchMetaStoppedOnOutput] =
                outcome.stopped_on_output ? 1 : 0;
            meta[kSpecBatchMetaAllSpeculativeAccepted] =
                outcome.all_speculative_accepted ? 1 : 0;
            meta[kSpecBatchMetaConsumedVerifierRows] =
                outcome.consumed_verifier_rows;
            meta[kSpecBatchMetaSampledTerminal] =
                outcome.sampled_terminal ? 1 : 0;
        }

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

        bool buildTemperatureOnlyDraftProposal(
            DeviceLogitsSource source,
            int row,
            int slot,
            const SamplingParams &params,
            int vocab_size)
        {
            if (!supports_stochastic_device_sampling_ ||
                source != DeviceLogitsSource::MTP ||
                row != 0 ||
                slot < 0 ||
                vocab_size != VOCAB_SIZE ||
                mtp_logits_.size() != VOCAB_SIZE)
            {
                return false;
            }

            SamplingParams proposal_params = params;
            proposal_params.top_k = 0;
            proposal_params.top_p = 1.0f;
            proposal_params.presence_penalty = 0.0f;
            proposal_params.frequency_penalty = 0.0f;
            proposal_params.dry_multiplier = 0.0f;
            proposal_params.dry_penalty_last_n = 0;

            Sampler proposal_sampler(params.seed);
            auto &target = deviceDistribution(DeviceDistributionBuffer::Draft, slot);
            target = proposal_sampler.compute_distribution(
                mtp_logits_.data(),
                VOCAB_SIZE,
                proposal_params);
            return !target.empty();
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

        int shiftedTargetForMainTokens(int main_tokens) const
        {
            return std::max(0, main_tokens - 1);
        }

        void appendSpeculativeSidecarShiftedRow()
        {
            if (!supports_mtp_shifted_row_reuse_from_sidecar_)
                return;
            mtp_shifted_cached_tokens_ =
                std::max(mtp_shifted_cached_tokens_, position_);
        }

        void adoptPublishedMainTokens(int target_cached_tokens)
        {
            position_ = target_cached_tokens;
            mtp_shifted_cached_tokens_ =
                shiftedTargetForMainTokens(target_cached_tokens);
        }

        bool appendOneShiftedMTPRow(int already_appended_tokens,
                                    int position_offset_override)
        {
            if (already_appended_tokens < 0)
                return false;
            const int position_offset =
                position_offset_override >= 0
                    ? position_offset_override
                    : position_;
            const int expected_cached_tokens =
                std::max(0, position_offset - 1 + already_appended_tokens);
            if (mtp_shifted_cached_tokens_ > expected_cached_tokens)
                return false;
            mtp_shifted_cached_tokens_ = expected_cached_tokens + 1;
            return true;
        }

        void setupBatchPrefillLogits()
        {
            batch_logits_.assign(
                static_cast<size_t>(sequence_lengths_.size()) *
                    static_cast<size_t>(std::max(0, padded_seq_len_)) *
                    static_cast<size_t>(VOCAB_SIZE),
                -10.0f);
            for (size_t seq = 0; seq < sequence_lengths_.size(); ++seq)
            {
                const int logical_length = sequence_lengths_[seq];
                if (logical_length <= 0 || padded_seq_len_ <= 0)
                    continue;

                const int token =
                    (PREFILL_ARGMAX_TOKEN + static_cast<int>(seq)) % VOCAB_SIZE;
                const size_t offset =
                    (seq * static_cast<size_t>(padded_seq_len_) +
                     static_cast<size_t>(logical_length - 1)) *
                    static_cast<size_t>(VOCAB_SIZE);
                batch_logits_[offset + static_cast<size_t>(token)] = 10.0f;
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
            const int logits_rows = row_indexed_all_position_logits_enabled_
                                        ? row_indexed_all_position_logits_row_count_
                                        : seq_len;
            all_position_logits_.assign(static_cast<size_t>(logits_rows) * VOCAB_SIZE, -10.0f);
            const int speculative_depth = std::max(0, seq_len - 1);
            const int accepted_prefix = nextVerifierAcceptedPrefix(speculative_depth);

            for (int row = 0; row < logits_rows; ++row)
            {
                int token = DECODE_ARGMAX_TOKEN;
                if (row < speculative_depth)
                {
                    if (row < accepted_prefix)
                        token = MTP_ARGMAX_TOKEN;
                    else
                        token = row == 0 ? nextVerifierRejectToken()
                                         : DECODE_ARGMAX_TOKEN;
                }
                all_position_logits_[static_cast<size_t>(row) * VOCAB_SIZE +
                                     static_cast<size_t>(token)] = 10.0f;
            }
            if (column_parallel_logits_)
            {
                resetLocalTensor(all_position_logits_local_, logits_rows);
                for (int row = 0; row < logits_rows; ++row)
                {
                    int token = DECODE_ARGMAX_TOKEN;
                    if (row < speculative_depth)
                    {
                        if (row < accepted_prefix)
                            token = MTP_ARGMAX_TOKEN;
                        else
                            token = row == 0 ? last_verifier_reject_token_
                                             : DECODE_ARGMAX_TOKEN;
                    }
                    setLocalToken(all_position_logits_local_, row, token, 10.0f);
                }
            }
        }

        int nextVerifierAcceptedPrefix(int speculative_depth)
        {
            int accepted_prefix = accept_mtp_token_ ? speculative_depth : 0;
            if (verifier_accepted_prefix_script_index_ <
                verifier_accepted_prefix_script_.size())
            {
                accepted_prefix =
                    verifier_accepted_prefix_script_[verifier_accepted_prefix_script_index_++];
            }
            return std::clamp(accepted_prefix, 0, speculative_depth);
        }

        int nextVerifierRejectToken()
        {
            int token = VERIFY_REJECT_TOKEN;
            if (verifier_reject_token_script_index_ <
                verifier_reject_token_script_.size())
            {
                token =
                    verifier_reject_token_script_[verifier_reject_token_script_index_++];
            }
            token = std::clamp(token, 0, VOCAB_SIZE - 1);
            last_verifier_reject_token_ = token;
            return token;
        }

        void setAllPositionToken(int row, int token)
        {
            if (row < 0 || token < 0 || token >= VOCAB_SIZE)
                return;
            const size_t offset =
                static_cast<size_t>(row) * static_cast<size_t>(VOCAB_SIZE) +
                static_cast<size_t>(token);
            if (offset < all_position_logits_.size())
                all_position_logits_[offset] = 10.0f;
        }

        void setupAllPositionLogitsForBatch(
            const std::vector<std::vector<int>> &token_batches)
        {
            const int logits_rows = row_indexed_all_position_logits_enabled_
                                        ? row_indexed_all_position_logits_row_count_
                                        : padded_seq_len_ * static_cast<int>(token_batches.size());
            all_position_logits_.assign(
                static_cast<size_t>(std::max(0, logits_rows)) *
                    static_cast<size_t>(VOCAB_SIZE),
                -10.0f);
            if (!mtp_spec_verifier_plan_installed_ ||
                !last_mtp_spec_verifier_plan_.ok ||
                last_mtp_spec_verifier_plan_.request_count !=
                    static_cast<int>(token_batches.size()) ||
                static_cast<int>(last_mtp_spec_verifier_plan_.query_start_locs.size()) <
                    static_cast<int>(token_batches.size()) + 1)
            {
                setupAllPositionLogits(padded_seq_len_);
                return;
            }

            for (int request = 0;
                 request < last_mtp_spec_verifier_plan_.request_count;
                 ++request)
            {
                const int start =
                    last_mtp_spec_verifier_plan_.query_start_locs[
                        static_cast<size_t>(request)];
                const int end =
                    last_mtp_spec_verifier_plan_.query_start_locs[
                        static_cast<size_t>(request + 1)];
                const int draft_count = end - start;
                const int speculative_depth = std::max(0, draft_count - 1);
                const int accepted_prefix =
                    nextVerifierAcceptedPrefix(speculative_depth);
                for (int rel = 0; rel < draft_count; ++rel)
                {
                    const int compact_row = start + rel;
                    if (compact_row < 0 || compact_row >= logits_rows)
                        continue;

                    int token = DECODE_ARGMAX_TOKEN;
                    if (rel < speculative_depth)
                    {
                        if (rel < accepted_prefix)
                        {
                            token =
                                last_mtp_spec_verifier_plan_.verifier_input_tokens[
                                    static_cast<size_t>(start + rel + 1)];
                        }
                        else
                        {
                            token = rel == 0 ? VERIFY_REJECT_TOKEN
                                             : DECODE_ARGMAX_TOKEN;
                        }
                    }
                    setAllPositionToken(compact_row, token);
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
        int forward_with_device_token_ids_count_{0};
        int prepare_mtp_verifier_input_tokens_on_device_count_{0};
        int forward_mtp_count_{0};
        int forward_mtp_from_last_draft_count_{0};
        int forward_mtp_for_device_sampling_count_{0};
        int forward_mtp_from_last_draft_for_device_sampling_count_{0};
        int forward_mtp_from_device_draft_for_device_sampling_count_{0};
        int forward_mtp_from_device_target_for_device_sampling_count_{0};
        int forward_mtp_and_sample_count_{0};
        int forward_mtp_batch_and_sample_count_{0};
        int forward_mtp_batch_and_sample_to_device_draft_slots_count_{0};
        int forward_mtp_batch_from_last_draft_and_sample_count_{0};
        int forward_mtp_batch_from_last_draft_and_sample_to_device_draft_slots_count_{0};
        int forward_mtp_from_last_draft_and_sample_count_{0};
        int flush_pending_mtp_work_count_{0};
        int clear_cache_count_{0};
        int restore_count_{0};
        mutable int capture_checkpoint_count_{0};
        int set_all_position_count_{0};
        int set_row_indexed_all_position_count_{0};
        int set_mtp_spec_verifier_plan_count_{0};
        int clear_mtp_spec_verifier_plan_count_{0};
        int commit_mtp_shifted_count_{0};
        int sequential_commit_mtp_shifted_count_{0};
        int last_commit_mtp_already_appended_{0};
        int last_commit_mtp_main_forward_token_count_{0};
        int last_commit_mtp_position_offset_override_{-1};
        int last_commit_mtp_already_appended_shifted_kv_{-1};
        mutable int get_logits_count_{0};
        int sample_main_logits_count_{0};
        int sample_main_logits_to_device_target_slot_count_{0};
        int sample_device_count_{0};
        int sample_main_logits_batch_rows_count_{0};
        int last_main_logits_batch_request_count_{0};
        std::vector<float> last_main_logits_batch_thresholds_;
        int sample_mtp_logits_count_{0};
        int sample_mtp_logits_to_device_draft_slot_count_{0};
        int sample_all_position_logits_count_{0};
        int sample_all_position_logits_batched_count_{0};
        int verify_greedy_all_position_batch_outcome_count_{0};
        int apply_main_penalties_count_{0};
        int apply_mtp_penalties_count_{0};
        int apply_all_position_penalties_count_{0};
        int device_distribution_build_count_{0};
        int device_probability_rows_build_count_{0};
        int device_processed_rows_build_count_{0};
        int device_distribution_sample_count_{0};
	        int device_distribution_sample_deferred_count_{0};
        int device_draft_temperature_proposal_count_{0};
        int device_draft_temperature_proposal_deferred_count_{0};
	        int device_distribution_verify_count_{0};
        int device_distribution_verify_batch_count_{0};
        int device_probability_verify_row_count_{0};
        int stage_stochastic_draft_tokens_count_{0};
        int device_distribution_request_batch_outcome_count_{0};
        uint64_t last_probability_row_inverse_sample_seed_{0};
        int last_probability_row_inverse_sample_logical_position_{0};
	        int device_distribution_batch_outcome_device_first_count_{0};
        bool batch_outcome_used_host_draft_tokens_{false};
        bool last_batch_outcome_used_vllm_probability_rejection_{false};
        uint64_t last_batch_outcome_inverse_sample_seed_{0};
        int last_batch_outcome_inverse_sample_first_logical_position_{0};
        int prepare_mtp_verifier_input_tokens_device_first_count_{0};
        int device_target_shifted_commit_count_{0};
        int resident_logical_state_shifted_commit_count_{0};
        int forward_mtp_from_resident_logical_state_for_device_sampling_count_{0};
        int resident_sidecar_count_at_last_host_bridge_{-1};
        int all_position_verifier_sync_deferral_set_count_{0};
        int all_position_verifier_sync_deferral_enable_count_{0};
        int all_position_verifier_sync_deferral_disable_count_{0};
        int last_sample_all_position_start_row_{-1};
        int last_sample_all_position_row_count_{0};
        int last_sample_mtp_logits_device_draft_slot_{-1};
        int last_sample_main_logits_device_target_slot_{-1};
        int last_mtp_condition_token_{-1};
        int last_chained_mtp_condition_token_{-1};
        int last_chained_mtp_position_id_{-1};
        int last_mtp_batch_device_draft_first_slot_{-1};
        int last_mtp_batch_device_draft_slot_stride_{1};
        std::vector<int> last_chained_mtp_batch_device_draft_first_slots_;
        std::vector<int> last_chained_mtp_batch_device_draft_slot_strides_;
        int last_device_target_shifted_commit_token_{-1};
        int last_resident_logical_state_shifted_commit_token_{-1};
        int last_resident_logical_state_sidecar_token_{-1};
        int last_prepare_mtp_verifier_first_token_{-1};
        int last_prepare_mtp_verifier_first_target_sample_slot_{-1};
        int last_prepare_mtp_verifier_first_draft_slot_{-1};
        int last_prepare_mtp_verifier_draft_token_count_{0};
        int last_prepare_mtp_verifier_total_tokens_{0};
        int last_forward_device_token_seq_len_{0};
        bool is_first_forward_in_cycle_{true};
        bool mtp_enabled_{false};
        bool accept_mtp_token_{true};
        int mtp_draft_tokens_{1};
        int mtp_shifted_cached_tokens_{0};
        bool all_position_logits_enabled_{false};
        bool row_indexed_all_position_logits_enabled_{false};
        bool mtp_spec_verifier_plan_installed_{false};
        bool column_parallel_logits_{false};
        bool supports_mtp_token_coordination_{false};
        bool supports_chained_mtp_drafts_{false};
        bool supports_mtp_sidecar_sample_fusion_{false};
        bool supports_mtp_sidecar_stream_handoff_{false};
        bool supports_mtp_device_draft_token_input_{false};
        bool supports_mtp_sidecar_preserves_main_state_{false};
        bool supports_mtp_shifted_row_reuse_from_sidecar_{false};
        bool supports_logical_mtp_verifier_base_checkpoint_{false};
        bool supports_mtp_spec_state_publication_{false};
        bool hide_mtp_spec_state_publication_from_policy_{false};
        bool supports_device_resident_mtp_spec_state_publication_{false};
        bool publish_mtp_spec_state_ok_{true};
        bool supports_stochastic_device_sampling_{false};
        bool supports_main_logits_batch_rows_on_device_{false};
        bool all_position_verifier_sync_deferral_enabled_{false};
        bool requires_mtp_decode_equivalent_replay_{false};
        bool hide_local_logits_{false};
        bool use_captured_snapshot_{false};
        bool last_commit_mtp_allow_speculative_discard_{false};
        bool flush_pending_mtp_work_ok_{true};
        DeviceId primary_device_{DeviceId::cpu()};
        int vocab_start_{0};
        int vocab_local_{VOCAB_SIZE};
        int row_indexed_all_position_logits_row_count_{0};
        std::string mtp_unsupported_reason_;
        SamplingParams last_main_logits_batch_sampling_params_;
        PrefixStateSnapshot captured_snapshot_;
        PrefixStateSnapshot last_restored_snapshot_;
        MTPSpecStepPlan last_published_mtp_spec_step_;
        MTPSpecStepPlanBatch last_published_mtp_spec_batch_;
        MTPVerifierEconomyCapability mtp_verifier_economy_capability_;
        DeviceSpeculativePublicationRequest last_device_resident_publication_request_;
        MTPSpecStepPlanBatch last_adopted_device_resident_host_state_;
        MTPSpecDecodeVerifierInputPlan last_mtp_spec_verifier_plan_;
        std::vector<int> last_forward_tokens_;
        std::vector<std::vector<int>> forward_history_;
        std::vector<std::vector<int>> last_forward_batch_;
        std::vector<int> sequence_lengths_;
        std::vector<float> batch_logits_;
        std::vector<int> last_commit_mtp_tokens_;
        std::vector<int32_t> last_mtp_spec_verifier_rows_;
        std::vector<int32_t> last_mtp_spec_verifier_tokens_;
        std::vector<int32_t> last_mtp_batch_condition_tokens_;
        std::vector<int> last_mtp_batch_position_ids_;
        std::vector<int32_t> last_chained_mtp_batch_condition_tokens_;
        std::vector<int> last_chained_mtp_batch_position_ids_;
        std::vector<int32_t> last_staged_stochastic_draft_tokens_;
        std::vector<int> last_staged_stochastic_draft_first_slots_;
        std::vector<int> last_batch_outcome_first_target_slots_;
        std::vector<int> last_batch_outcome_first_draft_slots_;
        std::vector<int> last_batch_outcome_bonus_target_slots_;
        std::vector<int> last_request_batch_outcome_request_ids_;
        std::vector<int> last_request_batch_outcome_row_counts_;
        std::vector<int> last_request_batch_outcome_first_target_slots_;
        std::vector<int> last_request_batch_outcome_first_draft_slots_;
        std::vector<int> last_request_batch_outcome_bonus_target_slots_;
        std::vector<int32_t> last_request_batch_outcome_first_tokens_;
        std::vector<std::vector<int32_t>> last_request_batch_outcome_draft_tokens_;
        std::vector<std::vector<float>> last_request_batch_outcome_accept_thresholds_;
        std::vector<std::vector<float>> last_request_batch_outcome_residual_thresholds_;
        std::vector<float> last_request_batch_outcome_bonus_thresholds_;
        std::vector<uint64_t> last_request_batch_outcome_inverse_sample_seeds_;
        std::vector<int> last_request_batch_outcome_inverse_sample_first_positions_;
        std::vector<bool> last_request_batch_outcome_derived_thresholds_;
        std::vector<std::string> publication_events_;
        std::vector<std::pair<int, int>> forced_request_batch_rejections_;
        std::vector<int> verifier_accepted_prefix_script_;
        std::vector<int> verifier_reject_token_script_;
        std::vector<int> decode_argmax_script_;
        mutable std::vector<PrefixStateSnapshot> captured_checkpoint_script_;
        std::array<std::vector<SamplingDistributionEntry>, 8> target_device_distributions_;
        std::array<std::vector<SamplingDistributionEntry>, 6> draft_device_distributions_;
        std::array<int32_t, 8> device_target_sample_tokens_ = {-1, -1, -1, -1, -1, -1, -1, -1};
        std::array<int32_t, 6> device_draft_sample_tokens_ = {-1, -1, -1, -1, -1, -1};
        std::array<int32_t, 8> device_verifier_input_tokens_ = {-1, -1, -1, -1, -1, -1, -1, -1};
        static constexpr int kMockResidentOutcomeRequestCapacity = 4;
        std::array<int32_t,
                   sampling_math::kSpeculativeBatchMaxOutputTokens *
                       kMockResidentOutcomeRequestCapacity>
            resident_output_tokens_{};
        std::array<int,
                   sampling_math::kSpeculativeBatchMetaCount *
                       kMockResidentOutcomeRequestCapacity>
            resident_meta_{};
        int resident_stream_token_{0};
        int resident_outcome_response_ready_event_token_{0};
        int resident_ready_event_token_{0};
        bool resident_logical_state_valid_{false};
        int resident_logical_state_request_count_{0};
        std::array<int32_t, kMockResidentOutcomeRequestCapacity>
            resident_target_positions_{};
        std::array<int32_t, kMockResidentOutcomeRequestCapacity>
            resident_target_sequence_lengths_{};
        std::array<int32_t, kMockResidentOutcomeRequestCapacity>
            resident_accepted_state_counts_{};
        std::array<int32_t, kMockResidentOutcomeRequestCapacity>
            resident_base_cached_tokens_ = {-1, -1, -1, -1};
        std::array<int32_t, kMockResidentOutcomeRequestCapacity>
            resident_next_condition_tokens_{};
        std::array<int32_t, kMockResidentOutcomeRequestCapacity>
            resident_all_drafts_accepted_flags_{};
        std::array<int32_t, kMockResidentOutcomeRequestCapacity>
            resident_stopped_flags_{};
        std::array<int32_t, kMockResidentOutcomeRequestCapacity>
            resident_publication_ok_flags_{};
        const void *last_forward_device_token_ids_{nullptr};
        std::vector<SamplingDistributionEntry> invalid_distribution_;
        size_t verifier_accepted_prefix_script_index_{0};
        size_t verifier_reject_token_script_index_{0};
        int last_verifier_reject_token_{VERIFY_REJECT_TOKEN};
        size_t decode_argmax_script_index_{0};
        mutable size_t captured_checkpoint_script_index_{0};
        int last_forward_seq_len_{0};
        int forward_batch_call_count_{0};
        int publish_mtp_spec_state_count_{0};
        int publish_mtp_spec_state_batch_count_{0};
        int publish_device_resident_mtp_spec_state_count_{0};
        int adopt_device_resident_host_state_count_{0};
        int position_{0};
        int batch_capacity_{1};
        int padded_seq_len_{0};
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
                                                                             MTPVerifyMode verify_mode = MTPVerifyMode::Greedy,
                                                                             bool local_pp_topology = false,
                                                                             int max_request_batch = 1)
        {
            auto mock = std::make_unique<MockInferenceRunner>();
            auto *mock_ptr = mock.get(); // Keep raw pointer for inspection
            if (mtp_enabled)
            {
                mock_ptr->enableMTP(mtp_accept);
            }
            mock_ptr->setMTPDraftTokens(mtp_draft_tokens);
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
            config.mtp.max_request_batch = max_request_batch;
            config.mtp.verify_mode = verify_mode;
            config.mtp.depth_policy = depth_policy;

            RankExecutionPlan runner_plan = plan_;
            if (local_pp_topology)
            {
                /*
                 * Only the topology shape is needed for these runner-level
                 * guard tests. Real PP execution is covered by RankOrchestrator
                 * and parity integration tests.
                 */
                runner_plan.local_pp_devices = {GlobalDeviceAddress::cpu(),
                                                GlobalDeviceAddress::cpu()};
                runner_plan.local_pp_layer_boundaries = {0, 12, 24};
            }

            std::unique_ptr<OrchestrationRunner> runner;
            if (mpi_ctx)
            {
                runner = std::make_unique<OrchestrationRunner>(
                    std::move(config), runner_plan, std::move(mock), std::move(mpi_ctx));
            }
            else
            {
                runner = std::make_unique<OrchestrationRunner>(
                    std::move(config), runner_plan, std::move(mock));
            }

            // Set greedy sampling (temperature=0)
            SamplingParams greedy;
            greedy.temperature = 0.0f;
            runner->setSamplingParams(greedy);

            runners_.push_back(std::move(runner));
            return {runners_.back().get(), mock_ptr};
        }

        std::pair<OrchestrationRunner *, MockInferenceRunner *>
        createSingleDeviceRequestBatchRunner(
            int max_request_batch = 2,
            int mtp_draft_tokens = 1,
            MTPVerifyMode verify_mode = MTPVerifyMode::Greedy)
        {
            auto [runner, mock] = createRunner(
                /*mtp_enabled=*/true,
                /*mtp_accept=*/true,
                /*mtp_unsupported_reason=*/{},
                /*mpi_ctx=*/nullptr,
                /*mtp_token_coordination=*/false,
                /*hide_local_logits=*/false,
                DeviceId::cpu(),
                mtp_draft_tokens,
                /*chained_mtp_support=*/mtp_draft_tokens > 1,
                /*sidecar_sample_fusion=*/false,
                MTPDepthPolicyConfig{},
                verify_mode,
                /*local_pp_topology=*/false,
                max_request_batch);
            mock->setBatchCapacity(max_request_batch);
            mock->enableMTPSpecStatePublication();
            mock->enableMTPTokenCoordination(/*hide_local_logits=*/false);
            return {runner, mock};
        }

        LocalTPRunnerHarness createLocalTPRunner(bool mtp_accept = true,
                                                 bool column_parallel_logits = false,
                                                 std::vector<GlobalDeviceAddress> devices = {},
                                                 int mtp_draft_tokens = 1,
                                                 MTPDepthPolicyConfig depth_policy = {},
                                                 bool spec_state_publication = false)
        {
            if (devices.empty())
            {
                devices = {GlobalDeviceAddress::cpu(), GlobalDeviceAddress::cpu()};
            }
            auto child0 = std::make_unique<MockInferenceRunner>();
            auto child1 = std::make_unique<MockInferenceRunner>();
            child0->setMTPDraftTokens(mtp_draft_tokens);
            child1->setMTPDraftTokens(mtp_draft_tokens);
            child0->enableMTP(mtp_accept);
            child1->enableMTP(mtp_accept);
            if (mtp_draft_tokens > 1)
            {
                child0->enableChainedMTPDrafts();
                child1->enableChainedMTPDrafts();
            }
            if (spec_state_publication)
            {
                child0->enableMTPSpecStatePublication();
                child1->enableMTPSpecStatePublication();
            }
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
            rank_config.mtp.draft_tokens = mtp_draft_tokens;
            rank_config.mtp.verify_mode = MTPVerifyMode::Greedy;
            rank_config.mtp.depth_policy = depth_policy;

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
            config.mtp.draft_tokens = mtp_draft_tokens;
            config.mtp.verify_mode = MTPVerifyMode::Greedy;
            config.mtp.depth_policy = depth_policy;

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

    TEST_F(Test__PrefillDecodeTransition, PrefillBatchInitializesRequestSlotsAndBlocksScalarDecode)
    {
        auto [runner, mock] = createSingleDeviceRequestBatchRunner(/*max_request_batch=*/3);

        EXPECT_FALSE(runner->supportsDecodeStepBatch(2));
        ASSERT_TRUE(runner->supportsPrefillBatch(2));
        ASSERT_TRUE(runner->prefillBatch({{1, 2, 3}, {4, 5}}))
            << runner->lastError();
        EXPECT_TRUE(runner->supportsDecodeStepBatch(2));

        EXPECT_EQ(mock->forwardCallCount(), 0)
            << "Request-batched prefill must not initialize only request 0";
        EXPECT_EQ(mock->forwardBatchCallCount(), 1);
        EXPECT_THAT(mock->lastForwardBatch(),
                    ElementsAre(ElementsAre(1, 2, 3), ElementsAre(4, 5)));
        EXPECT_THAT(mock->sequence_lengths(), ElementsAre(3, 2));

        GenerationResult scalar_decode = runner->decodeStep();
        EXPECT_FALSE(scalar_decode.error.empty());
        EXPECT_THAT(scalar_decode.error, HasSubstr("decodeStep() cannot consume"));
        EXPECT_EQ(mock->forwardCallCount(), 0)
            << "Scalar decode must fail before mutating batched live state";

        GenerationBatchResult batch_step = runner->decodeStepBatch(2);
        ASSERT_TRUE(batch_step.error.empty()) << batch_step.error;
        ASSERT_EQ(batch_step.requests.size(), 2u);
        EXPECT_THAT(batch_step.requests[0].tokens,
                    ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN));
        EXPECT_THAT(batch_step.requests[1].tokens,
                    ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN + 1));
        EXPECT_EQ(mock->forwardCallCount(), 0)
            << "Batched ready-token decode must consume terminal prefill logits";
        EXPECT_TRUE(runner->supportsDecodeStepBatch(2));

        GenerationBatchResult second_batch_step = runner->decodeStepBatch(2);
        ASSERT_TRUE(second_batch_step.error.empty()) << second_batch_step.error;
        ASSERT_EQ(second_batch_step.requests.size(), 2u);
        EXPECT_THAT(second_batch_step.requests[0].tokens,
                    ElementsAre(MockInferenceRunner::MTP_ARGMAX_TOKEN));
        EXPECT_THAT(second_batch_step.requests[1].tokens,
                    ElementsAre(MockInferenceRunner::MTP_ARGMAX_TOKEN));
        EXPECT_EQ(mock->forwardMTPBatchAndSampleCount(), 1);
        EXPECT_THAT(mock->lastMTPBatchConditionTokens(),
                    ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN,
                                MockInferenceRunner::PREFILL_ARGMAX_TOKEN + 1));
        EXPECT_THAT(mock->lastMTPBatchPositionIds(), ElementsAre(3, 2));
        EXPECT_EQ(mock->setMTPSpecVerifierPlanCount(), 1);
        EXPECT_THAT(mock->lastMTPSpecVerifierTokens(),
                    ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN,
                                MockInferenceRunner::MTP_ARGMAX_TOKEN,
                                MockInferenceRunner::PREFILL_ARGMAX_TOKEN + 1,
                                MockInferenceRunner::MTP_ARGMAX_TOKEN));
        EXPECT_EQ(mock->forwardBatchCallCount(), 2)
            << "The verifier continuation should use one padded batch forward";
        EXPECT_EQ(mock->sampleAllPositionLogitsBatchedCount(), 1);
        EXPECT_EQ(mock->publishMTPSpecStateBatchCount(), 1);
        EXPECT_EQ(mock->lastPublishedMTPSpecBatch().request_count, 2);
        EXPECT_THAT(mock->sequence_lengths(), ElementsAre(5, 4));

        GenerationBatchResult third_batch_step = runner->decodeStepBatch(2);
        ASSERT_TRUE(third_batch_step.error.empty()) << third_batch_step.error;
        ASSERT_EQ(third_batch_step.requests.size(), 2u);
        EXPECT_THAT(third_batch_step.requests[0].tokens,
                    ElementsAre(MockInferenceRunner::DECODE_ARGMAX_TOKEN));
        EXPECT_THAT(third_batch_step.requests[1].tokens,
                    ElementsAre(MockInferenceRunner::DECODE_ARGMAX_TOKEN));
        EXPECT_EQ(mock->forwardBatchCallCount(), 2)
            << "Ready bonus tokens should be consumed without another forward";

        runner->clearCache();
        ASSERT_TRUE(runner->prefill({9, 8}));
        GenerationResult scalar_after_clear = runner->decodeStep();
        EXPECT_TRUE(scalar_after_clear.success())
            << "clearCache() must release request-batched live-state ownership";
    }

    TEST_F(Test__PrefillDecodeTransition, RequestBatchedGpuPrefillSamplesReadyRowsOnDevice)
    {
        auto [runner, mock] = createSingleDeviceRequestBatchRunner(/*max_request_batch=*/2);
        mock->setPrimaryDevice(DeviceId::cuda(0));
        mock->enableMainLogitsBatchRowsOnDevice();

        ASSERT_TRUE(runner->prefillBatch({{1, 2, 3}, {4, 5}}))
            << runner->lastError();

        GenerationBatchResult batch_step = runner->decodeStepBatch(2);
        ASSERT_TRUE(batch_step.error.empty()) << batch_step.error;
        ASSERT_THAT(batch_step.requests, SizeIs(2));
        EXPECT_THAT(batch_step.requests[0].tokens,
                    ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN));
        EXPECT_THAT(batch_step.requests[1].tokens,
                    ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN + 1));
        EXPECT_EQ(mock->sampleMainLogitsBatchRowsCount(), 1)
            << "GPU request-batched prefill must sample terminal logits through "
               "the runner-owned device path";
        EXPECT_EQ(mock->lastMainLogitsBatchRequestCount(), 2);
        EXPECT_EQ(mock->getLogitsCallCount(), 0)
            << "The CPU Sampler must never receive GPU logits pointers";
    }

    TEST_F(Test__PrefillDecodeTransition, RequestBatchedStochasticGpuPrefillUsesPositionKeyedThresholds)
    {
        auto [runner, mock] =
            createSingleDeviceRequestBatchRunner(
                /*max_request_batch=*/2,
                /*mtp_draft_tokens=*/1,
                MTPVerifyMode::SpeculativeSampling);
        mock->setPrimaryDevice(DeviceId::cuda(0));
        mock->enableMainLogitsBatchRowsOnDevice();

        SamplingParams sampling;
        sampling.temperature = 0.6f;
        sampling.top_k = 20;
        sampling.top_p = 0.95f;
        sampling.seed = 1234;
        runner->setSamplingParams(sampling);

        ASSERT_TRUE(runner->prefillBatch({{1, 2, 3}, {4, 5}}))
            << runner->lastError();

        GenerationBatchResult batch_step = runner->decodeStepBatch(2);
        ASSERT_TRUE(batch_step.error.empty()) << batch_step.error;

        const std::vector<float> &thresholds =
            mock->lastMainLogitsBatchThresholds();
        ASSERT_THAT(thresholds, SizeIs(2));
        EXPECT_NEAR(thresholds[0],
                    sampling_math::uniform01(
                        sampling.seed,
                        /*logical_position=*/3u * 8u +
                            static_cast<uint64_t>(
                                0 /* MTPSpecStochasticDrawPurpose::Sample */)),
                    1e-7f);
        EXPECT_NEAR(thresholds[1],
                    sampling_math::uniform01(
                        sampling.seed,
                        /*logical_position=*/2u * 8u +
                            static_cast<uint64_t>(
                                0 /* MTPSpecStochasticDrawPurpose::Sample */)),
                    1e-7f);
        EXPECT_EQ(mock->getLogitsCallCount(), 0)
            << "Seeded stochastic request batches must remain on the GPU "
               "distribution path instead of falling back to host logits";
    }

    TEST_F(Test__PrefillDecodeTransition, DeviceSpeculativeOutcomeHandleRequiresStreamAndBuffers)
    {
        DeviceSpeculativeOutcomeHandle handle;
        EXPECT_FALSE(handle.valid());

        int32_t output_tokens[sampling_math::kSpeculativeBatchMaxOutputTokens] = {};
        int meta[sampling_math::kSpeculativeBatchMetaCount] = {};
        int stream_token = 0;
        int response_ready_event_token = 0;

        handle.output_tokens_device = output_tokens;
        handle.meta_device = meta;
        handle.request_count = 1;
        handle.stream = &stream_token;
        handle.response_ready_event =
            std::shared_ptr<void>(&response_ready_event_token, [](void *) {});
        EXPECT_TRUE(handle.valid());

        handle.response_ready_event.reset();
        EXPECT_FALSE(handle.valid());

        handle.response_ready_event =
            std::shared_ptr<void>(&response_ready_event_token, [](void *) {});
        handle.stream = nullptr;
        EXPECT_FALSE(handle.valid());

        handle.stream = &stream_token;
        handle.meta_stride = sampling_math::kSpeculativeBatchMetaCount - 1;
        EXPECT_FALSE(handle.valid());
    }

    TEST_F(Test__PrefillDecodeTransition, ScalarResidentOutcomeBuildsOneRequestDescriptor)
    {
        MockInferenceRunner runner;
        runner.setPrimaryDevice(DeviceId::cuda(0));
        const int32_t draft_tokens[] = {11, 12};
        const float accept_thresholds[] = {0.25f, 0.50f};
        const float residual_thresholds[] = {0.75f, 0.90f};
        const int32_t stop_tokens[] = {2, 3};

        DeviceSpeculativeOutcomeHandle handle;
        ASSERT_TRUE(runner.verifyStochasticDistributionsBatchOutcomeOnDeviceResident(
            /*first_target_slot=*/4,
            /*first_draft_slot=*/7,
            draft_tokens,
            accept_thresholds,
            residual_thresholds,
            /*row_count=*/2,
            /*first_token=*/9,
            stop_tokens,
            /*stop_token_count=*/2,
            /*bonus_target_slot=*/6,
            /*bonus_threshold=*/0.42f,
            &handle,
            /*inverse_sample_seed=*/1234,
            /*inverse_sample_first_logical_position=*/56,
            /*use_vllm_probability_rejection=*/true));

        EXPECT_TRUE(handle.valid());
        EXPECT_EQ(handle.device, DeviceId::cuda(0));
        EXPECT_EQ(handle.request_count, 1);
        EXPECT_EQ(runner.verifyStochasticRequestBatchOutcomeCount(), 1);
        EXPECT_THAT(runner.lastRequestBatchOutcomeRequestIds(), ElementsAre(0));
        EXPECT_THAT(runner.lastRequestBatchOutcomeRowCounts(), ElementsAre(2));
        EXPECT_THAT(runner.lastRequestBatchOutcomeFirstTargetSlots(), ElementsAre(4));
        EXPECT_THAT(runner.lastRequestBatchOutcomeFirstDraftSlots(), ElementsAre(7));
        EXPECT_THAT(runner.lastRequestBatchOutcomeBonusTargetSlots(), ElementsAre(6));
        EXPECT_THAT(runner.lastRequestBatchOutcomeFirstTokens(), ElementsAre(9));
        ASSERT_EQ(runner.lastRequestBatchOutcomeDraftTokens().size(), 1u);
        EXPECT_THAT(runner.lastRequestBatchOutcomeDraftTokens()[0],
                    ElementsAre(11, 12));
        ASSERT_EQ(runner.lastRequestBatchOutcomeAcceptThresholds().size(), 1u);
        EXPECT_THAT(runner.lastRequestBatchOutcomeAcceptThresholds()[0],
                    ElementsAre(FloatEq(0.25f), FloatEq(0.50f)));
        ASSERT_EQ(runner.lastRequestBatchOutcomeResidualThresholds().size(), 1u);
        EXPECT_THAT(runner.lastRequestBatchOutcomeResidualThresholds()[0],
                    ElementsAre(FloatEq(0.75f), FloatEq(0.90f)));
        EXPECT_THAT(runner.lastRequestBatchOutcomeBonusThresholds(),
                    ElementsAre(FloatEq(0.42f)));
        EXPECT_THAT(runner.lastRequestBatchOutcomeInverseSampleSeeds(),
                    ElementsAre(1234u));
        EXPECT_THAT(runner.lastRequestBatchOutcomeInverseSampleFirstPositions(),
                    ElementsAre(56));
        ASSERT_EQ(runner.lastRequestBatchOutcomeDerivedThresholds().size(), 1u);
        EXPECT_FALSE(runner.lastRequestBatchOutcomeDerivedThresholds()[0]);
    }

    TEST_F(Test__PrefillDecodeTransition, ScalarResidentOutcomeCanDeriveSeededThresholds)
    {
        MockInferenceRunner runner;
        runner.setPrimaryDevice(DeviceId::cuda(0));
        const int32_t draft_tokens[] = {11, 12};
        const int32_t stop_tokens[] = {2};
        constexpr uint64_t seed = 1234;

        DeviceSpeculativeOutcomeHandle handle;
        ASSERT_TRUE(runner.verifyStochasticDistributionsBatchOutcomeOnDeviceResident(
            /*first_target_slot=*/4,
            /*first_draft_slot=*/7,
            draft_tokens,
            /*accept_thresholds=*/nullptr,
            /*residual_thresholds=*/nullptr,
            /*row_count=*/2,
            /*first_token=*/9,
            stop_tokens,
            /*stop_token_count=*/1,
            /*bonus_target_slot=*/6,
            /*bonus_threshold=*/0.42f,
            &handle,
            seed,
            /*inverse_sample_first_logical_position=*/56,
            /*use_vllm_probability_rejection=*/true));

        EXPECT_TRUE(handle.valid());
        ASSERT_EQ(runner.lastRequestBatchOutcomeDerivedThresholds().size(), 1u);
        EXPECT_TRUE(runner.lastRequestBatchOutcomeDerivedThresholds()[0]);
        ASSERT_EQ(runner.lastRequestBatchOutcomeAcceptThresholds().size(), 1u);
        ASSERT_EQ(runner.lastRequestBatchOutcomeResidualThresholds().size(), 1u);
        EXPECT_THAT(runner.lastRequestBatchOutcomeAcceptThresholds()[0],
                    ElementsAre(
                        FloatEq(mtpSeededVerifierThreshold(
                            seed,
                            56,
                            1 /* MTPSpecStochasticDrawPurpose::Accept */)),
                        FloatEq(mtpSeededVerifierThreshold(
                            seed,
                            57,
                            1 /* MTPSpecStochasticDrawPurpose::Accept */))));
        EXPECT_THAT(runner.lastRequestBatchOutcomeResidualThresholds()[0],
                    ElementsAre(
                        FloatEq(mtpSeededVerifierThreshold(
                            seed,
                            56,
                            2 /* MTPSpecStochasticDrawPurpose::Residual */)),
                        FloatEq(mtpSeededVerifierThreshold(
                            seed,
                            57,
                            2 /* MTPSpecStochasticDrawPurpose::Residual */))));
    }

    TEST_F(Test__PrefillDecodeTransition, DeviceFirstScalarResidentOutcomeBuildsDeviceTokenDescriptor)
    {
        MockInferenceRunner runner;
        runner.setPrimaryDevice(DeviceId::cuda(0));
        const float accept_thresholds[] = {0.1f};
        const float residual_thresholds[] = {0.2f};

        DeviceSpeculativeOutcomeHandle handle;
        ASSERT_TRUE(runner.verifyStochasticDistributionsBatchOutcomeOnDeviceFirstTokenResident(
            /*first_target_slot=*/1,
            /*first_draft_slot=*/2,
            /*draft_tokens=*/nullptr,
            accept_thresholds,
            residual_thresholds,
            /*row_count=*/1,
            /*first_target_sample_slot=*/3,
            /*stop_tokens=*/nullptr,
            /*stop_token_count=*/0,
            /*bonus_target_slot=*/4,
            /*bonus_threshold=*/0.8f,
            &handle,
            /*inverse_sample_seed=*/99,
            /*inverse_sample_first_logical_position=*/12,
            /*use_vllm_probability_rejection=*/true));

        EXPECT_TRUE(handle.valid());
        EXPECT_EQ(runner.verifyStochasticRequestBatchOutcomeCount(), 1);
        EXPECT_THAT(runner.lastRequestBatchOutcomeFirstTargetSlots(), ElementsAre(1));
        EXPECT_THAT(runner.lastRequestBatchOutcomeFirstDraftSlots(), ElementsAre(2));
        EXPECT_THAT(runner.lastRequestBatchOutcomeFirstTokens(), ElementsAre(-1));
        ASSERT_EQ(runner.lastRequestBatchOutcomeDraftTokens().size(), 1u);
        EXPECT_THAT(runner.lastRequestBatchOutcomeDraftTokens()[0],
                    ElementsAre(-1));
        EXPECT_THAT(runner.lastRequestBatchOutcomeBonusTargetSlots(), ElementsAre(4));
        EXPECT_THAT(runner.lastRequestBatchOutcomeBonusThresholds(),
                    ElementsAre(FloatEq(0.8f)));
        EXPECT_THAT(runner.lastRequestBatchOutcomeInverseSampleSeeds(),
                    ElementsAre(99u));
        EXPECT_THAT(runner.lastRequestBatchOutcomeInverseSampleFirstPositions(),
                    ElementsAre(12));
        ASSERT_EQ(runner.lastRequestBatchOutcomeDerivedThresholds().size(), 1u);
        EXPECT_FALSE(runner.lastRequestBatchOutcomeDerivedThresholds()[0]);
    }

    TEST_F(Test__PrefillDecodeTransition, RequestBatchedMTPContinuationSupportsDepthThree)
    {
        auto [runner, mock] =
            createSingleDeviceRequestBatchRunner(
                /*max_request_batch=*/2,
                /*mtp_draft_tokens=*/3);

        ASSERT_TRUE(runner->prefillBatch({{1, 2, 3}, {4, 5}}))
            << runner->lastError();

        GenerationBatchResult first = runner->decodeStepBatch(2);
        ASSERT_TRUE(first.error.empty()) << first.error;
        ASSERT_THAT(first.requests, SizeIs(2));
        EXPECT_THAT(first.requests[0].tokens,
                    ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN));
        EXPECT_THAT(first.requests[1].tokens,
                    ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN + 1));
        EXPECT_TRUE(runner->supportsDecodeStepBatch(2))
            << "Depth-three continuation should advertise the same capability "
               "that decodeStepBatch() can execute";

        GenerationBatchResult second = runner->decodeStepBatch(2);
        ASSERT_TRUE(second.error.empty()) << second.error;
        ASSERT_THAT(second.requests, SizeIs(2));
        EXPECT_THAT(second.requests[0].tokens,
                    ElementsAre(MockInferenceRunner::MTP_ARGMAX_TOKEN,
                                MockInferenceRunner::MTP_ARGMAX_TOKEN,
                                MockInferenceRunner::MTP_ARGMAX_TOKEN));
        EXPECT_THAT(second.requests[1].tokens,
                    ElementsAre(MockInferenceRunner::MTP_ARGMAX_TOKEN,
                                MockInferenceRunner::MTP_ARGMAX_TOKEN,
                                MockInferenceRunner::MTP_ARGMAX_TOKEN));

        EXPECT_EQ(mock->forwardMTPBatchAndSampleCount(), 1);
        EXPECT_EQ(mock->forwardMTPBatchFromLastDraftAndSampleCount(), 2)
            << "Depth-three request batching must use batched chained sidecars";
        EXPECT_THAT(mock->lastMTPBatchConditionTokens(),
                    ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN,
                                MockInferenceRunner::PREFILL_ARGMAX_TOKEN + 1));
        EXPECT_THAT(mock->lastMTPBatchPositionIds(), ElementsAre(3, 2));
        EXPECT_THAT(mock->lastChainedMTPBatchConditionTokens(),
                    ElementsAre(MockInferenceRunner::MTP_ARGMAX_TOKEN,
                                MockInferenceRunner::MTP_ARGMAX_TOKEN));
        EXPECT_THAT(mock->lastChainedMTPBatchPositionIds(), ElementsAre(5, 4));
        EXPECT_THAT(mock->lastMTPSpecVerifierTokens(),
                    ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN,
                                MockInferenceRunner::MTP_ARGMAX_TOKEN,
                                MockInferenceRunner::MTP_ARGMAX_TOKEN,
                                MockInferenceRunner::MTP_ARGMAX_TOKEN,
                                MockInferenceRunner::PREFILL_ARGMAX_TOKEN + 1,
                                MockInferenceRunner::MTP_ARGMAX_TOKEN,
                                MockInferenceRunner::MTP_ARGMAX_TOKEN,
                                MockInferenceRunner::MTP_ARGMAX_TOKEN));
        EXPECT_EQ(mock->lastSampleAllPositionRowCount(), 8);
        EXPECT_EQ(mock->publishMTPSpecStateBatchCount(), 1);
        EXPECT_THAT(mock->sequence_lengths(), ElementsAre(7, 6));

        GenerationBatchResult third = runner->decodeStepBatch(2);
        ASSERT_TRUE(third.error.empty()) << third.error;
        EXPECT_THAT(third.requests[0].tokens,
                    ElementsAre(MockInferenceRunner::DECODE_ARGMAX_TOKEN));
        EXPECT_THAT(third.requests[1].tokens,
                    ElementsAre(MockInferenceRunner::DECODE_ARGMAX_TOKEN));
        EXPECT_EQ(mock->forwardBatchCallCount(), 2)
            << "Bonus-ready tokens should not trigger a verifier forward";
    }

    TEST_F(Test__PrefillDecodeTransition, RequestBatchedStochasticDepthOneUsesDeviceDraftSlots)
    {
        auto [runner, mock] =
            createSingleDeviceRequestBatchRunner(
                /*max_request_batch=*/2,
                /*mtp_draft_tokens=*/1,
                MTPVerifyMode::SpeculativeSampling);
        mock->setPrimaryDevice(DeviceId::cuda(0));
        mock->enableMainLogitsBatchRowsOnDevice();
        mock->enableStochasticDeviceSampling();
        mock->enableDeviceResidentMTPSpecStatePublication();

        SamplingParams sampling;
        sampling.temperature = 0.1f;
        sampling.top_k = 5;
        sampling.top_p = 1.0f;
        sampling.seed = 1234;
        runner->setSamplingParams(sampling);

        ASSERT_TRUE(runner->prefillBatch({{1, 2, 3}, {4, 5}}))
            << runner->lastError();

        GenerationBatchResult first = runner->decodeStepBatch(2);
        ASSERT_TRUE(first.error.empty()) << first.error;

        GenerationBatchResult second = runner->decodeStepBatch(2);
        ASSERT_TRUE(second.error.empty()) << second.error;
        ASSERT_THAT(second.requests, SizeIs(2));
        EXPECT_THAT(second.requests[0].tokens,
                    ElementsAre(MockInferenceRunner::MTP_ARGMAX_TOKEN));
        EXPECT_THAT(second.requests[1].tokens,
                    ElementsAre(MockInferenceRunner::MTP_ARGMAX_TOKEN));

        EXPECT_EQ(mock->forwardMTPBatchAndSampleToDeviceDraftSlotsCount(), 1)
            << "Depth-one stochastic request batching should sample sidecar "
               "drafts directly into runner-owned device slots.";
        EXPECT_EQ(mock->lastMTPBatchDeviceDraftFirstSlot(), 0);
        EXPECT_EQ(mock->stageStochasticDraftTokensCount(), 0)
            << "Device-slot sidecar sampling makes the old host-to-device "
               "draft staging hook redundant for depth-one request batches.";
        EXPECT_EQ(mock->verifyStochasticRequestBatchOutcomeCount(), 1);
        EXPECT_THAT(mock->lastRequestBatchOutcomeRowCounts(),
                    ElementsAre(1, 1));
        EXPECT_THAT(mock->lastRequestBatchOutcomeFirstDraftSlots(),
                    ElementsAre(0, 1));
        EXPECT_THAT(mock->lastBatchOutcomeFirstDraftSlots(),
                    ElementsAre(0, 1));
        EXPECT_TRUE(mock->lastBatchOutcomeUsedVLLMProbabilityRejection());
        EXPECT_FALSE(mock->batchOutcomeUsedHostDraftTokens())
            << "The compact verifier reducer must consume draft tokens from "
               "device slots even while metadata still keeps a host shadow.";
        EXPECT_EQ(mock->publishMTPSpecStateBatchCount(), 0)
            << "GPU stochastic request batches must not mutate live state "
               "through the host-plan publisher.";
        EXPECT_EQ(mock->publishDeviceResidentMTPSpecStateCount(), 1);
        EXPECT_EQ(mock->adoptDeviceResidentHostStateCount(), 1);
        EXPECT_THAT(mock->publicationEvents(),
                    ElementsAre("device_outcome_publish",
                                "host_state_adopt",
                                "host_outcome_bridge"));
        EXPECT_EQ(mock->lastDeviceResidentPublicationRequest().request_count, 2);
        EXPECT_EQ(mock->lastDeviceResidentPublicationRequest().max_draft_tokens, 2);
    }

    TEST_F(Test__PrefillDecodeTransition, RequestBatchedStochasticContinuationPublishesDeviceOutcomes)
    {
        auto [runner, mock] =
            createSingleDeviceRequestBatchRunner(
                /*max_request_batch=*/2,
                /*mtp_draft_tokens=*/2,
                MTPVerifyMode::SpeculativeSampling);
        mock->setPrimaryDevice(DeviceId::cuda(0));
        mock->enableMainLogitsBatchRowsOnDevice();
        mock->enableStochasticDeviceSampling();
        mock->enableDeviceResidentMTPSpecStatePublication();

        SamplingParams sampling;
        sampling.temperature = 0.1f;
        sampling.top_k = 5;
        sampling.top_p = 1.0f;
        sampling.seed = 1234;
        runner->setSamplingParams(sampling);

        ASSERT_TRUE(runner->prefillBatch({{1, 2, 3}, {4, 5}}))
            << runner->lastError();

        GenerationBatchResult first = runner->decodeStepBatch(2);
        ASSERT_TRUE(first.error.empty()) << first.error;
        EXPECT_TRUE(runner->supportsDecodeStepBatch(2))
            << "Stochastic request batching should advertise the continuation "
               "only after per-request sampler state exists";

        GenerationBatchResult second = runner->decodeStepBatch(2);
        ASSERT_TRUE(second.error.empty()) << second.error;
        ASSERT_THAT(second.requests, SizeIs(2));
        EXPECT_THAT(second.requests[0].tokens,
                    ElementsAre(MockInferenceRunner::MTP_ARGMAX_TOKEN,
                                MockInferenceRunner::MTP_ARGMAX_TOKEN));
        EXPECT_THAT(second.requests[1].tokens,
                    ElementsAre(MockInferenceRunner::MTP_ARGMAX_TOKEN,
                                MockInferenceRunner::MTP_ARGMAX_TOKEN));
        EXPECT_EQ(mock->forwardMTPBatchAndSampleCount(), 1);
        EXPECT_EQ(mock->forwardMTPBatchFromLastDraftAndSampleCount(), 1);
        EXPECT_EQ(mock->forwardMTPBatchAndSampleToDeviceDraftSlotsCount(), 1);
        EXPECT_EQ(mock->lastMTPBatchDeviceDraftFirstSlot(), 0);
        EXPECT_EQ(mock->lastMTPBatchDeviceDraftSlotStride(), 2);
        EXPECT_EQ(mock->forwardMTPBatchFromLastDraftAndSampleToDeviceDraftSlotsCount(), 1);
        EXPECT_THAT(mock->lastChainedMTPBatchDeviceDraftFirstSlots(),
                    ElementsAre(1));
        EXPECT_THAT(mock->lastChainedMTPBatchDeviceDraftSlotStrides(),
                    ElementsAre(2));
        EXPECT_EQ(mock->forwardBatchCallCount(), 2)
            << "Stochastic request batching should amortize one verifier "
               "forward across both requests";
        EXPECT_EQ(mock->stageStochasticDraftTokensCount(), 0)
            << "Request-batched GPU stochastic sidecars should publish every "
               "draft depth directly into request-major device slots.";
        EXPECT_EQ(mock->verifyStochasticRequestBatchOutcomeCount(), 1)
            << "decodeStepBatch() should hand the complete stochastic request "
               "batch to the runner once, after all row staging is complete";
        EXPECT_THAT(mock->lastRequestBatchOutcomeRequestIds(), ElementsAre(0, 1));
        EXPECT_THAT(mock->lastRequestBatchOutcomeRowCounts(), ElementsAre(2, 2));
        EXPECT_THAT(mock->lastRequestBatchOutcomeFirstTargetSlots(),
                    ElementsAre(0, 3));
        EXPECT_THAT(mock->lastRequestBatchOutcomeFirstDraftSlots(),
                    ElementsAre(0, 2));
        EXPECT_THAT(mock->lastRequestBatchOutcomeBonusTargetSlots(),
                    ElementsAre(2, 5));
        EXPECT_THAT(mock->lastBatchOutcomeFirstTargetSlots(),
                    ElementsAre(0, 3));
        EXPECT_THAT(mock->lastBatchOutcomeFirstDraftSlots(),
                    ElementsAre(0, 2));
        EXPECT_THAT(mock->lastBatchOutcomeBonusTargetSlots(),
                    ElementsAre(2, 5));
        EXPECT_TRUE(mock->lastBatchOutcomeUsedVLLMProbabilityRejection());
        EXPECT_FALSE(mock->batchOutcomeUsedHostDraftTokens())
            << "Request-batched stochastic verification must consume the "
               "runner-owned draft sample slots";
        EXPECT_EQ(mock->lastBatchOutcomeInverseSampleSeed(), sampling.seed);
        EXPECT_EQ(mock->publishMTPSpecStateBatchCount(), 0)
            << "The compatibility host-plan publisher must not run after "
               "resident request-batch publication succeeds.";
        EXPECT_EQ(mock->publishDeviceResidentMTPSpecStateCount(), 1);
        EXPECT_EQ(mock->adoptDeviceResidentHostStateCount(), 1);
        EXPECT_THAT(mock->publicationEvents(),
                    ElementsAre("device_outcome_publish",
                                "host_state_adopt",
                                "host_outcome_bridge"));
        EXPECT_EQ(mock->lastDeviceResidentPublicationRequest().request_count, 2);
        EXPECT_EQ(mock->lastDeviceResidentPublicationRequest().max_draft_tokens, 3);
        EXPECT_THAT(mock->sequence_lengths(), ElementsAre(6, 5));

        GenerationBatchResult third = runner->decodeStepBatch(2);
        ASSERT_TRUE(third.error.empty()) << third.error;
        EXPECT_THAT(third.requests[0].tokens,
                    ElementsAre(MockInferenceRunner::DECODE_ARGMAX_TOKEN));
        EXPECT_THAT(third.requests[1].tokens,
                    ElementsAre(MockInferenceRunner::DECODE_ARGMAX_TOKEN));
        EXPECT_EQ(mock->forwardBatchCallCount(), 2)
            << "Ready stochastic bonus tokens should be consumed without "
               "another verifier forward";
    }

    /**
     * @brief Pins the scalar-equivalent RNG contract for batched stochastic MTP.
     *
     * Request batching must not key accept, residual, or bonus draws by compact
     * batch row.  Each descriptor handed to the device verifier uses the same
     * logical-token positions that a standalone request would use, otherwise
     * RB=2 can pass functional tests while silently changing acceptance rates.
     */
    TEST_F(Test__PrefillDecodeTransition, RequestBatchedStochasticDepthThreeUsesLogicalPositionDraws)
    {
        ScopedEnv enable_stats("LLAMINAR_PERF_STATS_JSON", "1");
        PerfStatsCollector::reset();
        auto [runner, mock] =
            createSingleDeviceRequestBatchRunner(
                /*max_request_batch=*/2,
                /*mtp_draft_tokens=*/3,
                MTPVerifyMode::SpeculativeSampling);
        mock->setPrimaryDevice(DeviceId::cuda(0));
        mock->enableMainLogitsBatchRowsOnDevice();
        mock->enableStochasticDeviceSampling();
        mock->enableDeviceResidentMTPSpecStatePublication();
        mock->forceStochasticRequestBatchReject(/*request_id=*/1,
                                                /*correction_token=*/4);

        SamplingParams sampling;
        sampling.temperature = 0.1f;
        sampling.top_k = 5;
        sampling.top_p = 1.0f;
        sampling.seed = 1234;
        runner->setSamplingParams(sampling);

        ASSERT_TRUE(runner->prefillBatch({{1, 2, 3}, {4, 5}}))
            << runner->lastError();
        GenerationBatchResult first = runner->decodeStepBatch(2);
        ASSERT_TRUE(first.error.empty()) << first.error;

        GenerationBatchResult second = runner->decodeStepBatch(2);
        ASSERT_TRUE(second.error.empty()) << second.error;

        ASSERT_THAT(mock->lastRequestBatchOutcomeRequestIds(), ElementsAre(0, 1));
        EXPECT_THAT(mock->lastRequestBatchOutcomeRowCounts(), ElementsAre(3, 3));
        EXPECT_THAT(mock->lastRequestBatchOutcomeFirstTargetSlots(),
                    ElementsAre(0, 4));
        EXPECT_THAT(mock->lastRequestBatchOutcomeFirstDraftSlots(),
                    ElementsAre(0, 3));
        EXPECT_THAT(mock->lastRequestBatchOutcomeBonusTargetSlots(),
                    ElementsAre(3, 7));
        EXPECT_EQ(mock->forwardMTPBatchAndSampleToDeviceDraftSlotsCount(), 1);
        EXPECT_EQ(mock->lastMTPBatchDeviceDraftFirstSlot(), 0);
        EXPECT_EQ(mock->lastMTPBatchDeviceDraftSlotStride(), 3);
        EXPECT_EQ(mock->forwardMTPBatchFromLastDraftAndSampleToDeviceDraftSlotsCount(), 2);
        EXPECT_THAT(mock->lastChainedMTPBatchDeviceDraftFirstSlots(),
                    ElementsAre(1, 2));
        EXPECT_THAT(mock->lastChainedMTPBatchDeviceDraftSlotStrides(),
                    ElementsAre(3, 3));
        EXPECT_EQ(mock->stageStochasticDraftTokensCount(), 0)
            << "Depth-three request batching should use strided device-slot "
               "sidecar stores instead of per-request host staging.";
        EXPECT_THAT(mock->lastRequestBatchOutcomeFirstTokens(),
                    ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN,
                                MockInferenceRunner::PREFILL_ARGMAX_TOKEN + 1));
        ASSERT_THAT(mock->lastRequestBatchOutcomeDraftTokens(), SizeIs(2));
        EXPECT_THAT(mock->lastRequestBatchOutcomeDraftTokens()[0],
                    ElementsAre(MockInferenceRunner::MTP_ARGMAX_TOKEN,
                                MockInferenceRunner::MTP_ARGMAX_TOKEN,
                                MockInferenceRunner::MTP_ARGMAX_TOKEN));
        EXPECT_THAT(mock->lastRequestBatchOutcomeDraftTokens()[1],
                    ElementsAre(MockInferenceRunner::MTP_ARGMAX_TOKEN,
                                MockInferenceRunner::MTP_ARGMAX_TOKEN,
                                MockInferenceRunner::MTP_ARGMAX_TOKEN));
        EXPECT_THAT(mock->lastRequestBatchOutcomeInverseSampleSeeds(),
                    ElementsAre(static_cast<uint64_t>(sampling.seed),
                                static_cast<uint64_t>(sampling.seed)));
        EXPECT_THAT(mock->lastRequestBatchOutcomeInverseSampleFirstPositions(),
                    ElementsAre(4, 3));
        ASSERT_EQ(mock->lastRequestBatchOutcomeDerivedThresholds().size(), 2u);
        EXPECT_FALSE(mock->lastRequestBatchOutcomeDerivedThresholds()[0])
            << "Production request batches must pass explicit, value-owned "
               "thresholds until the complete served stochastic sampler stream "
               "is device-owned and benchmark-proven.";
        EXPECT_FALSE(mock->lastRequestBatchOutcomeDerivedThresholds()[1])
            << "Production request batches must pass explicit, value-owned "
               "thresholds until the complete served stochastic sampler stream "
               "is device-owned and benchmark-proven.";

        auto threshold = [&](int logical_position, int purpose) {
            return mtpSeededVerifierThreshold(
                static_cast<uint64_t>(sampling.seed),
                logical_position,
                purpose);
        };

        ASSERT_THAT(mock->lastRequestBatchOutcomeAcceptThresholds(), SizeIs(2));
        ASSERT_THAT(mock->lastRequestBatchOutcomeResidualThresholds(), SizeIs(2));
        ASSERT_THAT(mock->lastRequestBatchOutcomeBonusThresholds(), SizeIs(2));

        const std::array<int, 3> request0_positions = {4, 5, 6};
        const std::array<int, 3> request1_positions = {3, 4, 5};
        for (int row = 0; row < 3; ++row)
        {
            EXPECT_NEAR(mock->lastRequestBatchOutcomeAcceptThresholds()[0][row],
                        threshold(request0_positions[static_cast<size_t>(row)],
                                  1 /* MTPSpecStochasticDrawPurpose::Accept */),
                        1e-7f);
            EXPECT_NEAR(mock->lastRequestBatchOutcomeResidualThresholds()[0][row],
                        threshold(request0_positions[static_cast<size_t>(row)],
                                  2 /* MTPSpecStochasticDrawPurpose::Residual */),
                        1e-7f);
            EXPECT_NEAR(mock->lastRequestBatchOutcomeAcceptThresholds()[1][row],
                        threshold(request1_positions[static_cast<size_t>(row)],
                                  1 /* MTPSpecStochasticDrawPurpose::Accept */),
                        1e-7f);
            EXPECT_NEAR(mock->lastRequestBatchOutcomeResidualThresholds()[1][row],
                        threshold(request1_positions[static_cast<size_t>(row)],
                                  2 /* MTPSpecStochasticDrawPurpose::Residual */),
                        1e-7f);
        }
        EXPECT_NEAR(mock->lastRequestBatchOutcomeBonusThresholds()[0],
                    threshold(7, 0 /* MTPSpecStochasticDrawPurpose::Sample */),
                    1e-7f);
        EXPECT_NEAR(mock->lastRequestBatchOutcomeBonusThresholds()[1],
                    threshold(6, 0 /* MTPSpecStochasticDrawPurpose::Sample */),
                    1e-7f);

        const auto records = PerfStatsCollector::snapshot({"mtp"});
        const PerfStatsCollector::Tags batch_tags{
            {"implementation", "request_batch_device_outcome"},
            {"request_batch", "true"}};
        const PerfStatRecord *physical_rows =
            findPerfRecordWithTags(records,
                                   PerfStatRecord::Kind::Counter,
                                   "stochastic_device_physical_verify_rows",
                                   batch_tags);
        ASSERT_NE(physical_rows, nullptr);
        EXPECT_DOUBLE_EQ(physical_rows->value, 6.0);
        const PerfStatRecord *semantic_rows =
            findPerfRecordWithTags(records,
                                   PerfStatRecord::Kind::Counter,
                                   "stochastic_device_semantic_verify_rows",
                                   batch_tags);
        ASSERT_NE(semantic_rows, nullptr);
        EXPECT_DOUBLE_EQ(semantic_rows->value, 4.0);
        const PerfStatRecord *post_reject_rows =
            findPerfRecordWithTags(records,
                                   PerfStatRecord::Kind::Counter,
                                   "stochastic_device_post_reject_rows",
                                   batch_tags);
        ASSERT_NE(post_reject_rows, nullptr);
        EXPECT_DOUBLE_EQ(post_reject_rows->value, 2.0)
            << "Only request 1 rejects at row zero; its two later verifier "
               "rows are physical work that the next Phase 10 optimization "
               "should try to avoid.";
        EXPECT_EQ(mock->publishMTPSpecStateBatchCount(), 0);
        EXPECT_EQ(mock->publishDeviceResidentMTPSpecStateCount(), 1);
        EXPECT_EQ(mock->adoptDeviceResidentHostStateCount(), 1);
        EXPECT_EQ(mock->lastDeviceResidentPublicationRequest().request_count, 2);
        EXPECT_EQ(mock->lastDeviceResidentPublicationRequest().max_draft_tokens, 4);
        PerfStatsCollector::reset();
    }

    TEST_F(Test__PrefillDecodeTransition, RequestBatchedStochasticUnseededKeepsExplicitThresholds)
    {
        auto [runner, mock] =
            createSingleDeviceRequestBatchRunner(
                /*max_request_batch=*/2,
                /*mtp_draft_tokens=*/2,
                MTPVerifyMode::SpeculativeSampling);
        mock->enableStochasticDeviceSampling();

        SamplingParams sampling;
        sampling.temperature = 0.1f;
        sampling.top_k = 5;
        sampling.top_p = 1.0f;
        sampling.seed = 0;
        runner->setSamplingParams(sampling);

        ASSERT_TRUE(runner->prefillBatch({{1, 2, 3}, {4, 5}}))
            << runner->lastError();
        GenerationBatchResult first = runner->decodeStepBatch(2);
        ASSERT_TRUE(first.error.empty()) << first.error;

        GenerationBatchResult second = runner->decodeStepBatch(2);
        ASSERT_TRUE(second.error.empty()) << second.error;

        ASSERT_EQ(mock->lastRequestBatchOutcomeDerivedThresholds().size(), 2u);
        EXPECT_FALSE(mock->lastRequestBatchOutcomeDerivedThresholds()[0])
            << "Unseeded stochastic verification owns a request-local sampler "
               "stream, so thresholds must be explicit values rather than "
               "derived from a seed.";
        EXPECT_FALSE(mock->lastRequestBatchOutcomeDerivedThresholds()[1])
            << "Unseeded stochastic verification owns a request-local sampler "
               "stream, so thresholds must be explicit values rather than "
               "derived from a seed.";
        ASSERT_THAT(mock->lastRequestBatchOutcomeAcceptThresholds(), SizeIs(2));
        ASSERT_THAT(mock->lastRequestBatchOutcomeResidualThresholds(), SizeIs(2));
        EXPECT_THAT(mock->lastRequestBatchOutcomeAcceptThresholds()[0], SizeIs(2));
        EXPECT_THAT(mock->lastRequestBatchOutcomeAcceptThresholds()[1], SizeIs(2));
        EXPECT_THAT(mock->lastRequestBatchOutcomeResidualThresholds()[0], SizeIs(2));
        EXPECT_THAT(mock->lastRequestBatchOutcomeResidualThresholds()[1], SizeIs(2));
    }

    TEST_F(Test__PrefillDecodeTransition, RequestBatchedStochasticMixedReadyAndRejectStaysLockstep)
    {
        auto [runner, mock] =
            createSingleDeviceRequestBatchRunner(
                /*max_request_batch=*/2,
                /*mtp_draft_tokens=*/1,
                MTPVerifyMode::SpeculativeSampling);
        mock->enableStochasticDeviceSampling();
        mock->forceStochasticRequestBatchReject(/*request_id=*/1,
                                                /*correction_token=*/4);

        SamplingParams sampling;
        sampling.temperature = 0.1f;
        sampling.top_k = 5;
        sampling.top_p = 1.0f;
        sampling.seed = 1234;
        runner->setSamplingParams(sampling);

        ASSERT_TRUE(runner->prefillBatch({{1, 2, 3}, {4, 5}}))
            << runner->lastError();

        GenerationBatchResult first = runner->decodeStepBatch(2);
        ASSERT_TRUE(first.error.empty()) << first.error;

        GenerationBatchResult second = runner->decodeStepBatch(2);
        ASSERT_TRUE(second.error.empty()) << second.error;
        ASSERT_THAT(second.requests, SizeIs(2));
        EXPECT_THAT(second.requests[0].tokens,
                    ElementsAre(MockInferenceRunner::MTP_ARGMAX_TOKEN,
                                MockInferenceRunner::DECODE_ARGMAX_TOKEN))
            << "The all-accepted lane should emit its bonus-ready token inline "
               "instead of leaving a terminal-logit state beside a rejected lane";
        EXPECT_THAT(second.requests[1].tokens, ElementsAre(4));
        EXPECT_TRUE(runner->supportsDecodeStepBatch(2))
            << "Inlining the ready token should keep every live request in the "
               "same verifier-continuation state class";

        GenerationBatchResult third = runner->decodeStepBatch(2);
        ASSERT_TRUE(third.error.empty()) << third.error;
        ASSERT_THAT(third.requests, SizeIs(2));
        EXPECT_THAT(third.requests[0].tokens,
                    ElementsAre(MockInferenceRunner::MTP_ARGMAX_TOKEN,
                                MockInferenceRunner::DECODE_ARGMAX_TOKEN));
        EXPECT_THAT(third.requests[1].tokens, ElementsAre(4));
        EXPECT_EQ(mock->forwardBatchCallCount(), 3)
            << "The next lockstep step should run one verifier batch rather "
               "than failing on mixed ready/verifier ownership";
    }

    TEST_F(Test__PrefillDecodeTransition, PrefillBatchRejectsUndersizedRunnerCapacityBeforeForward)
    {
        auto [runner, mock] = createSingleDeviceRequestBatchRunner(/*max_request_batch=*/3);
        mock->setBatchCapacity(1);

        EXPECT_FALSE(runner->supportsPrefillBatch(2));
        EXPECT_FALSE(runner->prefillBatch({{1, 2}, {3, 4}}));
        EXPECT_THAT(runner->lastError(), HasSubstr("batch capacity"));
        EXPECT_EQ(mock->forwardBatchCallCount(), 0);
        EXPECT_EQ(mock->forwardCallCount(), 0);
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
     * @brief Forced policy tokens must keep the MTP shifted KV stream aligned.
     *
     * Thinking-budget exhaustion injects a model-authored stop-thinking phrase
     * through forceDecodeToken().  The first forced token often consumes ready
     * terminal logits without a main forward; later forced tokens forward the
     * previous injected token.  In both cases the emitted token needs the same
     * shifted-MTP sidecar row that ordinary decode publishes before forwarding
     * an accepted token, otherwise the next MTP step sees main KV/GDN state
     * ahead of the shifted sidecar cache.
     */
    TEST_F(Test__PrefillDecodeTransition, ForceDecodeTokenMaintainsMTPShiftedCacheAcrossInjectedSequence)
    {
        auto [runner, mock] = createRunner(/*mtp_enabled=*/true, /*mtp_accept=*/true);

        std::vector<int32_t> prompt = {1, 2, 3, 4, 5};
        ASSERT_TRUE(runner->prefill(prompt));
        EXPECT_EQ(mock->forwardCallCount(), 1);

        GenerationResult first_forced = runner->forceDecodeToken(2);
        ASSERT_TRUE(first_forced.success()) << first_forced.error;
        EXPECT_THAT(first_forced.tokens, ElementsAre(2));
        EXPECT_EQ(mock->forwardCallCount(), 1)
            << "Ready logits should let the first forced token avoid a main forward.";
        EXPECT_EQ(mock->commitMTPShiftedCount(), 1);
        EXPECT_EQ(mock->sequentialCommitMTPShiftedCount(), 1);
        EXPECT_EQ(mock->lastCommitMTPPositionOffsetOverride(), 5);
        EXPECT_THAT(mock->lastCommitMTPTokens(), ElementsAre(2));

        GenerationResult second_forced = runner->forceDecodeToken(4);
        ASSERT_TRUE(second_forced.success()) << second_forced.error;
        EXPECT_THAT(second_forced.tokens, ElementsAre(4));
        EXPECT_EQ(mock->forwardCallCount(), 2)
            << "The second forced token must first append the previous one.";
        EXPECT_THAT(mock->lastForwardTokens(), ElementsAre(2));
        EXPECT_EQ(mock->commitMTPShiftedCount(), 2);
        EXPECT_EQ(mock->sequentialCommitMTPShiftedCount(), 2);
        EXPECT_EQ(mock->lastCommitMTPPositionOffsetOverride(), 6);
        EXPECT_THAT(mock->lastCommitMTPTokens(), ElementsAre(4));
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

    TEST_F(Test__PrefillDecodeTransition, MTPGreedyPenaltiesUseSequentialVerifier)
    {
        PerfStatsCollector::reset();

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
        ASSERT_TRUE(step1.success()) << step1.error;
        EXPECT_THAT(step1.tokens,
                    ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN,
                                MockInferenceRunner::MTP_ARGMAX_TOKEN));
        EXPECT_EQ(mock->forwardMTPCount(), 1);
        EXPECT_EQ(mock->applyMainPenaltiesCount(), 3)
            << "penalty-greedy catch-up must mutate main logits after each "
               "forwarded verifier token, using a branch-local sampler "
               "history that includes the token just replayed";
        EXPECT_EQ(mock->applyMTPPenaltiesCount(), 1);
        EXPECT_EQ(mock->applyAllPositionPenaltiesCount(), 0);
        EXPECT_EQ(mock->setAllPositionCount(), 0)
            << "Greedy penalties must stay on the decode-equivalent verifier "
               "until the all-position path applies row-local penalty history.";

        const auto probe = runner->prefixStateProbe();
        EXPECT_FALSE(probe.mtp_bypassed);
        EXPECT_EQ(probe.mtp_bypasses, 0u);
        EXPECT_EQ(probe.mtp_draft_steps, 1u);
        EXPECT_EQ(probe.mtp_verifier_runs, 1u);
        EXPECT_EQ(probe.mtp_accepted_tokens, 1u);
        EXPECT_EQ(probe.mtp_transaction_validation_failures, 0u);

        PerfStatsCollector::reset();
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
            EXPECT_EQ(mock->captureCheckpointCount(), 1)
                << "a main-state-preserving sidecar should not export the "
                   "post-sidecar checkpoint that decode-equivalent replay "
                   "would immediately discard";
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
            EXPECT_EQ(skipped_restore->tags.at("discarded_sidecar_checkpoint"), "false");

            const PerfStatRecord *post_sidecar_capture =
                findPerfRecord(records,
                               PerfStatRecord::Kind::Timer,
                               "capture_post_sidecar_prefix_state");
            EXPECT_EQ(post_sidecar_capture, nullptr);
            const PerfStatRecord *skipped_post_sidecar_capture =
                findPerfRecord(records,
                               PerfStatRecord::Kind::Counter,
                               "post_sidecar_checkpoint_skipped_sidecar_preserved");
            ASSERT_NE(skipped_post_sidecar_capture, nullptr);
            EXPECT_DOUBLE_EQ(skipped_post_sidecar_capture->value, 1.0);
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
        depth_policy.use_generated_policy = false;

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

    TEST_F(Test__PrefillDecodeTransition, DynamicMTPDepthZeroBypassesDraftVerifierWithoutDisablingMTP)
    {
        MTPDepthPolicyConfig depth_policy;
        depth_policy.mode = MTPDepthPolicyMode::Dynamic;
        depth_policy.min_depth = 0;
        depth_policy.max_depth = 3;
        depth_policy.initial_depth = 1;
        depth_policy.window_size = 1;
        depth_policy.min_samples = 1;
        depth_policy.cooldown_steps = 2;
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
            MockInferenceRunner::DECODE_ARGMAX_TOKEN});

        ASSERT_TRUE(runner->prefill({1, 2, 3, 4, 5}));

        GenerationResult step1 = runner->decodeStep();
        ASSERT_TRUE(step1.success()) << step1.error;
        EXPECT_EQ(mock->forwardMTPCount(), 1);
        EXPECT_EQ(mock->forwardMTPFromLastDraftCount(), 0);
        auto probe = runner->prefixStateProbe();
        EXPECT_EQ(probe.mtp_current_depth, 0);
        EXPECT_EQ(probe.mtp_min_depth, 0);
        EXPECT_EQ(probe.mtp_depth_policy_demotions, 1u);
        EXPECT_EQ(probe.mtp_depth_policy_updates, 1u);
        EXPECT_EQ(probe.mtp_request.last_depth_policy_reason, "demote_zero_accept_rate");

        const int forward_mtp_count_after_demote = mock->forwardMTPCount();
        const int chained_mtp_count_after_demote = mock->forwardMTPFromLastDraftCount();
        const int shifted_commit_count_after_demote = mock->commitMTPShiftedCount();
        GenerationResult step2 = runner->decodeStep();
        ASSERT_TRUE(step2.success()) << step2.error;

        EXPECT_EQ(mock->forwardMTPCount(), forward_mtp_count_after_demote)
            << "depth-zero adaptive bypass must not run MTP draft sidecars";
        EXPECT_EQ(mock->forwardMTPFromLastDraftCount(), chained_mtp_count_after_demote)
            << "depth-zero adaptive bypass must not run chained MTP draft sidecars";
        EXPECT_EQ(mock->commitMTPShiftedCount(), shifted_commit_count_after_demote + 1)
            << "depth-zero adaptive bypass must still maintain shifted MTP cache for later probes";
        probe = runner->prefixStateProbe();
        EXPECT_TRUE(probe.mtp_request.enabled);
        EXPECT_FALSE(probe.mtp_request.bypassed)
            << "adaptive depth-zero is a per-step policy action, not an unsupported-feature bypass";
        EXPECT_EQ(probe.mtp_bypasses, 0u);
        EXPECT_EQ(probe.mtp_current_depth, 0);
        EXPECT_EQ(probe.mtp_request.last_depth_policy_reason, "depth_zero_bypass");
    }

    TEST_F(Test__PrefillDecodeTransition, DynamicMTPDepthResetsAcrossClearCachePrefillCycles)
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
        EXPECT_EQ(mock->forwardMTPFromLastDraftCount(), 4)
            << "clearCache() is a request boundary, so adaptive MTP must "
               "restart from the configured initial depth instead of carrying "
               "a learned depth into the next request";

        probe = runner->prefixStateProbe();
        EXPECT_EQ(probe.mtp_current_depth, 2);
        EXPECT_EQ(probe.mtp_depth_policy_demotions, 1u);
        EXPECT_EQ(probe.mtp_depth_policy_updates, 1u);
    }

    TEST_F(Test__PrefillDecodeTransition, DynamicMTPDepthHoldsBeforeDeepestWithoutGeneratedPolicy)
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
        depth_policy.use_generated_policy = false;

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
            << "second step should use depth 2 before evaluating the deepest lane";
        probe = runner->prefixStateProbe();
        EXPECT_EQ(probe.mtp_current_depth, 2)
            << "depth 3 is expensive enough that dynamic mode only enters it through generated policy evidence";
        EXPECT_EQ(probe.mtp_depth_policy_promotions, 1u);
        EXPECT_EQ(probe.mtp_depth_policy_updates, 1u);
        EXPECT_EQ(probe.mtp_request.last_depth_policy_reason, "hold");
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
            mock->enableMTPShiftedRowReuseFromSidecar();
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
            EXPECT_EQ(mock->setRowIndexedAllPositionCount(), 2)
                << "row-indexed verifier logits should be enabled and disabled around the verifier forward";
            EXPECT_EQ(mock->setMTPSpecVerifierPlanCount(), 1)
                << "the verifier row plan must be installed before the row-indexed forward";
            EXPECT_EQ(mock->clearMTPSpecVerifierPlanCount(), 1)
                << "the scoped verifier row plan must be cleared after the forward";
            EXPECT_FALSE(mock->mtpSpecVerifierPlanInstalled())
                << "stale verifier metadata must not survive a decode step";
            EXPECT_THAT(mock->lastMTPSpecVerifierRows(), ElementsAre(0, 1, 2));
            EXPECT_THAT(mock->lastMTPSpecVerifierTokens(),
                        ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN,
                                    MockInferenceRunner::MTP_ARGMAX_TOKEN,
                                    MockInferenceRunner::MTP_ARGMAX_TOKEN));
            EXPECT_EQ(mock->sampleAllPositionLogitsBatchedCount(), 1);
            EXPECT_EQ(mock->lastSampleAllPositionStartRow(), 0);
            EXPECT_EQ(mock->lastSampleAllPositionRowCount(), 3);
            EXPECT_EQ(mock->publishMTPSpecStateCount(), 1);
            EXPECT_EQ(mock->publishMTPSpecStateBatchCount(), 1);
            EXPECT_EQ(mock->lastPublishedMTPSpecBatch().request_count, 1);
            EXPECT_EQ(mock->sequentialCommitMTPShiftedCount(), 0)
                << "a main-state-preserving sidecar already appended the first shifted MTP row";
            EXPECT_EQ(mock->commitMTPShiftedCount(), 1)
                << "accepted verifier hidden rows should fill the remaining shifted prefix without sequential verifier replay";
            EXPECT_EQ(mock->lastCommitMTPAlreadyAppended(), 1);
            EXPECT_EQ(mock->lastCommitMTPMainForwardTokenCount(), 3);
            EXPECT_THAT(mock->lastCommitMTPTokens(),
                        ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN,
                                    MockInferenceRunner::MTP_ARGMAX_TOKEN,
                                    MockInferenceRunner::MTP_ARGMAX_TOKEN));
            EXPECT_EQ(mock->restoreCount(), 0);
            EXPECT_EQ(mock->captureCheckpointCount(), 1)
                << "all-position publication does not consume the old post-sidecar checkpoint";
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
            const PerfStatRecord *reused_first_shifted =
                findPerfRecord(records,
                               PerfStatRecord::Kind::Counter,
                               "all_position_initial_shifted_reused_sidecar_rows");
            ASSERT_NE(reused_first_shifted, nullptr);
            EXPECT_DOUBLE_EQ(reused_first_shifted->value, 1.0);

            const PerfStatRecord *publication_runs =
                findPerfRecordWithTags(records,
                                       PerfStatRecord::Kind::Counter,
                                       "all_position_state_publication_verifier_runs",
                                       {{"verifier_rows", "3"},
                                        {"correction_replay_tokens", "0"},
                                        {"accepted_state_count", "3"},
                                        {"target_cached_tokens", "8"}});
            ASSERT_NE(publication_runs, nullptr);

            const PerfStatRecord *post_sidecar_capture =
                findPerfRecord(records,
                               PerfStatRecord::Kind::Timer,
                               "capture_post_sidecar_prefix_state");
            EXPECT_EQ(post_sidecar_capture, nullptr);

            const PerfStatRecord *skipped_post_sidecar_capture =
                findPerfRecord(records,
                               PerfStatRecord::Kind::Counter,
                               "post_sidecar_checkpoint_skipped_all_position_publication");
            ASSERT_NE(skipped_post_sidecar_capture, nullptr);
            EXPECT_DOUBLE_EQ(skipped_post_sidecar_capture->value, 1.0);
        }
        std::filesystem::remove(export_path);
        PerfStatsCollector::reset();
    }

    TEST_F(Test__PrefillDecodeTransition, GreedyGPUAllPositionSpecPublicationScopesVerifierSyncDeferral)
    {
        auto [runner, mock] = createRunner(
            /*mtp_enabled=*/true,
            /*mtp_accept=*/true,
            /*mtp_unsupported_reason=*/{},
            /*mpi_ctx=*/nullptr,
            /*mtp_token_coordination=*/true,
            /*hide_local_logits=*/false,
            DeviceId::cuda(0),
            /*mtp_draft_tokens=*/2,
            /*chained_mtp_support=*/true);
        mock->enableMTPSidecarPreservesMainState();
        mock->requireMTPDecodeEquivalentReplay();
        mock->enableMTPSpecStatePublication();
        mock->setVerifierAcceptedPrefixScript({2});

        ASSERT_TRUE(runner->prefill({1, 2, 3, 4, 5}));
        GenerationResult step1 = runner->decodeStep();
        ASSERT_TRUE(step1.success()) << step1.error;

        EXPECT_EQ(mock->setAllPositionCount(), 2);
        EXPECT_EQ(mock->sampleAllPositionLogitsBatchedCount(), 1);
        EXPECT_EQ(mock->publishMTPSpecStateCount(), 1);
        EXPECT_EQ(mock->allPositionVerifierSyncDeferralEnableCount(), 1);
        EXPECT_EQ(mock->allPositionVerifierSyncDeferralDisableCount(), 1);
        EXPECT_EQ(mock->allPositionVerifierSyncDeferralSetCount(), 2);
        EXPECT_FALSE(mock->allPositionVerifierSyncDeferralEnabled());
    }

    TEST_F(Test__PrefillDecodeTransition, PenaltyGreedyGPUUsesRowLocalAllPositionVerifier)
    {
        const std::filesystem::path export_path =
            std::filesystem::temp_directory_path() /
            "llaminar_mtp_penalty_greedy_all_position_unit.json";
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
                /*chained_mtp_support=*/true);
            mock->enableMTPSidecarPreservesMainState();
            mock->enableMTPShiftedRowReuseFromSidecar();
            mock->enableMTPSpecStatePublication();
            mock->setVerifierAcceptedPrefixScript({2});

            SamplingParams sampling;
            sampling.temperature = 0.0f;
            sampling.presence_penalty = 0.25f;
            runner->setSamplingParams(sampling);

            ASSERT_TRUE(runner->prefill({1, 2, 3, 4, 5}));
            const int forward_count_after_prefill = mock->forwardCallCount();

            GenerationResult step = runner->decodeStep();
            ASSERT_TRUE(step.success()) << step.error;
            EXPECT_THAT(step.tokens,
                        ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN,
                                    MockInferenceRunner::MTP_ARGMAX_TOKEN,
                                    MockInferenceRunner::MTP_ARGMAX_TOKEN));

            EXPECT_EQ(mock->applyMainPenaltiesCount(), 1)
                << "the first temperature-zero target token is still sampled "
                   "from penalty-mutated logits";
            EXPECT_EQ(mock->applyMTPPenaltiesCount(), 2)
                << "each greedy MTP sidecar draft should see the sampler "
                   "history built by prior accepted tokens";
            EXPECT_EQ(mock->applyAllPositionPenaltiesCount(), 3)
                << "two verifier comparison rows plus the bonus-ready row "
                   "need row-local speculative penalty history";
            EXPECT_EQ(mock->setAllPositionCount(), 2);
            EXPECT_EQ(mock->sampleAllPositionLogitsBatchedCount(), 1)
                << "penalty-greedy should use the compact all-position reducer "
                   "after row-local penalty application";
            EXPECT_EQ(mock->verifyGreedyAllPositionBatchOutcomeCount(), 1);
            EXPECT_EQ(mock->publishMTPSpecStateCount(), 1);
            EXPECT_EQ(mock->forwardCallCount(), forward_count_after_prefill + 1)
                << "the promoted penalty-greedy lane must not fall back to "
                   "stepwise decode-equivalent replay";

            const auto records = PerfStatsCollector::snapshot({"mtp"});
            const PerfStatRecord *row_penalties =
                findPerfRecord(records,
                               PerfStatRecord::Kind::Counter,
                               "greedy_vllm_penalty_rows_preapplied");
            ASSERT_NE(row_penalties, nullptr);
            EXPECT_DOUBLE_EQ(row_penalties->value, 3.0);
        }
        std::filesystem::remove(export_path);
        PerfStatsCollector::reset();
    }

    TEST_F(Test__PrefillDecodeTransition, GreedyGPUAllPositionUsesDeviceDraftSlotsForVerifierInput)
    {
        const std::filesystem::path export_path =
            std::filesystem::temp_directory_path() /
            "llaminar_mtp_greedy_device_draft_slot_unit.json";
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
            mock->enableStochasticDeviceSampling();
            mock->enableMTPDeviceDraftTokenInput();
            mock->enableMTPSidecarPreservesMainState();
            mock->enableMTPShiftedRowReuseFromSidecar();
            mock->enableDeviceResidentMTPSpecStatePublication();
            mock->setVerifierAcceptedPrefixScript({2});

            ASSERT_TRUE(runner->prefill({1, 2, 3, 4, 5}));
            GenerationResult step1 = runner->decodeStep();
            ASSERT_TRUE(step1.success()) << step1.error;
            EXPECT_THAT(step1.tokens,
                        ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN,
                                    MockInferenceRunner::MTP_ARGMAX_TOKEN,
                                    MockInferenceRunner::MTP_ARGMAX_TOKEN));

            EXPECT_EQ(mock->sampleMTPLogitsToDeviceDraftSlotCount(), 2)
                << "Both fixed-depth greedy sidecars should publish their "
                   "sampled draft tokens into runner-owned device slots.";
            EXPECT_EQ(mock->sampleMainLogitsToDeviceTargetSlotCount(), 1)
                << "The first fixed-depth greedy target token should stay in "
                   "the runner-owned device target slot until verifier summary.";
            EXPECT_EQ(mock->lastSampleMainLogitsDeviceTargetSlot(), 0);
            EXPECT_EQ(mock->lastSampleMTPLogitsDeviceDraftSlot(), 1);
            EXPECT_EQ(mock->forwardMTPFromDeviceTargetForDeviceSamplingCount(), 1)
                << "The first sidecar should consume the deferred target slot "
                   "rather than a host-token condition.";
            EXPECT_EQ(mock->forwardMTPForDeviceSamplingCount(), 0)
                << "The opted-in device-token path must not fall back to the "
                   "host-token sidecar entry point.";
            EXPECT_EQ(mock->sampleMainLogitsCount(), 0)
                << "The deferred first-token path must not call the legacy "
                   "host-visible greedy main sampler.";
            EXPECT_EQ(mock->sampleMTPLogitsCount(), 0)
                << "The opted-in device-slot path must not quietly fall back "
                   "to the legacy host-visible greedy MTP sampler.";
            EXPECT_EQ(mock->prepareMTPVerifierInputTokensOnDeviceCount(), 1)
                << "The all-position verifier should consume the same "
                   "device-resident draft slots produced by sidecar sampling.";
            EXPECT_EQ(mock->prepareMTPVerifierInputTokensDeviceFirstCount(), 1);
            EXPECT_EQ(mock->lastPrepareMTPVerifierFirstDraftSlot(), 0);
            EXPECT_EQ(mock->lastPrepareMTPVerifierDraftTokenCount(), 2);
            EXPECT_EQ(mock->lastPrepareMTPVerifierTotalTokens(), 3);
            EXPECT_EQ(mock->forwardWithDeviceTokenIdsCount(), 1);
            EXPECT_EQ(mock->lastForwardDeviceTokenIds(),
                      mock->deviceVerifierInputTokens().data());
            EXPECT_EQ(mock->lastForwardDeviceTokenSeqLen(), 3);
            EXPECT_EQ(mock->publishDeviceResidentMTPSpecStateCount(), 1)
                << "Greedy all-position GPU verification should publish "
                   "accepted state from the resident compact outcome when "
                   "the runner advertises that capability.";
            EXPECT_EQ(mock->publishMTPSpecStateBatchCount(), 0)
                << "The host-plan publisher must not mutate live state after "
                   "device-resident greedy publication succeeds.";
            EXPECT_EQ(mock->adoptDeviceResidentHostStateCount(), 1);
            EXPECT_THAT(mock->publicationEvents(),
                        ElementsAre("device_outcome_publish",
                                    "host_outcome_bridge",
                                    "host_state_adopt"));

            const auto &device_verifier_tokens =
                mock->deviceVerifierInputTokens();
            EXPECT_THAT(
                std::vector<int32_t>(
                    device_verifier_tokens.begin(),
                    device_verifier_tokens.begin() + 4),
                ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN,
                            MockInferenceRunner::MTP_ARGMAX_TOKEN,
                            MockInferenceRunner::MTP_ARGMAX_TOKEN,
                            -1));

            const auto records = PerfStatsCollector::snapshot({"mtp"});
            EXPECT_NE(findPerfRecordWithTags(
                          records,
                          PerfStatRecord::Kind::Counter,
                          "mtp_token_greedy_device_slot_samples",
                          {{"draft_idx", "0"}}),
                      nullptr);
            EXPECT_NE(findPerfRecordWithTags(
                          records,
                          PerfStatRecord::Kind::Counter,
                          "mtp_token_greedy_device_slot_samples",
                          {{"draft_idx", "1"}}),
                      nullptr);
            EXPECT_NE(findPerfRecord(records,
                                     PerfStatRecord::Kind::Counter,
                                     "first_token_greedy_deferred_host_reads"),
                      nullptr);
            EXPECT_NE(findPerfRecordWithTags(
                          records,
                          PerfStatRecord::Kind::Counter,
                          "all_position_verifier_device_token_inputs",
                          {{"total_tokens", "3"}}),
                      nullptr);
            EXPECT_EQ(findPerfRecord(records,
                                     PerfStatRecord::Kind::Counter,
                                     "mtp_token_greedy_device_slot_failures"),
                      nullptr);
        }
        std::filesystem::remove(export_path);
        PerfStatsCollector::reset();
    }

    TEST_F(Test__PrefillDecodeTransition, GroupedGreedyDeviceResidentPublicationAvoidsReplayAndCheckpoint)
    {
        const std::filesystem::path export_path =
            std::filesystem::temp_directory_path() /
            "llaminar_mtp_grouped_greedy_device_publication_unit.json";
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
            mock->enableGroupedOutcomeDeviceResidentPublication(/*rows=*/4);
            mock->enableStochasticDeviceSampling();
            mock->enableDeviceResidentMTPSpecStatePublication();
            mock->hideMTPSpecStatePublicationFromPolicy();
            mock->enableMTPSidecarPreservesMainState();
            mock->enableMTPShiftedRowReuseFromSidecar();
            mock->enableMTPSidecarLogitsStreamHandoff();
            mock->enableMTPDeviceDraftTokenInput();
            mock->setVerifierAcceptedPrefixScript({2});

            ASSERT_TRUE(runner->prefill({1, 2, 3, 4, 5}));
            const int forward_count_after_prefill = mock->forwardCallCount();

            GenerationResult step = runner->decodeStep();
            ASSERT_TRUE(step.success()) << step.error;
            EXPECT_THAT(step.tokens,
                        ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN,
                                    MockInferenceRunner::MTP_ARGMAX_TOKEN,
                                    MockInferenceRunner::MTP_ARGMAX_TOKEN));

            EXPECT_EQ(mock->setMTPSpecVerifierPlanCount(), 1);
            EXPECT_EQ(mock->setRowIndexedAllPositionCount(), 2);
            EXPECT_EQ(mock->setAllPositionCount(), 2);
            EXPECT_EQ(mock->prepareMTPVerifierInputTokensOnDeviceCount(), 1);
            EXPECT_EQ(mock->prepareMTPVerifierInputTokensDeviceFirstCount(), 1);
            EXPECT_EQ(mock->verifyGreedyAllPositionBatchOutcomeCount(), 1);
            EXPECT_EQ(mock->publishMTPSpecStateCount(), 0);
            EXPECT_EQ(mock->publishMTPSpecStateBatchCount(), 0);
            EXPECT_EQ(mock->publishDeviceResidentMTPSpecStateCount(), 1);
            EXPECT_EQ(mock->adoptDeviceResidentHostStateCount(), 1);
            EXPECT_EQ(mock->restoreCount(), 0)
                << "grouped greedy resident publication must not restore a "
                   "payload checkpoint on the success path";
            EXPECT_EQ(mock->captureCheckpointCount(), 0)
                << "grouped greedy resident publication should carry only a "
                   "logical base stamp, not export hybrid payloads";
            EXPECT_EQ(mock->sequentialCommitMTPShiftedCount(), 0)
                << "accepted shifted rows should be published from compact "
                   "device metadata, not replayed serially";
            EXPECT_EQ(mock->forwardCallCount(), forward_count_after_prefill + 1)
                << "one grouped verifier forward; no catch-up forwards";
            EXPECT_EQ(mock->forwardMTPFromDeviceDraftForDeviceSamplingCount(), 1)
                << "The fused chained sidecar should consume the previous "
                   "draft from the runner-owned device slot instead of a host "
                   "shadow token.";
            EXPECT_THAT(mock->publicationEvents(),
                        ElementsAre("device_outcome_publish",
                                    "host_outcome_bridge",
                                    "host_state_adopt"));
            EXPECT_EQ(mock->residentSidecarCountAtLastHostBridge(), 0)
                << "Grouped greedy response materialization should not wait "
                   "behind speculative next-step sidecar prelaunch work.";

            const auto records = PerfStatsCollector::snapshot({"mtp"});
            EXPECT_EQ(findPerfRecord(records,
                                     PerfStatRecord::Kind::Counter,
                                     "decode_equivalent_sequential_verifier_runs"),
                      nullptr);
            EXPECT_EQ(findPerfRecord(records,
                                     PerfStatRecord::Kind::Timer,
                                     "capture_live_prefix_state"),
                      nullptr);
            EXPECT_EQ(findPerfRecord(records,
                                     PerfStatRecord::Kind::Timer,
                                     "decode_equivalent_catchup_forward_one"),
                      nullptr);
            EXPECT_NE(findPerfRecordWithTags(
                          records,
                          PerfStatRecord::Kind::Counter,
                          "mtp_token_greedy_device_slot_deferred_host_reads",
                          {{"draft_idx", "0"},
                           {"path", "fused_first_sidecar"}}),
                      nullptr);
            EXPECT_NE(findPerfRecordWithTags(
                          records,
                          PerfStatRecord::Kind::Counter,
                          "mtp_token_greedy_device_slot_deferred_host_reads",
                          {{"draft_idx", "1"},
                           {"path", "fused_chained_sidecar"}}),
                      nullptr);
            ASSERT_NE(findPerfRecord(records,
                                     PerfStatRecord::Kind::Timer,
                                     "grouped_outcome_greedy_verifier_forward"),
                      nullptr);
            ASSERT_NE(findPerfRecordWithTags(
                          records,
                          PerfStatRecord::Kind::Counter,
                          "grouped_decode_equivalent_greedy_verifier_runs",
                          {{"verifier_forward_tokens", "3"},
                           {"verifier_rows", "3"},
                           {"replay_forward_tokens", "0"},
                           {"shifted_commits", "0"},
                           {"state_publication", "device_resident"}}),
                      nullptr);
            ASSERT_NE(findPerfRecordWithTags(
                          records,
                          PerfStatRecord::Kind::Counter,
                          "acceptance_trace",
                          {{"verifier_path",
                            "grouped_decode_equivalent_greedy"},
                           {"catchup_implementation",
                            "device_batch_outcome_device_resident_publication"},
                           {"policy_path",
                            "grouped_outcome_device_resident_publication"},
                           {"decode_equivalent_replay_required", "false"}}),
                      nullptr);
            ASSERT_NE(findPerfRecordWithTags(
                          records,
                          PerfStatRecord::Kind::Counter,
                          "live_prefix_checkpoint_skipped_direct_publication",
                          {{"verifier_path", "grouped_decode_equivalent_outcome"}}),
                      nullptr);
            ASSERT_NE(findPerfRecordWithTags(
                          records,
                          PerfStatRecord::Kind::Counter,
                          "stochastic_first_sidecar_prelaunches",
                          {{"path", "grouped_outcome_device_resident_publication"},
                           {"resident_state_kind", "device_publication_mailbox"},
                           {"prelaunch_timing", "post_bridge"},
                           {"sampling", "greedy"}}),
                      nullptr);
        }
        std::filesystem::remove(export_path);
        PerfStatsCollector::reset();
    }

    TEST_F(Test__PrefillDecodeTransition, GroupedGreedyReusesResidentPrelaunchOnNextStep)
    {
        const std::filesystem::path export_path =
            std::filesystem::temp_directory_path() /
            "llaminar_mtp_grouped_greedy_prelaunch_reuse_unit.json";
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
            mock->enableGroupedOutcomeDeviceResidentPublication(/*rows=*/4);
            mock->enableStochasticDeviceSampling();
            mock->enableDeviceResidentMTPSpecStatePublication();
            mock->hideMTPSpecStatePublicationFromPolicy();
            mock->enableMTPSidecarPreservesMainState();
            mock->enableMTPShiftedRowReuseFromSidecar();
            mock->enableMTPSidecarLogitsStreamHandoff();
            mock->enableMTPDeviceDraftTokenInput();
            mock->setVerifierAcceptedPrefixScript({2, 2});

            ASSERT_TRUE(runner->prefill({1, 2, 3, 4, 5}));
            GenerationResult step1 = runner->decodeStep();
            ASSERT_TRUE(step1.success()) << step1.error;
            ASSERT_THAT(step1.tokens, SizeIs(3));
            EXPECT_EQ(mock->residentSidecarCountAtLastHostBridge(), 0);

            const int resident_sidecar_count_after_step1 =
                mock->forwardMTPFromResidentLogicalStateForDeviceSamplingCount();
            const int device_target_sidecar_count_after_step1 =
                mock->forwardMTPFromDeviceTargetForDeviceSamplingCount();

            GenerationResult step2 = runner->decodeStep();
            ASSERT_TRUE(step2.success()) << step2.error;
            ASSERT_THAT(step2.tokens, SizeIs(3));

            EXPECT_EQ(
                mock->forwardMTPFromResidentLogicalStateForDeviceSamplingCount(),
                resident_sidecar_count_after_step1 + 1)
                << "Step two should reuse the already queued first sidecar, "
                   "then enqueue exactly one replacement prelaunch for the "
                   "following step.";
            EXPECT_EQ(mock->forwardMTPFromDeviceTargetForDeviceSamplingCount(),
                      device_target_sidecar_count_after_step1)
                << "A matching resident prelaunch must replace the older "
                   "target-slot sidecar replay on the next greedy step.";

            const auto records = PerfStatsCollector::snapshot({"mtp"});
            ASSERT_NE(findPerfRecordWithTags(
                          records,
                          PerfStatRecord::Kind::Counter,
                          "stochastic_first_sidecar_prelaunch_reuses",
                          {{"sampling", "greedy"}}),
                      nullptr);
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
            mock->enableMTPShiftedRowReuseFromSidecar();
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
            EXPECT_EQ(mock->sequentialCommitMTPShiftedCount(), 0)
                << "the rejected correction is only a pending condition here; "
                   "its shifted row is appended when the next verifier step "
                   "consumes that condition token";
            EXPECT_EQ(mock->forwardCallCount(), forward_count_after_prefill + 1)
                << "only the all-position verifier forward runs in this step";

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
                                        {"correction_replay_tokens", "0"},
                                        {"deferred_correction_condition_tokens", "1"},
                                        {"ready_token", "-1"}});
            ASSERT_NE(trace, nullptr);

            const PerfStatRecord *publication_runs =
                findPerfRecordWithTags(records,
                                       PerfStatRecord::Kind::Counter,
                                       "all_position_state_publication_verifier_runs",
                                       {{"verifier_rows", "3"},
                                        {"correction_replay_tokens", "0"},
                                        {"accepted_state_count", "1"},
                                        {"target_cached_tokens", "6"}});
            ASSERT_NE(publication_runs, nullptr);
            const PerfStatRecord *reused_first_shifted =
                findPerfRecord(records,
                               PerfStatRecord::Kind::Counter,
                               "all_position_initial_shifted_reused_sidecar_rows");
            ASSERT_NE(reused_first_shifted, nullptr);
        }
        std::filesystem::remove(export_path);
        PerfStatsCollector::reset();
    }

    TEST_F(Test__PrefillDecodeTransition, AllPositionSpecPublicationWithoutSidecarReuseDoesNotOverpublishReject)
    {
        const std::filesystem::path export_path =
            std::filesystem::temp_directory_path() /
            "llaminar_mtp_all_position_no_sidecar_reuse_reject_unit.json";
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
                DeviceId::cuda(0),
                /*mtp_draft_tokens=*/2,
                /*chained_mtp_support=*/true);
            mock->enableMTPSpecStatePublication();
            mock->setVerifierAcceptedPrefixScript({0});

            ASSERT_TRUE(runner->prefill({1, 2, 3, 4, 5}));

            GenerationResult step1 = runner->decodeStep();
            ASSERT_TRUE(step1.success()) << step1.error;
            EXPECT_THAT(step1.tokens,
                        ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN,
                                    MockInferenceRunner::VERIFY_REJECT_TOKEN));

            EXPECT_EQ(mock->publishMTPSpecStateCount(), 1);
            EXPECT_EQ(mock->restoreCount(), 1)
                << "A non-preserving sidecar must restore the verifier base "
                   "before all-position publication.";
            EXPECT_EQ(mock->sequentialCommitMTPShiftedCount(), 1)
                << "Without a reusable sidecar row, row zero is still "
                   "published from the restored verifier-base terminal hidden.";
            EXPECT_EQ(mock->commitMTPShiftedCount(), 1);
            EXPECT_THAT(mock->lastCommitMTPTokens(),
                        ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN));
            EXPECT_EQ(mock->lastCommitMTPAlreadyAppended(), 0);
            EXPECT_EQ(mock->lastCommitMTPPositionOffsetOverride(), 5);
            EXPECT_TRUE(mock->lastPublishedMTPSpecStep().reuse_initial_mtp_shifted_kv_row)
                << "The plan flag means the initial shifted row is resident, "
                   "whether it came from a sidecar or verifier-base publication.";

            const auto records = PerfStatsCollector::snapshot({"mtp"});
            const PerfStatRecord *deferred_initial =
                findPerfRecordWithTags(
                    records,
                    PerfStatRecord::Kind::Counter,
                    "all_position_initial_shifted_deferred_to_verifier_rows",
                    {{"sidecar_preserves_main_state", "false"}});
            ASSERT_NE(deferred_initial, nullptr);
            EXPECT_DOUBLE_EQ(deferred_initial->value, 1.0);
            EXPECT_EQ(findPerfRecord(records,
                                     PerfStatRecord::Kind::Counter,
                                     "all_position_initial_shifted_reused_sidecar_rows"),
                      nullptr);
            const PerfStatRecord *initial_commit =
                findPerfRecordWithTags(
                    records,
                    PerfStatRecord::Kind::Counter,
                    "all_position_initial_shifted_commits",
                    {{"source", "verifier_base_terminal_hidden"}});
            ASSERT_NE(initial_commit, nullptr);
            EXPECT_DOUBLE_EQ(initial_commit->value, 1.0);
        }
        std::filesystem::remove(export_path);
        PerfStatsCollector::reset();
    }

    TEST_F(Test__PrefillDecodeTransition, AllPositionSpecPublicationWithoutSidecarReuseAnchorsShiftedCommitToVerifierBase)
    {
        const std::filesystem::path export_path =
            std::filesystem::temp_directory_path() /
            "llaminar_mtp_all_position_no_sidecar_reuse_accept_unit.json";
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
                /*chained_mtp_support=*/true);
            mock->enableMTPSpecStatePublication();
            mock->setVerifierAcceptedPrefixScript({2});

            ASSERT_TRUE(runner->prefill({1, 2, 3, 4, 5}));

            GenerationResult step1 = runner->decodeStep();
            ASSERT_TRUE(step1.success()) << step1.error;
            EXPECT_THAT(step1.tokens,
                        ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN,
                                    MockInferenceRunner::MTP_ARGMAX_TOKEN,
                                    MockInferenceRunner::MTP_ARGMAX_TOKEN));

            EXPECT_EQ(mock->restoreCount(), 1)
                << "Non-preserving sidecars must restore the verifier base "
                   "before publishing accepted verifier-row state.";
            EXPECT_EQ(mock->sequentialCommitMTPShiftedCount(), 1)
                << "Row zero should be published from restored verifier-base terminal hidden.";
            EXPECT_EQ(mock->commitMTPShiftedCount(), 2)
                << "Row zero plus accepted verifier rows after row zero both "
                   "publish shifted MTP KV.";
            EXPECT_EQ(mock->lastCommitMTPAlreadyAppended(), 1)
                << "Verifier row zero is skipped as the hidden source row.";
            EXPECT_EQ(mock->lastCommitMTPAlreadyAppendedShiftedKV(), 1)
                << "The row-zero verifier-base publication makes one skipped "
                   "shifted KV row resident before the batched prefix commit.";
            EXPECT_EQ(mock->lastCommitMTPPositionOffsetOverride(), 5)
                << "Non-reuse shifted commits must be anchored to the verifier "
                   "base cached-token count, not the later sidecar planning position.";
            EXPECT_TRUE(mock->lastPublishedMTPSpecStep().reuse_initial_mtp_shifted_kv_row);

            const auto records = PerfStatsCollector::snapshot({"mtp"});
            const PerfStatRecord *deferred_initial =
                findPerfRecordWithTags(
                    records,
                    PerfStatRecord::Kind::Counter,
                    "all_position_initial_shifted_deferred_to_verifier_rows",
                    {{"sidecar_preserves_main_state", "false"}});
            ASSERT_NE(deferred_initial, nullptr);
            EXPECT_EQ(findPerfRecord(records,
                                     PerfStatRecord::Kind::Counter,
                                     "all_position_initial_shifted_reused_sidecar_rows"),
                      nullptr);
            const PerfStatRecord *initial_commit =
                findPerfRecordWithTags(
                    records,
                    PerfStatRecord::Kind::Counter,
                    "all_position_initial_shifted_commits",
                    {{"source", "verifier_base_terminal_hidden"}});
            ASSERT_NE(initial_commit, nullptr);
            EXPECT_DOUBLE_EQ(initial_commit->value, 1.0);
        }
        std::filesystem::remove(export_path);
        PerfStatsCollector::reset();
    }

    TEST_F(Test__PrefillDecodeTransition, AllPositionSpecPublicationForcedRejectReplayCheckDerivesNextToken)
    {
        const std::filesystem::path export_path =
            std::filesystem::temp_directory_path() / "llaminar_mtp_all_position_reject_replay_check_unit.json";
        {
            ScopedEnv enable_stats("LLAMINAR_PERF_STATS_JSON", export_path.string().c_str());
            ScopedEnv enable_replay_check("LLAMINAR_MTP_VERIFY_COMMIT_REPLAY_CHECK", "1");
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
            mock->enableMTPShiftedRowReuseFromSidecar();
            mock->enableMTPSpecStatePublication();
            mock->setVerifierAcceptedPrefixScript({0});
            mock->setVerifierRejectTokenScript({
                MockInferenceRunner::VERIFY_REJECT_TOKEN,
                MockInferenceRunner::DECODE_ARGMAX_TOKEN,
            });
            mock->setDecodeArgmaxScript({
                MockInferenceRunner::DECODE_ARGMAX_TOKEN,
                MockInferenceRunner::DECODE_ARGMAX_TOKEN,
                MockInferenceRunner::DECODE_ARGMAX_TOKEN,
                MockInferenceRunner::DECODE_ARGMAX_TOKEN,
                MockInferenceRunner::DECODE_ARGMAX_TOKEN,
                MockInferenceRunner::DECODE_ARGMAX_TOKEN,
            });

            ASSERT_TRUE(runner->prefill({1, 2, 3, 4, 5}));

            GenerationResult rejected = runner->decodeStep();
            ASSERT_TRUE(rejected.success()) << rejected.error;
            EXPECT_THAT(rejected.tokens,
                        ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN,
                                    MockInferenceRunner::VERIFY_REJECT_TOKEN));

            const auto records = PerfStatsCollector::snapshot({"mtp"});
            const PerfStatRecord *derived =
                findPerfRecordWithTags(
                    records,
                    PerfStatRecord::Kind::Counter,
                    "commit_replay_check_derived_next_tokens",
                    {{"path", "all_position_state_publication_verifier"},
                     {"deferred_condition_token",
                      std::to_string(MockInferenceRunner::VERIFY_REJECT_TOKEN)},
                     {"next_token",
                      std::to_string(MockInferenceRunner::DECODE_ARGMAX_TOKEN)}});
            ASSERT_NE(derived, nullptr)
                << "forced-reject publication has no ready token, so the "
                   "debug replay oracle must derive one by forwarding the "
                   "rejected correction as the next condition token.";

            const PerfStatRecord *match =
                findPerfRecordWithTags(
                    records,
                    PerfStatRecord::Kind::Counter,
                    "commit_replay_check_matches",
                    {{"path", "all_position_state_publication_verifier"},
                     {"accepted_tokens", "7,4"},
                     {"next_token",
                      std::to_string(MockInferenceRunner::DECODE_ARGMAX_TOKEN)},
                     {"derived_next_token", "true"}});
            ASSERT_NE(match, nullptr);
            const PerfStatRecord *restore_after_derivation =
                findPerfRecordWithTags(
                    records,
                    PerfStatRecord::Kind::Counter,
                    "commit_replay_check_restores_after_ready_derivation",
                    {{"path", "all_position_state_publication_verifier"}});
            ASSERT_NE(restore_after_derivation, nullptr)
                << "ready-token derivation mutates committed state, so the "
                   "debug replay oracle must restore before feeding the "
                   "pending condition token for the continuation probe.";

            runner->setDecodeStepTokenBudget(1);
            GenerationResult ordinary = runner->decodeStep();
            runner->setDecodeStepTokenBudget(0);
            ASSERT_TRUE(ordinary.success()) << ordinary.error;
            EXPECT_THAT(ordinary.tokens,
                        ElementsAre(MockInferenceRunner::DECODE_ARGMAX_TOKEN))
                << "after a forced reject, the next one-token decode must "
                   "consume the rejected correction exactly once and continue "
                   "from the same token as a full replay.";
        }
        std::filesystem::remove(export_path);
        PerfStatsCollector::reset();
    }

    TEST_F(Test__PrefillDecodeTransition, AllPositionSpecPublicationConsumesPendingConditionWithoutConditionForward)
    {
        const std::filesystem::path export_path =
            std::filesystem::temp_directory_path() / "llaminar_mtp_all_position_synthetic_base_unit.json";
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
                /*mtp_draft_tokens=*/1,
                /*chained_mtp_support=*/false);
            mock->enableMTPSidecarPreservesMainState();
            mock->enableMTPShiftedRowReuseFromSidecar();
            mock->enableMTPSpecStatePublication();
            mock->setVerifierAcceptedPrefixScript({0, 1});
            mock->setDecodeArgmaxScript({MockInferenceRunner::DECODE_ARGMAX_TOKEN});

            ASSERT_TRUE(runner->prefill({1, 2, 3, 4, 5}));

            GenerationResult rejected = runner->decodeStep();
            ASSERT_TRUE(rejected.success()) << rejected.error;
            EXPECT_THAT(rejected.tokens,
                        ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN,
                                    MockInferenceRunner::VERIFY_REJECT_TOKEN));
            const int captures_after_reject = mock->captureCheckpointCount();
            ASSERT_EQ(captures_after_reject, 1)
                << "first ready-logits all-position step only needs the entry checkpoint";

            GenerationResult accepted = runner->decodeStep();
            ASSERT_TRUE(accepted.success()) << accepted.error;
            EXPECT_THAT(accepted.tokens,
                        ElementsAre(MockInferenceRunner::MTP_ARGMAX_TOKEN))
                << "the pending correction is verifier input, not output";
            EXPECT_EQ(mock->restoreCount(), 0)
                << "main-state-preserving all-position publication must not restore "
                   "the synthetic verifier base.";
            EXPECT_EQ(mock->captureCheckpointCount(), captures_after_reject + 1)
                << "the second step keeps its rollback checkpoint but skips the "
                   "standalone condition-forward verifier-base export.";

            const auto records = PerfStatsCollector::snapshot({"mtp"});
            const PerfStatRecord *pending_skip =
                findPerfRecord(records,
                               PerfStatRecord::Kind::Counter,
                               "condition_forward_skipped_pending_condition");
            ASSERT_NE(pending_skip, nullptr);
            EXPECT_DOUBLE_EQ(pending_skip->value, 1.0);
            const PerfStatRecord *verifier_base_capture =
                findPerfRecord(records,
                               PerfStatRecord::Kind::Timer,
                               "capture_verifier_base_prefix_state");
            EXPECT_EQ(verifier_base_capture, nullptr)
                << "the all-position state-publication fast path should not export "
                   "a verifier-base checkpoint after condition forward.";
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
            mock->enableMTPShiftedRowReuseFromSidecar();
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
            EXPECT_EQ(mock->sequentialCommitMTPShiftedCount(), 0)
                << "the rejected correction remains a pending condition, not a "
                   "same-step shifted-cache append";
            EXPECT_EQ(mock->commitMTPShiftedCount(), 1)
                << "only the accepted verifier prefix is committed from verifier hidden rows";
            EXPECT_EQ(mock->lastCommitMTPAlreadyAppended(), 1);
            EXPECT_EQ(mock->lastCommitMTPMainForwardTokenCount(), 3);
            EXPECT_THAT(mock->lastCommitMTPTokens(),
                        ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN,
                                    MockInferenceRunner::MTP_ARGMAX_TOKEN));
            EXPECT_EQ(mock->forwardCallCount(), forward_count_after_prefill + 1)
                << "only the all-position verifier forward runs in this step";

            const auto records = PerfStatsCollector::snapshot({"mtp"});
            const PerfStatRecord *publication_runs =
                findPerfRecordWithTags(records,
                                       PerfStatRecord::Kind::Counter,
                                       "all_position_state_publication_verifier_runs",
                                       {{"forward_tokens", "3"},
                                        {"verifier_rows", "3"},
                                        {"correction_replay_tokens", "0"},
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
            mock->enableMTPShiftedRowReuseFromSidecar();
            mock->enableMTPSidecarLogitsStreamHandoff();
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
            EXPECT_EQ(mock->allPositionVerifierSyncDeferralSetCount(), 0);
            EXPECT_EQ(mock->publishMTPSpecStateCount(), 1);
            EXPECT_EQ(mock->sequentialCommitMTPShiftedCount(), 0)
                << "the first stochastic sidecar row is reused for shifted MTP KV";
            EXPECT_EQ(mock->commitMTPShiftedCount(), 1)
                << "the accepted stochastic verifier row should fill the remaining shifted prefix without replay";
            EXPECT_EQ(mock->lastCommitMTPAlreadyAppended(), 1);
            EXPECT_EQ(mock->lastCommitMTPMainForwardTokenCount(), 2);
            EXPECT_EQ(mock->restoreCount(), 0);
            EXPECT_EQ(mock->forwardCallCount(), forward_count_after_prefill + 1);
            EXPECT_EQ(mock->deviceDistributionBuildCount(), 3)
                << "first target token, verifier row, and terminal ready-token "
                   "use compact distributions; the MTP draft proposal is "
                   "temperature-only full-probability";
            EXPECT_EQ(mock->deviceDraftTemperatureProposalCount(), 1);
            EXPECT_EQ(mock->deviceDraftTemperatureProposalDeferredCount(), 0);
            EXPECT_EQ(mock->deviceProbabilityRowsBuildCount(), 0);
            EXPECT_EQ(mock->deviceProcessedRowsBuildCount(), 0)
                << "the all-position verifier comparison and bonus rows should "
                   "use compact one-hot rows";
            EXPECT_EQ(mock->deviceDistributionSampleCount(), 2)
                << "first target token is compact and bonus is sampled inside the verifier; "
                   "the MTP draft sample is produced by the vLLM proposal kernel";
            EXPECT_EQ(mock->deviceDistributionVerifyBatchCount(), 1);
            EXPECT_TRUE(mock->lastBatchOutcomeUsedVLLMProbabilityRejection())
                << "history-dependent stochastic lanes should now batch through "
                   "the vLLM compact one-hot verifier";
            EXPECT_EQ(mock->lastBatchOutcomeInverseSampleSeed(), sampling.seed);
            EXPECT_GT(mock->lastBatchOutcomeInverseSampleFirstLogicalPosition(), 0);
            EXPECT_EQ(mock->deviceProbabilityVerifyRowCount(), 0)
                << "history-dependent stochastic lanes should not fall back to "
                   "the scalar row verifier";
            EXPECT_EQ(mock->applyMainPenaltiesCount(), 0)
                << "the first stochastic token has no prior sampler history, "
                   "so an empty penalty map must not hit the device hook";
            EXPECT_EQ(mock->applyMTPPenaltiesCount(), 0)
                << "vLLM-style draft proposal ignores draft-side penalties; "
                   "target-side rejection correction owns the final policy";
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
            EXPECT_EQ(findPerfRecord(records,
                                     PerfStatRecord::Kind::Timer,
                                     "decode_equivalent_stochastic_forward_one"),
                      nullptr)
                << "accepted all-position stochastic lanes must not fall back "
                   "to the decode-equivalent stepwise verifier";
        }
        std::filesystem::remove(export_path);
        PerfStatsCollector::reset();
    }

    TEST_F(Test__PrefillDecodeTransition, GroupedOutcomeDeviceResidentPublicationUsesBatchedStochasticVerifier)
    {
        const std::filesystem::path export_path =
            std::filesystem::temp_directory_path() /
            "llaminar_mtp_grouped_outcome_device_publication_unit.json";
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
        mock->enableGroupedOutcomeDeviceResidentPublication(/*rows=*/4);
        mock->enableDeviceResidentMTPSpecStatePublication();
        mock->hideMTPSpecStatePublicationFromPolicy();
        mock->setVerifierAcceptedPrefixScript({1});

        SamplingParams sampling;
        sampling.temperature = 0.8f;
        sampling.top_k = 2;
        sampling.top_p = 0.95f;
        sampling.seed = 123;
        runner->setSamplingParams(sampling);

        ASSERT_TRUE(runner->prefill({1, 2, 3, 4, 5}));
        const int forward_count_after_prefill = mock->forwardCallCount();

        GenerationResult step = runner->decodeStep();
        ASSERT_TRUE(step.success()) << step.error;
        EXPECT_THAT(step.tokens,
                    ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN,
                                MockInferenceRunner::MTP_ARGMAX_TOKEN));

        EXPECT_EQ(mock->setMTPSpecVerifierPlanCount(), 1);
        EXPECT_EQ(mock->setRowIndexedAllPositionCount(), 2);
        EXPECT_EQ(mock->setAllPositionCount(), 2);
        EXPECT_THAT(mock->lastMTPSpecVerifierRows(), ElementsAre(0, 1));
        EXPECT_EQ(mock->stageStochasticDraftTokensCount(), 1)
            << "host-visible draft shadows must be explicitly staged into "
               "runner-owned verifier slots before the compact outcome reducer";
        EXPECT_FALSE(mock->batchOutcomeUsedHostDraftTokens())
            << "the grouped outcome reducer must read device draft slots, not "
               "host draft-token pointers";
        EXPECT_EQ(mock->deviceDistributionVerifyBatchCount(), 1);
        EXPECT_TRUE(mock->lastBatchOutcomeUsedVLLMProbabilityRejection());
        EXPECT_EQ(mock->publishMTPSpecStateCount(), 0);
        EXPECT_EQ(mock->publishMTPSpecStateBatchCount(), 0);
        EXPECT_EQ(mock->publishDeviceResidentMTPSpecStateCount(), 1);
        EXPECT_EQ(mock->adoptDeviceResidentHostStateCount(), 1);
        EXPECT_EQ(mock->sequentialCommitMTPShiftedCount(), 0)
            << "grouped outcomes must publish state from compact device "
               "metadata instead of replaying shifted rows";
        EXPECT_EQ(mock->restoreCount(), 1)
            << "the runner restores before grouped verification, then direct "
               "publication consumes verifier rows without replay restore";
        EXPECT_EQ(mock->forwardCallCount(), forward_count_after_prefill + 1)
            << "one grouped target verifier forward; no replay forwards";

        const auto records = PerfStatsCollector::snapshot({"mtp"});
        EXPECT_EQ(findPerfRecord(records,
                                 PerfStatRecord::Kind::Timer,
                                 "decode_equivalent_stochastic_forward_one"),
                  nullptr)
            << "GroupedDecodeEquivalentOutcome must not silently use the "
               "scalar stochastic verifier loop.";
        ASSERT_NE(findPerfRecord(records,
                                 PerfStatRecord::Kind::Timer,
                                 "grouped_outcome_stochastic_verifier_forward"),
                  nullptr);
        ASSERT_NE(findPerfRecord(records,
                                 PerfStatRecord::Kind::Counter,
                                 "grouped_outcome_device_resident_publication_uses"),
                  nullptr);
        ASSERT_NE(findPerfRecord(records,
                                 PerfStatRecord::Kind::Counter,
                                 "grouped_outcome_device_resident_state_publications"),
                  nullptr);
        ASSERT_NE(findPerfRecordWithTags(
                      records,
                      PerfStatRecord::Kind::Counter,
                      "grouped_decode_equivalent_stochastic_verifier_runs",
                      {{"verifier_forward_tokens", "2"},
                       {"verifier_rows", "2"},
                       {"replay_forward_tokens", "0"},
                       {"shifted_commits", "0"},
                       {"state_publication", "device_resident"}}),
                  nullptr);
        ASSERT_NE(findPerfRecordWithTags(
                      records,
                      PerfStatRecord::Kind::Counter,
                      "acceptance_trace",
                      {{"verifier_path",
                        "grouped_decode_equivalent_stochastic"},
                       {"catchup_implementation",
                        "device_batch_outcome_device_resident_publication"},
                       {"policy_path",
                        "grouped_outcome_device_resident_publication"},
                       {"decode_equivalent_replay_required", "false"}}),
                  nullptr);

        std::filesystem::remove(export_path);
        PerfStatsCollector::reset();
    }

    TEST_F(Test__PrefillDecodeTransition, GroupedOutcomeDeviceResidentPublicationDefersFirstTokenAndDraftHostReads)
    {
        const std::filesystem::path export_path =
            std::filesystem::temp_directory_path() /
            "llaminar_mtp_grouped_outcome_device_first_unit.json";
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
            /*sidecar_sample_fusion=*/false,
            {},
            MTPVerifyMode::SpeculativeSampling);
        mock->enableStochasticDeviceSampling();
        mock->enableGroupedOutcomeDeviceResidentPublication(/*rows=*/4);
        mock->enableDeviceResidentMTPSpecStatePublication();
        mock->hideMTPSpecStatePublicationFromPolicy();
        mock->enableMTPSidecarPreservesMainState();
        mock->enableMTPShiftedRowReuseFromSidecar();
        mock->enableMTPSidecarLogitsStreamHandoff();
        mock->enableMTPDeviceDraftTokenInput();
        mock->setVerifierAcceptedPrefixScript({1});

        SamplingParams sampling;
        sampling.temperature = 0.8f;
        sampling.top_k = 2;
        sampling.top_p = 0.95f;
        sampling.seed = 123;
        runner->setSamplingParams(sampling);

        ASSERT_TRUE(runner->prefill({1, 2, 3, 4, 5}));
        GenerationResult step = runner->decodeStep();
        ASSERT_TRUE(step.success()) << step.error;
        EXPECT_THAT(step.tokens,
                    ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN,
                                MockInferenceRunner::MTP_ARGMAX_TOKEN));

        EXPECT_EQ(mock->deviceDistributionSampleDeferredCount(), 1)
            << "The first stochastic target token should stay in a device target slot.";
        EXPECT_EQ(mock->deviceDraftTemperatureProposalDeferredCount(), 1)
            << "The sidecar draft token should stay in a device draft slot.";
        EXPECT_EQ(mock->deviceDistributionSampleCount(), 1)
            << "The grouped outcome still samples the bonus-ready row inside the "
               "device batch outcome; first-token sampling is protected by the "
               "deferred counter above.";
        EXPECT_EQ(mock->forwardMTPFromDeviceTargetForDeviceSamplingCount(), 1);
        EXPECT_EQ(mock->forwardMTPForDeviceSamplingCount(), 0)
            << "The grouped sidecar must consume the device target slot, not a host token.";
        EXPECT_EQ(mock->prepareMTPVerifierInputTokensDeviceFirstCount(), 1);
        EXPECT_EQ(mock->lastPrepareMTPVerifierFirstDraftSlot(), 0);
        EXPECT_EQ(mock->lastPrepareMTPVerifierDraftTokenCount(), 1);
        EXPECT_EQ(mock->lastPrepareMTPVerifierTotalTokens(), 2);
        EXPECT_EQ(mock->forwardWithDeviceTokenIdsCount(), 1);
        EXPECT_EQ(mock->verifyStochasticRequestBatchOutcomeCount(), 1);
        EXPECT_THAT(mock->lastRequestBatchOutcomeFirstTokens(), ElementsAre(-1));
        ASSERT_THAT(mock->lastRequestBatchOutcomeDraftTokens(), SizeIs(1));
        EXPECT_THAT(mock->lastRequestBatchOutcomeDraftTokens()[0], ElementsAre(-1));
        EXPECT_EQ(mock->residentSidecarCountAtLastHostBridge(), 1)
            << "Grouped-outcome publication should prelaunch from the resolved "
               "device mailbox before the host response bridge.";

        const auto records = PerfStatsCollector::snapshot({"mtp"});
        ASSERT_NE(findPerfRecord(records,
                                 PerfStatRecord::Kind::Counter,
                                 "first_token_stochastic_deferred_host_reads"),
                  nullptr);
        ASSERT_NE(findPerfRecordWithTags(
                      records,
                      PerfStatRecord::Kind::Counter,
                      "mtp_token_stochastic_deferred_host_reads",
                      {{"draft_idx", "0"}}),
                  nullptr);
        ASSERT_NE(findPerfRecordWithTags(
                      records,
                      PerfStatRecord::Kind::Counter,
                      "grouped_outcome_verifier_device_token_inputs",
                      {{"total_tokens", "2"}}),
                  nullptr);
        ASSERT_NE(findPerfRecordWithTags(
                      records,
                      PerfStatRecord::Kind::Counter,
                      "stochastic_first_sidecar_prelaunches",
                      {{"path", "grouped_outcome_device_resident_publication"},
                       {"resident_state_kind", "device_publication_mailbox"},
                       {"prelaunch_timing", "pre_bridge"}}),
                  nullptr);

        std::filesystem::remove(export_path);
        PerfStatsCollector::reset();
    }

    TEST_F(Test__PrefillDecodeTransition, GroupedOutcomeDeviceResidentPublicationSkipsPayloadCheckpointWhenSidecarPreservesMainState)
    {
        const std::filesystem::path export_path =
            std::filesystem::temp_directory_path() /
            "llaminar_mtp_grouped_outcome_checkpoint_skip_unit.json";
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
        mock->enableGroupedOutcomeDeviceResidentPublication(/*rows=*/4);
        mock->enableDeviceResidentMTPSpecStatePublication();
        mock->hideMTPSpecStatePublicationFromPolicy();
        mock->enableMTPSidecarPreservesMainState();
        mock->enableMTPShiftedRowReuseFromSidecar();
        mock->enableMTPSidecarLogitsStreamHandoff();
        mock->setVerifierAcceptedPrefixScript({1});

        SamplingParams sampling;
        sampling.temperature = 0.8f;
        sampling.top_k = 2;
        sampling.top_p = 0.95f;
        sampling.seed = 123;
        runner->setSamplingParams(sampling);

        ASSERT_TRUE(runner->prefill({1, 2, 3, 4, 5}));

        GenerationResult step = runner->decodeStep();
        ASSERT_TRUE(step.success()) << step.error;
        EXPECT_THAT(step.tokens,
                    ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN,
                                MockInferenceRunner::MTP_ARGMAX_TOKEN));

        EXPECT_EQ(mock->publishDeviceResidentMTPSpecStateCount(), 1);
        EXPECT_EQ(mock->adoptDeviceResidentHostStateCount(), 1);
        EXPECT_EQ(mock->restoreCount(), 0)
            << "A grouped vLLM-style verifier with sidecar preservation should "
               "never restore a payload checkpoint on the success path.";
        EXPECT_EQ(mock->captureCheckpointCount(), 0)
            << "The grouped device-resident publication path should carry only "
               "a logical base stamp; exporting hybrid payloads here is the "
               "full-pipeline MoE bottleneck this regression protects.";

        const auto records = PerfStatsCollector::snapshot({"mtp"});
        ASSERT_NE(findPerfRecordWithTags(
                      records,
                      PerfStatRecord::Kind::Counter,
                      "live_prefix_checkpoint_skipped_direct_publication",
                      {{"verifier_path", "grouped_decode_equivalent_outcome"}}),
                  nullptr);
        ASSERT_NE(findPerfRecordWithTags(
                      records,
                      PerfStatRecord::Kind::Counter,
                      "post_sidecar_checkpoint_skipped_sidecar_preserved",
                      {{"verifier_path", "grouped_decode_equivalent_outcome"}}),
                  nullptr);
        EXPECT_EQ(findPerfRecord(records,
                                 PerfStatRecord::Kind::Timer,
                                 "capture_live_prefix_state"),
                  nullptr);
        EXPECT_EQ(findPerfRecord(records,
                                 PerfStatRecord::Kind::Timer,
                                 "capture_verifier_base_prefix_state"),
                  nullptr);

        std::filesystem::remove(export_path);
        PerfStatsCollector::reset();
    }

    TEST_F(Test__PrefillDecodeTransition, GroupedOutcomeDeviceResidentPublicationConsumesPendingConditionWithoutConditionForward)
    {
        const std::filesystem::path export_path =
            std::filesystem::temp_directory_path() /
            "llaminar_mtp_grouped_outcome_pending_condition_unit.json";
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
        mock->enableGroupedOutcomeDeviceResidentPublication(/*rows=*/4);
        mock->enableDeviceResidentMTPSpecStatePublication();
        mock->hideMTPSpecStatePublicationFromPolicy();
        mock->enableMTPSidecarPreservesMainState();
        mock->enableMTPShiftedRowReuseFromSidecar();
        mock->enableMTPSidecarLogitsStreamHandoff();
        mock->enableMTPDeviceDraftTokenInput();
        mock->setVerifierAcceptedPrefixScript({0, 1});

        SamplingParams sampling;
        sampling.temperature = 0.8f;
        sampling.top_k = 2;
        sampling.top_p = 0.95f;
        sampling.seed = 123;
        runner->setSamplingParams(sampling);

        ASSERT_TRUE(runner->prefill({1, 2, 3, 4, 5}));
        const int forward_count_after_prefill = mock->forwardCallCount();

        GenerationResult step1 = runner->decodeStep();
        ASSERT_TRUE(step1.success()) << step1.error;
        EXPECT_THAT(step1.tokens,
                    ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN,
                                MockInferenceRunner::VERIFY_REJECT_TOKEN));
        EXPECT_EQ(mock->publishDeviceResidentMTPSpecStateCount(), 1);
        EXPECT_EQ(mock->adoptDeviceResidentHostStateCount(), 1);
        EXPECT_EQ(mock->residentAcceptedStateCount(0), 1)
            << "Reject-first grouped publication should publish only the "
               "already-consumed verifier row and leave the correction as "
               "resident pending-condition state.";
        EXPECT_EQ(mock->forwardCallCount(), forward_count_after_prefill + 1)
            << "The first step should run one grouped verifier forward and no replay.";
        EXPECT_EQ(mock->residentSidecarCountAtLastHostBridge(), 1)
            << "Grouped rejected-token publication should prelaunch from the "
               "device-resolved mailbox before the host outcome bridge.";
        EXPECT_EQ(mock->lastResidentLogicalStateSidecarToken(),
                  MockInferenceRunner::VERIFY_REJECT_TOKEN)
            << "The post-outcome resident prelaunch should use the correction "
               "token selected by the grouped verifier.";

        const auto records_after_reject = PerfStatsCollector::snapshot({"mtp"});
        ASSERT_NE(findPerfRecordWithTags(
                      records_after_reject,
                      PerfStatRecord::Kind::Counter,
                      "acceptance_trace",
                      {{"verifier_path", "grouped_decode_equivalent_stochastic"},
                       {"pending_condition_input", "false"},
                       {"output_tokens", "2"},
                       {"next_pending_condition_token",
                        std::to_string(MockInferenceRunner::VERIFY_REJECT_TOKEN)}}),
                  nullptr);
        ASSERT_NE(findPerfRecordWithTags(
                      records_after_reject,
                      PerfStatRecord::Kind::Counter,
                      "stochastic_first_sidecar_prelaunches",
                      {{"path", "grouped_outcome_device_resident_publication"},
                       {"resident_state_kind", "device_publication_mailbox"},
                       {"prelaunch_timing", "pre_bridge"}}),
                  nullptr);

        const int forward_count_after_reject = mock->forwardCallCount();
        const int resident_sidecar_count_after_reject =
            mock->forwardMTPFromResidentLogicalStateForDeviceSamplingCount();
        const int host_sidecar_count_after_reject =
            mock->forwardMTPForDeviceSamplingCount();

        GenerationResult step2 = runner->decodeStep();
        ASSERT_TRUE(step2.success()) << step2.error;
        EXPECT_THAT(step2.tokens,
                    ElementsAre(MockInferenceRunner::MTP_ARGMAX_TOKEN))
            << "The pending correction was already emitted in step one; grouped "
               "verification must consume it as row zero and emit only the new "
               "accepted draft.";
        EXPECT_EQ(mock->forwardMTPFromResidentLogicalStateForDeviceSamplingCount(),
                  resident_sidecar_count_after_reject + 1);
        EXPECT_EQ(mock->forwardMTPForDeviceSamplingCount(),
                  host_sidecar_count_after_reject)
            << "Grouped resident pending-condition reuse must not fall back to "
               "the host-token sidecar entrypoint.";
        EXPECT_EQ(mock->forwardCallCount(), forward_count_after_reject + 1)
            << "Step two should run only the grouped verifier forward; the "
               "standalone condition_forward is the duplicated work this protects.";

        const auto records_after_step2 = PerfStatsCollector::snapshot({"mtp"});
        ASSERT_NE(findPerfRecord(records_after_step2,
                                 PerfStatRecord::Kind::Counter,
                                 "stochastic_first_sidecar_prelaunch_reuses"),
                  nullptr)
            << "The first grouped sidecar of step two should reuse the "
               "correction prelaunch rather than replaying from a host token.";
        const PerfStatRecord *pending_skip =
            findPerfRecord(records_after_step2,
                           PerfStatRecord::Kind::Counter,
                           "condition_forward_skipped_pending_condition");
        ASSERT_NE(pending_skip, nullptr);
        EXPECT_DOUBLE_EQ(pending_skip->value, 1.0);
        ASSERT_NE(findPerfRecordWithTags(
                      records_after_step2,
                      PerfStatRecord::Kind::Counter,
                      "acceptance_trace",
                      {{"verifier_path", "grouped_decode_equivalent_stochastic"},
                       {"draft_tokens", "4,-2"},
                       {"pending_condition_input", "true"},
                       {"output_tokens", "1"},
                       {"next_pending_condition_token", "none"}}),
                  nullptr);
        ASSERT_NE(findPerfRecordWithTags(
                      records_after_step2,
                      PerfStatRecord::Kind::Counter,
                      "transaction_commits",
                      {{"path", "grouped_decode_equivalent_stochastic_verifier"},
                       {"emitted_token_start_index", "1"},
                       {"emitted_tokens", "1"}}),
                  nullptr);

        std::filesystem::remove(export_path);
        PerfStatsCollector::reset();
    }

    TEST_F(Test__PrefillDecodeTransition, DeviceResidentOutcomePublishesStateBeforeHostFlush)
    {
        const std::filesystem::path export_path =
            std::filesystem::temp_directory_path() /
            "llaminar_mtp_device_resident_publication_unit.json";
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
        mock->enableMTPShiftedRowReuseFromSidecar();
        mock->enableMTPSidecarLogitsStreamHandoff();
        mock->requireMTPDecodeEquivalentReplay();
        mock->enableDeviceResidentMTPSpecStatePublication();
        mock->setVerifierAcceptedPrefixScript({1});

        SamplingParams sampling;
        sampling.temperature = 0.8f;
        sampling.top_k = 2;
        sampling.top_p = 0.95f;
        sampling.presence_penalty = 0.25f;
        sampling.seed = 123;
        runner->setSamplingParams(sampling);

        ASSERT_TRUE(runner->prefill({1, 2, 3, 4, 5}));

        GenerationResult step = runner->decodeStep();
        ASSERT_TRUE(step.success()) << step.error;
        EXPECT_THAT(step.tokens,
                    ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN,
                                MockInferenceRunner::MTP_ARGMAX_TOKEN));

        EXPECT_EQ(mock->publishDeviceResidentMTPSpecStateCount(), 1)
            << "A capable backend should publish accepted state directly from "
               "the compact device outcome.";
        EXPECT_EQ(mock->publishMTPSpecStateBatchCount(), 0)
            << "The compatibility host-plan publisher must not run after "
               "device-resident publication succeeds.";
        EXPECT_EQ(mock->adoptDeviceResidentHostStateCount(), 1)
            << "The compatibility host bridge should refresh host-visible "
               "positions/sequence lengths without re-publishing live state.";
        EXPECT_EQ(mock->captureCheckpointCount(), 0)
            << "the vLLM-style device-resident transaction carries only a "
               "logical base stamp on the success path; rollback checkpoints "
               "belong to legacy verifier lanes, even when debug replay would "
               "need a payload checkpoint to restore hybrid state.";
        EXPECT_EQ(mock->commitMTPShiftedCount(), 0)
            << "Accepted shifted-prefix commit is part of the direct "
               "device-resident publication contract.";
        EXPECT_THAT(mock->publicationEvents(),
                    ElementsAre("device_outcome_publish",
                                "host_outcome_bridge",
                                "host_state_adopt"))
            << "Host outcome copying should remain only an output flush after "
               "live state has already been published; host mirror adoption is "
               "a separate no-KV-mutation step.";

        const DeviceSpeculativePublicationRequest &request =
            mock->lastDeviceResidentPublicationRequest();
        EXPECT_TRUE(request.valid());
        EXPECT_EQ(request.request_count, 1);
        EXPECT_EQ(request.max_draft_tokens, 2);
        EXPECT_TRUE(request.base_cached_tokens.empty())
            << "Device-resident publication should use the pre-verifier "
               "device snapshot, not a host base-cache shadow.";
        EXPECT_EQ(request.base_sidecar_position, 5);
        ASSERT_TRUE(mock->lastAdoptedDeviceResidentHostState().ok);
        ASSERT_THAT(mock->lastAdoptedDeviceResidentHostState().steps,
                    SizeIs(1));
        EXPECT_EQ(
            mock->lastAdoptedDeviceResidentHostState()
                .steps.front()
                .target_cached_tokens,
            mock->get_position());

        const auto records = PerfStatsCollector::snapshot({"mtp"});
        const PerfStatRecord *skipped_checkpoint =
            findPerfRecord(records,
                           PerfStatRecord::Kind::Counter,
                           "live_prefix_checkpoint_skipped_direct_publication");
        ASSERT_NE(skipped_checkpoint, nullptr);
        EXPECT_DOUBLE_EQ(skipped_checkpoint->value, 1.0);
        EXPECT_EQ(findPerfRecord(records,
                                 PerfStatRecord::Kind::Timer,
                                 "capture_live_prefix_state"),
                  nullptr);
        std::filesystem::remove(export_path);
        PerfStatsCollector::reset();
    }

    TEST_F(Test__PrefillDecodeTransition, DeviceResidentOutcomeDefersResidentCorrectionAsPendingCondition)
    {
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
        mock->enableMTPShiftedRowReuseFromSidecar();
        mock->enableLogicalMTPVerifierBaseCheckpoint();
        mock->enableMTPSidecarLogitsStreamHandoff();
        mock->enableMTPDeviceDraftTokenInput();
        mock->requireMTPDecodeEquivalentReplay();
        mock->enableDeviceResidentMTPSpecStatePublication();
        mock->setVerifierAcceptedPrefixScript({0, 1});

        SamplingParams sampling;
        sampling.temperature = 0.8f;
        sampling.top_k = 2;
        sampling.top_p = 0.95f;
        sampling.seed = 123;
        runner->setSamplingParams(sampling);

        ASSERT_TRUE(runner->prefill({1, 2, 3, 4, 5}));

        GenerationResult step = runner->decodeStep();
        ASSERT_TRUE(step.success()) << step.error;
        EXPECT_THAT(step.tokens,
                    ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN,
                                MockInferenceRunner::VERIFY_REJECT_TOKEN));

        EXPECT_EQ(mock->publishDeviceResidentMTPSpecStateCount(), 1);
        EXPECT_EQ(mock->adoptDeviceResidentHostStateCount(), 1);
        EXPECT_EQ(mock->residentLogicalStateShiftedCommitCount(), 0)
            << "The rejected correction token should remain resident pending "
               "condition state, not a same-step shifted-cache append.";
        EXPECT_EQ(mock->sequentialCommitMTPShiftedCount(), 0)
            << "Direct resident publication must not fall back to the host-token "
               "shifted-row helper for a rejected correction.";
        EXPECT_EQ(mock->commitMTPShiftedCount(), 0);
        EXPECT_EQ(mock->residentAcceptedStateCount(0), 1)
            << "Reject-first stochastic publication should expose row zero as "
               "the accepted-state/correction boundary on device.";
        EXPECT_EQ(mock->lastResidentLogicalStateShiftedCommitToken(), -1);
        EXPECT_TRUE(mock->lastCommitMTPTokens().empty());
        EXPECT_EQ(mock->residentSidecarCountAtLastHostBridge(), 1)
            << "The next resident sidecar should be prelaunched before the "
               "compatibility host outcome bridge flushes served tokens.";

        const int resident_sidecar_count_after_reject =
            mock->forwardMTPFromResidentLogicalStateForDeviceSamplingCount();
        const int host_sidecar_count_after_reject =
            mock->forwardMTPForDeviceSamplingCount();

        GenerationResult next_step = runner->decodeStep();
        ASSERT_TRUE(next_step.success()) << next_step.error;
        EXPECT_THAT(next_step.tokens,
                    ElementsAre(MockInferenceRunner::MTP_ARGMAX_TOKEN))
            << "The previously emitted correction is verifier input for the "
               "next transaction, not a newly emitted output token.";
        EXPECT_EQ(mock->forwardMTPFromResidentLogicalStateForDeviceSamplingCount(),
                  resident_sidecar_count_after_reject + 1)
            << "Step two may discard an incompatible rejected-lane prelaunch, "
               "but it should still queue the following sidecar from its own "
               "fresh resident mailbox.";
        EXPECT_EQ(mock->forwardMTPForDeviceSamplingCount(),
                  host_sidecar_count_after_reject)
            << "A resident pending condition must not fall back to the "
               "host-token sidecar entry point.";
        EXPECT_EQ(mock->flushPendingMTPWorkCount(), 0)
            << "Resident stochastic verifier inputs own their sample-ready "
               "events, so the runner must not drain the MTP sidecar stream "
               "before the verifier graph can consume device token slots.";
    }

    TEST_F(Test__PrefillDecodeTransition, AllPositionSpecPublicationPenaltyFreeStochasticDefersVerifierSync)
    {
        const std::filesystem::path export_path =
            std::filesystem::temp_directory_path() / "llaminar_mtp_all_position_stochastic_deferred_sync_unit.json";
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
            mock->enableMTPShiftedRowReuseFromSidecar();
            mock->enableMTPSidecarLogitsStreamHandoff();
            mock->enableMTPDeviceDraftTokenInput();
            mock->requireMTPDecodeEquivalentReplay();
            mock->enableDeviceResidentMTPSpecStatePublication();
            mock->setVerifierAcceptedPrefixScript({1});

            SamplingParams sampling;
            sampling.temperature = 0.8f;
            sampling.top_k = 1;
            sampling.top_p = 0.95f;
            sampling.seed = 123;
            runner->setSamplingParams(sampling);

            ASSERT_TRUE(runner->prefill({1, 2, 3, 4, 5}));
            GenerationResult step1 = runner->decodeStep();
            ASSERT_TRUE(step1.success()) << step1.error;
            EXPECT_THAT(step1.tokens,
                        ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN,
                                    MockInferenceRunner::MTP_ARGMAX_TOKEN));

            EXPECT_EQ(mock->allPositionVerifierSyncDeferralEnableCount(), 1)
                << "penalty-free device-batched stochastic verification should "
                   "handoff the verifier graph stream to the sampler path";
            EXPECT_EQ(mock->allPositionVerifierSyncDeferralDisableCount(), 1);
            EXPECT_FALSE(mock->allPositionVerifierSyncDeferralEnabled());
            EXPECT_EQ(mock->forwardMTPForDeviceSamplingCount(), 0)
                << "the first sidecar should consume the prefill token from the "
                   "device target slot rather than a host-uploaded token";
            EXPECT_EQ(mock->forwardMTPFromDeviceTargetForDeviceSamplingCount(), 1);
            EXPECT_EQ(mock->forwardMTPFromLastDraftForDeviceSamplingCount(), 0);
            EXPECT_EQ(mock->flushPendingMTPWorkCount(), 0)
                << "resident sidecar and verifier inputs carry their own stream "
                   "dependencies and should not force a CPU-side drain";
            EXPECT_EQ(mock->deviceDistributionBuildCount(), 3)
                << "first target token, verifier row, and terminal ready-token "
                   "use compact distributions; MTP draft proposal bypasses "
                   "compact top-k/top-p tables";
            EXPECT_EQ(mock->deviceDraftTemperatureProposalCount(), 1);
            EXPECT_EQ(mock->deviceDraftTemperatureProposalDeferredCount(), 1);
            EXPECT_EQ(mock->deviceProbabilityRowsBuildCount(), 0);
            EXPECT_EQ(mock->deviceProcessedRowsBuildCount(), 0)
                << "the verifier and bonus rows should use compact one-hot rows";
            EXPECT_EQ(mock->deviceDistributionSampleCount(), 1)
                << "only the first target token should be sampled through the "
                   "compact distribution helper; bonus sampling stays folded "
                   "into the resident verifier outcome";
            EXPECT_EQ(mock->deviceDistributionSampleDeferredCount(), 1)
                << "the first target token is deferred into the device target "
                   "slot for the sidecar, not read back to the host";
            EXPECT_EQ(mock->deviceDistributionVerifyBatchCount(), 1);
            EXPECT_TRUE(mock->lastBatchOutcomeUsedVLLMProbabilityRejection())
                << "penalty-free all-position stochastic verifier should use "
                   "the vLLM compact one-hot verifier branch";
            EXPECT_EQ(mock->lastBatchOutcomeInverseSampleSeed(), sampling.seed);
            EXPECT_GT(mock->lastBatchOutcomeInverseSampleFirstLogicalPosition(), 0);
            EXPECT_FALSE(mock->batchOutcomeUsedHostDraftTokens())
                << "compact device outcome verification should read sampled "
                   "draft tokens from device slots, not from a host shadow";
            EXPECT_EQ(mock->applyAllPositionPenaltiesCount(), 0)
                << "penalties would make row distributions history-dependent "
                   "and must keep the synchronized verifier boundary";

            const auto records = PerfStatsCollector::snapshot({"mtp"});
            const PerfStatRecord *batched_rows =
                findPerfRecordWithTags(records,
                                       PerfStatRecord::Kind::Counter,
                                       "all_position_stochastic_device_batched_rows",
                                       {{"implementation", "device_batch_outcome"},
                                        {"verifier_path", "all_position_state_publication"}});
            ASSERT_NE(batched_rows, nullptr);
            const PerfStatRecord *deferred_draft =
                findPerfRecordWithTags(records,
                                       PerfStatRecord::Kind::Counter,
                                       "mtp_token_stochastic_deferred_host_reads",
                                       {{"draft_idx", "0"}});
            ASSERT_NE(deferred_draft, nullptr);
            EXPECT_EQ(findPerfRecord(records,
                                     PerfStatRecord::Kind::Timer,
                                     "decode_equivalent_stochastic_forward_one"),
                      nullptr)
                << "penalty-free compact device outcomes should stay on the "
                   "all-position verifier path";

            const int resident_sidecar_count_after_step1 =
                mock->forwardMTPFromResidentLogicalStateForDeviceSamplingCount();
            const int device_target_sidecar_count_after_step1 =
                mock->forwardMTPFromDeviceTargetForDeviceSamplingCount();
            const int host_condition_sidecar_count_after_step1 =
                mock->forwardMTPForDeviceSamplingCount();
            EXPECT_EQ(mock->residentSidecarCountAtLastHostBridge(), 1)
                << "The accepted ready-token path should prelaunch the next "
                   "sidecar from the resident mailbox before the host response "
                   "bridge materializes compact outcomes.";

            GenerationResult step2 = runner->decodeStep();
            ASSERT_TRUE(step2.success()) << step2.error;
            ASSERT_FALSE(step2.tokens.empty());
            EXPECT_EQ(
                mock->forwardMTPFromResidentLogicalStateForDeviceSamplingCount(),
                resident_sidecar_count_after_step1 + 1)
                << "Step two should reuse the sidecar prelaunched from the "
                   "bonus ready token, then queue the following resident "
                   "sidecar before the host bridge.";
            EXPECT_EQ(mock->forwardMTPFromDeviceTargetForDeviceSamplingCount(),
                      device_target_sidecar_count_after_step1)
                << "A resident ready token should supersede the older deferred "
                   "target-slot sidecar input path on the following step.";
            EXPECT_EQ(mock->forwardMTPForDeviceSamplingCount(),
                      host_condition_sidecar_count_after_step1)
                << "The ready-token handoff must not fall back to a host-token "
                   "sidecar condition.";
            EXPECT_EQ(step2.tokens.front(),
                      mock->lastResidentLogicalStateSidecarToken())
                << "The token emitted to the served response and the token fed "
                   "to the next sidecar must be the same ready verifier token.";
            const auto records_after_step2 = PerfStatsCollector::snapshot({"mtp"});
            const PerfStatRecord *prelaunch_reuse =
                findPerfRecord(records_after_step2,
                               PerfStatRecord::Kind::Counter,
                               "stochastic_first_sidecar_prelaunch_reuses");
            ASSERT_NE(prelaunch_reuse, nullptr)
                << "The first sidecar of step two should reuse the sidecar "
                   "prelaunched at the end of step one instead of replaying "
                   "resident-ready-token input.";
        }
        std::filesystem::remove(export_path);
        PerfStatsCollector::reset();
    }

    TEST_F(Test__PrefillDecodeTransition, ResidentPrelaunchReusesWithNonMatchingStopTokens)
    {
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
        mock->enableMTPSidecarLogitsStreamHandoff();
        mock->enableMTPDeviceDraftTokenInput();
        mock->requireMTPDecodeEquivalentReplay();
        mock->enableDeviceResidentMTPSpecStatePublication();
        mock->setVerifierAcceptedPrefixScript({1});

        SamplingParams sampling;
        sampling.temperature = 0.8f;
        sampling.top_k = 1;
        sampling.top_p = 0.95f;
        sampling.seed = 123;
        runner->setSamplingParams(sampling);
        runner->setStopTokens({1});

        ASSERT_TRUE(runner->prefill({1, 2, 3, 4, 5}));
        GenerationResult step1 = runner->decodeStep();
        ASSERT_TRUE(step1.success()) << step1.error;
        ASSERT_FALSE(step1.is_complete);
        EXPECT_THAT(step1.tokens,
                    ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN,
                                MockInferenceRunner::MTP_ARGMAX_TOKEN));
        EXPECT_EQ(mock->residentSidecarCountAtLastHostBridge(), 1)
            << "Stop-token-enabled served inference should still prelaunch "
               "the next sidecar before the response bridge when the step did "
               "not complete.";

        const int resident_sidecar_count_after_step1 =
            mock->forwardMTPFromResidentLogicalStateForDeviceSamplingCount();
        GenerationResult step2 = runner->decodeStep();
        ASSERT_TRUE(step2.success()) << step2.error;
        EXPECT_EQ(mock->forwardMTPFromResidentLogicalStateForDeviceSamplingCount(),
                  resident_sidecar_count_after_step1 + 1)
            << "The ready-token step should reuse the sidecar prelaunched "
               "with stop-token metadata present, then queue the next one.";
    }

    TEST_F(Test__PrefillDecodeTransition, ResidentPrelaunchIsDiscardedWhenStopTokenCompletes)
    {
        const std::filesystem::path export_path =
            std::filesystem::temp_directory_path() /
            "llaminar_mtp_stop_token_prelaunch_discard_unit.json";
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
        mock->enableMTPSidecarLogitsStreamHandoff();
        mock->enableMTPDeviceDraftTokenInput();
        mock->requireMTPDecodeEquivalentReplay();
        mock->enableDeviceResidentMTPSpecStatePublication();
        mock->setVerifierAcceptedPrefixScript({1});

        SamplingParams sampling;
        sampling.temperature = 0.8f;
        sampling.top_k = 1;
        sampling.top_p = 0.95f;
        sampling.seed = 123;
        runner->setSamplingParams(sampling);
        runner->setStopTokens({MockInferenceRunner::MTP_ARGMAX_TOKEN});

        ASSERT_TRUE(runner->prefill({1, 2, 3, 4, 5}));
        GenerationResult step = runner->decodeStep();
        ASSERT_TRUE(step.success()) << step.error;
        EXPECT_TRUE(step.is_complete);
        EXPECT_THAT(step.tokens,
                    ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN,
                                MockInferenceRunner::MTP_ARGMAX_TOKEN));
        EXPECT_EQ(mock->residentSidecarCountAtLastHostBridge(), 1)
            << "The sidecar may be prelaunched before the response bridge, "
               "but it must become discardable once host-visible stop status "
               "is known.";

        const auto records = PerfStatsCollector::snapshot({"mtp"});
        const PerfStatRecord *discarded =
            findPerfRecord(records,
                           PerfStatRecord::Kind::Counter,
                           "stochastic_first_sidecar_prelaunch_discarded_complete");
        ASSERT_NE(discarded, nullptr);
        EXPECT_DOUBLE_EQ(discarded->value, 1.0);
        std::filesystem::remove(export_path);
        PerfStatsCollector::reset();
    }

    TEST_F(Test__PrefillDecodeTransition, PenaltyFreeStochasticChainsSidecarFromDeviceDraftToken)
    {
        const std::filesystem::path export_path =
            std::filesystem::temp_directory_path() / "llaminar_mtp_stochastic_device_draft_token_unit.json";
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
                /*sidecar_sample_fusion=*/false,
                {},
                MTPVerifyMode::SpeculativeSampling);
            mock->enableStochasticDeviceSampling();
            mock->enableMTPSidecarPreservesMainState();
            mock->enableMTPShiftedRowReuseFromSidecar();
            mock->enableMTPSidecarLogitsStreamHandoff();
            mock->enableMTPDeviceDraftTokenInput();
            mock->enableMTPSpecStatePublication();
            mock->setVerifierAcceptedPrefixScript({2});

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
                                    MockInferenceRunner::MTP_ARGMAX_TOKEN,
                                    MockInferenceRunner::MTP_ARGMAX_TOKEN));

            EXPECT_EQ(mock->forwardMTPForDeviceSamplingCount(), 0);
            EXPECT_EQ(mock->forwardMTPFromDeviceTargetForDeviceSamplingCount(), 1)
                << "the first stochastic sidecar should consume the deferred "
                   "main-model target token from its device slot";
            EXPECT_EQ(mock->deviceTargetShiftedCommitCount(), 0)
                << "the all-position publication path reuses the first shifted "
                   "row appended by the device-target sidecar";
            EXPECT_EQ(mock->forwardMTPFromDeviceDraftForDeviceSamplingCount(), 1)
                << "depth>1 stochastic sidecar chaining should consume the "
                   "previous sampled draft token from the device slot";
            EXPECT_EQ(mock->forwardMTPFromLastDraftForDeviceSamplingCount(), 0)
                << "the device-token path must not upload draft_tokens.back() "
                   "through the legacy host-token chained sidecar";
            EXPECT_EQ(mock->prepareMTPVerifierInputTokensOnDeviceCount(), 1)
                << "penalty-free stochastic all-position verification should "
                   "compose verifier input IDs from device-resident draft slots";
            EXPECT_EQ(mock->forwardWithDeviceTokenIdsCount(), 1)
                << "the verifier embedding graph should consume the prepared "
                   "device token row while host tokens remain a metadata shadow";
            EXPECT_EQ(mock->prepareMTPVerifierInputTokensDeviceFirstCount(), 1);
            EXPECT_EQ(mock->lastPrepareMTPVerifierFirstDraftSlot(), 0);
            EXPECT_EQ(mock->lastPrepareMTPVerifierDraftTokenCount(), 2);
            EXPECT_EQ(mock->lastPrepareMTPVerifierTotalTokens(), 3);
            EXPECT_EQ(mock->lastForwardDeviceTokenIds(),
                      mock->deviceVerifierInputTokens().data());
            EXPECT_EQ(mock->lastForwardDeviceTokenSeqLen(), 3);
            const auto &device_verifier_tokens =
                mock->deviceVerifierInputTokens();
            EXPECT_THAT(
                std::vector<int32_t>(
                    device_verifier_tokens.begin(),
                    device_verifier_tokens.begin() + 4),
                ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN,
                            MockInferenceRunner::MTP_ARGMAX_TOKEN,
                            MockInferenceRunner::MTP_ARGMAX_TOKEN,
                            -1));
            EXPECT_EQ(mock->lastChainedMTPConditionToken(),
                      MockInferenceRunner::MTP_ARGMAX_TOKEN);
            EXPECT_EQ(mock->deviceDistributionBuildCount(), 4)
                << "first target token, both verifier rows, and terminal ready-token "
                   "use compact distributions; both MTP drafts use "
                   "temperature-only proposal rows";
            EXPECT_EQ(mock->deviceDraftTemperatureProposalCount(), 2);
            EXPECT_EQ(mock->deviceDraftTemperatureProposalDeferredCount(), 2);
            EXPECT_EQ(mock->deviceProbabilityRowsBuildCount(), 0);
            EXPECT_EQ(mock->deviceProcessedRowsBuildCount(), 0)
                << "both verifier comparison rows plus bonus should use compact one-hot rows";
            EXPECT_EQ(mock->deviceDistributionSampleCount(), 1)
                << "only the bonus ready-token sample should be host-visible; "
                   "the first target and draft samples stay device-resident";
            EXPECT_EQ(mock->deviceDistributionSampleDeferredCount(), 1)
                << "only the first target token uses compact deferred sampling; "
                   "both MTP drafts are deferred through proposal slots";
            EXPECT_EQ(mock->deviceDistributionVerifyBatchCount(), 1);
            EXPECT_EQ(mock->verifyStochasticDistributionsBatchOutcomeDeviceFirstCount(), 1);
            EXPECT_EQ(mock->flushPendingMTPWorkCount(), 0)
                << "The verifier input row is materialized on the verifier "
                   "execution stream from target/draft sample slots; flushing "
                   "the sidecar stream here would reintroduce a host sync.";
            EXPECT_TRUE(mock->lastBatchOutcomeUsedVLLMProbabilityRejection())
                << "device-first all-position stochastic verifier should still "
                   "use the vLLM compact one-hot verifier branch";
            EXPECT_EQ(mock->lastBatchOutcomeInverseSampleSeed(), sampling.seed);
            EXPECT_GT(mock->lastBatchOutcomeInverseSampleFirstLogicalPosition(), 0);
            EXPECT_FALSE(mock->batchOutcomeUsedHostDraftTokens())
                << "device-first stochastic MTP keeps all verifier draft "
                   "tokens in device slots until the summary is produced";

            const auto records = PerfStatsCollector::snapshot({"mtp"});
            const PerfStatRecord *device_input =
                findPerfRecordWithTags(records,
                                       PerfStatRecord::Kind::Counter,
                                       "stochastic_sidecar_device_token_inputs",
                                       {{"draft_idx", "1"}});
            ASSERT_NE(device_input, nullptr);
            const PerfStatRecord *verifier_device_input =
                findPerfRecordWithTags(records,
                                       PerfStatRecord::Kind::Counter,
                                       "all_position_verifier_device_token_inputs",
                                       {{"total_tokens", "3"}});
            ASSERT_NE(verifier_device_input, nullptr);
            EXPECT_NE(findPerfRecord(
                          records,
                          PerfStatRecord::Kind::Counter,
                          "sidecar_final_flush_skipped_device_verifier_tokens"),
                      nullptr)
                << "The resident stochastic lane should explicitly report "
                   "that the pre-verifier sidecar flush was skipped.";
            const PerfStatRecord *first_target_input =
                findPerfRecordWithTags(records,
                                       PerfStatRecord::Kind::Counter,
                                       "stochastic_first_sidecar_device_target_inputs",
                                       {});
            ASSERT_NE(first_target_input, nullptr);
            const PerfStatRecord *reused_first_shifted =
                findPerfRecord(records,
                               PerfStatRecord::Kind::Counter,
                               "all_position_initial_shifted_reused_sidecar_rows");
            ASSERT_NE(reused_first_shifted, nullptr);
            const PerfStatRecord *deferred_drafts =
                findPerfRecordWithTags(records,
                                       PerfStatRecord::Kind::Counter,
                                       "mtp_token_stochastic_deferred_host_reads",
                                       {{"draft_idx", "1"}});
            ASSERT_NE(deferred_drafts, nullptr);
        }
        std::filesystem::remove(export_path);
        PerfStatsCollector::reset();
    }

    TEST_F(Test__PrefillDecodeTransition, GroupedGreedyDefersMTPDraftHostReadsWhenVerifierOwnsDeviceSlots)
    {
        const std::filesystem::path export_path =
            std::filesystem::temp_directory_path() /
            "llaminar_mtp_grouped_greedy_deferred_draft_slots_unit.json";
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
                /*sidecar_sample_fusion=*/false);
            mock->enableGroupedOutcomeDeviceResidentPublication(/*rows=*/4);
            mock->enableStochasticDeviceSampling();
            mock->enableDeviceResidentMTPSpecStatePublication();
            mock->hideMTPSpecStatePublicationFromPolicy();
            mock->enableMTPSidecarPreservesMainState();
            mock->enableMTPShiftedRowReuseFromSidecar();
            mock->enableMTPSidecarLogitsStreamHandoff();
            mock->enableMTPDeviceDraftTokenInput();
            mock->setVerifierAcceptedPrefixScript({2});

            ASSERT_TRUE(runner->prefill({1, 2, 3, 4, 5}));
            GenerationResult step = runner->decodeStep();
            ASSERT_TRUE(step.success()) << step.error;
            EXPECT_THAT(step.tokens,
                        ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN,
                                    MockInferenceRunner::MTP_ARGMAX_TOKEN,
                                    MockInferenceRunner::MTP_ARGMAX_TOKEN));

            EXPECT_EQ(mock->sampleMTPLogitsToDeviceDraftSlotCount(), 2);
            EXPECT_EQ(mock->forwardMTPFromDeviceDraftForDeviceSamplingCount(), 1)
                << "The second sidecar should consume the first sampled draft "
                   "directly from the runner-owned device slot.";
            EXPECT_EQ(mock->forwardMTPFromLastDraftForDeviceSamplingCount(), 0)
                << "Grouped greedy must not upload a host draft shadow when "
                   "device draft slots are available.";
            EXPECT_EQ(mock->prepareMTPVerifierInputTokensOnDeviceCount(), 1);
            EXPECT_EQ(mock->verifyGreedyAllPositionBatchOutcomeCount(), 1);
            EXPECT_EQ(mock->publishDeviceResidentMTPSpecStateCount(), 1);

            const auto records = PerfStatsCollector::snapshot({"mtp"});
            EXPECT_NE(findPerfRecordWithTags(
                          records,
                          PerfStatRecord::Kind::Counter,
                          "mtp_token_greedy_device_slot_deferred_host_reads",
                          {{"draft_idx", "0"}}),
                      nullptr);
            EXPECT_EQ(findPerfRecordWithTags(
                          records,
                          PerfStatRecord::Kind::Counter,
                          "mtp_token_greedy_device_slot_deferred_host_reads",
                          {{"draft_idx", "1"}}),
                      nullptr);
            EXPECT_NE(findPerfRecordWithTags(
                          records,
                          PerfStatRecord::Kind::Counter,
                          "mtp_token_greedy_device_slot_samples",
                          {{"draft_idx", "1"}}),
                      nullptr);
            EXPECT_EQ(findPerfRecord(records,
                                     PerfStatRecord::Kind::Counter,
                                     "mtp_token_greedy_device_slot_failures"),
                      nullptr);
            EXPECT_NE(findPerfRecordWithTags(
                          records,
                          PerfStatRecord::Kind::Counter,
                          "grouped_outcome_verifier_device_token_inputs",
                          {{"total_tokens", "3"}}),
                      nullptr);
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
            mock->enableMTPShiftedRowReuseFromSidecar();
            mock->enableMTPSidecarLogitsStreamHandoff();
            mock->enableMTPSpecStatePublication();
            mock->setVerifierAcceptedPrefixScript({0, 1});
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
            EXPECT_EQ(mock->allPositionVerifierSyncDeferralSetCount(), 0);
            EXPECT_EQ(mock->forwardMTPForDeviceSamplingCount(), 0)
                << "penalty-bearing stochastic sampling is history-dependent, "
                   "so it must not use the deferred sidecar logits handoff";
            EXPECT_EQ(mock->flushPendingMTPWorkCount(), 2);
            EXPECT_EQ(mock->sequentialCommitMTPShiftedCount(), 0)
                << "the residual correction remains a pending condition until "
                   "the next verifier row consumes it";
            EXPECT_EQ(mock->forwardCallCount(), forward_count_after_prefill + 1)
                << "only the all-position verifier forward runs in this step";
            EXPECT_EQ(mock->deviceDistributionBuildCount(), 3)
                << "target, rejecting verifier, and terminal ready-token rows "
                   "use compact distributions; the MTP draft proposal is "
                   "temperature-only full-probability";
            EXPECT_EQ(mock->deviceDraftTemperatureProposalCount(), 1);
            EXPECT_EQ(mock->deviceDraftTemperatureProposalDeferredCount(), 0);
            EXPECT_EQ(mock->deviceProbabilityRowsBuildCount(), 0);
            EXPECT_EQ(mock->deviceProcessedRowsBuildCount(), 0)
                << "the rejecting verifier and bonus rows should use compact one-hot rows";
            EXPECT_EQ(mock->deviceDistributionSampleCount(), 2)
                << "target and bonus samples are host-visible; the draft sample "
                   "comes from the proposal path";
            EXPECT_EQ(mock->deviceDistributionVerifyBatchCount(), 1);
            EXPECT_TRUE(mock->lastBatchOutcomeUsedVLLMProbabilityRejection())
                << "residual-correction stochastic lanes should batch through "
                   "the vLLM compact one-hot verifier";
            EXPECT_EQ(mock->lastBatchOutcomeInverseSampleSeed(), sampling.seed);
            EXPECT_GT(mock->lastBatchOutcomeInverseSampleFirstLogicalPosition(), 0);
            EXPECT_EQ(mock->deviceProbabilityVerifyRowCount(), 0);
            EXPECT_EQ(mock->applyMainPenaltiesCount(), 0)
                << "the first stochastic token has no prior sampler history, "
                   "so an empty penalty map must not hit the device hook";
            EXPECT_EQ(mock->applyMTPPenaltiesCount(), 0)
                << "vLLM-style draft proposal ignores draft-side penalties; "
                   "target-side rejection correction owns the final policy";
            EXPECT_EQ(mock->applyAllPositionPenaltiesCount(), 2)
                << "the current vLLM batch prepares verifier and bonus rows "
                   "before the summary knows whether the bonus is consumed";

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
                                        {"correction_replay_tokens", "0"},
                                        {"deferred_correction_condition_tokens", "1"}});
            ASSERT_NE(trace, nullptr);

            const int forward_count_after_reject = mock->forwardCallCount();
            const int mtp_count_after_reject = mock->forwardMTPCount();

            GenerationResult step2 = runner->decodeStep();
            ASSERT_TRUE(step2.success()) << step2.error;
            EXPECT_THAT(step2.tokens, ElementsAre(MockInferenceRunner::MTP_ARGMAX_TOKEN))
                << "the pending correction is verifier input, not a newly emitted token";
            EXPECT_EQ(mock->lastMTPConditionToken(),
                      MockInferenceRunner::VERIFY_REJECT_TOKEN)
                << "the sidecar should draft from the rejected correction token";
            EXPECT_EQ(mock->forwardMTPCount(), mtp_count_after_reject + 1);
            EXPECT_EQ(mock->forwardCallCount(), forward_count_after_reject + 1)
                << "pending correction row should skip the standalone condition_forward";

            const auto records_after_step2 = PerfStatsCollector::snapshot({"mtp"});
            const PerfStatRecord *pending_skip =
                findPerfRecord(records_after_step2,
                               PerfStatRecord::Kind::Counter,
                               "condition_forward_skipped_pending_condition");
            ASSERT_NE(pending_skip, nullptr);
            EXPECT_DOUBLE_EQ(pending_skip->value, 1.0);
            const PerfStatRecord *pending_trace =
                findPerfRecordWithTags(records_after_step2,
                                       PerfStatRecord::Kind::Counter,
                                       "acceptance_trace",
                                       {{"draft_tokens",
                                         std::string("4,") +
                                             std::to_string(MockInferenceRunner::MTP_ARGMAX_TOKEN)},
                                        {"pending_condition_input", "true"},
                                        {"output_tokens", "1"},
                                        {"next_pending_condition_token", "none"}});
            ASSERT_NE(pending_trace, nullptr);
        }
        std::filesystem::remove(export_path);
        PerfStatsCollector::reset();
    }

    TEST_F(Test__PrefillDecodeTransition, AllPositionSpecPublicationStochasticHostAcceptsOnCPU)
    {
        const std::filesystem::path export_path =
            std::filesystem::temp_directory_path() / "llaminar_mtp_all_position_stochastic_host_accept_unit.json";
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
                /*mtp_draft_tokens=*/1,
                /*chained_mtp_support=*/false,
                /*sidecar_sample_fusion=*/false,
                {},
                MTPVerifyMode::SpeculativeSampling);
            mock->enableMTPSidecarPreservesMainState();
            mock->enableMTPShiftedRowReuseFromSidecar();
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
            EXPECT_EQ(mock->publishMTPSpecStateCount(), 1);
            EXPECT_EQ(mock->forwardCallCount(), forward_count_after_prefill + 1)
                << "CPU host stochastic publication should run one all-position verifier forward.";
            EXPECT_EQ(mock->restoreCount(), 0)
                << "graph-native sidecar state lets CPU publish without restoring the verifier base.";
            EXPECT_EQ(mock->deviceDistributionBuildCount(), 0);
            EXPECT_EQ(mock->deviceDistributionSampleCount(), 0);
            EXPECT_EQ(mock->deviceDistributionVerifyBatchCount(), 0);
            EXPECT_EQ(mock->applyMainPenaltiesCount(), 0);
            EXPECT_EQ(mock->applyMTPPenaltiesCount(), 0);
            EXPECT_EQ(mock->applyAllPositionPenaltiesCount(), 0);
            EXPECT_EQ(mock->commitMTPShiftedCount(), 1);
            EXPECT_EQ(mock->lastCommitMTPAlreadyAppended(), 1);
            EXPECT_EQ(mock->lastCommitMTPMainForwardTokenCount(), 2);

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
            EXPECT_EQ(probe.mtp_transaction_rollbacks, 0u);

            const auto records = PerfStatsCollector::snapshot({"mtp"});
            const PerfStatRecord *host_target =
                findPerfRecord(records,
                               PerfStatRecord::Kind::Timer,
                               "all_position_stochastic_host_target_distribution");
            ASSERT_NE(host_target, nullptr);
            const PerfStatRecord *host_terminal =
                findPerfRecord(records,
                               PerfStatRecord::Kind::Counter,
                               "stochastic_terminal_host_samples");
            ASSERT_NE(host_terminal, nullptr);
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

    TEST_F(Test__PrefillDecodeTransition, AllPositionSpecPublicationStochasticHostRejectsOnCPU)
    {
        const std::filesystem::path export_path =
            std::filesystem::temp_directory_path() / "llaminar_mtp_all_position_stochastic_host_reject_unit.json";
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
                /*mtp_draft_tokens=*/1,
                /*chained_mtp_support=*/false,
                /*sidecar_sample_fusion=*/false,
                {},
            MTPVerifyMode::SpeculativeSampling);
            mock->enableMTPSidecarPreservesMainState();
            mock->enableMTPShiftedRowReuseFromSidecar();
            mock->enableMTPSpecStatePublication();

            SamplingParams sampling;
            sampling.temperature = 0.8f;
            sampling.top_k = 2;
            sampling.top_p = 0.95f;
            sampling.presence_penalty = 0.25f;
            sampling.seed = 456;
            runner->setSamplingParams(sampling);
            mock->setVerifierAcceptedPrefixScript({0, 1});

            ASSERT_TRUE(runner->prefill({1, 2, 3, 4, 5}));
            const int forward_count_after_prefill = mock->forwardCallCount();

            GenerationResult step1 = runner->decodeStep();
            ASSERT_TRUE(step1.success()) << step1.error;
            EXPECT_THAT(step1.tokens,
                        ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN,
                                    MockInferenceRunner::VERIFY_REJECT_TOKEN));

            EXPECT_EQ(mock->setAllPositionCount(), 2);
            EXPECT_EQ(mock->publishMTPSpecStateCount(), 1);
            EXPECT_EQ(mock->forwardCallCount(), forward_count_after_prefill + 1)
                << "CPU host stochastic rejection should avoid same-step correction forward.";
            EXPECT_EQ(mock->restoreCount(), 0);
            EXPECT_EQ(mock->deviceDistributionBuildCount(), 0);
            EXPECT_EQ(mock->deviceDistributionSampleCount(), 0);
            EXPECT_EQ(mock->deviceDistributionVerifyBatchCount(), 0);
            EXPECT_EQ(mock->commitMTPShiftedCount(), 0);
            EXPECT_EQ(mock->sequentialCommitMTPShiftedCount(), 0)
                << "host stochastic rejection defers the correction shifted row "
                   "just like the GPU resident path";

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
            EXPECT_EQ(probe.mtp_transaction_rollbacks, 0u);

            const auto records = PerfStatsCollector::snapshot({"mtp"});
            const PerfStatRecord *host_residual =
                findPerfRecord(records,
                               PerfStatRecord::Kind::Counter,
                               "stochastic_residual_host_samples");
            ASSERT_NE(host_residual, nullptr);
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
                                        {"correction_replay_tokens", "0"},
                                        {"deferred_correction_condition_tokens", "1"}});
            ASSERT_NE(trace, nullptr);

            const int forward_count_after_reject = mock->forwardCallCount();
            const int mtp_count_after_reject = mock->forwardMTPCount();

            GenerationResult step2 = runner->decodeStep();
            ASSERT_TRUE(step2.success()) << step2.error;
            EXPECT_THAT(step2.tokens, ElementsAre(MockInferenceRunner::MTP_ARGMAX_TOKEN))
                << "CPU should also consume a rejected correction as a verifier row";
            EXPECT_EQ(mock->lastMTPConditionToken(),
                      MockInferenceRunner::VERIFY_REJECT_TOKEN);
            EXPECT_EQ(mock->forwardMTPCount(), mtp_count_after_reject + 1);
            EXPECT_EQ(mock->forwardCallCount(), forward_count_after_reject + 1)
                << "CPU pending correction row should avoid the standalone condition forward";

            const auto records_after_step2 = PerfStatsCollector::snapshot({"mtp"});
            const PerfStatRecord *pending_skip =
                findPerfRecord(records_after_step2,
                               PerfStatRecord::Kind::Counter,
                               "condition_forward_skipped_pending_condition");
            ASSERT_NE(pending_skip, nullptr);
            EXPECT_DOUBLE_EQ(pending_skip->value, 1.0);
            const PerfStatRecord *pending_trace =
                findPerfRecordWithTags(records_after_step2,
                                       PerfStatRecord::Kind::Counter,
                                       "acceptance_trace",
                                       {{"pending_condition_input", "true"},
                                        {"output_tokens", "1"},
                                        {"next_pending_condition_token", "none"}});
            ASSERT_NE(pending_trace, nullptr);
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
        sampling.temperature = 0.0f;
        sampling.seed = 1234;
        runner->setSamplingParams(sampling);

        GenerationResult step2 = runner->decodeStep();
        EXPECT_FALSE(step2.success());
        EXPECT_NE(step2.error.find("Ready MTP verifier token"), std::string::npos);
        EXPECT_EQ(mock->restoreCount(), 1)
            << "the ready-token sampling-contract guard fails before mutating "
               "runner state, so it should not capture and restore a fresh "
               "rollback checkpoint.";
        EXPECT_EQ(mock->forwardMTPCount(), 1)
            << "the stale ready-token guard must fail before launching another sidecar";
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
        EXPECT_EQ(mock->deviceDistributionBuildCount(), 3)
            << "first target token, verifier row, and terminal ready-token "
               "use compact distributions; MTP draft uses the temperature "
               "proposal path";
        EXPECT_EQ(mock->deviceDraftTemperatureProposalCount(), 1);
        EXPECT_EQ(mock->deviceDraftTemperatureProposalDeferredCount(), 0);
        EXPECT_EQ(mock->deviceDistributionSampleCount(), 2)
            << "first target token and terminal ready-token use compact sampling; "
               "the MTP draft sample comes from the proposal path";
        EXPECT_EQ(mock->deviceDistributionVerifyCount(), 0);
        EXPECT_EQ(mock->deviceDistributionVerifyBatchCount(), 1);
        EXPECT_EQ(mock->applyMainPenaltiesCount(), 2)
            << "empty first-token penalties are skipped; the sequential verifier "
               "and ready-token rows still apply non-empty history penalties";
        EXPECT_EQ(mock->applyMTPPenaltiesCount(), 0)
            << "vLLM-style draft proposal ignores draft-side penalties; "
               "target-side rejection correction owns the final policy";
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
        EXPECT_EQ(mock->deviceDistributionBuildCount(), 3)
            << "first target token, sequential target row, and ready token "
               "use compact distributions; MTP draft uses the proposal path";
        EXPECT_EQ(mock->deviceDraftTemperatureProposalCount(), 1);
        EXPECT_EQ(mock->deviceDraftTemperatureProposalDeferredCount(), 0);
        EXPECT_EQ(mock->deviceDistributionSampleCount(), 2);
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
        EXPECT_EQ(mock->deviceDistributionBuildCount(), 3)
            << "first target token, verifier row, and terminal row use compact "
               "distributions; MTP draft uses the proposal path";
        EXPECT_EQ(mock->deviceDraftTemperatureProposalCount(), 1);
        EXPECT_EQ(mock->deviceDraftTemperatureProposalDeferredCount(), 0);
        EXPECT_EQ(mock->deviceDistributionSampleCount(), 2)
            << "first target token and post-correction ready-token use compact "
               "sampling; the MTP draft sample comes from the proposal path";
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

    TEST_F(Test__PrefillDecodeTransition, MTPDynamicPPTopologyUsesCentralDepthController)
    {
        MTPDepthPolicyConfig dynamic_depth;
        dynamic_depth.mode = MTPDepthPolicyMode::Dynamic;
        dynamic_depth.min_depth = 1;
        dynamic_depth.max_depth = 3;
        dynamic_depth.initial_depth = 3;

        auto [runner, mock] = createRunner(
            /*mtp_enabled=*/true,
            /*mtp_accept=*/true,
            /*mtp_unsupported_reason=*/{},
            /*mpi_ctx=*/nullptr,
            /*mtp_token_coordination=*/false,
            /*hide_local_logits=*/false,
            /*primary_device=*/DeviceId::cpu(),
            /*mtp_draft_tokens=*/3,
            /*chained_mtp_support=*/true,
            /*sidecar_sample_fusion=*/false,
            dynamic_depth,
            MTPVerifyMode::Greedy,
            /*local_pp_topology=*/true);

        std::vector<int32_t> prompt = {1, 2, 3, 4, 5};
        ASSERT_TRUE(runner->prefill(prompt)) << runner->lastError();
        EXPECT_EQ(mock->forwardCallCount(), 1)
            << "The in-process OrchestrationRunner owns the dynamic-depth "
               "decision before PP fan-out.";

        GenerationResult step1 = runner->decodeStep();
        ASSERT_TRUE(step1.success()) << step1.error;
        EXPECT_GE(mock->forwardMTPCount(), 1)
            << "Dynamic PP MTP should enter the same sidecar draft path as a "
               "fixed-depth PP request once decode begins.";

        auto probe = runner->prefixStateProbe();
        EXPECT_TRUE(probe.mtp_config_enabled);
        EXPECT_FALSE(probe.mtp_bypassed);
        EXPECT_EQ(probe.mtp_bypasses, 0u);
        EXPECT_TRUE(probe.mtp_request.adaptive_depth_enabled);
        EXPECT_EQ(probe.mtp_request.depth_policy_mode, "dynamic");
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

    TEST_F(Test__PrefillDecodeTransition, GlobalTPMTPDepthThreeFansOutChainedSidecars)
    {
        auto mpi = std::make_shared<llaminar2::test::MockMPIContext>(0, 2);
        auto child = std::make_unique<MockInferenceRunner>();
        auto *child_ptr = child.get();
        child_ptr->enableMTP(/*accept_mtp_token=*/true);
        child_ptr->enableChainedMTPDrafts();

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
        config.mtp.draft_tokens = 3;
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
        EXPECT_EQ(child_ptr->forwardMTPCount(), 1);
        EXPECT_EQ(child_ptr->forwardMTPFromLastDraftCount(), 2);
        EXPECT_EQ(child_ptr->lastMTPConditionToken(), MockInferenceRunner::PREFILL_ARGMAX_TOKEN);
        EXPECT_EQ(child_ptr->lastChainedMTPConditionToken(), MockInferenceRunner::MTP_ARGMAX_TOKEN);
        EXPECT_EQ(child_ptr->lastChainedMTPPositionId(), 7);

        auto probe = runner->prefixStateProbe();
        EXPECT_FALSE(probe.mtp_bypassed) << probe.mtp_bypass_reason;
        EXPECT_EQ(probe.mtp_draft_steps, 3u);
        EXPECT_GE(probe.mtp_verifier_runs, 1u);
        EXPECT_GE(probe.mtp_verifier_token_count, 4u);
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

    TEST_F(Test__PrefillDecodeTransition, LocalTPDynamicMTPDepthUsesOneRankWideController)
    {
        MTPDepthPolicyConfig depth_policy;
        depth_policy.mode = MTPDepthPolicyMode::Dynamic;
        depth_policy.min_depth = 1;
        depth_policy.max_depth = 3;
        depth_policy.initial_depth = 3;
        depth_policy.window_size = 1;
        depth_policy.min_samples = 1;
        depth_policy.cooldown_steps = 0;
        depth_policy.use_generated_policy = false;

        auto harness = createLocalTPRunner(
            /*mtp_accept=*/true,
            /*column_parallel_logits=*/false,
            /*devices=*/{},
            /*mtp_draft_tokens=*/3,
            depth_policy,
            /*spec_state_publication=*/true);
        harness.child0->setVerifierAcceptedPrefixScript({3});
        harness.child1->setVerifierAcceptedPrefixScript({3});

        std::vector<int32_t> prompt = {1, 2, 3, 4, 5};
        ASSERT_TRUE(harness.runner->prefill(prompt));

        GenerationResult step1 = harness.runner->decodeStep();
        ASSERT_TRUE(step1.success()) << step1.error;

        EXPECT_EQ(harness.child0->forwardMTPCount(), 1);
        EXPECT_EQ(harness.child1->forwardMTPCount(), 1);
        EXPECT_EQ(harness.child0->forwardMTPFromLastDraftCount(), 2);
        EXPECT_EQ(harness.child1->forwardMTPFromLastDraftCount(), 2);
        EXPECT_EQ(harness.child0->publishMTPSpecStateCount(), 1);
        EXPECT_EQ(harness.child1->publishMTPSpecStateCount(), 1);

        const auto probe = harness.runner->prefixStateProbe();
        EXPECT_FALSE(probe.mtp_bypassed) << probe.mtp_bypass_reason;
        EXPECT_TRUE(probe.mtp_request.adaptive_depth_enabled);
        EXPECT_EQ(probe.mtp_request.depth_policy_mode, "dynamic");
        EXPECT_EQ(probe.mtp_max_depth, 3);
        EXPECT_GE(probe.mtp_depth_policy_windows, 1u);
        EXPECT_EQ(probe.mtp_draft_steps, 3u);
        EXPECT_EQ(probe.mtp_accepted_tokens, 3u);
        EXPECT_EQ(probe.mtp_rejected_tokens, 0u);
    }

    TEST_F(Test__PrefillDecodeTransition, LocalTPGPUAllPositionPublicationUsesShardedRowSampler)
    {
        auto harness = createLocalTPRunner(
            /*mtp_accept=*/true,
            /*column_parallel_logits=*/true,
            {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)},
            /*mtp_draft_tokens=*/1,
            {},
            /*spec_state_publication=*/true);
        harness.child0->enableMTPSidecarPreservesMainState();
        harness.child1->enableMTPSidecarPreservesMainState();
        harness.child0->requireMTPDecodeEquivalentReplay();
        harness.child1->requireMTPDecodeEquivalentReplay();
        harness.child0->setVerifierAcceptedPrefixScript({1});
        harness.child1->setVerifierAcceptedPrefixScript({1});

        ASSERT_TRUE(harness.runner->prefill({1, 2, 3, 4, 5}));

        GenerationResult step = harness.runner->decodeStep();
        ASSERT_TRUE(step.success()) << step.error;
        EXPECT_THAT(step.tokens,
                    ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN,
                                MockInferenceRunner::MTP_ARGMAX_TOKEN));

        EXPECT_EQ(harness.child0->verifyGreedyAllPositionBatchOutcomeCount(), 0);
        EXPECT_EQ(harness.child1->verifyGreedyAllPositionBatchOutcomeCount(), 0)
            << "LocalTP shards verifier logits, so the rank must not call a "
               "single-child compact device reducer.";
        EXPECT_EQ(harness.child0->sampleAllPositionLogitsBatchedCount(), 0);
        EXPECT_EQ(harness.child1->sampleAllPositionLogitsBatchedCount(), 0)
            << "Rank-level LocalTP sampling consumes child LogitsLocalInfo "
               "directly instead of asking one child to sample all rows.";
        EXPECT_EQ(harness.child0->publishMTPSpecStateCount(), 1);
        EXPECT_EQ(harness.child1->publishMTPSpecStateCount(), 1);
        EXPECT_EQ(harness.child0->setAllPositionCount(), 2);
        EXPECT_EQ(harness.child1->setAllPositionCount(), 2);
    }

    TEST_F(Test__PrefillDecodeTransition, MPIDynamicMTPDepthBroadcastsRankZeroDecision)
    {
        MTPDepthPolicyConfig depth_policy;
        depth_policy.mode = MTPDepthPolicyMode::Dynamic;
        depth_policy.min_depth = 1;
        depth_policy.max_depth = 3;
        depth_policy.initial_depth = 3;
        depth_policy.window_size = 1;
        depth_policy.min_samples = 1;
        depth_policy.cooldown_steps = 0;
        depth_policy.use_generated_policy = false;

        auto mpi = std::make_shared<test::MockMPIContext>(0, 2);
        auto *mpi_raw = mpi.get();
        auto [runner, mock] = createRunner(
            /*mtp_enabled=*/true,
            /*mtp_accept=*/true,
            /*mtp_unsupported_reason=*/{},
            std::move(mpi),
            /*mtp_token_coordination=*/true,
            /*hide_local_logits=*/false,
            DeviceId::cpu(),
            /*mtp_draft_tokens=*/3,
            /*chained_mtp_support=*/true,
            /*sidecar_sample_fusion=*/false,
            depth_policy);

        ASSERT_TRUE(runner->prefill({1, 2, 3, 4, 5})) << runner->lastError();
        mpi_raw->reset_call_counts();

        GenerationResult step1 = runner->decodeStep();
        ASSERT_TRUE(step1.success()) << step1.error;

        /*
         * Dynamic MPI/NodeLocalTP execution must coordinate the scalar draft
         * depth before launching sidecars.  The mock broadcast is intentionally
         * no-op for data, so this test verifies the structural contract: MTP no
         * longer hard-fails under MPI and the broadcast hook is exercised.
         */
        EXPECT_GE(mpi_raw->broadcast_call_count(), 2u)
            << "decodeStep() checks the requested depth once before entering MTP "
               "and once inside the MTP transaction";
        EXPECT_EQ(mock->forwardMTPCount(), 1);
        EXPECT_EQ(mock->forwardMTPFromLastDraftCount(), 2);

        const auto probe = runner->prefixStateProbe();
        EXPECT_FALSE(probe.mtp_bypassed) << probe.mtp_bypass_reason;
        EXPECT_TRUE(probe.mtp_request.adaptive_depth_enabled);
        EXPECT_EQ(probe.mtp_request.depth_policy_mode, "dynamic");
        EXPECT_EQ(probe.mtp_draft_steps, 3u);
        EXPECT_EQ(probe.mtp_accepted_tokens, 3u);
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
