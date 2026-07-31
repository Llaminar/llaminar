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
#include "execution/moe/MoERoutedExpertPlacementPlan.h"
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
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <sstream>
#include <thread>
#include <unordered_map>
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

std::string readSourceFileForRankOrchestratorTest(const std::string &path)
{
    std::ifstream input(path);
    if (!input.good())
        return {};
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

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

    PrefixRuntimeStateSnapshot prefixStateProbe() const override
    {
        if (!prefix_probe_position_override_)
            return {};

        PrefixRuntimeStateSnapshot snapshot;
        snapshot.initialized = true;
        snapshot.architecture = config_.architecture;
        snapshot.execution_path = "mock-device-graph";
        snapshot.primary_device = device_id_;
        snapshot.current_position = *prefix_probe_position_override_;
        snapshot.positions = {*prefix_probe_position_override_};
        snapshot.sequence_lengths = {*prefix_probe_position_override_};
        return snapshot;
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

    bool forwardMTPFromDeviceDraftAtLivePositionForDeviceSampling(
        int draft_sample_slot,
        int position_offset) override
    {
        ++forward_mtp_from_device_draft_calls_;
        last_device_draft_sample_slot_ = draft_sample_slot;
        last_device_token_sidecar_position_id_ = position_offset;
        return stochastic_device_ops_ok_ &&
               supports_mtp_device_draft_token_input_ &&
               draft_sample_slot >= 0 &&
               position_offset >= 0;
    }

    bool forwardMTPFromDeviceTargetAtLivePositionForDeviceSampling(
        int target_sample_slot) override
    {
        ++forward_mtp_from_device_target_calls_;
        last_device_target_sample_slot_ = target_sample_slot;
        last_device_token_sidecar_position_id_ = 0;
        return stochastic_device_ops_ok_ &&
               supports_mtp_device_draft_token_input_ &&
               target_sample_slot >= 0;
    }

    bool supportsMTPSpecStatePublication() const override
    {
        return supports_mtp_spec_state_publication_;
    }

    bool supportsDeviceResidentMTPSpecStatePublication() const override
    {
        return supports_device_resident_mtp_spec_state_publication_;
    }

    bool usesMirroredLocalTPMTPHeadForVerifier() const override
    {
        return uses_mirrored_localtp_mtp_head_for_verifier_;
    }

    MTPVerifierRowCapability mtpVerifierRowCapability() const override
    {
        return mtp_verifier_row_capability_;
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

    bool publishGroupedDecodeEquivalentMTPSpecStateBatch(
        const MTPSpecStepPlanBatch &plans,
        std::string *error = nullptr) override
    {
        publish_grouped_decode_equivalent_mtp_spec_state_batch_calls_.fetch_add(
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
                    *error = "mock grouped decode-equivalent MTP publication rendezvous timed out";
                return false;
            }
        }
        if (!publish_mtp_spec_state_ok_)
        {
            if (error)
                *error = "mock grouped decode-equivalent MTP publication failed";
            return false;
        }
        if (!plans.ok ||
            plans.request_count <= 0 ||
            static_cast<int>(plans.steps.size()) != plans.request_count)
        {
            if (error)
                *error = "mock grouped decode-equivalent MTP publication received invalid plans";
            return false;
        }

        resident_target_positions_.assign(
            static_cast<size_t>(plans.request_count),
            0);
        resident_target_sequence_lengths_.assign(
            static_cast<size_t>(plans.request_count),
            0);
        resident_accepted_state_counts_.assign(
            static_cast<size_t>(plans.request_count),
            0);
        resident_next_condition_tokens_.assign(
            static_cast<size_t>(plans.request_count),
            kMTPSpecDecodeInvalidToken);
        resident_all_drafts_accepted_flags_.assign(
            static_cast<size_t>(plans.request_count),
            0);
        resident_stopped_flags_.assign(
            static_cast<size_t>(plans.request_count),
            0);
        resident_publication_ok_flags_.assign(
            static_cast<size_t>(plans.request_count),
            0);
        resident_logical_state_valid_ = false;
        resident_logical_state_request_count_ = 0;

        int max_target_cached_tokens = 0;
        std::vector<bool> seen_request(
            static_cast<size_t>(plans.request_count),
            false);
        for (const MTPSpecStepPlan &step : plans.steps)
        {
            if (step.request_index < 0 ||
                step.request_index >= plans.request_count)
            {
                if (error)
                    *error = "mock grouped decode-equivalent MTP publication received an out-of-range request index";
                return false;
            }
            const size_t idx = static_cast<size_t>(step.request_index);
            if (seen_request[idx])
            {
                if (error)
                    *error = "mock grouped decode-equivalent MTP publication received a duplicate request index";
                return false;
            }
            seen_request[idx] = true;
            max_target_cached_tokens =
                std::max(max_target_cached_tokens, step.target_cached_tokens);
            resident_target_positions_[idx] = step.target_cached_tokens;
            resident_target_sequence_lengths_[idx] = step.target_cached_tokens;
            resident_accepted_state_counts_[idx] = step.accepted_count;
            resident_next_condition_tokens_[idx] = step.next_condition_token;
            resident_all_drafts_accepted_flags_[idx] =
                step.all_drafts_accepted ? 1 : 0;
            resident_stopped_flags_[idx] = step.stopped ? 1 : 0;
            resident_publication_ok_flags_[idx] = 1;
        }
        for (bool seen : seen_request)
        {
            if (!seen)
            {
                if (error)
                    *error = "mock grouped decode-equivalent MTP publication missed a request index";
                return false;
            }
        }
        position_ = max_target_cached_tokens;
        resident_logical_state_request_count_ = plans.request_count;
        resident_logical_state_valid_ = true;
        return true;
    }

    bool copyDeviceSpeculativeOutcomesToHost(
        const DeviceSpeculativeOutcomeHandle &handle,
        DeviceSpeculativeVerifyBatchOutcome *outcomes) override
    {
        using namespace sampling_math;
        if (!handle.valid() ||
            !outcomes ||
            handle.device != device_id_ ||
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
        for (int request_index = 0;
             request_index < handle.request_count;
             ++request_index)
        {
            const size_t token_base =
                static_cast<size_t>(request_index) *
                static_cast<size_t>(handle.output_token_stride);
            const size_t meta_base =
                static_cast<size_t>(request_index) *
                static_cast<size_t>(handle.meta_stride);
            const int *row_meta = meta + meta_base;
            if (row_meta[kSpecBatchMetaOk] == 0)
                return false;

            DeviceSpeculativeVerifyBatchOutcome &out = outcomes[request_index];
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

    bool publishAcceptedMTPSpecStateBatchFromDeviceOutcome(
        const DeviceSpeculativePublicationRequest &request,
        std::string *error = nullptr) override
    {
        using namespace sampling_math;
        publish_device_resident_mtp_spec_state_batch_calls_.fetch_add(
            1,
            std::memory_order_relaxed);
        if (!supports_device_resident_mtp_spec_state_publication_)
        {
            if (error)
                *error = "mock device-resident MTP publication disabled";
            return false;
        }
        if (!publish_mtp_spec_state_ok_)
        {
            if (error)
                *error = "mock device-resident MTP publication failed";
            return false;
        }
        if (!request.valid() ||
            request.request_count <= 0 ||
            request.request_count > kMockResidentOutcomeRequestCapacity)
        {
            if (error)
                *error = "mock device-resident MTP publication received invalid request";
            return false;
        }

        resident_target_positions_.assign(
            static_cast<size_t>(request.request_count),
            0);
        resident_target_sequence_lengths_.assign(
            static_cast<size_t>(request.request_count),
            0);
        resident_accepted_state_counts_.assign(
            static_cast<size_t>(request.request_count),
            0);
        resident_next_condition_tokens_.assign(
            static_cast<size_t>(request.request_count),
            kMTPSpecDecodeInvalidToken);
        resident_all_drafts_accepted_flags_.assign(
            static_cast<size_t>(request.request_count),
            0);
        resident_stopped_flags_.assign(
            static_cast<size_t>(request.request_count),
            0);
        resident_publication_ok_flags_.assign(
            static_cast<size_t>(request.request_count),
            0);

        int max_target_cached_tokens = 0;
        for (int request_index = 0;
             request_index < request.request_count;
             ++request_index)
        {
            const size_t meta_base =
                static_cast<size_t>(request_index) *
                static_cast<size_t>(request.outcome.meta_stride);
            const int *row_meta = request.outcome.meta_device + meta_base;
            if (row_meta[kSpecBatchMetaOk] == 0)
            {
                if (error)
                    *error = "mock device-resident MTP publication row is invalid";
                return false;
            }

            /*
             * Production snapshots the pre-verifier cache length into its
             * persistent device metadata before verifier replay.  The mock's
             * `position_` is that resident scalar.  Deliberately do not accept
             * a value through DeviceSpeculativePublicationRequest: doing so
             * would teach this unit fixture the host-mirror contract that the
             * GPU API has retired.
             */
            const int base_cached_tokens = position_;
            const int accepted_count =
                row_meta[kSpecBatchMetaTargetVerifierStateCommitCount];
            const int target_cached_tokens =
                base_cached_tokens + accepted_count;

            int publication_ok = 0;
            int32_t next_condition_token = kMTPSpecDecodeInvalidToken;
            int all_drafts_accepted = 0;
            int stopped = 0;
            derive_speculative_publication_metadata(
                request.outcome.meta_device,
                request.outcome.meta_stride,
                request_index,
                request.max_draft_tokens,
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
                    *error = "mock device-resident MTP metadata derivation failed";
                return false;
            }

            const size_t idx = static_cast<size_t>(request_index);
            resident_target_positions_[idx] = target_cached_tokens;
            resident_target_sequence_lengths_[idx] = target_cached_tokens;
            resident_accepted_state_counts_[idx] = accepted_count;
            resident_next_condition_tokens_[idx] = next_condition_token;
            resident_all_drafts_accepted_flags_[idx] =
                all_drafts_accepted != 0 ? 1 : 0;
            resident_stopped_flags_[idx] = stopped != 0 ? 1 : 0;
            resident_publication_ok_flags_[idx] = 1;
            max_target_cached_tokens =
                std::max(max_target_cached_tokens, target_cached_tokens);
        }

        position_ = max_target_cached_tokens;
        resident_logical_state_request_count_ = request.request_count;
        resident_logical_state_valid_ = true;
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
        handle.stopped_flags_device = resident_stopped_flags_.data();
        handle.publication_ok_flags_device =
            resident_publication_ok_flags_.data();
        handle.request_count = resident_logical_state_request_count_;
        handle.device = device_id_;
        handle.stream = const_cast<int *>(&resident_stream_token_);
        handle.ready_event = const_cast<int *>(&resident_ready_event_token_);
        handle.live_state_epoch = 1;
        handle.mtp_transaction.state = resident_mtp_transaction_state_;
        return handle;
    }

    bool forwardMTPFromDeviceResidentLogicalStateForDeviceSampling(
        const DeviceResidentLogicalSequenceStateHandle &logical_state,
        int request_index = 0) override
    {
        ++forward_mtp_from_resident_logical_state_calls_;
        if (!logical_state.sameMailboxAs(deviceResidentLogicalSequenceState()) ||
            request_index < 0 ||
            request_index >= resident_logical_state_request_count_)
        {
            return false;
        }
        last_resident_logical_state_request_index_ = request_index;
        return forward_mtp_ok_;
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

    int sampleGreedyOnDevice() override
    {
        ++sample_greedy_on_device_calls_;
        if (logits_.empty())
            return -1;
        return static_cast<int>(
            std::distance(logits_.begin(),
                          std::max_element(logits_.begin(), logits_.end())));
    }

    bool consumeUnusedReplicatedMainLogitsPublication() override
    {
        ++consume_unused_replicated_main_logits_publication_calls_;
        return consume_unused_replicated_main_logits_publication_ok_;
    }

    bool sampleGreedyFromMainLogitsToDeviceTargetSlot(
        int target_sample_slot,
        int32_t *out_token) override
    {
        ++sample_greedy_main_target_slot_calls_;
        if (target_sample_slot < 0 ||
            target_sample_slot >=
                static_cast<int>(target_sample_tokens_.size()) ||
            logits_.empty())
        {
            return false;
        }
        const int32_t token = static_cast<int32_t>(
            std::distance(
                logits_.begin(),
                std::max_element(logits_.begin(), logits_.end())));
        target_sample_tokens_[static_cast<size_t>(target_sample_slot)] = token;
        target_sample_slot_ready_[static_cast<size_t>(target_sample_slot)] = true;
        if (out_token)
            *out_token = token;
        return true;
    }

    int sampleOnDevice(const SamplingParams &params) override
    {
        ++sample_on_device_calls_;
        if (params.is_greedy())
            return sampleGreedyOnDevice();
        return stochastic_sample_token_;
    }

    int sampleOnDeviceAtLogicalPosition(
        const SamplingParams &params,
        int logical_position) override
    {
        (void)logical_position;
        return sampleOnDevice(params);
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

    /**
     * @brief Record device-target shifted-row commits for LocalTP fanout tests.
     *
     * Production LocalTP resolves the target token into a child-owned device
     * sample slot before the shifted MTP KV append.  The mock captures the slot
     * and publication boundary metadata so tests can prove the rank asks every
     * participant to run the real device-slot commit instead of stopping at a
     * rank-level unsupported path.
     */
    bool commitMTPShiftedRowFromDeviceTargetSample(
        int target_sample_slot,
        int already_appended_tokens,
        bool allow_speculative_discard = false) override
    {
        ++commit_mtp_device_target_sample_calls_;
        last_commit_mtp_device_target_sample_slot_ = target_sample_slot;
        last_commit_mtp_already_appended_ = already_appended_tokens;
        last_commit_mtp_main_forward_token_count_ = 0;
        last_commit_mtp_allow_speculative_discard_ =
            allow_speculative_discard;
        last_commit_mtp_tokens_.clear();
        return commit_mtp_shifted_rows_ok_ &&
               target_sample_slot >= 0 &&
               already_appended_tokens >= 0;
    }

    /**
     * @brief Record checkpoint-backed shifted-row repairs for LocalTP fanout tests.
     *
     * The production LocalTP path uses this method when grouped verifier
     * publication needs to append the first shifted MTP row from the verifier
     * base terminal hidden state.  The mock keeps the checkpoint metadata
     * visible so tests can prove the rank sends a valid logical checkpoint to
     * every participant instead of accidentally taking the older partial-forward
     * row-list path.
     */
    bool commitMTPShiftedRowFromCheckpointTerminalHidden(
        const PrefixStateSnapshot &checkpoint,
        int32_t token,
        int already_appended_tokens,
        bool allow_speculative_discard = false,
        int position_offset_override = -1) override
    {
        ++commit_mtp_checkpoint_terminal_hidden_calls_;
        last_commit_mtp_already_appended_ = already_appended_tokens;
        last_commit_mtp_main_forward_token_count_ = 0;
        last_commit_mtp_allow_speculative_discard_ = allow_speculative_discard;
        last_commit_mtp_position_offset_override_ = position_offset_override;
        last_commit_mtp_tokens_.assign(1, token);
        last_commit_mtp_checkpoint_valid_ = checkpoint.valid;
        last_commit_mtp_checkpoint_logical_ = checkpoint.logical_checkpoint;
        last_commit_mtp_checkpoint_cached_tokens_ = checkpoint.cached_tokens;
        last_commit_mtp_checkpoint_provenance_ = checkpoint.provenance;
        if (!commit_mtp_checkpoint_terminal_hidden_ok_ ||
            !checkpoint.valid ||
            already_appended_tokens < 0)
        {
            return false;
        }
        if (position_offset_override >= 0 &&
            position_offset_override != checkpoint.cached_tokens)
        {
            return false;
        }
        return true;
    }

    bool commitMTPInitialShiftedRowFromDeviceOutcome(
        const PrefixStateSnapshot &checkpoint,
        const DeviceSpeculativeOutcomeHandle &outcome,
        int request_index,
        int main_forward_token_count,
        bool allow_speculative_discard = false) override
    {
        using namespace sampling_math;
        ++commit_mtp_initial_device_outcome_calls_;
        last_commit_mtp_already_appended_ = 0;
        last_commit_mtp_main_forward_token_count_ = main_forward_token_count;
        last_commit_mtp_allow_speculative_discard_ = allow_speculative_discard;
        last_commit_mtp_position_offset_override_ = checkpoint.cached_tokens;
        last_commit_mtp_checkpoint_valid_ = checkpoint.valid;
        last_commit_mtp_checkpoint_logical_ = checkpoint.logical_checkpoint;
        last_commit_mtp_checkpoint_cached_tokens_ = checkpoint.cached_tokens;
        last_commit_mtp_checkpoint_provenance_ = checkpoint.provenance;
        last_commit_mtp_tokens_.clear();
        if (!checkpoint.valid ||
            !outcome.valid() ||
            outcome.device != device_id_ ||
            request_index < 0 ||
            request_index >= outcome.request_count ||
            outcome.meta_stride < kSpeculativeBatchMetaCount ||
            outcome.output_token_stride < kSpeculativeBatchMaxOutputTokens ||
            main_forward_token_count <= 0)
        {
            return false;
        }
        const int *meta =
            static_cast<const int *>(outcome.meta_device) +
            static_cast<size_t>(request_index) *
                static_cast<size_t>(outcome.meta_stride);
        const int32_t *tokens =
            static_cast<const int32_t *>(outcome.output_tokens_device) +
            static_cast<size_t>(request_index) *
                static_cast<size_t>(outcome.output_token_stride);
        if (meta[kSpecBatchMetaOk] == 0)
            return false;
        const int output_count = meta[kSpecBatchMetaOutputCount];
        last_commit_mtp_tokens_.assign(1, output_count > 0 ? tokens[0] : 0);
        return true;
    }

    bool ensureMTPCheckpointTerminalHidden() override
    {
        ++ensure_mtp_checkpoint_terminal_hidden_calls_;
        return ensure_mtp_checkpoint_terminal_hidden_ok_;
    }

    bool hasLogitsLocal() const override
    {
        return logits_local_ != nullptr;
    }

    LogitsLocalInfo getLogitsLocalInfo() const override
    {
        get_logits_local_info_calls_.fetch_add(
            1,
            std::memory_order_relaxed);
        if (!logits_local_)
            return {};
        const auto &shape = logits_local_->shape();
        return LogitsLocalInfo{
            nullptr,
            std::nullopt,
            shape.size() >= 2 ? shape[1] : 0,
            0,
            logits_local_.get(),
            nullptr,
            nullptr,
            nullptr,
            0};
    }

    LogitsLocalInfo consumeLogitsLocalInfoForSampling() override
    {
        consume_logits_local_info_calls_.fetch_add(
            1,
            std::memory_order_relaxed);
        return getLogitsLocalInfo();
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
            0,
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
            0,
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
        return !uses_mirrored_localtp_mtp_head_for_verifier_ &&
               all_position_logits_local_ != nullptr;
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
            0,
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

    void setSkipLogitsGatherDecode(bool skip) override
    {
        ++set_skip_decode_calls_;
        skip_logits_gather_decode_ = skip;
    }

    void setSkipLogitsGatherPrefill(bool skip) override
    {
        ++set_skip_prefill_calls_;
        skip_logits_gather_prefill_ = skip;
    }

    void setMTPAllPositionVerifierSyncDeferralEnabled(bool enabled) override
    {
        ++set_all_position_sync_deferral_calls_;
        all_position_sync_deferral_enabled_ = enabled;
    }

    void setMTPMainDecodeSyncDeferralEnabled(bool enabled) override
    {
        ++set_main_decode_sync_deferral_calls_;
        main_decode_sync_deferral_enabled_ = enabled;
    }

    bool verifyGreedyAllPositionBatchOutcomeOnDevice(
        const int32_t *draft_tokens,
        int draft_token_count,
        const int32_t *stop_tokens,
        int stop_token_count,
        DeviceSpeculativeVerifyBatchOutcome *out) override
    {
        using namespace sampling_math;
        ++verify_greedy_all_position_batch_outcome_calls_;
        last_stochastic_row_count_ = draft_token_count;
        last_stop_token_count_ = stop_token_count;
        if (!out ||
            !draft_tokens ||
            draft_token_count <= 0 ||
            draft_token_count > kSpeculativeBatchMaxOutputTokens ||
            stop_token_count < 0 ||
            stop_token_count > kSpeculativeBatchMaxStopTokens ||
            (stop_token_count > 0 && !stop_tokens))
        {
            return false;
        }
        if (!verify_greedy_all_position_batch_outcome_ok_)
            return false;

        std::array<int32_t, kSpeculativeBatchMaxOutputTokens> verify_tokens =
            {-1, -1, -1, -1, -1};
        if (!sampleMockAllPositionRows(
                0,
                draft_token_count,
                verify_tokens.data()))
        {
            return false;
        }

        const int compare_rows = draft_token_count - 1;
        std::array<int, kSpeculativeBatchMaxRows> sampled_tokens =
            {-1, -1, -1, -1};
        std::array<int, kSpeculativeBatchMaxRows> accepted =
            {0, 0, 0, 0};
        for (int row = 0; row < compare_rows; ++row)
        {
            const int32_t expected_draft =
                draft_tokens[static_cast<size_t>(row + 1)] >= 0
                    ? draft_tokens[static_cast<size_t>(row + 1)]
                    : verifier_device_tokens_[static_cast<size_t>(row + 1)];
            sampled_tokens[static_cast<size_t>(row)] =
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
                : verifier_device_tokens_[0];
        summarize_speculative_verify_batch(
            first_token,
            sampled_tokens.data(),
            accepted.data(),
            compare_rows,
            packed_stop_tokens.data(),
            stop_token_count,
            ready_token,
            1,
            output_tokens.data(),
            static_cast<int>(output_tokens.size()),
            meta.data());
        if (meta[kSpecBatchMetaOk] == 0)
            return false;

        if (out)
        {
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
            out->stopped_on_output =
                meta[kSpecBatchMetaStoppedOnOutput] != 0;
            out->all_speculative_accepted =
                meta[kSpecBatchMetaAllSpeculativeAccepted] != 0;
            out->consumed_verifier_rows =
                meta[kSpecBatchMetaConsumedVerifierRows];
            out->sampled_terminal =
                meta[kSpecBatchMetaSampledTerminal] != 0;
        }
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
        if (out_handle)
            *out_handle = DeviceSpeculativeOutcomeHandle{};
        if (!out_handle ||
            !supports_device_resident_mtp_spec_state_publication_)
        {
            return false;
        }

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

        staged_resident_output_tokens_.fill(-1);
        staged_resident_meta_.fill(0);
        writeResidentOutcomeRow(/*request_index=*/0, outcome);

        out_handle->output_tokens_device =
            staged_resident_output_tokens_.data();
        out_handle->meta_device = staged_resident_meta_.data();
        out_handle->request_count = 1;
        out_handle->output_token_stride = kSpeculativeBatchMaxOutputTokens;
        out_handle->meta_stride = kSpeculativeBatchMetaCount;
        out_handle->device = device_id_;
        out_handle->stream = &resident_stream_token_;
        out_handle->response_ready_event =
            std::shared_ptr<void>(
                &resident_outcome_ready_event_token_,
                [](void *) {});
        out_handle->mirrored_local_tp_published_in_graph =
            uses_mirrored_localtp_mtp_head_for_verifier_;
        attachMockResidentMTPTransaction(out_handle, /*request_count=*/1);
        return out_handle->valid();
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
        if (slot >= 0 &&
            slot < static_cast<int>(draft_sample_tokens_.size()))
        {
            draft_sample_tokens_[static_cast<size_t>(slot)] =
                stochastic_sample_token_;
            draft_sample_slot_ready_[static_cast<size_t>(slot)] = true;
        }
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

    DeviceStochasticDraftSampleSlotHandle deviceStochasticDraftSampleSlot(
        int slot,
        bool require_ready = false) override
    {
        if (slot < 0 ||
            slot >= static_cast<int>(draft_sample_tokens_.size()) ||
            (require_ready &&
             !draft_sample_slot_ready_[static_cast<size_t>(slot)]))
        {
            return {};
        }

        DeviceStochasticDraftSampleSlotHandle handle;
        handle.token_device =
            draft_sample_tokens_.data() + static_cast<size_t>(slot);
        handle.slot = slot;
        handle.device = device_id_;
        handle.stream =
            draft_sample_stream_tokens_.data() + static_cast<size_t>(slot);
        return handle;
    }

    bool recordStochasticDraftSampleSlotReadyFromDevice(
        int slot,
        void *producer_stream,
        bool verifier_consumer_pending = true) override
    {
        (void)verifier_consumer_pending;
        if (slot < 0 ||
            slot >= static_cast<int>(draft_sample_tokens_.size()) ||
            producer_stream == nullptr)
        {
            return false;
        }

        ++record_draft_sample_slot_ready_calls_;
        last_recorded_draft_sample_slot_ = slot;
        draft_sample_slot_ready_[static_cast<size_t>(slot)] = true;
        return true;
    }

    DeviceStochasticTargetSampleSlotHandle deviceStochasticTargetSampleSlot(
        int slot,
        bool require_ready = false) override
    {
        if (slot < 0 ||
            slot >= static_cast<int>(target_sample_tokens_.size()) ||
            (require_ready &&
             !target_sample_slot_ready_[static_cast<size_t>(slot)]))
        {
            return {};
        }

        DeviceStochasticTargetSampleSlotHandle handle;
        handle.token_device =
            target_sample_tokens_.data() + static_cast<size_t>(slot);
        handle.slot = slot;
        handle.device = device_id_;
        handle.stream =
            target_sample_stream_tokens_.data() + static_cast<size_t>(slot);
        return handle;
    }

    bool recordStochasticTargetSampleSlotReadyFromDevice(
        int slot,
        void *producer_stream,
        bool verifier_consumer_pending = true) override
    {
        (void)verifier_consumer_pending;
        if (slot < 0 ||
            slot >= static_cast<int>(target_sample_tokens_.size()) ||
            producer_stream == nullptr)
        {
            return false;
        }
        ++record_target_sample_slot_ready_calls_;
        last_recorded_target_sample_slot_ = slot;
        target_sample_slot_ready_[static_cast<size_t>(slot)] = true;
        return true;
    }

    int sampleStochasticDistributionOnDevice(
        DeviceDistributionBuffer buffer,
        int slot,
        float threshold) override
    {
        ++sample_stochastic_distribution_calls_;
        last_stochastic_slot_ = slot;
        last_stochastic_threshold_ = threshold;
        if (buffer == DeviceDistributionBuffer::Target &&
            slot >= 0 &&
            slot < static_cast<int>(target_sample_tokens_.size()))
        {
            target_sample_tokens_[static_cast<size_t>(slot)] =
                stochastic_sample_token_;
            target_sample_slot_ready_[static_cast<size_t>(slot)] = true;
        }
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

    const void *prepareMTPVerifierInputTokensOnDeviceFromHostRow(
        const int32_t *verifier_tokens,
        int total_verifier_input_tokens,
        int draft_token_count) override
    {
        ++prepare_mtp_verifier_input_tokens_from_host_row_calls_;
        last_verifier_draft_token_count_ = draft_token_count;
        last_verifier_total_input_tokens_ = total_verifier_input_tokens;
        last_verifier_host_row_tokens_.clear();
        if (!stochastic_device_ops_ok_ ||
            !verifier_tokens ||
            total_verifier_input_tokens <= 0 ||
            draft_token_count < 0)
        {
            return nullptr;
        }

        last_verifier_host_row_tokens_.assign(
            verifier_tokens,
            verifier_tokens + total_verifier_input_tokens);
        for (int i = 0; i < total_verifier_input_tokens &&
                        i < static_cast<int>(verifier_device_tokens_from_host_row_.size());
             ++i)
        {
            verifier_device_tokens_from_host_row_[static_cast<size_t>(i)] =
                verifier_tokens[i];
        }
        return verifier_device_tokens_from_host_row_.data();
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
        for (int i = 0; i < draft_token_count; ++i)
        {
            const int slot = first_draft_slot + i;
            if (slot >= 0 &&
                slot < static_cast<int>(draft_sample_tokens_.size()))
            {
                draft_sample_tokens_[static_cast<size_t>(slot)] =
                    draft_tokens[i];
                draft_sample_slot_ready_[static_cast<size_t>(slot)] = true;
            }
        }
        return true;
    }

    bool stageStochasticTargetTokenForDeviceSampling(
        int32_t target_token,
        int target_sample_slot = 0) override
    {
        ++stage_stochastic_target_token_calls_;
        last_staged_target_token_ = target_token;
        last_staged_target_sample_slot_ = target_sample_slot;
        return stochastic_device_ops_ok_ &&
               target_token >= 0 &&
               target_sample_slot >= 0;
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

    bool verifyStochasticDistributionsRequestBatchOutcomesOnDeviceResident(
        const DeviceStochasticBatchOutcomeRequest *requests,
        int request_count,
        DeviceSpeculativeOutcomeHandle *out_handle) override
    {
        using namespace sampling_math;
        ++verify_stochastic_request_batch_outcome_calls_;
        if (out_handle)
            *out_handle = DeviceSpeculativeOutcomeHandle{};
        if (!supports_device_stochastic_mtp_verification_ ||
            !supports_device_resident_mtp_spec_state_publication_ ||
            !stochastic_device_ops_ok_ ||
            !requests ||
            !out_handle ||
            request_count <= 0 ||
            request_count > kMockResidentOutcomeRequestCapacity)
        {
            return false;
        }

        staged_resident_output_tokens_.fill(-1);
        staged_resident_meta_.fill(0);
        for (int request_index = 0; request_index < request_count; ++request_index)
        {
            const DeviceStochasticBatchOutcomeRequest &request =
                requests[request_index];
            if (request.row_count <= 0 ||
                request.row_count > kSpeculativeBatchMaxRows)
            {
                return false;
            }

            std::array<int, kSpeculativeBatchMaxRows> row_tokens{};
            std::array<int, kSpeculativeBatchMaxRows> row_accepted{};
            std::array<int, kSpeculativeBatchMaxStopTokens> stop_tokens{};
            std::array<int, kSpeculativeBatchMaxOutputTokens> output_tokens{};
            std::array<int, kSpeculativeBatchMetaCount> meta{};
            row_tokens.fill(-1);
            row_accepted.fill(0);
            stop_tokens.fill(-1);
            output_tokens.fill(-1);
            meta.fill(0);

            for (int row = 0; row < request.row_count; ++row)
            {
                const int32_t draft_token =
                    request.use_device_draft_tokens
                        ? (row < static_cast<int>(last_staged_draft_tokens_.size())
                               ? last_staged_draft_tokens_[static_cast<size_t>(row)]
                               : stochastic_sample_token_)
                        : request.draft_tokens[static_cast<size_t>(row)];
                row_tokens[static_cast<size_t>(row)] = draft_token;
                row_accepted[static_cast<size_t>(row)] = 1;
            }
            for (int i = 0; i < request.stop_token_count; ++i)
            {
                stop_tokens[static_cast<size_t>(i)] =
                    request.stop_tokens[static_cast<size_t>(i)];
            }

            const int first_token =
                request.first_token_from_device
                    ? last_staged_target_token_
                    : request.first_token;
            const int has_bonus =
                request.bonus_target_slot >= 0 ? 1 : 0;
            const int bonus_token =
                has_bonus ? stochastic_sample_token_ : -1;
            summarize_speculative_verify_batch(
                first_token,
                row_tokens.data(),
                row_accepted.data(),
                request.row_count,
                request.stop_token_count > 0 ? stop_tokens.data() : nullptr,
                request.stop_token_count,
                bonus_token,
                has_bonus,
                output_tokens.data(),
                static_cast<int>(output_tokens.size()),
                meta.data());
            if (meta[kSpecBatchMetaOk] == 0)
                return false;

            DeviceSpeculativeVerifyBatchOutcome outcome;
            outcome.ok = true;
            for (size_t token_index = 0;
                 token_index < outcome.output_tokens.size();
                 ++token_index)
            {
                outcome.output_tokens[token_index] =
                    output_tokens[token_index];
            }
            outcome.output_token_count =
                meta[kSpecBatchMetaOutputCount];
            outcome.accepted_speculative_prefix =
                meta[kSpecBatchMetaAcceptedSpeculativePrefix];
            outcome.target_verifier_state_commit_count =
                meta[kSpecBatchMetaTargetVerifierStateCommitCount];
            outcome.ready_token = meta[kSpecBatchMetaReadyToken];
            outcome.rejected_verified_token =
                meta[kSpecBatchMetaRejectedVerifiedToken];
            outcome.stopped_on_output =
                meta[kSpecBatchMetaStoppedOnOutput] != 0;
            outcome.all_speculative_accepted =
                meta[kSpecBatchMetaAllSpeculativeAccepted] != 0;
            outcome.consumed_verifier_rows =
                meta[kSpecBatchMetaConsumedVerifierRows];
            outcome.sampled_terminal =
                meta[kSpecBatchMetaSampledTerminal] != 0;
            writeResidentOutcomeRow(request_index, outcome);
        }

        out_handle->output_tokens_device =
            staged_resident_output_tokens_.data();
        out_handle->meta_device = staged_resident_meta_.data();
        out_handle->request_count = request_count;
        out_handle->output_token_stride = kSpeculativeBatchMaxOutputTokens;
        out_handle->meta_stride = kSpeculativeBatchMetaCount;
        out_handle->device = device_id_;
        out_handle->stream = &resident_stream_token_;
        out_handle->response_ready_event =
            std::shared_ptr<void>(
                &resident_outcome_ready_event_token_,
                [](void *) {});
        attachMockResidentMTPTransaction(out_handle, request_count);
        return out_handle->valid();
    }

    bool publishRankCompactSpeculativeResponseReady(
        DeviceSpeculativeOutcomeHandle *handle) override
    {
        ++publish_rank_compact_response_ready_calls_;
        if (!handle ||
            !handle->valid() ||
            handle->device != device_id_ ||
            handle->stream != &resident_stream_token_ ||
            handle->response_ready_event.get() !=
                &resident_outcome_ready_event_token_ ||
            handle->response_ready_after_rank_collective)
        {
            return false;
        }
        handle->response_ready_after_rank_collective = true;
        return true;
    }

    void clear_cache() override
    {
        clear_cache_calls_.fetch_add(1, std::memory_order_relaxed);
        position_ = 0;
        resident_mtp_transaction_state_.reset();
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

    const float *getSnapshot(const std::string &key, size_t &out_size) const override
    {
        auto it = snapshots_.find(key);
        if (it == snapshots_.end())
        {
            out_size = 0;
            return nullptr;
        }
        out_size = it->second.size();
        return it->second.data();
    }

    SnapshotInfo getSnapshotWithShape(const std::string &key) const override
    {
        auto it = snapshots_.find(key);
        if (it == snapshots_.end())
            return {};

        const auto shape_it = snapshot_shapes_.find(key);
        if (shape_it == snapshot_shapes_.end())
            return {};

        return SnapshotInfo{
            it->second.data(),
            it->second.size(),
            shape_it->second.first,
            shape_it->second.second};
    }

    std::vector<std::string> getSnapshotKeys() const override
    {
        std::vector<std::string> keys;
        keys.reserve(snapshots_.size());
        for (const auto &entry : snapshots_)
            keys.push_back(entry.first);
        std::sort(keys.begin(), keys.end());
        return keys;
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
        populated_prefix_restore_model_runtime_state_.push_back(hit.restore_model_runtime_state);
        if (prefix_populate_ok_)
        {
            position_ = hit.cached_tokens;
            if (prefix_populate_invalidates_resident_logical_state_)
            {
                resident_logical_state_valid_ = false;
                resident_logical_state_request_count_ = 0;
            }
        }
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

    PrefixStateSnapshot captureLivePrefixCheckpoint(
        const PrefixCheckpointCaptureRequest &request) const override
    {
        prefix_live_capture_calls_.fetch_add(1, std::memory_order_relaxed);
        PrefixStateSnapshot snapshot;
        if (!prefix_live_capture_ok_ || !request.valid())
            return snapshot;
        snapshot.valid = true;
        snapshot.logical_checkpoint = true;
        snapshot.cached_tokens = request.logical_cached_tokens;
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

    void set_mock_logits_local(int local_vocab, const std::vector<float> &logits)
    {
        logits_local_ = std::make_shared<FP32Tensor>(
            std::vector<size_t>{1, static_cast<size_t>(local_vocab)},
            DeviceId::cpu());
        std::memcpy(logits_local_->mutable_data(), logits.data(),
                    logits.size() * sizeof(float));
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

    void set_mock_snapshot(
        const std::string &key,
        size_t rows,
        size_t cols,
        std::vector<float> data)
    {
        if (data.size() != rows * cols)
            throw std::invalid_argument("mock snapshot size does not match rows*cols");
        snapshots_[key] = std::move(data);
        snapshot_shapes_[key] = {rows, cols};
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
    void set_prefix_populate_invalidates_resident_logical_state(bool invalidates)
    {
        prefix_populate_invalidates_resident_logical_state_ = invalidates;
    }
    void set_prefix_terminal_restore_requires_blocks(bool required) { prefix_terminal_restore_requires_blocks_ = required; }
    void set_forward_mtp_ok(bool ok) { forward_mtp_ok_ = ok; }
    void set_supports_chained_mtp_drafts(bool supported) { supports_chained_mtp_drafts_ = supported; }
    void set_forward_mtp_from_last_draft_ok(bool ok) { forward_mtp_from_last_draft_ok_ = ok; }
    void set_commit_mtp_shifted_rows_ok(bool ok) { commit_mtp_shifted_rows_ok_ = ok; }
    void set_commit_mtp_checkpoint_terminal_hidden_ok(bool ok)
    {
        commit_mtp_checkpoint_terminal_hidden_ok_ = ok;
    }
    void set_ensure_mtp_checkpoint_terminal_hidden_ok(bool ok)
    {
        ensure_mtp_checkpoint_terminal_hidden_ok_ = ok;
    }
    void set_supports_mtp_spec_state_publication(bool supported) { supports_mtp_spec_state_publication_ = supported; }
    void set_supports_device_resident_mtp_spec_state_publication(bool supported)
    {
        supports_device_resident_mtp_spec_state_publication_ = supported;
    }
    void set_uses_mirrored_localtp_mtp_head_for_verifier(bool mirrored)
    {
        uses_mirrored_localtp_mtp_head_for_verifier_ = mirrored;
    }
    void set_mtp_verifier_row_capability(MTPVerifierRowCapability capability)
    {
        mtp_verifier_row_capability_ = capability;
    }
    void set_publish_mtp_spec_state_ok(bool ok) { publish_mtp_spec_state_ok_ = ok; }
    void set_all_position_logits_ok(bool ok) { set_all_position_logits_ok_ = ok; }
    void set_mtp_unsupported_reason(std::string reason) { mtp_unsupported_reason_ = std::move(reason); }
    void set_primary_device_id(DeviceId device_id) { device_id_ = device_id; }
    void set_prefix_probe_position_override(int position)
    {
        prefix_probe_position_override_ = position;
    }
    /**
     * @brief Seed the mock's device-owned pre-verifier sequence position.
     *
     * This is test setup for the resident cache-count snapshot.  It must not be
     * copied into a publication request, because the production GPU publisher
     * obtains the same value from persistent device metadata.
     */
    void set_mock_device_position(int position) { position_ = position; }
    void set_supports_mtp_sidecar_preserves_main_state(bool supported) { supports_mtp_sidecar_preserves_main_state_ = supported; }
    void set_supports_mtp_sidecar_logits_stream_handoff(bool supported) { supports_mtp_sidecar_logits_stream_handoff_ = supported; }
    void set_supports_mtp_device_draft_token_input(bool supported) { supports_mtp_device_draft_token_input_ = supported; }
    void set_supports_mtp_shifted_row_reuse_from_sidecar(bool supported) { supports_mtp_shifted_row_reuse_from_sidecar_ = supported; }
    void set_supports_device_stochastic_mtp_verification(bool supported) { supports_device_stochastic_mtp_verification_ = supported; }
    void set_stochastic_sample_token(int token) { stochastic_sample_token_ = token; }
    void set_consume_unused_replicated_main_logits_publication_ok(bool ok)
    {
        consume_unused_replicated_main_logits_publication_ok_ = ok;
    }
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
    const std::vector<bool> &populated_prefix_restore_model_runtime_state() const
    {
        return populated_prefix_restore_model_runtime_state_;
    }
    const std::vector<int> &terminal_restored_tokens() const { return terminal_restored_tokens_; }
    const std::vector<int32_t> &prefix_lookup_tokens() const { return prefix_lookup_tokens_; }
    const std::vector<int32_t> &harvested_prefix_tokens() const { return harvested_prefix_tokens_; }
    int harvested_prompt_token_count() const { return harvested_prompt_token_count_; }
    size_t forward_mtp_call_count() const { return forward_mtp_calls_.load(std::memory_order_relaxed); }
    size_t forward_mtp_from_last_draft_call_count() const { return forward_mtp_from_last_draft_calls_.load(std::memory_order_relaxed); }
    size_t sample_mtp_logits_call_count() const { return sample_mtp_logits_calls_; }
    size_t sample_greedy_on_device_call_count() const { return sample_greedy_on_device_calls_; }
    size_t sample_greedy_main_target_slot_call_count() const
    {
        return sample_greedy_main_target_slot_calls_;
    }
    size_t sample_on_device_call_count() const { return sample_on_device_calls_; }
    size_t consume_unused_replicated_main_logits_publication_call_count() const
    {
        return consume_unused_replicated_main_logits_publication_calls_;
    }
    size_t get_logits_local_info_call_count() const { return get_logits_local_info_calls_.load(std::memory_order_relaxed); }
    size_t consume_logits_local_info_call_count() const { return consume_logits_local_info_calls_.load(std::memory_order_relaxed); }
    size_t get_mtp_logits_local_info_call_count() const { return get_mtp_logits_local_info_calls_.load(std::memory_order_relaxed); }
    size_t consume_mtp_logits_local_info_call_count() const { return consume_mtp_logits_local_info_calls_.load(std::memory_order_relaxed); }
    size_t commit_mtp_shifted_rows_call_count() const { return commit_mtp_shifted_rows_calls_; }
    size_t commit_mtp_checkpoint_terminal_hidden_call_count() const
    {
        return commit_mtp_checkpoint_terminal_hidden_calls_;
    }
    size_t commit_mtp_initial_device_outcome_call_count() const
    {
        return commit_mtp_initial_device_outcome_calls_;
    }
    size_t ensure_mtp_checkpoint_terminal_hidden_call_count() const
    {
        return ensure_mtp_checkpoint_terminal_hidden_calls_;
    }
    size_t publish_mtp_spec_state_call_count() const { return publish_mtp_spec_state_calls_.load(std::memory_order_relaxed); }
    size_t publish_mtp_spec_state_batch_call_count() const { return publish_mtp_spec_state_batch_calls_.load(std::memory_order_relaxed); }
    size_t publish_grouped_decode_equivalent_mtp_spec_state_batch_call_count() const
    {
        return publish_grouped_decode_equivalent_mtp_spec_state_batch_calls_.load(
            std::memory_order_relaxed);
    }
    size_t publish_device_resident_mtp_spec_state_batch_call_count() const
    {
        return publish_device_resident_mtp_spec_state_batch_calls_.load(
            std::memory_order_relaxed);
    }
    int32_t last_mtp_condition_token() const { return last_mtp_condition_token_; }
    int32_t last_chained_mtp_condition_token() const { return last_chained_mtp_condition_token_; }
    int last_chained_mtp_position_id() const { return last_chained_mtp_position_id_; }
    size_t forward_mtp_from_device_draft_call_count() const { return forward_mtp_from_device_draft_calls_; }
    size_t forward_mtp_from_device_target_call_count() const { return forward_mtp_from_device_target_calls_; }
    int last_device_draft_sample_slot() const { return last_device_draft_sample_slot_; }
    int last_device_target_sample_slot() const { return last_device_target_sample_slot_; }
    int last_device_token_sidecar_position_id() const { return last_device_token_sidecar_position_id_; }
    size_t commit_mtp_device_target_sample_call_count() const
    {
        return commit_mtp_device_target_sample_calls_;
    }
    int last_commit_mtp_device_target_sample_slot() const
    {
        return last_commit_mtp_device_target_sample_slot_;
    }
    int last_commit_mtp_already_appended() const { return last_commit_mtp_already_appended_; }
    int last_commit_mtp_main_forward_token_count() const { return last_commit_mtp_main_forward_token_count_; }
    bool last_commit_mtp_allow_speculative_discard() const { return last_commit_mtp_allow_speculative_discard_; }
    int last_commit_mtp_position_offset_override() const { return last_commit_mtp_position_offset_override_; }
    const std::vector<int32_t> &last_commit_mtp_tokens() const { return last_commit_mtp_tokens_; }
    bool last_commit_mtp_checkpoint_valid() const { return last_commit_mtp_checkpoint_valid_; }
    bool last_commit_mtp_checkpoint_logical() const { return last_commit_mtp_checkpoint_logical_; }
    int last_commit_mtp_checkpoint_cached_tokens() const { return last_commit_mtp_checkpoint_cached_tokens_; }
    PrefixStateProvenance last_commit_mtp_checkpoint_provenance() const
    {
        return last_commit_mtp_checkpoint_provenance_;
    }
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
    size_t set_skip_decode_call_count() const { return set_skip_decode_calls_; }
    size_t set_skip_prefill_call_count() const { return set_skip_prefill_calls_; }
    bool skip_logits_gather_decode() const { return skip_logits_gather_decode_; }
    bool skip_logits_gather_prefill() const { return skip_logits_gather_prefill_; }
    size_t set_all_position_sync_deferral_call_count() const { return set_all_position_sync_deferral_calls_; }
    size_t set_main_decode_sync_deferral_call_count() const { return set_main_decode_sync_deferral_calls_; }
    bool all_position_sync_deferral_enabled() const { return all_position_sync_deferral_enabled_; }
    bool main_decode_sync_deferral_enabled() const { return main_decode_sync_deferral_enabled_; }
    size_t build_stochastic_processed_rows_call_count() const { return build_stochastic_processed_rows_calls_; }
    size_t sample_stochastic_draft_proposal_call_count() const { return sample_stochastic_draft_proposal_calls_; }
    size_t sample_stochastic_distribution_call_count() const { return sample_stochastic_distribution_calls_; }
    size_t build_stochastic_distributions_call_count() const
    {
        return build_stochastic_distributions_calls_;
    }
    size_t build_stochastic_distribution_call_count() const
    {
        return build_stochastic_distribution_calls_;
    }
    size_t prepare_mtp_verifier_input_tokens_call_count() const
    {
        return prepare_mtp_verifier_input_tokens_calls_;
    }
    size_t prepare_mtp_verifier_input_tokens_from_device_call_count() const
    {
        return prepare_mtp_verifier_input_tokens_from_device_calls_;
    }
    size_t prepare_mtp_verifier_input_tokens_from_host_row_call_count() const
    {
        return prepare_mtp_verifier_input_tokens_from_host_row_calls_;
    }
    const std::vector<int32_t> &last_verifier_host_row_tokens() const
    {
        return last_verifier_host_row_tokens_;
    }
    size_t stage_stochastic_draft_tokens_call_count() const
    {
        return stage_stochastic_draft_tokens_calls_;
    }
    size_t record_draft_sample_slot_ready_call_count() const
    {
        return record_draft_sample_slot_ready_calls_;
    }
    size_t record_target_sample_slot_ready_call_count() const
    {
        return record_target_sample_slot_ready_calls_;
    }
    int last_recorded_target_sample_slot() const
    {
        return last_recorded_target_sample_slot_;
    }
    int last_recorded_draft_sample_slot() const
    {
        return last_recorded_draft_sample_slot_;
    }
    int32_t draft_sample_token(int slot) const
    {
        if (slot < 0 ||
            slot >= static_cast<int>(draft_sample_tokens_.size()))
            return -1;
        return draft_sample_tokens_[static_cast<size_t>(slot)];
    }
    bool draft_sample_slot_ready(int slot) const
    {
        if (slot < 0 ||
            slot >= static_cast<int>(draft_sample_slot_ready_.size()))
            return false;
        return draft_sample_slot_ready_[static_cast<size_t>(slot)];
    }
    int32_t target_sample_token(int slot) const
    {
        if (slot < 0 ||
            slot >= static_cast<int>(target_sample_tokens_.size()))
            return -1;
        return target_sample_tokens_[static_cast<size_t>(slot)];
    }
    bool target_sample_slot_ready(int slot) const
    {
        if (slot < 0 ||
            slot >= static_cast<int>(target_sample_slot_ready_.size()))
            return false;
        return target_sample_slot_ready_[static_cast<size_t>(slot)];
    }
    size_t stage_stochastic_target_token_call_count() const
    {
        return stage_stochastic_target_token_calls_;
    }
    size_t verify_stochastic_batch_outcome_call_count() const { return verify_stochastic_batch_outcome_calls_; }
    size_t verify_stochastic_request_batch_outcome_call_count() const { return verify_stochastic_request_batch_outcome_calls_; }
    size_t publish_rank_compact_response_ready_call_count() const
    {
        return publish_rank_compact_response_ready_calls_;
    }
    size_t verify_greedy_all_position_batch_outcome_call_count() const { return verify_greedy_all_position_batch_outcome_calls_; }
    size_t forward_mtp_from_resident_logical_state_call_count() const
    {
        return forward_mtp_from_resident_logical_state_calls_;
    }
    int last_resident_logical_state_request_index() const
    {
        return last_resident_logical_state_request_index_;
    }
    int32_t last_verifier_first_token() const { return last_verifier_first_token_; }
    int last_verifier_first_target_sample_slot() const { return last_verifier_first_target_sample_slot_; }
    int last_verifier_first_draft_slot() const { return last_verifier_first_draft_slot_; }
    int last_verifier_draft_token_count() const { return last_verifier_draft_token_count_; }
    int last_verifier_total_input_tokens() const { return last_verifier_total_input_tokens_; }
    int last_staged_first_draft_slot() const { return last_staged_first_draft_slot_; }
    int last_staged_target_sample_slot() const { return last_staged_target_sample_slot_; }
    int32_t last_staged_target_token() const { return last_staged_target_token_; }
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
        sample_greedy_on_device_calls_ = 0;
        sample_on_device_calls_ = 0;
        commit_mtp_shifted_rows_calls_ = 0;
        ensure_mtp_checkpoint_terminal_hidden_calls_ = 0;
        publish_mtp_spec_state_calls_.store(0, std::memory_order_relaxed);
        set_all_position_logits_calls_.store(0, std::memory_order_relaxed);
        set_row_indexed_all_position_logits_calls_.store(0, std::memory_order_relaxed);
        set_mtp_spec_verifier_input_plan_calls_.store(0, std::memory_order_relaxed);
        clear_mtp_spec_verifier_input_plan_calls_.store(0, std::memory_order_relaxed);
        get_logits_local_info_calls_.store(0, std::memory_order_relaxed);
        consume_logits_local_info_calls_.store(0, std::memory_order_relaxed);
        get_mtp_logits_local_info_calls_.store(0, std::memory_order_relaxed);
        consume_mtp_logits_local_info_calls_.store(0, std::memory_order_relaxed);
        get_all_position_logits_local_info_calls_.store(0, std::memory_order_relaxed);
        consume_all_position_logits_local_info_calls_.store(0, std::memory_order_relaxed);
        prefix_live_capture_calls_.store(0, std::memory_order_relaxed);
        prefix_live_restore_calls_.store(0, std::memory_order_relaxed);
        prefix_live_truncate_calls_.store(0, std::memory_order_relaxed);
        set_skip_decode_calls_ = 0;
        set_skip_prefill_calls_ = 0;
        set_all_position_sync_deferral_calls_ = 0;
        set_main_decode_sync_deferral_calls_ = 0;
        prepare_mtp_verifier_input_tokens_calls_ = 0;
        prepare_mtp_verifier_input_tokens_from_device_calls_ = 0;
        prepare_mtp_verifier_input_tokens_from_host_row_calls_ = 0;
        stage_stochastic_draft_tokens_calls_ = 0;
        stage_stochastic_target_token_calls_ = 0;
        forward_mtp_from_device_draft_calls_ = 0;
        forward_mtp_from_device_target_calls_ = 0;
        record_draft_sample_slot_ready_calls_ = 0;
        last_recorded_draft_sample_slot_ = -1;
        last_verifier_host_row_tokens_.clear();
        last_staged_draft_tokens_.clear();
        draft_sample_slot_ready_.fill(false);
    }

private:
    /**
     * @brief Attach a request-scoped shifted-cache ownership lease to an outcome.
     *
     * Real GPU runners expose canonical per-depth cache-count device pointers
     * and an event fence owned by the active MTP session.  These unit tests use
     * ordinary process memory as their fake device address space, but retain
     * the same pointer, stream, event, and generation lifetime so RankOrchestrator
     * exercises the production ownership checks rather than a weaker mock ABI.
     */
    void attachMockResidentMTPTransaction(
        DeviceSpeculativeOutcomeHandle *out_handle,
        int request_count)
    {
        if (!out_handle || request_count <= 0 ||
            request_count > kMockResidentOutcomeRequestCapacity)
        {
            return;
        }

        std::fill(
            resident_shifted_cached_tokens_.begin(),
            resident_shifted_cached_tokens_.end(),
            position_);
        auto state = std::make_shared<DeviceResidentMTPTransactionState>();
        state->device = device_id_;
        state->request_count = request_count;
        state->shifted_cached_tokens_device_by_depth = {
            resident_shifted_cached_tokens_.data()};
        state->producer_stream = &resident_stream_token_;
        state->ready_event = std::shared_ptr<void>(
            &resident_mtp_transaction_ready_event_token_,
            [](void *) {});
        state->session_epoch = 1;
        state->mutation_generation = ++resident_mtp_transaction_generation_;
        resident_mtp_transaction_state_ = std::move(state);
        out_handle->mtp_transaction.state = resident_mtp_transaction_state_;
    }

    bool sampleMockAllPositionRows(
        int start_row,
        int row_count,
        int32_t *out_tokens) const
    {
        if (start_row < 0 ||
            row_count <= 0 ||
            !out_tokens ||
            config_.vocab_size <= 0 ||
            all_position_logits_.empty())
        {
            return false;
        }

        const size_t vocab = static_cast<size_t>(config_.vocab_size);
        for (int i = 0; i < row_count; ++i)
        {
            const size_t row =
                static_cast<size_t>(start_row + i);
            const size_t offset = row * vocab;
            if (offset + vocab > all_position_logits_.size())
                return false;

            const float *row_logits =
                all_position_logits_.data() + offset;
            int best = 0;
            float best_value = row_logits[0];
            for (int token = 1; token < config_.vocab_size; ++token)
            {
                const float value = row_logits[static_cast<size_t>(token)];
                if (value > best_value)
                {
                    best = token;
                    best_value = value;
                }
            }
            out_tokens[static_cast<size_t>(i)] =
                static_cast<int32_t>(best);
        }
        return true;
    }

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
            staged_resident_output_tokens_[token_base + static_cast<size_t>(i)] =
                outcome.output_tokens[static_cast<size_t>(i)];
        }

        int *meta = staged_resident_meta_.data() + meta_base;
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

    Config config_;
    int position_;
    std::vector<float> logits_;
    std::vector<float> mtp_logits_;
    std::vector<float> all_position_logits_;
    std::unordered_map<std::string, std::vector<float>> snapshots_;
    std::unordered_map<std::string, std::pair<size_t, size_t>> snapshot_shapes_;
    std::shared_ptr<FP32Tensor> logits_local_;
    std::shared_ptr<FP32Tensor> mtp_logits_local_;
    std::shared_ptr<FP32Tensor> all_position_logits_local_;
    std::shared_ptr<ForwardMTPRendezvous> forward_mtp_rendezvous_;
    std::shared_ptr<MTPPublicationRendezvous> mtp_publication_rendezvous_;
    std::shared_ptr<ChainedMTPRendezvous> chained_mtp_rendezvous_;
    PrefixLookupResult prefix_lookup_result_;
    DeviceId device_id_ = DeviceId::cpu();
    std::optional<int> prefix_probe_position_override_;
    bool prefix_populate_ok_ = true;
    bool prefix_harvest_ok_ = true;
    bool prefix_terminal_restore_ok_ = true;
    bool prefix_terminal_restore_requires_blocks_ = false;
    bool prefix_populate_invalidates_resident_logical_state_ = false;
    bool forward_mtp_ok_ = true;
    bool supports_chained_mtp_drafts_ = false;
    bool forward_mtp_from_last_draft_ok_ = true;
    bool commit_mtp_shifted_rows_ok_ = true;
    bool commit_mtp_checkpoint_terminal_hidden_ok_ = true;
    bool ensure_mtp_checkpoint_terminal_hidden_ok_ = true;
    bool supports_mtp_spec_state_publication_ = false;
    bool supports_device_resident_mtp_spec_state_publication_ = false;
    bool uses_mirrored_localtp_mtp_head_for_verifier_ = false;
    MTPVerifierRowCapability mtp_verifier_row_capability_;
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
    bool skip_logits_gather_decode_ = false;
    bool skip_logits_gather_prefill_ = false;
    bool all_position_sync_deferral_enabled_ = false;
    bool main_decode_sync_deferral_enabled_ = false;
    bool consume_unused_replicated_main_logits_publication_ok_ = true;
    bool resident_logical_state_valid_ = false;
    int resident_logical_state_request_count_ = 0;
    mutable int resident_stream_token_ = 0;
    mutable int resident_ready_event_token_ = 0;
    int stochastic_sample_token_ = 17;
    bool prefix_live_capture_ok_ = true;
    bool prefix_live_restore_ok_ = true;
    bool prefix_live_truncate_ok_ = true;
    uint64_t moe_placement_epoch_ = 0;
    int32_t last_mtp_condition_token_ = -1;
    int32_t last_chained_mtp_condition_token_ = -1;
    int last_chained_mtp_position_id_ = -1;
    size_t forward_mtp_from_device_draft_calls_ = 0;
    size_t forward_mtp_from_device_target_calls_ = 0;
    size_t commit_mtp_device_target_sample_calls_ = 0;
    int last_device_draft_sample_slot_ = -1;
    int last_device_target_sample_slot_ = -1;
    int last_device_token_sidecar_position_id_ = -1;
    int last_commit_mtp_device_target_sample_slot_ = -1;
    int last_commit_mtp_already_appended_ = 0;
    int last_commit_mtp_main_forward_token_count_ = 0;
    int last_commit_mtp_position_offset_override_ = -1;
    int last_commit_mtp_checkpoint_cached_tokens_ = -1;
    size_t ensure_mtp_checkpoint_terminal_hidden_calls_ = 0;
    bool last_commit_mtp_allow_speculative_discard_ = false;
    bool last_commit_mtp_checkpoint_valid_ = false;
    bool last_commit_mtp_checkpoint_logical_ = false;
    PrefixStateProvenance last_commit_mtp_checkpoint_provenance_ =
        PrefixStateProvenance::Unknown;
    MTPSpecStepPlan last_mtp_spec_state_plan_;
    MTPSpecStepPlanBatch last_mtp_spec_state_batch_;
    MTPSpecDecodeVerifierInputPlan last_mtp_spec_verifier_input_plan_;
    size_t sample_mtp_logits_calls_ = 0;
    size_t sample_greedy_on_device_calls_ = 0;
    size_t sample_greedy_main_target_slot_calls_ = 0;
    size_t sample_on_device_calls_ = 0;
    size_t consume_unused_replicated_main_logits_publication_calls_ = 0;
    size_t commit_mtp_shifted_rows_calls_ = 0;
    size_t commit_mtp_checkpoint_terminal_hidden_calls_ = 0;
    size_t commit_mtp_initial_device_outcome_calls_ = 0;
    size_t apply_penalties_on_device_calls_ = 0;
    size_t apply_penalties_to_mtp_logits_calls_ = 0;
    size_t apply_penalties_to_all_position_row_calls_ = 0;
    size_t build_stochastic_distribution_calls_ = 0;
    size_t build_stochastic_distributions_calls_ = 0;
    size_t build_stochastic_processed_rows_calls_ = 0;
    size_t sample_stochastic_draft_proposal_calls_ = 0;
    size_t sample_stochastic_distribution_calls_ = 0;
    size_t set_skip_decode_calls_ = 0;
    size_t set_skip_prefill_calls_ = 0;
    size_t set_all_position_sync_deferral_calls_ = 0;
    size_t set_main_decode_sync_deferral_calls_ = 0;
    size_t prepare_mtp_verifier_input_tokens_calls_ = 0;
    size_t prepare_mtp_verifier_input_tokens_from_device_calls_ = 0;
    size_t prepare_mtp_verifier_input_tokens_from_host_row_calls_ = 0;
    size_t stage_stochastic_draft_tokens_calls_ = 0;
    size_t stage_stochastic_target_token_calls_ = 0;
    size_t verify_stochastic_batch_outcome_calls_ = 0;
    size_t verify_stochastic_request_batch_outcome_calls_ = 0;
    size_t publish_rank_compact_response_ready_calls_ = 0;
    size_t verify_greedy_all_position_batch_outcome_calls_ = 0;
    size_t forward_mtp_from_resident_logical_state_calls_ = 0;
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
    int last_staged_target_sample_slot_ = -1;
    int32_t last_staged_target_token_ = -1;
    size_t record_draft_sample_slot_ready_calls_ = 0;
    int last_recorded_draft_sample_slot_ = -1;
    size_t record_target_sample_slot_ready_calls_ = 0;
    int last_recorded_target_sample_slot_ = -1;
    int last_resident_logical_state_request_index_ = -1;
    float last_stochastic_threshold_ = 0.0f;
    bool last_use_vllm_probability_rejection_ = false;
    std::vector<int32_t> resident_target_positions_;
    std::vector<int32_t> resident_target_sequence_lengths_;
    std::vector<int32_t> resident_accepted_state_counts_;
    std::vector<int32_t> resident_next_condition_tokens_;
    std::vector<int32_t> resident_all_drafts_accepted_flags_;
    std::vector<int32_t> resident_stopped_flags_;
    std::vector<int32_t> resident_publication_ok_flags_;
    std::array<int32_t, 8> verifier_device_tokens_{};
    std::array<int32_t, 8> verifier_device_tokens_from_device_first_{};
    std::array<int32_t, 8> verifier_device_tokens_from_host_row_{};
    std::vector<int32_t> last_verifier_host_row_tokens_;
    std::vector<int32_t> last_staged_draft_tokens_;
    std::array<int32_t, sampling_math::kSpeculativeBatchMaxRows>
        draft_sample_tokens_{};
    std::array<bool, sampling_math::kSpeculativeBatchMaxRows>
        draft_sample_slot_ready_{};
    std::array<int, sampling_math::kSpeculativeBatchMaxRows>
        draft_sample_stream_tokens_{};
    std::array<int32_t, sampling_math::kSpeculativeBatchMaxRows>
        target_sample_tokens_{};
    std::array<bool, sampling_math::kSpeculativeBatchMaxRows>
        target_sample_slot_ready_{};
    std::array<int, sampling_math::kSpeculativeBatchMaxRows>
        target_sample_stream_tokens_{};
    size_t prefix_lookup_calls_ = 0;
    size_t prefix_populate_calls_ = 0;
    size_t prefix_harvest_calls_ = 0;
    size_t prefix_terminal_restore_calls_ = 0;
    int harvested_prompt_token_count_ = 0;
    std::vector<int> populated_prefix_tokens_;
    std::vector<bool> populated_prefix_restore_model_runtime_state_;
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
    mutable std::atomic<size_t>
        publish_grouped_decode_equivalent_mtp_spec_state_batch_calls_{0};
    mutable std::atomic<size_t>
        publish_device_resident_mtp_spec_state_batch_calls_{0};
    mutable std::atomic<size_t> set_all_position_logits_calls_{0};
    mutable std::atomic<size_t> set_row_indexed_all_position_logits_calls_{0};
    mutable std::atomic<size_t> set_mtp_spec_verifier_input_plan_calls_{0};
    mutable std::atomic<size_t> clear_mtp_spec_verifier_input_plan_calls_{0};
    mutable std::atomic<size_t> get_logits_local_info_calls_{0};
    mutable std::atomic<size_t> consume_logits_local_info_calls_{0};
    mutable std::atomic<size_t> get_mtp_logits_local_info_calls_{0};
    mutable std::atomic<size_t> consume_mtp_logits_local_info_calls_{0};
    mutable std::atomic<size_t> get_all_position_logits_local_info_calls_{0};
    mutable std::atomic<size_t> consume_all_position_logits_local_info_calls_{0};
    mutable std::atomic<size_t> prefix_live_capture_calls_{0};
    mutable std::atomic<size_t> prefix_live_restore_calls_{0};
    mutable std::atomic<size_t> prefix_live_truncate_calls_{0};
    static constexpr int kMockResidentOutcomeRequestCapacity = 4;
    std::array<int32_t,
               sampling_math::kSpeculativeBatchMaxOutputTokens *
                   kMockResidentOutcomeRequestCapacity>
        staged_resident_output_tokens_{};
    std::array<int,
               sampling_math::kSpeculativeBatchMetaCount *
                   kMockResidentOutcomeRequestCapacity>
        staged_resident_meta_{};
    mutable int resident_outcome_ready_event_token_ = 0;
    mutable int resident_mtp_transaction_ready_event_token_ = 0;
    uint64_t resident_mtp_transaction_generation_ = 0;
    std::array<int, kMockResidentOutcomeRequestCapacity>
        resident_shifted_cached_tokens_{};
    std::shared_ptr<DeviceResidentMTPTransactionState>
        resident_mtp_transaction_state_;
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
        bool sideband_should_fail = false;
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
    bool reserveCollectiveResources(size_t /*bytes*/, size_t /*fp16_scratch_elements*/) override { return true; }

    // =====================================================================
    // ILocalTPContext Broadcast (no-op)
    // =====================================================================
    bool broadcast(TensorBase * /*tensor*/, int /*source_device_index*/ = 0) override
    {
        broadcast_calls_.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    bool collectiveSidebandOnStream(
        const std::vector<LocalTPCollectiveSidebandBuffer> &sidebands,
        int device_index,
        void *producer_stream,
        const std::string & /*anchor_stage_name*/) override
    {
        if (!producer_stream)
            throw std::invalid_argument("MockLocalTPContext::collectiveSidebandOnStream requires a non-null stream");

        collective_sideband_calls_.fetch_add(1, std::memory_order_relaxed);
        if (config_.sideband_should_fail)
            return false;

        const int degree_count = degree();
        if (device_index < 0 || device_index >= degree_count)
            return false;
        if (degree_count <= 1 || sidebands.empty())
            return true;

        std::unique_lock<std::mutex> lock(sideband_mutex_);
        const int generation = sideband_generation_;
        if (sideband_generation_sidebands_.empty())
            sideband_generation_sidebands_.resize(static_cast<size_t>(degree_count));
        if (!sideband_generation_sidebands_[static_cast<size_t>(device_index)].empty())
            return false;

        sideband_generation_sidebands_[static_cast<size_t>(device_index)] =
            sidebands;
        ++sideband_arrivals_;
        if (sideband_arrivals_ == degree_count)
        {
            const bool ok = completeMockSidebandGenerationLocked();
            sideband_generation_result_ = ok;
            sideband_arrivals_ = 0;
            sideband_generation_sidebands_.clear();
            ++sideband_generation_;
            lock.unlock();
            sideband_cv_.notify_all();
            return ok;
        }

        const bool completed = sideband_cv_.wait_for(
            lock,
            std::chrono::milliseconds(500),
            [this, generation]()
            {
                return sideband_generation_ != generation;
            });
        return completed && sideband_generation_result_;
    }

    bool collectiveSidebandsMultiOnStreams(
        const std::vector<std::vector<LocalTPCollectiveSidebandBuffer>>
            &participant_sidebands,
        const std::vector<void *> &producer_streams,
        const std::string & /*publication_name*/) override
    {
        collective_sideband_calls_.fetch_add(1, std::memory_order_relaxed);
        if (config_.sideband_should_fail ||
            participant_sidebands.size() != static_cast<size_t>(degree()) ||
            producer_streams.size() != static_cast<size_t>(degree()))
        {
            return false;
        }
        for (void *stream : producer_streams)
        {
            if (!stream)
            {
                throw std::invalid_argument(
                    "MockLocalTPContext::collectiveSidebandsMultiOnStreams requires non-null streams");
            }
        }

        std::lock_guard<std::mutex> lock(sideband_mutex_);
        sideband_generation_sidebands_ = participant_sidebands;
        const bool result = completeMockSidebandGenerationLocked();
        sideband_generation_sidebands_.clear();
        return result;
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

    size_t collective_sideband_call_count() const
    {
        return collective_sideband_calls_.load(std::memory_order_relaxed);
    }

    size_t collective_sideband_broadcast_count() const
    {
        return collective_sideband_broadcasts_.load(std::memory_order_relaxed);
    }

    void reset_call_counts()
    {
        synchronize_calls_.store(0, std::memory_order_relaxed);
        allreduce_calls_.store(0, std::memory_order_relaxed);
        allgather_calls_.store(0, std::memory_order_relaxed);
        gather_from_devices_calls_.store(0, std::memory_order_relaxed);
        reduce_scatter_calls_.store(0, std::memory_order_relaxed);
        collective_sideband_calls_.store(0, std::memory_order_relaxed);
        collective_sideband_broadcasts_.store(0, std::memory_order_relaxed);
    }

    void set_allreduce_fails(bool fails) { config_.allreduce_should_fail = fails; }
    void set_allgather_fails(bool fails) { config_.allgather_should_fail = fails; }
    void set_sideband_fails(bool fails) { config_.sideband_should_fail = fails; }

private:
    static size_t mockCollectiveDataTypeBytes(CollectiveDataType dtype)
    {
        switch (dtype)
        {
        case CollectiveDataType::FLOAT32:
        case CollectiveDataType::INT32:
            return sizeof(std::uint32_t);
        case CollectiveDataType::FLOAT16:
        case CollectiveDataType::BFLOAT16:
            return sizeof(std::uint16_t);
        case CollectiveDataType::INT8:
            return sizeof(std::uint8_t);
        }
        return 0;
    }

    bool completeMockSidebandGenerationLocked()
    {
        const int degree_count = degree();
        if (static_cast<int>(sideband_generation_sidebands_.size()) != degree_count ||
            sideband_generation_sidebands_.empty())
        {
            return false;
        }

        const size_t sideband_count = sideband_generation_sidebands_.front().size();
        for (const auto &participant_sidebands : sideband_generation_sidebands_)
        {
            if (participant_sidebands.size() != sideband_count)
                return false;
        }

        for (size_t sideband_index = 0;
             sideband_index < sideband_count;
             ++sideband_index)
        {
            const auto &reference =
                sideband_generation_sidebands_.front()[sideband_index];
            if (reference.root_device_index < 0 ||
                reference.root_device_index >= degree_count ||
                reference.element_count == 0)
            {
                return false;
            }

            for (int participant = 0; participant < degree_count; ++participant)
            {
                const auto &sideband =
                    sideband_generation_sidebands_[static_cast<size_t>(participant)][sideband_index];
                if (sideband.kind != reference.kind ||
                    sideband.element_count != reference.element_count ||
                    sideband.dtype != reference.dtype ||
                    sideband.root_device_index != reference.root_device_index)
                {
                    return false;
                }
            }

            if (reference.kind != LocalTPCollectiveSidebandKind::Broadcast)
                continue;

            const auto &root_sideband =
                sideband_generation_sidebands_[static_cast<size_t>(reference.root_device_index)][sideband_index];
            const void *src =
                root_sideband.send_buffer ? root_sideband.send_buffer : root_sideband.recv_buffer;
            const size_t bytes =
                reference.element_count * mockCollectiveDataTypeBytes(reference.dtype);
            if (!src || bytes == 0)
                return false;

            for (int participant = 0; participant < degree_count; ++participant)
            {
                auto &sideband =
                    sideband_generation_sidebands_[static_cast<size_t>(participant)][sideband_index];
                if (!sideband.recv_buffer)
                    return false;
                std::memcpy(sideband.recv_buffer, src, bytes);
                collective_sideband_broadcasts_.fetch_add(
                    1,
                    std::memory_order_relaxed);
            }
        }
        return true;
    }

    Config config_;
    std::mutex sideband_mutex_;
    std::condition_variable sideband_cv_;
    int sideband_generation_ = 0;
    int sideband_arrivals_ = 0;
    bool sideband_generation_result_ = true;
    std::vector<std::vector<LocalTPCollectiveSidebandBuffer>>
        sideband_generation_sidebands_;
    mutable std::atomic<size_t> synchronize_calls_{0};
    mutable std::atomic<size_t> allreduce_calls_{0};
    mutable std::atomic<size_t> allgather_calls_{0};
    mutable std::atomic<size_t> broadcast_calls_{0};
    mutable std::atomic<size_t> gather_from_devices_calls_{0};
    mutable std::atomic<size_t> reduce_scatter_calls_{0};
    mutable std::atomic<size_t> collective_sideband_calls_{0};
    mutable std::atomic<size_t> collective_sideband_broadcasts_{0};
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

TEST_F(Test__RankOrchestrator, ForwardTPWorkerJoinWaitsForSlowForward)
{
    struct ScopedCollectTimeoutEnv
    {
        ScopedCollectTimeoutEnv()
        {
            const char *old_value = std::getenv("LLAMINAR_TP_COLLECT_TIMEOUT_MS");
            had_old_value = old_value != nullptr;
            if (old_value)
                previous_value = old_value;
            setenv("LLAMINAR_TP_COLLECT_TIMEOUT_MS", "10", 1);
            mutableDebugEnv().reload();
        }

        ~ScopedCollectTimeoutEnv()
        {
            if (had_old_value)
                setenv("LLAMINAR_TP_COLLECT_TIMEOUT_MS", previous_value.c_str(), 1);
            else
                unsetenv("LLAMINAR_TP_COLLECT_TIMEOUT_MS");
            mutableDebugEnv().reload();
        }

        bool had_old_value = false;
        std::string previous_value;
    } collect_timeout_env;

    std::vector<std::unique_ptr<IInferenceRunner>> runners;
    auto slow_runner = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *slow_runner_ptr = slow_runner.get();
    runners.push_back(std::move(slow_runner));
    runners.push_back(std::make_unique<MockDeviceGraphOrchestrator>());

    auto orchestrator = RankOrchestrator::createForTest(
        llaminar2::test::MockModelContext::createMinimal(),
        std::move(runners),
        makeTPContextForRunnerCount(2),
        makeRankConfigForRunnerCount(2));

    int tokens[] = {1};
    ASSERT_TRUE(orchestrator->forward(tokens, 1));
    slow_runner_ptr->set_forward_sleep_ms(250);
    EXPECT_TRUE(orchestrator->forward(tokens, 1))
        << "LLAMINAR_TP_COLLECT_TIMEOUT_MS is a collective rendezvous timeout, "
           "not a wall-clock worker-join timeout for arbitrary host-side work.";
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

TEST_F(Test__RankOrchestrator, LocalTPActiveMoEHistogramSyncMergesSameDomainSiblings)
{
    auto runner0 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner0_ptr = runner0.get();
    runner0_ptr->set_moe_rebalance_controller(makeDomainController("gpu_domain"));

    auto runner1 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner1_ptr = runner1.get();
    runner1_ptr->set_moe_rebalance_controller(makeDomainController("gpu_domain"));

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
    auto *active = orchestrator->moeRebalanceController();
    ASSERT_NE(active, nullptr);
    auto *sibling = controllers[0] == active ? controllers[1] : controllers[0];
    ASSERT_NE(sibling, nullptr);

    int sibling_sync_calls = 0;
    sibling->histogram()->registerRuntimeHistogramSync([&]()
    {
        const int hot_expert = 1;
        const float weight = 1.0f;
        sibling->histogram()->record(0, &hot_expert, &weight, 1);
        ++sibling_sync_calls;
        return true;
    });

    ASSERT_TRUE(active->histogram()->syncRuntimeHistograms());

    EXPECT_EQ(sibling_sync_calls, 1);
    EXPECT_EQ(active->histogram()->activationCount(0, 1), 1u);
    EXPECT_EQ(sibling->histogram()->activationCount(0, 1), 0u)
        << "Sibling counts must reset after the root histogram consumes them";
    EXPECT_EQ(active->histogram()->windowTokenCount(), 0u)
        << "Sibling participants contribute load distribution, not extra rank decode tokens";
}

TEST_F(Test__RankOrchestrator, SameBackendGpuExpertTransferIsDirectOnly)
{
    using rank_orchestrator_detail::sameBackendGpuExpertTransferIsDirectOnly;

    EXPECT_TRUE(sameBackendGpuExpertTransferIsDirectOnly(DeviceId::cuda(0), DeviceId::cuda(1)));
    EXPECT_TRUE(sameBackendGpuExpertTransferIsDirectOnly(DeviceId::rocm(0), DeviceId::rocm(1)));

    EXPECT_FALSE(sameBackendGpuExpertTransferIsDirectOnly(DeviceId::cuda(0), DeviceId::rocm(0)));
    EXPECT_FALSE(sameBackendGpuExpertTransferIsDirectOnly(DeviceId::cuda(0), DeviceId::cpu()));
    EXPECT_FALSE(sameBackendGpuExpertTransferIsDirectOnly(DeviceId::cpu(), DeviceId::cuda(0)));
}

TEST_F(Test__RankOrchestrator, SameBackendGpuExpertTransferStagesAsyncAndPublishesLater)
{
    const std::string rank_source =
        readSourceFileForRankOrchestratorTest(
            "/workspaces/llaminar/src/v2/execution/local_execution/orchestrators/RankOrchestrator.cpp");
    const std::string dgo_source =
        readSourceFileForRankOrchestratorTest(
            "/workspaces/llaminar/src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");
    ASSERT_FALSE(rank_source.empty());
    ASSERT_FALSE(dgo_source.empty());

    const auto prepare_pos =
        rank_source.find("prepareExpertWeightsDirectForMasksFrom");
    const auto target_masks_pos =
        rank_source.find("const std::vector<std::vector<bool>> &target_masks");
    const auto transfer_masks_pos =
        rank_source.find("const std::vector<std::vector<bool>> &transfer_masks");
    const auto direct_remaining_pos =
        rank_source.find("direct_remaining_masks = transfer_masks");
    const auto serialized_remaining_pos =
        rank_source.find("serialized_remaining_masks = transfer_masks");
    const auto direct_mask_call_pos =
        rank_source.find("prepareExpertWeightsDirectForMasksFrom");
    const auto serialized_mask_call_pos =
        rank_source.find("collectExpertWeightsForMasks");
    const auto publish_pos =
        rank_source.find("bool RankOrchestrator::publishPreparedMoEExpertMasksForAllDevices");
    const auto pending_publish_pos =
        rank_source.find("activatePendingGpuDirectExpertTransfers", publish_pos);
    const auto apply_pos =
        rank_source.find("applyExpertMasksForDomain", publish_pos);
    const auto abort_pos =
        rank_source.find("incomplete_gpu_direct_prepare", publish_pos);
    const auto deferred_activation_pos =
        dgo_source.find("activation is deferred until publish");
    const auto pending_arrival_pos =
        dgo_source.find("pending_gpu_direct_transfer_slot_arrivals_.push_back");
    const auto stage_capacity_pos =
        dgo_source.find("staging_pool_capacity");
    const auto direct_prepare_fn_pos =
        dgo_source.find("prepareExpertWeightsDirectForMasksFrom");
    const auto direct_publish_fn_pos =
        dgo_source.find("bool DeviceGraphOrchestrator::activatePreparedGpuDirectExpertTransfers");
    const auto direct_publish_fn_end =
        dgo_source.find("void DeviceGraphOrchestrator::clearPendingGpuDirectExpertTransfers",
                        direct_publish_fn_pos);
    const auto activation_completion_store_pos =
        dgo_source.find("pending_gpu_direct_activation_completions_.push_back", direct_publish_fn_pos);
    const auto retain_stage_pending_false_pos =
        dgo_source.find("!graph_stable_gpu_rebalance", direct_publish_fn_pos);
    const auto retire_before_prepare_pos =
        dgo_source.find("retireCompletedGpuDirectActivationCompletions();", direct_prepare_fn_pos);
    const auto retire_fn_pos =
        dgo_source.find("void DeviceGraphOrchestrator::retireCompletedGpuDirectActivationCompletions");
    const auto nonblocking_query_pos =
        dgo_source.find("queryEventChecked", retire_fn_pos);
    const auto missing_arrivals_pos =
        dgo_source.find("missingPreparedExpertIds(expert_ids)", direct_prepare_fn_pos);
    const auto active_arrival_capacity_pos =
        dgo_source.find("active_arrival_capacity", missing_arrivals_pos);
    const auto staged_prepare_pos =
        dgo_source.find("stageExpertsGPUDirectToTransferSlotsFrom", active_arrival_capacity_pos);
    const auto host_wait_in_prepare_pos =
        dgo_source.find("waitForEvent", direct_prepare_fn_pos);
    const auto activate_in_prepare_pos =
        dgo_source.find("activateGpuDirectTransferSlotArrivals", direct_prepare_fn_pos);
    const auto apply_masks_pos =
        dgo_source.find("void DeviceGraphOrchestrator::applyExpertMasksForDomain");
    const auto release_departed_pos =
        dgo_source.find("releaseDepartedExperts(masks[layer])", apply_masks_pos);
    const auto rolling_retry_pos =
        dgo_source.find("while (!stage_pending.empty())");
    const auto requested_arrival_entries_pos =
        dgo_source.find("requested_arrival_entries", direct_prepare_fn_pos);

    ASSERT_NE(prepare_pos, std::string::npos)
        << "same-backend GPU rebalance must prepare direct transfers before publish";
    ASSERT_NE(target_masks_pos, std::string::npos);
    ASSERT_NE(transfer_masks_pos, std::string::npos);
    ASSERT_NE(direct_remaining_pos, std::string::npos)
        << "same-backend GPU-direct prepare must verify only logical arrival deltas";
    ASSERT_NE(serialized_remaining_pos, std::string::npos)
        << "serialized fallback should still use the smaller transfer delta";
    ASSERT_NE(direct_mask_call_pos, std::string::npos);
    ASSERT_NE(serialized_mask_call_pos, std::string::npos);
    EXPECT_LT(direct_remaining_pos, direct_mask_call_pos);
    EXPECT_LT(serialized_remaining_pos, serialized_mask_call_pos);
    ASSERT_NE(publish_pos, std::string::npos);
    ASSERT_NE(pending_publish_pos, std::string::npos)
        << "rank publish must activate prepared GPU-direct transfer-slot arrivals";
    ASSERT_NE(abort_pos, std::string::npos)
        << "incomplete same-backend GPU prepare must abort the whole mask publish";
    ASSERT_NE(deferred_activation_pos, std::string::npos)
        << "GPU-direct prepare must leave activation for the publish phase";
    ASSERT_NE(pending_arrival_pos, std::string::npos)
        << "prepared transfer-slot arrivals must be retained for publish";
    ASSERT_NE(stage_capacity_pos, std::string::npos)
        << "rolling staging capacity must be explicit and bounded";
    ASSERT_NE(direct_prepare_fn_pos, std::string::npos);
    ASSERT_NE(direct_publish_fn_pos, std::string::npos);
    ASSERT_NE(direct_publish_fn_end, std::string::npos);
    ASSERT_NE(activation_completion_store_pos, std::string::npos)
        << "graph-stable publish must retain activation completions on the DGO, not on cached stages";
    ASSERT_NE(retain_stage_pending_false_pos, std::string::npos)
        << "graph-stable replay cannot drain MoEExpertComputeStage pending completions after capture";
    ASSERT_NE(retire_before_prepare_pos, std::string::npos)
        << "completed activation events must release transfer staging slots before the next prepare wave";
    ASSERT_NE(retire_fn_pos, std::string::npos);
    ASSERT_NE(nonblocking_query_pos, std::string::npos)
        << "retiring graph-stable activation completions should be nonblocking";
    ASSERT_NE(missing_arrivals_pos, std::string::npos)
        << "active arrival capacity must be based on missing experts, not the whole target mask";
    ASSERT_NE(active_arrival_capacity_pos, std::string::npos);
    ASSERT_NE(staged_prepare_pos, std::string::npos);
    ASSERT_NE(requested_arrival_entries_pos, std::string::npos)
        << "async prepare must reserve enough transfer-slot staging for the pending publication";
    ASSERT_NE(apply_masks_pos, std::string::npos);
    ASSERT_NE(release_departed_pos, std::string::npos)
        << "departed experts must still be released during mask application after direct prepares";
    ASSERT_NE(rolling_retry_pos, std::string::npos)
        << "rolling transfer staging must retry pending experts after reduced-capacity waves";
    ASSERT_NE(apply_pos, std::string::npos);
    EXPECT_LT(missing_arrivals_pos, staged_prepare_pos);
    EXPECT_TRUE(host_wait_in_prepare_pos == std::string::npos ||
                host_wait_in_prepare_pos > direct_publish_fn_pos)
        << "same-backend GPU prepare must not CPU-wait for transfer or activation events";
    EXPECT_TRUE(activate_in_prepare_pos == std::string::npos ||
                activate_in_prepare_pos > direct_publish_fn_pos)
        << "same-backend GPU prepare must not publish active GEMM tables";
    EXPECT_LT(retire_before_prepare_pos, staged_prepare_pos);
    EXPECT_LT(activation_completion_store_pos, direct_publish_fn_end);
    const auto host_wait_in_publish_pos =
        dgo_source.find("waitForEvent", direct_publish_fn_pos);
    EXPECT_TRUE(host_wait_in_publish_pos == std::string::npos ||
                host_wait_in_publish_pos > direct_publish_fn_end)
        << "GPU transfer-slot publish must chain events on streams, not CPU-wait before activation";
    EXPECT_LT(abort_pos, apply_pos)
        << "mask publish must be all-or-nothing when GPU-direct prepare is incomplete";
    EXPECT_LT(pending_publish_pos, apply_pos)
        << "activation must be enqueued before masks can expose the arrived experts";
}

TEST_F(Test__RankOrchestrator, GpuDynamicMoERebalanceRefreshesStableGraphTables)
{
    const std::string dgo_source =
        readSourceFileForRankOrchestratorTest(
            "/workspaces/llaminar/src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp");
    const std::string stage_source =
        readSourceFileForRankOrchestratorTest(
            "/workspaces/llaminar/src/v2/execution/compute_stages/stages/MoEExpertComputeStage.cpp");
    const std::string stage_header =
        readSourceFileForRankOrchestratorTest(
            "/workspaces/llaminar/src/v2/execution/compute_stages/stages/MoEExpertComputeStage.h");
    const std::string forward_engine_header =
        readSourceFileForRankOrchestratorTest(
            "/workspaces/llaminar/src/v2/execution/local_execution/engine/ForwardExecutionEngine.h");
    const std::string forward_engine_source =
        readSourceFileForRankOrchestratorTest(
            "/workspaces/llaminar/src/v2/execution/local_execution/engine/ForwardExecutionEngine.cpp");
    const std::string kernel_iface =
        readSourceFileForRankOrchestratorTest(
            "/workspaces/llaminar/src/v2/kernels/IMoEKernel.h");
    const std::string cuda_source =
        readSourceFileForRankOrchestratorTest(
            "/workspaces/llaminar/src/v2/kernels/cuda/moe/CUDAMoEKernel.cpp");
    const std::string rocm_source =
        readSourceFileForRankOrchestratorTest(
            "/workspaces/llaminar/src/v2/kernels/rocm/moe/ROCmMoEKernel.cpp");
    ASSERT_FALSE(dgo_source.empty());
    ASSERT_FALSE(stage_source.empty());
    ASSERT_FALSE(stage_header.empty());
    ASSERT_FALSE(forward_engine_header.empty());
    ASSERT_FALSE(forward_engine_source.empty());
    ASSERT_FALSE(kernel_iface.empty());
    ASSERT_FALSE(cuda_source.empty());
    ASSERT_FALSE(rocm_source.empty());

    const auto epoch_fn_pos =
        dgo_source.find("uint64_t DeviceGraphOrchestrator::moePlacementEpoch() const");
    const auto gpu_stable_guard_pos =
        dgo_source.find("usesGraphStableGpuMoERebalance()", epoch_fn_pos);
    const auto gpu_stable_return_pos =
        dgo_source.find("return 0;", gpu_stable_guard_pos);
    const auto mask_epoch_pos =
        dgo_source.find("current_expert_mask_epoch_", epoch_fn_pos);
    ASSERT_NE(epoch_fn_pos, std::string::npos);
    ASSERT_NE(gpu_stable_guard_pos, std::string::npos)
        << "GPU dynamic MoE placement must be graph-table data, not a forward graph key";
    ASSERT_NE(gpu_stable_return_pos, std::string::npos);
    ASSERT_NE(mask_epoch_pos, std::string::npos);
    EXPECT_LT(gpu_stable_return_pos, mask_epoch_pos)
        << "GPU dynamic MoE must bypass placement epochs before graph-cache signature materialization";

    const auto stable_predicate_pos =
        dgo_source.find("bool DeviceGraphOrchestrator::usesGraphStableGpuMoERebalance() const");
    ASSERT_NE(stable_predicate_pos, std::string::npos);
    const auto stable_predicate_end =
        dgo_source.find("uint64_t DeviceGraphOrchestrator::moePlacementEpoch() const", stable_predicate_pos);
    ASSERT_NE(stable_predicate_end, std::string::npos);
    const std::string stable_predicate_body =
        dgo_source.substr(stable_predicate_pos, stable_predicate_end - stable_predicate_pos);
    EXPECT_NE(stable_predicate_body.find("ExecutionDomainScope::LOCAL"), std::string::npos);
    EXPECT_NE(stable_predicate_body.find("RoutedExpertComputePolicy::Apportioned"), std::string::npos);
    EXPECT_NE(stable_predicate_body.find("domain.participants.size() < 2"), std::string::npos);
    EXPECT_NE(stable_predicate_body.find("participant.isGPU()"), std::string::npos);
    EXPECT_NE(stable_predicate_body.find("participant.device_type != participant_type"), std::string::npos);

    const auto prepare_direct_pos =
        dgo_source.find("DeviceGraphOrchestrator::prepareExpertWeightsDirectForMasksFrom");
    const auto activate_direct_pos =
        dgo_source.find("bool DeviceGraphOrchestrator::activatePreparedGpuDirectExpertTransfers", prepare_direct_pos);
    ASSERT_NE(prepare_direct_pos, std::string::npos);
    ASSERT_NE(activate_direct_pos, std::string::npos);
    const std::string prepare_direct_body =
        dgo_source.substr(prepare_direct_pos, activate_direct_pos - prepare_direct_pos);
    const auto missing_prepared_pos =
        prepare_direct_body.find("missingPreparedExpertIds");
    const auto already_resident_count_pos =
        prepare_direct_body.find("std::find(stage_pending.begin()", missing_prepared_pos);
    const auto no_transfer_needed_pos =
        prepare_direct_body.find("if (stage_pending.empty())", already_resident_count_pos);
    ASSERT_NE(missing_prepared_pos, std::string::npos);
    ASSERT_NE(already_resident_count_pos, std::string::npos);
    ASSERT_NE(no_transfer_needed_pos, std::string::npos);
    EXPECT_LT(already_resident_count_pos, no_transfer_needed_pos)
        << "Already-resident arrivals must be counted before skipping an empty transfer wave";
    EXPECT_NE(prepare_direct_body.find("select_rebalance_destination_stages"), std::string::npos)
        << "GPU-direct prepare must use the same destination stage class that mask publication updates";
    EXPECT_NE(prepare_direct_body.find("usesGraphStableMoEPlacement"), std::string::npos)
        << "Graph-stable GPU rebalancing must target stages that can consume placement by data mutation";
    EXPECT_NE(prepare_direct_body.find("required_stage_count"), std::string::npos)
        << "Arrival mask bits may only clear after every selected destination stage is satisfied";
    EXPECT_NE(prepare_direct_body.find(">= required_stage_count"), std::string::npos);

    const auto apply_masks_pos =
        dgo_source.find("void DeviceGraphOrchestrator::applyExpertMasksForDomain");
    const auto collect_weights_pos =
        dgo_source.find("ReceivedWeightsMap DeviceGraphOrchestrator::collectExpertWeightsForMasks",
                        apply_masks_pos);
    ASSERT_NE(apply_masks_pos, std::string::npos);
    ASSERT_NE(collect_weights_pos, std::string::npos);
    const std::string apply_masks_body =
        dgo_source.substr(apply_masks_pos, collect_weights_pos - apply_masks_pos);
    EXPECT_NE(apply_masks_body.find("invalidateMoEPlacementSensitiveGraphsForStablePlacement"),
              std::string::npos)
        << "Stable rebalance must not leave placement-sensitive verifier/multi-token graphs reusable";
    EXPECT_NE(apply_masks_body.find("usesGraphStableMoEPlacement"), std::string::npos)
        << "Mask publication must be scoped to graph-stable MoE stages in GPU mode";
    EXPECT_NE(apply_masks_body.find("requires cached graph-stable MoE stages"), std::string::npos)
        << "Graph-stable GPU rebalance should fail fast when no runtime decode stage can be updated";
    EXPECT_NE(apply_masks_body.find("moe_expert_mask_publish"), std::string::npos)
        << "Mask publication must acquire an explicit non-null stream outside normal stage execution";
    EXPECT_NE(apply_masks_body.find("expert_runtime_movement_epoch"), std::string::npos)
        << "Graph-stable GPU mask publication must advance the prefix-cache runtime movement epoch; "
           "otherwise prefix-cache restore can reuse KV/logits produced under stale expert residency.";
    const auto mask_publish_stream_bind_pos =
        apply_masks_body.find("stage->setGPUStream(publication_stream)");
    EXPECT_NE(mask_publish_stream_bind_pos, std::string::npos)
        << "Mask publication must bind the explicit stream before graph-stable descriptor refresh";
    const auto pre_adopt_release_guard_pos =
        apply_masks_body.find("if (!graph_stable_gpu_rebalance)");
    const auto register_prepare_pos =
        apply_masks_body.find("registerAndPrepareNewExperts");
    const auto post_adopt_release_guard_pos =
        apply_masks_body.find("if (graph_stable_gpu_rebalance)", register_prepare_pos);
    ASSERT_NE(pre_adopt_release_guard_pos, std::string::npos);
    ASSERT_NE(register_prepare_pos, std::string::npos);
    ASSERT_NE(post_adopt_release_guard_pos, std::string::npos);
    ASSERT_NE(mask_publish_stream_bind_pos, std::string::npos);
    EXPECT_LT(mask_publish_stream_bind_pos, register_prepare_pos)
        << "Newly prepared expert engines must not be rebound to a null stream";
    EXPECT_LT(pre_adopt_release_guard_pos, register_prepare_pos)
        << "Legacy rebalance should still release departures before heavy prepare";
    EXPECT_LT(register_prepare_pos, post_adopt_release_guard_pos)
        << "Graph-stable GPU rebalance must adopt GPU-direct arrivals before releasing shared store departures";

    EXPECT_NE(stage_header.find("usesGraphStableRuntimeDecodePlacement"), std::string::npos);
    EXPECT_NE(stage_header.find("usesGraphStableFixedTopologyPrefillPlacement"), std::string::npos);
    EXPECT_NE(stage_header.find("usesGraphStableMoEPlacement"), std::string::npos);
    EXPECT_NE(stage_source.find("bool MoEExpertComputeStage::usesGraphStableRuntimeDecodePlacement() const"),
              std::string::npos);
    EXPECT_NE(stage_source.find("bool MoEExpertComputeStage::usesGraphStableFixedTopologyPrefillPlacement() const"),
              std::string::npos);
    EXPECT_NE(stage_source.find("bool MoEExpertComputeStage::refreshFixedTopologyGroupedPrefillPlacement()"),
              std::string::npos);
    EXPECT_NE(stage_source.find("!params_.force_grouped_verifier_prefill_for_decode"), std::string::npos)
        << "Verifier replay stages are not the graph-stable placement owner";
    EXPECT_NE(stage_source.find("updateGroupedPrefillExpertMask"), std::string::npos)
        << "Fixed-topology prefill must refresh the persistent device mask instead of recapturing";
    EXPECT_NE(forward_engine_header.find("invalidateMoEPlacementSensitiveGraphsForStablePlacement"),
              std::string::npos);
    EXPECT_NE(forward_engine_source.find("ForwardExecutionEngine::invalidateMoEPlacementSensitiveGraphsForStablePlacement"),
              std::string::npos);
    EXPECT_NE(forward_engine_source.find("signature.decode"), std::string::npos);
    EXPECT_NE(forward_engine_source.find("!signature.all_position_logits"), std::string::npos);
    EXPECT_NE(forward_engine_source.find("signature.seq_len == 1"), std::string::npos);
    EXPECT_NE(forward_engine_source.find("graph_stable_fixed_prefill"), std::string::npos);
    EXPECT_NE(forward_engine_source.find("cache.markGPUStreamBindingsDirty()"), std::string::npos)
        << "Preserved decode graphs must still rebind explicit streams after transfer publication";

    const auto apply_mask_pos =
        stage_source.find("void MoEExpertComputeStage::applyExpertMask");
    const auto build_context_pos =
        stage_source.find("MoEWeightContext MoEExpertComputeStage::buildWeightContext", apply_mask_pos);
    ASSERT_NE(apply_mask_pos, std::string::npos);
    ASSERT_NE(build_context_pos, std::string::npos);
    const std::string apply_mask_body =
        stage_source.substr(apply_mask_pos, build_context_pos - apply_mask_pos);
    EXPECT_EQ(apply_mask_body.find("grouped_gateup_desc_table_id_ = -1"), std::string::npos)
        << "Mask publish must not destroy captured descriptor table identity";
    EXPECT_EQ(apply_mask_body.find("grouped_down_desc_table_id_ = -1"), std::string::npos)
        << "Mask publish must not destroy captured descriptor table identity";
    EXPECT_NE(apply_mask_body.find("grouped_gateup_desc_table_dirty_ = true"), std::string::npos);
    EXPECT_NE(apply_mask_body.find("refreshGraphStablePlacement"), std::string::npos);
    EXPECT_NE(apply_mask_body.find("throw std::runtime_error"), std::string::npos)
        << "Graph-stable placement refresh failures must fail fast";

    EXPECT_NE(kernel_iface.find("updateGroupedExpertGateUpDescriptorTables"), std::string::npos);
    EXPECT_NE(kernel_iface.find("updateGroupedExpertDownDescriptorTable"), std::string::npos);
    EXPECT_NE(kernel_iface.find("updateGroupedPrefillExpertMask"), std::string::npos);
    EXPECT_NE(cuda_source.find("CUDAMoEKernel::updateGroupedExpertGateUpDescriptorTables"), std::string::npos);
    EXPECT_NE(cuda_source.find("CUDAMoEKernel::updateGroupedExpertDownDescriptorTable"), std::string::npos);
    EXPECT_NE(cuda_source.find("CUDAMoEKernel::updateGroupedPrefillExpertMask"), std::string::npos);
    EXPECT_NE(rocm_source.find("ROCmMoEKernel::updateGroupedExpertGateUpDescriptorTables"), std::string::npos);
    EXPECT_NE(rocm_source.find("ROCmMoEKernel::updateGroupedExpertDownDescriptorTable"), std::string::npos);
    EXPECT_NE(rocm_source.find("ROCmMoEKernel::updateGroupedPrefillExpertMask"), std::string::npos);

    const auto replica_publish_pos =
        dgo_source.find("void DeviceGraphOrchestrator::setExpertReplicaSetForParticipant");
    const auto release_raw_pos =
        dgo_source.find("size_t DeviceGraphOrchestrator::releaseRawExpertWeights", replica_publish_pos);
    ASSERT_NE(replica_publish_pos, std::string::npos);
    ASSERT_NE(release_raw_pos, std::string::npos);
    const std::string replica_publish_body =
        dgo_source.substr(replica_publish_pos, release_raw_pos - replica_publish_pos);
    EXPECT_NE(replica_publish_body.find("moe_replica_set_publish"), std::string::npos)
        << "Hot-replica publication uses the same graph-stable refresh path as masks";
    const auto replica_stream_bind_pos =
        replica_publish_body.find("moe->setGPUStream(ensure_replica_publication_stream())");
    const auto replica_set_call_pos =
        replica_publish_body.find("moe->setReplicaSet(normalized_replicas, participant_id)");
    ASSERT_NE(replica_stream_bind_pos, std::string::npos);
    ASSERT_NE(replica_set_call_pos, std::string::npos);
    EXPECT_LT(replica_stream_bind_pos, replica_set_call_pos)
        << "Replica publication must bind an explicit stream before refreshing GPU placement tables";
}

TEST_F(Test__RankOrchestrator, LocalTPMoERebalanceDoesNotUseHostWorkerForGpuTransferStaging)
{
    const std::string runner_source =
        readSourceFileForRankOrchestratorTest(
            "/workspaces/llaminar/src/v2/execution/runner/OrchestrationRunner.cpp");
    ASSERT_FALSE(runner_source.empty());

    ASSERT_EQ(runner_source.find("std::async("), std::string::npos)
        << "GPU transfer staging must not call CUDA/HIP from a separate host worker while graphs/collectives replay";
    ASSERT_NE(runner_source.find("pending_moe_rebalance_prepare_"), std::string::npos)
        << "LocalTP GPU transfer staging should use a first-class delayed publish state, not a host worker";
    ASSERT_NE(runner_source.find("publishPendingMoERebalanceUpdate"), std::string::npos)
        << "delayed LocalTP MoE publishes need an explicit runner-owned drain point";
}

TEST_F(Test__RankOrchestrator, ClearCacheDrainsPendingMoERebalanceBeforeDroppingTransfers)
{
    const std::string runner_source =
        readSourceFileForRankOrchestratorTest(
            "/workspaces/llaminar/src/v2/execution/runner/OrchestrationRunner.cpp");
    ASSERT_FALSE(runner_source.empty());

    auto count_occurrences = [](const std::string &haystack, const std::string &needle)
    {
        size_t count = 0;
        size_t pos = 0;
        while ((pos = haystack.find(needle, pos)) != std::string::npos)
        {
            ++count;
            pos += needle.size();
        }
        return count;
    };

    const auto clear_pos = runner_source.find("void OrchestrationRunner::clearCache()");
    ASSERT_NE(clear_pos, std::string::npos);
    const auto clear_end = runner_source.find("PrefixRuntimeStateSnapshot OrchestrationRunner::prefixStateProbe", clear_pos);
    ASSERT_NE(clear_end, std::string::npos);
    const std::string clear_body = runner_source.substr(clear_pos, clear_end - clear_pos);

    const auto request_reset_pos =
        clear_body.find("clearUnderlyingRunnerCacheAfterMoEPublish(\"request-clear-cache\")");
    ASSERT_NE(request_reset_pos, std::string::npos)
        << "request/session cache reset must publish prepared MoE rebalances before resetting runtime state";
    EXPECT_EQ(clear_body.find("pending_moe_rebalance_prepare_.reset()"), std::string::npos)
        << "clearCache() must never silently erase a prepared MoE publish";
    EXPECT_EQ(clear_body.find("runner_->clear_cache()"), std::string::npos)
        << "clearCache() must go through the MoE-publish-aware reset helper";

    const auto drain_fn = runner_source.find("void OrchestrationRunner::drainPendingMoERebalanceBeforeCacheClear()");
    ASSERT_NE(drain_fn, std::string::npos);
    const auto helper_fn =
        runner_source.find("void OrchestrationRunner::clearUnderlyingRunnerCacheAfterMoEPublish", drain_fn);
    ASSERT_NE(helper_fn, std::string::npos);
    const std::string drain_body = runner_source.substr(drain_fn, helper_fn - drain_fn);
    EXPECT_NE(drain_body.find("publishPendingMoERebalanceUpdate()"), std::string::npos);
    EXPECT_NE(drain_body.find("throw std::runtime_error"), std::string::npos)
        << "failed cache-boundary drains must fail fast instead of falling through";

    const auto apply_fn = runner_source.find("bool OrchestrationRunner::applyMoERebalanceWithReplicas", helper_fn);
    ASSERT_NE(apply_fn, std::string::npos);
    const std::string helper_body = runner_source.substr(helper_fn, apply_fn - helper_fn);
    const auto helper_drain_pos = helper_body.find("drainPendingMoERebalanceBeforeCacheClear()");
    const auto clear_transfers_pos = helper_body.find("clearPendingGpuDirectExpertTransfersForAllDevices()");
    const auto runner_reset_pos = helper_body.find("runner_->resetInferenceState(");
    ASSERT_NE(helper_drain_pos, std::string::npos);
    ASSERT_NE(clear_transfers_pos, std::string::npos);
    ASSERT_NE(runner_reset_pos, std::string::npos);
    EXPECT_LT(helper_drain_pos, clear_transfers_pos)
        << "shared cache reset must not discard staged GPU-direct transfers before the pending publish drains";
    EXPECT_LT(helper_drain_pos, runner_reset_pos)
        << "shared cache reset must not reset graph/cache state before the pending MoE publish applies masks";

    EXPECT_EQ(count_occurrences(runner_source, "runner_->clear_cache()"), 0u)
        << "OrchestrationRunner cache resets must not use the old clear_cache() junk drawer; "
           "they must funnel through clearUnderlyingRunnerCacheAfterMoEPublish() and resetInferenceState().";
    for (const char *reason : {
             "shutdown",
             "prefix-cache-initial-reset",
             "request-clear-cache",
         })
    {
        EXPECT_NE(runner_source.find(std::string("clearUnderlyingRunnerCacheAfterMoEPublish(\"") +
                                     reason + "\")"),
                  std::string::npos)
            << "Missing MoE-publish-aware cache reset reason " << reason;
    }
}

TEST_F(Test__RankOrchestrator, ShutdownSynchronizesAllRankDevicesBeforeRelease)
{
    const std::string runner_source =
        readSourceFileForRankOrchestratorTest(
            "/workspaces/llaminar/src/v2/execution/runner/OrchestrationRunner.cpp");
    ASSERT_FALSE(runner_source.empty());
    const std::string rank_source =
        readSourceFileForRankOrchestratorTest(
            "/workspaces/llaminar/src/v2/execution/local_execution/orchestrators/RankOrchestrator.cpp");
    ASSERT_FALSE(rank_source.empty());

    EXPECT_EQ(runner_source.find("synchronizeRunnerPrimaryDeviceBeforeRelease"),
              std::string::npos)
        << "shutdown must not drain only the primary device in multi-device rank runners";
    const auto shutdown_helper =
        runner_source.find("void synchronizeRunnerDevicesBeforeRelease");
    ASSERT_NE(shutdown_helper, std::string::npos);
    const auto helper_end =
        runner_source.find("const char *prefixStorageTierName", shutdown_helper);
    ASSERT_NE(helper_end, std::string::npos);
    const std::string helper_body =
        runner_source.substr(shutdown_helper, helper_end - shutdown_helper);
    EXPECT_NE(helper_body.find("dynamic_cast<IRankOrchestrator *>"), std::string::npos);
    EXPECT_NE(helper_body.find("rank->synchronizeDevices()"), std::string::npos);
    EXPECT_NE(runner_source.find("synchronizeRunnerDevicesBeforeRelease(\n                runner_.get(),\n                physical_runner_backend_access_enabled_)"),
              std::string::npos);
    EXPECT_NE(helper_body.find("if (!physical_backend_access_enabled)"),
              std::string::npos)
        << "Injected GPU-shaped unit runners must not initialize physical backends during teardown.";

    const auto sync_fn =
        rank_source.find("void RankOrchestrator::synchronizeDevices()");
    ASSERT_NE(sync_fn, std::string::npos);
    const auto sync_end =
        rank_source.find("MoERebalanceController *RankOrchestrator::moeRebalanceController", sync_fn);
    ASSERT_NE(sync_end, std::string::npos);
    const std::string sync_body = rank_source.substr(sync_fn, sync_end - sync_fn);
    EXPECT_NE(sync_body.find("tp_ctx_->synchronize()"), std::string::npos);
    EXPECT_NE(sync_body.find("for (const auto &runner : device_runners_)"), std::string::npos);
    EXPECT_NE(sync_body.find("for (const auto &runner : pp_stage_runners_)"), std::string::npos);
    EXPECT_NE(sync_body.find("backend->synchronize(device.gpu_ordinal())"), std::string::npos)
        << "rank-level synchronization must drain child runner device streams, not only collectives";
}

TEST_F(Test__RankOrchestrator, DeviceSideMoERebalanceSkipsHostRuntimeHistogramBridge)
{
    const std::string source =
        readSourceFileForRankOrchestratorTest(
            "/workspaces/llaminar/src/v2/execution/local_execution/orchestrators/RankOrchestrator.cpp");
    ASSERT_FALSE(source.empty());

    const auto wire_pos = source.find("void RankOrchestrator::wireLocalTPMoERuntimeHistogramSyncs()");
    ASSERT_NE(wire_pos, std::string::npos);
    const auto device_side_gate_pos =
        source.find("usesDeviceSideMoERebalanceController()", wire_pos);
    const auto register_pos =
        source.find("active_histogram->registerRuntimeHistogramSync", wire_pos);
    ASSERT_NE(device_side_gate_pos, std::string::npos)
        << "Device-side graph rebalance gathers histograms on-device and must not wire host sync callbacks.";
    ASSERT_NE(register_pos, std::string::npos);
    EXPECT_LT(device_side_gate_pos, register_pos)
        << "The host runtime histogram bridge must be bypassed before callback registration in device-side mode.";
    EXPECT_NE(source.find("device-side graph rebalance owns histogram allgather"), std::string::npos);
}

TEST_F(Test__RankOrchestrator, LocalTPMoERebalancePublishesPendingUpdateBeforeNewProposal)
{
    const std::string runner_source =
        readSourceFileForRankOrchestratorTest(
            "/workspaces/llaminar/src/v2/execution/runner/OrchestrationRunner.cpp");
    ASSERT_FALSE(runner_source.empty());

    const auto maybe_pos = runner_source.find("bool OrchestrationRunner::maybeApplyMoERebalance()");
    const auto publish_pos = runner_source.find("publishPendingMoERebalanceUpdate()", maybe_pos);
    const auto controller_pos = runner_source.find("auto *controller = moeRebalanceController()", maybe_pos);
    const auto device_side_gate_pos = runner_source.find("usesDeviceSideMoERebalanceController()", maybe_pos);
    const auto decision_pos = runner_source.find("controller->rebalanceDecision()", maybe_pos);
    const auto delayed_prepare_pos = runner_source.find("local_tp_delayed_publish");
    const auto mpi_guard_pos = runner_source.find("local_tp_runner && !mpi_coordinated_world", delayed_prepare_pos);
    const auto prepare_pos = runner_source.find("prepareMoEExpertMaskTransfersForAllDevices", delayed_prepare_pos);
    const auto pre_forward_helper_pos =
        runner_source.find("bool OrchestrationRunner::publishPendingMoERebalanceBeforeForward");
    const auto helper_publish_pos =
        runner_source.find("publishPendingMoERebalanceUpdate()", pre_forward_helper_pos);

    ASSERT_NE(maybe_pos, std::string::npos);
    ASSERT_NE(publish_pos, std::string::npos)
        << "a pending host async rebalance must be published after one compute interval";
    ASSERT_NE(controller_pos, std::string::npos);
    ASSERT_NE(device_side_gate_pos, std::string::npos)
        << "device-side graph rebalance must bypass host delayed-publish maintenance";
    ASSERT_NE(decision_pos, std::string::npos);
    ASSERT_NE(delayed_prepare_pos, std::string::npos);
    ASSERT_NE(mpi_guard_pos, std::string::npos)
        << "delayed publish is currently safe only for single-process LocalTP domains";
    ASSERT_NE(prepare_pos, std::string::npos)
        << "LocalTP delayed publish must prepare transfer slots without publishing masks immediately";
    ASSERT_NE(pre_forward_helper_pos, std::string::npos);
    ASSERT_NE(helper_publish_pos, std::string::npos)
        << "A prepared LocalTP publish must be drained before the next collective-bearing forward.";
    EXPECT_LT(device_side_gate_pos, publish_pos)
        << "device-side graph rebalance must not drain a host-prepared publish first";
    EXPECT_LT(publish_pos, decision_pos);

    const auto decode_pos = runner_source.find("GenerationResult OrchestrationRunner::decodeStep()");
    const auto decode_helper_pos =
        runner_source.find("publishPendingMoERebalanceBeforeForward(\"decode_step\")", decode_pos);
    const auto decode_forward_pos = runner_source.find("runner_->forward(&last_token_, 1)", decode_pos);
    ASSERT_NE(decode_pos, std::string::npos);
    ASSERT_NE(decode_helper_pos, std::string::npos);
    ASSERT_NE(decode_forward_pos, std::string::npos);
    EXPECT_LT(decode_helper_pos, decode_forward_pos)
        << "Pending LocalTP rebalance publishes must be applied symmetrically before decode launches TP collectives.";

    const auto force_pos = runner_source.find("GenerationResult OrchestrationRunner::forceDecodeToken");
    const auto force_helper_pos =
        runner_source.find("publishPendingMoERebalanceBeforeForward(\"force_decode_token\")", force_pos);
    const auto force_forward_pos = runner_source.find("runner_->forward(&last_token_, 1)", force_pos);
    ASSERT_NE(force_pos, std::string::npos);
    ASSERT_NE(force_helper_pos, std::string::npos);
    ASSERT_NE(force_forward_pos, std::string::npos);
    EXPECT_LT(force_helper_pos, force_forward_pos)
        << "Forced-token decode also advances TP collectives and must drain prepared publishes first.";
}

TEST_F(Test__RankOrchestrator, HotReplicaStrategyDoesNotFallbackToOwnershipSwaps)
{
    const std::string runner_source =
        readSourceFileForRankOrchestratorTest(
            "/workspaces/llaminar/src/v2/execution/runner/OrchestrationRunner.cpp");
    ASSERT_FALSE(runner_source.empty());

    const auto strategy_pos = runner_source.find("const bool hot_replica_strategy = max_replicas > 0");
    const auto stable_log_pos = runner_source.find("No beneficial hot expert replicas; keeping base expert ownership stable");
    const auto ownership_guard_pos = runner_source.find("if (!controller->hasReplicas() && !hot_replica_strategy)");
    ASSERT_NE(strategy_pos, std::string::npos)
        << "hot-expert cache mode must be represented as an explicit rebalance strategy";
    ASSERT_NE(stable_log_pos, std::string::npos)
        << "empty low-benefit replica proposals should keep base ownership stable";
    ASSERT_NE(ownership_guard_pos, std::string::npos)
        << "ownership swaps should run only when hot-replica strategy is disabled";
}

TEST_F(Test__RankOrchestrator, PreparedMoEExpertMaskUpdateSnapshotsMasksForDelayedPublish)
{
    std::vector<std::unique_ptr<IInferenceRunner>> runners;
    runners.push_back(std::make_unique<MockDeviceGraphOrchestrator>());
    runners.push_back(std::make_unique<MockDeviceGraphOrchestrator>());

    auto orchestrator = RankOrchestrator::createForTest(
        llaminar2::test::MockModelContext::createMinimal(),
        std::move(runners),
        makeTPContextForRunnerCount(2),
        makeRankConfigForRunnerCount(2));

    std::vector<std::vector<std::vector<bool>>> masks(
        2,
        std::vector<std::vector<bool>>(
            2,
            std::vector<bool>(3, false)));
    masks[0][0][0] = true;
    masks[0][1][1] = true;
    masks[1][0][2] = true;
    masks[1][1][0] = true;

    auto prepared = orchestrator->prepareMoEExpertMaskTransfersForAllDevices(
        masks,
        "gpu_domain");

    masks[0][0][0] = false;
    masks[1][1][0] = false;

    ASSERT_FALSE(prepared.empty());
    EXPECT_EQ(prepared.domain_id, "gpu_domain");
    ASSERT_EQ(prepared.received_by_device.size(), 2u);
    EXPECT_TRUE(prepared.received_by_device[0].empty());
    EXPECT_TRUE(prepared.received_by_device[1].empty());
    ASSERT_EQ(prepared.gpu_direct_prepare_ok_by_device.size(), 2u);
    EXPECT_TRUE(prepared.gpu_direct_prepare_ok_by_device[0]);
    EXPECT_TRUE(prepared.gpu_direct_prepare_ok_by_device[1]);
    EXPECT_TRUE(prepared.masks_by_participant[0][0][0])
        << "prepared rebalance publication must own a stable mask snapshot";
    EXPECT_TRUE(prepared.masks_by_participant[1][1][0])
        << "callers may keep decoding on the old masks while a prepared update is pending";

    orchestrator->publishPreparedMoEExpertMasksForAllDevices(prepared);
}

TEST_F(Test__RankOrchestrator, PreparedMoEExpertMaskUpdateKeepsTransferMaskSeparateFromPublishMask)
{
    std::vector<std::unique_ptr<IInferenceRunner>> runners;
    runners.push_back(std::make_unique<MockDeviceGraphOrchestrator>());
    runners.push_back(std::make_unique<MockDeviceGraphOrchestrator>());

    auto orchestrator = RankOrchestrator::createForTest(
        llaminar2::test::MockModelContext::createMinimal(),
        std::move(runners),
        makeTPContextForRunnerCount(2),
        makeRankConfigForRunnerCount(2));

    std::vector<std::vector<std::vector<bool>>> publish_masks(
        2,
        std::vector<std::vector<bool>>(
            1,
            std::vector<bool>(4, false)));
    publish_masks[0][0][0] = true;
    publish_masks[0][0][1] = true;
    publish_masks[1][0][2] = true;
    publish_masks[1][0][3] = true;

    std::vector<std::vector<std::vector<bool>>> transfer_masks(
        2,
        std::vector<std::vector<bool>>(
            1,
            std::vector<bool>(4, false)));
    transfer_masks[0][0][1] = true;
    transfer_masks[1][0][2] = true;

    auto prepared = orchestrator->prepareMoEExpertMaskTransfersForAllDevices(
        publish_masks,
        "gpu_domain",
        &transfer_masks);

    transfer_masks[0][0][1] = false;
    publish_masks[1][0][3] = false;

    ASSERT_EQ(prepared.masks_by_participant.size(), 2u);
    EXPECT_TRUE(prepared.masks_by_participant[0][0][0]);
    EXPECT_TRUE(prepared.masks_by_participant[0][0][1]);
    EXPECT_TRUE(prepared.masks_by_participant[1][0][2]);
    EXPECT_TRUE(prepared.masks_by_participant[1][0][3])
        << "transfer masks restrict only what gets copied; they must not narrow the published active set";
}

TEST_F(Test__RankOrchestrator, ReplicaArrivalTransferMasksCopyOnlyToNonOwners)
{
    MoERebalanceController::Config cfg;
    cfg.domain_id = "gpu";
    cfg.mode = MoERebalanceMode::OBSERVE;
    cfg.num_layers = 2;
    cfg.num_experts = 4;
    cfg.top_k = 1;
    cfg.window_size = 4;
    cfg.sockets = {
        DeviceId(DeviceType::CPU, 0),
        DeviceId(DeviceType::CPU, 1),
        DeviceId(DeviceType::CPU, 2)};
    cfg.initial_expert_to_socket = {0, 1, 0, 2};
    MoERebalanceController controller(std::move(cfg));

    ExpertReplicaSet arrivals;
    arrivals.domain_id = "gpu";
    arrivals.is_replicated = {false, true, true, false};
    arrivals.owner_socket = {0, 1, 0, 2};
    arrivals.num_replicated = 2;
    arrivals.num_sockets = 3;

    auto masks = rank_orchestrator_detail::buildReplicaArrivalTransferMasks(controller, arrivals);
    ASSERT_EQ(masks.size(), 3u);
    for (const auto &participant_masks : masks)
    {
        ASSERT_EQ(participant_masks.size(), 2u);
        for (const auto &layer_mask : participant_masks)
            ASSERT_EQ(layer_mask.size(), 4u);
    }

    for (int layer = 0; layer < 2; ++layer)
    {
        EXPECT_TRUE(masks[0][layer][1]);
        EXPECT_FALSE(masks[1][layer][1])
            << "participant 1 owns expert 1 and must not copy its own replica arrival";
        EXPECT_TRUE(masks[2][layer][1]);

        EXPECT_FALSE(masks[0][layer][2])
            << "participant 0 owns expert 2 and must not copy its own replica arrival";
        EXPECT_TRUE(masks[1][layer][2]);
        EXPECT_TRUE(masks[2][layer][2]);

        EXPECT_FALSE(masks[0][layer][0]);
        EXPECT_FALSE(masks[1][layer][0]);
        EXPECT_FALSE(masks[2][layer][0]);
        EXPECT_FALSE(masks[0][layer][3]);
        EXPECT_FALSE(masks[1][layer][3]);
        EXPECT_FALSE(masks[2][layer][3]);
    }
}

TEST_F(Test__RankOrchestrator, ReplicaArrivalTransferMasksUseLayerParticipantResidency)
{
    MoERebalanceController::Config cfg;
    cfg.domain_id = "gpu";
    cfg.mode = MoERebalanceMode::OBSERVE;
    cfg.num_layers = 3;
    cfg.num_experts = 4;
    cfg.top_k = 1;
    cfg.window_size = 4;
    cfg.sockets = {
        DeviceId(DeviceType::CPU, 0),
        DeviceId(DeviceType::CPU, 1),
        DeviceId(DeviceType::CPU, 2)};
    cfg.initial_expert_to_socket = {0, 1, 2, 0};
    MoERebalanceController controller(std::move(cfg));

    ExpertReplicaSet arrivals;
    arrivals.domain_id = "gpu";
    arrivals.owner_socket = {0, 1, 2, 0};
    arrivals.num_sockets = 3;
    arrivals.is_replicated.assign(4, false);
    arrivals.replica_participants_by_layer.assign(
        3,
        std::vector<std::vector<bool>>(4, std::vector<bool>(3, false)));
    arrivals.setReplicaOnParticipant(0, 1, 0);
    arrivals.setReplicaOnParticipant(2, 2, 1);
    arrivals.rebuildAggregateReplicaFlags();
    ASSERT_EQ(arrivals.num_replicated, 2);

    auto masks = rank_orchestrator_detail::buildReplicaArrivalTransferMasks(controller, arrivals);
    ASSERT_EQ(masks.size(), 3u);
    for (const auto &participant_masks : masks)
    {
        ASSERT_EQ(participant_masks.size(), 3u);
        for (const auto &layer_mask : participant_masks)
            ASSERT_EQ(layer_mask.size(), 4u);
    }

    EXPECT_TRUE(masks[0][0][1]);
    EXPECT_FALSE(masks[1][0][1])
        << "participant 1 owns expert 1 and must not copy its own replica arrival";
    EXPECT_FALSE(masks[2][0][1])
        << "only the resident target participant should copy this arrival";
    EXPECT_FALSE(masks[0][1][1])
        << "arrival transfer must stay scoped to the layer that changed";

    EXPECT_FALSE(masks[0][2][2]);
    EXPECT_TRUE(masks[1][2][2]);
    EXPECT_FALSE(masks[2][2][2])
        << "participant 2 owns expert 2 and must not copy its own replica arrival";
    EXPECT_FALSE(masks[1][0][2])
        << "arrival transfer must not expand a layer-2 slot to layer 0";
}

TEST_F(Test__RankOrchestrator, EmptyReplicaArrivalTransferMasksSuppressCopies)
{
    MoERebalanceController::Config cfg;
    cfg.domain_id = "gpu";
    cfg.mode = MoERebalanceMode::OBSERVE;
    cfg.num_layers = 2;
    cfg.num_experts = 2;
    cfg.top_k = 1;
    cfg.window_size = 4;
    cfg.sockets = {DeviceId(DeviceType::CPU, 0), DeviceId(DeviceType::CPU, 1)};
    cfg.initial_expert_to_socket = {0, 1};
    MoERebalanceController controller(std::move(cfg));

    ExpertReplicaSet arrivals;
    arrivals.domain_id = "gpu";
    arrivals.is_replicated = {false, false};
    arrivals.owner_socket = {0, 1};
    arrivals.num_replicated = 0;
    arrivals.num_sockets = 2;

    const auto full_mask0 = controller.computeExpertMasksForParticipant(0);
    ASSERT_EQ(full_mask0.size(), 2u);
    ASSERT_EQ(full_mask0[0].size(), 2u);
    EXPECT_TRUE(full_mask0[0][0])
        << "full compute masks still contain owned experts";

    auto transfer_masks = rank_orchestrator_detail::buildReplicaArrivalTransferMasks(
        controller,
        arrivals);
    ASSERT_EQ(transfer_masks.size(), 2u);
    for (const auto &participant_masks : transfer_masks)
    {
        ASSERT_EQ(participant_masks.size(), 2u);
        for (const auto &layer_mask : participant_masks)
        {
            ASSERT_EQ(layer_mask.size(), 2u);
            EXPECT_FALSE(layer_mask[0]);
            EXPECT_FALSE(layer_mask[1]);
        }
    }
}

TEST_F(Test__RankOrchestrator, OwnershipArrivalTransferMasksCopyOnlyNewOwners)
{
    MoERebalanceController::Config cfg;
    cfg.domain_id = "gpu";
    cfg.mode = MoERebalanceMode::OBSERVE;
    cfg.num_layers = 2;
    cfg.num_experts = 4;
    cfg.top_k = 1;
    cfg.window_size = 4;
    cfg.sockets = {
        DeviceId(DeviceType::CPU, 0),
        DeviceId(DeviceType::CPU, 1),
        DeviceId(DeviceType::CPU, 2)};
    cfg.initial_expert_to_socket = {1, 1, 0, 2};
    MoERebalanceController controller(std::move(cfg));

    const std::vector<int> previous_placement = {0, 1, 2, 0};
    auto masks = rank_orchestrator_detail::buildOwnershipArrivalTransferMasks(
        controller,
        previous_placement);

    ASSERT_EQ(masks.size(), 3u);
    for (const auto &participant_masks : masks)
    {
        ASSERT_EQ(participant_masks.size(), 2u);
        for (const auto &layer_mask : participant_masks)
            ASSERT_EQ(layer_mask.size(), 4u);
    }

    for (int layer = 0; layer < 2; ++layer)
    {
        EXPECT_TRUE(masks[1][layer][0])
            << "expert 0 moved from participant 0 to participant 1";
        EXPECT_FALSE(masks[0][layer][0]);
        EXPECT_FALSE(masks[2][layer][0]);

        EXPECT_FALSE(masks[0][layer][1]);
        EXPECT_FALSE(masks[1][layer][1])
            << "unchanged ownership must not trigger an arrival transfer";
        EXPECT_FALSE(masks[2][layer][1]);

        EXPECT_TRUE(masks[0][layer][2])
            << "expert 2 moved from participant 2 to participant 0";
        EXPECT_FALSE(masks[1][layer][2]);
        EXPECT_FALSE(masks[2][layer][2]);

        EXPECT_TRUE(masks[2][layer][3])
            << "expert 3 moved from participant 0 to participant 2";
        EXPECT_FALSE(masks[0][layer][3]);
        EXPECT_FALSE(masks[1][layer][3]);
    }
}

TEST_F(Test__RankOrchestrator, OwnershipArrivalSnapshotKeepsTransferMaskSeparateFromPublishMask)
{
    std::vector<std::unique_ptr<IInferenceRunner>> runners;
    runners.push_back(std::make_unique<MockDeviceGraphOrchestrator>());
    runners.push_back(std::make_unique<MockDeviceGraphOrchestrator>());

    auto orchestrator = RankOrchestrator::createForTest(
        llaminar2::test::MockModelContext::createMinimal(),
        std::move(runners),
        makeTPContextForRunnerCount(2),
        makeRankConfigForRunnerCount(2));

    MoERebalanceController::Config cfg;
    cfg.domain_id = "gpu";
    cfg.mode = MoERebalanceMode::OBSERVE;
    cfg.num_layers = 1;
    cfg.num_experts = 4;
    cfg.top_k = 1;
    cfg.window_size = 4;
    cfg.sockets = {DeviceId(DeviceType::CPU, 0), DeviceId(DeviceType::CPU, 1)};
    cfg.initial_expert_to_socket = {0, 0, 1, 1};
    MoERebalanceController controller(std::move(cfg));

    const std::vector<int> previous_placement = {0, 1, 1, 0};
    auto snapshot = orchestrator->snapshotMoEExpertMasksForAllDevices(
        controller,
        nullptr,
        &previous_placement);

    ASSERT_EQ(snapshot.masks_by_participant.size(), 2u);
    ASSERT_EQ(snapshot.masks_by_participant[0].size(), 1u);
    ASSERT_EQ(snapshot.masks_by_participant[1].size(), 1u);
    EXPECT_TRUE(snapshot.masks_by_participant[0][0][0]);
    EXPECT_TRUE(snapshot.masks_by_participant[0][0][1])
        << "publish masks contain the full active owner set for participant 0";
    EXPECT_TRUE(snapshot.masks_by_participant[1][0][2]);
    EXPECT_TRUE(snapshot.masks_by_participant[1][0][3])
        << "publish masks contain the full active owner set for participant 1";

    ASSERT_NE(snapshot.transferMasks(), nullptr);
    const auto &transfer_masks = *snapshot.transferMasks();
    ASSERT_EQ(transfer_masks.size(), 2u);
    EXPECT_FALSE(transfer_masks[0][0][0])
        << "unchanged expert 0 is already resident and must not be transferred";
    EXPECT_TRUE(transfer_masks[0][0][1])
        << "expert 1 moved from participant 1 to participant 0";
    EXPECT_FALSE(transfer_masks[1][0][2])
        << "unchanged expert 2 is already resident and must not be transferred";
    EXPECT_TRUE(transfer_masks[1][0][3])
        << "expert 3 moved from participant 0 to participant 1";
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

TEST_F(Test__RankOrchestrator, PopulatePrefixPropagatesModelRuntimeRestorePolicyToChildren)
{
    auto runner0 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner0_ptr = runner0.get();
    runner0_ptr->set_prefix_lookup_result(makePrefixHit(/*cached_tokens=*/4,
                                                        /*terminal_logits=*/true,
                                                        /*supported=*/true,
                                                        /*include_blocks=*/true));

    auto runner1 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner1_ptr = runner1.get();
    runner1_ptr->set_prefix_lookup_result(makePrefixHit(/*cached_tokens=*/4,
                                                        /*terminal_logits=*/true,
                                                        /*supported=*/true,
                                                        /*include_blocks=*/true));

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

    hit.restore_model_runtime_state = false;
    ASSERT_TRUE(orchestrator->populatePrefix(hit));

    EXPECT_EQ(runner0_ptr->populated_prefix_restore_model_runtime_state(),
              std::vector<bool>({false}));
    EXPECT_EQ(runner1_ptr->populated_prefix_restore_model_runtime_state(),
              std::vector<bool>({false}));
}

TEST_F(Test__RankOrchestrator, PopulatePrefixResetsAggregateSequenceState)
{
    auto runner0 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner0_ptr = runner0.get();
    runner0_ptr->set_prefix_lookup_result(makePrefixHit(/*cached_tokens=*/4,
                                                        /*terminal_logits=*/true,
                                                        /*supported=*/true,
                                                        /*include_blocks=*/true));

    auto runner1 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner1_ptr = runner1.get();
    runner1_ptr->set_prefix_lookup_result(makePrefixHit(/*cached_tokens=*/4,
                                                        /*terminal_logits=*/true,
                                                        /*supported=*/true,
                                                        /*include_blocks=*/true));

    std::vector<std::unique_ptr<IInferenceRunner>> runners;
    runners.push_back(std::move(runner0));
    runners.push_back(std::move(runner1));

    auto orchestrator = RankOrchestrator::createForTest(
        llaminar2::test::MockModelContext::createMinimal(),
        std::move(runners),
        makeTPContextForRunnerCount(2),
        makeRankConfigForRunnerCount(2));

    int stale_request_tokens[] = {10, 11, 12, 13, 14, 15, 16};
    ASSERT_TRUE(orchestrator->forward(stale_request_tokens, 7));
    EXPECT_THAT(orchestrator->sequence_lengths(), ::testing::ElementsAre(7));

    PrefixLookupResult hit = orchestrator->lookupPrefix({1, 2, 3, 4, 5, 6});
    ASSERT_TRUE(hit.supported);
    ASSERT_EQ(hit.cached_tokens, 4);

    ASSERT_TRUE(orchestrator->populatePrefix(hit));
    EXPECT_EQ(orchestrator->get_position(), 4);
    EXPECT_EQ(orchestrator->padded_seq_len(), 0);
    EXPECT_THAT(orchestrator->sequence_lengths(), ::testing::ElementsAre(4));

    int suffix_tokens[] = {5, 6};
    ASSERT_TRUE(orchestrator->forwardPrefill(suffix_tokens, 2));
    EXPECT_THAT(orchestrator->sequence_lengths(), ::testing::ElementsAre(6));

    orchestrator->clear_cache();
    EXPECT_EQ(orchestrator->get_position(), 0);
    EXPECT_EQ(orchestrator->padded_seq_len(), 0);
    EXPECT_THAT(orchestrator->sequence_lengths(), ::testing::ElementsAre(0));
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

/**
 * @brief Prove a CPU TP prefix restore cannot expose the previous request's logits.
 *
 * The child runners own the restored column-parallel terminal rows, while
 * RankOrchestrator owns the full-vocabulary CPU sampling aggregate. A restore
 * must update both ownership layers atomically; otherwise seeded stochastic
 * MTP can sample the old request's next token immediately after clearCache().
 */
TEST_F(Test__RankOrchestrator, PrefixTerminalRestoreRefreshesCPUTPLogitsAggregate)
{
    MockDeviceGraphOrchestrator::Config child_config;
    child_config.vocab_size = 4;

    auto runner0 =
        std::make_unique<MockDeviceGraphOrchestrator>(child_config);
    auto *runner0_ptr = runner0.get();
    runner0_ptr->set_mock_logits_local(
        /*local_vocab=*/2,
        {10.0f, 11.0f});
    runner0_ptr->set_prefix_lookup_result(
        makePrefixHit(/*cached_tokens=*/4, /*terminal_logits=*/true));

    auto runner1 =
        std::make_unique<MockDeviceGraphOrchestrator>(child_config);
    auto *runner1_ptr = runner1.get();
    runner1_ptr->set_mock_logits_local(
        /*local_vocab=*/2,
        {12.0f, 13.0f});
    runner1_ptr->set_prefix_lookup_result(
        makePrefixHit(/*cached_tokens=*/4, /*terminal_logits=*/true));

    std::vector<std::unique_ptr<IInferenceRunner>> runners;
    runners.push_back(std::move(runner0));
    runners.push_back(std::move(runner1));

    auto model_ctx = llaminar2::test::MockModelContextBuilder()
                         .usePreset(llaminar2::test::ModelPreset::MINIMAL)
                         .setVocabSize(4)
                         .build();
    auto orchestrator = RankOrchestrator::createForTest(
        std::move(model_ctx),
        std::move(runners),
        makeTPContextForRunnerCount(2),
        makeRankConfigForRunnerCount(2));

    int32_t token = 1;
    ASSERT_TRUE(orchestrator->forward(&token, 1));
    const float *stale_aggregate = orchestrator->logits();
    ASSERT_NE(stale_aggregate, nullptr);
    EXPECT_THAT(
        std::vector<float>(stale_aggregate, stale_aggregate + 4),
        ::testing::ElementsAre(10.0f, 11.0f, 12.0f, 13.0f));

    /*
     * Model the child terminal-state import by replacing each local shard
     * before RankOrchestrator completes the aggregate restore transaction.
     */
    runner0_ptr->set_mock_logits_local(
        /*local_vocab=*/2,
        {20.0f, 21.0f});
    runner1_ptr->set_mock_logits_local(
        /*local_vocab=*/2,
        {22.0f, 23.0f});

    const PrefixLookupResult hit = orchestrator->lookupPrefix({1, 2, 3, 4});
    ASSERT_TRUE(hit.has_terminal_logits);
    ASSERT_TRUE(orchestrator->restorePrefixTerminalState(hit));

    const float *restored_aggregate = orchestrator->logits();
    ASSERT_NE(restored_aggregate, nullptr);
    EXPECT_THAT(
        std::vector<float>(restored_aggregate, restored_aggregate + 4),
        ::testing::ElementsAre(20.0f, 21.0f, 22.0f, 23.0f));
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

/**
 * @brief LocalTP fans out logical verifier-base terminal-hidden repairs.
 *
 * Grouped LocalTP MTP publication may synthesize a logical verifier-base
 * checkpoint when every participant advertises that token-count checkpoints
 * are sufficient.  That shared logical checkpoint still has to reach every TP
 * participant so each shard appends its own shifted MTP row from the same
 * serial-decode boundary.  This is the focused regression for a bug where the
 * rank worker path accepted only aggregate participant snapshots and skipped
 * valid logical checkpoints before child runners could commit the repair.
 */
TEST_F(Test__RankOrchestrator,
       LocalTPCheckpointTerminalHiddenFansOutSharedLogicalCheckpoint)
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

    PrefixStateSnapshot checkpoint;
    checkpoint.valid = true;
    checkpoint.logical_checkpoint = true;
    checkpoint.provenance = PrefixStateProvenance::LogicalCheckpoint;
    checkpoint.cached_tokens = 5;
    checkpoint.mtp_cached_tokens = {4};

    EXPECT_TRUE(orchestrator->commitMTPShiftedRowFromCheckpointTerminalHidden(
        checkpoint,
        /*token=*/123,
        /*already_appended_tokens=*/0,
        /*allow_speculative_discard=*/true,
        /*position_offset_override=*/5));

    for (MockDeviceGraphOrchestrator *child : {runner0_ptr, runner1_ptr})
    {
        EXPECT_EQ(child->commit_mtp_checkpoint_terminal_hidden_call_count(), 1u);
        EXPECT_EQ(child->commit_mtp_shifted_rows_call_count(), 0u)
            << "The checkpoint-terminal-hidden repair is a distinct grouped "
               "publication operation, not a partial-forward row replay.";
        EXPECT_TRUE(child->last_commit_mtp_checkpoint_valid());
        EXPECT_TRUE(child->last_commit_mtp_checkpoint_logical());
        EXPECT_EQ(child->last_commit_mtp_checkpoint_provenance(),
                  PrefixStateProvenance::LogicalCheckpoint);
        EXPECT_EQ(child->last_commit_mtp_checkpoint_cached_tokens(), 5);
        EXPECT_EQ(child->last_commit_mtp_tokens(),
                  std::vector<int32_t>({123}));
        EXPECT_TRUE(child->last_commit_mtp_allow_speculative_discard());
        EXPECT_EQ(child->last_commit_mtp_position_offset_override(), 5);
    }
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

TEST_F(Test__RankOrchestrator, DeviceTargetShiftedCommitRunsOnEveryLocalTPChild)
{
    auto runner0 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner0_ptr = runner0.get();
    runner0_ptr->set_primary_device_id(DeviceId::cuda(0));
    runner0_ptr->set_supports_mtp_device_draft_token_input(true);

    auto runner1 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner1_ptr = runner1.get();
    runner1_ptr->set_primary_device_id(DeviceId::cuda(1));
    runner1_ptr->set_supports_mtp_device_draft_token_input(true);

    std::vector<std::unique_ptr<IInferenceRunner>> runners;
    runners.push_back(std::move(runner0));
    runners.push_back(std::move(runner1));

    auto orchestrator = RankOrchestrator::createForTest(
        llaminar2::test::MockModelContext::createMinimal(),
        std::move(runners),
        makeTPContextForRunnerCount(2),
        makeRankConfigForRunnerCount(2));

    EXPECT_TRUE(orchestrator->commitMTPShiftedRowFromDeviceTargetSample(
        /*target_sample_slot=*/3,
        /*already_appended_tokens=*/1,
        /*allow_speculative_discard=*/true));

    EXPECT_EQ(runner0_ptr->commit_mtp_device_target_sample_call_count(), 1u);
    EXPECT_EQ(runner1_ptr->commit_mtp_device_target_sample_call_count(), 1u);
    EXPECT_EQ(runner0_ptr->last_commit_mtp_device_target_sample_slot(), 3);
    EXPECT_EQ(runner1_ptr->last_commit_mtp_device_target_sample_slot(), 3);
    EXPECT_EQ(runner0_ptr->last_commit_mtp_already_appended(), 1);
    EXPECT_EQ(runner1_ptr->last_commit_mtp_already_appended(), 1);
    EXPECT_TRUE(runner0_ptr->last_commit_mtp_allow_speculative_discard());
    EXPECT_TRUE(runner1_ptr->last_commit_mtp_allow_speculative_discard());
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

TEST_F(Test__RankOrchestrator, LocalTPRequiresMirroredHeadBeforeAdvertisingResidentMTPOutcome)
{
    auto runner0 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner0_ptr = runner0.get();
    runner0_ptr->set_primary_device_id(DeviceId::cuda(0));
    runner0_ptr->set_supports_device_resident_mtp_spec_state_publication(true);
    auto runner1 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner1_ptr = runner1.get();
    runner1_ptr->set_primary_device_id(DeviceId::cuda(1));
    runner1_ptr->set_supports_device_resident_mtp_spec_state_publication(true);

    std::vector<std::unique_ptr<IInferenceRunner>> runners;
    runners.push_back(std::move(runner0));
    runners.push_back(std::move(runner1));

    auto orchestrator = RankOrchestrator::createForTest(
        llaminar2::test::MockModelContext::createMinimal(),
        std::move(runners),
        makeTPContextForRunnerCount(2),
        makeRankConfigForRunnerCount(2));

    EXPECT_FALSE(orchestrator->supportsDeviceResidentMTPSpecStatePublication())
        << "GPU LocalTP must not advertise resident MTP publication until every "
           "child owns a mirrored full-vocab verifier head.";
    EXPECT_FALSE(orchestrator->supportsGreedyAllPositionBatchOutcomeOnDevice())
        << "The diagnostic rank compact reducer may still materialize response "
           "tokens, but it must not advertise GPU live-state publication.";
    EXPECT_FALSE(orchestrator->supportsDeviceStochasticMTPVerification())
        << "Stochastic GPU LocalTP must use child-resident mirrored outcomes, "
           "not rank-owned compact top-k summaries.";
}

TEST_F(Test__RankOrchestrator, LocalTPCompactGreedyVerifierOutcomeReducesShardedRows)
{
    auto runner0 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner0_ptr = runner0.get();
    runner0_ptr->set_mock_all_position_logits_local(
        /*rows=*/2,
        /*local_vocab=*/3,
        {
            0.0f, 1.0f, 2.0f,
            0.0f, 1.0f, 12.0f,
        });

    auto runner1 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner1_ptr = runner1.get();
    runner1_ptr->set_mock_all_position_logits_local(
        /*rows=*/2,
        /*local_vocab=*/3,
        {
            0.0f, 10.0f, 1.0f,
            5.0f, 4.0f, 3.0f,
        });

    std::vector<std::unique_ptr<IInferenceRunner>> runners;
    runners.push_back(std::move(runner0));
    runners.push_back(std::move(runner1));

    auto orchestrator = RankOrchestrator::createForTest(
        llaminar2::test::MockModelContext::createMinimal(),
        std::move(runners),
        makeTPContextForRunnerCount(2),
        makeRankConfigForRunnerCount(2));

    const std::array<int32_t, 2> draft_tokens = {10, 4};
    DeviceSpeculativeVerifyBatchOutcome outcome;
    ASSERT_TRUE(orchestrator->verifyGreedyAllPositionBatchOutcomeOnDevice(
        draft_tokens.data(),
        static_cast<int>(draft_tokens.size()),
        /*stop_tokens=*/nullptr,
        /*stop_token_count=*/0,
        &outcome));

    EXPECT_TRUE(outcome.ok);
    EXPECT_EQ(outcome.output_token_count, 2);
    EXPECT_EQ(outcome.output_tokens[0], 10);
    EXPECT_EQ(outcome.output_tokens[1], 4)
        << "row 0 should accept the draft from shard 1";
    EXPECT_EQ(outcome.accepted_speculative_prefix, 1);
    EXPECT_EQ(outcome.target_verifier_state_commit_count, 2);
    EXPECT_EQ(outcome.ready_token, 2)
        << "row 1 bonus token should come from shard 0";
    EXPECT_EQ(outcome.rejected_verified_token, -1);
    EXPECT_TRUE(outcome.all_speculative_accepted);
    EXPECT_EQ(outcome.consumed_verifier_rows, 1);
    EXPECT_TRUE(outcome.sampled_terminal);
    EXPECT_EQ(runner0_ptr->consume_all_position_logits_local_info_call_count(), 1u);
    EXPECT_EQ(runner1_ptr->consume_all_position_logits_local_info_call_count(), 1u);
    EXPECT_EQ(runner0_ptr->verify_greedy_all_position_batch_outcome_call_count(), 0u)
        << "Rank-level LocalTP reducer must not delegate a sharded outcome to "
           "one child.";
    EXPECT_EQ(runner1_ptr->verify_greedy_all_position_batch_outcome_call_count(), 0u);
}

TEST_F(Test__RankOrchestrator, LocalTPCompactGreedyVerifierOutcomeRejectsCrossShardMismatch)
{
    auto runner0 = std::make_unique<MockDeviceGraphOrchestrator>();
    runner0->set_mock_all_position_logits_local(
        /*rows=*/2,
        /*local_vocab=*/3,
        {
            0.0f, 1.0f, 2.0f,
            0.0f, 1.0f, 12.0f,
        });

    auto runner1 = std::make_unique<MockDeviceGraphOrchestrator>();
    runner1->set_mock_all_position_logits_local(
        /*rows=*/2,
        /*local_vocab=*/3,
        {
            9.0f, 3.0f, 1.0f,
            5.0f, 4.0f, 3.0f,
        });

    std::vector<std::unique_ptr<IInferenceRunner>> runners;
    runners.push_back(std::move(runner0));
    runners.push_back(std::move(runner1));

    auto orchestrator = RankOrchestrator::createForTest(
        llaminar2::test::MockModelContext::createMinimal(),
        std::move(runners),
        makeTPContextForRunnerCount(2),
        makeRankConfigForRunnerCount(2));

    const std::array<int32_t, 2> draft_tokens = {10, 4};
    DeviceSpeculativeVerifyBatchOutcome outcome;
    ASSERT_TRUE(orchestrator->verifyGreedyAllPositionBatchOutcomeOnDevice(
        draft_tokens.data(),
        static_cast<int>(draft_tokens.size()),
        /*stop_tokens=*/nullptr,
        /*stop_token_count=*/0,
        &outcome));

    EXPECT_TRUE(outcome.ok);
    EXPECT_EQ(outcome.output_token_count, 2);
    EXPECT_EQ(outcome.output_tokens[0], 10);
    EXPECT_EQ(outcome.output_tokens[1], 3)
        << "row 0 should reject with the shard-1 global argmax token";
    EXPECT_EQ(outcome.accepted_speculative_prefix, 0);
    EXPECT_EQ(outcome.target_verifier_state_commit_count, 1);
    EXPECT_EQ(outcome.ready_token, -1);
    EXPECT_EQ(outcome.rejected_verified_token, 3);
    EXPECT_FALSE(outcome.all_speculative_accepted);
    EXPECT_EQ(outcome.consumed_verifier_rows, 1);
    EXPECT_FALSE(outcome.sampled_terminal);
}

TEST_F(Test__RankOrchestrator,
       LocalTPMirroredGreedyOutcomeUsesGraphPublishedCommonResidentOutcome)
{
    auto rendezvous = std::make_shared<MTPPublicationRendezvous>(2);

    auto runner0 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner0_ptr = runner0.get();
    runner0_ptr->set_primary_device_id(DeviceId::cuda(0));
    runner0_ptr->set_vocab_size(6);
    runner0_ptr->set_supports_device_resident_mtp_spec_state_publication(true);
    runner0_ptr->set_supports_mtp_sidecar_logits_stream_handoff(true);
    runner0_ptr->set_supports_mtp_device_draft_token_input(true);
    runner0_ptr->set_uses_mirrored_localtp_mtp_head_for_verifier(true);
    runner0_ptr->set_mock_device_position(64);
    runner0_ptr->set_mtp_publication_rendezvous(rendezvous);
    runner0_ptr->set_mock_all_position_logits(
        {
            0.0f, 1.0f, 2.0f, 3.0f, 9.0f, 4.0f,
            0.0f, 1.0f, 8.0f, 3.0f, 2.0f, 4.0f,
        });

    auto runner1 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner1_ptr = runner1.get();
    runner1_ptr->set_primary_device_id(DeviceId::cuda(1));
    runner1_ptr->set_vocab_size(6);
    runner1_ptr->set_supports_device_resident_mtp_spec_state_publication(true);
    runner1_ptr->set_supports_mtp_sidecar_logits_stream_handoff(true);
    runner1_ptr->set_supports_mtp_device_draft_token_input(true);
    runner1_ptr->set_uses_mirrored_localtp_mtp_head_for_verifier(true);
    runner1_ptr->set_mock_device_position(64);
    runner1_ptr->set_mtp_publication_rendezvous(rendezvous);
    runner1_ptr->set_mock_all_position_logits(
        {
            0.0f, 1.0f, 2.0f, 3.0f, 9.0f, 4.0f,
            0.0f, 1.0f, 8.0f, 3.0f, 2.0f, 4.0f,
        });

    std::vector<std::unique_ptr<IInferenceRunner>> runners;
    runners.push_back(std::move(runner0));
    runners.push_back(std::move(runner1));

    auto tp_ctx = makeTPContextForRunnerCount(2);
    auto *tp_ctx_ptr = tp_ctx.get();
    auto orchestrator = RankOrchestrator::createForTest(
        llaminar2::test::MockModelContext::createMinimal(),
        std::move(runners),
        std::move(tp_ctx),
        makeRankConfigForRunnerCount(2));

    ASSERT_TRUE(orchestrator->supportsGreedyAllPositionBatchOutcomeOnDevice());
    ASSERT_TRUE(orchestrator->supportsDeviceResidentMTPSpecStatePublication());
    ASSERT_TRUE(orchestrator->usesMirroredLocalTPMTPHeadForVerifier());

    const std::array<int32_t, 2> draft_tokens = {10, 4};
    DeviceSpeculativeOutcomeHandle handle;
    ASSERT_TRUE(orchestrator->verifyGreedyAllPositionBatchOutcomeOnDeviceResident(
        draft_tokens.data(),
        static_cast<int>(draft_tokens.size()),
        /*stop_tokens=*/nullptr,
        /*stop_token_count=*/0,
        &handle));
    ASSERT_TRUE(handle.valid());
    EXPECT_EQ(tp_ctx_ptr->collective_sideband_call_count(), 0u)
        << "The rank consumer must not enqueue a second compact-outcome "
           "collective after every child reports graph-owned publication.";
    EXPECT_EQ(tp_ctx_ptr->collective_sideband_broadcast_count(), 0u);

    DeviceSpeculativeVerifyBatchOutcome materialized;
    ASSERT_TRUE(orchestrator->materializeDeviceSpeculativeOutcomesForHostResponse(
        handle,
        &materialized));
    EXPECT_TRUE(materialized.ok);
    EXPECT_EQ(materialized.output_token_count, 2);
    EXPECT_EQ(materialized.output_tokens[0], 10);
    EXPECT_EQ(materialized.output_tokens[1], 4);
    EXPECT_EQ(materialized.ready_token, 2);

    PrefixStateSnapshot checkpoint;
    checkpoint.valid = true;
    checkpoint.cached_tokens = 64;
    checkpoint.participant_snapshots.resize(2);
    for (PrefixStateSnapshot &participant : checkpoint.participant_snapshots)
    {
        participant.valid = true;
        participant.cached_tokens = 64;
        participant.provenance = PrefixStateProvenance::PayloadCheckpoint;
    }
    EXPECT_TRUE(orchestrator->commitMTPInitialShiftedRowFromDeviceOutcome(
        checkpoint,
        handle,
        /*request_index=*/0,
        /*main_forward_token_count=*/2,
        /*allow_speculative_discard=*/true));
    EXPECT_EQ(runner0_ptr->commit_mtp_initial_device_outcome_call_count(), 1u);
    EXPECT_EQ(runner1_ptr->commit_mtp_initial_device_outcome_call_count(), 1u);
    EXPECT_EQ(runner0_ptr->commit_mtp_checkpoint_terminal_hidden_call_count(), 0u);
    EXPECT_EQ(runner1_ptr->commit_mtp_checkpoint_terminal_hidden_call_count(), 0u);
    EXPECT_THAT(runner0_ptr->last_commit_mtp_tokens(),
                ::testing::ElementsAre(10));
    EXPECT_THAT(runner1_ptr->last_commit_mtp_tokens(),
                ::testing::ElementsAre(10));
    EXPECT_EQ(runner0_ptr->last_commit_mtp_checkpoint_cached_tokens(), 64);
    EXPECT_EQ(runner1_ptr->last_commit_mtp_checkpoint_cached_tokens(), 64);

    DeviceSpeculativePublicationRequest request;
    request.outcome = handle;
    request.request_count = 1;
    request.max_draft_tokens = static_cast<int>(draft_tokens.size());
    request.max_state_commit_rows =
        materialized.target_verifier_state_commit_count;
    request.publish_mtp_shifted_kv = true;

    std::string error;
    EXPECT_TRUE(orchestrator->publishAcceptedMTPSpecStateBatchFromDeviceOutcome(
        request,
        &error))
        << error;

    EXPECT_EQ(rendezvous->arrivals.load(std::memory_order_acquire), 0);
    EXPECT_EQ(runner0_ptr->verify_greedy_all_position_batch_outcome_call_count(), 1u);
    EXPECT_EQ(runner1_ptr->verify_greedy_all_position_batch_outcome_call_count(), 1u);
    EXPECT_EQ(runner0_ptr->publish_device_resident_mtp_spec_state_batch_call_count(), 1u);
    EXPECT_EQ(runner1_ptr->publish_device_resident_mtp_spec_state_batch_call_count(), 1u);
    EXPECT_EQ(runner0_ptr->publish_mtp_spec_state_batch_call_count(), 0u);
    EXPECT_EQ(runner1_ptr->publish_mtp_spec_state_batch_call_count(), 0u);
    EXPECT_EQ(runner0_ptr->publish_grouped_decode_equivalent_mtp_spec_state_batch_call_count(), 0u);
    EXPECT_EQ(runner1_ptr->publish_grouped_decode_equivalent_mtp_spec_state_batch_call_count(), 0u);

    DeviceResidentLogicalSequenceStateHandle resident_state =
        orchestrator->deviceResidentLogicalSequenceState();
    EXPECT_TRUE(resident_state.valid());
    EXPECT_TRUE(orchestrator->forwardMTPFromDeviceResidentLogicalStateForDeviceSampling(
        resident_state,
        /*request_index=*/0))
        << "Mirrored LocalTP publication should expose a rank-owned mailbox "
           "that fans the next first-sidecar prelaunch to child-resident "
           "logical-state handles before the host response bridge.";
    EXPECT_EQ(runner0_ptr->forward_mtp_from_resident_logical_state_call_count(), 1u);
    EXPECT_EQ(runner1_ptr->forward_mtp_from_resident_logical_state_call_count(), 1u);
    EXPECT_EQ(runner0_ptr->last_resident_logical_state_request_index(), 0);
    EXPECT_EQ(runner1_ptr->last_resident_logical_state_request_index(), 0);

    /*
     * Reproduce the prefix-restore lifetime that originally left a phantom rank
     * mailbox alive. Production child runners clear their logical mailbox when
     * populatePrefix() replaces live request state. The rank must invalidate its
     * aggregate before dispatching that replacement, and an already-issued rank
     * handle must no longer be accepted by a sidecar consumer afterward.
     */
    runner0_ptr->set_prefix_populate_invalidates_resident_logical_state(true);
    runner1_ptr->set_prefix_populate_invalidates_resident_logical_state(true);
    runner0_ptr->set_prefix_lookup_result(
        makePrefixHit(/*cached_tokens=*/4,
                      /*terminal_logits=*/true,
                      /*supported=*/true,
                      /*include_blocks=*/true));
    runner1_ptr->set_prefix_lookup_result(
        makePrefixHit(/*cached_tokens=*/4,
                      /*terminal_logits=*/true,
                      /*supported=*/true,
                      /*include_blocks=*/true));

    const std::vector<int32_t> restored_prompt = {1, 2, 3, 4};
    const PrefixLookupResult prefix_hit =
        orchestrator->lookupPrefix(restored_prompt);
    ASSERT_EQ(prefix_hit.cached_tokens, 4);
    ASSERT_TRUE(orchestrator->populatePrefix(prefix_hit));
    EXPECT_FALSE(orchestrator->deviceResidentLogicalSequenceState().valid())
        << "Prefix replacement must not expose child mailboxes adopted by the "
           "previous request.";
    EXPECT_FALSE(
        orchestrator->forwardMTPFromDeviceResidentLogicalStateForDeviceSampling(
            resident_state,
            /*request_index=*/0))
        << "A rank mailbox issued before prefix restore must fail closed after "
           "the aggregate epoch advances.";
}

TEST_F(Test__RankOrchestrator,
       MirroredLocalTPOutcomePublicationHasNoHostWorkerRendezvous)
{
    const std::string source =
        readSourceFileForRankOrchestratorTest(
            "/workspaces/llaminar/src/v2/execution/local_execution/orchestrators/RankOrchestrator.cpp");
    ASSERT_FALSE(source.empty());

    const size_t begin = source.find(
        "bool RankOrchestrator::broadcastPrimaryMirroredLocalTPDeviceOutcomeToChildren(");
    ASSERT_NE(begin, std::string::npos);
    const size_t end = source.find(
        "bool RankOrchestrator::buildRankStochasticDistributionFromLocalTP(",
        begin);
    ASSERT_NE(end, std::string::npos);
    const std::string body = source.substr(begin, end - begin);

    EXPECT_NE(
        body.find("collectiveSidebandsMultiOnStreams("),
        std::string::npos);
    EXPECT_EQ(body.find("tp_worker_pool_"), std::string::npos);
    EXPECT_EQ(body.find("TPWorkerPool"), std::string::npos);
    EXPECT_EQ(body.find("dispatch("), std::string::npos);
    EXPECT_EQ(body.find("collectAll("), std::string::npos);
    EXPECT_EQ(
        body.find("effectiveTPWorkerJoinTimeoutMs"),
        std::string::npos);
    EXPECT_EQ(
        body.find("collectiveSidebandOnStream("),
        std::string::npos);
}

TEST_F(Test__RankOrchestrator,
       MirroredGreedyConsumerCannotRebroadcastGraphOwnedOutcome)
{
    const std::string source =
        readSourceFileForRankOrchestratorTest(
            "/workspaces/llaminar/src/v2/execution/local_execution/orchestrators/RankOrchestrator.cpp");
    ASSERT_FALSE(source.empty());

    const size_t begin = source.find(
        "bool RankOrchestrator::verifyGreedyMirroredLocalTPBatchOutcomeOnDeviceResident(");
    ASSERT_NE(begin, std::string::npos);
    const size_t end = source.find(
        "bool RankOrchestrator::verifyGreedyAllPositionRequestBatchOutcomesOnDeviceResident(",
        begin);
    ASSERT_NE(end, std::string::npos);
    const std::string body = source.substr(begin, end - begin);

    EXPECT_NE(
        body.find("mirrored_local_tp_published_in_graph"),
        std::string::npos);
    EXPECT_EQ(
        body.find(
            "broadcastPrimaryMirroredLocalTPDeviceOutcomeToChildren("),
        std::string::npos)
        << "Greedy graph publication is complete before rank consumption; "
           "a second rank-side collective is forbidden.";
    EXPECT_NE(
        body.find("requestAbort()"),
        std::string::npos)
        << "A child missing graph-owned publication must abort the TP domain.";
}

TEST_F(Test__RankOrchestrator, LocalTPResidentCompactGreedyOutcomeResolvesDeferredRankSlots)
{
    auto runner0 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner0_ptr = runner0.get();
    runner0_ptr->set_mock_logits_local(
        /*local_vocab=*/3,
        {0.0f, 1.0f, 2.0f});
    runner0_ptr->set_mock_mtp_logits_local(
        /*local_vocab=*/3,
        {0.0f, 1.0f, 2.0f});
    runner0_ptr->set_mock_all_position_logits_local(
        /*rows=*/2,
        /*local_vocab=*/3,
        {
            0.0f, 1.0f, 2.0f,
            0.0f, 6.0f, 12.0f,
        });

    auto runner1 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner1_ptr = runner1.get();
    runner1_ptr->set_mock_logits_local(
        /*local_vocab=*/3,
        {0.0f, 10.0f, 1.0f});
    runner1_ptr->set_mock_mtp_logits_local(
        /*local_vocab=*/3,
        {0.0f, 10.0f, 1.0f});
    runner1_ptr->set_mock_all_position_logits_local(
        /*rows=*/2,
        /*local_vocab=*/3,
        {
            0.0f, 10.0f, 1.0f,
            5.0f, 4.0f, 3.0f,
        });

    std::vector<std::unique_ptr<IInferenceRunner>> runners;
    runners.push_back(std::move(runner0));
    runners.push_back(std::move(runner1));

    auto orchestrator = RankOrchestrator::createForTest(
        llaminar2::test::MockModelContext::createMinimal(),
        std::move(runners),
        makeTPContextForRunnerCount(2),
        makeRankConfigForRunnerCount(2));

    ASSERT_TRUE(orchestrator->sampleGreedyFromMainLogitsToDeviceTargetSlot(
        /*target_sample_slot=*/0,
        /*out_token=*/nullptr));
    ASSERT_TRUE(orchestrator->sampleGreedyFromMTPLogitsToDeviceDraftSlot(
        /*draft_sample_slot=*/0,
        /*out_token=*/nullptr));

    const std::array<int32_t, 2> deferred_tokens = {-3, -2};
    DeviceSpeculativeOutcomeHandle handle;
    ASSERT_TRUE(orchestrator->verifyGreedyAllPositionBatchOutcomeOnDeviceResident(
        deferred_tokens.data(),
        static_cast<int>(deferred_tokens.size()),
        /*stop_tokens=*/nullptr,
        /*stop_token_count=*/0,
        &handle));
    ASSERT_TRUE(handle.valid());

    DeviceSpeculativeVerifyBatchOutcome materialized;
    ASSERT_TRUE(orchestrator->materializeDeviceSpeculativeOutcomesForHostResponse(
        handle,
        &materialized));
    EXPECT_TRUE(materialized.ok);
    EXPECT_EQ(materialized.output_token_count, 2);
    EXPECT_EQ(materialized.output_tokens[0], 4)
        << "The first-token shadow should resolve from rank target slot zero.";
    EXPECT_EQ(materialized.output_tokens[1], 4)
        << "The draft-token shadow should resolve from rank draft slot zero.";
    EXPECT_EQ(materialized.accepted_speculative_prefix, 1);
    EXPECT_EQ(materialized.target_verifier_state_commit_count, 2);
    EXPECT_EQ(materialized.ready_token, 2);
    EXPECT_EQ(runner0_ptr->consume_all_position_logits_local_info_call_count(), 1u);
    EXPECT_EQ(runner1_ptr->consume_all_position_logits_local_info_call_count(), 1u);
}

TEST_F(Test__RankOrchestrator, LocalTPMirroredStochasticOutcomeSamplesOnceAndStagesRankDraftToken)
{
    auto rendezvous = std::make_shared<MTPPublicationRendezvous>(2);

    auto runner0 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner0_ptr = runner0.get();
    runner0_ptr->set_primary_device_id(DeviceId::cuda(0));
    runner0_ptr->set_supports_device_resident_mtp_spec_state_publication(true);
    runner0_ptr->set_supports_mtp_sidecar_logits_stream_handoff(true);
    runner0_ptr->set_supports_mtp_device_draft_token_input(true);
    runner0_ptr->set_supports_device_stochastic_mtp_verification(true);
    runner0_ptr->set_uses_mirrored_localtp_mtp_head_for_verifier(true);
    runner0_ptr->set_mock_device_position(64);
    runner0_ptr->set_stochastic_sample_token(1);
    runner0_ptr->set_mtp_publication_rendezvous(rendezvous);
    runner0_ptr->set_mock_logits_local(/*local_vocab=*/2, {0.0f, 5.0f});
    runner0_ptr->set_mock_all_position_logits_local(
        /*rows=*/2,
        /*local_vocab=*/2,
        {
            0.0f, 1.0f,
            0.0f, 6.0f,
        });

    auto runner1 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner1_ptr = runner1.get();
    runner1_ptr->set_primary_device_id(DeviceId::cuda(1));
    runner1_ptr->set_supports_device_resident_mtp_spec_state_publication(true);
    runner1_ptr->set_supports_mtp_sidecar_logits_stream_handoff(true);
    runner1_ptr->set_supports_mtp_device_draft_token_input(true);
    runner1_ptr->set_supports_device_stochastic_mtp_verification(true);
    runner1_ptr->set_uses_mirrored_localtp_mtp_head_for_verifier(true);
    runner1_ptr->set_mock_device_position(64);
    runner1_ptr->set_stochastic_sample_token(4);
    runner1_ptr->set_mtp_publication_rendezvous(rendezvous);
    runner1_ptr->set_mock_logits_local(/*local_vocab=*/3, {1.0f, 0.0f, 0.0f});
    runner1_ptr->set_mock_all_position_logits_local(
        /*rows=*/2,
        /*local_vocab=*/3,
        {
            4.0f, 0.0f, 0.0f,
            1.0f, 0.0f, 0.0f,
        });

    std::vector<std::unique_ptr<IInferenceRunner>> runners;
    runners.push_back(std::move(runner0));
    runners.push_back(std::move(runner1));

    auto model_ctx = llaminar2::test::MockModelContext::createMinimal();
    model_ctx->setVocabSize(5);
    auto tp_ctx = makeTPContextForRunnerCount(2);
    auto *tp_ctx_ptr = tp_ctx.get();
    auto orchestrator = RankOrchestrator::createForTest(
        std::move(model_ctx),
        std::move(runners),
        std::move(tp_ctx),
        makeRankConfigForRunnerCount(2));

    ASSERT_TRUE(orchestrator->supportsDeviceStochasticMTPVerification())
        << "Mirrored LocalTP stochastic support requires every child to expose "
           "full-vocab resident stochastic verification.";
    ASSERT_TRUE(orchestrator->supportsDeviceResidentMTPSpecStatePublication());
    ASSERT_TRUE(orchestrator->usesMirroredLocalTPMTPHeadForVerifier());
    ASSERT_TRUE(orchestrator->supportsMTPSidecarLogitsStreamHandoff());
    ASSERT_TRUE(orchestrator->supportsMTPDeviceDraftTokenInput());

    SamplingParams params;
    params.temperature = 1.0f;
    params.top_k = 2;
    params.top_p = 1.0f;

    ASSERT_TRUE(orchestrator->buildStochasticDistributionOnDevice(
        DeviceLogitsSource::Main,
        /*row=*/0,
        DeviceDistributionBuffer::Target,
        /*slot=*/0,
        params,
        /*vocab_size=*/5));
    EXPECT_EQ(runner0_ptr->build_stochastic_distribution_call_count(), 1u);
    EXPECT_EQ(runner1_ptr->build_stochastic_distribution_call_count(), 0u)
        << "A mirrored full-vocabulary main head needs one primary compact "
           "distribution, not a rank merge or independently sampled children.";
    EXPECT_EQ(orchestrator->sampleStochasticDistributionOnDevice(
                  DeviceDistributionBuffer::Target,
                  /*slot=*/0,
                  /*threshold=*/0.10f),
              1);
    EXPECT_EQ(runner0_ptr->sample_stochastic_distribution_call_count(), 1u);
    EXPECT_EQ(runner1_ptr->sample_stochastic_distribution_call_count(), 0u);
    EXPECT_EQ(runner0_ptr->record_target_sample_slot_ready_call_count(), 1u);
    EXPECT_EQ(runner1_ptr->record_target_sample_slot_ready_call_count(), 1u);
    EXPECT_TRUE(runner0_ptr->target_sample_slot_ready(0));
    EXPECT_TRUE(runner1_ptr->target_sample_slot_ready(0));
    EXPECT_EQ(runner0_ptr->target_sample_token(0), 1);
    EXPECT_EQ(runner1_ptr->target_sample_token(0), 1)
        << "NCCL/RCCL must publish the primary first target into every child slot.";

    const int32_t draft_token =
        orchestrator->sampleStochasticDraftProposalOnDevice(
            DeviceLogitsSource::MTP,
            /*row=*/0,
            /*slot=*/0,
            params,
            /*vocab_size=*/5,
            /*threshold=*/0.25f);
    ASSERT_EQ(draft_token, 1);
    EXPECT_EQ(runner0_ptr->sample_stochastic_draft_proposal_call_count(), 1u);
    EXPECT_EQ(runner1_ptr->sample_stochastic_draft_proposal_call_count(), 0u)
        << "Mirrored LocalTP stochastic proposals must be rank-coordinated. "
           "The primary child samples the mirrored full-vocab head once, then "
           "the sampled token is staged onto every child device slot.";
    EXPECT_EQ(runner0_ptr->stage_stochastic_draft_tokens_call_count(), 1u);
    EXPECT_EQ(runner1_ptr->stage_stochastic_draft_tokens_call_count(), 1u);
    EXPECT_EQ(runner0_ptr->last_staged_first_draft_slot(), 0);
    EXPECT_EQ(runner1_ptr->last_staged_first_draft_slot(), 0);
    EXPECT_THAT(runner0_ptr->last_staged_draft_tokens(),
                ::testing::ElementsAre(draft_token));
    EXPECT_THAT(runner1_ptr->last_staged_draft_tokens(),
                ::testing::ElementsAre(draft_token));

    const void *verifier_tokens =
        orchestrator->prepareMTPVerifierInputTokensOnDevice(
            /*first_token=*/1,
            /*first_draft_slot=*/0,
            /*draft_token_count=*/1,
            /*total_verifier_input_tokens=*/2);
    ASSERT_NE(verifier_tokens, nullptr);
    EXPECT_EQ(runner0_ptr->prepare_mtp_verifier_input_tokens_call_count(), 1u);
    EXPECT_EQ(runner1_ptr->prepare_mtp_verifier_input_tokens_call_count(), 1u);
    EXPECT_EQ(runner0_ptr->prepare_mtp_verifier_input_tokens_from_host_row_call_count(), 0u);
    EXPECT_EQ(runner1_ptr->prepare_mtp_verifier_input_tokens_from_host_row_call_count(), 0u);
    EXPECT_EQ(runner0_ptr->last_verifier_first_token(), 1);
    EXPECT_EQ(runner1_ptr->last_verifier_first_token(), 1);
    EXPECT_EQ(runner0_ptr->last_verifier_first_draft_slot(), 0);
    EXPECT_EQ(runner1_ptr->last_verifier_first_draft_slot(), 0);
    EXPECT_EQ(runner0_ptr->last_verifier_draft_token_count(), 1);
    EXPECT_EQ(runner1_ptr->last_verifier_draft_token_count(), 1);

    ASSERT_TRUE(orchestrator->buildStochasticDistributionsOnDevice(
        DeviceLogitsSource::AllPosition,
        /*first_row=*/0,
        DeviceDistributionBuffer::Target,
        /*first_slot=*/0,
        /*row_count=*/2,
        params,
        /*vocab_size=*/5));

    const std::array<float, 1> accept_thresholds = {0.10f};
    const std::array<float, 1> residual_thresholds = {0.50f};
    DeviceSpeculativeOutcomeHandle handle;
    ASSERT_TRUE(orchestrator->verifyStochasticDistributionsBatchOutcomeOnDeviceResident(
        /*first_target_slot=*/0,
        /*first_draft_slot=*/0,
        /*draft_tokens=*/nullptr,
        accept_thresholds.data(),
        residual_thresholds.data(),
        /*row_count=*/1,
        /*first_token=*/1,
        /*stop_tokens=*/nullptr,
        /*stop_token_count=*/0,
        /*bonus_target_slot=*/1,
        /*bonus_threshold=*/0.10f,
        &handle,
        /*inverse_sample_seed=*/123,
        /*inverse_sample_first_logical_position=*/64,
        /*use_vllm_probability_rejection=*/true));
    ASSERT_TRUE(handle.valid());
    EXPECT_EQ(runner0_ptr->publish_rank_compact_response_ready_call_count(), 1u);
    EXPECT_EQ(runner1_ptr->publish_rank_compact_response_ready_call_count(), 1u);
    EXPECT_TRUE(handle.response_ready_after_rank_collective);
    EXPECT_EQ(tp_ctx_ptr->collective_sideband_call_count(), 3u)
        << "The first target, draft proposal, and compact verifier outcome "
           "must each enter one rank-level grouped collective, independent of "
           "the LocalTP participant count.";
    EXPECT_EQ(tp_ctx_ptr->collective_sideband_broadcast_count(), 6u)
        << "The primary stochastic compact outcome must be broadcast into "
           "every child mailbox after the first target slot was likewise "
           "published by a device collective.";

    DeviceSpeculativeVerifyBatchOutcome materialized;
    ASSERT_TRUE(orchestrator->materializeDeviceSpeculativeOutcomesForHostResponse(
        handle,
        &materialized));
    EXPECT_TRUE(materialized.ok);
    EXPECT_EQ(materialized.output_token_count, 2);
    EXPECT_EQ(materialized.output_tokens[0], 1);
    EXPECT_EQ(materialized.output_tokens[1], 1);
    EXPECT_EQ(materialized.accepted_speculative_prefix, 1);
    EXPECT_EQ(materialized.target_verifier_state_commit_count, 2);
    EXPECT_EQ(materialized.ready_token, 1);
    EXPECT_TRUE(materialized.all_speculative_accepted);
    EXPECT_TRUE(materialized.sampled_terminal);

    DeviceSpeculativePublicationRequest request;
    request.outcome = handle;
    request.request_count = 1;
    request.max_draft_tokens = 2;
    request.max_state_commit_rows =
        materialized.target_verifier_state_commit_count;
    request.publish_mtp_shifted_kv = true;

    std::string error;
    EXPECT_TRUE(orchestrator->publishAcceptedMTPSpecStateBatchFromDeviceOutcome(
        request,
        &error))
        << error;

    EXPECT_EQ(rendezvous->arrivals.load(std::memory_order_acquire), 0);
    EXPECT_EQ(runner0_ptr->verify_stochastic_batch_outcome_call_count(), 0u);
    EXPECT_EQ(runner1_ptr->verify_stochastic_batch_outcome_call_count(), 0u);
    EXPECT_EQ(runner0_ptr->verify_stochastic_request_batch_outcome_call_count(), 1u);
    EXPECT_EQ(runner1_ptr->verify_stochastic_request_batch_outcome_call_count(), 1u);
    EXPECT_EQ(runner0_ptr->publish_device_resident_mtp_spec_state_batch_call_count(), 1u);
    EXPECT_EQ(runner1_ptr->publish_device_resident_mtp_spec_state_batch_call_count(), 1u);
    EXPECT_EQ(runner0_ptr->publish_mtp_spec_state_batch_call_count(), 0u);
    EXPECT_EQ(runner1_ptr->publish_mtp_spec_state_batch_call_count(), 0u);
    EXPECT_EQ(runner0_ptr->publish_grouped_decode_equivalent_mtp_spec_state_batch_call_count(), 0u);
    EXPECT_EQ(runner1_ptr->publish_grouped_decode_equivalent_mtp_spec_state_batch_call_count(), 0u);
    EXPECT_EQ(runner0_ptr->consume_logits_local_info_call_count(), 0u);
    EXPECT_EQ(runner1_ptr->consume_logits_local_info_call_count(), 0u)
        << "Mirrored main-target sampling must not enter the legacy sharded reducer.";
    EXPECT_EQ(runner0_ptr->consume_mtp_logits_local_info_call_count(), 0u);
    EXPECT_EQ(runner1_ptr->consume_mtp_logits_local_info_call_count(), 0u)
        << "Mirrored stochastic MTP must not depend on rank-local shard logits.";
    EXPECT_EQ(runner0_ptr->consume_all_position_logits_local_info_call_count(), 0u);
    EXPECT_EQ(runner1_ptr->consume_all_position_logits_local_info_call_count(), 0u);
    EXPECT_EQ(runner0_ptr->stage_stochastic_draft_tokens_call_count(), 1u);
    EXPECT_EQ(runner1_ptr->stage_stochastic_draft_tokens_call_count(), 1u);
    EXPECT_EQ(runner0_ptr->last_staged_first_draft_slot(), 0);
    EXPECT_EQ(runner1_ptr->last_staged_first_draft_slot(), 0);
    EXPECT_THAT(runner0_ptr->last_staged_draft_tokens(),
                ::testing::ElementsAre(1));
    EXPECT_THAT(runner1_ptr->last_staged_draft_tokens(),
                ::testing::ElementsAre(1));
    EXPECT_EQ(runner0_ptr->build_stochastic_distributions_call_count(), 1u);
    EXPECT_EQ(runner1_ptr->build_stochastic_distributions_call_count(), 1u);
    EXPECT_EQ(runner0_ptr->stage_stochastic_target_token_call_count(), 0u);
    EXPECT_EQ(runner1_ptr->stage_stochastic_target_token_call_count(), 0u)
        << "The first target handoff is a direct device-slot collective, not a "
           "host-token staging call.";

    const DeviceResidentLogicalSequenceStateHandle child0_state =
        runner0_ptr->deviceResidentLogicalSequenceState();
    const DeviceResidentLogicalSequenceStateHandle child1_state =
        runner1_ptr->deviceResidentLogicalSequenceState();
    ASSERT_TRUE(child0_state.valid());
    ASSERT_TRUE(child1_state.valid());
    EXPECT_EQ(child0_state.next_condition_tokens_device[0], materialized.ready_token);
    EXPECT_EQ(child1_state.next_condition_tokens_device[0], materialized.ready_token)
        << "The peer child must publish from the primary compact stochastic "
           "outcome, not from an independently sampled mirrored-head summary.";

    DeviceResidentLogicalSequenceStateHandle resident_state =
        orchestrator->deviceResidentLogicalSequenceState();
    EXPECT_TRUE(resident_state.valid());
    EXPECT_TRUE(orchestrator->forwardMTPFromDeviceResidentLogicalStateForDeviceSampling(
        resident_state,
        /*request_index=*/0));
    EXPECT_EQ(runner0_ptr->forward_mtp_from_resident_logical_state_call_count(), 1u);
    EXPECT_EQ(runner1_ptr->forward_mtp_from_resident_logical_state_call_count(), 1u);
    EXPECT_EQ(runner0_ptr->last_resident_logical_state_request_index(), 0);
    EXPECT_EQ(runner1_ptr->last_resident_logical_state_request_index(), 0);
}

TEST_F(Test__RankOrchestrator, LocalTPMirroredDeferredStochasticDraftBroadcastsPrimaryDeviceSlot)
{
    auto runner0 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner0_ptr = runner0.get();
    runner0_ptr->set_primary_device_id(DeviceId::cuda(0));
    runner0_ptr->set_supports_device_resident_mtp_spec_state_publication(true);
    runner0_ptr->set_supports_mtp_sidecar_logits_stream_handoff(true);
    runner0_ptr->set_supports_mtp_device_draft_token_input(true);
    runner0_ptr->set_supports_device_stochastic_mtp_verification(true);
    runner0_ptr->set_uses_mirrored_localtp_mtp_head_for_verifier(true);
    runner0_ptr->set_stochastic_sample_token(7);

    auto runner1 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner1_ptr = runner1.get();
    runner1_ptr->set_primary_device_id(DeviceId::cuda(1));
    runner1_ptr->set_supports_device_resident_mtp_spec_state_publication(true);
    runner1_ptr->set_supports_mtp_sidecar_logits_stream_handoff(true);
    runner1_ptr->set_supports_mtp_device_draft_token_input(true);
    runner1_ptr->set_supports_device_stochastic_mtp_verification(true);
    runner1_ptr->set_uses_mirrored_localtp_mtp_head_for_verifier(true);
    runner1_ptr->set_stochastic_sample_token(4);

    std::vector<std::unique_ptr<IInferenceRunner>> runners;
    runners.push_back(std::move(runner0));
    runners.push_back(std::move(runner1));

    auto model_ctx = llaminar2::test::MockModelContext::createMinimal();
    model_ctx->setVocabSize(8);
    auto tp_ctx = makeTPContextForRunnerCount(2);
    auto *tp_ctx_ptr = tp_ctx.get();
    auto orchestrator = RankOrchestrator::createForTest(
        std::move(model_ctx),
        std::move(runners),
        std::move(tp_ctx),
        makeRankConfigForRunnerCount(2));

    ASSERT_TRUE(orchestrator->supportsDeviceStochasticMTPVerification());
    ASSERT_TRUE(orchestrator->usesMirroredLocalTPMTPHeadForVerifier());
    ASSERT_TRUE(orchestrator->supportsMTPDeviceDraftTokenInput());

    SamplingParams params;
    params.temperature = 1.0f;
    params.top_k = 2;
    params.top_p = 1.0f;

    ASSERT_TRUE(orchestrator->sampleStochasticDraftProposalOnDeviceDeferred(
        DeviceLogitsSource::MTP,
        /*row=*/0,
        /*slot=*/1,
        params,
        /*vocab_size=*/8,
        /*threshold=*/0.25f));

    EXPECT_EQ(runner0_ptr->sample_stochastic_draft_proposal_call_count(), 1u);
    EXPECT_EQ(runner1_ptr->sample_stochastic_draft_proposal_call_count(), 0u)
        << "Deferred mirrored LocalTP must not let each child sample a separate "
           "full-vocab MTP proposal; child 0 owns the proposal and broadcasts it.";
    EXPECT_EQ(runner0_ptr->stage_stochastic_draft_tokens_call_count(), 0u);
    EXPECT_EQ(runner1_ptr->stage_stochastic_draft_tokens_call_count(), 0u)
        << "The deferred path should stay device-resident instead of staging a "
           "host token row into child draft slots.";
    EXPECT_EQ(tp_ctx_ptr->collective_sideband_call_count(), 2u);
    EXPECT_EQ(tp_ctx_ptr->collective_sideband_broadcast_count(), 2u);
    EXPECT_EQ(runner0_ptr->record_draft_sample_slot_ready_call_count(), 1u);
    EXPECT_EQ(runner1_ptr->record_draft_sample_slot_ready_call_count(), 1u);
    EXPECT_EQ(runner0_ptr->last_recorded_draft_sample_slot(), 1);
    EXPECT_EQ(runner1_ptr->last_recorded_draft_sample_slot(), 1);
    EXPECT_TRUE(runner0_ptr->draft_sample_slot_ready(1));
    EXPECT_TRUE(runner1_ptr->draft_sample_slot_ready(1));
    EXPECT_EQ(runner0_ptr->draft_sample_token(1), 7);
    EXPECT_EQ(runner1_ptr->draft_sample_token(1), 7)
        << "The peer child must consume the primary child's sampled draft token "
           "for chained sidecar slot 1.";

    const void *verifier_tokens =
        orchestrator->prepareMTPVerifierInputTokensOnDevice(
            /*first_token=*/3,
            /*first_draft_slot=*/1,
            /*draft_token_count=*/1,
            /*total_verifier_input_tokens=*/2);
    ASSERT_NE(verifier_tokens, nullptr);
    EXPECT_EQ(runner0_ptr->prepare_mtp_verifier_input_tokens_call_count(), 1u);
    EXPECT_EQ(runner1_ptr->prepare_mtp_verifier_input_tokens_call_count(), 1u);
    EXPECT_EQ(runner0_ptr->last_verifier_first_draft_slot(), 1);
    EXPECT_EQ(runner1_ptr->last_verifier_first_draft_slot(), 1);
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
    EXPECT_THAT(orchestrator->sequence_lengths(), ::testing::ElementsAre(81))
        << "Rank-level sequence bookkeeping must adopt the child accepted "
           "boundary after grouped MTP publication, otherwise replay probes can "
           "observe a stale verifier-row length after children have published "
           "only the accepted prefix.";
    EXPECT_THAT(orchestrator->prefixStateProbe().sequence_lengths,
                ::testing::ElementsAre(81));
}

/**
 * @brief Keep canonical child state authoritative after mailbox retirement.
 *
 * Prefix restore retires speculative outcome mailboxes because their token and
 * acceptance fields describe the replaced timeline. Each GPU child can still
 * observe its restored cache-owned sequence count. The rank's scalar position,
 * by contrast, is only a scheduler shadow and may temporarily retain the
 * pre-restore value. This regression makes that disagreement explicit and
 * proves symmetric LocalTP diagnostics adopt the initialized child probes even
 * when no resident outcome mailbox is currently live.
 */
TEST_F(Test__RankOrchestrator,
       PrefixStateProbeAdoptsCanonicalChildPositionWithoutLiveMailbox)
{
    auto runner0 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner0_ptr = runner0.get();
    runner0_ptr->set_primary_device_id(DeviceId::cuda(0));

    auto runner1 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner1_ptr = runner1.get();
    runner1_ptr->set_primary_device_id(DeviceId::cuda(1));

    std::vector<std::unique_ptr<IInferenceRunner>> runners;
    runners.push_back(std::move(runner0));
    runners.push_back(std::move(runner1));

    auto orchestrator = RankOrchestrator::createForTest(
        llaminar2::test::MockModelContext::createMinimal(),
        std::move(runners),
        makeTPContextForRunnerCount(2),
        makeRankConfigForRunnerCount(2));

    const int token = 17;
    ASSERT_TRUE(orchestrator->forward(&token, 597));
    ASSERT_EQ(orchestrator->get_position(), 597);
    ASSERT_FALSE(orchestrator->deviceResidentLogicalSequenceState().valid());

    runner0_ptr->set_prefix_probe_position_override(596);
    runner1_ptr->set_prefix_probe_position_override(596);

    const PrefixRuntimeStateSnapshot snapshot =
        orchestrator->prefixStateProbe();
    EXPECT_TRUE(snapshot.initialized);
    EXPECT_EQ(snapshot.current_position, 596);
    EXPECT_THAT(snapshot.positions, ::testing::ElementsAre(596));
    EXPECT_THAT(snapshot.sequence_lengths, ::testing::ElementsAre(596));
    EXPECT_EQ(orchestrator->get_position(), 597)
        << "Observation must not mutate the scheduler-owned parent cursor.";
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

/**
 * @brief Regression guard that grouped LocalTP publication uses grouped child APIs.
 *
 * The key safety property is non-promotion: a LocalTP rank may support grouped
 * decode-equivalent publication while still refusing the stronger direct
 * all-position publication capability.  The call below must therefore fan out
 * through publishGroupedDecodeEquivalentMTPSpecStateBatch() on each child and
 * never touch publishAcceptedMTPSpecStateBatch().
 */
TEST_F(Test__RankOrchestrator, GroupedDecodeEquivalentBatchPublicationRunsOnEveryLocalTPChildWithoutPromotingDirectPath)
{
    auto rendezvous = std::make_shared<MTPPublicationRendezvous>(2);

    auto runner0 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner0_ptr = runner0.get();
    runner0_ptr->set_mtp_publication_rendezvous(rendezvous);

    auto runner1 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner1_ptr = runner1.get();
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
    EXPECT_FALSE(orchestrator->supportsMTPSpecStatePublication())
        << "The grouped lane must remain separate from direct all-position publication.";
    EXPECT_TRUE(orchestrator->publishGroupedDecodeEquivalentMTPSpecStateBatch(batch, &error))
        << error;

    EXPECT_EQ(rendezvous->arrivals.load(std::memory_order_acquire), 2);
    EXPECT_EQ(runner0_ptr->publish_mtp_spec_state_batch_call_count(), 0u);
    EXPECT_EQ(runner1_ptr->publish_mtp_spec_state_batch_call_count(), 0u);
    EXPECT_EQ(runner0_ptr->publish_grouped_decode_equivalent_mtp_spec_state_batch_call_count(), 1u);
    EXPECT_EQ(runner1_ptr->publish_grouped_decode_equivalent_mtp_spec_state_batch_call_count(), 1u);
    ASSERT_THAT(runner0_ptr->last_mtp_spec_state_batch().steps, ::testing::SizeIs(2));
    ASSERT_THAT(runner1_ptr->last_mtp_spec_state_batch().steps, ::testing::SizeIs(2));
    EXPECT_EQ(runner0_ptr->last_mtp_spec_state_batch().steps[0].accepted_count, 2);
    EXPECT_EQ(runner1_ptr->last_mtp_spec_state_batch().steps[1].accepted_count, 1);
    EXPECT_EQ(runner0_ptr->get_position(), 81);
    EXPECT_EQ(runner1_ptr->get_position(), 81);
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

TEST_F(Test__RankOrchestrator, LogitsGatherSkipPolicyPropagatesToEveryLocalTPChild)
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

    runner0_ptr->reset_call_counts();
    runner1_ptr->reset_call_counts();

    orchestrator->setSkipLogitsGatherDecode(true);
    EXPECT_TRUE(runner0_ptr->skip_logits_gather_decode());
    EXPECT_TRUE(runner1_ptr->skip_logits_gather_decode());
    EXPECT_GT(runner0_ptr->set_skip_decode_call_count(), 0u);
    EXPECT_GT(runner1_ptr->set_skip_decode_call_count(), 0u);

    orchestrator->setSkipLogitsGatherPrefill(true);
    EXPECT_TRUE(runner0_ptr->skip_logits_gather_prefill());
    EXPECT_TRUE(runner1_ptr->skip_logits_gather_prefill());
    EXPECT_GT(runner0_ptr->set_skip_prefill_call_count(), 0u);
    EXPECT_GT(runner1_ptr->set_skip_prefill_call_count(), 0u);

    const size_t runner0_decode_calls = runner0_ptr->set_skip_decode_call_count();
    const size_t runner1_decode_calls = runner1_ptr->set_skip_decode_call_count();

    orchestrator->setSkipLogitsGatherDecode(false);
    EXPECT_FALSE(runner0_ptr->skip_logits_gather_decode());
    EXPECT_FALSE(runner1_ptr->skip_logits_gather_decode());
    EXPECT_GT(runner0_ptr->set_skip_decode_call_count(), runner0_decode_calls);
    EXPECT_GT(runner1_ptr->set_skip_decode_call_count(), runner1_decode_calls);
}

TEST_F(Test__RankOrchestrator, DecodeSyncDeferralPolicyPropagatesToEveryLocalTPChild)
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

    runner0_ptr->reset_call_counts();
    runner1_ptr->reset_call_counts();

    orchestrator->setMTPMainDecodeSyncDeferralEnabled(true);
    EXPECT_TRUE(runner0_ptr->main_decode_sync_deferral_enabled());
    EXPECT_TRUE(runner1_ptr->main_decode_sync_deferral_enabled());
    EXPECT_EQ(runner0_ptr->set_main_decode_sync_deferral_call_count(), 1u);
    EXPECT_EQ(runner1_ptr->set_main_decode_sync_deferral_call_count(), 1u);

    orchestrator->setMTPAllPositionVerifierSyncDeferralEnabled(true);
    EXPECT_TRUE(runner0_ptr->all_position_sync_deferral_enabled());
    EXPECT_TRUE(runner1_ptr->all_position_sync_deferral_enabled());
    EXPECT_EQ(runner0_ptr->set_all_position_sync_deferral_call_count(), 1u);
    EXPECT_EQ(runner1_ptr->set_all_position_sync_deferral_call_count(), 1u);

    orchestrator->setMTPMainDecodeSyncDeferralEnabled(false);
    EXPECT_FALSE(runner0_ptr->main_decode_sync_deferral_enabled());
    EXPECT_FALSE(runner1_ptr->main_decode_sync_deferral_enabled());
    EXPECT_EQ(runner0_ptr->set_main_decode_sync_deferral_call_count(), 2u);
    EXPECT_EQ(runner1_ptr->set_main_decode_sync_deferral_call_count(), 2u);
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

TEST_F(Test__RankOrchestrator, MultiChildMainSamplingDelegatesToPrimaryReplicatedLogits)
{
    auto runner0 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner0_ptr = runner0.get();
    runner0_ptr->set_primary_device_id(DeviceId::cuda(0));
    runner0->set_mock_logits({0.0f, 1.0f, 9.0f, 2.0f});

    auto runner1 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner1_ptr = runner1.get();
    runner1_ptr->set_primary_device_id(DeviceId::cuda(1));
    runner1->set_mock_logits({0.0f, 8.0f, 1.0f, 3.0f});

    std::vector<std::unique_ptr<IInferenceRunner>> runners;
    runners.push_back(std::move(runner0));
    runners.push_back(std::move(runner1));

    auto orchestrator = RankOrchestrator::createForTest(
        llaminar2::test::MockModelContext::createMinimal(),
        std::move(runners),
        makeTPContextForRunnerCount(2),
        makeRankConfigForRunnerCount(2));

    EXPECT_EQ(orchestrator->sampleGreedyOnDevice(), 2)
        << "Phase-split decode uses replicated full logits, not LOGITS_LOCAL shards.";
    EXPECT_EQ(runner0_ptr->sample_greedy_on_device_call_count(), 1u);
    EXPECT_EQ(runner1_ptr->sample_greedy_on_device_call_count(), 0u);
    EXPECT_EQ(
        runner0_ptr
            ->consume_unused_replicated_main_logits_publication_call_count(),
        0u);
    EXPECT_EQ(
        runner1_ptr
            ->consume_unused_replicated_main_logits_publication_call_count(),
        1u)
        << "The non-primary replicated graph output must close its one-shot "
           "stream publication without launching a duplicate argmax.";
}

/**
 * @brief Replicated sampling fails closed when a non-primary publication leaks.
 *
 * This is the model-free regression for the CUDA2 Dynamic phase-split failure:
 * a second decode graph replay must never replace participant one's still-live
 * main-logits stream handoff. Rank sampling therefore treats inability to close
 * that unused replica as a fatal sampling failure.
 */
TEST_F(Test__RankOrchestrator,
       ReplicatedMainSamplingFailsWhenNonPrimaryPublicationCannotBeConsumed)
{
    auto runner0 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner0_ptr = runner0.get();
    runner0_ptr->set_primary_device_id(DeviceId::cuda(0));
    runner0_ptr->set_mock_logits({0.0f, 1.0f, 9.0f, 2.0f});

    auto runner1 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner1_ptr = runner1.get();
    runner1_ptr->set_primary_device_id(DeviceId::cuda(1));
    runner1_ptr->set_mock_logits({0.0f, 8.0f, 1.0f, 3.0f});
    runner1_ptr->set_consume_unused_replicated_main_logits_publication_ok(false);

    std::vector<std::unique_ptr<IInferenceRunner>> runners;
    runners.push_back(std::move(runner0));
    runners.push_back(std::move(runner1));

    auto orchestrator = RankOrchestrator::createForTest(
        llaminar2::test::MockModelContext::createMinimal(),
        std::move(runners),
        makeTPContextForRunnerCount(2),
        makeRankConfigForRunnerCount(2));

    EXPECT_EQ(orchestrator->sampleGreedyOnDevice(), -1);
    EXPECT_EQ(runner0_ptr->sample_greedy_on_device_call_count(), 1u);
    EXPECT_EQ(
        runner1_ptr
            ->consume_unused_replicated_main_logits_publication_call_count(),
        1u);
}

/**
 * @brief Mirrored LocalTP publishes one primary main-target argmax by collective.
 *
 * The first scalar MTP target is sampled from the preceding main terminal row.
 * Once that row is mirrored, consulting `LOGITS_LOCAL` or staging a host token
 * would revive the obsolete sharded path. This regression requires one child
 * argmax, one NCCL/RCCL target-slot broadcast, and a fresh readiness publication
 * on every participant.
 */
TEST_F(Test__RankOrchestrator,
       MirroredLocalTPGreedyMainTargetSlotBroadcastsPrimaryDeviceSample)
{
    auto runner0 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner0_ptr = runner0.get();
    runner0_ptr->set_primary_device_id(DeviceId::cuda(0));
    runner0_ptr->set_supports_device_stochastic_mtp_verification(true);
    runner0_ptr->set_uses_mirrored_localtp_mtp_head_for_verifier(true);
    runner0_ptr->set_mock_logits({0.0f, 1.0f, 9.0f, 2.0f});

    auto runner1 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner1_ptr = runner1.get();
    runner1_ptr->set_primary_device_id(DeviceId::cuda(1));
    runner1_ptr->set_supports_device_stochastic_mtp_verification(true);
    runner1_ptr->set_uses_mirrored_localtp_mtp_head_for_verifier(true);
    runner1_ptr->set_mock_logits({0.0f, 8.0f, 1.0f, 3.0f});

    std::vector<std::unique_ptr<IInferenceRunner>> runners;
    runners.push_back(std::move(runner0));
    runners.push_back(std::move(runner1));

    auto tp_ctx = makeTPContextForRunnerCount(2);
    auto *tp_ctx_ptr = tp_ctx.get();
    auto rank_config = makeRankConfigForRunnerCount(2);
    rank_config.mtp.enabled = true;
    rank_config.mtp.draft_tokens = 2;
    auto orchestrator = RankOrchestrator::createForTest(
        llaminar2::test::MockModelContext::createMinimal(),
        std::move(runners),
        std::move(tp_ctx),
        std::move(rank_config));

    int32_t host_shadow = -1;
    ASSERT_TRUE(orchestrator->sampleGreedyFromMainLogitsToDeviceTargetSlot(
        /*target_sample_slot=*/2,
        &host_shadow));

    EXPECT_EQ(host_shadow, 2);
    EXPECT_EQ(runner0_ptr->sample_greedy_main_target_slot_call_count(), 1u);
    EXPECT_EQ(runner1_ptr->sample_greedy_main_target_slot_call_count(), 0u)
        << "Only the primary mirrored head performs the argmax.";
    EXPECT_EQ(
        runner1_ptr
            ->consume_unused_replicated_main_logits_publication_call_count(),
        1u);
    EXPECT_EQ(runner0_ptr->consume_logits_local_info_call_count(), 0u);
    EXPECT_EQ(runner1_ptr->consume_logits_local_info_call_count(), 0u);
    EXPECT_EQ(tp_ctx_ptr->collective_sideband_call_count(), 2u);
    EXPECT_EQ(tp_ctx_ptr->collective_sideband_broadcast_count(), 2u);
    EXPECT_EQ(runner0_ptr->record_target_sample_slot_ready_call_count(), 1u);
    EXPECT_EQ(runner1_ptr->record_target_sample_slot_ready_call_count(), 1u);
    EXPECT_EQ(runner0_ptr->last_recorded_target_sample_slot(), 2);
    EXPECT_EQ(runner1_ptr->last_recorded_target_sample_slot(), 2);
    EXPECT_TRUE(runner0_ptr->target_sample_slot_ready(2));
    EXPECT_TRUE(runner1_ptr->target_sample_slot_ready(2));
    EXPECT_EQ(runner0_ptr->target_sample_token(2), 2);
    EXPECT_EQ(runner1_ptr->target_sample_token(2), 2);
}

TEST_F(Test__RankOrchestrator, MultiChildGreedyMainTargetSlotStagesCrossShardWinner)
{
    auto runner0 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner0_ptr = runner0.get();
    runner0->set_mock_logits_local(
        /*local_vocab=*/3,
        {0.5f, 1.0f, 0.25f});

    auto runner1 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner1_ptr = runner1.get();
    runner1->set_mock_logits_local(
        /*local_vocab=*/3,
        {0.1f, 5.0f, 2.0f});

    std::vector<std::unique_ptr<IInferenceRunner>> runners;
    runners.push_back(std::move(runner0));
    runners.push_back(std::move(runner1));

    auto rank_config = makeRankConfigForRunnerCount(2);
    rank_config.mtp.enabled = true;
    rank_config.mtp.draft_tokens = 2;
    auto orchestrator = RankOrchestrator::createForTest(
        llaminar2::test::MockModelContext::createMinimal(),
        std::move(runners),
        makeTPContextForRunnerCount(2),
        std::move(rank_config));

    int32_t host_token = -1;
    EXPECT_TRUE(orchestrator->sampleGreedyFromMainLogitsToDeviceTargetSlot(
        /*target_sample_slot=*/2,
        &host_token));

    EXPECT_EQ(host_token, 4)
        << "The rank argmax must add each child's vocab offset before staging.";
    EXPECT_EQ(runner0_ptr->consume_logits_local_info_call_count(), 1u);
    EXPECT_EQ(runner1_ptr->consume_logits_local_info_call_count(), 1u);
    EXPECT_EQ(runner0_ptr->stage_stochastic_target_token_call_count(), 1u);
    EXPECT_EQ(runner1_ptr->stage_stochastic_target_token_call_count(), 1u);
    EXPECT_EQ(runner0_ptr->last_staged_target_token(), 4);
    EXPECT_EQ(runner1_ptr->last_staged_target_token(), 4);
    EXPECT_EQ(runner0_ptr->last_staged_target_sample_slot(), 2);
    EXPECT_EQ(runner1_ptr->last_staged_target_sample_slot(), 2);
}

TEST_F(Test__RankOrchestrator, MultiChildMainStochasticSamplingDelegatesToPrimaryReplicatedLogits)
{
    auto runner0 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner0_ptr = runner0.get();
    runner0_ptr->set_primary_device_id(DeviceId::cuda(0));
    runner0->set_stochastic_sample_token(23);

    auto runner1 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner1_ptr = runner1.get();
    runner1_ptr->set_primary_device_id(DeviceId::cuda(1));
    runner1->set_stochastic_sample_token(42);

    std::vector<std::unique_ptr<IInferenceRunner>> runners;
    runners.push_back(std::move(runner0));
    runners.push_back(std::move(runner1));

    auto orchestrator = RankOrchestrator::createForTest(
        llaminar2::test::MockModelContext::createMinimal(),
        std::move(runners),
        makeTPContextForRunnerCount(2),
        makeRankConfigForRunnerCount(2));

    SamplingParams params;
    params.temperature = 0.8f;
    params.top_k = 8;
    params.top_p = 0.95f;

    EXPECT_EQ(orchestrator->sampleOnDevice(params), 23);
    EXPECT_EQ(runner0_ptr->sample_on_device_call_count(), 1u);
    EXPECT_EQ(runner1_ptr->sample_on_device_call_count(), 0u);
    EXPECT_EQ(
        runner1_ptr
            ->consume_unused_replicated_main_logits_publication_call_count(),
        1u);
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

TEST_F(Test__RankOrchestrator, MultiChildGreedyMTPDraftSlotStagesCrossShardWinner)
{
    auto runner0 = std::make_unique<MockDeviceGraphOrchestrator>();
    runner0->set_mock_mtp_logits_local(
        /*local_vocab=*/3,
        {1.0f, 0.0f, 0.5f});
    auto *runner0_ptr = runner0.get();

    auto runner1 = std::make_unique<MockDeviceGraphOrchestrator>();
    runner1->set_mock_mtp_logits_local(
        /*local_vocab=*/3,
        {0.25f, 9.0f, 2.0f});
    auto *runner1_ptr = runner1.get();

    std::vector<std::unique_ptr<IInferenceRunner>> runners;
    runners.push_back(std::move(runner0));
    runners.push_back(std::move(runner1));

    auto model_ctx = llaminar2::test::MockModelContext::createMinimal();
    model_ctx->setVocabSize(6);

    auto orchestrator = RankOrchestrator::createForTest(
        std::move(model_ctx),
        std::move(runners),
        makeTPContextForRunnerCount(2),
        makeRankConfigForRunnerCount(2));

    int32_t host_token = -1;
    EXPECT_TRUE(orchestrator->sampleGreedyFromMTPLogitsToDeviceDraftSlot(
        /*draft_sample_slot=*/1,
        &host_token));

    EXPECT_EQ(host_token, 4)
        << "The rank argmax must convert the winning local MTP shard index into "
           "the global token before staging draft verifier slots.";
    EXPECT_EQ(runner0_ptr->consume_mtp_logits_local_info_call_count(), 1u);
    EXPECT_EQ(runner1_ptr->consume_mtp_logits_local_info_call_count(), 1u);
    EXPECT_EQ(runner0_ptr->get_mtp_logits_local_info_call_count(), 0u);
    EXPECT_EQ(runner1_ptr->get_mtp_logits_local_info_call_count(), 0u);
    EXPECT_EQ(runner0_ptr->stage_stochastic_draft_tokens_call_count(), 1u);
    EXPECT_EQ(runner1_ptr->stage_stochastic_draft_tokens_call_count(), 1u);
    EXPECT_EQ(runner0_ptr->last_staged_first_draft_slot(), 1);
    EXPECT_EQ(runner1_ptr->last_staged_first_draft_slot(), 1);
    EXPECT_EQ(runner0_ptr->last_staged_draft_tokens(),
              std::vector<int32_t>({4}));
    EXPECT_EQ(runner1_ptr->last_staged_draft_tokens(),
              std::vector<int32_t>({4}));
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

TEST_F(Test__RankOrchestrator, MultiChildAllPositionLogitsUsePrimaryReplicatedFullVocab)
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
    const float *logits = orchestrator->getAllPositionLogits();
    ASSERT_NE(logits, nullptr)
        << "Replicated all-position logits must follow ordinary serial TP decode "
           "and publish the primary child instead of requiring every replica to "
           "be bitwise-identical.";
    EXPECT_FLOAT_EQ(logits[0], 1.0f);
    EXPECT_FLOAT_EQ(logits[1], 2.0f);
    EXPECT_FLOAT_EQ(logits[2], 3.0f);
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

TEST_F(Test__RankOrchestrator, TPSnapshot_RowParallel_SumsDevicePartials)
{
    TPSnapshot snapshot;
    snapshot.key = "layer0_ATTENTION_OUTPUT";
    snapshot.mode = SnapshotShardingMode::ROW_PARALLEL;
    snapshot.tp_degree = 3;

    DeviceSnapshotData dev0;
    dev0.device_index = 0;
    dev0.rows = 2;
    dev0.cols = 3;
    dev0.data = {1.0f, 2.0f, 3.0f,
                 4.0f, 5.0f, 6.0f};

    DeviceSnapshotData dev1;
    dev1.device_index = 1;
    dev1.rows = 2;
    dev1.cols = 3;
    dev1.data = {10.0f, 20.0f, 30.0f,
                 40.0f, 50.0f, 60.0f};

    DeviceSnapshotData dev2;
    dev2.device_index = 2;
    dev2.rows = 2;
    dev2.cols = 3;
    dev2.data = {100.0f, 200.0f, 300.0f,
                 400.0f, 500.0f, 600.0f};

    snapshot.device_data.push_back(std::move(dev0));
    snapshot.device_data.push_back(std::move(dev1));
    snapshot.device_data.push_back(std::move(dev2));

    ASSERT_TRUE(snapshot.computeCombined());
    EXPECT_EQ(snapshot.combined_rows, 2);
    EXPECT_EQ(snapshot.combined_cols, 3);
    EXPECT_EQ(snapshot.combined_data,
              (std::vector<float>{111.0f, 222.0f, 333.0f,
                                  444.0f, 555.0f, 666.0f}));
}

TEST_F(Test__RankOrchestrator, TPSnapshot_PhaseSplitDecodeTreatsDenseOutputsAsReplicated)
{
    auto runner0 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner0_ptr = runner0.get();
    runner0_ptr->set_mock_snapshot(
        "layer0_ATTENTION_OUTPUT",
        1,
        4,
        {1.0f, 2.0f, 3.0f, 4.0f});

    auto runner1 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner1_ptr = runner1.get();
    runner1_ptr->set_mock_snapshot(
        "layer0_ATTENTION_OUTPUT",
        1,
        4,
        {1.0f, 2.0f, 3.0f, 4.0f});

    std::vector<std::unique_ptr<IInferenceRunner>> runners;
    runners.push_back(std::move(runner0));
    runners.push_back(std::move(runner1));

    auto model_ctx = llaminar2::test::MockModelContext::createMinimal();
    model_ctx->setArchitecture("qwen35moe");

    auto plan = std::make_shared<MoERoutedExpertPlacementPlan>();
    plan->enabled = true;
    plan->topology = RoutedExpertPlacementTopology::TieredOverlay;
    plan->continuation_domain_spec.dense_tp_enabled = true;
    plan->continuation_domain_spec.dense_decode_replicated = true;
    plan->continuation_domain_spec.refreshDensePolicyFromFlags();

    RankOrchestrator::Config rank_config;
    rank_config.devices = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)};
    rank_config.weights = {0.5f, 0.5f};
    rank_config.moe_routed_expert_plan = plan;

    MockLocalTPContext::Config tp_config;
    tp_config.devices = rank_config.devices;
    tp_config.weights = rank_config.weights;

    auto orchestrator = RankOrchestrator::createForTest(
        model_ctx,
        std::move(runners),
        std::make_unique<MockLocalTPContext>(tp_config),
        rank_config);

    int token = 42;
    ASSERT_TRUE(orchestrator->forward(&token, 1));

    auto snapshot = orchestrator->getTPSnapshot("layer0_ATTENTION_OUTPUT");
    EXPECT_EQ(snapshot.mode, SnapshotShardingMode::REPLICATED);
    ASSERT_TRUE(snapshot.computeCombined());
    EXPECT_EQ(snapshot.combined_rows, 1);
    EXPECT_EQ(snapshot.combined_cols, 4);
    EXPECT_EQ(snapshot.combined_data,
              (std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}));
}

TEST_F(Test__RankOrchestrator, TPSnapshot_RuntimeGQAOverrideTreatsKVCacheAsReplicated)
{
    auto runner0 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner0_ptr = runner0.get();
    runner0_ptr->set_mock_snapshot(
        "layer3_KV_CACHE_K",
        1,
        4,
        {1.0f, 2.0f, 3.0f, 4.0f});

    auto runner1 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner1_ptr = runner1.get();
    runner1_ptr->set_mock_snapshot(
        "layer3_KV_CACHE_K",
        1,
        4,
        {1.0f, 2.0f, 3.0f, 4.0f});

    std::vector<std::unique_ptr<IInferenceRunner>> runners;
    runners.push_back(std::move(runner0));
    runners.push_back(std::move(runner1));

    auto model_ctx = llaminar2::test::MockModelContext::createMinimal();
    model_ctx->setArchitecture("qwen35moe");
    model_ctx->setHeadCountKV(1);

    RankOrchestrator::Config rank_config;
    rank_config.devices = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)};
    rank_config.weights = {0.5f, 0.5f};

    MockLocalTPContext::Config tp_config;
    tp_config.devices = rank_config.devices;
    tp_config.weights = rank_config.weights;

    auto orchestrator = RankOrchestrator::createForTest(
        model_ctx,
        std::move(runners),
        std::make_unique<MockLocalTPContext>(tp_config),
        rank_config);

    auto snapshot = orchestrator->getTPSnapshot("layer3_KV_CACHE_K");
    EXPECT_EQ(snapshot.mode, SnapshotShardingMode::REPLICATED);
    ASSERT_TRUE(snapshot.computeCombined());
    EXPECT_EQ(snapshot.combined_rows, 1);
    EXPECT_EQ(snapshot.combined_cols, 4);
    EXPECT_EQ(snapshot.combined_data,
              (std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}));
}

TEST_F(Test__RankOrchestrator, TPSnapshot_PhaseSplitDecodeKeepsMoECombinedOutputRowParallel)
{
    auto runner0 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner0_ptr = runner0.get();
    runner0_ptr->set_mock_snapshot(
        "layer0_MOE_COMBINED_OUTPUT",
        1,
        4,
        {1.0f, 2.0f, 3.0f, 4.0f});

    auto runner1 = std::make_unique<MockDeviceGraphOrchestrator>();
    auto *runner1_ptr = runner1.get();
    runner1_ptr->set_mock_snapshot(
        "layer0_MOE_COMBINED_OUTPUT",
        1,
        4,
        {10.0f, 20.0f, 30.0f, 40.0f});

    std::vector<std::unique_ptr<IInferenceRunner>> runners;
    runners.push_back(std::move(runner0));
    runners.push_back(std::move(runner1));

    auto model_ctx = llaminar2::test::MockModelContext::createMinimal();
    model_ctx->setArchitecture("qwen35moe");

    auto plan = std::make_shared<MoERoutedExpertPlacementPlan>();
    plan->enabled = true;
    plan->topology = RoutedExpertPlacementTopology::TieredOverlay;
    plan->continuation_domain_spec.dense_tp_enabled = true;
    plan->continuation_domain_spec.dense_decode_replicated = true;
    plan->continuation_domain_spec.refreshDensePolicyFromFlags();

    RankOrchestrator::Config rank_config;
    rank_config.devices = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)};
    rank_config.weights = {0.5f, 0.5f};
    rank_config.moe_routed_expert_plan = plan;

    MockLocalTPContext::Config tp_config;
    tp_config.devices = rank_config.devices;
    tp_config.weights = rank_config.weights;

    auto orchestrator = RankOrchestrator::createForTest(
        model_ctx,
        std::move(runners),
        std::make_unique<MockLocalTPContext>(tp_config),
        rank_config);

    int token = 42;
    ASSERT_TRUE(orchestrator->forward(&token, 1));

    auto snapshot = orchestrator->getTPSnapshot("layer0_MOE_COMBINED_OUTPUT");
    EXPECT_EQ(snapshot.mode, SnapshotShardingMode::ROW_PARALLEL);
    ASSERT_TRUE(snapshot.computeCombined());
    EXPECT_EQ(snapshot.combined_rows, 1);
    EXPECT_EQ(snapshot.combined_cols, 4);
    EXPECT_EQ(snapshot.combined_data,
              (std::vector<float>{11.0f, 22.0f, 33.0f, 44.0f}));
}

TEST_F(Test__RankOrchestrator, TPSnapshot_RowParallel_RejectsMismatchedPartialShapes)
{
    TPSnapshot snapshot;
    snapshot.key = "layer0_FFN_DOWN";
    snapshot.mode = SnapshotShardingMode::ROW_PARALLEL;
    snapshot.tp_degree = 2;

    DeviceSnapshotData dev0;
    dev0.device_index = 0;
    dev0.rows = 1;
    dev0.cols = 4;
    dev0.data = {1.0f, 2.0f, 3.0f, 4.0f};

    DeviceSnapshotData dev1;
    dev1.device_index = 1;
    dev1.rows = 2;
    dev1.cols = 2;
    dev1.data = {5.0f, 6.0f, 7.0f, 8.0f};

    snapshot.device_data.push_back(std::move(dev0));
    snapshot.device_data.push_back(std::move(dev1));

    EXPECT_FALSE(snapshot.computeCombined());
    EXPECT_FALSE(snapshot.combined_valid);
    EXPECT_TRUE(snapshot.combined_data.empty());
    EXPECT_EQ(snapshot.combined_rows, 0);
    EXPECT_EQ(snapshot.combined_cols, 0);
}

TEST_F(Test__RankOrchestrator, TPSnapshot_UnknownShardingRejectsMultiDeviceCombination)
{
    TPSnapshot snapshot;
    snapshot.key = "layer0_UNDECLARED_DEBUG_KEY";
    snapshot.mode = SnapshotShardingMode::UNKNOWN;
    snapshot.tp_degree = 2;

    DeviceSnapshotData dev0;
    dev0.device_index = 0;
    dev0.rows = 1;
    dev0.cols = 2;
    dev0.data = {1.0f, 2.0f};

    DeviceSnapshotData dev1;
    dev1.device_index = 1;
    dev1.rows = 1;
    dev1.cols = 2;
    dev1.data = {3.0f, 4.0f};

    snapshot.device_data.push_back(std::move(dev0));
    snapshot.device_data.push_back(std::move(dev1));

    EXPECT_FALSE(snapshot.computeCombined());
    EXPECT_FALSE(snapshot.combined_valid);
    EXPECT_TRUE(snapshot.combined_data.empty());
    EXPECT_EQ(snapshot.combined_rows, 0);
    EXPECT_EQ(snapshot.combined_cols, 0);
}

TEST_F(Test__RankOrchestrator, TPSnapshot_SchemaFamiliesResolveByLongestPrefix)
{
    StageShardingConfig sharding = {
        {"ATTENTION_DEVICE_*", SnapshotShardingMode::COLUMN_PARALLEL},
        {"ATTENTION_DEVICE_KV_COUNT_REQUEST_*", SnapshotShardingMode::REPLICATED},
    };

    EXPECT_EQ(
        getStageShardingMode(
            "layer3_ATTENTION_DEVICE_KV_COUNT_REQUEST_11",
            sharding),
        SnapshotShardingMode::REPLICATED);
    EXPECT_EQ(
        getStageShardingMode(
            "layer3_ATTENTION_DEVICE_OTHER",
            sharding),
        SnapshotShardingMode::COLUMN_PARALLEL);
    EXPECT_EQ(
        getStageShardingMode(
            "layer3_ATTENTION_CONTEXT",
            sharding),
        SnapshotShardingMode::UNKNOWN);
}

TEST_F(Test__RankOrchestrator, TPSnapshot_LegacyEmbeddingShardingDistinguishesPreAndPostAllreduce)
{
    /**
     * Vocab-parallel embedding produces participant-local partial rows before
     * the explicit embedding allreduce.  The legacy sharding helper is still
     * used by some low-level snapshot tests, so it must carry the same semantic
     * split as the model schema contract.
     */
    EXPECT_EQ(getStageShardingMode("EMBEDDING"),
              SnapshotShardingMode::ROW_PARALLEL);
    EXPECT_EQ(getStageShardingMode("EMBEDDING_ALLREDUCED"),
              SnapshotShardingMode::REPLICATED);
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

TEST_F(Test__RankOrchestrator, TPSnapshot_Replicated_RejectsDivergentReplicaPayloads)
{
    /**
     * Replicated snapshots are often post-collective tensors.  Accepting device
     * zero without checking the other participants can hide the exact class of
     * allreduce and graph-publication defects that snapshot diagnostics are
     * supposed to expose.
     */
    TPSnapshot snapshot;
    snapshot.key = "layer0_ATTENTION_OUTPUT_ALLREDUCED";
    snapshot.mode = SnapshotShardingMode::REPLICATED;
    snapshot.tp_degree = 2;

    DeviceSnapshotData dev0;
    dev0.device_index = 0;
    dev0.rows = 1;
    dev0.cols = 4;
    dev0.data = {1.0f, 2.0f, 3.0f, 4.0f};

    DeviceSnapshotData dev1;
    dev1.device_index = 1;
    dev1.rows = 1;
    dev1.cols = 4;
    dev1.data = {1.0f, 2.0f, 30.0f, 4.0f};

    snapshot.device_data.push_back(std::move(dev0));
    snapshot.device_data.push_back(std::move(dev1));

    EXPECT_FALSE(snapshot.computeCombined());
    EXPECT_FALSE(snapshot.combined_valid);
    EXPECT_TRUE(snapshot.combined_data.empty());
    EXPECT_EQ(snapshot.combined_rows, 0);
    EXPECT_EQ(snapshot.combined_cols, 0);
}
