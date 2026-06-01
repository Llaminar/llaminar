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
#include "config/OrchestrationConfig.h"
#include "execution/mpi_orchestration/RankExecutionPlan.h"
#include "backends/GlobalDeviceAddress.h"
#include "tensors/Tensors.h"
#include "utils/PerfStatsCollector.h"
#include "../../../mocks/MockModelContext.h"
#include "../../../mocks/MockMPIContext.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
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
        }

        ~ScopedEnv()
        {
            if (had_old_)
                setenv(name_.c_str(), old_value_.c_str(), 1);
            else
                unsetenv(name_.c_str());
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

        const float *mtpLogits() const override
        {
            if (hide_local_logits_)
                return nullptr;
            if (column_parallel_logits_)
                return nullptr;
            return mtp_logits_.empty() ? nullptr : mtp_logits_.data();
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

        bool supportsMTPTokenCoordination() const override
        {
            return supports_mtp_token_coordination_;
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

        // =====================================================================
        // Test inspection methods
        // =====================================================================

        int forwardCallCount() const { return forward_call_count_; }
        int clearCacheCount() const { return clear_cache_count_; }
        int forwardMTPCount() const { return forward_mtp_count_; }
        int restoreCount() const { return restore_count_; }
        int setAllPositionCount() const { return set_all_position_count_; }
        int lastMTPConditionToken() const { return last_mtp_condition_token_; }
        int sampleMainLogitsCount() const { return sample_main_logits_count_; }
        int sampleMTPLogitsCount() const { return sample_mtp_logits_count_; }
        int sampleAllPositionLogitsCount() const { return sample_all_position_logits_count_; }
        const PrefixStateSnapshot &lastRestoredSnapshot() const { return last_restored_snapshot_; }
        const std::vector<int> &lastForwardTokens() const { return last_forward_tokens_; }
        int lastForwardSeqLen() const { return last_forward_seq_len_; }
        void enableMTP(bool accept_mtp_token)
        {
            mtp_enabled_ = true;
            accept_mtp_token_ = accept_mtp_token;
        }
        void setMTPUnsupportedReason(std::string reason)
        {
            mtp_unsupported_reason_ = std::move(reason);
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
        void setCapturedSnapshot(PrefixStateSnapshot snapshot)
        {
            captured_snapshot_ = std::move(snapshot);
            use_captured_snapshot_ = true;
        }
        PrefixStateSnapshot captureLivePrefixState(int seq_idx = 0) const override
        {
            (void)seq_idx;
            if (use_captured_snapshot_)
            {
                PrefixStateSnapshot snapshot = captured_snapshot_;
                snapshot.cached_tokens = position_;
                return snapshot;
            }
            PrefixStateSnapshot snapshot;
            snapshot.valid = mtp_enabled_;
            snapshot.cached_tokens = position_;
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
            logits_[DECODE_ARGMAX_TOKEN] = 10.0f; // Token 3 is argmax
            if (column_parallel_logits_)
            {
                resetLocalTensor(logits_local_, 1);
                setLocalToken(logits_local_, 0, DECODE_ARGMAX_TOKEN, 10.0f);
            }
        }

        void setupAllPositionLogits(int seq_len)
        {
            all_position_logits_.assign(static_cast<size_t>(seq_len) * VOCAB_SIZE, -10.0f);
            const int row0_token = accept_mtp_token_ ? MTP_ARGMAX_TOKEN : VERIFY_REJECT_TOKEN;
            all_position_logits_[row0_token] = 10.0f;
            if (seq_len > 1)
            {
                all_position_logits_[VOCAB_SIZE + DECODE_ARGMAX_TOKEN] = 10.0f;
            }
            if (column_parallel_logits_)
            {
                resetLocalTensor(all_position_logits_local_, seq_len);
                setLocalToken(all_position_logits_local_, 0, row0_token, 10.0f);
                if (seq_len > 1)
                {
                    setLocalToken(all_position_logits_local_, 1, DECODE_ARGMAX_TOKEN, 10.0f);
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
        int clear_cache_count_{0};
        int restore_count_{0};
        int set_all_position_count_{0};
        int sample_main_logits_count_{0};
        int sample_mtp_logits_count_{0};
        int sample_all_position_logits_count_{0};
        int last_mtp_condition_token_{-1};
        bool is_first_forward_in_cycle_{true};
        bool mtp_enabled_{false};
        bool accept_mtp_token_{true};
        bool all_position_logits_enabled_{false};
        bool column_parallel_logits_{false};
        bool supports_mtp_token_coordination_{false};
        bool hide_local_logits_{false};
        bool use_captured_snapshot_{false};
        int vocab_start_{0};
        int vocab_local_{VOCAB_SIZE};
        std::string mtp_unsupported_reason_;
        PrefixStateSnapshot captured_snapshot_;
        PrefixStateSnapshot last_restored_snapshot_;
        std::vector<int> last_forward_tokens_;
        int last_forward_seq_len_{0};
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
                                                                             bool hide_local_logits = false)
        {
            auto mock = std::make_unique<MockInferenceRunner>();
            auto *mock_ptr = mock.get(); // Keep raw pointer for inspection
            if (mtp_enabled)
            {
                mock_ptr->enableMTP(mtp_accept);
            }
            mock_ptr->setMTPUnsupportedReason(std::move(mtp_unsupported_reason));
            if (mtp_token_coordination)
            {
                mock_ptr->enableMTPTokenCoordination(hide_local_logits);
            }

            OrchestrationConfig config;
            config.device_for_this_rank = GlobalDeviceAddress::cpu();
            config.mtp.enabled = mtp_enabled;
            config.mtp.draft_tokens = 1;
            config.mtp.verify_mode = MTPVerifyMode::Greedy;

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
                                                 bool column_parallel_logits = false)
        {
            auto child0 = std::make_unique<MockInferenceRunner>();
            auto child1 = std::make_unique<MockInferenceRunner>();
            child0->enableMTP(mtp_accept);
            child1->enableMTP(mtp_accept);
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
            rank_config.devices = {GlobalDeviceAddress::cpu(), GlobalDeviceAddress::cpu()};
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
            config.device_for_this_rank = GlobalDeviceAddress::cpu();
            config.mtp.enabled = true;
            config.mtp.draft_tokens = 1;
            config.mtp.verify_mode = MTPVerifyMode::Greedy;

            auto runner = std::make_unique<OrchestrationRunner>(
                std::move(config), plan_, std::move(rank_runner));

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

    TEST_F(Test__PrefillDecodeTransition, MTPFirstDecodeAcceptsGreedyDraftAndReplaysFromPrefillCheckpoint)
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
        EXPECT_EQ(mock->restoreCount(), 1);
        EXPECT_GE(mock->setAllPositionCount(), 2);
        EXPECT_THAT(mock->lastForwardTokens(),
                    ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN,
                                MockInferenceRunner::MTP_ARGMAX_TOKEN));

        const auto probe = runner->prefixStateProbe();
        EXPECT_EQ(probe.mtp_draft_steps, 1u);
        EXPECT_EQ(probe.mtp_verifier_runs, 1u);
        EXPECT_EQ(probe.mtp_verifier_token_count, 2u);
        EXPECT_EQ(probe.mtp_rejected_tokens, 0u);
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

            const auto records = PerfStatsCollector::snapshot({"mtp"});
            const PerfStatRecord *step_calls =
                findPerfRecord(records, PerfStatRecord::Kind::Counter, "decode_step_calls");
            ASSERT_NE(step_calls, nullptr);
            EXPECT_DOUBLE_EQ(step_calls->value, 1.0);

            const PerfStatRecord *capture_timer =
                findPerfRecord(records, PerfStatRecord::Kind::Timer, "capture_live_prefix_state");
            ASSERT_NE(capture_timer, nullptr);
            EXPECT_EQ(capture_timer->count, 1u);

            const PerfStatRecord *sidecar_timer =
                findPerfRecord(records, PerfStatRecord::Kind::Timer, "sidecar_forward");
            ASSERT_NE(sidecar_timer, nullptr);
            EXPECT_EQ(sidecar_timer->count, 1u);

            const PerfStatRecord *verifier_timer =
                findPerfRecord(records, PerfStatRecord::Kind::Timer, "verifier_forward");
            ASSERT_NE(verifier_timer, nullptr);
            EXPECT_EQ(verifier_timer->count, 1u);
        }
        std::filesystem::remove(export_path);
        PerfStatsCollector::reset();
    }

    TEST_F(Test__PrefillDecodeTransition, MTPSecondDecodeUsesReplayTerminalLogitsWithoutRefeedingPreviousToken)
    {
        auto [runner, mock] = createRunner(/*mtp_enabled=*/true, /*mtp_accept=*/true);

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
                    ElementsAre(MockInferenceRunner::DECODE_ARGMAX_TOKEN,
                                MockInferenceRunner::MTP_ARGMAX_TOKEN));
    }

    TEST_F(Test__PrefillDecodeTransition, MTPFirstDecodeForcedRejectReplaysVerifiedToken)
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
        EXPECT_THAT(mock->lastForwardTokens(),
                    ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN,
                                MockInferenceRunner::VERIFY_REJECT_TOKEN));

        const auto probe = runner->prefixStateProbe();
        EXPECT_EQ(probe.mtp_draft_steps, 1u);
        EXPECT_EQ(probe.mtp_verifier_runs, 1u);
        EXPECT_EQ(probe.mtp_verifier_token_count, 2u);
        EXPECT_EQ(probe.mtp_rejected_tokens, 1u);
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
        EXPECT_EQ(mock->sampleAllPositionLogitsCount(), 1);

        auto probe = runner->prefixStateProbe();
        EXPECT_FALSE(probe.mtp_bypassed);
        EXPECT_EQ(probe.mtp_draft_steps, 1u);
        EXPECT_EQ(probe.mtp_accepted_tokens, 2u);
        EXPECT_EQ(probe.mtp_rollbacks, 1u);
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
        EXPECT_EQ(child_ptr->restoreCount(), 1);

        auto probe = runner->prefixStateProbe();
        EXPECT_FALSE(probe.mtp_bypassed);
        EXPECT_EQ(probe.mtp_draft_steps, 1u);
        EXPECT_EQ(probe.mtp_accepted_tokens, 2u);
        EXPECT_EQ(probe.mtp_rollbacks, 1u);
    }

    TEST_F(Test__PrefillDecodeTransition, LocalTPMTPDecodeRunsEveryParticipantAndRecordsOneRollback)
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
        EXPECT_GE(harness.child0->setAllPositionCount(), 2);
        EXPECT_GE(harness.child1->setAllPositionCount(), 2);
        EXPECT_THAT(harness.child0->lastForwardTokens(),
                    ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN,
                                MockInferenceRunner::MTP_ARGMAX_TOKEN));
        EXPECT_THAT(harness.child1->lastForwardTokens(),
                    ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN,
                                MockInferenceRunner::MTP_ARGMAX_TOKEN));

        const auto probe = harness.runner->prefixStateProbe();
        EXPECT_EQ(probe.mtp_draft_steps, 1u);
        EXPECT_EQ(probe.mtp_verifier_runs, 1u);
        EXPECT_EQ(probe.mtp_verifier_token_count, 2u);
        EXPECT_EQ(probe.mtp_accepted_tokens, 2u);
        EXPECT_EQ(probe.mtp_rejected_tokens, 0u);
        EXPECT_EQ(probe.mtp_rollbacks, 1u);
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
        EXPECT_THAT(harness.child0->lastForwardTokens(),
                    ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN,
                                MockInferenceRunner::VERIFY_REJECT_TOKEN));
        EXPECT_THAT(harness.child1->lastForwardTokens(),
                    ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN,
                                MockInferenceRunner::VERIFY_REJECT_TOKEN));

        const auto probe = harness.runner->prefixStateProbe();
        EXPECT_EQ(probe.mtp_draft_steps, 1u);
        EXPECT_EQ(probe.mtp_verifier_runs, 1u);
        EXPECT_EQ(probe.mtp_verifier_token_count, 2u);
        EXPECT_EQ(probe.mtp_accepted_tokens, 2u);
        EXPECT_EQ(probe.mtp_rejected_tokens, 1u);
        EXPECT_EQ(probe.mtp_rollbacks, 1u);
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
                    ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN,
                                MockInferenceRunner::MTP_ARGMAX_TOKEN));
        EXPECT_THAT(harness.child1->lastForwardTokens(),
                    ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN,
                                MockInferenceRunner::MTP_ARGMAX_TOKEN));

        const auto probe = harness.runner->prefixStateProbe();
        EXPECT_EQ(probe.mtp_draft_steps, 1u);
        EXPECT_EQ(probe.mtp_verifier_runs, 1u);
        EXPECT_EQ(probe.mtp_verifier_token_count, 2u);
        EXPECT_EQ(probe.mtp_accepted_tokens, 2u);
        EXPECT_EQ(probe.mtp_rejected_tokens, 0u);
        EXPECT_EQ(probe.mtp_rollbacks, 1u);
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
        EXPECT_THAT(harness.child0->lastForwardTokens(),
                    ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN,
                                MockInferenceRunner::VERIFY_REJECT_TOKEN));
        EXPECT_THAT(harness.child1->lastForwardTokens(),
                    ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN,
                                MockInferenceRunner::VERIFY_REJECT_TOKEN));

        const auto probe = harness.runner->prefixStateProbe();
        EXPECT_EQ(probe.mtp_draft_steps, 1u);
        EXPECT_EQ(probe.mtp_verifier_runs, 1u);
        EXPECT_EQ(probe.mtp_verifier_token_count, 2u);
        EXPECT_EQ(probe.mtp_accepted_tokens, 2u);
        EXPECT_EQ(probe.mtp_rejected_tokens, 1u);
        EXPECT_EQ(probe.mtp_rollbacks, 1u);
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
        auto [runner, mock] = createRunner(/*mtp_enabled=*/true, /*mtp_accept=*/true);

        SamplingParams greedy;
        greedy.temperature = 0.0f;

        std::vector<int32_t> prompt = {1, 2, 3};
        GenerationResult result = runner->generate(prompt, 1, greedy);

        ASSERT_TRUE(result.success()) << result.error;
        EXPECT_THAT(result.tokens,
                    ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN));
        EXPECT_EQ(mock->forwardMTPCount(), 1);
        EXPECT_THAT(mock->lastForwardTokens(),
                    ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN));

        const auto probe = runner->prefixStateProbe();
        EXPECT_EQ(probe.mtp_draft_steps, 1u);
        EXPECT_EQ(probe.mtp_accepted_tokens, 1u);
        EXPECT_EQ(probe.mtp_rollbacks, 1u);
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
        EXPECT_EQ(mock->forwardMTPCount(), 1);
        EXPECT_THAT(mock->lastForwardTokens(),
                    ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN));

        const auto probe = runner->prefixStateProbe();
        EXPECT_EQ(probe.mtp_draft_steps, 1u);
        EXPECT_EQ(probe.mtp_accepted_tokens, 1u);
        EXPECT_EQ(probe.mtp_rollbacks, 1u);
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
                    ElementsAre(MockInferenceRunner::PREFILL_ARGMAX_TOKEN,
                                MockInferenceRunner::MTP_ARGMAX_TOKEN));
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
