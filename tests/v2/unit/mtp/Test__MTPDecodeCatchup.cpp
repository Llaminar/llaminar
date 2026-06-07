#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "execution/mtp/MTPDecodeCatchup.h"
#include "execution/local_execution/orchestrators/IInferenceRunner.h"
#include "utils/PerfStatsCollector.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

using namespace llaminar2;
using namespace testing;

namespace
{
    class FakeCatchupRunner : public IInferenceRunner
    {
    public:
        bool forward(const int *tokens, int seq_len) override
        {
            if (!tokens || seq_len != 1)
                return false;
            ++forward_count;
            forward_tokens.push_back(tokens[0]);
            ++position;
            return forward_ok;
        }

        const float *logits() const override { return nullptr; }
        int vocab_size() const override { return 16; }
        void clear_cache() override
        {
            position = 0;
            forward_tokens.clear();
            committed_tokens.clear();
            committed_indices.clear();
        }
        int get_position() const override { return position; }
        ExecutionPath executionPath() const override { return ExecutionPath::GRAPH; }
        const char *architecture() const override { return "fake-catchup"; }

        bool commitMTPShiftedRowFromCurrentTerminalHidden(
            int32_t token,
            int already_appended_tokens,
            bool allow_speculative_discard = false,
            int position_offset_override = -1) override
        {
            if (!commit_ok)
                return false;
            ++commit_count;
            committed_tokens.push_back(token);
            committed_indices.push_back(already_appended_tokens);
            committed_allow_discard.push_back(allow_speculative_discard);
            committed_position_offsets.push_back(position_offset_override);
            return true;
        }

        bool forward_ok = true;
        bool commit_ok = true;
        int position = 0;
        int forward_count = 0;
        int commit_count = 0;
        std::vector<int> forward_tokens;
        std::vector<int32_t> committed_tokens;
        std::vector<int> committed_indices;
        std::vector<bool> committed_allow_discard;
        std::vector<int> committed_position_offsets;
    };

    class ScriptedSampler
    {
    public:
        explicit ScriptedSampler(std::vector<int32_t> tokens)
            : tokens_(std::move(tokens))
        {
        }

        int32_t operator()()
        {
            if (index_ >= tokens_.size())
                return -1;
            return tokens_[index_++];
        }

        size_t calls() const { return index_; }

    private:
        std::vector<int32_t> tokens_;
        size_t index_ = 0;
    };

    class ScopedEnv
    {
    public:
        ScopedEnv(const char *name, const char *value)
            : name_(name),
              had_old_(std::getenv(name) != nullptr),
              old_value_(had_old_ ? std::getenv(name) : "")
        {
            setenv(name_.c_str(), value, 1);
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

} // namespace

TEST(Test__MTPDecodeCatchup, SharedStepwiseAcceptsMultiRowDraftAndReturnsReadyToken)
{
    FakeCatchupRunner runner;
    ScriptedSampler sampler({9, 8, 6, 4});

    MTPDecodeCatchupGreedyRequest request;
    request.draft_tokens = {7, 9, 8, 6};
    request.base_sidecar_position = 42;

    MTPDecodeCatchupGreedyResult result =
        runSharedStepwiseMTPDecodeCatchupGreedy(
            runner,
            request,
            [&]() { return sampler(); });

    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_THAT(result.accepted_tokens, ElementsAre(7, 9, 8, 6));
    EXPECT_THAT(result.verifier_tokens, ElementsAre(9, 8, 6));
    EXPECT_TRUE(result.all_speculative_accepted);
    EXPECT_EQ(result.accepted_speculative_prefix, 3);
    EXPECT_EQ(result.ready_token, 4);
    EXPECT_EQ(result.main_forward_token_count, 4);
    EXPECT_EQ(result.shifted_commit_count, 4);
    EXPECT_THAT(runner.forward_tokens, ElementsAre(7, 9, 8, 6));
    EXPECT_THAT(runner.committed_tokens, ElementsAre(7, 9, 8, 6));
    EXPECT_THAT(runner.committed_indices, ElementsAre(0, 1, 2, 3));
    EXPECT_THAT(runner.committed_position_offsets, ElementsAre(42, 42, 42, 42));
    EXPECT_TRUE(std::all_of(
        runner.committed_allow_discard.begin(),
        runner.committed_allow_discard.end(),
        [](bool v) { return v; }));
}

TEST(Test__MTPDecodeCatchup, SharedStepwiseUsesRequestImplementationNameForCounters)
{
    ScopedEnv enable_perf_stats("LLAMINAR_PERF_STATS_JSON", "1");
    PerfStatsCollector::reset();
    FakeCatchupRunner runner;
    ScriptedSampler sampler({9, 4});

    MTPDecodeCatchupGreedyRequest request;
    request.draft_tokens = {7, 9};
    request.implementation_name = "device_graph_stepwise";

    MTPDecodeCatchupGreedyResult result =
        runSharedStepwiseMTPDecodeCatchupGreedy(
            runner,
            request,
            [&]() { return sampler(); });

    ASSERT_TRUE(result.ok) << result.error;

    const auto records = PerfStatsCollector::snapshot({"mtp"});
    bool found_custom_run_counter = false;
    for (const auto &record : records)
    {
        if (record.name != "decode_equivalent_catchup_runs")
            continue;
        auto it = record.tags.find("implementation");
        if (it != record.tags.end() && it->second == "device_graph_stepwise")
        {
            found_custom_run_counter = true;
            break;
        }
    }
    EXPECT_TRUE(found_custom_run_counter);
    PerfStatsCollector::reset();
}

TEST(Test__MTPDecodeCatchup, SharedStepwiseRejectsAndForwardsCorrectionForReadyToken)
{
    FakeCatchupRunner runner;
    ScriptedSampler sampler({9, 3, 5});

    MTPDecodeCatchupGreedyRequest request;
    request.draft_tokens = {7, 9, 9};
    request.base_sidecar_position = 5;

    MTPDecodeCatchupGreedyResult result =
        runSharedStepwiseMTPDecodeCatchupGreedy(
            runner,
            request,
            [&]() { return sampler(); });

    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_THAT(result.accepted_tokens, ElementsAre(7, 9, 3));
    EXPECT_THAT(result.verifier_tokens, ElementsAre(9, 3));
    EXPECT_FALSE(result.all_speculative_accepted);
    EXPECT_EQ(result.accepted_speculative_prefix, 1);
    EXPECT_EQ(result.rejected_verified_token, 3);
    EXPECT_EQ(result.ready_token, 5);
    EXPECT_EQ(result.main_forward_token_count, 3);
    EXPECT_THAT(runner.forward_tokens, ElementsAre(7, 9, 3));
    EXPECT_THAT(runner.committed_tokens, ElementsAre(7, 9, 3));
    EXPECT_THAT(runner.committed_indices, ElementsAre(0, 1, 2));
}

TEST(Test__MTPDecodeCatchup, EquivalenceAllowsCandidateWithFewerMainForwards)
{
    MTPDecodeCatchupGreedyResult oracle;
    oracle.ok = true;
    oracle.accepted_tokens = {7, 9, 8};
    oracle.verifier_tokens = {9, 8};
    oracle.all_speculative_accepted = true;
    oracle.stopped_on_output = false;
    oracle.accepted_speculative_prefix = 2;
    oracle.ready_token = 4;
    oracle.main_forward_token_count = 3;
    oracle.shifted_commit_count = 3;

    MTPDecodeCatchupGreedyResult candidate = oracle;
    candidate.main_forward_token_count = 1;

    MTPDecodeCatchupGreedyEquivalence eq =
        compareMTPDecodeCatchupGreedyResults(oracle, candidate);
    EXPECT_TRUE(eq.ok) << eq.error;
}

TEST(Test__MTPDecodeCatchup, EquivalenceRejectsCommittedTokenDrift)
{
    MTPDecodeCatchupGreedyResult oracle;
    oracle.ok = true;
    oracle.accepted_tokens = {7, 9, 8};
    oracle.verifier_tokens = {9, 8};
    oracle.all_speculative_accepted = true;
    oracle.accepted_speculative_prefix = 2;
    oracle.ready_token = 4;
    oracle.shifted_commit_count = 3;

    MTPDecodeCatchupGreedyResult candidate = oracle;
    candidate.accepted_tokens = {7, 9, 3};

    MTPDecodeCatchupGreedyEquivalence eq =
        compareMTPDecodeCatchupGreedyResults(oracle, candidate);
    EXPECT_FALSE(eq.ok);
    EXPECT_THAT(eq.error, HasSubstr("accepted tokens mismatch"));
}

TEST(Test__MTPDecodeCatchup, EquivalenceRejectsReadyTokenDrift)
{
    MTPDecodeCatchupGreedyResult oracle;
    oracle.ok = true;
    oracle.accepted_tokens = {7, 9};
    oracle.verifier_tokens = {9};
    oracle.all_speculative_accepted = true;
    oracle.accepted_speculative_prefix = 1;
    oracle.ready_token = 4;
    oracle.shifted_commit_count = 2;

    MTPDecodeCatchupGreedyResult candidate = oracle;
    candidate.ready_token = 5;

    MTPDecodeCatchupGreedyEquivalence eq =
        compareMTPDecodeCatchupGreedyResults(oracle, candidate);
    EXPECT_FALSE(eq.ok);
    EXPECT_THAT(eq.error, HasSubstr("ready token mismatch"));
}

TEST(Test__MTPDecodeCatchup, EquivalenceRejectsShiftedCommitCountDrift)
{
    MTPDecodeCatchupGreedyResult oracle;
    oracle.ok = true;
    oracle.accepted_tokens = {7, 77};
    oracle.verifier_tokens = {77};
    oracle.all_speculative_accepted = false;
    oracle.accepted_speculative_prefix = 0;
    oracle.rejected_verified_token = 77;
    oracle.ready_token = 5;
    oracle.shifted_commit_count = 2;

    MTPDecodeCatchupGreedyResult candidate = oracle;
    candidate.shifted_commit_count = 1;

    MTPDecodeCatchupGreedyEquivalence eq =
        compareMTPDecodeCatchupGreedyResults(oracle, candidate);
    EXPECT_FALSE(eq.ok);
    EXPECT_THAT(eq.error, HasSubstr("shifted MTP commit count mismatch"));
}

TEST(Test__MTPDecodeCatchup, BuildsVerifierRowsResultForAcceptAll)
{
    MTPDecodeCatchupGreedyRequest request;
    request.draft_tokens = {7, 9, 8};

    MTPDecodeCatchupGreedyResult result =
        buildMTPDecodeCatchupGreedyResultFromVerifierRows(
            request,
            /*sampled_verifier_tokens=*/{9, 8, 4});

    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_THAT(result.accepted_tokens, ElementsAre(7, 9, 8));
    EXPECT_THAT(result.verifier_tokens, ElementsAre(9, 8));
    EXPECT_TRUE(result.all_speculative_accepted);
    EXPECT_EQ(result.accepted_speculative_prefix, 2);
    EXPECT_EQ(result.ready_token, 4);
    EXPECT_EQ(result.main_forward_token_count, 3);
    EXPECT_EQ(result.shifted_commit_count, 3);
    EXPECT_EQ(result.target_verifier_state_commit_count, 3);
}

TEST(Test__MTPDecodeCatchup, BuildsVerifierRowsResultForRejectAfterPrefix)
{
    MTPDecodeCatchupGreedyRequest request;
    request.draft_tokens = {7, 9, 8};

    MTPDecodeCatchupGreedyResult result =
        buildMTPDecodeCatchupGreedyResultFromVerifierRows(
            request,
            /*sampled_verifier_tokens=*/{9, 3, 4});

    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_THAT(result.accepted_tokens, ElementsAre(7, 9, 3));
    EXPECT_THAT(result.verifier_tokens, ElementsAre(9, 3));
    EXPECT_FALSE(result.all_speculative_accepted);
    EXPECT_EQ(result.accepted_speculative_prefix, 1);
    EXPECT_EQ(result.rejected_verified_token, 3);
    EXPECT_EQ(result.ready_token, -1)
        << "the correction token has not been forwarded by the target verifier rows";
    EXPECT_EQ(result.main_forward_token_count, 3);
    EXPECT_EQ(result.shifted_commit_count, 3);
    EXPECT_EQ(result.target_verifier_state_commit_count, 2)
        << "state is valid through first token plus accepted speculative prefix only";
}

TEST(Test__MTPDecodeCatchup, BuildsVerifierRowsResultWhenFirstDraftStops)
{
    MTPDecodeCatchupGreedyRequest request;
    request.draft_tokens = {7, 9, 8};
    request.stop_tokens = {7};

    MTPDecodeCatchupGreedyResult result =
        buildMTPDecodeCatchupGreedyResultFromVerifierRows(
            request,
            /*sampled_verifier_tokens=*/{-1, -1, -1});

    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_THAT(result.accepted_tokens, ElementsAre(7));
    EXPECT_TRUE(result.verifier_tokens.empty());
    EXPECT_TRUE(result.stopped_on_output);
    EXPECT_TRUE(result.all_speculative_accepted)
        << "no speculative row was rejected before the stop";
    EXPECT_EQ(result.accepted_speculative_prefix, 0);
    EXPECT_EQ(result.ready_token, -1);
    EXPECT_EQ(result.main_forward_token_count, 3)
        << "the promoted verifier graph still ran the fixed target row count";
    EXPECT_EQ(result.shifted_commit_count, 1);
    EXPECT_EQ(result.target_verifier_state_commit_count, 1);
}

TEST(Test__MTPDecodeCatchup, BuildsVerifierRowsResultWhenAcceptedSpeculativeTokenStops)
{
    MTPDecodeCatchupGreedyRequest request;
    request.draft_tokens = {7, 9, 8};
    request.stop_tokens = {9};

    MTPDecodeCatchupGreedyResult result =
        buildMTPDecodeCatchupGreedyResultFromVerifierRows(
            request,
            /*sampled_verifier_tokens=*/{9, -1, -1});

    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_THAT(result.accepted_tokens, ElementsAre(7, 9));
    EXPECT_THAT(result.verifier_tokens, ElementsAre(9));
    EXPECT_TRUE(result.stopped_on_output);
    EXPECT_TRUE(result.all_speculative_accepted)
        << "a stop after an accepted speculative token is not a rejection";
    EXPECT_EQ(result.accepted_speculative_prefix, 1);
    EXPECT_EQ(result.ready_token, -1);
    EXPECT_EQ(result.main_forward_token_count, 3);
    EXPECT_EQ(result.shifted_commit_count, 2);
    EXPECT_EQ(result.target_verifier_state_commit_count, 2);
}

TEST(Test__MTPDecodeCatchup, BuildsVerifierRowsResultWhenRejectedCorrectionStops)
{
    MTPDecodeCatchupGreedyRequest request;
    request.draft_tokens = {7, 9, 8};
    request.stop_tokens = {3};

    MTPDecodeCatchupGreedyResult result =
        buildMTPDecodeCatchupGreedyResultFromVerifierRows(
            request,
            /*sampled_verifier_tokens=*/{3, -1, -1});

    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_THAT(result.accepted_tokens, ElementsAre(7, 3));
    EXPECT_THAT(result.verifier_tokens, ElementsAre(3));
    EXPECT_TRUE(result.stopped_on_output);
    EXPECT_FALSE(result.all_speculative_accepted);
    EXPECT_EQ(result.accepted_speculative_prefix, 0);
    EXPECT_EQ(result.rejected_verified_token, 3);
    EXPECT_EQ(result.ready_token, -1);
    EXPECT_EQ(result.main_forward_token_count, 3);
    EXPECT_EQ(result.shifted_commit_count, 2);
    EXPECT_EQ(result.target_verifier_state_commit_count, 1)
        << "the stopped correction token has not been forwarded as live state";
}

TEST(Test__MTPDecodeCatchup, RejectsVerifierRowsAcceptAllWithoutReadyToken)
{
    MTPDecodeCatchupGreedyRequest request;
    request.draft_tokens = {7, 9, 8};

    MTPDecodeCatchupGreedyResult result =
        buildMTPDecodeCatchupGreedyResultFromVerifierRows(
            request,
            /*sampled_verifier_tokens=*/{9, 8, -1});

    EXPECT_FALSE(result.ok);
    EXPECT_THAT(result.error, HasSubstr("ready token"));
}

TEST(Test__MTPDecodeCatchup, SharedStepwiseStopTokenDiscardsReadyToken)
{
    FakeCatchupRunner runner;
    ScriptedSampler sampler({9, 5});

    MTPDecodeCatchupGreedyRequest request;
    request.draft_tokens = {7, 9, 8};
    request.stop_tokens = {9};

    MTPDecodeCatchupGreedyResult result =
        runSharedStepwiseMTPDecodeCatchupGreedy(
            runner,
            request,
            [&]() { return sampler(); });

    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_THAT(result.accepted_tokens, ElementsAre(7, 9));
    EXPECT_THAT(result.verifier_tokens, ElementsAre(9));
    EXPECT_TRUE(result.stopped_on_output);
    EXPECT_EQ(result.ready_token, -1);
    EXPECT_EQ(result.main_forward_token_count, 2)
        << "the accepted stop output is still forwarded before completion";
    EXPECT_THAT(runner.forward_tokens, ElementsAre(7, 9));
    EXPECT_THAT(runner.committed_tokens, ElementsAre(7, 9));
}

TEST(Test__MTPDecodeCatchup, SharedStepwiseHardFailsWithoutDraftTokens)
{
    FakeCatchupRunner runner;
    ScriptedSampler sampler({1});

    MTPDecodeCatchupGreedyRequest request;
    MTPDecodeCatchupGreedyResult result =
        runSharedStepwiseMTPDecodeCatchupGreedy(
            runner,
            request,
            [&]() { return sampler(); });

    EXPECT_FALSE(result.ok);
    EXPECT_THAT(result.error, HasSubstr("no draft tokens"));
    EXPECT_EQ(runner.forward_count, 0);
    EXPECT_EQ(runner.commit_count, 0);
    EXPECT_EQ(sampler.calls(), 0u);
}

TEST(Test__MTPDecodeCatchup, SharedStepwiseHardFailsWithoutSamplerCallback)
{
    FakeCatchupRunner runner;

    MTPDecodeCatchupGreedyRequest request;
    request.draft_tokens = {7};

    MTPDecodeCatchupGreedyResult result =
        runSharedStepwiseMTPDecodeCatchupGreedy(
            runner,
            request,
            MTPDecodeCatchupGreedySampler{});

    EXPECT_FALSE(result.ok);
    EXPECT_THAT(result.error, HasSubstr("no sampler callback"));
    EXPECT_EQ(runner.forward_count, 0);
    EXPECT_EQ(runner.commit_count, 0);
}

TEST(Test__MTPDecodeCatchup, SharedStepwiseHardFailsWhenInitialShiftedCommitFails)
{
    FakeCatchupRunner runner;
    runner.commit_ok = false;
    ScriptedSampler sampler({9});

    MTPDecodeCatchupGreedyRequest request;
    request.draft_tokens = {7, 9};

    MTPDecodeCatchupGreedyResult result =
        runSharedStepwiseMTPDecodeCatchupGreedy(
            runner,
            request,
            [&]() { return sampler(); });

    EXPECT_FALSE(result.ok);
    EXPECT_THAT(result.error, HasSubstr("initial shifted-cache commit failed"));
    EXPECT_EQ(runner.commit_count, 0);
    EXPECT_EQ(runner.forward_count, 0);
    EXPECT_EQ(sampler.calls(), 0u);
}

TEST(Test__MTPDecodeCatchup, SharedStepwiseHardFailsWhenForwardFails)
{
    FakeCatchupRunner runner;
    runner.forward_ok = false;
    ScriptedSampler sampler({9});

    MTPDecodeCatchupGreedyRequest request;
    request.draft_tokens = {7, 9};

    MTPDecodeCatchupGreedyResult result =
        runSharedStepwiseMTPDecodeCatchupGreedy(
            runner,
            request,
            [&]() { return sampler(); });

    EXPECT_FALSE(result.ok);
    EXPECT_THAT(result.error, HasSubstr("failed to forward/sample first token"));
    EXPECT_EQ(runner.commit_count, 1);
    EXPECT_EQ(runner.forward_count, 1);
    EXPECT_EQ(sampler.calls(), 0u);
}

TEST(Test__MTPDecodeCatchup, SharedStepwiseHardFailsWhenSamplerFails)
{
    FakeCatchupRunner runner;
    ScriptedSampler sampler({});

    MTPDecodeCatchupGreedyRequest request;
    request.draft_tokens = {7, 9};

    MTPDecodeCatchupGreedyResult result =
        runSharedStepwiseMTPDecodeCatchupGreedy(
            runner,
            request,
            [&]() { return sampler(); });

    EXPECT_FALSE(result.ok);
    EXPECT_THAT(result.error, HasSubstr("failed to forward/sample first token"));
    EXPECT_EQ(runner.commit_count, 1);
    EXPECT_EQ(runner.forward_count, 1);
    EXPECT_EQ(sampler.calls(), 0u);
}

TEST(Test__MTPDecodeCatchup, SharedStepwiseHardFailsWhenLaterShiftedCommitFails)
{
    class LaterCommitFailureRunner : public FakeCatchupRunner
    {
    public:
        bool commitMTPShiftedRowFromCurrentTerminalHidden(
            int32_t token,
            int already_appended_tokens,
            bool allow_speculative_discard = false,
            int position_offset_override = -1) override
        {
            if (already_appended_tokens >= 1)
                return false;
            return FakeCatchupRunner::commitMTPShiftedRowFromCurrentTerminalHidden(
                token,
                already_appended_tokens,
                allow_speculative_discard,
                position_offset_override);
        }
    };

    LaterCommitFailureRunner runner;
    ScriptedSampler sampler({9});

    MTPDecodeCatchupGreedyRequest request;
    request.draft_tokens = {7, 9};

    MTPDecodeCatchupGreedyResult result =
        runSharedStepwiseMTPDecodeCatchupGreedy(
            runner,
            request,
            [&]() { return sampler(); });

    EXPECT_FALSE(result.ok);
    EXPECT_THAT(result.error, HasSubstr("shifted-cache commit failed"));
    EXPECT_THAT(result.accepted_tokens, ElementsAre(7, 9));
    EXPECT_EQ(result.main_forward_token_count, 1);
    EXPECT_EQ(result.shifted_commit_count, 1);
}
