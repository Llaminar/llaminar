/**
 * @file Test__ServingStepReplay.cpp
 * @brief Recorded-input replay of the continuous-batching serving step.
 *
 * Replays a real-model recording (tests/v2/fixtures/serving_step_replay)
 * through the production ServingSession, device-free: the session must issue
 * exactly the recorded forwards (slots, token rows), admit the recorded
 * requests into the recorded slots, and emit identical events (tokens,
 * finishes, counts) — covering greedy and seeded/entropy-seeded stochastic
 * sampling, paced grants, a forced token, chunked prompts, cancellation while
 * queued and admission into retired slots. Negative controls prove the replay
 * detects a changed logit, a missing forward and a changed outcome.
 */

#include "utils/ServingStepTrace.h"

#include <gtest/gtest.h>

#ifndef LLAMINAR_SERVING_TRACE_FIXTURE
#error "LLAMINAR_SERVING_TRACE_FIXTURE must name the recorded trace"
#endif

namespace llaminar2::test
{
    namespace
    {
        ServingStepTrace fixture() { return ServingStepTrace::read(LLAMINAR_SERVING_TRACE_FIXTURE); }
    } // namespace

    TEST(ServingStepReplay, RecordedRealModelTraceReplaysExactly)
    {
        const auto trace = fixture();
        ASSERT_GT(trace.forwards.size(), 0u);
        ASSERT_EQ(trace.admissions.size(), 4u);
        EXPECT_EQ(ServingStepTraceReplayer::replay(trace), "");
    }

    TEST(ServingStepReplay, ReplayIsRepeatable)
    {
        const auto trace = fixture();
        for (int i = 0; i < 3; ++i)
            EXPECT_EQ(ServingStepTraceReplayer::replay(trace), "") << "replay " << i;
    }

    TEST(ServingStepReplay, ChangedLogitIsDetected)
    {
        auto trace = fixture();
        // Promote the second candidate of the first greedy decision above
        // the first: the recorded token can no longer be produced.
        bool changed = false;
        for (auto &forward : trace.forwards)
            for (auto &row : forward.top)
                if (!changed && row.size() >= 2)
                {
                    row[1].second = row[0].second + 1.0f;
                    changed = true;
                }
        ASSERT_TRUE(changed);
        EXPECT_NE(ServingStepTraceReplayer::replay(trace), "");
    }

    TEST(ServingStepReplay, MissingForwardIsDetected)
    {
        auto trace = fixture();
        trace.forwards.pop_back();
        EXPECT_NE(ServingStepTraceReplayer::replay(trace), "");
    }

    TEST(ServingStepReplay, ChangedOutcomeIsDetected)
    {
        auto trace = fixture();
        bool changed = false;
        for (auto &step : trace.steps)
            for (auto &event : step.events)
                if (!changed && event.finished())
                {
                    event.finish = event.finish == ServingFinishReason::Length ? ServingFinishReason::Stop
                                                                               : ServingFinishReason::Length;
                    changed = true;
                }
        ASSERT_TRUE(changed);
        EXPECT_NE(ServingStepTraceReplayer::replay(trace), "");
    }

    namespace
    {
        /** @brief Minimal scripted rows: token 3 always wins; failures on demand. */
        class ScriptedRows final : public IServingRowExecutor
        {
        public:
            static constexpr int kVocab = 8;
            bool fail_forward = false;
            bool fail_reset = false;
            bool forwardServingRows(const std::vector<int> &, const std::vector<std::vector<int>> &rows) override
            {
                if (fail_forward)
                    return false;
                logits_.assign(rows.size() * kVocab, 0.0f);
                for (std::size_t r = 0; r < rows.size(); ++r)
                    logits_[r * kVocab + 3] = 1.0f;
                return true;
            }
            const float *servingRowLogits(int row) const override { return logits_.data() + row * kVocab; }
            bool resetServingRow(int) override { return !fail_reset; }
            int restoreServingRowPrefix(int, const std::vector<int32_t> &, PrefixLookupResult &admission) override
            {
                admission = {};
                return 0;
            }
            bool harvestServingRowPrefix(int, const PrefixLookupResult &, const std::vector<int32_t> &) override
            {
                return true;
            }
            std::string servingRowError() const override { return fail_forward ? "scripted forward failure" : ""; }

        private:
            std::vector<float> logits_;
        };

        ServingBatchRequest greedy(std::uint64_t id, int max_new)
        {
            ServingBatchRequest request;
            request.id = id;
            request.prompt = {1, 2, 3};
            request.max_new_tokens = max_new;
            request.sampling.temperature = 0.0f;
            request.sampling.top_k = 1;
            request.sampling.seed = 1;
            return request;
        }
    } // namespace

    TEST(ServingSessionLifecycle, FailedSlotResetAtRetirementSealsTheSession)
    {
        ScriptedRows rows;
        ServingSession session(rows, 64, ScriptedRows::kVocab);
        std::string error;
        ASSERT_TRUE(session.begin(2, 2, error)) << error;
        ASSERT_TRUE(session.submit(greedy(1, 1), error));
        ASSERT_TRUE(session.submit(greedy(2, 30), error));
        rows.fail_reset = true; // the finishing slot cannot be emptied
        std::map<std::uint64_t, ServingFinishReason> finished;
        for (int step = 0; step < 4 && finished.size() < 2; ++step)
            for (const auto &event : session.step())
                if (event.finished())
                    finished[event.id] = event.finish;
        EXPECT_EQ(finished[1], ServingFinishReason::Length) << "the finished request keeps its outcome";
        EXPECT_EQ(finished[2], ServingFinishReason::Error) << "co-resident requests fail with the seal";
        EXPECT_NE(session.failure().find("could not be reset"), std::string::npos) << session.failure();
        EXPECT_FALSE(session.submit(greedy(3, 1), error)) << "nothing is admitted onto stale slot state";
    }

    TEST(ServingSessionLifecycle, FinishedPeersKeepTheirOutcomeWhenAResetFails)
    {
        // Requests 1 and 2 finish in the same step (3 and 2 tokens, one
        // prompt chunk per step); request 3 is still live. A failed reset must
        // not turn the finished peers' Length into Error.
        ScriptedRows rows;
        ServingSession session(rows, 64, ScriptedRows::kVocab);
        std::string error;
        ASSERT_TRUE(session.begin(3, 3, error)) << error;
        ASSERT_TRUE(session.submit(greedy(1, 3), error));
        ASSERT_TRUE(session.submit(greedy(2, 2), error));
        ASSERT_TRUE(session.submit(greedy(3, 30), error));
        std::map<std::uint64_t, ServingFinishReason> finished;
        for (int step = 0; step < 2; ++step)
            for (const auto &event : session.step())
                if (event.finished())
                    finished[event.id] = event.finish;
        ASSERT_TRUE(finished.empty());
        rows.fail_reset = true;
        for (const auto &event : session.step())
            if (event.finished())
                finished[event.id] = event.finish;
        EXPECT_EQ(finished[1], ServingFinishReason::Length);
        EXPECT_EQ(finished[2], ServingFinishReason::Length);
        EXPECT_EQ(finished[3], ServingFinishReason::Error);
        EXPECT_FALSE(session.failure().empty());
    }

    TEST(ServingSessionLifecycle, ForwardFailureSealsWithTheRunnerDiagnostic)
    {
        ScriptedRows rows;
        ServingSession session(rows, 64, ScriptedRows::kVocab);
        std::string error;
        ASSERT_TRUE(session.begin(1, 1, error)) << error;
        ASSERT_TRUE(session.submit(greedy(1, 4), error));
        rows.fail_forward = true;
        const auto events = session.step();
        ASSERT_EQ(events.size(), 1u);
        EXPECT_EQ(events[0].finish, ServingFinishReason::Error);
        EXPECT_EQ(events[0].error, "scripted forward failure");
        EXPECT_EQ(session.failure(), "scripted forward failure");
    }
} // namespace llaminar2::test
