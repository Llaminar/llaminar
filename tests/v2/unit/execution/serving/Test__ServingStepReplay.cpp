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
} // namespace llaminar2::test
