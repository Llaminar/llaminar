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
#include <filesystem>
#include <chrono>

#ifndef LLAMINAR_SERVING_TRACE_FIXTURE
#error "LLAMINAR_SERVING_TRACE_FIXTURE must name the recorded trace"
#endif

namespace llaminar2::test
{
    namespace
    {
        ServingStepTrace fixture() { return ServingStepTrace::read(LLAMINAR_SERVING_TRACE_FIXTURE); }
    } // namespace

    /** Full coverage is an observation contract, not permission to invent missing logits. */
    TEST(ServingTraceContract, FullVocabularyRejectsMissingAndDuplicateObservations)
    {
        std::vector<float> row;
        using Coverage = ServingStepTrace::LogitCoverage;
        EXPECT_THROW(ServingStepTrace::denseRow({{0, 1.0f}}, 2, row, Coverage::FullVocabulary), std::invalid_argument);
        EXPECT_THROW(ServingStepTrace::denseRow({{0, 1.0f}, {0, 2.0f}}, 2, row, Coverage::FullVocabulary), std::invalid_argument);
        EXPECT_THROW(ServingStepTrace::denseRow({{0, 1.0f}, {2, 2.0f}}, 2, row, Coverage::FullVocabulary), std::invalid_argument);
        ServingStepTrace::denseRow({{1, -3.25f}, {0, 0.125f}}, 2, row, Coverage::FullVocabulary);
        EXPECT_EQ(row, (std::vector<float>{0.125f, -3.25f}));
    }

    /** Serialization preserves penalties and escaped DRY breakers without loss. */
    TEST(ServingTraceContract, FullSamplingLawRoundTrips)
    {
        const auto path = std::filesystem::path(::testing::TempDir()) /
            ("serving-law-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        struct Cleanup {
            std::filesystem::path path;
            ~Cleanup() { std::error_code ec; std::filesystem::remove(path, ec); }
        } cleanup{path};
        ServingStepTrace trace;
        trace.coverage = ServingStepTrace::LogitCoverage::FullVocabulary;
        trace.context = 32; trace.vocab = 2; trace.capacity = 1;
        trace.steps.emplace_back();
        auto &command = trace.commands.emplace_back();
        command.request.id = 1;
        command.request.prompt = {1};
        auto &p = command.request.sampling;
        p.top_k = 0; p.top_p = 0.91f; p.seed = 123;
        p.presence_penalty = 0.3f; p.frequency_penalty = 0.2f; p.repetition_penalty = 1.1f;
        p.dry_multiplier = 0.7f; p.dry_base = 1.9f; p.dry_allowed_length = 3; p.dry_penalty_last_n = 17;
        p.dry_sequence_breakers = {"\n", "\"", "\\", "multi word"};
        EXPECT_NO_THROW(ServingStepTrace::requireReplayableLaw(p, trace.coverage));
        EXPECT_THROW(ServingStepTrace::requireReplayableLaw(p), std::invalid_argument);
        trace.write(path.string());
        const auto restored = ServingStepTrace::read(path.string());
        ASSERT_EQ(restored.commands.size(), 1u);
        EXPECT_EQ(restored.commands[0].request.sampling, p);
        std::ifstream valid_file(path);
        const std::string valid((std::istreambuf_iterator<char>(valid_file)), std::istreambuf_iterator<char>());
        valid_file.close();
        const auto law_start = valid.find("law ") + 4;
        const auto law_end = valid.find('\n', law_start);
        const auto original_law = nlohmann::json::parse(valid.substr(law_start, law_end - law_start));
        auto reject_law = [&](const std::string &law) {
            std::ofstream broken(path);
            broken << valid.substr(0, law_start) << law << valid.substr(law_end);
            broken.close();
            EXPECT_THROW(ServingStepTrace::read(path.string()), std::exception) << law;
        };
        auto unknown = original_law;
        unknown["unknown"] = 0;
        reject_law(unknown.dump());
        auto missing = original_law;
        missing.erase("dry_base");
        reject_law(missing.dump());
        reject_law(original_law.dump().substr(0, original_law.dump().size() - 1) +
                   ",\"dry_base\":\"0x1p+0\"}");
        for (const auto &invalid : std::vector<nlohmann::json>{1.5, true, 2147483648ULL, -2147483649LL, "2"})
        {
            auto law = original_law;
            law["dry_allowed_length"] = invalid;
            reject_law(law.dump());
        }
        auto sentinel_law = original_law;
        sentinel_law["dry_penalty_last_n"] = -1;
        {
            std::ofstream sentinel(path);
            sentinel << valid.substr(0, law_start) << sentinel_law.dump() << valid.substr(law_end);
        }
        EXPECT_EQ(ServingStepTrace::read(path.string()).commands[0].request.sampling.dry_penalty_last_n, -1);
        for (const auto &schedule : {std::string("submit -1"), std::string("submit 1")})
        {
            auto invalid = valid;
            invalid.replace(invalid.find("submit 0"), 8, schedule);
            std::ofstream bad_schedule(path);
            bad_schedule << invalid;
            bad_schedule.close();
            EXPECT_THROW(ServingStepTrace::read(path.string()), std::runtime_error);
        }
        auto no_steps = valid;
        no_steps.erase(no_steps.find("step 0\n"), 7);
        {
            std::ofstream empty_schedule(path);
            empty_schedule << no_steps;
        }
        EXPECT_THROW(ServingStepTrace::read(path.string()), std::runtime_error);
        // A v2 submit without its law must never inherit default penalties.
        std::ofstream truncated(path);
        truncated << "llaminar-serving-trace v2-full\ngeometry 32 2 1\nsubmit 0 1 1 0 0x1p+0 0 0x1p+0 123 0 1 1\n";
        truncated.close();
        EXPECT_THROW(ServingStepTrace::read(path.string()), std::runtime_error);
        for (const auto &malformed : std::vector<std::string>{
                 "llaminar-serving-trace v2-full\ngeometry 32 2 1\n", // missing end
                 "llaminar-serving-trace v2-full\ngeometry 32 2 1\nstep 1\nend\n", // missing event
                 "llaminar-serving-trace v2-full\ngeometry 32 2 1\nend\nstep 0\n", // trailing record
                 "llaminar-serving-trace v1\ngeometry 32 2 1 junk\n", // trailing field
                 "llaminar-serving-trace v1\ngeometry 32 2 1\nforward 1 0 1 1\ntop 1 0 invalid\n",
                 "llaminar-serving-trace v1\ngeometry 32 2 1\nforward 1 0 1 1\ntop 1 0 0x1p+0junk\n",
                 "llaminar-serving-trace v1\ngeometry 32 2 1\nstep 1\nevent 1 0 1 1\n"})
        {
            std::ofstream broken(path);
            broken << malformed;
            broken.close();
            EXPECT_THROW(ServingStepTrace::read(path.string()), std::runtime_error) << malformed;
        }
    }

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

    /** In-memory recordings cannot hide commands outside the replayed step domain. */
    TEST(ServingTraceContract, CommandsOutsideRecordedStepsAreRejected)
    {
        for (int step : {-1, 1000000})
        {
            auto trace = fixture();
            trace.commands.front().step = step;
            EXPECT_NE(ServingStepTraceReplayer::replay(trace), "");
        }
        auto trace = fixture();
        trace.steps.clear();
        EXPECT_NE(ServingStepTraceReplayer::replay(trace), "");
    }

    /** Production admission must authenticate request identity, seed and observation order. */
    TEST(ServingTraceContract, AdmissionIdentityAndAuthoredSeedMustMatch)
    {
        auto trace = fixture();
        trace.admissions.front().id = 999999;
        EXPECT_NE(ServingStepTraceReplayer::replay(trace), "");
        trace = fixture();
        trace.admissions[1].id = trace.admissions[0].id;
        EXPECT_NE(ServingStepTraceReplayer::replay(trace), "");
        trace = fixture();
        ++trace.admissions.front().seed;
        EXPECT_NE(ServingStepTraceReplayer::replay(trace), "");
        trace = fixture();
        std::swap(trace.admissions[0].id, trace.admissions[1].id);
        std::swap(trace.admissions[0].seed, trace.admissions[1].seed);
        EXPECT_NE(ServingStepTraceReplayer::replay(trace), "");
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

    /** A rejected duplicate cannot acquire state or consume the original owner's cancellation. */
    TEST(ServingSessionLifecycle, DuplicateLiveIdentityIsRejectedBeforeMutation)
    {
        ScriptedRows rows;
        ServingSession session(rows, 64, ScriptedRows::kVocab);
        std::string error;
        ASSERT_TRUE(session.begin(2, 2, error));
        ASSERT_TRUE(session.submit(greedy(7, 8), error));
        EXPECT_FALSE(session.submit(greedy(7, 1), error));
        EXPECT_NE(error.find("already live"), std::string::npos);
        EXPECT_EQ(session.occupancy().queued, 1);
        (void)session.step();
        EXPECT_EQ(session.occupancy().resident, 1);
        EXPECT_FALSE(session.submit(greedy(7, 1), error));
        EXPECT_EQ(session.occupancy().queued, 0);
        ASSERT_TRUE(session.submit(greedy(9, 3), error));
        session.cancel(7);
        std::vector<ServingBatchEvent> terminal;
        for (int step = 0; step < 10 && !session.idle(); ++step)
            for (const auto &event : session.step())
                if (event.finished()) terminal.push_back(event);
        ASSERT_TRUE(session.idle());
        ASSERT_EQ(terminal.size(), 2u);
        EXPECT_EQ(terminal[0].id, 7u);
        EXPECT_EQ(terminal[0].finish, ServingFinishReason::Cancelled);
        EXPECT_EQ(terminal[1].id, 9u);
        EXPECT_EQ(terminal[1].finish, ServingFinishReason::Length);
        EXPECT_EQ(terminal[1].completion_tokens, 3);
        EXPECT_TRUE(session.failure().empty());
    }

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
