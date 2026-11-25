/**
 * @file Test__ContinuousBatchingCPU.cpp
 * @brief Real-model continuous batching on a CPU runner.
 *
 * Proves the serving session's core contract against actual model execution:
 * requests share forward steps, a new request enters a slot while others keep
 * decoding (A continues, B finishes, C enters), and every request's tokens are
 * identical to running it alone — greedy and seeded stochastic alike, including
 * a prompt that streams in over several chunks and a cancellation.
 *
 * Requires LLAMINAR_CB_TEST_MODEL (a dense GGUF without MTP heads).
 */

#include <gtest/gtest.h>

#include "backends/BackendManager.h"
#include "execution/runner/IOrchestrationRunner.h"
#include "utils/ServingStepTrace.h"
#include "utils/TestOrchestrationHelper.h"

#include <cstdlib>
#include <filesystem>
#include <unistd.h>
#include <algorithm>
#include <map>
#include <memory>
#include <vector>

namespace llaminar2::test
{
    namespace
    {
        constexpr int kCapacity = 4;

        std::vector<int32_t> promptTokens(int length, int salt)
        {
            std::vector<int32_t> tokens;
            tokens.reserve(static_cast<std::size_t>(length));
            std::uint32_t state = 2166136261u ^ static_cast<std::uint32_t>(salt);
            for (int i = 0; i < length; ++i)
            {
                state = state * 16777619u + 977u;
                tokens.push_back(static_cast<int32_t>(100 + state % 30000u));
            }
            return tokens;
        }

        ServingBatchRequest makeRequest(std::uint64_t id, int prompt_length, int max_new,
                                        float temperature, unsigned int seed)
        {
            ServingBatchRequest request;
            request.id = id;
            request.prompt = promptTokens(prompt_length, static_cast<int>(id));
            request.max_new_tokens = max_new;
            request.sampling.temperature = temperature;
            request.sampling.top_k = temperature == 0.0f ? 1 : 40;
            request.sampling.top_p = temperature == 0.0f ? 1.0f : 0.95f;
            request.sampling.seed = seed;
            return request;
        }

        struct Outcome
        {
            std::vector<int32_t> tokens;
            ServingFinishReason finish = ServingFinishReason::None;
            int finished_at_step = -1;
            int first_token_step = -1;
            int cached_prompt_tokens = 0;
            std::vector<std::uint64_t> logit_digests;
        };

        class ContinuousBatchingCPU : public ::testing::Test
        {
        protected:
            virtual bool prefixCache() const { return false; }

            void SetUp() override
            {
                // The V2_Models fixture provides the canonical model under the
                // project root (the test's working directory); the variable
                // only overrides it. A missing model fails: never a skip.
                const char *override_model = std::getenv("LLAMINAR_CB_TEST_MODEL");
                const std::string model = override_model && *override_model
                                              ? std::string(override_model)
                                              : std::string("models/qwen2.5-0.5b-instruct-q4_0.gguf");
                ASSERT_TRUE(std::filesystem::exists(model)) << "continuous-batching test model missing: " << model;
                if (!hasCPUBackend())
                    initCPUBackend(0);
                OrchestrationConfig config =
                    TestOrchestrationHelper::buildSimpleConfig(model, DeviceId::cpu(), 1024);
                config.batch_size = kCapacity;
                config.deterministic = false;
                config.prefix_cache.enabled = prefixCache();
                if (prefixCache())
                {
                    // Hermetic tiers: the default disk tier persists across
                    // processes, which would make a "cold" request warm.
                    char dir[] = "/tmp/llaminar-cb-prefix-XXXXXX";
                    ASSERT_NE(::mkdtemp(dir), nullptr);
                    config.prefix_cache.disk_dir = dir;
                }
                runner_ = TestOrchestrationHelper::create(config);
                ASSERT_NE(runner_, nullptr);
                ASSERT_TRUE(runner_->initialize()) << runner_->lastError();
                ASSERT_GE(runner_->servingBatchCapacity(), kCapacity)
                    << "runner does not serve continuous batches";
            }

            /** Run requests through the session; `entries[step]` submits at that step. */
            std::map<std::uint64_t, Outcome> run(
                const std::map<int, std::vector<ServingBatchRequest>> &entries,
                const std::map<int, std::uint64_t> &cancels = {})
            {
                std::map<std::uint64_t, Outcome> outcomes;
                EXPECT_TRUE(runner_->beginServingBatch(kCapacity)) << runner_->lastError();
                for (int step = 0; step < 400; ++step)
                {
                    if (auto it = entries.find(step); it != entries.end())
                        for (const auto &request : it->second)
                            EXPECT_TRUE(runner_->submitServingRequest(request));
                    if (auto it = cancels.find(step); it != cancels.end())
                        runner_->cancelServingRequest(it->second);
                    for (const auto &event : runner_->stepServingBatch())
                    {
                        Outcome &outcome = outcomes[event.id];
                        if (!event.tokens.empty() && outcome.first_token_step < 0)
                            outcome.first_token_step = step;
                        outcome.tokens.insert(outcome.tokens.end(), event.tokens.begin(), event.tokens.end());
                        outcome.cached_prompt_tokens = std::max(outcome.cached_prompt_tokens,
                                                                event.cached_prompt_tokens);
                        outcome.logit_digests.insert(outcome.logit_digests.end(),
                                                     event.logit_digests.begin(), event.logit_digests.end());
                        if (event.finished())
                        {
                            EXPECT_EQ(outcome.finish, ServingFinishReason::None) << "finished twice";
                            EXPECT_TRUE(event.error.empty()) << event.error;
                            outcome.finish = event.finish;
                            outcome.finished_at_step = step;
                        }
                    }
                    bool more_entries = false;
                    for (const auto &[at, _] : entries)
                        more_entries = more_entries || at > step;
                    if (runner_->servingBatchIdle() && !more_entries)
                        break;
                }
                runner_->endServingBatch();
                return outcomes;
            }

            /** Tokens of @p request generated alone. */
            Outcome solo(const ServingBatchRequest &request)
            {
                return run({{0, {request}}}).at(request.id);
            }

            std::unique_ptr<IOrchestrationRunner> runner_;
        };

        /** Same session with the (default-on) prefix cache enabled. */
        class ContinuousBatchingCPUPrefix : public ContinuousBatchingCPU
        {
        protected:
            bool prefixCache() const override { return true; }
        };
    } // namespace

    TEST_F(ContinuousBatchingCPU, SoloServingMatchesScalarGeneration)
    {
        const auto request = makeRequest(1, 12, 16, 0.0f, 1);
        SamplingParams greedy = request.sampling;
        const auto scalar = runner_->generate(request.prompt, request.max_new_tokens, greedy);
        ASSERT_TRUE(scalar.error.empty()) << scalar.error;
        const auto served = solo(request);
        EXPECT_EQ(served.finish, ServingFinishReason::Length);
        EXPECT_EQ(served.tokens, scalar.tokens);
    }

    TEST_F(ContinuousBatchingCPU, LogitsBytesAreIndependentOfBatchComposition)
    {
        // Serial-row byte equivalence: the exact logits behind every decision
        // of request A are identical alone, with a co-prefilling request, with
        // a request entering mid-decode and at full capacity.
        auto a = makeRequest(1, 37, 24, 0.0f, 1);
        a.record_logit_digests = true;
        const auto alone = solo(a);
        ASSERT_EQ(alone.logit_digests.size(), alone.tokens.size());

        const auto b = makeRequest(2, 150, 30, 0.7f, 5);  // long prompt: 3 chunks
        const auto c = makeRequest(3, 12, 30, 0.0f, 1);
        const auto d = makeRequest(4, 70, 30, 0.9f, 9);
        const std::vector<std::pair<const char *, std::map<int, std::vector<ServingBatchRequest>>>> compositions = {
            {"co-prefill", {{0, {a, b}}}},
            {"enters mid-decode", {{0, {a}}, {4, {c}}}},
            {"full capacity, staggered", {{0, {b, a}}, {2, {c}}, {5, {d}}}},
        };
        for (const auto &[name, entries] : compositions)
        {
            const auto outcome = run(entries).at(1);
            EXPECT_EQ(outcome.tokens, alone.tokens) << name;
            EXPECT_EQ(outcome.logit_digests, alone.logit_digests) << name;
        }
    }

    TEST_F(ContinuousBatchingCPU, StopTokenIsReturnedAndRetiresOnlyItsRequest)
    {
        // Choose a stop token the greedy request actually produces, so its
        // stream must end with (and include) the first occurrence.
        const auto base = makeRequest(1, 14, 24, 0.0f, 1);
        const auto free_run = solo(base);
        ASSERT_GE(free_run.tokens.size(), 6u);
        const int32_t stop = free_run.tokens[5];
        const auto first = std::find(free_run.tokens.begin(), free_run.tokens.end(), stop);
        const std::vector<int32_t> expected(free_run.tokens.begin(), first + 1);

        auto stopping = base;
        stopping.stop_tokens = {stop};
        auto other = makeRequest(2, 30, 24, 0.0f, 1);
        const auto other_solo = solo(other);

        const auto batched = run({{0, {stopping, other}}});
        EXPECT_EQ(batched.at(1).finish, ServingFinishReason::Stop);
        EXPECT_EQ(batched.at(1).tokens, expected);
        // The co-resident request is unaffected by the other row's stop.
        EXPECT_EQ(batched.at(2).finish, ServingFinishReason::Length);
        EXPECT_EQ(batched.at(2).tokens, other_solo.tokens);
    }

    TEST_F(ContinuousBatchingCPU, AContinuesBFinishesCEntersAndAllMatchSolo)
    {
        const auto a = makeRequest(1, 9, 40, 0.0f, 1);   // long, greedy
        const auto b = makeRequest(2, 21, 6, 0.0f, 1);   // short: finishes early
        const auto c = makeRequest(3, 150, 12, 0.0f, 1); // enters later; 3 prompt chunks
        const auto d = makeRequest(4, 33, 30, 0.8f, 7);  // seeded stochastic

        std::map<std::uint64_t, Outcome> reference;
        for (const auto &request : {a, b, c, d})
            reference[request.id] = solo(request);

        // A, B, D start together; C arrives at step 3 and waits for nothing:
        // a slot is free (capacity 4), then B retires while A and D continue.
        const auto batched = run({{0, {a, b, d}}, {3, {c}}});
        for (const auto &request : {a, b, c, d})
        {
            const auto &got = batched.at(request.id);
            EXPECT_EQ(got.tokens, reference.at(request.id).tokens) << "request " << request.id;
            EXPECT_EQ(got.finish, reference.at(request.id).finish) << "request " << request.id;
        }
        // Shared execution: C entered while A was still decoding, and A was
        // still running after B finished.
        EXPECT_LT(batched.at(3).first_token_step, batched.at(1).finished_at_step);
        EXPECT_LT(batched.at(2).finished_at_step, batched.at(1).finished_at_step);
    }

    TEST_F(ContinuousBatchingCPU, NewRequestTakesARetiredSlotAtFullCapacity)
    {
        // Five requests on four slots: E enters only after one retires, in the
        // compacted slot, and still matches its solo stream.
        std::vector<ServingBatchRequest> first;
        for (std::uint64_t id = 1; id <= 4; ++id)
            first.push_back(makeRequest(id, 7 + static_cast<int>(id), id == 2 ? 3 : 20, 0.0f, 1));
        const auto e = makeRequest(5, 13, 10, 0.7f, 11);
        const auto e_solo = solo(e);
        const auto two_solo = solo(first[1]);
        const auto four_solo = solo(first[3]);

        const auto batched = run({{0, first}, {1, {e}}});
        EXPECT_EQ(batched.at(5).tokens, e_solo.tokens);
        EXPECT_EQ(batched.at(2).tokens, two_solo.tokens);
        EXPECT_EQ(batched.at(4).tokens, four_solo.tokens);
        EXPECT_GT(batched.at(5).first_token_step, batched.at(2).finished_at_step - 1);
        EXPECT_LT(batched.at(5).first_token_step, batched.at(1).finished_at_step);
    }

    TEST_F(ContinuousBatchingCPU, CancellationRetiresOnlyThatRequest)
    {
        const auto a = makeRequest(1, 10, 24, 0.0f, 1);
        const auto b = makeRequest(2, 10, 24, 0.0f, 1);
        const auto a_solo = solo(a);
        const auto batched = run({{0, {a, b}}}, {{4, 2}});
        EXPECT_EQ(batched.at(2).finish, ServingFinishReason::Cancelled);
        EXPECT_LT(batched.at(2).tokens.size(), 24u);
        EXPECT_EQ(batched.at(1).tokens, a_solo.tokens);
    }
    TEST_F(ContinuousBatchingCPU, PacedGrantsAndForcedTokensMatchScalarPath)
    {
        // Scalar reference: 3 sampled tokens, one forced token, 4 more.
        auto paced = makeRequest(1, 11, 64, 0.0f, 1);
        constexpr int32_t kForced = 151644;
        runner_->setSamplingParams(paced.sampling);
        ASSERT_TRUE(runner_->prefill(paced.prompt)) << runner_->lastError();
        std::vector<int32_t> scalar;
        for (int i = 0; i < 3; ++i)
        {
            const auto step = runner_->decodeStep();
            ASSERT_TRUE(step.error.empty()) << step.error;
            scalar.insert(scalar.end(), step.tokens.begin(), step.tokens.end());
        }
        const auto forced = runner_->forceDecodeToken(kForced);
        ASSERT_TRUE(forced.error.empty()) << forced.error;
        scalar.insert(scalar.end(), forced.tokens.begin(), forced.tokens.end());
        for (int i = 0; i < 4; ++i)
        {
            const auto step = runner_->decodeStep();
            ASSERT_TRUE(step.error.empty()) << step.error;
            scalar.insert(scalar.end(), step.tokens.begin(), step.tokens.end());
        }
        runner_->clearCache();

        // Batched: the paced request blocks between grants while another
        // request keeps decoding beside it.
        paced.paced = true;
        const auto other = makeRequest(2, 17, 30, 0.0f, 1);
        const auto other_solo = solo(other);
        ASSERT_TRUE(runner_->beginServingBatch(kCapacity));
        ASSERT_TRUE(runner_->submitServingRequest(paced));
        ASSERT_TRUE(runner_->submitServingRequest(other));
        runner_->grantServingTokens(1, 3);
        std::vector<int32_t> got, other_got;
        auto pump = [&](int steps) {
            for (int i = 0; i < steps; ++i)
                for (const auto &event : runner_->stepServingBatch())
                {
                    auto &sink = event.id == 1 ? got : other_got;
                    sink.insert(sink.end(), event.tokens.begin(), event.tokens.end());
                }
        };
        pump(8);
        EXPECT_EQ(got.size(), 3u) << "paced request must block after its grant";
        runner_->forceServingToken(1, kForced);
        pump(2);
        EXPECT_EQ(got.size(), 4u);
        runner_->grantServingTokens(1, 4);
        pump(40);
        runner_->cancelServingRequest(1);
        pump(1);
        runner_->endServingBatch();
        EXPECT_EQ(got, scalar);
        EXPECT_EQ(other_got, other_solo.tokens);
    }
} // namespace llaminar2::test

namespace llaminar2::test
{
    TEST_F(ContinuousBatchingCPUPrefix, SlotRestoresCachedPrefixWithoutDisturbingOthers)
    {
        // Cold: computes and harvests the prompt from its slot.
        const auto b = makeRequest(11, 150, 12, 0.0f, 1);
        const auto cold = solo(b);
        EXPECT_EQ(cold.cached_prompt_tokens, 0);

        // Warm, co-batched with an unrelated request: B restores its cached
        // prefix into its own slot (at least one token is recomputed) and
        // still produces the cold token stream; A is unaffected.
        const auto a = makeRequest(12, 40, 20, 0.0f, 1);
        const auto a_solo = solo(a);
        const auto batched = run({{0, {a}}, {2, {b}}});
        EXPECT_GT(batched.at(11).cached_prompt_tokens, 0);
        EXPECT_LT(batched.at(11).cached_prompt_tokens, 150);
        EXPECT_EQ(batched.at(11).tokens, cold.tokens);
        EXPECT_EQ(batched.at(12).tokens, a_solo.tokens);

        // A different suffix after the shared prefix reuses the common blocks.
        auto c = b;
        c.id = 13;
        for (std::size_t i = 140; i < c.prompt.size(); ++i)
            c.prompt[i] = static_cast<int32_t>(200 + i);
        const auto c_out = solo(c);
        EXPECT_GT(c_out.cached_prompt_tokens, 0);
        EXPECT_LE(c_out.cached_prompt_tokens, 140);
        EXPECT_EQ(c_out.finish, ServingFinishReason::Length);
    }

    TEST_F(ContinuousBatchingCPUPrefix, RestoredRowLogitsBytesMatchColdServing)
    {
        // Byte equality must include prefix-restored rows: decisions after a
        // restored prefix see exactly the logits of cold serving.
        auto b = makeRequest(21, 150, 10, 0.0f, 1);
        b.record_logit_digests = true;
        const auto cold = solo(b);
        ASSERT_EQ(cold.cached_prompt_tokens, 0);
        const auto other = makeRequest(22, 33, 16, 0.7f, 3);
        const auto warm = run({{0, {other}}, {1, {b}}}).at(21);
        ASSERT_GT(warm.cached_prompt_tokens, 0);
        EXPECT_EQ(warm.tokens, cold.tokens);
        EXPECT_EQ(warm.logit_digests, cold.logit_digests);
    }

    TEST_F(ContinuousBatchingCPUPrefix, ScalarConsumerRecomputesServingArchivedTerminalBlocks)
    {
        // Serving rows archive KV only (no terminal logits/hidden). A later
        // capacity-one consumer of the same prompt must treat those blocks as
        // non-terminal and recompute, producing the cold result.
        const auto request = makeRequest(31, 128, 12, 0.0f, 1); // block-aligned: a full hit
        const auto served = solo(request);
        ASSERT_EQ(served.finish, ServingFinishReason::Length);
        SamplingParams greedy = request.sampling;
        const auto scalar = runner_->generate(request.prompt, request.max_new_tokens, greedy);
        ASSERT_TRUE(scalar.error.empty()) << scalar.error;
        EXPECT_EQ(scalar.tokens, served.tokens);
        const auto prefix = runner_->requestRuntimeSummary().prefix_request;
        EXPECT_TRUE(prefix.enabled);
        EXPECT_GT(prefix.matched_tokens, 0) << "the scalar request reused the serving-archived blocks";
        EXPECT_FALSE(prefix.terminal_logits_restored) << "serving archives carry no terminal logits";
    }
} // namespace llaminar2::test

namespace llaminar2::test
{
    /**
     * Records the production serving session on the real model for the
     * device-free recorded-input replay (V2_Unit_ServingStepReplay) and
     * replays the recording at once through a fresh ServingSession. With
     * LLAMINAR_CB_RECORD_TRACE=<path> the trace is written there (fixture).
     */
    TEST_F(ContinuousBatchingCPU, RecordedServingTraceReplaysExactly)
    {
        ServingStepTrace trace;
        trace.context = 1024;
        trace.vocab = runner_->vocabSize();
        trace.capacity = 3;
        runner_->setServingTraceRecorder(std::make_shared<ServingStepTraceRecorder>(trace));

        auto submit = [&](int step, ServingBatchRequest request) {
            ServingStepTrace::Command c;
            c.kind = ServingStepTrace::Command::Kind::Submit;
            c.step = step;
            c.id = request.id;
            c.request = std::move(request);
            trace.commands.push_back(std::move(c));
        };
        auto command = [&](ServingStepTrace::Command::Kind kind, int step, std::uint64_t id, int value = 0) {
            ServingStepTrace::Command c;
            c.kind = kind;
            c.step = step;
            c.id = id;
            c.value = value;
            trace.commands.push_back(c);
        };
        auto a = makeRequest(1, 9, 14, 0.0f, 1);   // greedy, unpaced
        auto b = makeRequest(2, 70, 10, 0.8f, 7);  // seeded stochastic, two prompt chunks
        auto c = makeRequest(3, 21, 6, 0.0f, 1);   // greedy, paced with grants + forced token
        c.paced = true;
        auto d = makeRequest(4, 12, 8, 0.9f, 0);   // entropy seed resolved at admission
        d.sampling.top_k = 20;
        auto e = makeRequest(5, 30, 30, 0.0f, 1);  // queued, cancelled before entering
        submit(0, a);
        submit(0, b);
        submit(0, c);
        command(ServingStepTrace::Command::Kind::Grant, 1, 3, 2);
        submit(2, e);
        submit(3, d);
        command(ServingStepTrace::Command::Kind::Force, 4, 3, 42);
        command(ServingStepTrace::Command::Kind::Grant, 5, 3, 4);
        command(ServingStepTrace::Command::Kind::Cancel, 7, 5);

        ASSERT_TRUE(runner_->beginServingBatch(trace.capacity)) << runner_->lastError();
        int last_command_step = 0;
        for (const auto &cmd : trace.commands)
            last_command_step = std::max(last_command_step, cmd.step);
        for (int step = 0; step < 200; ++step)
        {
            for (const auto &cmd : trace.commands)
            {
                if (cmd.step != step)
                    continue;
                switch (cmd.kind)
                {
                case ServingStepTrace::Command::Kind::Submit:
                    ASSERT_TRUE(runner_->submitServingRequest(cmd.request)) << runner_->lastError();
                    break;
                case ServingStepTrace::Command::Kind::Grant: runner_->grantServingTokens(cmd.id, cmd.value); break;
                case ServingStepTrace::Command::Kind::Force: runner_->forceServingToken(cmd.id, cmd.value); break;
                case ServingStepTrace::Command::Kind::Cancel: runner_->cancelServingRequest(cmd.id); break;
                }
            }
            (void)runner_->stepServingBatch();
            if (step >= last_command_step && runner_->servingBatchIdle())
                break;
        }
        runner_->endServingBatch();
        runner_->setServingTraceRecorder(nullptr);

        ASSERT_TRUE(runner_->servingBatchIdle());
        EXPECT_EQ(trace.admissions.size(), 4u) << "E is cancelled while queued";
        EXPECT_NE(trace.admissions.back().seed, 0u) << "entropy seed is resolved and recorded";
        EXPECT_EQ(ServingStepTraceReplayer::replay(trace), "");
        if (const char *path = std::getenv("LLAMINAR_CB_RECORD_TRACE"); path && *path)
            trace.write(path);
    }
} // namespace llaminar2::test
