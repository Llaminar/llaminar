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
#include "utils/TestOrchestrationHelper.h"

#include <cstdlib>
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
        };

        class ContinuousBatchingCPU : public ::testing::Test
        {
        protected:
            void SetUp() override
            {
                const char *model = std::getenv("LLAMINAR_CB_TEST_MODEL");
                if (!model || !*model)
                    GTEST_SKIP() << "LLAMINAR_CB_TEST_MODEL not set";
                if (!hasCPUBackend())
                    initCPUBackend(0);
                OrchestrationConfig config =
                    TestOrchestrationHelper::buildSimpleConfig(model, DeviceId::cpu(), 1024);
                config.batch_size = kCapacity;
                config.deterministic = false;
                config.prefix_cache.enabled = false;
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
