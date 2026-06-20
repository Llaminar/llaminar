/**
 * @file Test__ChatCompletionHandler.cpp
 * @brief Unit tests for ChatCompletionHandler
 *
 * Tests request parsing, sampling parameter wiring, inference flow,
 * error handling, and response formatting — all via mock interfaces.
 */

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "app/modes/ChatCompletionHandler.h"
#include "mocks/MockOrchestrationRunner.h"
#include "mocks/MockTokenizer.h"
#include "nlohmann/json.hpp"

#include <ctime>
#include <stdexcept>

using namespace llaminar2;
using namespace llaminar2::test;
using json = nlohmann::json;
using ::testing::_;
using ::testing::AnyNumber;
using ::testing::AtLeast;
using ::testing::Invoke;
using ::testing::Return;
using ::testing::Throw;

// =============================================================================
// Test fixture
// =============================================================================

class Test__ChatCompletionHandler : public ::testing::Test
{
protected:
    void SetUp() override
    {
        runner_ = std::make_unique<MockOrchestrationRunner>();
        tokenizer_ = std::make_unique<MockTokenizer>();

        // Default: runner is initialized
        runner_->simulateInitialized();
        EXPECT_CALL(*runner_, maybeApplyMoERebalance())
            .Times(AnyNumber())
            .WillRepeatedly(Return(true));
        EXPECT_CALL(*runner_, prefixStateProbe())
            .Times(AnyNumber())
            .WillRepeatedly(Return(PrefixRuntimeStateSnapshot{}));

        ON_CALL(*tokenizer_, encodeChat(_, _, _, _))
            .WillByDefault(Invoke([this](const std::vector<ChatMessage> &messages,
                                         bool add_generation_prompt,
                                         const std::string &tools_json,
                                         bool /*enable_thinking*/)
                                  { return tokenizer_->encodeChat(messages, add_generation_prompt, tools_json); }));
    }

    /// Build a handler using the current mocks
    std::unique_ptr<ChatCompletionHandler> makeHandler()
    {
        return std::make_unique<ChatCompletionHandler>(*runner_, *tokenizer_);
    }

    /// Build a minimal valid request JSON
    static std::string minimalRequest(json overrides = json::object())
    {
        json body = {
            {"messages", json::array({json{{"role", "user"}, {"content", "Hello"}}})}};
        body.merge_patch(overrides);
        return body.dump();
    }

    /// Helper: make a successful single-token decode result
    static GenerationResult makeToken(int32_t token_id, bool is_complete = false)
    {
        GenerationResult r;
        r.tokens = {token_id};
        r.is_complete = is_complete;
        return r;
    }

    static GenerationResult makeTokens(std::initializer_list<int32_t> token_ids,
                                       bool is_complete = false)
    {
        GenerationResult r;
        r.tokens.assign(token_ids.begin(), token_ids.end());
        r.is_complete = is_complete;
        return r;
    }

    /// Helper: make an empty decode result (no more tokens)
    static GenerationResult makeEmpty()
    {
        GenerationResult r;
        return r;
    }

    /// Helper: make a failed decode result
    static GenerationResult makeFailed(const std::string &error)
    {
        GenerationResult r;
        r.error = error;
        return r;
    }

    /// Helper: make a minimal template with Qwen-style thinking tags.
    static std::unique_ptr<ChatTemplate> makeThinkingTemplate()
    {
        return ChatTemplate::create(R"(
{%- for message in messages %}
<|im_start|>{{ message['role'] }}
{{ message['content'] }}<|im_end|>
{% endfor %}
{%- if add_generation_prompt %}
<|im_start|>assistant
{%- if enable_thinking is defined and enable_thinking is true %}
<think>
{%- else %}
<think>

</think>

{%- endif %}
{%- endif %})",
                                    "",
                                    "");
    }

    std::unique_ptr<MockOrchestrationRunner> runner_;
    std::unique_ptr<MockTokenizer> tokenizer_;
};

// =============================================================================
// Request parsing tests (static — no runner/tokenizer needed)
// =============================================================================

TEST_F(Test__ChatCompletionHandler, ParseRequest_InvalidJSON_Returns400)
{
    ChatCompletionResponse error;
    auto result = ChatCompletionHandler::parseRequest("not json{{{", error);

    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(error.http_status, 400);

    auto body = json::parse(error.json_body);
    EXPECT_TRUE(body.contains("error"));
    EXPECT_EQ(body["error"]["type"], "invalid_request_error");
}

TEST_F(Test__ChatCompletionHandler, ParseRequest_MissingMessages_Returns400)
{
    ChatCompletionResponse error;
    auto result = ChatCompletionHandler::parseRequest(R"({"max_tokens": 10})", error);

    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(error.http_status, 400);
}

TEST_F(Test__ChatCompletionHandler, ParseRequest_EmptyMessages_Returns400)
{
    ChatCompletionResponse error;
    auto result = ChatCompletionHandler::parseRequest(R"({"messages": []})", error);

    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(error.http_status, 400);
}

TEST_F(Test__ChatCompletionHandler, ParseRequest_MessageMissingRole_Returns400)
{
    ChatCompletionResponse error;
    auto result = ChatCompletionHandler::parseRequest(
        R"({"messages": [{"content": "hello"}]})", error);

    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(error.http_status, 400);
}

TEST_F(Test__ChatCompletionHandler, ParseRequest_MessageMissingContent_Returns400)
{
    ChatCompletionResponse error;
    auto result = ChatCompletionHandler::parseRequest(
        R"({"messages": [{"role": "user"}]})", error);

    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(error.http_status, 400);
}

TEST_F(Test__ChatCompletionHandler, ParseRequest_MinimalValid_Succeeds)
{
    ChatCompletionResponse error;
    auto result = ChatCompletionHandler::parseRequest(minimalRequest(), error);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->messages.size(), 1u);
    EXPECT_EQ(result->messages[0].role, "user");
    EXPECT_EQ(result->messages[0].content, "Hello");
}

TEST_F(Test__ChatCompletionHandler, ParseRequest_DefaultMaxTokens_IsSentinelForFullContext)
{
    // When the client does not specify max_tokens, parseRequest leaves the field
    // at the sentinel value -1. The handler then defaults to (context_window -
    // prompt_tokens) at decode time so the model can fill the remaining context.
    ChatCompletionResponse error;
    auto result = ChatCompletionHandler::parseRequest(minimalRequest(), error);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->max_tokens, -1);
}

TEST_F(Test__ChatCompletionHandler, ParseRequest_CustomMaxTokens)
{
    ChatCompletionResponse error;
    auto result = ChatCompletionHandler::parseRequest(
        minimalRequest({{"max_tokens", 42}}), error);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->max_tokens, 42);
}

// =============================================================================
// Sampling parameter parsing tests — THE BUG FIX
// =============================================================================

TEST_F(Test__ChatCompletionHandler, ParseRequest_DefaultTemperature_UsesSamplingParamsDefault)
{
    ChatCompletionResponse error;
    auto result = ChatCompletionHandler::parseRequest(minimalRequest(), error);

    ASSERT_TRUE(result.has_value());
    // When no temperature is specified, SamplingParams default (1.0) is preserved.
    // handleRequest() will later merge model-recommended defaults if applicable.
    SamplingParams defaults;
    EXPECT_FLOAT_EQ(result->sampling.temperature, defaults.temperature);
}

TEST_F(Test__ChatCompletionHandler, ParseRequest_ExplicitTemperature_Parsed)
{
    ChatCompletionResponse error;
    auto result = ChatCompletionHandler::parseRequest(
        minimalRequest({{"temperature", 0.7}}), error);

    ASSERT_TRUE(result.has_value());
    EXPECT_FLOAT_EQ(result->sampling.temperature, 0.7f);
    EXPECT_FALSE(result->sampling.is_greedy());
}

TEST_F(Test__ChatCompletionHandler, ParseRequest_ZeroTemperature_IsGreedy)
{
    ChatCompletionResponse error;
    auto result = ChatCompletionHandler::parseRequest(
        minimalRequest({{"temperature", 0.0}}), error);

    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result->sampling.is_greedy());
}

TEST_F(Test__ChatCompletionHandler, ParseRequest_TopP_Parsed)
{
    ChatCompletionResponse error;
    auto result = ChatCompletionHandler::parseRequest(
        minimalRequest({{"top_p", 0.9}}), error);

    ASSERT_TRUE(result.has_value());
    EXPECT_FLOAT_EQ(result->sampling.top_p, 0.9f);
}

TEST_F(Test__ChatCompletionHandler, ParseRequest_TopK_Parsed)
{
    ChatCompletionResponse error;
    auto result = ChatCompletionHandler::parseRequest(
        minimalRequest({{"top_k", 40}}), error);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->sampling.top_k, 40);
}

TEST_F(Test__ChatCompletionHandler, ParseRequest_Seed_Parsed)
{
    ChatCompletionResponse error;
    auto result = ChatCompletionHandler::parseRequest(
        minimalRequest({{"seed", 12345}}), error);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->sampling.seed, 12345u);
}

TEST_F(Test__ChatCompletionHandler, ParseRequest_DefaultTopP_Is1)
{
    ChatCompletionResponse error;
    auto result = ChatCompletionHandler::parseRequest(minimalRequest(), error);

    ASSERT_TRUE(result.has_value());
    EXPECT_FLOAT_EQ(result->sampling.top_p, 1.0f);
}

TEST_F(Test__ChatCompletionHandler, ParseRequest_DefaultTopK_Is0)
{
    ChatCompletionResponse error;
    auto result = ChatCompletionHandler::parseRequest(minimalRequest(), error);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->sampling.top_k, 0);
}

TEST_F(Test__ChatCompletionHandler, ParseRequest_DefaultSeed_Is0)
{
    ChatCompletionResponse error;
    auto result = ChatCompletionHandler::parseRequest(minimalRequest(), error);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->sampling.seed, 0u);
}

TEST_F(Test__ChatCompletionHandler, ParseRequest_AllSamplingParams_Combined)
{
    ChatCompletionResponse error;
    auto result = ChatCompletionHandler::parseRequest(
        minimalRequest({{"temperature", 0.8},
                        {"top_p", 0.95},
                        {"top_k", 50},
                        {"seed", 42}}),
        error);

    ASSERT_TRUE(result.has_value());
    EXPECT_FLOAT_EQ(result->sampling.temperature, 0.8f);
    EXPECT_FLOAT_EQ(result->sampling.top_p, 0.95f);
    EXPECT_EQ(result->sampling.top_k, 50);
    EXPECT_EQ(result->sampling.seed, 42u);
}

// =============================================================================
// Multi-message parsing
// =============================================================================

TEST_F(Test__ChatCompletionHandler, ParseRequest_MultiTurnMessages)
{
    json body = {
        {"messages", json::array({
                         json{{"role", "system"}, {"content", "You are helpful."}},
                         json{{"role", "user"}, {"content", "Hi"}},
                         json{{"role", "assistant"}, {"content", "Hello!"}},
                         json{{"role", "user"}, {"content", "Bye"}},
                     })}};

    ChatCompletionResponse error;
    auto result = ChatCompletionHandler::parseRequest(body.dump(), error);

    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result->messages.size(), 4u);
    EXPECT_EQ(result->messages[0].role, "system");
    EXPECT_EQ(result->messages[1].role, "user");
    EXPECT_EQ(result->messages[2].role, "assistant");
    EXPECT_EQ(result->messages[3].role, "user");
}

// =============================================================================
// Inference flow tests — sampling params wired to runner
// =============================================================================

TEST_F(Test__ChatCompletionHandler, HandleRequest_SetsSamplingParams_BeforePrefill)
{
    auto handler = makeHandler();

    // Track call order
    std::vector<std::string> call_order;

    ON_CALL(*tokenizer_, encodeChat(_, _, _))
        .WillByDefault(Return(std::vector<int>{1, 2, 3}));

    EXPECT_CALL(*runner_, setSamplingParams(_))
        .WillOnce(Invoke([&](const SamplingParams &)
                         { call_order.push_back("setSamplingParams"); }));

    EXPECT_CALL(*runner_, clearCache())
        .Times(2)
        .WillOnce(Invoke([&]()
                         { call_order.push_back("clearCache"); }))
        .WillOnce(Invoke([&]()
                         { call_order.push_back("cleanupClearCache"); }));

    EXPECT_CALL(*runner_, prefill(_))
        .WillOnce(Invoke([&](const std::vector<int32_t> &) -> bool
                         { call_order.push_back("prefill"); return true; }));

    EXPECT_CALL(*runner_, decodeStep())
        .WillOnce(Return(makeToken(42, /*is_complete=*/true)));

    ON_CALL(*tokenizer_, is_stop_token(_))
        .WillByDefault(Return(false));

    ChatCompletionRequest request;
    request.messages = {ChatMessage("user", "test")};
    request.max_tokens = 5;
    request.sampling.temperature = 0.5f;

    auto response = handler->handleRequest(request);
    EXPECT_TRUE(response.ok);

    // setSamplingParams must happen before prefill
    ASSERT_GE(call_order.size(), 3u);
    auto sp_pos = std::find(call_order.begin(), call_order.end(), "setSamplingParams");
    auto prefill_pos = std::find(call_order.begin(), call_order.end(), "prefill");
    EXPECT_LT(sp_pos, prefill_pos) << "setSamplingParams must be called before prefill";
}

TEST_F(Test__ChatCompletionHandler, HandleRequest_PassesSamplingParams_ToRunner)
{
    auto handler = makeHandler();

    ON_CALL(*tokenizer_, encodeChat(_, _, _))
        .WillByDefault(Return(std::vector<int>{1}));
    ON_CALL(*runner_, prefill(_))
        .WillByDefault(Return(true));
    ON_CALL(*tokenizer_, is_stop_token(_))
        .WillByDefault(Return(false));

    SamplingParams captured;
    EXPECT_CALL(*runner_, setSamplingParams(_))
        .WillOnce(Invoke([&](const SamplingParams &params)
                         { captured = params; }));

    EXPECT_CALL(*runner_, decodeStep())
        .WillOnce(Return(makeToken(42, true)));

    ChatCompletionRequest request;
    request.messages = {ChatMessage("user", "test")};
    request.max_tokens = 1;
    request.sampling.temperature = 0.8f;
    request.sampling.top_p = 0.95f;
    request.sampling.top_k = 40;
    request.sampling.seed = 999;
    request.sampling_set.temperature = true;
    request.sampling_set.top_p = true;
    request.sampling_set.top_k = true;
    request.sampling_set.seed = true;

    handler->handleRequest(request);

    EXPECT_FLOAT_EQ(captured.temperature, 0.8f);
    EXPECT_FLOAT_EQ(captured.top_p, 0.95f);
    EXPECT_EQ(captured.top_k, 40);
    EXPECT_EQ(captured.seed, 999u);
}

TEST_F(Test__ChatCompletionHandler, HandleRequest_GreedySampling_WhenTemp0)
{
    auto handler = makeHandler();

    ON_CALL(*tokenizer_, encodeChat(_, _, _))
        .WillByDefault(Return(std::vector<int>{1}));
    ON_CALL(*runner_, prefill(_))
        .WillByDefault(Return(true));
    ON_CALL(*tokenizer_, is_stop_token(_))
        .WillByDefault(Return(false));

    SamplingParams captured;
    EXPECT_CALL(*runner_, setSamplingParams(_))
        .WillOnce(Invoke([&](const SamplingParams &params)
                         { captured = params; }));

    EXPECT_CALL(*runner_, decodeStep())
        .WillOnce(Return(makeToken(42, true)));

    ChatCompletionRequest request;
    request.messages = {ChatMessage("user", "2+2?")};
    request.sampling.temperature = 0.0f;
    request.sampling_set.temperature = true;

    handler->handleRequest(request);

    EXPECT_TRUE(captured.is_greedy()) << "temperature=0 must result in greedy sampling";
}

// =============================================================================
// Full end-to-end: handleRawRequest
// =============================================================================

TEST_F(Test__ChatCompletionHandler, HandleRawRequest_UsesModelDefaultsWhenNoUserSampling)
{
    auto handler = makeHandler();

    ON_CALL(*tokenizer_, encodeChat(_, _, _))
        .WillByDefault(Return(std::vector<int>{10, 20}));
    ON_CALL(*runner_, prefill(_))
        .WillByDefault(Return(true));
    ON_CALL(*tokenizer_, is_stop_token(_))
        .WillByDefault(Return(false));
    ON_CALL(*tokenizer_, decode_token(42))
        .WillByDefault(Return("answer"));

    // When no user sampling params are specified, model defaults are used
    SamplingParams model_defaults;
    model_defaults.temperature = 0.6f;
    model_defaults.presence_penalty = 1.5f;
    ON_CALL(*runner_, getRecommendedSamplingParams())
        .WillByDefault(Return(model_defaults));

    SamplingParams captured;
    EXPECT_CALL(*runner_, setSamplingParams(_))
        .WillOnce(Invoke([&](const SamplingParams &p)
                         { captured = p; }));

    EXPECT_CALL(*runner_, decodeStep())
        .WillOnce(Return(makeToken(42)))
        .WillOnce(Return(makeToken(0, true)));

    auto response = handler->handleRawRequest(minimalRequest());

    EXPECT_TRUE(response.ok);
    EXPECT_EQ(response.http_status, 200);
    EXPECT_FLOAT_EQ(captured.temperature, 0.6f) << "Should use model default temperature";
    EXPECT_FLOAT_EQ(captured.presence_penalty, 1.5f) << "Should use model default penalty";
}

TEST_F(Test__ChatCompletionHandler, HandleRawRequest_RegenerateWithTemp_NotGreedy)
{
    auto handler = makeHandler();

    ON_CALL(*tokenizer_, encodeChat(_, _, _))
        .WillByDefault(Return(std::vector<int>{1}));
    ON_CALL(*runner_, prefill(_))
        .WillByDefault(Return(true));
    ON_CALL(*tokenizer_, is_stop_token(_))
        .WillByDefault(Return(false));

    SamplingParams captured;
    EXPECT_CALL(*runner_, setSamplingParams(_))
        .WillOnce(Invoke([&](const SamplingParams &p)
                         { captured = p; }));

    EXPECT_CALL(*runner_, decodeStep())
        .WillOnce(Return(makeToken(1, true)));

    auto response = handler->handleRawRequest(
        minimalRequest({{"temperature", 1.0}}));

    EXPECT_TRUE(response.ok);
    EXPECT_FALSE(captured.is_greedy());
    EXPECT_FLOAT_EQ(captured.temperature, 1.0f);
}

// =============================================================================
// Response format tests
// =============================================================================

TEST_F(Test__ChatCompletionHandler, HandleRequest_ResponseFormat_OpenAICompatible)
{
    auto handler = makeHandler();

    ON_CALL(*tokenizer_, encodeChat(_, _, _))
        .WillByDefault(Return(std::vector<int>{1, 2, 3}));
    ON_CALL(*runner_, prefill(_))
        .WillByDefault(Return(true));
    ON_CALL(*tokenizer_, is_stop_token(_))
        .WillByDefault(Return(false));
    ON_CALL(*tokenizer_, decode_token(100))
        .WillByDefault(Return("hello"));
    ON_CALL(*tokenizer_, decode_token(200))
        .WillByDefault(Return(" world"));

    EXPECT_CALL(*runner_, decodeStep())
        .WillOnce(Return(makeToken(100)))
        .WillOnce(Return(makeToken(200)))
        .WillOnce(Return(makeToken(0, true)));

    ChatCompletionRequest request;
    request.messages = {ChatMessage("user", "greet")};
    request.max_tokens = 10;

    auto response = handler->handleRequest(request);

    EXPECT_TRUE(response.ok);
    EXPECT_EQ(response.http_status, 200);

    auto body = json::parse(response.json_body);
    EXPECT_EQ(body["object"], "chat.completion");
    EXPECT_EQ(body["choices"][0]["message"]["role"], "assistant");
    EXPECT_EQ(body["choices"][0]["message"]["content"], "hello world");
    EXPECT_EQ(body["choices"][0]["finish_reason"], "stop");
    EXPECT_EQ(body["usage"]["prompt_tokens"], 3);
    EXPECT_EQ(body["usage"]["completion_tokens"], 3); // 100, 200, 0(stop)
    EXPECT_EQ(body["usage"]["total_tokens"], 6);
}

TEST_F(Test__ChatCompletionHandler, HandleRequest_ConsumesMultiTokenDecodeStep)
{
    auto handler = makeHandler();

    ON_CALL(*tokenizer_, encodeChat(_, _, _))
        .WillByDefault(Return(std::vector<int>{1, 2}));
    ON_CALL(*runner_, prefill(_))
        .WillByDefault(Return(true));
    ON_CALL(*tokenizer_, is_stop_token(_))
        .WillByDefault(Return(false));
    ON_CALL(*tokenizer_, decode_token(10))
        .WillByDefault(Return("A"));
    ON_CALL(*tokenizer_, decode_token(11))
        .WillByDefault(Return("B"));
    ON_CALL(*tokenizer_, decode_token(12))
        .WillByDefault(Return("C"));

    EXPECT_CALL(*runner_, setDecodeStepTokenBudget(2)).Times(1);
    EXPECT_CALL(*runner_, setDecodeStepTokenBudget(0)).Times(1);
    EXPECT_CALL(*runner_, decodeStep())
        .WillOnce(Return(makeTokens({10, 11, 12})));

    ChatCompletionRequest request;
    request.messages = {ChatMessage("user", "greet")};
    request.max_tokens = 2;
    request.enable_thinking = false;

    auto response = handler->handleRequest(request);

    ASSERT_TRUE(response.ok);
    auto body = json::parse(response.json_body);
    EXPECT_EQ(body["choices"][0]["message"]["content"], "AB");
    EXPECT_EQ(body["usage"]["completion_tokens"], 2);
}

TEST_F(Test__ChatCompletionHandler, HandleRequest_ProbesRuntimeStateForServeSummary)
{
    auto handler = makeHandler();

    ON_CALL(*tokenizer_, encodeChat(_, _, _))
        .WillByDefault(Return(std::vector<int>{1, 2}));
    ON_CALL(*runner_, prefill(_))
        .WillByDefault(Return(true));
    ON_CALL(*tokenizer_, is_stop_token(_))
        .WillByDefault(Return(false));
    ON_CALL(*tokenizer_, decode_token(10))
        .WillByDefault(Return("A"));

    PrefixRuntimeStateSnapshot snapshot;
    snapshot.mtp_config_enabled = true;
    snapshot.mtp_request.enabled = true;
    snapshot.mtp_request.adaptive_depth_enabled = true;
    snapshot.mtp_request.depth_policy_mode = "dynamic";
    snapshot.mtp_request.current_depth = 1;
    snapshot.mtp_request.min_depth = 1;
    snapshot.mtp_request.max_depth = 3;
    snapshot.mtp_request.draft_steps = 2;
    snapshot.mtp_request.accepted_tokens = 1;
    snapshot.mtp_request.rejected_tokens = 1;
    snapshot.mtp_request.acceptance_rate = 0.5;
    snapshot.mtp_verifier_runs = 2;
    snapshot.mtp_verifier_token_count = 4;

    EXPECT_CALL(*runner_, decodeStep())
        .WillOnce(Return(makeToken(10)));
    EXPECT_CALL(*runner_, prefixStateProbe())
        .Times(1)
        .WillOnce(Return(snapshot));

    ChatCompletionRequest request;
    request.messages = {ChatMessage("user", "test")};
    request.max_tokens = 1;
    request.enable_thinking = false;

    auto response = handler->handleRequest(request);

    EXPECT_TRUE(response.ok);
    EXPECT_EQ(response.http_status, 200);
}

TEST_F(Test__ChatCompletionHandler, HandleRequest_ReplacesInvalidUtf8InGeneratedContent)
{
    auto handler = makeHandler();

    ON_CALL(*tokenizer_, encodeChat(_, _, _))
        .WillByDefault(Return(std::vector<int>{1}));
    ON_CALL(*runner_, prefill(_))
        .WillByDefault(Return(true));
    ON_CALL(*tokenizer_, is_stop_token(_))
        .WillByDefault(Return(false));
    ON_CALL(*tokenizer_, decode_token(10))
        .WillByDefault(Return(std::string(1, static_cast<char>(0xA2))));

    EXPECT_CALL(*runner_, decodeStep())
        .WillOnce(Return(makeToken(10)))
        .WillOnce(Return(makeToken(0, true)));

    ChatCompletionRequest request;
    request.messages = {ChatMessage("user", "emit invalid byte")};
    request.max_tokens = 4;

    auto response = handler->handleRequest(request);

    EXPECT_TRUE(response.ok);
    EXPECT_EQ(response.http_status, 200);

    auto body = json::parse(response.json_body);
    EXPECT_EQ(body["choices"][0]["message"]["content"].get<std::string>(),
              std::string("\xEF\xBF\xBD"));
}

// =============================================================================
// Error handling tests
// =============================================================================

TEST_F(Test__ChatCompletionHandler, HandleRequest_EncodeEmpty_Returns500)
{
    auto handler = makeHandler();

    ON_CALL(*tokenizer_, encodeChat(_, _, _))
        .WillByDefault(Return(std::vector<int>{}));

    ChatCompletionRequest request;
    request.messages = {ChatMessage("user", "test")};

    auto response = handler->handleRequest(request);

    EXPECT_FALSE(response.ok);
    EXPECT_EQ(response.http_status, 500);

    auto body = json::parse(response.json_body);
    EXPECT_EQ(body["error"]["type"], "server_error");
}

TEST_F(Test__ChatCompletionHandler, HandleRequest_PrefillFails_Returns500)
{
    auto handler = makeHandler();

    ON_CALL(*tokenizer_, encodeChat(_, _, _))
        .WillByDefault(Return(std::vector<int>{1}));

    std::string prefill_error = "Out of memory";
    ON_CALL(*runner_, prefill(_))
        .WillByDefault(Return(false));
    ON_CALL(*runner_, lastError())
        .WillByDefault(testing::ReturnRef(prefill_error));

    ChatCompletionRequest request;
    request.messages = {ChatMessage("user", "test")};

    auto response = handler->handleRequest(request);

    EXPECT_FALSE(response.ok);
    EXPECT_EQ(response.http_status, 500);

    auto body = json::parse(response.json_body);
    EXPECT_TRUE(body["error"]["message"].get<std::string>().find("Out of memory") != std::string::npos);
}

TEST_F(Test__ChatCompletionHandler, HandleRequest_DecodeFails_Returns500)
{
    auto handler = makeHandler();

    ON_CALL(*tokenizer_, encodeChat(_, _, _))
        .WillByDefault(Return(std::vector<int>{1}));
    ON_CALL(*runner_, prefill(_))
        .WillByDefault(Return(true));

    EXPECT_CALL(*runner_, decodeStep())
        .WillOnce(Return(makeFailed("CUDA error")));

    ChatCompletionRequest request;
    request.messages = {ChatMessage("user", "test")};

    auto response = handler->handleRequest(request);

    EXPECT_FALSE(response.ok);
    EXPECT_EQ(response.http_status, 500);

    auto body = json::parse(response.json_body);
    EXPECT_TRUE(body["error"]["message"].get<std::string>().find("CUDA error") != std::string::npos);
}

// =============================================================================
// Stop token and max_tokens boundary tests
// =============================================================================

TEST_F(Test__ChatCompletionHandler, HandleRequest_StopsOnStopToken)
{
    auto handler = makeHandler();

    ON_CALL(*tokenizer_, encodeChat(_, _, _))
        .WillByDefault(Return(std::vector<int>{1}));
    ON_CALL(*runner_, prefill(_))
        .WillByDefault(Return(true));
    ON_CALL(*tokenizer_, decode_token(10))
        .WillByDefault(Return("a"));
    ON_CALL(*tokenizer_, decode_token(11))
        .WillByDefault(Return("b"));

    // Token 99 is a stop token
    ON_CALL(*tokenizer_, is_stop_token(10))
        .WillByDefault(Return(false));
    ON_CALL(*tokenizer_, is_stop_token(11))
        .WillByDefault(Return(false));
    ON_CALL(*tokenizer_, is_stop_token(99))
        .WillByDefault(Return(true));

    EXPECT_CALL(*runner_, decodeStep())
        .WillOnce(Return(makeToken(10)))
        .WillOnce(Return(makeToken(11)))
        .WillOnce(Return(makeToken(99)));

    ChatCompletionRequest request;
    request.messages = {ChatMessage("user", "test")};
    request.max_tokens = 100;

    auto response = handler->handleRequest(request);

    EXPECT_TRUE(response.ok);
    auto body = json::parse(response.json_body);
    // Stop token should NOT be decoded into text
    EXPECT_EQ(body["choices"][0]["message"]["content"], "ab");
    EXPECT_EQ(body["usage"]["completion_tokens"], 3);
}

TEST_F(Test__ChatCompletionHandler, HandleRequest_StopsAtMaxTokens)
{
    auto handler = makeHandler();

    ON_CALL(*tokenizer_, encodeChat(_, _, _))
        .WillByDefault(Return(std::vector<int>{1}));
    ON_CALL(*runner_, prefill(_))
        .WillByDefault(Return(true));
    ON_CALL(*tokenizer_, is_stop_token(_))
        .WillByDefault(Return(false));
    ON_CALL(*tokenizer_, decode_token(_))
        .WillByDefault(Return("x"));

    EXPECT_CALL(*runner_, decodeStep())
        .Times(3)
        .WillRepeatedly(Return(makeToken(10)));

    ChatCompletionRequest request;
    request.messages = {ChatMessage("user", "test")};
    request.max_tokens = 3;

    auto response = handler->handleRequest(request);

    EXPECT_TRUE(response.ok);
    auto body = json::parse(response.json_body);
    EXPECT_EQ(body["usage"]["completion_tokens"], 3);
}

TEST_F(Test__ChatCompletionHandler, HandleRequest_AppliesRebalanceHookAfterDecodeSteps)
{
    auto handler = makeHandler();

    ON_CALL(*tokenizer_, encodeChat(_, _, _))
        .WillByDefault(Return(std::vector<int>{1, 2}));
    ON_CALL(*runner_, prefill(_))
        .WillByDefault(Return(true));
    ON_CALL(*tokenizer_, is_stop_token(_))
        .WillByDefault(Return(false));
    ON_CALL(*tokenizer_, decode_token(_))
        .WillByDefault(Return("x"));

    EXPECT_CALL(*runner_, clearCache()).Times(2);
    EXPECT_CALL(*runner_, setSamplingParams(_)).Times(1);
    EXPECT_CALL(*runner_, decodeStep())
        .Times(3)
        .WillRepeatedly(Return(makeToken(42, false)));
    EXPECT_CALL(*runner_, maybeApplyMoERebalance())
        .Times(3)
        .WillRepeatedly(Return(true));

    ChatCompletionRequest request;
    request.messages = {ChatMessage("user", "Hello")};
    request.max_tokens = 3;
    request.enable_thinking = false;

    auto response = handler->handleRequest(request);
    EXPECT_TRUE(response.ok);
}

TEST_F(Test__ChatCompletionHandler, HandleRequest_EmptyDecode_StopsGracefully)
{
    auto handler = makeHandler();

    ON_CALL(*tokenizer_, encodeChat(_, _, _))
        .WillByDefault(Return(std::vector<int>{1}));
    ON_CALL(*runner_, prefill(_))
        .WillByDefault(Return(true));

    EXPECT_CALL(*runner_, decodeStep())
        .WillOnce(Return(makeEmpty()));

    ChatCompletionRequest request;
    request.messages = {ChatMessage("user", "test")};

    auto response = handler->handleRequest(request);

    EXPECT_TRUE(response.ok);
    auto body = json::parse(response.json_body);
    EXPECT_EQ(body["choices"][0]["message"]["content"], "");
    EXPECT_EQ(body["usage"]["completion_tokens"], 0);
}

// =============================================================================
// Cache clearing test
// =============================================================================

TEST_F(Test__ChatCompletionHandler, HandleRequest_ClearsCacheBeforeAndAfterRequest)
{
    auto handler = makeHandler();

    ON_CALL(*tokenizer_, encodeChat(_, _, _))
        .WillByDefault(Return(std::vector<int>{1}));
    ON_CALL(*runner_, prefill(_))
        .WillByDefault(Return(true));

    EXPECT_CALL(*runner_, clearCache()).Times(2);
    EXPECT_CALL(*runner_, decodeStep())
        .WillOnce(Return(makeToken(0, true)));
    ON_CALL(*tokenizer_, is_stop_token(_))
        .WillByDefault(Return(false));

    ChatCompletionRequest request;
    request.messages = {ChatMessage("user", "test")};

    handler->handleRequest(request);
}

TEST_F(Test__ChatCompletionHandler, HandleRequest_ConsecutiveRequestsResetCacheAndSamplingEachTime)
{
    auto handler = makeHandler();

    ON_CALL(*tokenizer_, encodeChat(_, _, _))
        .WillByDefault(Return(std::vector<int>{1}));
    ON_CALL(*tokenizer_, is_stop_token(_))
        .WillByDefault(Return(false));
    ON_CALL(*runner_, getRecommendedSamplingParams())
        .WillByDefault(Return(SamplingParams{}));

    EXPECT_CALL(*runner_, clearCache()).Times(4);
    EXPECT_CALL(*runner_, setSamplingParams(_)).Times(2);
    EXPECT_CALL(*runner_, prefill(_)).Times(2).WillRepeatedly(Return(true));
    EXPECT_CALL(*runner_, decodeStep()).Times(2).WillRepeatedly(Return(makeToken(0, true)));

    ChatCompletionRequest request;
    request.messages = {ChatMessage("user", "test")};
    request.max_tokens = 1;

    auto first = handler->handleRequest(request);
    auto second = handler->handleRequest(request);

    EXPECT_TRUE(first.ok);
    EXPECT_TRUE(second.ok);
}

TEST_F(Test__ChatCompletionHandler, HandleRequest_ExceptionAfterBoundaryReturns500AndResetsCache)
{
    auto handler = makeHandler();

    EXPECT_CALL(*runner_, clearCache()).Times(AtLeast(2));
    EXPECT_CALL(*runner_, setSamplingParams(_))
        .WillOnce(Throw(std::runtime_error("sampling setup exploded")));
    EXPECT_CALL(*runner_, prefill(_)).Times(0);

    ChatCompletionRequest request;
    request.messages = {ChatMessage("user", "test")};

    auto response = handler->handleRequest(request);

    EXPECT_FALSE(response.ok);
    EXPECT_EQ(response.http_status, 500);
    auto body = json::parse(response.json_body);
    EXPECT_EQ(body["error"]["type"], "server_error");
    EXPECT_NE(body["error"]["message"].get<std::string>().find("sampling setup exploded"), std::string::npos);
}

TEST_F(Test__ChatCompletionHandler, HandleRequest_ForwardsThinkingModeToTokenizer)
{
    auto handler = makeHandler();

    ON_CALL(*runner_, prefill(_))
        .WillByDefault(Return(true));
    ON_CALL(*tokenizer_, is_stop_token(_))
        .WillByDefault(Return(false));

    EXPECT_CALL(*runner_, clearCache()).Times(4);
    EXPECT_CALL(*tokenizer_, encodeChat(_, true, "", false))
        .WillOnce(Return(std::vector<int>{1}));
    EXPECT_CALL(*tokenizer_, encodeChat(_, true, "", true))
        .WillOnce(Return(std::vector<int>{1}));
    EXPECT_CALL(*runner_, decodeStep())
        .Times(2)
        .WillRepeatedly(Return(makeToken(0, true)));

    ChatCompletionRequest non_thinking;
    non_thinking.messages = {ChatMessage("user", "test")};
    non_thinking.enable_thinking = false;
    non_thinking.max_tokens = 1;

    ChatCompletionRequest thinking = non_thinking;
    thinking.enable_thinking = true;

    auto first = handler->handleRequest(non_thinking);
    auto second = handler->handleRequest(thinking);

    EXPECT_TRUE(first.ok);
    EXPECT_TRUE(second.ok);
}

// =============================================================================
// Full roundtrip: raw JSON → response
// =============================================================================

TEST_F(Test__ChatCompletionHandler, HandleRawRequest_InvalidJSON_Returns400)
{
    auto handler = makeHandler();

    auto response = handler->handleRawRequest("broken{json");

    EXPECT_FALSE(response.ok);
    EXPECT_EQ(response.http_status, 400);
}

TEST_F(Test__ChatCompletionHandler, HandleRawRequest_MissingMessages_Returns400)
{
    auto handler = makeHandler();

    auto response = handler->handleRawRequest(R"({"max_tokens": 10})");

    EXPECT_FALSE(response.ok);
    EXPECT_EQ(response.http_status, 400);
}

TEST_F(Test__ChatCompletionHandler, HandleRawRequest_FullPipeline)
{
    auto handler = makeHandler();

    ON_CALL(*tokenizer_, encodeChat(_, _, _))
        .WillByDefault(Return(std::vector<int>{1, 2}));
    ON_CALL(*runner_, prefill(_))
        .WillByDefault(Return(true));
    ON_CALL(*tokenizer_, is_stop_token(_))
        .WillByDefault(Return(false));
    ON_CALL(*tokenizer_, decode_token(42))
        .WillByDefault(Return("4"));

    SamplingParams captured;
    EXPECT_CALL(*runner_, setSamplingParams(_))
        .WillOnce(Invoke([&](const SamplingParams &p)
                         { captured = p; }));

    EXPECT_CALL(*runner_, decodeStep())
        .WillOnce(Return(makeToken(42)))
        .WillOnce(Return(makeToken(0, true)));

    json body = {
        {"messages", json::array({json{{"role", "user"}, {"content", "2+2?"}}})},
        {"max_tokens", 10},
        {"temperature", 0.0},
        {"top_k", 1}};

    auto response = handler->handleRawRequest(body.dump());

    EXPECT_TRUE(response.ok);
    EXPECT_TRUE(captured.is_greedy());

    auto resp_body = json::parse(response.json_body);
    EXPECT_EQ(resp_body["choices"][0]["message"]["content"], "4");
}

// =============================================================================
// Penalty parameter parsing
// =============================================================================

TEST_F(Test__ChatCompletionHandler, ParseRequest_PresencePenalty_Parsed)
{
    ChatCompletionResponse error;
    json body = {
        {"messages", json::array({json{{"role", "user"}, {"content", "Hi"}}})},
        {"presence_penalty", 1.5}};
    auto req = ChatCompletionHandler::parseRequest(body.dump(), error);
    ASSERT_TRUE(req.has_value());
    EXPECT_FLOAT_EQ(req->sampling.presence_penalty, 1.5f);
}

TEST_F(Test__ChatCompletionHandler, ParseRequest_FrequencyPenalty_Parsed)
{
    ChatCompletionResponse error;
    json body = {
        {"messages", json::array({json{{"role", "user"}, {"content", "Hi"}}})},
        {"frequency_penalty", 0.7}};
    auto req = ChatCompletionHandler::parseRequest(body.dump(), error);
    ASSERT_TRUE(req.has_value());
    EXPECT_FLOAT_EQ(req->sampling.frequency_penalty, 0.7f);
}

TEST_F(Test__ChatCompletionHandler, ParseRequest_BothPenalties_Parsed)
{
    ChatCompletionResponse error;
    json body = {
        {"messages", json::array({json{{"role", "user"}, {"content", "Hello"}}})},
        {"presence_penalty", 2.0},
        {"frequency_penalty", 0.3}};
    auto req = ChatCompletionHandler::parseRequest(body.dump(), error);
    ASSERT_TRUE(req.has_value());
    EXPECT_FLOAT_EQ(req->sampling.presence_penalty, 2.0f);
    EXPECT_FLOAT_EQ(req->sampling.frequency_penalty, 0.3f);
    EXPECT_TRUE(req->sampling.has_penalties());
}

TEST_F(Test__ChatCompletionHandler, ParseRequest_NegativePenalty_Parsed)
{
    ChatCompletionResponse error;
    json body = {
        {"messages", json::array({json{{"role", "user"}, {"content", "Hi"}}})},
        {"presence_penalty", -1.0}};
    auto req = ChatCompletionHandler::parseRequest(body.dump(), error);
    ASSERT_TRUE(req.has_value());
    EXPECT_FLOAT_EQ(req->sampling.presence_penalty, -1.0f)
        << "Negative penalties (token reward) should be accepted";
}

TEST_F(Test__ChatCompletionHandler, ParseRequest_ZeroPenalty_Parsed)
{
    ChatCompletionResponse error;
    json body = {
        {"messages", json::array({json{{"role", "user"}, {"content", "Hi"}}})},
        {"presence_penalty", 0.0},
        {"frequency_penalty", 0.0}};
    auto req = ChatCompletionHandler::parseRequest(body.dump(), error);
    ASSERT_TRUE(req.has_value());
    EXPECT_FLOAT_EQ(req->sampling.presence_penalty, 0.0f);
    EXPECT_FLOAT_EQ(req->sampling.frequency_penalty, 0.0f);
    EXPECT_FALSE(req->sampling.has_penalties());
}

TEST_F(Test__ChatCompletionHandler, ParseRequest_DefaultPenalties_AreZero)
{
    ChatCompletionResponse error;
    auto req = ChatCompletionHandler::parseRequest(minimalRequest(), error);
    ASSERT_TRUE(req.has_value());
    EXPECT_FLOAT_EQ(req->sampling.presence_penalty, 0.0f);
    EXPECT_FLOAT_EQ(req->sampling.frequency_penalty, 0.0f);
    EXPECT_FALSE(req->sampling.has_penalties());
}

TEST_F(Test__ChatCompletionHandler, ParseRequest_AllParams_Combined)
{
    ChatCompletionResponse error;
    json body = {
        {"messages", json::array({json{{"role", "user"}, {"content", "Hi"}}})},
        {"temperature", 0.7},
        {"top_p", 0.9},
        {"top_k", 50},
        {"seed", 123},
        {"presence_penalty", 1.5},
        {"frequency_penalty", 0.5},
        {"max_tokens", 256}};
    auto req = ChatCompletionHandler::parseRequest(body.dump(), error);
    ASSERT_TRUE(req.has_value());
    EXPECT_FLOAT_EQ(req->sampling.temperature, 0.7f);
    EXPECT_FLOAT_EQ(req->sampling.top_p, 0.9f);
    EXPECT_EQ(req->sampling.top_k, 50);
    EXPECT_EQ(req->sampling.seed, 123u);
    EXPECT_FLOAT_EQ(req->sampling.presence_penalty, 1.5f);
    EXPECT_FLOAT_EQ(req->sampling.frequency_penalty, 0.5f);
    EXPECT_EQ(req->max_tokens, 256);
}

// =============================================================================
// Model defaults merging
// =============================================================================

TEST_F(Test__ChatCompletionHandler, HandleRawRequest_ModelDefaultsMergedPerFieldWhenUserSpecifiesTemp)
{
    auto handler = makeHandler();

    ON_CALL(*tokenizer_, encodeChat(_, _, _))
        .WillByDefault(Return(std::vector<int>{10, 20}));
    ON_CALL(*runner_, prefill(_))
        .WillByDefault(Return(true));
    ON_CALL(*tokenizer_, is_stop_token(_))
        .WillByDefault(Return(false));
    ON_CALL(*tokenizer_, decode_token(_))
        .WillByDefault(Return("x"));

    SamplingParams model_defaults;
    model_defaults.temperature = 0.6f;
    model_defaults.presence_penalty = 1.5f;
    ON_CALL(*runner_, getRecommendedSamplingParams())
        .WillByDefault(Return(model_defaults));

    SamplingParams captured;
    EXPECT_CALL(*runner_, setSamplingParams(_))
        .WillOnce(Invoke([&](const SamplingParams &p)
                         { captured = p; }));

    EXPECT_CALL(*runner_, decodeStep())
        .WillOnce(Return(makeToken(42, true)));

    // User sets only temperature — other fields must still receive model defaults.
    json body = {
        {"messages", json::array({json{{"role", "user"}, {"content", "test"}}})},
        {"temperature", 0.3}};

    auto response = handler->handleRawRequest(body.dump());
    EXPECT_TRUE(response.ok);
    EXPECT_FLOAT_EQ(captured.temperature, 0.3f)
        << "Should use user-specified temperature";
    EXPECT_FLOAT_EQ(captured.presence_penalty, 1.5f)
        << "Model defaults must still be applied to fields the user did NOT specify";
}

TEST_F(Test__ChatCompletionHandler, HandleRawRequest_ModelDefaultsMergedPerFieldWhenUserSpecifiesPenalty)
{
    auto handler = makeHandler();

    ON_CALL(*tokenizer_, encodeChat(_, _, _))
        .WillByDefault(Return(std::vector<int>{10, 20}));
    ON_CALL(*runner_, prefill(_))
        .WillByDefault(Return(true));
    ON_CALL(*tokenizer_, is_stop_token(_))
        .WillByDefault(Return(false));
    ON_CALL(*tokenizer_, decode_token(_))
        .WillByDefault(Return("x"));

    SamplingParams model_defaults;
    model_defaults.temperature = 0.6f;
    model_defaults.presence_penalty = 1.5f;
    ON_CALL(*runner_, getRecommendedSamplingParams())
        .WillByDefault(Return(model_defaults));

    SamplingParams captured;
    EXPECT_CALL(*runner_, setSamplingParams(_))
        .WillOnce(Invoke([&](const SamplingParams &p)
                         { captured = p; }));

    EXPECT_CALL(*runner_, decodeStep())
        .WillOnce(Return(makeToken(42, true)));

    // User sets only presence_penalty — other fields must still receive model defaults.
    json body = {
        {"messages", json::array({json{{"role", "user"}, {"content", "test"}}})},
        {"presence_penalty", 0.5}};

    auto response = handler->handleRawRequest(body.dump());
    EXPECT_TRUE(response.ok);
    EXPECT_FLOAT_EQ(captured.presence_penalty, 0.5f)
        << "Should use user-specified penalty";
    EXPECT_FLOAT_EQ(captured.temperature, 0.6f)
        << "Non-specified params must receive model defaults (per-field merge)";
}

TEST_F(Test__ChatCompletionHandler, HandleRawRequest_PenaltiesPassedToRunner)
{
    auto handler = makeHandler();

    ON_CALL(*tokenizer_, encodeChat(_, _, _))
        .WillByDefault(Return(std::vector<int>{10}));
    ON_CALL(*runner_, prefill(_))
        .WillByDefault(Return(true));
    ON_CALL(*tokenizer_, is_stop_token(_))
        .WillByDefault(Return(false));
    ON_CALL(*tokenizer_, decode_token(_))
        .WillByDefault(Return("y"));

    ON_CALL(*runner_, getRecommendedSamplingParams())
        .WillByDefault(Return(SamplingParams{}));

    SamplingParams captured;
    EXPECT_CALL(*runner_, setSamplingParams(_))
        .WillOnce(Invoke([&](const SamplingParams &p)
                         { captured = p; }));

    EXPECT_CALL(*runner_, decodeStep())
        .WillOnce(Return(makeToken(42, true)));

    json body = {
        {"messages", json::array({json{{"role", "user"}, {"content", "test"}}})},
        {"presence_penalty", 2.0},
        {"frequency_penalty", 0.8},
        {"temperature", 0.9}};

    auto response = handler->handleRawRequest(body.dump());
    EXPECT_TRUE(response.ok);
    EXPECT_FLOAT_EQ(captured.presence_penalty, 2.0f);
    EXPECT_FLOAT_EQ(captured.frequency_penalty, 0.8f);
    EXPECT_FLOAT_EQ(captured.temperature, 0.9f);
}

// =============================================================================
// Thinking model response handling
// =============================================================================

TEST_F(Test__ChatCompletionHandler, HandleRequest_ThinkingModel_ExtractsReasoningContent)
{
    // Create a thinking model template
    std::string jinja_template = R"(
{%- for message in messages %}
<|im_start|>{{ message['role'] }}
{{ message['content'] }}<|im_end|>
{% endfor %}
{%- if add_generation_prompt %}
<|im_start|>assistant
{%- if enable_thinking is defined and enable_thinking is true %}
<think>
{%- else %}
<think>

</think>

{%- endif %}
{%- endif %})";

    auto tmpl = ChatTemplate::create(jinja_template, "", "");
    ASSERT_TRUE(tmpl->isThinkingModel());

    auto handler = makeHandler();

    ON_CALL(*tokenizer_, hasChatTemplate())
        .WillByDefault(Return(true));
    ON_CALL(*tokenizer_, getChatTemplate())
        .WillByDefault(::testing::ReturnRef(*tmpl));
    ON_CALL(*tokenizer_, encodeChat(_, _, _))
        .WillByDefault(Return(std::vector<int>{10, 20, 30}));
    ON_CALL(*runner_, prefill(_))
        .WillByDefault(Return(true));
    ON_CALL(*tokenizer_, is_stop_token(_))
        .WillByDefault(Return(false));
    ON_CALL(*runner_, getRecommendedSamplingParams())
        .WillByDefault(Return(SamplingParams{}));
    EXPECT_CALL(*runner_, setSamplingParams(_)).Times(1);

    // Simulate generating: "Let me think...\n</think>\n\nThe answer is 4"
    std::string thinking_text = "Let me think...\n</think>\n\nThe answer is 4";
    std::vector<std::string> token_strs;
    for (char c : thinking_text)
        token_strs.push_back(std::string(1, c));

    int token_id = 100;
    auto call_sequence = testing::InSequence{};
    for (size_t i = 0; i < token_strs.size(); ++i)
    {
        int tid = token_id + static_cast<int>(i);
        ON_CALL(*tokenizer_, decode_token(tid))
            .WillByDefault(Return(token_strs[i]));
        EXPECT_CALL(*runner_, decodeStep())
            .WillOnce(Return(makeToken(tid)))
            .RetiresOnSaturation();
    }
    // Final stop token
    EXPECT_CALL(*runner_, decodeStep())
        .WillOnce(Return(makeToken(0, true)))
        .RetiresOnSaturation();

    ChatCompletionRequest request;
    request.messages = {ChatMessage("user", "What is 2+2?")};
    request.max_tokens = 200;

    auto response = handler->handleRequest(request);
    ASSERT_TRUE(response.ok);
    ASSERT_EQ(response.http_status, 200);

    auto resp_body = json::parse(response.json_body);
    auto message = resp_body["choices"][0]["message"];

    EXPECT_EQ(message["content"], "The answer is 4")
        << "Content should be the part after </think> tag";
    EXPECT_TRUE(message.contains("reasoning_content"))
        << "Response should include reasoning_content for thinking models";
    EXPECT_EQ(message["reasoning_content"], "Let me think...\n")
        << "Reasoning content should be the part before </think>";
}

TEST_F(Test__ChatCompletionHandler, HandleRequest_NonThinkingModel_NoReasoningField)
{
    auto handler = makeHandler();

    auto tmpl = ChatTemplate::create(ChatTemplateType::CHATML);
    ASSERT_FALSE(tmpl->isThinkingModel());

    ON_CALL(*tokenizer_, hasChatTemplate())
        .WillByDefault(Return(true));
    ON_CALL(*tokenizer_, getChatTemplate())
        .WillByDefault(::testing::ReturnRef(*tmpl));
    ON_CALL(*tokenizer_, encodeChat(_, _, _))
        .WillByDefault(Return(std::vector<int>{10, 20}));
    ON_CALL(*runner_, prefill(_))
        .WillByDefault(Return(true));
    ON_CALL(*tokenizer_, is_stop_token(_))
        .WillByDefault(Return(false));
    ON_CALL(*tokenizer_, decode_token(42))
        .WillByDefault(Return("Hello!"));
    ON_CALL(*runner_, getRecommendedSamplingParams())
        .WillByDefault(Return(SamplingParams{}));
    EXPECT_CALL(*runner_, setSamplingParams(_)).Times(1);
    EXPECT_CALL(*runner_, decodeStep())
        .WillOnce(Return(makeToken(42)))
        .WillOnce(Return(makeToken(0, true)));

    ChatCompletionRequest request;
    request.messages = {ChatMessage("user", "Hi")};
    request.max_tokens = 10;

    auto response = handler->handleRequest(request);
    ASSERT_TRUE(response.ok);

    auto resp_body = json::parse(response.json_body);
    auto message = resp_body["choices"][0]["message"];

    EXPECT_EQ(message["content"], "Hello!");
    EXPECT_FALSE(message.contains("reasoning_content"))
        << "Non-thinking models should NOT include reasoning_content field";
}

TEST_F(Test__ChatCompletionHandler, HandleRequest_NoChatTemplate_NoReasoningField)
{
    auto handler = makeHandler();

    ON_CALL(*tokenizer_, hasChatTemplate())
        .WillByDefault(Return(false));
    ON_CALL(*tokenizer_, encodeChat(_, _, _))
        .WillByDefault(Return(std::vector<int>{10}));
    ON_CALL(*runner_, prefill(_))
        .WillByDefault(Return(true));
    ON_CALL(*tokenizer_, is_stop_token(_))
        .WillByDefault(Return(false));
    ON_CALL(*tokenizer_, decode_token(42))
        .WillByDefault(Return("response"));
    ON_CALL(*runner_, getRecommendedSamplingParams())
        .WillByDefault(Return(SamplingParams{}));
    EXPECT_CALL(*runner_, setSamplingParams(_)).Times(1);
    EXPECT_CALL(*runner_, decodeStep())
        .WillOnce(Return(makeToken(42)))
        .WillOnce(Return(makeToken(0, true)));

    ChatCompletionRequest request;
    request.messages = {ChatMessage("user", "test")};
    request.max_tokens = 10;

    auto response = handler->handleRequest(request);
    ASSERT_TRUE(response.ok);

    auto resp_body = json::parse(response.json_body);
    EXPECT_FALSE(resp_body["choices"][0]["message"].contains("reasoning_content"));
}

// =============================================================================
// Response format validation
// =============================================================================

TEST_F(Test__ChatCompletionHandler, HandleRawRequest_ResponseContainsAllOpenAIFields)
{
    auto handler = makeHandler();

    ON_CALL(*tokenizer_, encodeChat(_, _, _))
        .WillByDefault(Return(std::vector<int>{10, 20, 30}));
    ON_CALL(*runner_, prefill(_))
        .WillByDefault(Return(true));
    ON_CALL(*tokenizer_, is_stop_token(_))
        .WillByDefault(Return(false));
    ON_CALL(*tokenizer_, decode_token(_))
        .WillByDefault(Return("word"));
    ON_CALL(*runner_, getRecommendedSamplingParams())
        .WillByDefault(Return(SamplingParams{}));
    EXPECT_CALL(*runner_, setSamplingParams(_)).Times(1);
    EXPECT_CALL(*runner_, decodeStep())
        .WillOnce(Return(makeToken(42)))
        .WillOnce(Return(makeToken(43)))
        .WillOnce(Return(makeToken(0, true)));

    auto response = handler->handleRawRequest(minimalRequest());
    ASSERT_TRUE(response.ok);
    ASSERT_EQ(response.http_status, 200);

    auto resp = json::parse(response.json_body);

    // Required OpenAI-compatible fields
    EXPECT_TRUE(resp.contains("id"));
    EXPECT_EQ(resp["object"], "chat.completion");
    EXPECT_TRUE(resp.contains("choices"));
    EXPECT_TRUE(resp.contains("usage"));

    // Choices structure
    ASSERT_EQ(resp["choices"].size(), 1u);
    auto choice = resp["choices"][0];
    EXPECT_EQ(choice["index"], 0);
    EXPECT_TRUE(choice.contains("message"));
    EXPECT_TRUE(choice.contains("finish_reason"));
    EXPECT_EQ(choice["message"]["role"], "assistant");
    EXPECT_TRUE(choice["message"].contains("content"));

    // Usage structure
    auto usage = resp["usage"];
    EXPECT_EQ(usage["prompt_tokens"], 3);
    EXPECT_EQ(usage["completion_tokens"], 3); // 2 content + 1 stop token
    EXPECT_EQ(usage["total_tokens"], 6);
}

TEST_F(Test__ChatCompletionHandler, HandleRawRequest_FinishReasonStop)
{
    auto handler = makeHandler();

    ON_CALL(*tokenizer_, encodeChat(_, _, _))
        .WillByDefault(Return(std::vector<int>{10}));
    ON_CALL(*runner_, prefill(_))
        .WillByDefault(Return(true));
    ON_CALL(*runner_, getRecommendedSamplingParams())
        .WillByDefault(Return(SamplingParams{}));
    EXPECT_CALL(*runner_, setSamplingParams(_)).Times(1);

    // Immediately returns a stop token
    EXPECT_CALL(*runner_, decodeStep())
        .WillOnce(Return(makeToken(0, true)));
    ON_CALL(*tokenizer_, is_stop_token(_))
        .WillByDefault(Return(false));

    auto response = handler->handleRawRequest(minimalRequest());
    ASSERT_TRUE(response.ok);

    auto resp = json::parse(response.json_body);
    EXPECT_EQ(resp["choices"][0]["finish_reason"], "stop");
}

TEST_F(Test__ChatCompletionHandler, HandleRawRequest_FinishReasonLength)
{
    auto handler = makeHandler();

    ON_CALL(*tokenizer_, encodeChat(_, _, _))
        .WillByDefault(Return(std::vector<int>{10}));
    ON_CALL(*runner_, prefill(_))
        .WillByDefault(Return(true));
    ON_CALL(*tokenizer_, is_stop_token(_))
        .WillByDefault(Return(false));
    ON_CALL(*tokenizer_, decode_token(_))
        .WillByDefault(Return("w"));
    ON_CALL(*runner_, getRecommendedSamplingParams())
        .WillByDefault(Return(SamplingParams{}));
    EXPECT_CALL(*runner_, setSamplingParams(_)).Times(1);

    // Never stops — will hit max_tokens
    EXPECT_CALL(*runner_, decodeStep())
        .WillRepeatedly(Return(makeToken(42)));

    json body = {
        {"messages", json::array({json{{"role", "user"}, {"content", "Hi"}}})},
        {"max_tokens", 3}};

    auto response = handler->handleRawRequest(body.dump());
    ASSERT_TRUE(response.ok);

    auto resp = json::parse(response.json_body);
    EXPECT_EQ(resp["choices"][0]["finish_reason"], "length");
    EXPECT_EQ(resp["usage"]["completion_tokens"], 3);
}

// =============================================================================
// System message handling
// =============================================================================

TEST_F(Test__ChatCompletionHandler, ParseRequest_SystemMessage_Parsed)
{
    ChatCompletionResponse error;
    json body = {
        {"messages", json::array({json{{"role", "system"}, {"content", "You are helpful."}},
                                  json{{"role", "user"}, {"content", "Hi"}}})}};
    auto req = ChatCompletionHandler::parseRequest(body.dump(), error);
    ASSERT_TRUE(req.has_value());
    ASSERT_EQ(req->messages.size(), 2u);
    EXPECT_EQ(req->messages[0].role, "system");
    EXPECT_EQ(req->messages[0].content, "You are helpful.");
    EXPECT_EQ(req->messages[1].role, "user");
    EXPECT_EQ(req->messages[1].content, "Hi");
}

TEST_F(Test__ChatCompletionHandler, ParseRequest_MultiTurn_OrderPreserved)
{
    ChatCompletionResponse error;
    json body = {
        {"messages", json::array({json{{"role", "system"}, {"content", "sys"}},
                                  json{{"role", "user"}, {"content", "q1"}},
                                  json{{"role", "assistant"}, {"content", "a1"}},
                                  json{{"role", "user"}, {"content", "q2"}}})}};
    auto req = ChatCompletionHandler::parseRequest(body.dump(), error);
    ASSERT_TRUE(req.has_value());
    ASSERT_EQ(req->messages.size(), 4u);
    EXPECT_EQ(req->messages[0].role, "system");
    EXPECT_EQ(req->messages[1].role, "user");
    EXPECT_EQ(req->messages[2].role, "assistant");
    EXPECT_EQ(req->messages[3].role, "user");
}

// =============================================================================
// Context window validation + reporting tests
// =============================================================================

TEST_F(Test__ChatCompletionHandler, HandleRequest_PromptExceedsContextWindow_Returns400)
{
    // Set a small context window
    OrchestrationConfig small_ctx_config;
    small_ctx_config.max_seq_len = 8;
    runner_->setConfig(small_ctx_config);

    auto handler = makeHandler();

    // Encode returns 10 tokens > max_seq_len of 8
    ON_CALL(*tokenizer_, encodeChat(_, _, _))
        .WillByDefault(Return(std::vector<int>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}));

    ChatCompletionRequest request;
    request.messages = {ChatMessage("user", "long prompt")};

    auto response = handler->handleRequest(request);

    EXPECT_FALSE(response.ok);
    EXPECT_EQ(response.http_status, 400);

    auto body = json::parse(response.json_body);
    EXPECT_EQ(body["error"]["type"], "invalid_request_error");
    EXPECT_EQ(body["error"]["param"], "messages");
    // Message should contain both the prompt size and context window size
    std::string msg = body["error"]["message"];
    EXPECT_NE(msg.find("10"), std::string::npos) << "Should mention prompt token count";
    EXPECT_NE(msg.find("8"), std::string::npos) << "Should mention context window size";
}

TEST_F(Test__ChatCompletionHandler, HandleRequest_PromptExactlyFitsContextWindow_Succeeds)
{
    OrchestrationConfig config;
    config.max_seq_len = 5;
    runner_->setConfig(config);

    auto handler = makeHandler();

    // Encode returns exactly 5 tokens = max_seq_len
    ON_CALL(*tokenizer_, encodeChat(_, _, _))
        .WillByDefault(Return(std::vector<int>{1, 2, 3, 4, 5}));
    ON_CALL(*runner_, prefill(_))
        .WillByDefault(Return(true));
    ON_CALL(*tokenizer_, is_stop_token(_))
        .WillByDefault(Return(false));
    ON_CALL(*tokenizer_, decode_token(_))
        .WillByDefault(Return("x"));

    EXPECT_CALL(*runner_, decodeStep())
        .WillOnce(Return(makeToken(42, true)));

    ChatCompletionRequest request;
    request.messages = {ChatMessage("user", "hi")};
    request.max_tokens = 1;

    auto response = handler->handleRequest(request);

    EXPECT_TRUE(response.ok);
    EXPECT_EQ(response.http_status, 200);
}

TEST_F(Test__ChatCompletionHandler, HandleRequest_ResponseContainsContextWindow)
{
    OrchestrationConfig config;
    config.max_seq_len = 2048;
    runner_->setConfig(config);

    auto handler = makeHandler();

    ON_CALL(*tokenizer_, encodeChat(_, _, _))
        .WillByDefault(Return(std::vector<int>{1, 2, 3}));
    ON_CALL(*runner_, prefill(_))
        .WillByDefault(Return(true));
    ON_CALL(*tokenizer_, is_stop_token(_))
        .WillByDefault(Return(false));
    ON_CALL(*tokenizer_, decode_token(_))
        .WillByDefault(Return("x"));

    EXPECT_CALL(*runner_, decodeStep())
        .WillOnce(Return(makeToken(42)))
        .WillOnce(Return(makeToken(0, true)));

    ChatCompletionRequest request;
    request.messages = {ChatMessage("user", "test")};
    request.max_tokens = 10;

    auto response = handler->handleRequest(request);

    EXPECT_TRUE(response.ok);
    auto body = json::parse(response.json_body);
    ASSERT_TRUE(body["usage"].contains("context_window"));
    EXPECT_EQ(body["usage"]["context_window"], 2048);
}

TEST_F(Test__ChatCompletionHandler, HandleRequest_ResponseContainsContextUsed)
{
    auto handler = makeHandler();

    // 3 prompt tokens
    ON_CALL(*tokenizer_, encodeChat(_, _, _))
        .WillByDefault(Return(std::vector<int>{1, 2, 3}));
    ON_CALL(*runner_, prefill(_))
        .WillByDefault(Return(true));
    ON_CALL(*tokenizer_, is_stop_token(_))
        .WillByDefault(Return(false));
    ON_CALL(*tokenizer_, decode_token(_))
        .WillByDefault(Return("x"));

    // 2 completion tokens (token 42, then EOS)
    EXPECT_CALL(*runner_, decodeStep())
        .WillOnce(Return(makeToken(42)))
        .WillOnce(Return(makeToken(0, true)));

    ChatCompletionRequest request;
    request.messages = {ChatMessage("user", "test")};
    request.max_tokens = 10;

    auto response = handler->handleRequest(request);

    EXPECT_TRUE(response.ok);
    auto body = json::parse(response.json_body);
    ASSERT_TRUE(body["usage"].contains("context_used"));
    // context_used = prompt_tokens(3) + completion_tokens(2)
    EXPECT_EQ(body["usage"]["context_used"], 5);
}

TEST_F(Test__ChatCompletionHandler, HandleRequest_ContextUsedEqualsPromptPlusCompletion)
{
    auto handler = makeHandler();

    // 5 prompt tokens
    ON_CALL(*tokenizer_, encodeChat(_, _, _))
        .WillByDefault(Return(std::vector<int>{10, 20, 30, 40, 50}));
    ON_CALL(*runner_, prefill(_))
        .WillByDefault(Return(true));
    ON_CALL(*tokenizer_, is_stop_token(_))
        .WillByDefault(Return(false));
    ON_CALL(*tokenizer_, decode_token(_))
        .WillByDefault(Return("y"));

    // 3 completion tokens
    EXPECT_CALL(*runner_, decodeStep())
        .WillOnce(Return(makeToken(100)))
        .WillOnce(Return(makeToken(101)))
        .WillOnce(Return(makeToken(102, true)));

    ChatCompletionRequest request;
    request.messages = {ChatMessage("user", "hello world")};
    request.max_tokens = 50;

    auto response = handler->handleRequest(request);

    EXPECT_TRUE(response.ok);
    auto body = json::parse(response.json_body);
    int prompt = body["usage"]["prompt_tokens"];
    int completion = body["usage"]["completion_tokens"];
    int used = body["usage"]["context_used"];
    EXPECT_EQ(prompt, 5);
    EXPECT_EQ(completion, 3);
    EXPECT_EQ(used, prompt + completion);
}

TEST_F(Test__ChatCompletionHandler, HandleRawRequest_PromptExceedsContext_Returns400)
{
    OrchestrationConfig config;
    config.max_seq_len = 4;
    runner_->setConfig(config);

    auto handler = makeHandler();

    ON_CALL(*tokenizer_, encodeChat(_, _, _))
        .WillByDefault(Return(std::vector<int>{1, 2, 3, 4, 5}));

    auto response = handler->handleRawRequest(minimalRequest());

    EXPECT_FALSE(response.ok);
    EXPECT_EQ(response.http_status, 400);
    auto body = json::parse(response.json_body);
    EXPECT_EQ(body["error"]["type"], "invalid_request_error");
}

// =============================================================================
// Response metadata tests (Phase 1: id, model, created, system_fingerprint)
// =============================================================================

TEST_F(Test__ChatCompletionHandler, ResponseMetadata_UniqueId_HasChatcmplPrefix)
{
    std::string id = ChatCompletionHandler::generateRequestId();
    EXPECT_TRUE(id.substr(0, 9) == "chatcmpl-")
        << "ID should start with 'chatcmpl-', got: " << id;
    EXPECT_GT(id.size(), 9u) << "ID should have hex suffix after prefix";
}

TEST_F(Test__ChatCompletionHandler, ResponseMetadata_UniqueId_TwoCallsDiffer)
{
    std::string id1 = ChatCompletionHandler::generateRequestId();
    std::string id2 = ChatCompletionHandler::generateRequestId();
    EXPECT_NE(id1, id2) << "Two generated IDs should be different";
}

TEST_F(Test__ChatCompletionHandler, ResponseMetadata_ModelFieldPresent)
{
    auto handler = std::make_unique<ChatCompletionHandler>(*runner_, *tokenizer_, "test-model");

    ON_CALL(*tokenizer_, encodeChat(_, _, _))
        .WillByDefault(Return(std::vector<int>{1}));
    ON_CALL(*runner_, prefill(_))
        .WillByDefault(Return(true));
    ON_CALL(*tokenizer_, is_stop_token(_))
        .WillByDefault(Return(false));
    ON_CALL(*tokenizer_, decode_token(_))
        .WillByDefault(Return("hi"));

    EXPECT_CALL(*runner_, decodeStep())
        .WillOnce(Return(makeToken(1, true)));

    ChatCompletionRequest request;
    request.messages = {ChatMessage("user", "test")};

    auto response = handler->handleRequest(request);
    EXPECT_TRUE(response.ok);

    auto body = json::parse(response.json_body);
    EXPECT_EQ(body["model"], "test-model");
}

TEST_F(Test__ChatCompletionHandler, ResponseMetadata_CreatedTimestamp)
{
    auto handler = makeHandler();

    ON_CALL(*tokenizer_, encodeChat(_, _, _))
        .WillByDefault(Return(std::vector<int>{1}));
    ON_CALL(*runner_, prefill(_))
        .WillByDefault(Return(true));
    ON_CALL(*tokenizer_, is_stop_token(_))
        .WillByDefault(Return(false));

    EXPECT_CALL(*runner_, decodeStep())
        .WillOnce(Return(makeToken(1, true)));

    ChatCompletionRequest request;
    request.messages = {ChatMessage("user", "test")};

    int64_t before = static_cast<int64_t>(std::time(nullptr));
    auto response = handler->handleRequest(request);
    int64_t after = static_cast<int64_t>(std::time(nullptr));

    EXPECT_TRUE(response.ok);
    auto body = json::parse(response.json_body);
    EXPECT_TRUE(body.contains("created"));
    int64_t created = body["created"].get<int64_t>();
    EXPECT_GE(created, before);
    EXPECT_LE(created, after);
}

TEST_F(Test__ChatCompletionHandler, ResponseMetadata_SystemFingerprint)
{
    auto handler = makeHandler();

    ON_CALL(*tokenizer_, encodeChat(_, _, _))
        .WillByDefault(Return(std::vector<int>{1}));
    ON_CALL(*runner_, prefill(_))
        .WillByDefault(Return(true));
    ON_CALL(*tokenizer_, is_stop_token(_))
        .WillByDefault(Return(false));

    EXPECT_CALL(*runner_, decodeStep())
        .WillOnce(Return(makeToken(1, true)));

    ChatCompletionRequest request;
    request.messages = {ChatMessage("user", "test")};

    auto response = handler->handleRequest(request);
    EXPECT_TRUE(response.ok);

    auto body = json::parse(response.json_body);
    EXPECT_EQ(body["system_fingerprint"], "llaminar-v2");
}

TEST_F(Test__ChatCompletionHandler, ResponseMetadata_IdIsChatcmplFormat)
{
    auto handler = makeHandler();

    ON_CALL(*tokenizer_, encodeChat(_, _, _))
        .WillByDefault(Return(std::vector<int>{1}));
    ON_CALL(*runner_, prefill(_))
        .WillByDefault(Return(true));
    ON_CALL(*tokenizer_, is_stop_token(_))
        .WillByDefault(Return(false));

    EXPECT_CALL(*runner_, decodeStep())
        .WillOnce(Return(makeToken(1, true)));

    ChatCompletionRequest request;
    request.messages = {ChatMessage("user", "test")};

    auto response = handler->handleRequest(request);
    auto body = json::parse(response.json_body);
    std::string id = body["id"].get<std::string>();
    EXPECT_TRUE(id.substr(0, 9) == "chatcmpl-") << "Response id should have chatcmpl- prefix, got: " << id;
}

TEST_F(Test__ChatCompletionHandler, ResponseMetadata_ModelFromRequest_OverridesDefault)
{
    auto handler = std::make_unique<ChatCompletionHandler>(*runner_, *tokenizer_, "default-model");

    ON_CALL(*tokenizer_, encodeChat(_, _, _))
        .WillByDefault(Return(std::vector<int>{1}));
    ON_CALL(*runner_, prefill(_))
        .WillByDefault(Return(true));
    ON_CALL(*tokenizer_, is_stop_token(_))
        .WillByDefault(Return(false));

    EXPECT_CALL(*runner_, decodeStep())
        .WillOnce(Return(makeToken(1, true)));

    ChatCompletionRequest request;
    request.messages = {ChatMessage("user", "test")};
    request.model = "user-specified-model";

    auto response = handler->handleRequest(request);
    auto body = json::parse(response.json_body);
    EXPECT_EQ(body["model"], "user-specified-model");
}

// =============================================================================
// Request parsing: stream and enable_thinking (Phase 2)
// =============================================================================

TEST_F(Test__ChatCompletionHandler, ParseRequest_StreamTrue_Parsed)
{
    ChatCompletionResponse error;
    auto result = ChatCompletionHandler::parseRequest(
        minimalRequest({{"stream", true}}), error);
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result->stream);
}

TEST_F(Test__ChatCompletionHandler, ParseRequest_StreamFalse_Parsed)
{
    ChatCompletionResponse error;
    auto result = ChatCompletionHandler::parseRequest(
        minimalRequest({{"stream", false}}), error);
    ASSERT_TRUE(result.has_value());
    EXPECT_FALSE(result->stream);
}

TEST_F(Test__ChatCompletionHandler, ParseRequest_StreamDefault_IsFalse)
{
    ChatCompletionResponse error;
    auto result = ChatCompletionHandler::parseRequest(minimalRequest(), error);
    ASSERT_TRUE(result.has_value());
    EXPECT_FALSE(result->stream);
}

TEST_F(Test__ChatCompletionHandler, ParseRequest_EnableThinkingFalse_Parsed)
{
    ChatCompletionResponse error;
    auto result = ChatCompletionHandler::parseRequest(
        minimalRequest({{"enable_thinking", false}}), error);
    ASSERT_TRUE(result.has_value());
    EXPECT_FALSE(result->enable_thinking);
}

TEST_F(Test__ChatCompletionHandler, ParseRequest_EnableThinkingDefault_IsTrue)
{
    ChatCompletionResponse error;
    auto result = ChatCompletionHandler::parseRequest(minimalRequest(), error);
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result->enable_thinking);
}

TEST_F(Test__ChatCompletionHandler, ParseRequest_ModelField_Parsed)
{
    ChatCompletionResponse error;
    auto result = ChatCompletionHandler::parseRequest(
        minimalRequest({{"model", "gpt-4"}}), error);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->model, "gpt-4");
}

TEST_F(Test__ChatCompletionHandler, ParseRequest_ModelField_DefaultEmpty)
{
    ChatCompletionResponse error;
    auto result = ChatCompletionHandler::parseRequest(minimalRequest(), error);
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result->model.empty());
}

// =============================================================================
// Streaming response tests (Phase 3)
// =============================================================================

TEST_F(Test__ChatCompletionHandler, Streaming_FirstChunk_HasRoleAssistant)
{
    auto handler = makeHandler();

    ON_CALL(*tokenizer_, encodeChat(_, _, _))
        .WillByDefault(Return(std::vector<int>{1}));
    ON_CALL(*runner_, prefill(_))
        .WillByDefault(Return(true));
    ON_CALL(*tokenizer_, is_stop_token(_))
        .WillByDefault(Return(false));

    EXPECT_CALL(*runner_, decodeStep())
        .WillOnce(Return(makeToken(1, true)));

    ChatCompletionRequest request;
    request.messages = {ChatMessage("user", "test")};
    request.stream = true;
    request.max_tokens = 5;

    std::vector<std::string> chunks;
    auto cb = [&](const std::string &line) -> bool
    {
        chunks.push_back(line);
        return true;
    };

    auto response = handler->handleStreamingRequest(request, cb);
    EXPECT_TRUE(response.ok);

    // First chunk should be role announcement
    ASSERT_GE(chunks.size(), 1u);
    // Parse first SSE line: "data: {...}\n\n"
    std::string first = chunks[0];
    ASSERT_TRUE(first.substr(0, 6) == "data: ") << "SSE line should start with 'data: '";
    auto first_json = json::parse(first.substr(6, first.find("\n\n") - 6));
    EXPECT_EQ(first_json["choices"][0]["delta"]["role"], "assistant");
    EXPECT_EQ(first_json["object"], "chat.completion.chunk");
}

TEST_F(Test__ChatCompletionHandler, Streaming_ProbesRuntimeStateForServeSummary)
{
    auto handler = makeHandler();

    ON_CALL(*tokenizer_, encodeChat(_, _, _))
        .WillByDefault(Return(std::vector<int>{1}));
    ON_CALL(*runner_, prefill(_))
        .WillByDefault(Return(true));
    ON_CALL(*tokenizer_, is_stop_token(_))
        .WillByDefault(Return(false));
    ON_CALL(*tokenizer_, decode_token(10))
        .WillByDefault(Return("A"));

    PrefixRuntimeStateSnapshot snapshot;
    snapshot.prefix_cache_config_enabled = true;
    snapshot.prefix_request.enabled = true;
    snapshot.prefix_request.requested_tokens = 1;
    snapshot.prefix_request.matched_tokens = 1;
    snapshot.prefix_request.hit = true;
    snapshot.prefix_request.storage_tier = "ram";

    EXPECT_CALL(*runner_, decodeStep())
        .WillOnce(Return(makeToken(10)));
    EXPECT_CALL(*runner_, prefixStateProbe())
        .Times(1)
        .WillOnce(Return(snapshot));

    ChatCompletionRequest request;
    request.messages = {ChatMessage("user", "test")};
    request.stream = true;
    request.max_tokens = 1;
    request.enable_thinking = false;

    std::vector<std::string> chunks;
    auto cb = [&](const std::string &line) -> bool
    {
        chunks.push_back(line);
        return true;
    };

    auto response = handler->handleStreamingRequest(request, cb);

    EXPECT_TRUE(response.ok);
    EXPECT_EQ(response.http_status, 200);
    ASSERT_FALSE(chunks.empty());
    EXPECT_EQ(chunks.back(), "data: [DONE]\n\n");
}

TEST_F(Test__ChatCompletionHandler, Streaming_TokenByToken_ContentInDelta)
{
    auto handler = makeHandler();

    ON_CALL(*tokenizer_, encodeChat(_, _, _))
        .WillByDefault(Return(std::vector<int>{1}));
    ON_CALL(*runner_, prefill(_))
        .WillByDefault(Return(true));
    ON_CALL(*tokenizer_, is_stop_token(_))
        .WillByDefault(Return(false));
    ON_CALL(*tokenizer_, decode_token(10))
        .WillByDefault(Return("Hello"));
    ON_CALL(*tokenizer_, decode_token(20))
        .WillByDefault(Return(" world"));

    EXPECT_CALL(*runner_, decodeStep())
        .WillOnce(Return(makeToken(10)))
        .WillOnce(Return(makeToken(20)))
        .WillOnce(Return(makeToken(0, true)));

    ChatCompletionRequest request;
    request.messages = {ChatMessage("user", "test")};
    request.stream = true;
    request.max_tokens = 10;

    std::vector<std::string> chunks;
    auto cb = [&](const std::string &line) -> bool
    {
        chunks.push_back(line);
        return true;
    };

    handler->handleStreamingRequest(request, cb);

    // chunks: role, "Hello", " world", finish, [DONE]
    ASSERT_GE(chunks.size(), 4u);

    // Second chunk: "Hello"
    auto c1 = json::parse(chunks[1].substr(6, chunks[1].find("\n\n") - 6));
    EXPECT_EQ(c1["choices"][0]["delta"]["content"], "Hello");

    // Third chunk: " world"
    auto c2 = json::parse(chunks[2].substr(6, chunks[2].find("\n\n") - 6));
    EXPECT_EQ(c2["choices"][0]["delta"]["content"], " world");
}

TEST_F(Test__ChatCompletionHandler, Streaming_EmitsEachTokenFromMultiTokenDecodeStep)
{
    auto handler = makeHandler();

    ON_CALL(*tokenizer_, encodeChat(_, _, _))
        .WillByDefault(Return(std::vector<int>{1}));
    ON_CALL(*runner_, prefill(_))
        .WillByDefault(Return(true));
    ON_CALL(*tokenizer_, is_stop_token(_))
        .WillByDefault(Return(false));
    ON_CALL(*tokenizer_, decode_token(10))
        .WillByDefault(Return("A"));
    ON_CALL(*tokenizer_, decode_token(11))
        .WillByDefault(Return("B"));
    ON_CALL(*tokenizer_, decode_token(12))
        .WillByDefault(Return("C"));

    EXPECT_CALL(*runner_, setDecodeStepTokenBudget(2)).Times(1);
    EXPECT_CALL(*runner_, setDecodeStepTokenBudget(0)).Times(1);
    EXPECT_CALL(*runner_, decodeStep())
        .WillOnce(Return(makeTokens({10, 11, 12})));

    ChatCompletionRequest request;
    request.messages = {ChatMessage("user", "test")};
    request.stream = true;
    request.max_tokens = 2;
    request.enable_thinking = false;

    std::vector<std::string> chunks;
    auto cb = [&](const std::string &line) -> bool
    {
        chunks.push_back(line);
        return true;
    };

    auto response = handler->handleStreamingRequest(request, cb);
    ASSERT_TRUE(response.ok);
    ASSERT_GE(chunks.size(), 4u);

    auto c1 = json::parse(chunks[1].substr(6, chunks[1].find("\n\n") - 6));
    auto c2 = json::parse(chunks[2].substr(6, chunks[2].find("\n\n") - 6));
    EXPECT_EQ(c1["choices"][0]["delta"]["content"], "A");
    EXPECT_EQ(c2["choices"][0]["delta"]["content"], "B");
}

TEST_F(Test__ChatCompletionHandler, Streaming_ReplacesInvalidUtf8InContentDelta)
{
    auto handler = makeHandler();

    ON_CALL(*tokenizer_, encodeChat(_, _, _))
        .WillByDefault(Return(std::vector<int>{1}));
    ON_CALL(*runner_, prefill(_))
        .WillByDefault(Return(true));
    ON_CALL(*tokenizer_, is_stop_token(_))
        .WillByDefault(Return(false));
    ON_CALL(*tokenizer_, decode_token(10))
        .WillByDefault(Return(std::string(1, static_cast<char>(0xAA))));

    EXPECT_CALL(*runner_, decodeStep())
        .WillOnce(Return(makeToken(10)))
        .WillOnce(Return(makeToken(0, true)));

    ChatCompletionRequest request;
    request.messages = {ChatMessage("user", "stream invalid byte")};
    request.stream = true;
    request.max_tokens = 4;

    std::vector<std::string> chunks;
    auto cb = [&](const std::string &line) -> bool
    {
        chunks.push_back(line);
        return true;
    };

    auto response = handler->handleStreamingRequest(request, cb);

    EXPECT_TRUE(response.ok);
    ASSERT_GE(chunks.size(), 4u); // role, content, finish, [DONE]

    auto content_chunk = json::parse(chunks[1].substr(6, chunks[1].find("\n\n") - 6));
    EXPECT_EQ(content_chunk["choices"][0]["delta"]["content"].get<std::string>(),
              std::string("\xEF\xBF\xBD"));
}

TEST_F(Test__ChatCompletionHandler, Streaming_FinalChunk_HasFinishReason)
{
    auto handler = makeHandler();

    ON_CALL(*tokenizer_, encodeChat(_, _, _))
        .WillByDefault(Return(std::vector<int>{1}));
    ON_CALL(*runner_, prefill(_))
        .WillByDefault(Return(true));
    ON_CALL(*tokenizer_, is_stop_token(_))
        .WillByDefault(Return(false));
    ON_CALL(*tokenizer_, decode_token(_))
        .WillByDefault(Return("x"));

    EXPECT_CALL(*runner_, decodeStep())
        .WillOnce(Return(makeToken(1, true)));

    ChatCompletionRequest request;
    request.messages = {ChatMessage("user", "test")};
    request.stream = true;

    std::vector<std::string> chunks;
    auto cb = [&](const std::string &line) -> bool
    {
        chunks.push_back(line);
        return true;
    };

    handler->handleStreamingRequest(request, cb);

    // Last real chunk (before [DONE]) should have finish_reason
    ASSERT_GE(chunks.size(), 3u);          // role + finish + [DONE]
    size_t finish_idx = chunks.size() - 2; // before [DONE]
    auto finish_json = json::parse(chunks[finish_idx].substr(6, chunks[finish_idx].find("\n\n") - 6));
    EXPECT_EQ(finish_json["choices"][0]["finish_reason"], "stop");
}

TEST_F(Test__ChatCompletionHandler, Streaming_DoneSentinel_EmittedLast)
{
    auto handler = makeHandler();

    ON_CALL(*tokenizer_, encodeChat(_, _, _))
        .WillByDefault(Return(std::vector<int>{1}));
    ON_CALL(*runner_, prefill(_))
        .WillByDefault(Return(true));
    ON_CALL(*tokenizer_, is_stop_token(_))
        .WillByDefault(Return(false));

    EXPECT_CALL(*runner_, decodeStep())
        .WillOnce(Return(makeToken(1, true)));

    ChatCompletionRequest request;
    request.messages = {ChatMessage("user", "test")};
    request.stream = true;

    std::vector<std::string> chunks;
    auto cb = [&](const std::string &line) -> bool
    {
        chunks.push_back(line);
        return true;
    };

    handler->handleStreamingRequest(request, cb);

    ASSERT_GE(chunks.size(), 1u);
    EXPECT_EQ(chunks.back(), "data: [DONE]\n\n");
}

TEST_F(Test__ChatCompletionHandler, Streaming_ConsistentId_AcrossChunks)
{
    auto handler = makeHandler();

    ON_CALL(*tokenizer_, encodeChat(_, _, _))
        .WillByDefault(Return(std::vector<int>{1}));
    ON_CALL(*runner_, prefill(_))
        .WillByDefault(Return(true));
    ON_CALL(*tokenizer_, is_stop_token(_))
        .WillByDefault(Return(false));
    ON_CALL(*tokenizer_, decode_token(_))
        .WillByDefault(Return("x"));

    EXPECT_CALL(*runner_, decodeStep())
        .WillOnce(Return(makeToken(10)))
        .WillOnce(Return(makeToken(0, true)));

    ChatCompletionRequest request;
    request.messages = {ChatMessage("user", "test")};
    request.stream = true;
    request.max_tokens = 10;

    std::vector<std::string> chunks;
    auto cb = [&](const std::string &line) -> bool
    {
        chunks.push_back(line);
        return true;
    };

    handler->handleStreamingRequest(request, cb);

    // All non-[DONE] chunks should have the same id
    std::string first_id;
    for (const auto &chunk : chunks)
    {
        if (chunk == "data: [DONE]\n\n")
            continue;
        auto j = json::parse(chunk.substr(6, chunk.find("\n\n") - 6));
        std::string id = j["id"].get<std::string>();
        if (first_id.empty())
            first_id = id;
        else
            EXPECT_EQ(id, first_id) << "All chunks should have the same id";
    }
    EXPECT_FALSE(first_id.empty());
}

TEST_F(Test__ChatCompletionHandler, Streaming_StopToken_FinishReasonIsStop)
{
    auto handler = makeHandler();

    ON_CALL(*tokenizer_, encodeChat(_, _, _))
        .WillByDefault(Return(std::vector<int>{1}));
    ON_CALL(*runner_, prefill(_))
        .WillByDefault(Return(true));
    ON_CALL(*tokenizer_, decode_token(10))
        .WillByDefault(Return("hi"));
    ON_CALL(*tokenizer_, is_stop_token(10))
        .WillByDefault(Return(false));
    ON_CALL(*tokenizer_, is_stop_token(99))
        .WillByDefault(Return(true));

    EXPECT_CALL(*runner_, decodeStep())
        .WillOnce(Return(makeToken(10)))
        .WillOnce(Return(makeToken(99)));

    ChatCompletionRequest request;
    request.messages = {ChatMessage("user", "test")};
    request.stream = true;
    request.max_tokens = 100;

    std::vector<std::string> chunks;
    auto cb = [&](const std::string &line) -> bool
    {
        chunks.push_back(line);
        return true;
    };

    handler->handleStreamingRequest(request, cb);

    // Find finish chunk (second to last, before [DONE])
    ASSERT_GE(chunks.size(), 3u);
    auto finish_json = json::parse(chunks[chunks.size() - 2].substr(6));
    EXPECT_EQ(finish_json["choices"][0]["finish_reason"], "stop");
}

TEST_F(Test__ChatCompletionHandler, Streaming_MaxTokens_FinishReasonIsLength)
{
    auto handler = makeHandler();

    ON_CALL(*tokenizer_, encodeChat(_, _, _))
        .WillByDefault(Return(std::vector<int>{1}));
    ON_CALL(*runner_, prefill(_))
        .WillByDefault(Return(true));
    ON_CALL(*tokenizer_, is_stop_token(_))
        .WillByDefault(Return(false));
    ON_CALL(*tokenizer_, decode_token(_))
        .WillByDefault(Return("x"));

    EXPECT_CALL(*runner_, decodeStep())
        .Times(3)
        .WillRepeatedly(Return(makeToken(10)));

    ChatCompletionRequest request;
    request.messages = {ChatMessage("user", "test")};
    request.stream = true;
    request.max_tokens = 3;

    std::vector<std::string> chunks;
    auto cb = [&](const std::string &line) -> bool
    {
        chunks.push_back(line);
        return true;
    };

    handler->handleStreamingRequest(request, cb);

    // Finish chunk should say "length"
    auto finish_json = json::parse(chunks[chunks.size() - 2].substr(6));
    EXPECT_EQ(finish_json["choices"][0]["finish_reason"], "length");
}

TEST_F(Test__ChatCompletionHandler, Streaming_PrefillFailure_ReturnsError)
{
    auto handler = makeHandler();

    ON_CALL(*tokenizer_, encodeChat(_, _, _))
        .WillByDefault(Return(std::vector<int>{1}));
    std::string prefill_error = "Out of memory";
    ON_CALL(*runner_, prefill(_))
        .WillByDefault(Return(false));
    ON_CALL(*runner_, lastError())
        .WillByDefault(testing::ReturnRef(prefill_error));

    ChatCompletionRequest request;
    request.messages = {ChatMessage("user", "test")};
    request.stream = true;

    std::vector<std::string> chunks;
    auto cb = [&](const std::string &line) -> bool
    {
        chunks.push_back(line);
        return true;
    };

    auto response = handler->handleStreamingRequest(request, cb);
    EXPECT_FALSE(response.ok);
    EXPECT_EQ(response.http_status, 500);
    // No SSE chunks should be emitted on pre-inference error
    EXPECT_TRUE(chunks.empty());
}

TEST_F(Test__ChatCompletionHandler, Streaming_ClearsCacheBeforeAndAfterRequest)
{
    auto handler = makeHandler();

    ON_CALL(*tokenizer_, encodeChat(_, _, _))
        .WillByDefault(Return(std::vector<int>{1}));
    ON_CALL(*runner_, prefill(_))
        .WillByDefault(Return(true));
    ON_CALL(*tokenizer_, is_stop_token(_))
        .WillByDefault(Return(false));

    EXPECT_CALL(*runner_, clearCache()).Times(2);
    EXPECT_CALL(*runner_, decodeStep())
        .WillOnce(Return(makeToken(1, true)));

    ChatCompletionRequest request;
    request.messages = {ChatMessage("user", "test")};
    request.stream = true;

    std::vector<std::string> chunks;
    auto cb = [&](const std::string &line) -> bool
    {
        chunks.push_back(line);
        return true;
    };

    auto response = handler->handleStreamingRequest(request, cb);

    EXPECT_TRUE(response.ok);
    ASSERT_FALSE(chunks.empty());
    EXPECT_EQ(chunks.back(), "data: [DONE]\n\n");
}

TEST_F(Test__ChatCompletionHandler, Streaming_ChunkObjectType)
{
    auto handler = makeHandler();

    ON_CALL(*tokenizer_, encodeChat(_, _, _))
        .WillByDefault(Return(std::vector<int>{1}));
    ON_CALL(*runner_, prefill(_))
        .WillByDefault(Return(true));
    ON_CALL(*tokenizer_, is_stop_token(_))
        .WillByDefault(Return(false));

    EXPECT_CALL(*runner_, decodeStep())
        .WillOnce(Return(makeToken(1, true)));

    ChatCompletionRequest request;
    request.messages = {ChatMessage("user", "test")};
    request.stream = true;

    std::vector<std::string> chunks;
    auto cb = [&](const std::string &line) -> bool
    {
        chunks.push_back(line);
        return true;
    };

    handler->handleStreamingRequest(request, cb);

    // All non-[DONE] chunks should have object "chat.completion.chunk"
    for (const auto &chunk : chunks)
    {
        if (chunk == "data: [DONE]\n\n")
            continue;
        auto j = json::parse(chunk.substr(6, chunk.find("\n\n") - 6));
        EXPECT_EQ(j["object"], "chat.completion.chunk");
    }
}

TEST_F(Test__ChatCompletionHandler, Streaming_SystemFingerprint_InChunks)
{
    auto handler = makeHandler();

    ON_CALL(*tokenizer_, encodeChat(_, _, _))
        .WillByDefault(Return(std::vector<int>{1}));
    ON_CALL(*runner_, prefill(_))
        .WillByDefault(Return(true));
    ON_CALL(*tokenizer_, is_stop_token(_))
        .WillByDefault(Return(false));

    EXPECT_CALL(*runner_, decodeStep())
        .WillOnce(Return(makeToken(1, true)));

    ChatCompletionRequest request;
    request.messages = {ChatMessage("user", "test")};
    request.stream = true;

    std::vector<std::string> chunks;
    auto cb = [&](const std::string &line) -> bool
    {
        chunks.push_back(line);
        return true;
    };

    handler->handleStreamingRequest(request, cb);

    for (const auto &chunk : chunks)
    {
        if (chunk == "data: [DONE]\n\n")
            continue;
        auto j = json::parse(chunk.substr(6, chunk.find("\n\n") - 6));
        EXPECT_EQ(j["system_fingerprint"], "llaminar-v2");
    }
}

// =============================================================================
// Streaming Think/Content Split tests (Phase 4)
// =============================================================================

TEST_F(Test__ChatCompletionHandler, ThinkSplitter_NoEndTag_AllContent)
{
    StreamingThinkSplitter splitter;
    auto result = splitter.process("hello");
    EXPECT_EQ(result.field, "content");
    EXPECT_EQ(result.text, "hello");
}

TEST_F(Test__ChatCompletionHandler, ThinkSplitter_InThinking_ReasoningContent)
{
    StreamingThinkSplitter splitter("</think>");
    auto result = splitter.process("reasoning");
    EXPECT_EQ(result.field, "reasoning_content");
    EXPECT_EQ(result.text, "reasoning");
    EXPECT_TRUE(splitter.inThinking());
}

TEST_F(Test__ChatCompletionHandler, ThinkSplitter_EndTag_TransitionsToContent)
{
    StreamingThinkSplitter splitter("</think>");

    // Process some reasoning
    auto r1 = splitter.process("thinking ");
    EXPECT_EQ(r1.field, "reasoning_content");
    EXPECT_TRUE(splitter.inThinking());

    // Process text with end tag
    auto r2 = splitter.process("done</think>\n\nAnswer here");
    // r2 should be the reasoning part (before </think>)
    EXPECT_EQ(r2.field, "reasoning_content");
    EXPECT_FALSE(splitter.inThinking());

    // Flush should give the content part
    auto flushed = splitter.flush();
    EXPECT_EQ(flushed.field, "content");
    EXPECT_EQ(flushed.text, "Answer here");
}

TEST_F(Test__ChatCompletionHandler, ThinkSplitter_AfterTransition_ContentField)
{
    StreamingThinkSplitter splitter("</think>");

    // End thinking immediately
    splitter.process("</think>\n\n");
    EXPECT_FALSE(splitter.inThinking());

    // Subsequent text should be content
    auto r = splitter.process("answer");
    EXPECT_EQ(r.field, "content");
    EXPECT_EQ(r.text, "answer");
}

TEST_F(Test__ChatCompletionHandler, ThinkSplitter_DuplicateEndTagStopsContent)
{
    StreamingThinkSplitter splitter("</think>");

    // The first marker closes reasoning; later markers are malformed answer
    // text and should stop the stream before the marker is emitted.
    auto r1 = splitter.process("reasoning</think>\n\nAnswer ");
    EXPECT_EQ(r1.field, "reasoning_content");
    EXPECT_FALSE(splitter.inThinking());

    auto first_content = splitter.flush();
    EXPECT_EQ(first_content.field, "content");
    EXPECT_EQ(first_content.text, "Answer ");
    EXPECT_FALSE(first_content.stop_generation);

    auto r2 = splitter.process("</th");
    EXPECT_EQ(r2.field, "content");
    EXPECT_TRUE(r2.text.empty());
    EXPECT_FALSE(r2.stop_generation);

    auto r3 = splitter.process("ink>\n\nIgnored");
    EXPECT_EQ(r3.field, "content");
    EXPECT_TRUE(r3.text.empty());
    EXPECT_TRUE(r3.stop_generation);
}

TEST_F(Test__ChatCompletionHandler, ThinkSplitter_EndTagOnly_EmptyReasoning)
{
    StreamingThinkSplitter splitter("</think>");

    auto r = splitter.process("</think>\n\nHello");
    // The reasoning before </think> is empty
    // After transition, the flush should give "Hello"
    EXPECT_FALSE(splitter.inThinking());

    auto flushed = splitter.flush();
    if (r.text.empty())
    {
        // Content was buffered
        EXPECT_EQ(flushed.text, "Hello");
        EXPECT_EQ(flushed.field, "content");
    }
}

TEST_F(Test__ChatCompletionHandler, ThinkSplitter_FlushEmpty_NoCrash)
{
    StreamingThinkSplitter splitter("</think>");
    auto r = splitter.flush();
    EXPECT_TRUE(r.text.empty());
}

TEST_F(Test__ChatCompletionHandler, ThinkSplitter_PartialEndTag_Buffered)
{
    StreamingThinkSplitter splitter("</think>");

    // Send partial end tag across tokens
    auto r1 = splitter.process("reasoning</th");
    // The "reasoning" part should be emitted, "</th" buffered
    EXPECT_EQ(r1.field, "reasoning_content");
    EXPECT_TRUE(splitter.inThinking());
    // Either "reasoning" is emitted or everything is buffered for safety

    // Complete the tag
    auto r2 = splitter.process("ink>\n\nAnswer");
    // Should transition to content
    EXPECT_FALSE(splitter.inThinking());

    auto flushed = splitter.flush();
    if (!flushed.text.empty())
    {
        EXPECT_EQ(flushed.field, "content");
    }
}

// =============================================================================
// handleRawRequest routing: stream=true routes to streaming
// =============================================================================

TEST_F(Test__ChatCompletionHandler, HandleRawRequest_StreamTrue_WithCallback_UsesStreaming)
{
    auto handler = makeHandler();

    ON_CALL(*tokenizer_, encodeChat(_, _, _))
        .WillByDefault(Return(std::vector<int>{1}));
    ON_CALL(*runner_, prefill(_))
        .WillByDefault(Return(true));
    ON_CALL(*tokenizer_, is_stop_token(_))
        .WillByDefault(Return(false));

    EXPECT_CALL(*runner_, decodeStep())
        .WillOnce(Return(makeToken(1, true)));

    std::vector<std::string> chunks;
    auto stream_cb = [&](const std::string &line) -> bool
    {
        chunks.push_back(line);
        return true;
    };

    json body = {
        {"messages", json::array({json{{"role", "user"}, {"content", "test"}}})},
        {"stream", true}};

    auto response = handler->handleRawRequest(body.dump(), stream_cb);
    EXPECT_TRUE(response.ok);
    EXPECT_FALSE(chunks.empty()) << "Streaming callback should have been invoked";
    EXPECT_EQ(chunks.back(), "data: [DONE]\n\n");
}

TEST_F(Test__ChatCompletionHandler, HandleRawRequest_StreamTrue_NoCallback_FallsBackToNonStreaming)
{
    auto handler = makeHandler();

    ON_CALL(*tokenizer_, encodeChat(_, _, _))
        .WillByDefault(Return(std::vector<int>{1}));
    ON_CALL(*runner_, prefill(_))
        .WillByDefault(Return(true));
    ON_CALL(*tokenizer_, is_stop_token(_))
        .WillByDefault(Return(false));

    EXPECT_CALL(*runner_, decodeStep())
        .WillOnce(Return(makeToken(1, true)));

    json body = {
        {"messages", json::array({json{{"role", "user"}, {"content", "test"}}})},
        {"stream", true}};

    // No stream callback provided — falls back to non-streaming
    auto response = handler->handleRawRequest(body.dump());
    EXPECT_TRUE(response.ok);
    // Should have a valid JSON body (non-streaming response)
    auto resp_body = json::parse(response.json_body);
    EXPECT_EQ(resp_body["object"], "chat.completion");
}

// =============================================================================
// enable_thinking=false disables reasoning extraction
// =============================================================================

TEST_F(Test__ChatCompletionHandler, HandleRequest_EnableThinkingFalse_NoReasoningSplit)
{
    auto handler = makeHandler();

    // Create a mock chat template that reports thinking support
    auto mock_template = ChatTemplate::create(ChatTemplateType::CHATML);

    ON_CALL(*tokenizer_, encodeChat(_, _, _))
        .WillByDefault(Return(std::vector<int>{1}));
    ON_CALL(*runner_, prefill(_))
        .WillByDefault(Return(true));
    ON_CALL(*tokenizer_, is_stop_token(_))
        .WillByDefault(Return(false));
    ON_CALL(*tokenizer_, decode_token(10))
        .WillByDefault(Return("thinking here</think>\n\nAnswer"));
    ON_CALL(*tokenizer_, hasChatTemplate())
        .WillByDefault(Return(true));
    ON_CALL(*tokenizer_, getChatTemplate())
        .WillByDefault(testing::ReturnRef(*mock_template));

    EXPECT_CALL(*runner_, decodeStep())
        .WillOnce(Return(makeToken(10)))
        .WillOnce(Return(makeToken(0, true)));

    ChatCompletionRequest request;
    request.messages = {ChatMessage("user", "test")};
    request.enable_thinking = false; // Disable reasoning extraction

    auto response = handler->handleRequest(request);
    EXPECT_TRUE(response.ok);

    auto body = json::parse(response.json_body);
    // With enable_thinking=false, the entire output should be in content
    // No reasoning_content field should be present
    EXPECT_FALSE(body["choices"][0]["message"].contains("reasoning_content"))
        << "With enable_thinking=false, reasoning_content should not be extracted";
}

// =============================================================================
// Thinking Budget tests
// =============================================================================

TEST_F(Test__ChatCompletionHandler, HandleRequest_ThinkingBudget_InjectsStopSequence)
{
    auto handler = makeHandler();

    ON_CALL(*tokenizer_, encodeChat(_, _, _))
        .WillByDefault(Return(std::vector<int>{1, 2, 3}));
    ON_CALL(*tokenizer_, is_stop_token(_))
        .WillByDefault(Return(false));
    ON_CALL(*tokenizer_, hasChatTemplate())
        .WillByDefault(Return(false));

    // Stop-thinking prompt tokenizes to [90, 91, 92]
    ON_CALL(*runner_, getStopThinkingPrompt())
        .WillByDefault(Return("stop thinking now"));
    ON_CALL(*tokenizer_, encode("stop thinking now", _, _))
        .WillByDefault(Return(std::vector<int>{90, 91, 92}));
    EXPECT_CALL(*tokenizer_, encode("stop thinking now", false, false))
        .Times(1);

    // Decode 2 thinking tokens, then budget exhaustion schedules the stop
    // sequence for subsequent forced positions. The forced-token calls are the
    // important regression guard: text injection without runner-state commits
    // leaves later decode on the wrong KV/GDN state.
    ON_CALL(*tokenizer_, decode_token(10))
        .WillByDefault(Return("thinking1"));
    ON_CALL(*tokenizer_, decode_token(11))
        .WillByDefault(Return("thinking2"));
    ON_CALL(*tokenizer_, decode_token(90))
        .WillByDefault(Return("stop"));
    ON_CALL(*tokenizer_, decode_token(91))
        .WillByDefault(Return(" thinking"));
    ON_CALL(*tokenizer_, decode_token(92))
        .WillByDefault(Return(" now"));

    ON_CALL(*runner_, prefill(_))
        .WillByDefault(Return(true));

    EXPECT_CALL(*runner_, decodeStep())
        .WillOnce(Return(makeToken(10)))       // Thinking token 1
        .WillOnce(Return(makeToken(11)))       // Thinking token 2, then budget exhausted
        .WillOnce(Return(makeToken(0, true))); // Normal completion
    EXPECT_CALL(*runner_, forceDecodeToken(90))
        .WillOnce(Return(makeToken(90)));
    EXPECT_CALL(*runner_, forceDecodeToken(91))
        .WillOnce(Return(makeToken(91)));
    EXPECT_CALL(*runner_, forceDecodeToken(92))
        .WillOnce(Return(makeToken(92)));

    ChatCompletionRequest request;
    request.messages = {ChatMessage("user", "think about this")};
    request.max_tokens = 20;
    request.enable_thinking = true;
    request.thinking_budget_tokens = 2; // Exhaust after 2 thinking tokens

    auto response = handler->handleRequest(request);
    EXPECT_TRUE(response.ok);

    auto body = json::parse(response.json_body);
    auto content = body["choices"][0]["message"]["content"].get<std::string>();

    // Generated text should contain the injected stop tokens
    EXPECT_NE(content.find("stop"), std::string::npos)
        << "Injected stop-thinking tokens should appear in output";
    EXPECT_NE(content.find(" thinking"), std::string::npos);
    EXPECT_NE(content.find(" now"), std::string::npos);
}

TEST_F(Test__ChatCompletionHandler, HandleRequest_ThinkingBudget_StopsAtDuplicateEndTag)
{
    auto handler = makeHandler();
    auto tmpl = makeThinkingTemplate();
    ASSERT_TRUE(tmpl->isThinkingModel());

    ON_CALL(*tokenizer_, hasChatTemplate())
        .WillByDefault(Return(true));
    ON_CALL(*tokenizer_, getChatTemplate())
        .WillByDefault(::testing::ReturnRef(*tmpl));
    ON_CALL(*tokenizer_, encodeChat(_, _, _))
        .WillByDefault(Return(std::vector<int>{1, 2, 3}));
    ON_CALL(*tokenizer_, is_stop_token(_))
        .WillByDefault(Return(false));
    ON_CALL(*runner_, prefill(_))
        .WillByDefault(Return(true));

    ON_CALL(*runner_, getStopThinkingPrompt())
        .WillByDefault(Return("Considering the limited time.\n</think>\n\n"));
    ON_CALL(*tokenizer_, encode("Considering the limited time.\n</think>\n\n", _, _))
        .WillByDefault(Return(std::vector<int>{90}));
    EXPECT_CALL(*tokenizer_, encode("Considering the limited time.\n</think>\n\n", false, false))
        .Times(1);

    ON_CALL(*tokenizer_, decode_token(10))
        .WillByDefault(Return("reasoning"));
    ON_CALL(*tokenizer_, decode_token(90))
        .WillByDefault(Return("Considering the limited time.\n</think>\n\n"));
    ON_CALL(*tokenizer_, decode_token(11))
        .WillByDefault(Return("13"));
    ON_CALL(*tokenizer_, decode_token(12))
        .WillByDefault(Return("\n</think>\n\n"));
    ON_CALL(*tokenizer_, decode_token(13))
        .WillByDefault(Return("13"));

    EXPECT_CALL(*runner_, decodeStep())
        .WillOnce(Return(makeToken(10))) // Exhausts the thinking budget.
        .WillOnce(Return(makeToken(11))) // First answer token after the forced close.
        .WillOnce(Return(makeToken(12))); // Duplicate close tag stops generation.
    EXPECT_CALL(*runner_, forceDecodeToken(90))
        .WillOnce(Return(makeToken(90)));

    ChatCompletionRequest request;
    request.messages = {ChatMessage("user", "think briefly")};
    request.max_tokens = 20;
    request.enable_thinking = true;
    request.thinking_budget_tokens = 1;

    auto response = handler->handleRequest(request);
    ASSERT_TRUE(response.ok);

    auto body = json::parse(response.json_body);
    auto message = body["choices"][0]["message"];
    const auto content = message["content"].get<std::string>();

    EXPECT_EQ(content, "13\n");
    EXPECT_EQ(content.find("</think>"), std::string::npos)
        << "Duplicate thinking end tags must not leak into answer content";
    ASSERT_TRUE(message.contains("reasoning_content"));
    EXPECT_NE(message["reasoning_content"].get<std::string>().find("reasoning"),
              std::string::npos);
}

TEST_F(Test__ChatCompletionHandler, HandleRequest_ThinkingBudget_DisabledByDefault)
{
    auto handler = makeHandler();

    ON_CALL(*tokenizer_, encodeChat(_, _, _))
        .WillByDefault(Return(std::vector<int>{1, 2, 3}));
    ON_CALL(*tokenizer_, is_stop_token(_))
        .WillByDefault(Return(false));
    ON_CALL(*tokenizer_, hasChatTemplate())
        .WillByDefault(Return(false));
    ON_CALL(*tokenizer_, decode_token(_))
        .WillByDefault(Return("tok"));

    ON_CALL(*runner_, prefill(_))
        .WillByDefault(Return(true));

    // No thinking budget set — should decode normally without injection
    EXPECT_CALL(*runner_, decodeStep())
        .WillOnce(Return(makeToken(10)))
        .WillOnce(Return(makeToken(11)))
        .WillOnce(Return(makeToken(0, true)));

    ChatCompletionRequest request;
    request.messages = {ChatMessage("user", "test")};
    request.max_tokens = 10;
    // thinking_budget_tokens is -1 by default (disabled)

    auto response = handler->handleRequest(request);
    EXPECT_TRUE(response.ok);

    auto body = json::parse(response.json_body);
    EXPECT_EQ(body["usage"]["completion_tokens"], 3); // 10, 11, 0(stop)
}

TEST_F(Test__ChatCompletionHandler, HandleRequest_ThinkingBudget_NotActiveWhenThinkingDisabled)
{
    auto handler = makeHandler();

    ON_CALL(*tokenizer_, encodeChat(_, _, _))
        .WillByDefault(Return(std::vector<int>{1, 2, 3}));
    ON_CALL(*tokenizer_, is_stop_token(_))
        .WillByDefault(Return(false));
    ON_CALL(*tokenizer_, hasChatTemplate())
        .WillByDefault(Return(false));
    ON_CALL(*tokenizer_, decode_token(_))
        .WillByDefault(Return("tok"));

    ON_CALL(*runner_, prefill(_))
        .WillByDefault(Return(true));

    // Even with budget set, if enable_thinking=false, budget is inactive
    EXPECT_CALL(*runner_, decodeStep())
        .WillOnce(Return(makeToken(10)))
        .WillOnce(Return(makeToken(11)))
        .WillOnce(Return(makeToken(12)))
        .WillOnce(Return(makeToken(0, true)));

    ChatCompletionRequest request;
    request.messages = {ChatMessage("user", "test")};
    request.max_tokens = 10;
    request.enable_thinking = false;
    request.thinking_budget_tokens = 1; // Would trigger after 1 token if active

    auto response = handler->handleRequest(request);
    EXPECT_TRUE(response.ok);

    auto body = json::parse(response.json_body);
    // All 4 tokens decoded normally (no injection)
    EXPECT_EQ(body["usage"]["completion_tokens"], 4);
}

TEST_F(Test__ChatCompletionHandler, ParseRequest_ThinkingBudgetTokens_Parsed)
{
    ChatCompletionResponse error;
    auto result = ChatCompletionHandler::parseRequest(
        minimalRequest({{"thinking_budget_tokens", 50}}), error);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->thinking_budget_tokens, 50);
}

TEST_F(Test__ChatCompletionHandler, ParseRequest_DRYParams_Parsed)
{
    ChatCompletionResponse error;
    auto result = ChatCompletionHandler::parseRequest(
        minimalRequest({{"dry_multiplier", 0.8},
                        {"dry_base", 2.0},
                        {"dry_allowed_length", 3},
                        {"dry_penalty_last_n", 256},
                        {"dry_sequence_breakers", json::array({"\\n", ":"})}}),
        error);

    ASSERT_TRUE(result.has_value());
    EXPECT_FLOAT_EQ(result->sampling.dry_multiplier, 0.8f);
    EXPECT_FLOAT_EQ(result->sampling.dry_base, 2.0f);
    EXPECT_EQ(result->sampling.dry_allowed_length, 3);
    EXPECT_EQ(result->sampling.dry_penalty_last_n, 256);
    EXPECT_EQ(result->sampling.dry_sequence_breakers.size(), 2u);
    EXPECT_EQ(result->sampling.dry_sequence_breakers[0], "\\n");
    EXPECT_EQ(result->sampling.dry_sequence_breakers[1], ":");
}

// =============================================================================
// Tool calling tests
// =============================================================================

// --- Request parsing ---

TEST_F(Test__ChatCompletionHandler, ParseRequest_ToolDefinitions_Parsed)
{
    auto body = json::parse(R"({
        "messages": [{"role": "user", "content": "What's the weather?"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a location",
                "parameters": {"type": "object", "properties": {"location": {"type": "string"}}}
            }
        }],
        "tool_choice": "auto",
        "parallel_tool_calls": true
    })");

    ChatCompletionResponse error;
    auto result = ChatCompletionHandler::parseRequest(body.dump(), error);

    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result->tools.is_array());
    EXPECT_EQ(result->tools.size(), 1u);
    EXPECT_EQ(result->tools[0]["function"]["name"], "get_weather");
    EXPECT_EQ(result->tool_choice, "auto");
    EXPECT_TRUE(result->parallel_tool_calls);
}

TEST_F(Test__ChatCompletionHandler, ParseRequest_ToolMessage_Parsed)
{
    auto body = json::parse(R"({
        "messages": [
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_abc",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{\"location\":\"Paris\"}"}
                }]
            },
            {"role": "tool", "content": "{\"temp\": 22}", "tool_call_id": "call_abc", "name": "get_weather"}
        ]
    })");

    ChatCompletionResponse error;
    auto result = ChatCompletionHandler::parseRequest(body.dump(), error);

    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result->messages.size(), 3u);

    // Assistant message with tool_calls
    EXPECT_EQ(result->messages[1].role, "assistant");
    EXPECT_TRUE(result->messages[1].content.empty());
    EXPECT_TRUE(result->messages[1].hasToolCalls());
    EXPECT_EQ(result->messages[1].tool_calls.size(), 1u);

    // Tool result message
    EXPECT_EQ(result->messages[2].role, "tool");
    EXPECT_EQ(result->messages[2].tool_call_id, "call_abc");
    EXPECT_EQ(result->messages[2].name, "get_weather");
    EXPECT_EQ(result->messages[2].isToolResult(), true);
}

TEST_F(Test__ChatCompletionHandler, ParseRequest_ToolMessage_MissingToolCallId_Returns400)
{
    auto body = json::parse(R"({
        "messages": [{"role": "tool", "content": "result data"}]
    })");

    ChatCompletionResponse error;
    auto result = ChatCompletionHandler::parseRequest(body.dump(), error);

    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(error.http_status, 400);
}

// --- Non-streaming response: tool calls detected ---

TEST_F(Test__ChatCompletionHandler, HandleRequest_ToolCallsDetected_InResponse)
{
    auto handler = makeHandler();

    // Model output contains Hermes 2 Pro tool call tags
    std::string model_output = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Paris\"}}\n</tool_call>";

    ON_CALL(*tokenizer_, encodeChat(_, _, _))
        .WillByDefault(Return(std::vector<int>{1, 2, 3}));
    ON_CALL(*runner_, prefill(_))
        .WillByDefault(Return(true));
    ON_CALL(*tokenizer_, is_stop_token(_))
        .WillByDefault(Return(false));
    ON_CALL(*runner_, getToolCallFormat())
        .WillByDefault(Return(ToolCallFormat::HERMES_2_PRO));

    // Emit one token per character then stop
    int call_count = 0;
    EXPECT_CALL(*runner_, decodeStep())
        .WillRepeatedly(Invoke([&]() -> GenerationResult
                               {
            if (call_count < static_cast<int>(model_output.size()))
            {
                return makeToken(100 + call_count++);
            }
            return makeToken(0, true); }));

    // Each token decodes to one character of the model output
    ON_CALL(*tokenizer_, decode_token(_))
        .WillByDefault(Invoke([&](int token_id) -> std::string
                              {
            int idx = token_id - 100;
            if (idx >= 0 && idx < static_cast<int>(model_output.size()))
                return std::string(1, model_output[idx]);
            return ""; }));

    ChatCompletionRequest request;
    request.messages = {ChatMessage("user", "What's the weather?")};
    request.max_tokens = 200;
    request.tools = json::array({json{{"type", "function"}, {"function", {{"name", "get_weather"}}}}});

    auto response = handler->handleRequest(request);

    EXPECT_TRUE(response.ok);
    EXPECT_EQ(response.http_status, 200);

    auto body = json::parse(response.json_body);
    EXPECT_EQ(body["choices"][0]["finish_reason"], "tool_calls");

    auto &message = body["choices"][0]["message"];
    EXPECT_EQ(message["role"], "assistant");
    EXPECT_TRUE(message.contains("tool_calls"));
    EXPECT_TRUE(message["tool_calls"].is_array());
    EXPECT_EQ(message["tool_calls"].size(), 1u);
    EXPECT_EQ(message["tool_calls"][0]["type"], "function");
    EXPECT_EQ(message["tool_calls"][0]["function"]["name"], "get_weather");
    EXPECT_EQ(message["tool_calls"][0]["function"]["arguments"], R"({"location":"Paris"})");
    EXPECT_FALSE(message["tool_calls"][0]["id"].get<std::string>().empty());
}

// --- Non-streaming: no tool calls in normal output ---

TEST_F(Test__ChatCompletionHandler, HandleRequest_NoToolCalls_NormalResponse)
{
    auto handler = makeHandler();

    ON_CALL(*tokenizer_, encodeChat(_, _, _))
        .WillByDefault(Return(std::vector<int>{1, 2}));
    ON_CALL(*runner_, prefill(_))
        .WillByDefault(Return(true));
    ON_CALL(*tokenizer_, is_stop_token(_))
        .WillByDefault(Return(false));
    ON_CALL(*runner_, getToolCallFormat())
        .WillByDefault(Return(ToolCallFormat::HERMES_2_PRO));
    ON_CALL(*tokenizer_, decode_token(100))
        .WillByDefault(Return("The weather is sunny."));

    EXPECT_CALL(*runner_, decodeStep())
        .WillOnce(Return(makeToken(100)))
        .WillOnce(Return(makeToken(0, true)));

    ChatCompletionRequest request;
    request.messages = {ChatMessage("user", "What's the weather?")};
    request.max_tokens = 10;
    request.tools = json::array({json{{"type", "function"}, {"function", {{"name", "get_weather"}}}}});

    auto response = handler->handleRequest(request);

    EXPECT_TRUE(response.ok);
    auto body = json::parse(response.json_body);
    EXPECT_EQ(body["choices"][0]["finish_reason"], "stop");
    EXPECT_EQ(body["choices"][0]["message"]["content"], "The weather is sunny.");
    EXPECT_FALSE(body["choices"][0]["message"].contains("tool_calls"));
}

// --- Streaming: tool calls detected ---

TEST_F(Test__ChatCompletionHandler, HandleStreamingRequest_ToolCallsDetected)
{
    auto handler = makeHandler();

    std::string model_output = "<tool_call>\n{\"name\": \"search\", \"arguments\": {\"q\": \"test\"}}\n</tool_call>";

    ON_CALL(*tokenizer_, encodeChat(_, _, _))
        .WillByDefault(Return(std::vector<int>{1, 2, 3}));
    ON_CALL(*runner_, prefill(_))
        .WillByDefault(Return(true));
    ON_CALL(*tokenizer_, is_stop_token(_))
        .WillByDefault(Return(false));
    ON_CALL(*runner_, getToolCallFormat())
        .WillByDefault(Return(ToolCallFormat::HERMES_2_PRO));
    ON_CALL(*tokenizer_, hasChatTemplate())
        .WillByDefault(Return(false));

    int call_count = 0;
    EXPECT_CALL(*runner_, decodeStep())
        .WillRepeatedly(Invoke([&]() -> GenerationResult
                               {
            if (call_count < static_cast<int>(model_output.size()))
                return makeToken(100 + call_count++);
            return makeToken(0, true); }));

    ON_CALL(*tokenizer_, decode_token(_))
        .WillByDefault(Invoke([&](int token_id) -> std::string
                              {
            int idx = token_id - 100;
            if (idx >= 0 && idx < static_cast<int>(model_output.size()))
                return std::string(1, model_output[idx]);
            return ""; }));

    ChatCompletionRequest request;
    request.messages = {ChatMessage("user", "search for test")};
    request.max_tokens = 200;
    request.stream = true;
    request.tools = json::array({json{{"type", "function"}, {"function", {{"name", "search"}}}}});

    // Collect SSE chunks
    std::vector<std::string> chunks;
    auto cb = [&](const std::string &chunk) -> bool
    {
        chunks.push_back(chunk);
        return true;
    };

    auto response = handler->handleStreamingRequest(request, cb);
    EXPECT_TRUE(response.ok);

    // Find the chunk containing tool_calls delta
    bool found_tool_calls = false;
    bool found_tool_calls_finish = false;
    for (const auto &chunk : chunks)
    {
        if (chunk.find("[DONE]") != std::string::npos)
            continue;
        if (chunk.substr(0, 6) != "data: ")
            continue;

        auto data = json::parse(chunk.substr(6));
        auto &delta = data["choices"][0]["delta"];

        if (delta.contains("tool_calls"))
        {
            found_tool_calls = true;
            EXPECT_EQ(delta["tool_calls"][0]["function"]["name"], "search");
            EXPECT_EQ(delta["tool_calls"][0]["function"]["arguments"], R"({"q":"test"})");
        }

        if (data["choices"][0].contains("finish_reason") &&
            data["choices"][0]["finish_reason"] == "tool_calls")
        {
            found_tool_calls_finish = true;
        }
    }

    EXPECT_TRUE(found_tool_calls) << "Expected tool_calls delta in SSE stream";
    EXPECT_TRUE(found_tool_calls_finish) << "Expected finish_reason=tool_calls in SSE stream";
}

// --- Without tools in request, tool-like output is passed through as content ---

TEST_F(Test__ChatCompletionHandler, HandleRequest_NoToolsRequested_ToolLikeOutputIsContent)
{
    auto handler = makeHandler();

    ON_CALL(*tokenizer_, encodeChat(_, _, _))
        .WillByDefault(Return(std::vector<int>{1}));
    ON_CALL(*runner_, prefill(_))
        .WillByDefault(Return(true));
    ON_CALL(*tokenizer_, is_stop_token(_))
        .WillByDefault(Return(false));
    ON_CALL(*tokenizer_, decode_token(100))
        .WillByDefault(Return("<tool_call>{\"name\":\"fn\"}</tool_call>"));

    EXPECT_CALL(*runner_, decodeStep())
        .WillOnce(Return(makeToken(100)))
        .WillOnce(Return(makeToken(0, true)));

    ChatCompletionRequest request;
    request.messages = {ChatMessage("user", "test")};
    request.max_tokens = 10;
    // No tools set — tool detection should NOT activate

    auto response = handler->handleRequest(request);
    EXPECT_TRUE(response.ok);

    auto body = json::parse(response.json_body);
    // Should be normal content, not parsed as tool_calls
    EXPECT_EQ(body["choices"][0]["finish_reason"], "stop");
    EXPECT_FALSE(body["choices"][0]["message"].contains("tool_calls"));
    EXPECT_TRUE(body["choices"][0]["message"]["content"].get<std::string>().find("<tool_call>") != std::string::npos);
}
