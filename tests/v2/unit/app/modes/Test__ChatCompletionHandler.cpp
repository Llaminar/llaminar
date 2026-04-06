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

using namespace llaminar2;
using namespace llaminar2::test;
using json = nlohmann::json;
using ::testing::_;
using ::testing::Invoke;
using ::testing::Return;

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

TEST_F(Test__ChatCompletionHandler, ParseRequest_DefaultMaxTokens_Is128)
{
    ChatCompletionResponse error;
    auto result = ChatCompletionHandler::parseRequest(minimalRequest(), error);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->max_tokens, 128);
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

    ON_CALL(*tokenizer_, encodeChat(_, _))
        .WillByDefault(Return(std::vector<int>{1, 2, 3}));

    EXPECT_CALL(*runner_, setSamplingParams(_))
        .WillOnce(Invoke([&](const SamplingParams &)
                         { call_order.push_back("setSamplingParams"); }));

    EXPECT_CALL(*runner_, clearCache())
        .WillOnce(Invoke([&]()
                         { call_order.push_back("clearCache"); }));

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

    ON_CALL(*tokenizer_, encodeChat(_, _))
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

    handler->handleRequest(request);

    EXPECT_FLOAT_EQ(captured.temperature, 0.8f);
    EXPECT_FLOAT_EQ(captured.top_p, 0.95f);
    EXPECT_EQ(captured.top_k, 40);
    EXPECT_EQ(captured.seed, 999u);
}

TEST_F(Test__ChatCompletionHandler, HandleRequest_GreedySampling_WhenTemp0)
{
    auto handler = makeHandler();

    ON_CALL(*tokenizer_, encodeChat(_, _))
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

    handler->handleRequest(request);

    EXPECT_TRUE(captured.is_greedy()) << "temperature=0 must result in greedy sampling";
}

// =============================================================================
// Full end-to-end: handleRawRequest
// =============================================================================

TEST_F(Test__ChatCompletionHandler, HandleRawRequest_UsesModelDefaultsWhenNoUserSampling)
{
    auto handler = makeHandler();

    ON_CALL(*tokenizer_, encodeChat(_, _))
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

    ON_CALL(*tokenizer_, encodeChat(_, _))
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

    ON_CALL(*tokenizer_, encodeChat(_, _))
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

// =============================================================================
// Error handling tests
// =============================================================================

TEST_F(Test__ChatCompletionHandler, HandleRequest_EncodeEmpty_Returns500)
{
    auto handler = makeHandler();

    ON_CALL(*tokenizer_, encodeChat(_, _))
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

    ON_CALL(*tokenizer_, encodeChat(_, _))
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

    ON_CALL(*tokenizer_, encodeChat(_, _))
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

    ON_CALL(*tokenizer_, encodeChat(_, _))
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

    ON_CALL(*tokenizer_, encodeChat(_, _))
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

TEST_F(Test__ChatCompletionHandler, HandleRequest_EmptyDecode_StopsGracefully)
{
    auto handler = makeHandler();

    ON_CALL(*tokenizer_, encodeChat(_, _))
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

TEST_F(Test__ChatCompletionHandler, HandleRequest_ClearsCacheBeforeEachRequest)
{
    auto handler = makeHandler();

    ON_CALL(*tokenizer_, encodeChat(_, _))
        .WillByDefault(Return(std::vector<int>{1}));
    ON_CALL(*runner_, prefill(_))
        .WillByDefault(Return(true));

    EXPECT_CALL(*runner_, clearCache()).Times(1);
    EXPECT_CALL(*runner_, decodeStep())
        .WillOnce(Return(makeToken(0, true)));
    ON_CALL(*tokenizer_, is_stop_token(_))
        .WillByDefault(Return(false));

    ChatCompletionRequest request;
    request.messages = {ChatMessage("user", "test")};

    handler->handleRequest(request);
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

    ON_CALL(*tokenizer_, encodeChat(_, _))
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

TEST_F(Test__ChatCompletionHandler, HandleRawRequest_ModelDefaultsNotUsedWhenUserSpecifiesTemp)
{
    auto handler = makeHandler();

    ON_CALL(*tokenizer_, encodeChat(_, _))
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

    // User explicitly sets temperature — model defaults should NOT be applied
    json body = {
        {"messages", json::array({json{{"role", "user"}, {"content", "test"}}})},
        {"temperature", 0.3}};

    auto response = handler->handleRawRequest(body.dump());
    EXPECT_TRUE(response.ok);
    EXPECT_FLOAT_EQ(captured.temperature, 0.3f)
        << "Should use user-specified temperature, not model default";
    EXPECT_FLOAT_EQ(captured.presence_penalty, 0.0f)
        << "Model defaults should not be merged when user specified any param";
}

TEST_F(Test__ChatCompletionHandler, HandleRawRequest_ModelDefaultsNotUsedWhenUserSpecifiesPenalty)
{
    auto handler = makeHandler();

    ON_CALL(*tokenizer_, encodeChat(_, _))
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

    // User explicitly sets presence_penalty — model defaults should NOT be applied
    json body = {
        {"messages", json::array({json{{"role", "user"}, {"content", "test"}}})},
        {"presence_penalty", 0.5}};

    auto response = handler->handleRawRequest(body.dump());
    EXPECT_TRUE(response.ok);
    EXPECT_FLOAT_EQ(captured.presence_penalty, 0.5f)
        << "Should use user-specified penalty";
    EXPECT_FLOAT_EQ(captured.temperature, 1.0f)
        << "Non-specified params should stay at API defaults, not model defaults";
}

TEST_F(Test__ChatCompletionHandler, HandleRawRequest_PenaltiesPassedToRunner)
{
    auto handler = makeHandler();

    ON_CALL(*tokenizer_, encodeChat(_, _))
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
    ON_CALL(*tokenizer_, encodeChat(_, _))
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
    ON_CALL(*tokenizer_, encodeChat(_, _))
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
    ON_CALL(*tokenizer_, encodeChat(_, _))
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

    ON_CALL(*tokenizer_, encodeChat(_, _))
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

    ON_CALL(*tokenizer_, encodeChat(_, _))
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

    ON_CALL(*tokenizer_, encodeChat(_, _))
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
    ON_CALL(*tokenizer_, encodeChat(_, _))
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
    ON_CALL(*tokenizer_, encodeChat(_, _))
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

    ON_CALL(*tokenizer_, encodeChat(_, _))
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
    ON_CALL(*tokenizer_, encodeChat(_, _))
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
    ON_CALL(*tokenizer_, encodeChat(_, _))
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

    ON_CALL(*tokenizer_, encodeChat(_, _))
        .WillByDefault(Return(std::vector<int>{1, 2, 3, 4, 5}));

    auto response = handler->handleRawRequest(minimalRequest());

    EXPECT_FALSE(response.ok);
    EXPECT_EQ(response.http_status, 400);
    auto body = json::parse(response.json_body);
    EXPECT_EQ(body["error"]["type"], "invalid_request_error");
}
