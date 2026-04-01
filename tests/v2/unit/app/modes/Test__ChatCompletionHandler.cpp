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
using ::testing::Return;
using ::testing::Invoke;

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

TEST_F(Test__ChatCompletionHandler, ParseRequest_DefaultTemperature_IsGreedy)
{
    ChatCompletionResponse error;
    auto result = ChatCompletionHandler::parseRequest(minimalRequest(), error);

    ASSERT_TRUE(result.has_value());
    EXPECT_FLOAT_EQ(result->sampling.temperature, 0.0f);
    EXPECT_TRUE(result->sampling.is_greedy());
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

TEST_F(Test__ChatCompletionHandler, HandleRawRequest_GreedyByDefault)
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
    EXPECT_TRUE(captured.is_greedy()) << "Default server request must use greedy sampling";
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
