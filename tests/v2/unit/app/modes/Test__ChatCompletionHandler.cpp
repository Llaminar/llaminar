/**
 * @file Test__ChatCompletionHandler.cpp
 * @brief Unit tests for ChatCompletionHandler
 *
 * Tests request parsing, sampling parameter wiring, inference flow,
 * error handling, and response formatting — all via mock interfaces.
 * Completion summaries must consume existing terminal observations, never
 * invoke an intrusive live-state probe on either HTTP response path.
 * Streaming token accounting authenticates terminal counts, framing, and
 * disconnect/error boundaries used by real coding clients for compaction.
 * Opt-in token traces retain terminal IDs and their stopping authority without
 * exposing those control tokens to the client or probing mutable runner state.
 * Native tool-string framing also survives adjacent literal parameter tags and
 * byte-fragmented Unicode on both HTTP response paths.
 */

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "app/modes/ChatCompletionHandler.h"
#include "app/modes/MoEMovementLedgerJson.h"
#include "execution/moe/MoERoutedExpertPlacementPlan.h"
#include "execution/mtp/MTPRequestSamplingPolicy.h"
#include "mocks/MockOrchestrationRunner.h"
#include "mocks/MockTokenizer.h"
#include "utils/Logger.h"
#include "utils/DebugEnv.h"
#include "nlohmann/json.hpp"

#include <ctime>
#include <future>
#include <atomic>
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

/** @brief Device-free HTTP fixture that forbids live probes and unsolicited ledger copies. */
class Test__ChatCompletionHandler : public ::testing::Test
{
protected:
    /** @brief Install inert dependencies and strict default observation boundaries. */
    void SetUp() override
    {
        runner_ = std::make_unique<MockOrchestrationRunner>();
        tokenizer_ = std::make_unique<MockTokenizer>();

        // Default: runner is initialized
        runner_->simulateInitialized();
        EXPECT_CALL(*runner_, maybeApplyMoERebalance(_))
            .Times(AnyNumber())
            .WillRepeatedly(Return(true));
        EXPECT_CALL(*runner_, prefixStateProbe(testing::_))
            .Times(0);
        EXPECT_CALL(*runner_, requestRuntimeSummary())
            .Times(AnyNumber())
            .WillRepeatedly(Return(RequestRuntimeSummary{}));
        // INFO logging and ordinary replies must never copy the lifetime ledger.
        EXPECT_CALL(*runner_, moeOptimizationMovementLedger()).Times(0);
        previous_log_level_ = Logger::getInstance().getLogLevel();
        previous_token_trace_ = debugEnv().runtime_debug.trace_generated_tokens;
        Logger::getInstance().setLogLevel(LogLevel::INFO);

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

    /** Restore process-local logging so the fixture cannot affect other suites. */
    void TearDown() override
    {
        Logger::getInstance().setLogLevel(previous_log_level_);
        mutableDebugEnv().runtime_debug.trace_generated_tokens = previous_token_trace_;
    }

    LogLevel previous_log_level_ = LogLevel::INFO;
    bool previous_token_trace_ = false; ///< Restore the caller's opt-in observation policy.

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

/** @test OpenCode reasoning survives JSON parsing and the model's next-turn template. */
TEST_F(Test__ChatCompletionHandler, AssistantReasoningHistorySurvivesTemplateRoundTrip)
{
    // This is the native template boundary: reasoning is a separate message
    // member, including on assistant messages whose only output is a tool call.
    auto tmpl = ChatTemplate::create(R"({%- for message in messages %}
{{- '<|im_start|>' + message.role + '\n' }}
{%- if message.reasoning_content is string %}
{{- '<think>\n' + message.reasoning_content + '\n</think>\n\n' }}
{%- endif %}
{{- message.content if message.content is string else '' }}
{{- '<|im_end|>\n' }}
{%- endfor %})");
    ASSERT_TRUE(tmpl->hasJinjaSupport());
    const std::string first = "Retain indentation and literal \\n. 🧪\nUse atomic writes.";
    const std::string second = "The previous tests passed. 🙂";
    for (bool escaped : {false, true})
    {
        for (bool thinking : {false, true})
        {
            json body = {{"messages", json::array({
                {{"role", "user"}, {"content", "Build the app."}},
                {{"role", "assistant"}, {"content", nullptr}, {"reasoning_content", first},
                 {"tool_calls", json::array({{{"id", "call_1"}, {"type", "function"},
                     {"function", {{"name", "write"}, {"arguments", "{}"}}}}})}},
                {{"role", "tool"}, {"content", "Written"}, {"tool_call_id", "call_1"}},
                {{"role", "assistant"}, {"content", "Done."}, {"reasoning_content", second}},
                {{"role", "user"}, {"content", "Add persistence."}}
            })}};
            ChatCompletionResponse error;
            auto request = ChatCompletionHandler::parseRequest(body.dump(-1, ' ', escaped), error);
            ASSERT_TRUE(request.has_value()) << error.json_body;
            const auto rendered = tmpl->apply(request->messages, true, thinking);
            EXPECT_NE(rendered.find("<think>\n" + first + "\n</think>"), std::string::npos);
            EXPECT_NE(rendered.find("<think>\n" + second + "\n</think>"), std::string::npos);
        }
    }
}

/** @test Invalid reasoning is rejected before relaxed tool-call content validation. */
TEST_F(Test__ChatCompletionHandler, ReasoningHistoryRejectsNonStringValues)
{
    for (const json &invalid : {json(1), json(false), json::array(), json::object()})
    {
        for (bool tool_call : {false, true})
        {
            json message = {{"role", "assistant"}, {"content", "answer"},
                            {"reasoning_content", invalid}};
            if (tool_call)
                message["tool_calls"] = json::array();
            ChatCompletionResponse error;
            auto request = ChatCompletionHandler::parseRequest(
                json{{"messages", json::array({message})}}.dump(), error);
            EXPECT_FALSE(request.has_value());
            EXPECT_EQ(error.http_status, 400);
        }
    }
    for (const json &empty : {json(nullptr), json("")})
    {
        ChatCompletionResponse error;
        EXPECT_TRUE(ChatCompletionHandler::parseRequest(json{{"messages", json::array({
            {{"role", "assistant"}, {"content", "answer"}, {"reasoning_content", empty}}
        })}}.dump(), error).has_value());
    }
}

/** @test Both HTTP paths select the request mode and preserve every explicit field. */
TEST_F(Test__ChatCompletionHandler, ThinkingSamplingDefaultsRespectEveryExplicitOverride)
{
    auto handler = makeHandler();
    ON_CALL(*tokenizer_, encodeChat(_, _, _)).WillByDefault(Return(std::vector<int>{10, 20}));
    ON_CALL(*runner_, prefill(_)).WillByDefault(Return(true));
    ON_CALL(*tokenizer_, is_stop_token(_)).WillByDefault(Return(false));
    ON_CALL(*runner_, decodeStep()).WillByDefault(Return(makeToken(0, true)));
    for (bool default_thinking : {false, true})
    {
        ON_CALL(*runner_, getDefaultThinkingMode()).WillByDefault(Return(
            default_thinking ? ThinkingMode::Enabled : ThinkingMode::Disabled));
        for (bool stream : {false, true})
        {
            // Omitted, top-level and official nested spelling share one policy.
            for (int mode : {0, 1, 2, 3, 4})
            {
                const bool thinking = mode == 2 ? default_thinking : mode == 1 || mode == 4;
                SamplingParams defaults;
                defaults.temperature = thinking ? 1.0f : 0.7f;
                defaults.top_p = thinking ? 0.95f : 0.80f;
                defaults.top_k = 20;
                defaults.presence_penalty = thinking ? 0.0f : 1.5f;
                defaults.frequency_penalty = 0.25f;
                defaults.repetition_penalty = 1.05f;
                for (unsigned fields = 0; fields < 64; ++fields)
                {
                    SCOPED_TRACE(::testing::Message() << default_thinking << '/' << stream << '/' << mode << '/' << fields);
                    json overrides = {{"stream", stream}, {"max_tokens", 1}};
                    if (mode < 2) overrides["enable_thinking"] = thinking;
                    if (mode > 2) overrides["chat_template_kwargs"] = {{"enable_thinking", thinking}};
                    if (fields & 1) overrides["temperature"] = 0.0;
                    if (fields & 2) overrides["top_p"] = 1.0;
                    if (fields & 4) overrides["top_k"] = 0;
                    if (fields & 8) overrides["presence_penalty"] = -0.5;
                    if (fields & 16) overrides["repetition_penalty"] = 1.0;
                    if (fields & 32) overrides["frequency_penalty"] = 0.0;
                    EXPECT_CALL(*runner_, getRecommendedSamplingParams(
                        thinking ? ThinkingMode::Enabled : ThinkingMode::Disabled))
                        .WillOnce(Return(defaults));
                    EXPECT_CALL(*tokenizer_, encodeChat(_, _, _, thinking))
                        .WillOnce(Return(std::vector<int>{10, 20}));
                    SamplingParams actual;
                    EXPECT_CALL(*runner_, setSamplingParams(_))
                        .WillOnce(Invoke([&](const SamplingParams &params) { actual = params; }));
                    if (stream)
                    {
                        const auto error = handler->handleRawRequest(minimalRequest(overrides),
                            [](const std::string &) { return true; });
                        EXPECT_TRUE(error.ok) << error.json_body;
                    }
                    else
                    {
                        const auto response = handler->handleRawRequest(minimalRequest(overrides));
                        EXPECT_TRUE(response.ok) << response.json_body;
                    }
                    EXPECT_FLOAT_EQ(actual.temperature, fields & 1 ? 0.0f : defaults.temperature);
                    EXPECT_FLOAT_EQ(actual.top_p, fields & 2 ? 1.0f : defaults.top_p);
                    EXPECT_EQ(actual.top_k, fields & 4 ? 0 : defaults.top_k);
                    EXPECT_FLOAT_EQ(actual.presence_penalty, fields & 8 ? -0.5f : defaults.presence_penalty);
                    EXPECT_FLOAT_EQ(actual.repetition_penalty, fields & 16 ? 1.0f : defaults.repetition_penalty);
                    EXPECT_FLOAT_EQ(actual.frequency_penalty, fields & 32 ? 0.0f : defaults.frequency_penalty);
                }
            }
        }
    }
}

/** @test Reasoning controls reject conflicting or mistyped requests before inference. */
TEST_F(Test__ChatCompletionHandler, ThinkingModeRejectsInvalidOrConflictingOverrides)
{
    for (const json &invalid : {json(nullptr), json(0), json("false"), json::array(), json::object()})
    {
        for (bool nested : {false, true})
        {
            json overrides = nested ? json{{"chat_template_kwargs", {{"enable_thinking", invalid}}}}
                                    : json{{"enable_thinking", invalid}};
            ChatCompletionResponse error;
            EXPECT_FALSE(ChatCompletionHandler::parseRequest([&] { auto body = json::parse(minimalRequest()); body.update(overrides); return body.dump(); }(), error));
            EXPECT_EQ(error.http_status, 400);
        }
    }
    for (bool top : {false, true})
    {
        for (bool nested : {false, true})
        {
            ChatCompletionResponse error;
            const auto request = ChatCompletionHandler::parseRequest(minimalRequest({
                {"enable_thinking", top}, {"chat_template_kwargs", {{"enable_thinking", nested}}}}), error);
            EXPECT_EQ(request.has_value(), top == nested);
            if (request) EXPECT_EQ(request->enable_thinking, top);
            else EXPECT_EQ(error.http_status, 400);
        }
    }
    for (const json &invalid : {json(7), json(false), json::array()})
    {
        ChatCompletionResponse error;
        EXPECT_FALSE(ChatCompletionHandler::parseRequest(minimalRequest({{"chat_template_kwargs", invalid}}), error));
        EXPECT_EQ(error.http_status, 400);
    }
}

/** @test Repetition factors are positive finite scalars; neutral one is still explicit. */
TEST_F(Test__ChatCompletionHandler, RepetitionPenaltyAdmission)
{
    for (const json &invalid : {json(nullptr), json(false), json("1.05"), json::array(), json::object(),
                               json(0), json(-1.0), json(1e100)})
    {
        ChatCompletionResponse error;
        // merge_patch removes null, so construct the full body directly here.
        auto body = json::parse(minimalRequest());
        body["repetition_penalty"] = invalid;
        EXPECT_FALSE(ChatCompletionHandler::parseRequest(body.dump(), error));
        EXPECT_EQ(error.http_status, 400);
    }
    for (float value : {0.5f, 1.0f, 1.05f, 1.1f, 2.0f})
    {
        ChatCompletionResponse error;
        const auto request = ChatCompletionHandler::parseRequest(minimalRequest({{"repetition_penalty", value}}), error);
        ASSERT_TRUE(request) << error.json_body;
        EXPECT_TRUE(request->sampling_set.repetition_penalty);
        EXPECT_FLOAT_EQ(request->sampling.repetition_penalty, value);
    }
}

/** @test An explicitly empty reasoning string remains distinct from absent/null history. */
TEST_F(Test__ChatCompletionHandler, EmptyReasoningHistoryPreservesTemplatePresence)
{
    const auto tmpl = ChatTemplate::create(
        "{% for message in messages %}{% if message.reasoning_content is string %}string:{{ message.reasoning_content }}{% else %}absent{% endif %}{% endfor %}");
    for (int form : {0, 1, 2})
    {
        json message = {{"role", "assistant"}, {"content", "answer"}};
        if (form != 0) message["reasoning_content"] = form == 1 ? json(nullptr) : json("");
        ChatCompletionResponse error;
        const auto request = ChatCompletionHandler::parseRequest(json{{"messages", json::array({message})}}.dump(), error);
        ASSERT_TRUE(request);
        EXPECT_EQ(request->messages.front().reasoning_content.has_value(), form == 2);
        EXPECT_EQ(tmpl->apply(request->messages, false, true), form == 2 ? "string:" : "absent");
    }
}

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
    ON_CALL(*runner_, getRecommendedSamplingParams(_))
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
    EXPECT_FALSE(body.contains("token_ids"));
    EXPECT_FALSE(body.contains("runtime_summary"));
    EXPECT_EQ(body["object"], "chat.completion");
    EXPECT_EQ(body["choices"][0]["message"]["role"], "assistant");
    EXPECT_EQ(body["choices"][0]["message"]["content"], "hello world");
    EXPECT_EQ(body["choices"][0]["finish_reason"], "stop");
    EXPECT_EQ(body["usage"]["prompt_tokens"], 3);
    EXPECT_EQ(body["usage"]["completion_tokens"], 3); // 100, 200, 0(stop)
    EXPECT_EQ(body["usage"]["total_tokens"], 6);
}

/** Optional token IDs are terminal output, never a new inference mode. */
TEST_F(Test__ChatCompletionHandler, TokenIds_RequestPolicyIsExplicitAndNonStreaming)
{
    ChatCompletionResponse error;
    auto request = ChatCompletionHandler::parseRequest(minimalRequest(), error);
    ASSERT_TRUE(request);
    EXPECT_EQ(request->token_output, CompletionTokenOutput::TextOnly);
    request = ChatCompletionHandler::parseRequest(minimalRequest({{"return_token_ids", true}}), error);
    ASSERT_TRUE(request);
    EXPECT_EQ(request->token_output, CompletionTokenOutput::TextAndIds);
    for (const json &invalid : {json(nullptr), json(1), json("true"), json::array()})
    {
        auto body = json::parse(minimalRequest());
        body["return_token_ids"] = invalid;
        EXPECT_FALSE(ChatCompletionHandler::parseRequest(body.dump(), error));
        EXPECT_EQ(error.http_status, 400);
    }
    EXPECT_FALSE(ChatCompletionHandler::parseRequest(
        minimalRequest({{"return_token_ids", true}, {"stream", true}}), error));
    EXPECT_EQ(error.http_status, 400);
    EXPECT_TRUE(ChatCompletionHandler::parseRequest(
        minimalRequest({{"return_token_ids", false}, {"stream", true}}), error));
}

/** A typed streaming caller cannot silently drop the requested representation. */
TEST_F(Test__ChatCompletionHandler, TokenIds_StreamingRejectionPrecedesInference)
{
    EXPECT_CALL(*runner_, clearCache()).Times(0);
    EXPECT_CALL(*runner_, prefill(_)).Times(0);
    EXPECT_CALL(*runner_, decodeStep()).Times(0);
    ChatCompletionRequest request;
    request.token_output = CompletionTokenOutput::TextAndIds;
    request.stream = true;
    auto response = makeHandler()->handleStreamingRequest(request, [](const std::string &) {
        ADD_FAILURE() << "unsupported response must not emit chunks";
        return true;
    });
    EXPECT_FALSE(response.ok);
    EXPECT_EQ(response.http_status, 400);
}

/** The optional completed summary has explicit admission, not truthy JSON flags. */
TEST_F(Test__ChatCompletionHandler, RuntimeSummary_ParsesTypedNonStreamingPolicy)
{
    ChatCompletionResponse error;
    auto request = ChatCompletionHandler::parseRequest(minimalRequest(), error);
    ASSERT_TRUE(request);
    EXPECT_EQ(request->runtime_output, CompletionRuntimeOutput::Omit);
    request = ChatCompletionHandler::parseRequest(minimalRequest({{"return_runtime_summary", true}}), error);
    ASSERT_TRUE(request);
    EXPECT_EQ(request->runtime_output, CompletionRuntimeOutput::Include);
    for (const json &invalid : {json(nullptr), json(1), json("true"), json::array()})
    {
        auto body = json::parse(minimalRequest());
        body["return_runtime_summary"] = invalid;
        EXPECT_FALSE(ChatCompletionHandler::parseRequest(
            body.dump(), error));
        EXPECT_EQ(error.http_status, 400);
    }
    EXPECT_FALSE(ChatCompletionHandler::parseRequest(
        minimalRequest({{"return_runtime_summary", true}, {"stream", true}}), error));
    EXPECT_EQ(error.http_status, 400);
    EXPECT_TRUE(ChatCompletionHandler::parseRequest(
        minimalRequest({{"return_runtime_summary", false}, {"stream", true}}), error));
}

/** A typed streaming caller cannot silently discard requested terminal evidence. */
TEST_F(Test__ChatCompletionHandler, RuntimeSummary_RejectsStreamingBeforeInference)
{
    EXPECT_CALL(*runner_, clearCache()).Times(0);
    EXPECT_CALL(*runner_, prefill(_)).Times(0);
    EXPECT_CALL(*runner_, decodeStep()).Times(0);
    EXPECT_CALL(*runner_, requestRuntimeSummary()).Times(0);
    ChatCompletionRequest request;
    request.runtime_output = CompletionRuntimeOutput::Include;
    EXPECT_EQ(makeHandler()->handleStreamingRequest(request,
        [](const std::string &) { return true; }).http_status, 400);
}

/** JSON and logs share one terminal observation even when INFO is disabled. */
TEST_F(Test__ChatCompletionHandler, RuntimeSummary_ProjectsOutcomeWithoutLiveProbe)
{
    ON_CALL(*tokenizer_, encodeChat(_, _, _)).WillByDefault(Return(std::vector<int>{7, 8}));
    ON_CALL(*runner_, prefill(_)).WillByDefault(Return(true));
    ON_CALL(*tokenizer_, is_stop_token(_)).WillByDefault(Return(false));
    ON_CALL(*tokenizer_, decode_token(10)).WillByDefault(Return("A"));
    EXPECT_CALL(*runner_, decodeStep()).Times(2).WillRepeatedly(Return(makeToken(10)));
    EXPECT_CALL(*runner_, prefixStateProbe(_)).Times(0);
    RequestRuntimeSummary outcome;
    outcome.prefix_request.enabled = true;
    outcome.prefix_request.hit = false; // Full-hit and partial-hit outcomes are exclusive.
    outcome.prefix_request.partial_hit = true;
    outcome.prefix_request.requested_tokens = 2;
    outcome.prefix_request.matched_tokens = 1;
    outcome.prefix_request.hybrid_state_restored = true;
    outcome.prefix_request.storage_tier = "ram";
    outcome.prefix_request.admission_placement_epochs = PrefixPlacementEpochSpan::covering(17, 19);
    outcome.prefix_request.completion_movement_epoch = 23;
    outcome.mtp_request.enabled = true;
    outcome.mtp_request.draft_steps = 27;
    outcome.mtp_request.adaptive_depth_enabled = true;
    outcome.mtp_request.max_depth = 15;
    outcome.mtp_verifier_runs = 9;
    EXPECT_CALL(*runner_, requestRuntimeSummary()).Times(2).WillRepeatedly(Return(outcome));
    EXPECT_CALL(*runner_, moeOptimizationMovementLedger())
        .Times(2).WillRepeatedly(Return(MoEOptimizationMovementLedger{}));
    ChatCompletionRequest request;
    request.messages = {{"user", "story"}};
    request.max_tokens = 1;
    request.enable_thinking = false;
    request.runtime_output = CompletionRuntimeOutput::Include;
    for (const auto level : {LogLevel::INFO, LogLevel::WARN})
    {
        Logger::getInstance().setLogLevel(level);
        const auto response = makeHandler()->handleRequest(request);
        ASSERT_TRUE(response.ok);
        const auto body = json::parse(response.json_body);
        EXPECT_FALSE(body.contains("token_ids")); // Independent extension.
        const auto &summary = body.at("runtime_summary");
        EXPECT_EQ(summary.at("schema"), 1);
        const auto &prefix = summary.at("prefix_cache");
        EXPECT_EQ(prefix.at("partial_hit"), true);
        EXPECT_EQ(prefix.at("hit"), false);
        EXPECT_EQ(prefix.at("matched_tokens"), 1);
        EXPECT_EQ(prefix.at("hybrid_state_restored"), true);
        EXPECT_EQ(prefix.at("storage_tier"), "ram");
        EXPECT_EQ(prefix.at("admission_epoch_earliest"), 17);
        EXPECT_EQ(prefix.at("admission_epoch_latest"), 19);
        EXPECT_EQ(prefix.at("completion_movement_epoch"), 23);
        EXPECT_EQ(summary.at("mtp").at("draft_steps"), 27);
        EXPECT_EQ(summary.at("mtp").at("verifier_runs"), 9);
        EXPECT_EQ(summary.at("mtp").at("max_depth"), 15);
        EXPECT_EQ(summary.at("expert_movement").at("scope"), "model_lifetime");
        EXPECT_TRUE(summary.at("expert_movement").at("edges").empty());
    }
}

/** Independent pipeline observations stay scoped on the real HTTP response path. */
TEST_F(Test__ChatCompletionHandler, RuntimeSummary_PipelineEpochsStayStageScoped)
{
    ON_CALL(*tokenizer_, encodeChat(_, _, _)).WillByDefault(Return(std::vector<int>{7, 8}));
    ON_CALL(*runner_, prefill(_)).WillByDefault(Return(true));
    ON_CALL(*tokenizer_, is_stop_token(_)).WillByDefault(Return(false));
    ON_CALL(*tokenizer_, decode_token(10)).WillByDefault(Return("A"));
    EXPECT_CALL(*runner_, decodeStep()).Times(1).WillOnce(Return(makeToken(10)));
    EXPECT_CALL(*runner_, prefixStateProbe(_)).Times(0);
    RequestRuntimeSummary outcome;
    outcome.prefix_request.enabled = true;
    outcome.prefix_request.movement_stages = MoEOptimizationStages<PrefixMovementEpochObservation>::seal({
        {{0, 0, 2, 2, {GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)}, false},
            {.admission = PrefixPlacementEpochSpan::at(100), .completion = 100}},
        {{1, 2, 4, 5, {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)}, true},
            {.admission = PrefixPlacementEpochSpan::covering(1, 2), .completion = 3}}});
    EXPECT_CALL(*runner_, requestRuntimeSummary()).Times(1).WillOnce(Return(outcome));
    EXPECT_CALL(*runner_, moeOptimizationMovementLedger()).Times(1).WillOnce(Return(MoEOptimizationMovementLedger{}));
    ChatCompletionRequest request;
    request.messages = {{"user", "story"}};
    request.max_tokens = 1;
    request.enable_thinking = false;
    request.runtime_output = CompletionRuntimeOutput::Include;
    const auto response = makeHandler()->handleRequest(request);
    ASSERT_TRUE(response.ok);
    const auto summary = json::parse(response.json_body).at("runtime_summary");
    EXPECT_EQ(summary.at("schema"), 2);
    const auto &prefix = summary.at("prefix_cache");
    EXPECT_TRUE(prefix.at("admission_epoch_earliest").is_null());
    EXPECT_TRUE(prefix.at("admission_epoch_latest").is_null());
    EXPECT_TRUE(prefix.at("completion_movement_epoch").is_null());
    const auto &stages = prefix.at("movement_epochs").at("stages");
    ASSERT_EQ(stages.size(), 2);
    EXPECT_EQ(stages[0].at("epochs").at("completion_movement_epoch"), 100);
    EXPECT_EQ(stages[1].at("epochs").at("admission_epoch_earliest"), 1);
    EXPECT_EQ(stages[1].at("epochs").at("admission_epoch_latest"), 2);
    EXPECT_EQ(stages[1].at("epochs").at("completion_movement_epoch"), 3);
    EXPECT_EQ(stages[1].at("identity").at("routed_last_layer"), 5);
}

namespace
{
    /**
     * @brief Build a complete two-edge cycle without creating any device state.
     * @param authority Host or device owner whose evidence is being projected.
     * @return Valid publication, economy and (for Host) admission records.
     */
    MoEOptimizationMovementLedger completedMovementFixture(MoEOptimizationAuthority authority)
    {
        MoEOptimizationMovementEdge promotion{
            .authority = authority, .transaction = 2, .candidate_epoch = 2,
            .layer = 31, .expert = 75, .cycle_index = 0, .cycle_size = 2,
            .direction = MoEOptimizationMovementDirection::Promotion,
            .axis = MoEOptimizationMovementAxis::Combined,
            .source_participant = 4, .destination_participant = 1,
            .source_priority = 17, .destination_priority = -20,
            .source_device = authority == MoEOptimizationAuthority::Host
                ? DeviceId::cpu() : DeviceId::cuda(1),
            .destination_device = DeviceId::rocm(3),
            .source_world_rank = 7, .destination_world_rank = 99,
            .source_world_rank_known = true, .destination_world_rank_known = false,
            .estimated_weight_bytes = (std::uint64_t{1} << 55) + 3,
            .activation_count = 123, .blocking_inference = false,
        };
        auto demotion = promotion;
        demotion.expert = 108;
        demotion.direction = MoEOptimizationMovementDirection::Demotion;
        std::swap(demotion.source_participant, demotion.destination_participant);
        std::swap(demotion.source_priority, demotion.destination_priority);
        std::swap(demotion.source_device, demotion.destination_device);
        std::swap(demotion.source_world_rank, demotion.destination_world_rank);
        std::swap(demotion.source_world_rank_known, demotion.destination_world_rank_known);
        MoEOptimizationMovementLedger ledger;
        ledger.edges = {promotion, demotion};
        ledger.economy.push_back({
            .authority = authority, .transaction = 2, .candidate_epoch = 2,
            .command_count = 2, .cycle_count = 1,
            .proof = MoEOptimizationTimeEconomy{
            .projected_service_gain_ns = 1000,
            .projected_transfer_and_repack_ns = 100,
            .projected_inference_interference_ns = 10,
            .projected_net_benefit_ns = 890,
            },
        });
        if (authority == MoEOptimizationAuthority::Host)
            ledger.host_admissions.push_back({
                .authority = authority, .transaction = 2, .candidate_epoch = 2,
                .maximum_concurrent_cycles = 3,
                .candidate_cycles = 3, .policy_eligible_cycles = 2,
                .policy_eligible_axes = {.tier_residency = 1, .combined = 1},
                .admitted_candidate_cycles = 1, .admitted_candidate_axes = {.combined = 1},
                .admitted_physical_cycles = 1, .admitted_physical_axes = {.combined = 1},
                .individual_policy_rejected_cycles = 1, .dependent_payoff_rejected_cycles = 1,
                .dependent_cohort_candidates = 2, .dependent_cohort_payoff_rejections = 1,
                .policy_bounded = true,
            });
        return ledger;
    }
}

/** Device-free projection retains exact identities, integer widths and every proof family. */
TEST(Test__MoEMovementLedgerJson, ProjectsHostAndDeviceEvidenceWithoutRecounting)
{
    for (const auto authority : {MoEOptimizationAuthority::Host, MoEOptimizationAuthority::Device})
    {
        const auto ledger = completedMovementFixture(authority);
        const auto wire = json::parse(moeMovementLedgerJson(ledger).dump());
        EXPECT_EQ(wire.at("schema"), 2);
        EXPECT_EQ(wire.at("scope"), "model_lifetime");
        EXPECT_EQ(wire.at("complete"), true);
        ASSERT_EQ(wire.at("edges").size(), 2u);
        const auto &edge = wire.at("edges").at(0);
        EXPECT_EQ(edge.at("authority"), authority == MoEOptimizationAuthority::Host ? "host" : "device");
        EXPECT_EQ(edge.at("transaction"), 2);
        EXPECT_EQ(edge.at("candidate_epoch"), 2);
        EXPECT_EQ(edge.at("layer"), 31);
        EXPECT_EQ(edge.at("expert"), 75);
        EXPECT_EQ(edge.at("cycle_index"), 0);
        EXPECT_EQ(edge.at("cycle_size"), 2);
        EXPECT_EQ(edge.at("source_participant"), 4);
        EXPECT_EQ(edge.at("destination_participant"), 1);
        EXPECT_EQ(edge.at("source_priority"), 17);
        EXPECT_EQ(edge.at("destination_priority"), -20);
        EXPECT_EQ(edge.at("source_device"), ledger.edges[0].source_device.toString());
        EXPECT_EQ(edge.at("destination_device"), DeviceId::rocm(3).toString());
        EXPECT_EQ(edge.at("source_world_rank"), 7);
        EXPECT_TRUE(edge.at("destination_world_rank").is_null());
        EXPECT_TRUE(edge.at("estimated_weight_bytes").is_number_unsigned());
        EXPECT_EQ(edge.at("estimated_weight_bytes").get<std::uint64_t>(), ledger.edges[0].estimated_weight_bytes);
        EXPECT_EQ(edge.at("activation_count"), 123);
        EXPECT_EQ(edge.at("blocking_inference"), false);
        EXPECT_EQ(edge.at("movement_axis"), "combined");
        EXPECT_EQ(edge.at("direction"), "promotion");
        EXPECT_EQ(wire.at("edges").at(1).at("direction"), "demotion");
        EXPECT_EQ(wire.at("edges").at(1).at("expert"), 108);
        ASSERT_EQ(wire.at("economy").size(), 1u);
        EXPECT_EQ(wire.at("economy").at(0).at("policy"), "time_ns");
        EXPECT_EQ(wire.at("economy").at(0).at("projected_net_benefit_ns"), 890);
        EXPECT_EQ(wire.at("economy").at(0).at("command_count"), 2);
        if (authority == MoEOptimizationAuthority::Device)
            EXPECT_TRUE(wire.at("host_admissions").empty());
        else
        {
            ASSERT_EQ(wire.at("host_admissions").size(), 1u);
            const auto &admission = wire.at("host_admissions").at(0);
            EXPECT_EQ(admission.at("cycle_capacity_kind"), "bounded");
            EXPECT_EQ(admission.at("maximum_concurrent_cycles"), 3);
            EXPECT_EQ(admission.at("policy_eligible_axes").at("tier_residency"), 1);
            EXPECT_EQ(admission.at("admitted_physical_axes").at("combined"), 1);
            EXPECT_EQ(admission.at("dependent_cohort_candidates"), 2);
            EXPECT_EQ(admission.at("policy_bounded"), true);
            auto unbounded = ledger;
            unbounded.host_admissions[0].cycle_capacity_kind = MoEOptimizationCycleCapacityKind::Unbounded;
            unbounded.host_admissions[0].maximum_concurrent_cycles = 0;
            const auto alternative = moeMovementLedgerJson(unbounded);
            EXPECT_EQ(alternative.at("host_admissions").at(0).at("cycle_capacity_kind"), "unbounded");
            EXPECT_EQ(alternative.at("host_admissions").at(0).at("maximum_concurrent_cycles"), 0);
        }
    }
}

/** Native policy proofs preserve dimensionless thresholds and exact load counts. */
TEST(Test__MoEMovementLedgerJson, NativePolicyRetainsLoadUnitsWithoutNanosecondEstimates)
{
    auto ledger = completedMovementFixture(MoEOptimizationAuthority::Device);
    const DeviceMoERebalanceLoadSpreadProof load{
        .accepted_spread_improvement = 40, .pre_wave_spread = 100, .post_wave_spread = 60,
        .pre_wave_total = 200, .post_wave_total = 200,
        .pre_participant_spread = 0, .post_participant_spread = 20,
        .pre_participant_total = 200, .post_participant_total = 200,
        .requested_payload_slots = 1, .minimum_improvement_per_slot = 40,
        .maximum_post_spread_per_mille = 300, .ownership_swap_accepts = 1};
    ledger.economy[0].proof = load;
    const auto wire = moeMovementLedgerJson(ledger).at("economy").at(0);
    EXPECT_EQ(wire.at("policy"), "native_load_spread");
    EXPECT_EQ(wire.at("pre_wave_spread"), 100);
    EXPECT_EQ(wire.at("post_wave_spread"), 60);
    EXPECT_EQ(wire.at("minimum_improvement_per_slot"), 40);
    EXPECT_EQ(wire.at("maximum_post_spread_per_mille"), 300);
    EXPECT_EQ(wire.at("ownership_swap_accepts"), 1);
    EXPECT_FALSE(wire.contains("projected_service_gain_ns"));
    EXPECT_FALSE(wire.contains("projected_transfer_and_repack_ns"));
    EXPECT_FALSE(wire.contains("projected_inference_interference_ns"));
    EXPECT_FALSE(wire.contains("projected_net_benefit_ns"));
    ledger.economy[0].authority = MoEOptimizationAuthority::Host;
    EXPECT_THROW(moeMovementLedgerJson(ledger), std::invalid_argument);
    ledger.economy[0].authority = MoEOptimizationAuthority::Device;
    ledger.economy[0].command_count += 1;
    EXPECT_THROW(moeMovementLedgerJson(ledger), std::invalid_argument);
}

/** Static/non-overlay execution supplies a complete empty lifetime observation. */
TEST(Test__MoEMovementLedgerJson, EmptyLedgerIsExplicitAndDoesNotInventAnAuthority)
{
    const auto wire = moeMovementLedgerJson({});
    EXPECT_EQ(wire.at("complete"), true);
    EXPECT_TRUE(wire.at("edges").empty());
    EXPECT_TRUE(wire.at("economy").empty());
    EXPECT_TRUE(wire.at("host_admissions").empty());
    EXPECT_FALSE(wire.contains("authority"));
}

/** Geometry is independent of journal activity, and rejects impossible owner/axis combinations. */
TEST(Test__MoEMovementLedgerJson, FrozenTopologyPreservesEveryAuthorityAndAxis)
{
    EXPECT_EQ(moeMovementTopologyJson({})["authority"], "none");
    for (const auto authority : {MoEOptimizationAuthority::Host, MoEOptimizationAuthority::Device})
        for (const auto axes : {MoEOptimizationMovementAxes::None, MoEOptimizationMovementAxes::TierResidency,
                               MoEOptimizationMovementAxes::ParticipantPlacement, MoEOptimizationMovementAxes::Both})
        {
            const auto wire = moeMovementTopologyJson({.authority = authority, .axes = axes});
            EXPECT_EQ(wire["schema"], 1);
            EXPECT_EQ(wire["scope"], "model_lifetime");
            EXPECT_EQ(wire["authority"], authority == MoEOptimizationAuthority::Host ? "host" : "device");
            EXPECT_EQ(wire["available_axes"].size(),
                      int(hasTierResidencyAxis(axes)) + int(hasParticipantPlacementAxis(axes)));
        }
    for (const auto invalid : {
             MoEOptimizationMovementTopology{.axes = MoEOptimizationMovementAxes::Both},
             MoEOptimizationMovementTopology{.authority = static_cast<MoEOptimizationAuthority>(99)},
             MoEOptimizationMovementTopology{.authority = MoEOptimizationAuthority::Device,
                                            .axes = static_cast<MoEOptimizationMovementAxes>(99)}})
        EXPECT_THROW(moeMovementTopologyJson(invalid), std::invalid_argument);
}

/** Logical objectives cannot be collapsed into physical promotion/demotion direction. */
TEST(Test__MoEMovementLedgerJson, PreservesEveryAxisAndSamePriorityDirection)
{
    auto ledger = completedMovementFixture(MoEOptimizationAuthority::Device);
    auto &edge = ledger.edges[0];
    edge.direction = MoEOptimizationMovementDirection::SamePriority;
    edge.destination_priority = edge.source_priority;
    for (const auto &[axis, name] : {
            std::pair{MoEOptimizationMovementAxis::TierResidency, "tier_residency"},
            std::pair{MoEOptimizationMovementAxis::ParticipantPlacement, "participant_placement"},
            std::pair{MoEOptimizationMovementAxis::Combined, "combined"}})
    {
        edge.axis = axis;
        const auto wire = moeMovementLedgerJson(ledger);
        EXPECT_EQ(wire.at("edges").at(0).at("movement_axis"), name);
        EXPECT_EQ(wire.at("edges").at(0).at("direction"), "same_priority");
    }
}

/** Lost, malformed or contradictory evidence must never become a successful observation. */
TEST(Test__MoEMovementLedgerJson, RejectsTruncationAndMalformedOwnerRecords)
{
    const auto valid = completedMovementFixture(MoEOptimizationAuthority::Host);
    for (auto field : {&MoEOptimizationMovementLedger::discarded_edges,
                       &MoEOptimizationMovementLedger::discarded_economy_records,
                       &MoEOptimizationMovementLedger::discarded_host_admission_records})
    {
        auto ledger = valid;
        ledger.*field = 1;
        EXPECT_THROW(moeMovementLedgerJson(ledger), std::invalid_argument);
    }
    auto ledger = valid;
    ledger.edges.push_back({});
    EXPECT_THROW(moeMovementLedgerJson(ledger), std::invalid_argument);
    ledger = valid;
    ledger.edges[0].axis = static_cast<MoEOptimizationMovementAxis>(255);
    EXPECT_THROW(moeMovementLedgerJson(ledger), std::invalid_argument);
    ledger = valid;
    ledger.edges[0].direction = static_cast<MoEOptimizationMovementDirection>(255);
    EXPECT_THROW(moeMovementLedgerJson(ledger), std::invalid_argument);
    ledger = valid;
    ledger.edges[0].authority = static_cast<MoEOptimizationAuthority>(255);
    EXPECT_THROW(moeMovementLedgerJson(ledger), std::invalid_argument);
    ledger = valid;
    std::get<MoEOptimizationTimeEconomy>(ledger.economy[0].proof).projected_net_benefit_ns += 1;
    EXPECT_THROW(moeMovementLedgerJson(ledger), std::invalid_argument);
    ledger = valid;
    ledger.host_admissions[0].candidate_cycles += 1;
    EXPECT_THROW(moeMovementLedgerJson(ledger), std::invalid_argument);
}

/** Opt-in HTTP observes once, after generation and before the cleanup reset. */
TEST_F(Test__ChatCompletionHandler, RuntimeSummary_MovementIsTerminalPassiveAndBeforeCleanup)
{
    ON_CALL(*tokenizer_, encodeChat(_, _, _)).WillByDefault(Return(std::vector<int>{7, 8}));
    ON_CALL(*runner_, prefill(_)).WillByDefault(Return(true));
    ON_CALL(*tokenizer_, is_stop_token(_)).WillByDefault(Return(false));
    ON_CALL(*tokenizer_, decode_token(10)).WillByDefault(Return("A"));
    const MoEOptimizationStatus optimization_status{
        .authority = MoEOptimizationAuthority::Host,
        .state = MoEOptimizationLifecycleState::Active,
        .activity = MoEOptimizationActivityState::CollectingDemand,
        .published_movement_waves = 1u,
        .completed_movement = {
            .transactions = 1u,
            .commands = 2u,
            .physical_bytes = 4096u,
            .promotions = 1u,
            .demotions = 1u,
        },
        .demand_window = {
            .generation = 7u,
            .collected_routed_rows = 11u,
            .capacity_routed_rows = 256u,
            .scope = MoEOptimizationDemandScope::RoutedRows,
        },
        .completed_decision_windows = 3u,
        .last_decision = MoEOptimizationDecisionReceipt{
            .transaction = 3u,
            .candidate_epoch = 4u,
            .snapshot_observations = 256u,
            .rejected_cycles = 2u,
            .phase_tradeoff_candidates = 1u,
            .payoff_rejected_cycles = 1u,
            .priority_cost_before = 90u,
            .priority_cost_after = 90u,
            .same_priority_makespan_before = 12u,
            .same_priority_makespan_after = 12u,
            .projected_service_gain_ns = 700u,
            .projected_transfer_and_repack_ns = 900u,
            .layer_scan_start = 8u,
            .layer_scan_next = 9u,
        },
    };
    EXPECT_CALL(*runner_, moeOptimizationStatus())
        .Times(1)
        .WillOnce(Return(optimization_status));
    const auto ledger = completedMovementFixture(MoEOptimizationAuthority::Host);
    OrchestrationConfig resolved_config;
    auto plan = std::make_shared<MoERoutedExpertPlacementPlan>();
    plan->enabled = true;
    plan->domains = {{.name = "cpu", .participants = {GlobalDeviceAddress::cpu(0), GlobalDeviceAddress::cpu(1)}},
                     {.name = "gpu", .participants = {GlobalDeviceAddress::rocm(0)}}};
    plan->routed_tiers = {{.name = "a", .domain = "cpu", .priority = 9},
                          {.name = "b", .domain = "gpu", .priority = -3}};
    plan->placements = {{.layer = 0, .routed_expert_tier = {0, 0, 1}}};
    plan->authority_execution = resolveMoEOverlayAuthorityExecutionKind(*plan);
    resolved_config.moe_routed_expert_plan = plan;
    ON_CALL(*runner_, config()).WillByDefault(testing::ReturnRef(resolved_config));
    int completed_steps = 0;
    bool observed = false;
    // A grouped result followed by a terminal row catches both per-step and
    // per-token ledger reads: neither is a completed-request observation.
    EXPECT_CALL(*runner_, decodeStep()).Times(2).WillRepeatedly(Invoke([&] {
        ++completed_steps;
        return completed_steps == 1 ? makeTokens({10, 10}) : makeToken(10);
    }));
    EXPECT_CALL(*runner_, moeOptimizationMovementLedger()).Times(1).WillOnce(Invoke([&] {
        EXPECT_EQ(completed_steps, 2);
        observed = true;
        return ledger;
    }));
    EXPECT_CALL(*runner_, clearCache()).Times(2).WillRepeatedly(Invoke([&] {
        if (completed_steps != 0)
            EXPECT_TRUE(observed);
    }));
    ChatCompletionRequest request;
    request.messages = {{"user", "story"}};
    request.max_tokens = 3;
    request.enable_thinking = false;
    request.runtime_output = CompletionRuntimeOutput::Include;
    const auto response = makeHandler()->handleRequest(request);
    ASSERT_TRUE(response.ok);
    EXPECT_EQ(json::parse(response.json_body).at("usage").at("completion_tokens"), 3);
    EXPECT_EQ(json::parse(response.json_body).at("runtime_summary").at("expert_movement"),
              moeMovementLedgerJson(ledger));
    EXPECT_EQ(json::parse(response.json_body)["runtime_summary"]["expert_movement_topology"],
              moeMovementTopologyJson({.authority = MoEOptimizationAuthority::Host, .axes = MoEOptimizationMovementAxes::Both}));
    EXPECT_EQ(json::parse(response.json_body)["runtime_summary"]["expert_optimization"],
              moeOptimizationStatusJson(optimization_status));
}

/** A requested but truncated movement proof is an HTTP failure, not an empty Static claim. */
TEST_F(Test__ChatCompletionHandler, RuntimeSummary_RejectsIncompleteMovementEvidence)
{
    ON_CALL(*tokenizer_, encodeChat(_, _, _)).WillByDefault(Return(std::vector<int>{7, 8}));
    ON_CALL(*runner_, prefill(_)).WillByDefault(Return(true));
    ON_CALL(*tokenizer_, is_stop_token(_)).WillByDefault(Return(false));
    ON_CALL(*tokenizer_, decode_token(10)).WillByDefault(Return("A"));
    EXPECT_CALL(*runner_, decodeStep()).WillOnce(Return(makeToken(10)));
    MoEOptimizationMovementLedger ledger;
    ledger.discarded_edges = 1;
    EXPECT_CALL(*runner_, moeOptimizationMovementLedger()).Times(1).WillOnce(Return(ledger));
    ChatCompletionRequest request;
    request.messages = {{"user", "story"}};
    request.max_tokens = 1;
    request.enable_thinking = false;
    request.runtime_output = CompletionRuntimeOutput::Include;
    const auto response = makeHandler()->handleRequest(request);
    EXPECT_FALSE(response.ok);
    EXPECT_EQ(response.http_status, 500);
    EXPECT_NE(response.json_body.find("movement ledger"), std::string::npos);
}

/** Preserve grouped order, hidden EOS, actual prompt IDs, and per-request reset. */
TEST_F(Test__ChatCompletionHandler, TokenIds_ObserveCommittedOutputWithoutRetokenizingOrProbing)
{
    ON_CALL(*tokenizer_, encodeChat(_, _, _)).WillByDefault(Return(std::vector<int>{7, 8}));
    ON_CALL(*runner_, prefill(_)).WillByDefault(Return(true));
    ON_CALL(*tokenizer_, is_stop_token(_)).WillByDefault(Return(false));
    ON_CALL(*tokenizer_, decode_token(_)).WillByDefault(Return("same text"));
    EXPECT_CALL(*tokenizer_, encode(_, _, _)).Times(0);
    EXPECT_CALL(*runner_, decodeStep())
        .WillOnce(Return(makeTokens({10, 11, 0}, true)))
        .WillOnce(Return(makeToken(42, true)));
    auto handler = makeHandler();
    ChatCompletionRequest request;
    request.messages = {ChatMessage("user", "observe")};
    request.max_tokens = 8;
    request.enable_thinking = false;
    request.token_output = CompletionTokenOutput::TextAndIds;
    for (const auto expected : {std::vector<int>{10, 11, 0}, std::vector<int>{42}})
    {
        const auto response = handler->handleRequest(request);
        ASSERT_TRUE(response.ok);
        const auto body = json::parse(response.json_body);
        EXPECT_EQ(body.at("token_ids").at("prompt"), (std::vector<int>{7, 8}));
        EXPECT_EQ(body.at("token_ids").at("completion"), expected);
        EXPECT_EQ(body.at("usage").at("completion_tokens"), expected.size());
        EXPECT_EQ(body.at("choices")[0].at("finish_reason"), "stop");
    }
}

/** Do not report tokens that the existing bounded HTTP response did not consume. */
TEST_F(Test__ChatCompletionHandler, TokenIds_RespectResponseBudgetAndEmptyTerminalResult)
{
    ON_CALL(*tokenizer_, encodeChat(_, _, _)).WillByDefault(Return(std::vector<int>{1}));
    ON_CALL(*runner_, prefill(_)).WillByDefault(Return(true));
    ON_CALL(*tokenizer_, is_stop_token(_)).WillByDefault(Return(false));
    ON_CALL(*tokenizer_, decode_token(_)).WillByDefault(Return("text"));
    EXPECT_CALL(*runner_, decodeStep())
        .WillOnce(Return(makeTokens({10, 11, 12})))
        .WillOnce(Return(makeTokens({}, true)));
    auto handler = makeHandler();
    ChatCompletionRequest request;
    request.messages = {ChatMessage("user", "observe")};
    request.max_tokens = 2;
    request.enable_thinking = false;
    request.token_output = CompletionTokenOutput::TextAndIds;
    for (const auto expected : {std::vector<int>{10, 11}, std::vector<int>{}})
    {
        const auto response = handler->handleRequest(request);
        ASSERT_TRUE(response.ok);
        const auto body = json::parse(response.json_body);
        EXPECT_EQ(body.at("token_ids").at("completion"), expected);
        EXPECT_EQ(body.at("usage").at("completion_tokens"), expected.size());
    }
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

/** Token traces account for EOS and runner completion on both public paths. */
TEST_F(Test__ChatCompletionHandler, GeneratedTokenTraceRetainsTerminalAuthority)
{
    ON_CALL(*tokenizer_, encodeChat(_, _, _))
        .WillByDefault(Return(std::vector<int>{1, 2}));
    ON_CALL(*runner_, prefill(_)).WillByDefault(Return(true));
    ON_CALL(*tokenizer_, is_stop_token(_)).WillByDefault(Return(false));
    EXPECT_CALL(*tokenizer_, decode_token(10)).Times(12).WillRepeatedly(Return("A"));
    EXPECT_CALL(*tokenizer_, decode_token(11)).Times(12).WillRepeatedly(Return("B"));
    // Only the six explicitly traced responses may decode the withheld token.
    // Ordinary requests keep the original CPU work and observation boundary.
    EXPECT_CALL(*tokenizer_, decode_token(12)).Times(6)
        .WillRepeatedly(Return("<terminal>"));
    bool tokenizer_stop = false;
    bool runner_complete = false;
    ON_CALL(*tokenizer_, is_stop_token(12))
        .WillByDefault(Invoke([&](int) { return tokenizer_stop; }));
    EXPECT_CALL(*runner_, decodeStep()).Times(12)
        .WillRepeatedly(Invoke([&] { return makeTokens({10, 11, 12}, runner_complete); }));
    ChatCompletionRequest request;
    request.messages = {{"user", "trace this completion"}};
    request.max_tokens = 8;
    request.enable_thinking = false;
    for (const bool trace : {false, true})
    for (const bool stream : {false, true})
    for (const int ending : {0, 1, 2})
    {
        SCOPED_TRACE(::testing::Message() << "trace=" << trace << " stream=" << stream
                                        << " ending=" << ending);
        mutableDebugEnv().runtime_debug.trace_generated_tokens = trace;
        tokenizer_stop = ending != 1;
        runner_complete = ending != 0;
        request.stream = stream;
        std::string wire;
        testing::internal::CaptureStderr();
        const auto response = stream
            ? makeHandler()->handleStreamingRequest(request, [&](const std::string &chunk) {
                  wire += chunk;
                  return true;
              })
            : makeHandler()->handleRequest(request);
        const auto logs = testing::internal::GetCapturedStderr();
        ASSERT_TRUE(response.ok);
        if (!stream)
        {
            const auto body = json::parse(response.json_body);
            EXPECT_EQ(body.at("choices")[0].at("message").at("content"), "AB");
            EXPECT_EQ(body.at("usage").at("completion_tokens"), 3);
            wire = response.json_body;
        }
        EXPECT_EQ(wire.find("<terminal>"), std::string::npos);
        if (!trace)
        {
            EXPECT_EQ(logs.find("[ChatCompletion/token]"), std::string::npos);
            continue;
        }
        const std::string path = stream ? "stream" : "nonstream";
        for (const int index : {0, 1, 2})
        {
            const auto marker = "[ChatCompletion/token] path=" + path + " index=" +
                std::to_string(index) + " token=" + std::to_string(10 + index);
            const auto position = logs.find(marker);
            EXPECT_NE(position, std::string::npos) << logs;
            if (position != std::string::npos)
                EXPECT_EQ(logs.find(marker, position + marker.size()), std::string::npos);
        }
        EXPECT_NE(logs.find(std::string("disposition=") +
                           (tokenizer_stop ? "stop_token" : "runner_complete") +
                           " text=\"<terminal>\""), std::string::npos) << logs;
    }
}

/** Ordinary completion logs must not inspect or synchronize live GPU state. */
TEST_F(Test__ChatCompletionHandler, RuntimeSummaryNeverProbesLiveInferenceState)
{
    auto handler = makeHandler();
    ON_CALL(*tokenizer_, encodeChat(_, _, _))
        .WillByDefault(Return(std::vector<int>{1}));
    ON_CALL(*runner_, prefill(_)).WillByDefault(Return(true));
    ON_CALL(*tokenizer_, is_stop_token(_)).WillByDefault(Return(false));
    ON_CALL(*tokenizer_, decode_token(10)).WillByDefault(Return("A"));
    EXPECT_CALL(*runner_, decodeStep())
        .Times(4).WillRepeatedly(Return(makeToken(10)));
    // A real probe reads ring metadata and can wait for unrelated maintenance.
    // Both HTTP modes must use completed request observations instead.
    EXPECT_CALL(*runner_, prefixStateProbe(testing::_)).Times(0);
    // Passive endpoint accounting remains available with INFO logging disabled.
    EXPECT_CALL(*runner_, requestRuntimeSummary())
        .Times(4).WillRepeatedly(Return(RequestRuntimeSummary{}));
    ChatCompletionRequest request;
    request.messages = {ChatMessage("user", "test")};
    request.max_tokens = 1;
    request.enable_thinking = false;
    for (const auto level : {LogLevel::INFO, LogLevel::WARN})
    {
        Logger::getInstance().setLogLevel(level);
        request.stream = false;
        EXPECT_TRUE(handler->handleRequest(request).ok);
        request.stream = true;
        EXPECT_TRUE(handler->handleStreamingRequest(request,
            [](const std::string &) { return true; }).ok);
    }
}

TEST_F(Test__ChatCompletionHandler, HandleRequest_UsesTerminalRequestSummary)
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

    RequestRuntimeSummary snapshot;
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
    EXPECT_CALL(*runner_, requestRuntimeSummary())
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

/** @brief Both public response modes reject unsupported sampling before prefill or SSE. */
TEST_F(Test__ChatCompletionHandler, UnsupportedMTPSamplingReturns400BeforeInference)
{
    EXPECT_CALL(*runner_, setSamplingParams(_))
        .Times(2).WillRepeatedly(Throw(UnsupportedMTPSamplingRequest{}));
    EXPECT_CALL(*runner_, prefill(_)).Times(0);
    EXPECT_CALL(*runner_, decodeStep()).Times(0);
    EXPECT_CALL(*tokenizer_, encodeChat(_, _, _, _)).Times(0);
    ChatCompletionResponse error;
    auto request = ChatCompletionHandler::parseRequest(
        minimalRequest({{"dry_multiplier", 0.8}, {"temperature", 0.8}}), error);
    ASSERT_TRUE(request);
    auto handler = makeHandler();
    auto response = handler->handleRequest(*request);
    EXPECT_EQ(response.http_status, 400);
    auto body = json::parse(response.json_body);
    EXPECT_EQ(body["error"]["type"], "invalid_request_error");
    EXPECT_EQ(body["error"]["code"], UnsupportedMTPSamplingRequest::code);
    EXPECT_EQ(body["error"]["param"], "dry_multiplier");
    request->stream = true;
    int publications = 0;
    response = handler->handleStreamingRequest(*request, [&](const std::string &) {
        ++publications;
        return true;
    });
    EXPECT_EQ(response.http_status, 400);
    EXPECT_EQ(publications, 0);
    EXPECT_EQ(json::parse(response.json_body)["error"]["code"], UnsupportedMTPSamplingRequest::code);
}

/** @brief Typed admission rejection must not disguise a broken graph as a client error. */
TEST_F(Test__ChatCompletionHandler, SamplingBackendFailureRemains500)
{
    EXPECT_CALL(*runner_, setSamplingParams(_))
        .WillOnce(Throw(std::runtime_error("graph construction failed")));
    EXPECT_CALL(*runner_, prefill(_)).Times(0);
    ChatCompletionRequest request;
    request.messages = {ChatMessage("user", "test")};
    const auto response = makeHandler()->handleRequest(request);
    EXPECT_EQ(response.http_status, 500);
    EXPECT_EQ(json::parse(response.json_body)["error"]["type"], "server_error");
}

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
    EXPECT_CALL(*runner_, maybeApplyMoERebalance(1u))
        .Times(3)
        .WillRepeatedly(Return(true));

    ChatCompletionRequest request;
    request.messages = {ChatMessage("user", "Hello")};
    request.max_tokens = 3;
    request.enable_thinking = false;

    auto response = handler->handleRequest(request);
    EXPECT_TRUE(response.ok);
}

TEST_F(Test__ChatCompletionHandler, HandleRequest_AppliesRebalanceHookAfterFinalCompletedStep)
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
    EXPECT_CALL(*runner_, decodeStep())
        .WillOnce(Return(makeToken(42, true)));
    EXPECT_CALL(*runner_, maybeApplyMoERebalance(1u))
        .Times(1)
        .WillOnce(Return(true));

    ChatCompletionRequest request;
    request.messages = {ChatMessage("user", "Hello")};
    request.max_tokens = 8;
    request.enable_thinking = false;

    auto response = handler->handleRequest(request);
    EXPECT_TRUE(response.ok);
}

TEST_F(Test__ChatCompletionHandler, HandleRequest_UsesUnifiedDecodeBoundaryMaintenance)
{
    auto handler = makeHandler();

    ON_CALL(*tokenizer_, encodeChat(_, _, _))
        .WillByDefault(Return(std::vector<int>{1, 2}));
    ON_CALL(*runner_, prefill(_))
        .WillByDefault(Return(true));
    ON_CALL(*tokenizer_, is_stop_token(_))
        .WillByDefault(Return(false));

    EXPECT_CALL(*runner_, clearCache()).Times(2);
    EXPECT_CALL(*runner_, decodeStep())
        .WillOnce(Return(makeToken(42, true)));
    EXPECT_CALL(*runner_, maybeApplyMoERebalance(1u))
        .Times(1)
        .WillOnce(Return(true));

    ChatCompletionRequest request;
    request.messages = {ChatMessage("user", "Hello")};
    request.max_tokens = 8;
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
    ON_CALL(*runner_, getRecommendedSamplingParams(_))
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
    ON_CALL(*runner_, getRecommendedSamplingParams(_))
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
    ON_CALL(*runner_, getRecommendedSamplingParams(_))
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

    ON_CALL(*runner_, getRecommendedSamplingParams(_))
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
    ON_CALL(*runner_, getRecommendedSamplingParams(_))
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
    ON_CALL(*runner_, getRecommendedSamplingParams(_))
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
    ON_CALL(*runner_, getRecommendedSamplingParams(_))
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
    ON_CALL(*runner_, getRecommendedSamplingParams(_))
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
    ON_CALL(*runner_, getRecommendedSamplingParams(_))
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
    ON_CALL(*runner_, getRecommendedSamplingParams(_))
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
    EXPECT_EQ(result->enable_thinking, false);
}

TEST_F(Test__ChatCompletionHandler, ParseRequest_EnableThinkingDefault_IsTrue)
{
    ChatCompletionResponse error;
    auto result = ChatCompletionHandler::parseRequest(minimalRequest(), error);
    ASSERT_TRUE(result.has_value());
    EXPECT_FALSE(result->enable_thinking.has_value());
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

TEST_F(Test__ChatCompletionHandler, StreamingUsageOptionsAreTypedAndModeBound)
{
    // JSON Merge Patch deletes null-valued fields. Install this field directly
    // so include_usage:null reaches the production parser as an invalid type.
    const auto request_with_options = [](const json &options)
    {
        auto body = json::parse(minimalRequest());
        body["stream"] = true;
        body["stream_options"] = options;
        return body.dump();
    };
    for (const json options : {json(nullptr), json::object(),
                              json{{"include_usage", false}}, json{{"include_usage", true}}})
    {
        ChatCompletionResponse error;
        auto request = ChatCompletionHandler::parseRequest(
            request_with_options(options), error);
        ASSERT_TRUE(request.has_value());
        EXPECT_EQ(request->streaming_usage,
            options.is_object() && options.value("include_usage", false)
                ? StreamingUsageOutput::Include : StreamingUsageOutput::Omit);
    }
    for (const json options : {json(true), json(1), json("usage"), json::array(),
                              json{{"include_usage", nullptr}}, json{{"include_usage", "true"}}})
    {
        ChatCompletionResponse error;
        EXPECT_FALSE(ChatCompletionHandler::parseRequest(
            request_with_options(options), error));
        EXPECT_EQ(error.http_status, 400);
    }
    ChatCompletionResponse error;
    EXPECT_FALSE(ChatCompletionHandler::parseRequest(minimalRequest(
        {{"stream_options", {{"include_usage", true}}}}), error));
    EXPECT_EQ(error.http_status, 400);
    ChatCompletionRequest typed;
    typed.streaming_usage = StreamingUsageOutput::Include;
    EXPECT_CALL(*runner_, prefill(_)).Times(0);
    EXPECT_EQ(makeHandler()->handleRequest(typed).http_status, 400);
}

TEST_F(Test__ChatCompletionHandler, StreamingUsagePublishesExactCountsAfterTerminalChoice)
{
    ON_CALL(*tokenizer_, encodeChat(_, _, _)).WillByDefault(Return(std::vector<int>{1, 2, 3, 4, 5}));
    ON_CALL(*runner_, prefill(_)).WillByDefault(Return(true));
    ON_CALL(*tokenizer_, is_stop_token(_)).WillByDefault(Return(false));
    ON_CALL(*runner_, getToolCallFormat()).WillByDefault(Return(ToolCallFormat::QWEN_3_XML));
    size_t cursor = 0;
    bool batched = false, terminal_length = false;
    EXPECT_CALL(*runner_, decodeStep()).Times(28).WillRepeatedly(Invoke([&] {
        if (batched) return terminal_length ? makeTokens({10, 11}) : makeTokens({10, 11, 0}, true);
        const auto index = cursor++;
        return index < 2 ? makeToken(10 + static_cast<int>(index)) : makeToken(0, true);
    }));
    for (const bool batch : {false, true})
    for (const bool include : {false, true})
        for (const bool tool : {false, true})
            for (const bool length : {false, true})
            {
                SCOPED_TRACE(::testing::Message() << include << " tool=" << tool << " length=" << length);
                cursor = 0;
                batched = batch;
                terminal_length = length;
                // Byte fragments and native tool framing are still just two
                // committed tokens; counting visible JSON characters is wrong.
                ON_CALL(*tokenizer_, decode_token(10)).WillByDefault(Return(tool
                    ? "<tool_call>\n<function=write>\n<parameter=content>\n🙂\n</parameter>\n</function>\n</tool_call>"
                    : std::string("\xf0\x9f", 2)));
                ON_CALL(*tokenizer_, decode_token(11)).WillByDefault(Return(tool
                    ? std::string{} : std::string("\x99\x82", 2)));
                ChatCompletionRequest request;
                request.messages = {ChatMessage("user", "Count committed tokens")};
                request.stream = true;
                request.enable_thinking = false;
                request.max_tokens = length ? 2 : 3;
                request.streaming_usage = include ? StreamingUsageOutput::Include : StreamingUsageOutput::Omit;
                if (tool) request.tools = json::parse(R"([{"type":"function","function":{"name":"write","parameters":{"type":"object","properties":{"content":{"type":"string"}},"required":["content"]}}}])");
                std::vector<json> chunks;
                size_t done = 0;
                const auto response = makeHandler()->publishStreamingRequest(request, [&](const std::string &line) {
                    if (line == "data: [DONE]\n\n") ++done;
                    else chunks.push_back(json::parse(line.substr(6)));
                    return true;
                });
                ASSERT_TRUE(response.ok);
                EXPECT_EQ(done, 1u);
                ASSERT_GE(chunks.size(), 3u);
                const auto &terminal = chunks.at(chunks.size() - (include ? 2 : 1));
                EXPECT_EQ(terminal["choices"][0]["finish_reason"], tool ? "tool_calls" : length ? "length" : "stop");
                for (size_t i = 0; i < chunks.size() - (include ? 1 : 0); ++i)
                {
                    EXPECT_EQ(chunks[i]["id"], chunks[0]["id"]);
                    if (include) EXPECT_TRUE(chunks[i]["usage"].is_null());
                    else EXPECT_FALSE(chunks[i].contains("usage"));
                }
                if (include)
                {
                    EXPECT_TRUE(chunks.back()["choices"].empty());
                    EXPECT_EQ(chunks.back()["id"], chunks[0]["id"]);
                    EXPECT_EQ(chunks.back()["usage"], (json{{"prompt_tokens", 5},
                        {"completion_tokens", length ? 2 : 3}, {"total_tokens", length ? 7 : 8}}));
                }
            }
}

TEST_F(Test__ChatCompletionHandler, StreamingUsageErrorsAndDisconnectsCannotPublishSuccessTail)
{
    ON_CALL(*tokenizer_, encodeChat(_, _, _)).WillByDefault(Return(std::vector<int>{1}));
    ON_CALL(*runner_, prefill(_)).WillByDefault(Return(true));
    ON_CALL(*tokenizer_, is_stop_token(_)).WillByDefault(Return(false));
    ON_CALL(*tokenizer_, decode_token(_)).WillByDefault(Return("hello"));
    ChatCompletionRequest request;
    request.messages = {ChatMessage("user", "Count")};
    request.stream = true;
    request.enable_thinking = false;
    request.streaming_usage = StreamingUsageOutput::Include;
    request.max_tokens = 1;
    for (size_t disconnect_at = 1; disconnect_at <= 4; ++disconnect_at)
    {
        SCOPED_TRACE(disconnect_at);
        size_t attempts = 0;
        ON_CALL(*runner_, decodeStep()).WillByDefault(Return(makeToken(10)));
        const auto response = makeHandler()->publishStreamingRequest(request, [&](const std::string &) {
            return ++attempts != disconnect_at;
        });
        EXPECT_TRUE(response.ok);
        EXPECT_EQ(attempts, disconnect_at);
    }
    for (const bool prefill_error : {false, true})
    {
        ON_CALL(*runner_, prefill(_)).WillByDefault(Return(!prefill_error));
        ON_CALL(*runner_, decodeStep()).WillByDefault(Return(makeFailed("decode failure")));
        size_t successful_usage = 0, done = 0;
        const auto response = makeHandler()->publishStreamingRequest(request, [&](const std::string &line) {
            if (line == "data: [DONE]\n\n") ++done;
            else
            {
                const auto chunk = json::parse(line.substr(6));
                successful_usage += chunk.contains("usage") && chunk["usage"].is_object();
            }
            return true;
        });
        EXPECT_FALSE(response.ok);
        EXPECT_EQ(successful_usage, 0u);
        EXPECT_EQ(done, 1u);
    }
}

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

TEST_F(Test__ChatCompletionHandler, Streaming_UsesTerminalRequestSummary)
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

    RequestRuntimeSummary snapshot;
    snapshot.prefix_request.enabled = true;
    snapshot.prefix_request.requested_tokens = 1;
    snapshot.prefix_request.matched_tokens = 1;
    snapshot.prefix_request.hit = true;
    snapshot.prefix_request.storage_tier = "ram";

    EXPECT_CALL(*runner_, decodeStep())
        .WillOnce(Return(makeToken(10)));
    EXPECT_CALL(*runner_, requestRuntimeSummary())
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

/**
 * @brief Streaming bounds a complete graph admission so HTTP can flush progress.
 *
 * The runner may execute an entire response budget device-side. Giving it the
 * full remaining request made every nominal SSE token arrive only after the
 * terminal result. The window retains grouped/MTP execution while proving the
 * first response is published before later generation windows are submitted.
 */
TEST_F(Test__ChatCompletionHandler, Streaming_BoundsDeviceGenerationPublicationWindow)
{
    auto handler = makeHandler();
    ON_CALL(*tokenizer_, encodeChat(_, _, _))
        .WillByDefault(Return(std::vector<int>{1}));
    ON_CALL(*runner_, prefill(_)).WillByDefault(Return(true));
    ON_CALL(*tokenizer_, is_stop_token(_)).WillByDefault(Return(false));
    ON_CALL(*tokenizer_, decode_token(_)).WillByDefault(Return("x"));

    std::vector<int> nonzero_budgets;
    EXPECT_CALL(*runner_, setDecodeStepTokenBudget(_))
        .Times(6)
        .WillRepeatedly(Invoke([&](int budget)
        {
            if (budget > 0)
                nonzero_budgets.push_back(budget);
        }));
    int decode_calls = 0;
    EXPECT_CALL(*runner_, decodeStep())
        .Times(3)
        .WillRepeatedly(Invoke([&]
        {
            ++decode_calls;
            const int count = decode_calls < 3 ? 16 : 8;
            GenerationResult result;
            result.tokens.assign(static_cast<size_t>(count), 10);
            return result;
        }));

    std::vector<int> nonempty_content_observation;
    ChatCompletionRequest request;
    request.messages = {ChatMessage("user", "stream")};
    request.max_tokens = 40;
    request.stream = true;
    request.enable_thinking = false;
    const auto response = handler->handleStreamingRequest(
        request,
        [&](const std::string &line)
        {
            if (line.starts_with("data: ") &&
                !line.starts_with("data: [DONE]"))
            {
                const auto delta =
                    json::parse(line.substr(6))["choices"][0]["delta"];
                if (!delta.value("content", "").empty())
                    nonempty_content_observation.push_back(decode_calls);
            }
            return true;
        });

    ASSERT_TRUE(response.ok);
    ASSERT_EQ(nonempty_content_observation.size(), 40u);
    EXPECT_EQ(nonzero_budgets, (std::vector<int>{16, 16, 8}));
    EXPECT_EQ(nonempty_content_observation.front(), 1);
    EXPECT_EQ(nonempty_content_observation[16], 2);
    EXPECT_EQ(nonempty_content_observation[32], 3);
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

TEST_F(Test__ChatCompletionHandler, Streaming_AppliesRebalanceHookAfterFinalCompletedStep)
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
    EXPECT_CALL(*runner_, maybeApplyMoERebalance(1u))
        .Times(1)
        .WillOnce(Return(true));

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
    ASSERT_GE(chunks.size(), 1u);
    EXPECT_EQ(chunks.back(), "data: [DONE]\n\n");
}

TEST_F(Test__ChatCompletionHandler, Streaming_UsesUnifiedDecodeBoundaryMaintenance)
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
    EXPECT_CALL(*runner_, maybeApplyMoERebalance(1u))
        .Times(1)
        .WillOnce(Return(true));

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
    ASSERT_GE(chunks.size(), 1u);
    EXPECT_EQ(chunks.back(), "data: [DONE]\n\n");
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

TEST_F(Test__ChatCompletionHandler, ThinkSplitter_DuplicateEndTagPreservesFollowingContent)
{
    StreamingThinkSplitter splitter("</think>");

    // All closing markers are framing, not EOS. A later natural marker must
    // not hide the final answer following a budget-injected marker.
    auto r1 = splitter.process("reasoning</think>\n\nAnswer ");
    EXPECT_EQ(r1.field, "reasoning_content");
    EXPECT_FALSE(splitter.inThinking());

    auto first_content = splitter.flush();
    EXPECT_EQ(first_content.field, "content");
    EXPECT_EQ(first_content.text, "Answer ");

    auto r2 = splitter.process("</th");
    EXPECT_EQ(r2.field, "content");
    EXPECT_TRUE(r2.text.empty());

    auto r3 = splitter.process("ink>\n\nFinal answer");
    EXPECT_EQ(r3.field, "content");
    EXPECT_EQ(r3.text, "\n\nFinal answer");
}

/** @brief Closing markers are idempotent before the answer for every chunk split. */
TEST_F(Test__ChatCompletionHandler, ThinkSplitter_RedundantCloseBeforeAnswerIsChunkInvariant)
{
    const std::string text = "reasoning</think>\n </think>\t</think>\n13</think> preserved";
    for (size_t width = 1; width <= text.size(); ++width)
    {
        SCOPED_TRACE(width);
        StreamingThinkSplitter splitter("</think>");
        std::string reasoning, content;
        const auto collect = [&](const StreamingThinkSplitter::SplitResult &part)
        {
            (part.field == "content" ? content : reasoning) += part.text;
        };
        for (size_t offset = 0; offset < text.size(); offset += width)
            collect(splitter.process(text.substr(offset, width)));
        collect(splitter.flush());
        EXPECT_EQ(reasoning, "reasoning");
        EXPECT_EQ(content, "13 preserved");
        EXPECT_FALSE(splitter.inThinking());
        EXPECT_EQ(splitter.process(" more content").text, " more content");
    }
}

/** @brief Whitespace and partial duplicate markers cannot invent a visible answer. */
TEST_F(Test__ChatCompletionHandler, ThinkSplitter_AwaitingAnswerDoesNotStopAtRepeatedClose)
{
    StreamingThinkSplitter splitter("</think>");
    for (const std::string piece : {"</think>", "\n", "</th", "ink>", "\t"})
    {
        const auto part = splitter.process(piece);
        EXPECT_TRUE(part.text.empty());
    }
    EXPECT_TRUE(splitter.flush().text.empty());
    EXPECT_EQ(splitter.process("13").text, "13");
    EXPECT_TRUE(splitter.process("</think>").text.empty());
    EXPECT_EQ(splitter.process(" final").text, " final");
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
    request.token_output = CompletionTokenOutput::TextAndIds;

    auto response = handler->handleRequest(request);
    EXPECT_TRUE(response.ok);

    auto body = json::parse(response.json_body);
    auto content = body["choices"][0]["message"]["content"].get<std::string>();

    // Forced thinking suffixes are real committed positions, not text-only
    // framing. A token regression trace must retain them and terminal EOS.
    EXPECT_EQ(body.at("token_ids").at("completion"),
              (std::vector<int>{10, 11, 90, 91, 92, 0}));

    // Generated text should contain the injected stop tokens
    EXPECT_NE(content.find("stop"), std::string::npos)
        << "Injected stop-thinking tokens should appear in output";
    EXPECT_NE(content.find(" thinking"), std::string::npos);
    EXPECT_NE(content.find(" now"), std::string::npos);
}

TEST_F(Test__ChatCompletionHandler, HandleRequest_ThinkingBudget_ConsumesEOSAfterDuplicateEndTag)
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
        .WillOnce(Return(makeToken(12))) // Delimiter is not EOS.
        .WillOnce(Return(makeToken(13))) // Preserve the model's final answer.
        .WillOnce(Return(makeToken(0, true)));
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

    EXPECT_EQ(content, "13\n\n\n13");
    EXPECT_EQ(content.find("</think>"), std::string::npos)
        << "Duplicate thinking end tags must not leak into answer content";
    ASSERT_TRUE(message.contains("reasoning_content"));
    EXPECT_NE(message["reasoning_content"].get<std::string>().find("reasoning"),
              std::string::npos);
}

/**
 * @brief HTTP and SSE must both reach the answer after a forced/natural double close.
 *
 * Ornith emitted its own closing marker immediately after the injected budget
 * phrase. The old handler stopped before sampling any visible answer. Keep
 * the forced token, duplicate marker, actual answer and EOS as separate calls
 * so a premature stop violates both mock expectations and response content.
 */
TEST_F(Test__ChatCompletionHandler, ThinkingBudget_RedundantCloseBeforeAnswer_HTTPAndSSE)
{
    auto tmpl = makeThinkingTemplate();
    ON_CALL(*tokenizer_, hasChatTemplate()).WillByDefault(Return(true));
    ON_CALL(*tokenizer_, getChatTemplate()).WillByDefault(::testing::ReturnRef(*tmpl));
    ON_CALL(*tokenizer_, encodeChat(_, _, _)).WillByDefault(Return(std::vector<int>{1, 2, 3}));
    ON_CALL(*tokenizer_, is_stop_token(_)).WillByDefault(Return(false));
    ON_CALL(*runner_, prefill(_)).WillByDefault(Return(true));
    ON_CALL(*runner_, getStopThinkingPrompt()).WillByDefault(Return("budget close</think>\n\n"));
    ON_CALL(*tokenizer_, encode("budget close</think>\n\n", false, false))
        .WillByDefault(Return(std::vector<int>{90}));
    ON_CALL(*tokenizer_, decode_token(10)).WillByDefault(Return("reasoning"));
    ON_CALL(*tokenizer_, decode_token(90)).WillByDefault(Return("budget close</think>\n\n"));
    ON_CALL(*tokenizer_, decode_token(11)).WillByDefault(Return("</th"));
    ON_CALL(*tokenizer_, decode_token(12)).WillByDefault(Return("ink>\n\n"));
    ON_CALL(*tokenizer_, decode_token(13)).WillByDefault(Return("13"));

    for (bool streaming : {false, true})
    {
        SCOPED_TRACE(streaming);
        EXPECT_CALL(*runner_, decodeStep())
            .WillOnce(Return(makeToken(10)))
            .WillOnce(Return(makeToken(11)))
            .WillOnce(Return(makeToken(12)))
            .WillOnce(Return(makeToken(13)))
            .WillOnce(Return(makeToken(0, true)));
        EXPECT_CALL(*runner_, forceDecodeToken(90)).WillOnce(Return(makeToken(90)));
        ChatCompletionRequest request;
        request.messages = {ChatMessage("user", "think briefly")};
        request.max_tokens = 20;
        request.enable_thinking = true;
        request.thinking_budget_tokens = 1;
        auto handler = makeHandler();
        std::string content, reasoning;
        if (streaming)
        {
            auto response = handler->handleStreamingRequest(request, [&](const std::string &line)
            {
                if (line.starts_with("data: ") && !line.starts_with("data: [DONE]"))
                {
                    const auto delta = json::parse(line.substr(6))["choices"][0]["delta"];
                    content += delta.value("content", "");
                    reasoning += delta.value("reasoning_content", "");
                }
                return true;
            });
            EXPECT_TRUE(response.ok);
        }
        else
        {
            auto response = handler->handleRequest(request);
            ASSERT_TRUE(response.ok);
            const auto message = json::parse(response.json_body)["choices"][0]["message"];
            content = message.value("content", "");
            reasoning = message.value("reasoning_content", "");
        }
        EXPECT_EQ(content, "13");
        EXPECT_EQ(reasoning, "reasoningbudget close");
        EXPECT_TRUE(::testing::Mock::VerifyAndClearExpectations(runner_.get()));
    }
}

/**
 * @brief A reasoning delimiter cannot terminate HTTP or SSE generation.
 *
 * The real Qwen3.6 CUDA E2E emitted more reasoning after the budget's forced
 * close, then its natural close before the numeric answer. The field splitter
 * must not treat that second delimiter as EOS and discard the actual answer.
 * Ordered mock calls prove that both response modes consume the answer and EOS.
 */
TEST_F(Test__ChatCompletionHandler, ThinkingBudget_ContinuedReasoningCloseIsNotEOS_HTTPAndSSE)
{
    auto tmpl = makeThinkingTemplate();
    ON_CALL(*tokenizer_, hasChatTemplate()).WillByDefault(Return(true));
    ON_CALL(*tokenizer_, getChatTemplate()).WillByDefault(::testing::ReturnRef(*tmpl));
    ON_CALL(*tokenizer_, encodeChat(_, _, _)).WillByDefault(Return(std::vector<int>{1, 2, 3}));
    ON_CALL(*tokenizer_, is_stop_token(_)).WillByDefault(Return(false));
    ON_CALL(*runner_, prefill(_)).WillByDefault(Return(true));
    ON_CALL(*runner_, getStopThinkingPrompt()).WillByDefault(Return("budget close</think>\n\n"));
    ON_CALL(*tokenizer_, encode("budget close</think>\n\n", false, false))
        .WillByDefault(Return(std::vector<int>{90}));
    ON_CALL(*tokenizer_, decode_token(10)).WillByDefault(Return("reasoning"));
    ON_CALL(*tokenizer_, decode_token(90)).WillByDefault(Return("budget close</think>\n\n"));
    ON_CALL(*tokenizer_, decode_token(11)).WillByDefault(Return("6. Final check complete."));
    ON_CALL(*tokenizer_, decode_token(12)).WillByDefault(Return("</th"));
    ON_CALL(*tokenizer_, decode_token(13)).WillByDefault(Return("ink>\n\n"));
    ON_CALL(*tokenizer_, decode_token(14)).WillByDefault(Return("13"));

    for (bool streaming : {false, true})
    {
        SCOPED_TRACE(streaming);
        std::vector<int> decode_budgets;
        ON_CALL(*runner_, setDecodeStepTokenBudget(_))
            .WillByDefault(Invoke([&](int budget)
            {
                if (budget > 0)
                    decode_budgets.push_back(budget);
            }));
        EXPECT_CALL(*runner_, decodeStep())
            .WillOnce(Return(makeToken(10)))
            .WillOnce(Return(makeToken(11)))
            .WillOnce(Return(makeToken(12)))
            .WillOnce(Return(makeToken(13)))
            .WillOnce(Return(makeToken(14)))
            .WillOnce(Return(makeToken(0, true)));
        EXPECT_CALL(*runner_, forceDecodeToken(90)).WillOnce(Return(makeToken(90)));
        ChatCompletionRequest request;
        request.messages = {ChatMessage("user", "think briefly")};
        request.max_tokens = 20;
        request.enable_thinking = true;
        request.thinking_budget_tokens = 1;
        auto handler = makeHandler();
        std::string content, finish;
        if (streaming)
        {
            const auto response = handler->handleStreamingRequest(request, [&](const std::string &line)
            {
                if (line.starts_with("data: ") && !line.starts_with("data: [DONE]"))
                {
                    const auto choice = json::parse(line.substr(6))["choices"][0];
                    content += choice["delta"].value("content", "");
                    if (!choice["finish_reason"].is_null())
                        finish = choice["finish_reason"].get<std::string>();
                }
                return true;
            });
            EXPECT_TRUE(response.ok);
        }
        else
        {
            const auto response = handler->handleRequest(request);
            ASSERT_TRUE(response.ok);
            const auto choice = json::parse(response.json_body)["choices"][0];
            content = choice["message"].value("content", "");
            finish = choice["finish_reason"].get<std::string>();
        }
        EXPECT_EQ(content, "6. Final check complete.\n\n13");
        EXPECT_EQ(finish, "stop");
        ASSERT_EQ(decode_budgets.size(), 6u);
        EXPECT_EQ(decode_budgets.front(), 1)
            << "Reasoning must use a one-token observation window";
        EXPECT_EQ(decode_budgets[1], streaming ? 16 : 18)
            << "A completed reasoning close must restore the normal answer window";
        EXPECT_TRUE(::testing::Mock::VerifyAndClearExpectations(runner_.get()));
    }
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
    EXPECT_EQ(result->tool_choice.mode, ToolChoiceMode::Auto);
    EXPECT_TRUE(result->parallel_tool_calls);
}

TEST_F(Test__ChatCompletionHandler, ParseRequest_SpecificToolChoiceIsTypedAndValidated)
{
    const auto body = json::parse(R"({
        "messages": [{"role": "user", "content": "Use the weather service."}],
        "tools": [
            {"type":"function","function":{"name":"get_weather","parameters":{"type":"object"}}},
            {"type":"function","function":{"name":"get_time","parameters":{"type":"object"}}}
        ],
        "tool_choice": {"type":"function","function":{"name":"get_weather"}}
    })");

    ChatCompletionResponse error;
    const auto result = ChatCompletionHandler::parseRequest(body.dump(), error);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->tool_choice.mode, ToolChoiceMode::SpecificFunction);
    EXPECT_EQ(result->tool_choice.function_name, "get_weather");
}

TEST_F(Test__ChatCompletionHandler, ParseRequest_UnknownSpecificToolChoiceReturns400)
{
    const auto body = json::parse(R"({
        "messages": [{"role": "user", "content": "Use a service."}],
        "tools": [
            {"type":"function","function":{"name":"get_weather","parameters":{"type":"object"}}}
        ],
        "tool_choice": {"type":"function","function":{"name":"missing"}}
    })");

    ChatCompletionResponse error;
    EXPECT_FALSE(ChatCompletionHandler::parseRequest(body.dump(), error).has_value());
    EXPECT_EQ(error.http_status, 400);
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

/** Reproduce the native payload returned by Qwen 3.5 MoE and Qwen 3.8 dense. */
TEST_F(Test__ChatCompletionHandler, HandleRequest_QwenNativeToolCallBecomesOpenAIResponse)
{
    auto handler = makeHandler();
    const std::string model_output = R"(<tool_call>
<function=get_weather>
<parameter=city>
Paris
</parameter>
</function>
</tool_call>)";

    ON_CALL(*tokenizer_, encodeChat(_, _, _))
        .WillByDefault(Return(std::vector<int>{1, 2, 3}));
    ON_CALL(*runner_, prefill(_))
        .WillByDefault(Return(true));
    ON_CALL(*tokenizer_, is_stop_token(_))
        .WillByDefault(Return(false));
    ON_CALL(*runner_, getToolCallFormat())
        .WillByDefault(Return(ToolCallFormat::QWEN_3_XML));

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
            const int index = token_id - 100;
            return index >= 0 && index < static_cast<int>(model_output.size())
                       ? std::string(1, model_output[index])
                       : std::string{}; }));

    ChatCompletionRequest request;
    request.messages = {ChatMessage("user", "What's the weather?")};
    request.max_tokens = 200;
    request.tools = json::parse(R"([{"type":"function","function":{"name":"get_weather",
        "parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}])");

    const auto response = handler->handleRequest(request);
    ASSERT_TRUE(response.ok);
    const auto body = json::parse(response.json_body);
    EXPECT_EQ(body["choices"][0]["finish_reason"], "tool_calls");
    const auto &message = body["choices"][0]["message"];
    EXPECT_TRUE(message["content"].is_null());
    ASSERT_EQ(message["tool_calls"].size(), 1u);
    EXPECT_EQ(message["tool_calls"][0]["function"]["name"], "get_weather");
    EXPECT_EQ(
        json::parse(message["tool_calls"][0]["function"]["arguments"].get<std::string>()),
        json({{"city", "Paris"}}));
}

/** @test Admission, decode and late exceptions each publish one valid SSE termination. */
TEST_F(Test__ChatCompletionHandler, StreamingTransportErrorsTerminateExactlyOnce)
{
    bool prefill_ok = true, throw_on_decode = false;
    ON_CALL(*tokenizer_, encodeChat(_, _, _)).WillByDefault(Return(std::vector<int>{1}));
    ON_CALL(*runner_, prefill(_)).WillByDefault(Invoke([&](const std::vector<int32_t> &) { return prefill_ok; }));
    EXPECT_CALL(*runner_, decodeStep()).WillRepeatedly(Invoke([&]() -> GenerationResult {
        if (throw_on_decode) throw std::runtime_error("late decoder failure");
        return makeFailed("verifier counts invalid");
    }));
    ChatCompletionRequest request;
    request.messages = {ChatMessage("user", "Hello")};
    request.enable_thinking = false;
    request.max_tokens = 10;
    for (int phase : {0, 1, 2})
    {
        SCOPED_TRACE(phase);
        prefill_ok = phase != 0;
        throw_on_decode = phase == 2;
        int done = 0, errors = 0;
        auto handler = makeHandler();
        const auto response = handler->publishStreamingRequest(request, [&](const std::string &line) {
            EXPECT_EQ(done, 0) << "payload appeared after terminal sentinel";
            if (line == "data: [DONE]\n\n") ++done;
            else
            {
                EXPECT_TRUE(line.starts_with("data: "));
                const auto body = json::parse(line.substr(6));
                if (body.contains("error") || (body.contains("choices") && body["choices"][0]["delta"].contains("error"))) ++errors;
            }
            return true;
        });
        EXPECT_FALSE(response.ok);
        EXPECT_EQ(done, 1);
        EXPECT_EQ(errors, 1);
    }
    prefill_ok = true;
    int writes = 0;
    auto handler = makeHandler();
    handler->publishStreamingRequest(request, [&](const std::string &) { ++writes; return false; });
    EXPECT_EQ(writes, 1) << "disconnection must retire the transport writer";
}

/** @test Literal UTF-8 and JSON surrogate pairs preserve emoji in every chat role. */
TEST_F(Test__ChatCompletionHandler, EmojiChatMessagesParseLiteralAndEscapedJSONExactly)
{
    const std::string emoji = "🙂👩🏽‍💻🇬🇧❤️1️⃣🚀 中文λ";
    const json body = {{"messages", json::array({
        {{"role", "system"}, {"content", emoji}},
        {{"role", "user"}, {"content", emoji}},
        {{"role", "assistant"}, {"content", emoji}},
        {{"role", "tool"}, {"content", emoji}, {"tool_call_id", "call_emoji"}}})}};
    for (bool escaped : {false, true})
    {
        ChatCompletionResponse error;
        const auto request = ChatCompletionHandler::parseRequest(body.dump(-1, ' ', escaped), error);
        ASSERT_TRUE(request);
        ASSERT_EQ(request->messages.size(), 4U);
        for (const auto &message : request->messages) EXPECT_EQ(message.content, emoji);
        EXPECT_EQ(request->messages.back().tool_call_id, "call_emoji");
    }
    ChatCompletionResponse error;
    EXPECT_FALSE(ChatCompletionHandler::parseRequest(
        R"({"messages":[{"role":"user","content":"\ud83d"}]})", error));
    EXPECT_EQ(error.http_status, 400);
}

/** @test Token boundaries cannot corrupt emoji in content, reasoning or tool arguments. */
TEST_F(Test__ChatCompletionHandler, EmojiByteFragmentsSurviveJSONAndSSEAtEveryBoundary)
{
    const std::string emoji = "🙂👩🏽‍💻🇬🇧❤️1️⃣🚀\n中文λ";
    std::vector<std::string> pieces;
    size_t cursor = 0;
    auto thinking_template = makeThinkingTemplate();
    ON_CALL(*tokenizer_, encodeChat(_, _, _)).WillByDefault(Return(std::vector<int>{1}));
    ON_CALL(*runner_, prefill(_)).WillByDefault(Return(true));
    ON_CALL(*runner_, getToolCallFormat()).WillByDefault(Return(ToolCallFormat::QWEN_3_XML));
    ON_CALL(*tokenizer_, is_stop_token(_)).WillByDefault(Return(false));
    ON_CALL(*tokenizer_, hasChatTemplate()).WillByDefault(Return(true));
    ON_CALL(*tokenizer_, getChatTemplate()).WillByDefault(testing::ReturnRef(*thinking_template));
    EXPECT_CALL(*runner_, decodeStep()).WillRepeatedly(Invoke([&] {
        return cursor < pieces.size() ? makeToken(100 + cursor++) : makeToken(0, true);
    }));
    ON_CALL(*tokenizer_, decode_token(_)).WillByDefault(Invoke([&](int token) {
        return token >= 100 && static_cast<size_t>(token - 100) < pieces.size()
            ? pieces[token - 100] : std::string{};
    }));
    for (int lane : {0, 1, 2})
    {
        const std::string output = lane == 1 ? emoji + "</think>" + emoji
            : lane == 2 ? "<tool_call>\n<function=write>\n<parameter=content>\n" + emoji
                + "\n</parameter>\n</function>\n</tool_call>" : emoji;
        for (size_t cut = 0; cut <= output.size() + 1; ++cut)
            for (bool stream : {false, true})
            {
                SCOPED_TRACE(::testing::Message() << "lane=" << lane << " cut=" << cut << " stream=" << stream);
                pieces.clear();
                if (cut == output.size() + 1)
                    for (char byte : output) pieces.emplace_back(1, byte);
                else
                {
                    pieces.push_back(output.substr(0, cut));
                    pieces.push_back(output.substr(cut));
                }
                cursor = 0;
                ChatCompletionRequest request;
                request.messages = {ChatMessage("user", emoji)};
                request.max_tokens = 512;
                request.enable_thinking = lane == 1;
                request.tools = lane == 2 ? json::parse(R"([{"type":"function","function":{"name":"write",
                    "parameters":{"type":"object","properties":{"content":{"type":"string"}}}}}])") : json::array();
                std::string content, reasoning;
                json calls = json::array();
                auto handler = makeHandler();
                ChatCompletionResponse response;
                if (stream)
                    response = handler->handleStreamingRequest(request, [&](const std::string &line) {
                        if (line == "data: [DONE]\n\n") return true;
                        const auto delta = json::parse(line.substr(6))["choices"][0]["delta"];
                        content += delta.value("content", "");
                        reasoning += delta.value("reasoning_content", "");
                        if (delta.contains("tool_calls")) calls = delta["tool_calls"];
                        return true;
                    });
                else
                {
                    response = handler->handleRequest(request);
                    const auto message = json::parse(response.json_body)["choices"][0]["message"];
                    if (message.contains("content") && !message["content"].is_null()) content = message["content"];
                    reasoning = message.value("reasoning_content", "");
                    calls = message.value("tool_calls", json::array());
                }
                ASSERT_TRUE(response.ok);
                if (lane == 2)
                {
                    ASSERT_EQ(calls.size(), 1U);
                    EXPECT_EQ(json::parse(calls[0]["function"]["arguments"].get<std::string>())["content"], emoji);
                }
                else EXPECT_EQ(content, emoji);
                EXPECT_EQ(reasoning, lane == 1 ? emoji : "");
            }
    }
}

/** @test Both HTTP response modes apply OpenCode's string schemas to native parameters. */
TEST_F(Test__ChatCompletionHandler, QwenStringToolArgumentsSurviveJSONAndSSE)
{
    std::string model_output;
    size_t cursor = 0;
    ON_CALL(*tokenizer_, encodeChat(_, _, _)).WillByDefault(Return(std::vector<int>{1}));
    ON_CALL(*runner_, prefill(_)).WillByDefault(Return(true));
    ON_CALL(*runner_, getToolCallFormat()).WillByDefault(Return(ToolCallFormat::QWEN_3_XML));
    ON_CALL(*tokenizer_, is_stop_token(_)).WillByDefault(Return(false));
    EXPECT_CALL(*runner_, decodeStep()).WillRepeatedly(Invoke([&] {
        return cursor < model_output.size() ? makeToken(100 + cursor++) : makeToken(0, true);
    }));
    ON_CALL(*tokenizer_, decode_token(_)).WillByDefault(Invoke([&](int token) {
        return token >= 100 && static_cast<size_t>(token - 100) < model_output.size()
            ? model_output.substr(token - 100, 1) : std::string{};
    }));
    for (const std::string value : {"123", "true", "null", "{\"count\":123}",
                                    "[1,2]", "\"quoted\"", "    first\n\tsecond  \n",
                                    "A</parameter><parameter=x>KEEP</parameter>B\n",
                                    "A</parameter><parameter=filePath>KEEP</parameter>B\n",
                                    "A\n</parameter>\n<parameter=x>\nKEEP\n</parameter>\nB\n",
                                    "🙂</parameter><parameter=x>👩🏽‍💻</parameter>🚀\n",
                                    R"(parts = text.split("\n"); output.write("\t" + "\n".join(parts)))",
                                    R"(pattern = r"\w+\s+\u263a"; path = "C:\\temp\\code.py")"})
        for (const bool stream : {false, true})
        for (const bool inline_close : {false, true})
        {
            SCOPED_TRACE(::testing::Message() << value << " stream=" << stream
                                            << " inline_close=" << inline_close);
            cursor = 0;
            const std::string close = inline_close ? "</parameter>\n" : "\n</parameter>\n";
            // A literal terminal newline requires the separate framing newline
            // used by the template; omitting it is not an unambiguous encoding.
            const std::string content_close = value.ends_with('\n') ? "\n</parameter>\n" : close;
            model_output = "<tool_call>\n<function=write>\n<parameter=content>\n" + value +
                content_close + "<parameter=filePath>\n/tmp/result.txt" + close + "</function>\n</tool_call>";
            ChatCompletionRequest request;
            request.messages = {ChatMessage("user", "Write the exact content")};
            request.max_tokens = 512;
            request.enable_thinking = false;
            request.stream = stream;
            request.tools = json::parse(R"([{"type":"function","function":{"name":"write",
                "parameters":{"type":"object","properties":{"content":{"type":"string"},"filePath":{"type":"string"}},"required":["content","filePath"]}}}])");
            json calls = json::array();
            std::string finish;
            auto handler = makeHandler();
            if (stream)
            {
                const auto response = handler->handleStreamingRequest(request, [&](const std::string &line) {
                    if (!line.starts_with("data: ") || line.starts_with("data: [DONE]")) return true;
                    const auto choice = json::parse(line.substr(6))["choices"][0];
                    if (choice["delta"].contains("tool_calls")) calls = choice["delta"]["tool_calls"];
                    if (!choice["finish_reason"].is_null()) finish = choice["finish_reason"];
                    return true;
                });
                ASSERT_TRUE(response.ok);
            }
            else
            {
                const auto response = handler->handleRequest(request);
                ASSERT_TRUE(response.ok);
                const auto choice = json::parse(response.json_body)["choices"][0];
                calls = choice["message"]["tool_calls"];
                finish = choice["finish_reason"];
            }
            EXPECT_EQ(finish, "tool_calls");
            ASSERT_EQ(calls.size(), 1u);
            EXPECT_EQ(calls[0]["function"]["name"], "write");
            const auto arguments = json::parse(calls[0]["function"]["arguments"].get<std::string>());
            EXPECT_EQ(arguments["content"], value);
            EXPECT_EQ(arguments["filePath"], "/tmp/result.txt");
        }
}

TEST_F(Test__ChatCompletionHandler, HandleRequest_RequiredChoicePublishesPromptPolicy)
{
    auto handler = makeHandler();
    EXPECT_CALL(*tokenizer_, encodeChat(_, true, _, true))
        .WillOnce(Invoke([](const std::vector<ChatMessage> &messages,
                            bool,
                            const std::string &tools_json,
                            bool)
                         {
            EXPECT_FALSE(messages.empty());
            if (messages.empty())
                return std::vector<int>{};
            EXPECT_EQ(messages.front().role, "system");
            EXPECT_NE(messages.front().content.find("MUST call one"), std::string::npos);
            EXPECT_NE(tools_json.find("get_weather"), std::string::npos);
            return std::vector<int>{1, 2, 3}; }));
    ON_CALL(*runner_, prefill(_)).WillByDefault(Return(true));
    ON_CALL(*tokenizer_, is_stop_token(_)).WillByDefault(Return(false));
    ON_CALL(*runner_, getToolCallFormat())
        .WillByDefault(Return(ToolCallFormat::QWEN_3_XML));
    const std::string call =
        "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n"
        "</parameter>\n</function>\n</tool_call>";
    ON_CALL(*tokenizer_, decode_token(100)).WillByDefault(Return(call));
    EXPECT_CALL(*runner_, decodeStep())
        .WillOnce(Return(makeToken(100)))
        .WillOnce(Return(makeToken(0, true)));

    ChatCompletionRequest request;
    request.messages = {ChatMessage("user", "Give me current weather.")};
    request.max_tokens = 10;
    request.tools = json::array(
        {json{{"type", "function"},
              {"function", {{"name", "get_weather"},
                            {"parameters", {{"type", "object"},
                                            {"properties", {{"city", {{"type", "string"}}}}}}}}}}});
    request.tool_choice.mode = ToolChoiceMode::Required;

    const auto response = handler->handleRequest(request);
    ASSERT_TRUE(response.ok);
    EXPECT_EQ(json::parse(response.json_body)["choices"][0]["finish_reason"],
              "tool_calls");
}

TEST_F(Test__ChatCompletionHandler, HandleRequest_NoneChoiceNeitherExposesNorExecutesTools)
{
    auto handler = makeHandler();
    EXPECT_CALL(*tokenizer_, encodeChat(_, true, "", true))
        .WillOnce(Return(std::vector<int>{1, 2, 3}));
    ON_CALL(*runner_, prefill(_)).WillByDefault(Return(true));
    ON_CALL(*tokenizer_, is_stop_token(_)).WillByDefault(Return(false));
    ON_CALL(*runner_, getToolCallFormat())
        .WillByDefault(Return(ToolCallFormat::QWEN_3_XML));
    const std::string literal =
        "<tool_call>\n<function=get_weather>\n</function>\n</tool_call>";
    ON_CALL(*tokenizer_, decode_token(100)).WillByDefault(Return(literal));
    EXPECT_CALL(*runner_, decodeStep())
        .WillOnce(Return(makeToken(100)))
        .WillOnce(Return(makeToken(0, true)));

    ChatCompletionRequest request;
    request.messages = {ChatMessage("user", "Do not use tools.")};
    request.max_tokens = 10;
    request.tools = json::array(
        {json{{"type", "function"},
              {"function", {{"name", "get_weather"}}}}});
    request.tool_choice.mode = ToolChoiceMode::None;

    const auto response = handler->handleRequest(request);
    ASSERT_TRUE(response.ok);
    const auto body = json::parse(response.json_body);
    EXPECT_EQ(body["choices"][0]["finish_reason"], "stop");
    EXPECT_FALSE(body["choices"][0]["message"].contains("tool_calls"));
    EXPECT_EQ(body["choices"][0]["message"]["content"], literal);
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

/**
 * @brief Tool-enabled Qwen streams keep reasoning, answer, and calls distinct.
 *
 * OpenWebUI supplies tools on ordinary agentic requests. The historical path
 * buffered all generated text, then collapsed reasoning into one terminal
 * content delta. This captured Qwen grammar proves each safe field is emitted
 * before EOS and no native call markup leaks to the client.
 */
TEST_F(Test__ChatCompletionHandler, StreamingQwenToolsPreserveLiveReasoningAndContent)
{
    auto template_ = makeThinkingTemplate();
    ASSERT_TRUE(template_->isThinkingModel());
    ON_CALL(*tokenizer_, hasChatTemplate()).WillByDefault(Return(true));
    ON_CALL(*tokenizer_, getChatTemplate())
        .WillByDefault(::testing::ReturnRef(*template_));
    ON_CALL(*tokenizer_, encodeChat(_, _, _))
        .WillByDefault(Return(std::vector<int>{1}));
    ON_CALL(*runner_, prefill(_)).WillByDefault(Return(true));
    ON_CALL(*runner_, getToolCallFormat())
        .WillByDefault(Return(ToolCallFormat::QWEN_3_XML));
    ON_CALL(*tokenizer_, is_stop_token(_)).WillByDefault(Return(false));

    const std::vector<std::string> pieces = {
        "private thought",
        "</think>\n\n",
        "I will check. ",
        "<tool_",
        "call><function=get_weather><parameter=city>Paris</parameter>"
        "</function></tool_call>"};
    ON_CALL(*tokenizer_, decode_token(_))
        .WillByDefault(Invoke([&](int token)
        {
            const int index = token - 100;
            return index >= 0 && index < static_cast<int>(pieces.size())
                ? pieces[static_cast<size_t>(index)]
                : std::string{};
        }));

    int decode_calls = 0;
    EXPECT_CALL(*runner_, decodeStep())
        .Times(static_cast<int>(pieces.size()) + 1)
        .WillRepeatedly(Invoke([&]
        {
            if (decode_calls < static_cast<int>(pieces.size()))
                return makeToken(100 + decode_calls++);
            ++decode_calls;
            return makeToken(0, true);
        }));

    ChatCompletionRequest request;
    request.messages = {ChatMessage("user", "weather")};
    request.max_tokens = 32;
    request.stream = true;
    request.enable_thinking = true;
    request.tools = json::parse(R"([{"type":"function","function":{"name":"get_weather",
        "parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}])");

    std::string reasoning;
    std::string content;
    std::string finish;
    int reasoning_observation = 0;
    int content_observation = 0;
    int call_observation = 0;
    std::string tool_name;
    const auto response = makeHandler()->handleStreamingRequest(
        request,
        [&](const std::string &line)
        {
            if (!line.starts_with("data: ") ||
                line.starts_with("data: [DONE]"))
                return true;
            const auto choice = json::parse(line.substr(6))["choices"][0];
            const auto &delta = choice["delta"];
            if (delta.contains("reasoning_content"))
            {
                reasoning += delta["reasoning_content"].get<std::string>();
                if (reasoning_observation == 0)
                    reasoning_observation = decode_calls;
            }
            if (delta.contains("content"))
            {
                content += delta["content"].get<std::string>();
                if (content_observation == 0)
                    content_observation = decode_calls;
            }
            if (delta.contains("tool_calls"))
            {
                call_observation = decode_calls;
                tool_name = delta["tool_calls"][0]["function"]["name"];
            }
            if (!choice["finish_reason"].is_null())
                finish = choice["finish_reason"].get<std::string>();
            return true;
        });

    ASSERT_TRUE(response.ok);
    EXPECT_EQ(reasoning, "private thought");
    EXPECT_EQ(content, "I will check. ");
    EXPECT_EQ(tool_name, "get_weather");
    EXPECT_EQ(finish, "tool_calls");
    EXPECT_EQ(reasoning_observation, 1);
    EXPECT_EQ(content_observation, 3);
    EXPECT_EQ(call_observation, 5);
    EXPECT_EQ(content.find("<tool_call>"), std::string::npos);
    EXPECT_EQ(reasoning.find("I will check"), std::string::npos);
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

/** @brief Disconnected text publication cannot erase tokens already committed by a native batch. */
TEST_F(Test__ChatCompletionHandler, RuntimeStatsDisconnectKeepsCommittedBatch)
{
    auto handler = makeHandler();
    ON_CALL(*tokenizer_, encodeChat(_, _, _)).WillByDefault(Return(std::vector<int>{1, 2}));
    ON_CALL(*runner_, prefill(_)).WillByDefault(Return(true));
    ON_CALL(*tokenizer_, is_stop_token(_)).WillByDefault(Return(false));
    ON_CALL(*tokenizer_, decode_token(_)).WillByDefault(Return("A"));
    EXPECT_CALL(*runner_, decodeStep()).WillOnce(Return(makeTokens({10, 11, 12, 13})));
    ChatCompletionRequest request;
    request.messages = {ChatMessage("user", "test")};
    request.stream = true;
    request.max_tokens = 4;
    request.enable_thinking = false;
    int publications = 0;
    EXPECT_TRUE(handler->handleStreamingRequest(request, [&](const std::string &) {
        return ++publications == 1; // Accept role only, disconnect on the first text delta.
    }).ok);
    const auto stats = handler->runtimeStats();
    EXPECT_EQ(stats["requests"]["disconnected"], 1);
    EXPECT_EQ(stats["requests"]["completed"], 0);
    EXPECT_EQ(stats["tokens"]["completion"], 4);
    EXPECT_EQ(stats["last_request"]["completion_tokens"], 4);
    EXPECT_EQ(stats["last_request"]["outcome"], "disconnected");
    EXPECT_FALSE(stats["last_request"]["timings"]["ttft_seconds"].is_null());
    EXPECT_TRUE(stats["last_request"]["timings"]["first_output_seconds"].is_null());
}

/** @brief Role metadata and terminal-only output are not the first model text token. */
TEST_F(Test__ChatCompletionHandler, RuntimeStatsTTFTExcludesRoleAndTerminalFrames)
{
    using Clock = std::chrono::steady_clock;
    ON_CALL(*tokenizer_, encodeChat(_, _, _)).WillByDefault(Return(std::vector<int>{1, 2}));
    ON_CALL(*runner_, prefill(_)).WillByDefault(Return(true));
    ON_CALL(*tokenizer_, is_stop_token(_)).WillByDefault(Return(false));
    ON_CALL(*tokenizer_, decode_token(_)).WillByDefault(Return("A"));
    auto handler = makeHandler();
    ChatCompletionRequest request;
    request.messages = {ChatMessage("user", "test")};
    request.stream = true;
    request.max_tokens = 1;
    request.enable_thinking = false;
    for (const bool terminal_only : {false, true})
    {
        const HttpRequestArrival arrival;
        Clock::time_point decode_return;
        EXPECT_CALL(*runner_, decodeStep()).WillOnce(Invoke([&] {
            decode_return = Clock::now();
            return makeToken(10, terminal_only);
        }));
        const auto response = handler->handleStreamingRequest(request,
            [](const std::string &) { return true; }, arrival);
        ASSERT_TRUE(response.ok);
        const auto last = handler->runtimeStats()["last_request"];
        if (terminal_only)
        {
            EXPECT_TRUE(last["timings"]["ttft_seconds"].is_null());
            EXPECT_TRUE(last["timings"]["first_output_seconds"].is_null());
        }
        else
            EXPECT_GE(last["timings"]["ttft_seconds"].get<double>(),
                std::chrono::duration<double>(decode_return - arrival.received).count());
    }
}

/** @brief Last-request publication waits for cache cleanup, preserving accurate service latency. */
TEST_F(Test__ChatCompletionHandler, RuntimeStatsFreshnessIncludesCleanupRetirement)
{
    auto handler = makeHandler();
    ON_CALL(*tokenizer_, encodeChat(_, _, _)).WillByDefault(Return(std::vector<int>{1, 2}));
    ON_CALL(*runner_, prefill(_)).WillByDefault(Return(true));
    ON_CALL(*tokenizer_, is_stop_token(_)).WillByDefault(Return(false));
    ON_CALL(*tokenizer_, decode_token(_)).WillByDefault(Return("A"));
    EXPECT_CALL(*runner_, decodeStep()).WillOnce(Return(makeToken(10)));
    std::promise<void> entered, release;
    const auto released = release.get_future().share();
    EXPECT_CALL(*runner_, clearCache()).WillOnce(Return()).WillOnce(Invoke([&] {
        entered.set_value();
        released.wait();
    }));
    ChatCompletionRequest request;
    request.messages = {ChatMessage("user", "test")};
    request.max_tokens = 1;
    request.enable_thinking = false;
    auto response = std::async(std::launch::async, [&] { return handler->handleRequest(request); });
    // Always release before std::future joins, including after failed assertions.
    struct ReleaseOnExit
    {
        std::promise<void> &release;
        ~ReleaseOnExit() { release.set_value(); }
    };
    {
        ReleaseOnExit retire{release};
        ASSERT_EQ(entered.get_future().wait_for(std::chrono::seconds(3)), std::future_status::ready);
        const auto pending = handler->runtimeStats();
        EXPECT_EQ(pending["requests"]["active"], 1);
        EXPECT_EQ(pending["requests"]["completed"], 0);
        EXPECT_TRUE(pending["last_request"].is_null());
    }
    const auto result = response.get();
    ASSERT_TRUE(result.ok);
    const auto usage = json::parse(result.json_body)["usage"];
    const auto complete = handler->runtimeStats();
    EXPECT_EQ(complete["requests"]["active"], 0);
    EXPECT_EQ(complete["last_request"]["prompt_tokens"], usage["prompt_tokens"]);
    EXPECT_EQ(complete["last_request"]["completion_tokens"], usage["completion_tokens"]);
    EXPECT_EQ(complete["tokens"]["completion"], 1);
    EXPECT_EQ(complete["timings"]["latency"]["samples"], 1);
}
