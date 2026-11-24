/**
 * @file Test__DeepSeekV4ChatEncoding.cpp
 * @brief DeepSeek-V4 chat protocol: prompt bytes, request mapping and DSML tool-call output.
 *
 * The production comparison target is vLLM's DeepSeek-V4 tokenizer mode. The
 * golden prompts were generated from the production vLLM encoder by
 * scripts/deepseek4/make_chat_encoding_golden.py; every case goes through the
 * real HTTP request parser and the native encoder, so request mapping
 * (reasoning_effort, thinking aliases, reasoning replay, text parts, tool key
 * order) is covered together with the rendering. DSML parsing follows
 * vllm/parser/deepseek_v4.py; streaming must equal batch parsing for every
 * byte-level split.
 */
#include "app/modes/ChatCompletionHandler.h"
#include "models/deepseek4/DeepSeekV4Schema.h"
#include "utils/ChatTemplate.h"
#include "utils/DeepSeekV4ChatEncoding.h"
#include "utils/PythonJson.h"
#include "utils/ToolCallParser.h"

#include "DeepSeekV4ChatGolden.generated.h"

#include <gtest/gtest.h>
#include <limits>
#include <nlohmann/json.hpp>

using namespace llaminar2;
using ojson = nlohmann::ordered_json;

namespace
{
/** @brief Mirror the handler: parse the HTTP body, then render with its controls. */
std::string renderThroughHandler(const ojson &request_body, ChatCompletionResponse &error)
{
    const auto request = ChatCompletionHandler::parseRequest(request_body.dump(), error);
    if (!request) return {};
    ChatEncodingOptions options;
    // thinkingEnabled(): "none" disables thinking; DeepSeek-V4 defaults it on.
    options.enable_thinking = request->reasoning_effort && *request->reasoning_effort == "none"
                                  ? false : request->enable_thinking.value_or(true);
    options.reasoning_effort = request->reasoning_effort;
    if (request->tools_ordered.is_array() && !request->tools_ordered.empty())
        options.tools_json = request->tools_ordered.dump();
    return ChatTemplate::create(ChatTemplateType::DEEPSEEK_V4)->apply(request->messages, options);
}

const std::string kDsml = "｜DSML｜";

std::string invoke(const std::string &name, const std::string &params)
{
    return "<" + kDsml + "invoke name=\"" + name + "\">\n" + params + "</" + kDsml + "invoke>\n";
}

std::string param(const std::string &name, const std::string &value, const char *is_string)
{
    std::string head = "<" + kDsml + "parameter name=\"" + name + "\"";
    if (is_string) head += std::string(" string=\"") + is_string + "\"";
    return head + ">" + value + "</" + kDsml + "parameter>\n";
}

std::string section(const std::string &invokes)
{
    return "\n\n<" + kDsml + "tool_calls>\n" + invokes + "</" + kDsml + "tool_calls>";
}

ToolCallParseResult parseDsml(const std::string &text, const nlohmann::json &tools = nlohmann::json::array())
{
    return parseToolCalls(text, ToolCallFormat::DEEPSEEK_V4_DSML, tools);
}
} // namespace

/** @test Every production golden prompt is reproduced byte for byte through the HTTP request parser. */
TEST(Test__DeepSeekV4ChatEncoding, GoldenPromptsMatchProductionEncoder)
{
    const auto golden = ojson::parse(test::dsv4::kChatEncodingGolden);
    ASSERT_GE(golden["cases"].size(), 20u);
    for (const auto &item : golden["cases"])
    {
        SCOPED_TRACE(item["name"].get<std::string>());
        ChatCompletionResponse error;
        const std::string prompt = renderThroughHandler(item["request"], error);
        ASSERT_FALSE(prompt.empty()) << error.json_body;
        EXPECT_EQ(prompt, item["prompt"].get<std::string>());
    }
}

/** @test vLLM's reasoning-effort aliases and thinking defaults. */
TEST(Test__DeepSeekV4ChatEncoding, RequestPolicyMatchesVllmMapping)
{
    using deepseek4::ChatReasoningEffort;
    using deepseek4::ChatThinkingMode;
    const auto policy = [](bool thinking, std::optional<std::string> effort) {
        return deepseek4::resolveChatPromptPolicy(thinking, effort);
    };
    EXPECT_EQ(policy(true, std::nullopt).effort, ChatReasoningEffort::High);
    EXPECT_EQ(policy(true, "none").thinking, ChatThinkingMode::Chat);
    for (const char *low : {"low", "minimal", "medium"})
        EXPECT_EQ(policy(true, std::string(low)).effort, ChatReasoningEffort::Low) << low;
    for (const char *high : {"high", "xhigh"})
        EXPECT_EQ(policy(true, std::string(high)).effort, ChatReasoningEffort::High) << high;
    EXPECT_EQ(policy(true, "max").effort, ChatReasoningEffort::Max);
    EXPECT_EQ(policy(false, "max").thinking, ChatThinkingMode::Chat);
}

/** @test Python json.dumps spelling: separators, float repr, escapes, raw UTF-8. */
TEST(Test__DeepSeekV4ChatEncoding, PythonJsonDumpsSpelling)
{
    const std::pair<double, const char *> floats[] = {
        {0.5, "0.5"}, {100.0, "100.0"}, {1e-07, "1e-07"}, {1e-05, "1e-05"}, {0.0001, "0.0001"},
        {1e16, "1e+16"}, {1.5e16, "1.5e+16"}, {1234567890123456.0, "1234567890123456.0"},
        {9007199254740992.0, "9007199254740992.0"}, {-2.5, "-2.5"}, {0.1, "0.1"},
        {1.0 / 3.0, "0.3333333333333333"}, {123.456, "123.456"}, {1e22, "1e+22"},
        {5e-324, "5e-324"}, {-0.0, "-0.0"}, {0.0, "0.0"}};
    for (const auto &[value, expected] : floats)
        EXPECT_EQ(pythonFloatRepr(value), expected) << expected;
    EXPECT_THROW(pythonFloatRepr(std::numeric_limits<double>::infinity()), std::invalid_argument);

    const auto value = ojson::parse(R"({"z": [1, "x\n\"\\\u0001é", true, null], "a": {}, "n": -3, "f": 2.0})");
    EXPECT_EQ(pythonJsonDumps(value), "{\"z\": [1, \"x\\n\\\"\\\\\\u0001é\", true, null], \"a\": {}, \"n\": -3, \"f\": 2.0}");
}

/** @test Tool schemas keep the client's key order in both the request and the prompt. */
TEST(Test__DeepSeekV4ChatEncoding, ToolSchemaKeyOrderSurvivesRequestParsing)
{
    const std::string body = R"({"messages": [{"role": "user", "content": "x"}], "tools": [{"type": "function",
        "function": {"name": "f", "parameters": {"type": "object", "properties": {"zeta": {"type": "string"},
        "alpha": {"type": "integer"}}}}}]})";
    ChatCompletionResponse error;
    const auto request = ChatCompletionHandler::parseRequest(body, error);
    ASSERT_TRUE(request) << error.json_body;
    const std::string dumped = request->tools_ordered.dump();
    EXPECT_LT(dumped.find("zeta"), dumped.find("alpha"));
    const std::string prompt = renderThroughHandler(ojson::parse(body), error);
    EXPECT_NE(prompt.find("\"properties\": {\"zeta\": {\"type\": \"string\"}, \"alpha\""), std::string::npos) << prompt;
}

/** @test Image parts keep their slot and render as the placeholder exactly as vLLM's encoder flattens them. */
TEST(Test__DeepSeekV4ChatEncoding, ImagePartsRenderAsPlaceholdersInOrder)
{
    ChatCompletionResponse error;
    const auto request = ChatCompletionHandler::parseRequest(
        R"({"messages": [{"role": "user", "content": [{"type": "text", "text": "What is this?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            {"type": "text", "text": "Be brief."}]}], "chat_template_kwargs": {"thinking": false}})", error);
    ASSERT_TRUE(request) << error.json_body;
    ASSERT_EQ(request->image_parts.size(), 1u);
    EXPECT_EQ(request->image_parts[0].message, 0u);
    EXPECT_EQ(request->image_parts[0].part, 1u);
    EXPECT_EQ(request->image_parts[0].url, "data:image/png;base64,AAAA");
    const auto rendered = ChatCompletionHandler::withImagePlaceholders(*request, "<｜deepseek_image｜>");
    ChatEncodingOptions options;
    options.enable_thinking = false;
    // deepseek_v4_encoding.encode_messages(..., thinking_mode="chat") of the same request.
    EXPECT_EQ(ChatTemplate::create(ChatTemplateType::DEEPSEEK_V4)->apply(rendered.messages, options),
              "<｜begin▁of▁sentence｜><｜User｜>What is this?\n\n<｜deepseek_image｜>\n\nBe brief.<｜Assistant｜></think>");
}

/** @test Request admission: images, unknown efforts and conflicting thinking controls are 400s. */
TEST(Test__DeepSeekV4ChatEncoding, RequestAdmissionRejectsUnsupportedOrConflictingControls)
{
    const auto reject = [](const std::string &body) {
        ChatCompletionResponse error;
        EXPECT_FALSE(ChatCompletionHandler::parseRequest(body, error)) << body;
        EXPECT_EQ(error.http_status, 400) << body;
    };
    reject(R"({"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]})");
    reject(R"({"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://a/b.png"}}]}]})");
    reject(R"({"messages": [{"role": "system", "content": [{"type": "image_url", "image_url": {"url": "data:,"}}]}]})");
    reject(R"({"messages": [{"role": "user", "content": [{"type": "image", "image": "data:,"}]}]})");
    reject(R"({"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {}}]}]})");
    reject(R"({"messages": [{"role": "user", "content": [{"type": "text"}]}]})");
    reject(R"({"messages": [{"role": "user", "content": 7}]})");
    reject(R"({"messages": [{"role": "user", "content": "x"}], "reasoning_effort": "ultra"})");
    reject(R"({"messages": [{"role": "user", "content": "x"}], "reasoning_effort": "low",
               "chat_template_kwargs": {"reasoning_effort": "max"}})");
    reject(R"({"messages": [{"role": "user", "content": "x"}],
               "chat_template_kwargs": {"thinking": true, "enable_thinking": false}})");
    reject(R"({"messages": [{"role": "assistant", "content": "x", "reasoning": 3}]})");

    ChatCompletionResponse error;
    const auto request = ChatCompletionHandler::parseRequest(
        R"({"messages": [{"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]},
            {"role": "assistant", "content": "c", "reasoning": "new", "reasoning_content": "old"}],
            "chat_template_kwargs": {"thinking": false}})", error);
    ASSERT_TRUE(request) << error.json_body;
    EXPECT_EQ(request->messages[0].content, "a\nb");
    EXPECT_EQ(request->messages[0].content_parts, (std::vector<std::string>{"a", "b"}));
    EXPECT_EQ(request->messages[1].reasoning_content, std::optional<std::string>("new"));
    EXPECT_EQ(request->enable_thinking, std::optional<bool>(false));
}

/** @test The DeepSeek-V4 schema selects DSML and a delimiter-terminated budget close. */
TEST(Test__DeepSeekV4ChatEncoding, SchemaSelectsDsmlToolFormat)
{
    DeepSeekV4SchemaFactory factory;
    EXPECT_EQ(factory.getToolCallFormat(), ToolCallFormat::DEEPSEEK_V4_DSML);
    const auto close = factory.getStopThinkingPrompt();
    EXPECT_EQ(close.substr(close.size() - 8), "</think>");
}

/** @test Content, typed parameters and parallel invokes; the protocol separator is not content. */
TEST(Test__DeepSeekV4ChatEncoding, DsmlParsesProductionToolCalls)
{
    const std::string text = "Let me check." +
        section(invoke("get_weather", param("city", "杭州", "true") + param("days", "5", "false") +
                                          param("tags", "[\"a\", 1.0]", "false")) +
                invoke("search", param("query", "line1\nline2 \"q\"", "true"))) +
        "\ntrailing text after the turn";
    const auto result = parseDsml(text);
    EXPECT_EQ(result.content, "Let me check.");
    ASSERT_EQ(result.tool_calls.size(), 2u);
    EXPECT_EQ(result.tool_calls[0].name, "get_weather");
    EXPECT_EQ(result.tool_calls[0].arguments, "{\"city\": \"杭州\", \"days\": 5, \"tags\": [\"a\", 1.0]}");
    EXPECT_EQ(result.tool_calls[1].name, "search");
    EXPECT_EQ(result.tool_calls[1].arguments, "{\"query\": \"line1\\nline2 \\\"q\\\"\"}");
    EXPECT_NE(result.tool_calls[0].id, result.tool_calls[1].id);
}

/** @test vLLM's tolerances: missing string attribute, unclosed parameter, misspelled opener, bare invoke. */
TEST(Test__DeepSeekV4ChatEncoding, DsmlToleratesProductionVariants)
{
    auto result = parseDsml(section(invoke("f", param("list", "[1, 2]", nullptr) + param("word", "not json", nullptr))));
    ASSERT_EQ(result.tool_calls.size(), 1u);
    EXPECT_EQ(result.tool_calls[0].arguments, "{\"list\": [1, 2], \"word\": \"not json\"}");

    const std::string unclosed = "<" + kDsml + "parameter name=\"a\" string=\"true\">one\n" +
                                 "<" + kDsml + "parameter name=\"b\" string=\"false\">2</" + kDsml + "parameter>\n";
    result = parseDsml(section(invoke("f", unclosed)));
    ASSERT_EQ(result.tool_calls.size(), 1u);
    EXPECT_EQ(result.tool_calls[0].arguments, "{\"a\": \"one\\n\", \"b\": 2}");

    result = parseDsml("x\n\n<" + kDsml + "toolcalls>\n" + invoke("g", param("k", "v", "true")) + "</" + kDsml + "tool_calls>");
    ASSERT_EQ(result.tool_calls.size(), 1u);
    EXPECT_EQ(result.content, "x");

    result = parseDsml("y" + invoke("h", ""));
    ASSERT_EQ(result.tool_calls.size(), 1u);
    EXPECT_EQ(result.content, "y");
    EXPECT_EQ(result.tool_calls[0].arguments, "{}");
}

/** @test A spurious {"arguments": ...} wrapper is removed only when the schema proves it spurious. */
TEST(Test__DeepSeekV4ChatEncoding, DsmlUnwrapsSpuriousArgumentWrapper)
{
    const auto tools = nlohmann::json::parse(R"([{"type": "function", "function": {"name": "f",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}])");
    auto result = parseDsml(section(invoke("f", param("arguments", "{\"city\": \"x\"}", "true"))), tools);
    ASSERT_EQ(result.tool_calls.size(), 1u);
    EXPECT_EQ(result.tool_calls[0].arguments, "{\"city\": \"x\"}");
    result = parseDsml(section(invoke("f", param("arguments", "{\"other\": 1}", "true"))), tools);
    EXPECT_EQ(result.tool_calls[0].arguments, "{\"arguments\": \"{\\\"other\\\": 1}\"}");
}

/** @test Fail closed: a section without a complete invoke stays content; an incomplete tail follows the calls. */
TEST(Test__DeepSeekV4ChatEncoding, DsmlIncompleteSyntaxFailsClosed)
{
    const std::string truncated = "a\n\n<" + kDsml + "tool_calls>\n<" + kDsml + "invoke name=\"f\">\n" +
                                  param("k", "v", "true");
    auto result = parseDsml(truncated);
    EXPECT_TRUE(result.tool_calls.empty());
    EXPECT_EQ(result.content, truncated);

    const std::string tail = "<" + kDsml + "invoke name=\"g\">\n";
    result = parseDsml("\n\n<" + kDsml + "tool_calls>\n" + invoke("f", "") + tail);
    ASSERT_EQ(result.tool_calls.size(), 1u);
    EXPECT_EQ(result.content, tail);
}

/** @test Streaming emits the same content and calls as batch parsing for byte-by-byte delivery. */
TEST(Test__DeepSeekV4ChatEncoding, DsmlStreamingEqualsBatchForEverySplit)
{
    const std::string text = "Answer: 1 < 2." +
        section(invoke("get_weather", param("city", "Grüße", "true") + param("n", "3", "false")) +
                invoke("search", param("q", "x", "true")));
    const auto batch = parseDsml(text);
    ASSERT_EQ(batch.tool_calls.size(), 2u);
    for (std::size_t chunk : {std::size_t{1}, std::size_t{2}, std::size_t{3}, std::size_t{7}, std::size_t{64}})
    {
        SCOPED_TRACE(chunk);
        StreamingToolCallSplitter splitter(ToolCallFormat::DEEPSEEK_V4_DSML);
        std::string content;
        std::vector<ToolCall> calls;
        const auto fold = [&](std::vector<StreamingToolCallSplitter::Event> events) {
            for (auto &event : events)
                if (event.kind == StreamingToolCallSplitter::Event::Kind::ToolCall) calls.push_back(event.tool_call);
                else content += event.content;
        };
        for (std::size_t i = 0; i < text.size(); i += chunk) fold(splitter.process(text.substr(i, chunk)));
        fold(splitter.flush());
        EXPECT_EQ(content, batch.content);
        ASSERT_EQ(calls.size(), batch.tool_calls.size());
        for (std::size_t i = 0; i < calls.size(); ++i)
        {
            EXPECT_EQ(calls[i].name, batch.tool_calls[i].name);
            EXPECT_EQ(calls[i].arguments, batch.tool_calls[i].arguments);
        }
    }
}

/** @test Wrongly typed fields and integers beyond 64 bits are 400s, never a 500 or a rounded value. */
TEST(Test__DeepSeekV4ChatEncoding, MalformedTypesAndHugeIntegersAreRejected)
{
    const auto reject = [](const std::string &body) {
        ChatCompletionResponse error;
        EXPECT_FALSE(ChatCompletionHandler::parseRequest(body, error)) << body;
        EXPECT_EQ(error.http_status, 400) << body;
    };
    reject(R"({"messages": [{"role": "user", "content": [{"type": 7, "text": "x"}]}]})");
    reject(R"({"messages": [{"role": "user", "content": "x"}], "tools": [{"type": "function",
        "function": {"name": "f", "parameters": {"type": "object", "properties": {"id": {"maximum": 18446744073709551617}}}}}]})");
    EXPECT_TRUE(hasOutOfRangeInteger("[18446744073709551616]"));
    EXPECT_TRUE(hasOutOfRangeInteger("{\"a\": -9223372036854775809}"));
    EXPECT_FALSE(hasOutOfRangeInteger("[18446744073709551615, -9223372036854775808, 1e30, 2.5]"));

    // A numeric tool-call name parses as a request but cannot be rendered: invalid_argument, i.e. a 400.
    ChatCompletionResponse error;
    const auto request = ChatCompletionHandler::parseRequest(
        R"({"messages": [{"role": "user", "content": "x"}, {"role": "assistant", "content": null,
            "tool_calls": [{"id": "c", "type": "function", "function": {"name": 7, "arguments": "{}"}}]}]})", error);
    ASSERT_TRUE(request) << error.json_body;
    EXPECT_THROW(ChatTemplate::create(ChatTemplateType::DEEPSEEK_V4)->apply(request->messages, ChatEncodingOptions{}),
                 std::invalid_argument);

    // OpenAI arguments are a JSON string: its inner numbers get the same exactness rule.
    const auto string_args = ChatCompletionHandler::parseRequest(
        R"({"messages": [{"role": "user", "content": "x"}, {"role": "assistant", "content": null,
            "tool_calls": [{"id": "c", "type": "function", "function": {"name": "f",
            "arguments": "{\"id\": 18446744073709551617}"}}]}]})", error);
    ASSERT_TRUE(string_args) << error.json_body;
    EXPECT_THROW(ChatTemplate::create(ChatTemplateType::DEEPSEEK_V4)->apply(string_args->messages, ChatEncodingOptions{}),
                 std::invalid_argument);

    // Model output: an invoke with an inexact typed value, directly or inside a
    // wrapper string, is never executed with a changed number.
    const std::string huge = section(invoke("f", param("id", "18446744073709551617", "false")));
    auto result = parseDsml("Lead." + huge);
    EXPECT_TRUE(result.tool_calls.empty());
    EXPECT_EQ(result.content, "Lead." + huge); // exact original text, published once
    // Mixed section: the valid call executes; the rejected invoke is literal once.
    const std::string bad = invoke("g", param("id", "18446744073709551617", "false"));
    result = parseDsml(section(invoke("f", param("id", "7", "false")) + bad));
    ASSERT_EQ(result.tool_calls.size(), 1u);
    EXPECT_EQ(result.tool_calls[0].arguments, "{\"id\": 7}");
    EXPECT_EQ(result.content, bad.substr(0, bad.size() - 1)); // the invoke, through its close tag
    // Streaming publishes the same literal content for every chunking.
    for (std::size_t chunk : {std::size_t{1}, std::size_t{5}, std::size_t{64}})
    {
        StreamingToolCallSplitter splitter(ToolCallFormat::DEEPSEEK_V4_DSML);
        std::string content;
        std::size_t calls = 0;
        const std::string text = "Lead." + huge;
        const auto fold = [&](std::vector<StreamingToolCallSplitter::Event> events) {
            for (auto &event : events)
                if (event.kind == StreamingToolCallSplitter::Event::Kind::ToolCall) ++calls;
                else content += event.content;
        };
        for (std::size_t i = 0; i < text.size(); i += chunk) fold(splitter.process(text.substr(i, chunk)));
        fold(splitter.flush());
        EXPECT_EQ(calls, 0u);
        EXPECT_EQ(content, text) << chunk;
    }
    const auto tools = nlohmann::json::parse(R"([{"type": "function", "function": {"name": "f",
        "parameters": {"type": "object", "properties": {"id": {"type": "integer"}}}}}])");
    result = parseDsml(section(invoke("f", param("arguments", "{\"id\": 18446744073709551617}", "true"))), tools);
    ASSERT_EQ(result.tool_calls.size(), 1u);
    EXPECT_EQ(result.tool_calls[0].arguments, "{\"arguments\": \"{\\\"id\\\": 18446744073709551617}\"}");
    // Other formats share the same lossless authority.
    EXPECT_TRUE(parseToolCalls("<tool_call>\n{\"name\": \"f\", \"arguments\": {\"id\": 18446744073709551617}}\n</tool_call>",
                               ToolCallFormat::HERMES_2_PRO).tool_calls.empty());
}
