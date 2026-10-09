/**
 * @file Test__ToolCallParser.cpp
 * @brief Unit tests for architecture-selected tool-call protocols
 *
 * These tests use model-native text captured from production requests.  They
 * protect the boundary that turns inert generated text into executable OpenAI
 * tool calls, so malformed input must always fail closed as ordinary content.
 * Literal protocol tags inside file strings retain their byte boundaries;
 * native framing and the admitted schema distinguish executable arguments.
 */

#include "models/qwen35/Qwen35Schema.h"
#include "models/qwen35moe/Qwen35MoESchema.h"
#include "utils/ToolCallParser.h"
#include "utils/ToolCallTypes.h"
#include <gtest/gtest.h>

using namespace llaminar2;

namespace
{
    /** @return Function schemas, including OpenCode's write string argument contract. */
    const nlohmann::json &qwenTestTools()
    {
        static const auto tools = nlohmann::json::parse(R"([
          {"type":"function","function":{"name":"write","parameters":{"type":"object","properties":{"content":{"type":"string"},"filePath":{"type":"string"}}}}},
          {"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}},
          {"type":"function","function":{"name":"get_time","parameters":{"type":"object","properties":{"timezone":{"type":"string"}}}}},
          {"type":"function","function":{"name":"search_records","parameters":{"type":"object","properties":{"query":{"type":"string"},"limit":{"type":"integer"},"filters":{"type":"object"}}}}}
        ])");
        return tools;
    }
}

/** @test OpenCode write/edit string fields must retain literal JSON-looking bytes. */
TEST(Test__ToolCallParser, QwenStringArgumentsPreserveLiteralBytes)
{
    for (const std::string value : {"123", "true", "null", "{\"count\":123}",
                                    "[1,2]", "\"quoted\"", "    first\n\tsecond  \n", "", "λ 中文 🙂",
                                    "print(\"<tool_call></tool_call><parameter=x></parameter>\")\n",
                                    R"(parts = text.split("\n"); output.write("\t" + "\n".join(parts)))",
                                    R"(pattern = r"\w+\s+\u263a"; path = "C:\\temp\\code.py")"})
    {
        SCOPED_TRACE(value);
        const std::string text = "<tool_call>\n<function=write>\n<parameter=content>\n" +
            value + "\n</parameter>\n<parameter=filePath>\n/tmp/result.txt\n</parameter>\n"
            "</function>\n</tool_call>";
        const auto result = parseToolCalls(text, ToolCallFormat::QWEN_3_XML, qwenTestTools());
        ASSERT_EQ(result.tool_calls.size(), 1u);
        const auto arguments = nlohmann::json::parse(result.tool_calls[0].arguments);
        EXPECT_TRUE(arguments["content"].is_string());
        EXPECT_EQ(arguments["content"], value);
    }
}

/** @test Every SSE split point retains the exact same schema-owned string bytes. */
TEST(Test__ToolCallParser, StreamingQwenStringArgumentsPreserveEveryChunkBoundary)
{
    for (const std::string value : {"123", "true", "null", "{\"count\":123}",
                                    "[1,2]", "\"quoted\"", "    first\n\tsecond  \n", "", "λ 中文 🙂",
                                    "print(\"<tool_call></tool_call><parameter=x></parameter>\")\n",
                                    R"(parts = text.split("\n"); output.write("\t" + "\n".join(parts)))",
                                    R"(pattern = r"\w+\s+\u263a"; path = "C:\\temp\\code.py")"})
    {
        const std::string text = "<tool_call>\n<function=write>\n<parameter=content>\n" +
            value + "\n</parameter>\n<parameter=filePath>\n/tmp/result.txt\n</parameter>\n"
            "</function>\n</tool_call>";
        for (size_t split = 0; split <= text.size(); ++split)
        {
            SCOPED_TRACE(::testing::Message() << value << " split=" << split);
            StreamingToolCallSplitter splitter(ToolCallFormat::QWEN_3_XML, qwenTestTools());
            std::vector<StreamingToolCallSplitter::Event> events;
            for (auto fragment : {std::string_view(text).substr(0, split),
                                  std::string_view(text).substr(split)})
            {
                auto produced = splitter.process(fragment);
                events.insert(events.end(), produced.begin(), produced.end());
            }
            const auto terminal = splitter.flush();
            events.insert(events.end(), terminal.begin(), terminal.end());
            ASSERT_EQ(events.size(), 1u);
            ASSERT_EQ(events[0].kind, StreamingToolCallSplitter::Event::Kind::ToolCall);
            EXPECT_EQ(nlohmann::json::parse(events[0].tool_call.arguments)["content"], value);
        }
    }
}

/** @test Adjacent literal parameter tags cannot discard an otherwise valid write. */
TEST(Test__ToolCallParser, QwenLiteralAdjacentParametersPreserveNativeCalls)
{
    for (const std::string value : {
            "A</parameter><parameter=x>KEEP</parameter>B\n",
            "A</parameter><parameter=filePath>KEEP</parameter>B\n",
            "A</parameter><parameter=content>KEEP</parameter>B\n",
            "A\n</parameter>\n<parameter=x>\nKEEP\n</parameter>\nB\n",
            "🙂</parameter><parameter=x>👩🏽‍💻</parameter>🚀\n"})
    {
        const std::string text = "<tool_call>\n<function=write>\n<parameter=content>\n" +
            value + "\n</parameter>\n<parameter=filePath>\n/tmp/probe5.txt\n</parameter>\n"
            "</function>\n</tool_call>";
        const auto expected = nlohmann::json{{"content", value}, {"filePath", "/tmp/probe5.txt"}};
        SCOPED_TRACE(value);
        const auto parsed = parseToolCalls(text, ToolCallFormat::QWEN_3_XML, qwenTestTools());
        ASSERT_EQ(parsed.tool_calls.size(), 1u);
        EXPECT_TRUE(parsed.content.empty());
        EXPECT_EQ(nlohmann::json::parse(parsed.tool_calls[0].arguments), expected);
        for (size_t split = 0; split <= text.size(); ++split)
        {
            SCOPED_TRACE(split);
            StreamingToolCallSplitter splitter(ToolCallFormat::QWEN_3_XML, qwenTestTools());
            std::vector<StreamingToolCallSplitter::Event> events;
            for (const auto fragment : {std::string_view(text).substr(0, split),
                                        std::string_view(text).substr(split)})
            {
                auto produced = splitter.process(fragment);
                events.insert(events.end(), produced.begin(), produced.end());
            }
            const auto terminal = splitter.flush();
            events.insert(events.end(), terminal.begin(), terminal.end());
            ASSERT_EQ(events.size(), 1u);
            ASSERT_EQ(events[0].kind, StreamingToolCallSplitter::Event::Kind::ToolCall);
            EXPECT_EQ(nlohmann::json::parse(events[0].tool_call.arguments), expected);
        }
    }
}

/** @test A real OpenCode write may close a value before, rather than after, a newline. */
TEST(Test__ToolCallParser, QwenMixedParameterFramingPreservesEveryChunkBoundary)
{
    auto tools = qwenTestTools();
    tools[0]["function"]["parameters"]["required"] = {"content", "filePath"};
    for (const std::string newline : {"\n", "\r\n"})
        for (const std::string value : {"123", "true", "", "λ 中文 🙂", "    first\n\tsecond  ",
                                      "A</parameter><parameter=filePath>KEEP</parameter>B"})
            for (const bool close_on_newline : {false, true})
            for (const bool path_close_on_newline : {false, true})
            {
                // The 215K-context OpenCode app trace used an opening newline,
                // an inline content terminator, and a newline before filePath.
                // Cover both argument orders so the final parameter obeys the
                // same framing contract as an intermediate one.
                for (const bool content_first : {false, true})
                {
                    const auto parameter = [&](const std::string& name, const std::string& body,
                                               bool closing_newline) {
                        return "<parameter=" + name + ">" + newline + body +
                            (closing_newline ? newline : "") + "</parameter>" + newline;
                    };
                    const auto content = parameter("content", value, close_on_newline);
                    const auto path = parameter("filePath", "/tmp/result.txt", path_close_on_newline);
                    const std::string text = "<tool_call>" + newline + "<function=write>" + newline +
                        (content_first ? content + path : path + content) +
                        "</function>" + newline + "</tool_call>";
                    const auto expected = nlohmann::json{{"content", value}, {"filePath", "/tmp/result.txt"}};
                    SCOPED_TRACE(::testing::Message() << text);
                    const auto parsed = parseToolCalls(text, ToolCallFormat::QWEN_3_XML, tools);
                    ASSERT_EQ(parsed.tool_calls.size(), 1u);
                    EXPECT_TRUE(parsed.content.empty());
                    EXPECT_EQ(nlohmann::json::parse(parsed.tool_calls[0].arguments), expected);
                    for (size_t split = 0; split <= text.size(); ++split)
                    {
                        SCOPED_TRACE(split);
                        StreamingToolCallSplitter splitter(ToolCallFormat::QWEN_3_XML, tools);
                        std::vector<StreamingToolCallSplitter::Event> events;
                        for (const auto fragment : {std::string_view(text).substr(0, split),
                                                    std::string_view(text).substr(split)})
                        {
                            auto produced = splitter.process(fragment);
                            events.insert(events.end(), produced.begin(), produced.end());
                        }
                        const auto terminal = splitter.flush();
                        events.insert(events.end(), terminal.begin(), terminal.end());
                        ASSERT_EQ(events.size(), 1u);
                        ASSERT_EQ(events[0].kind, StreamingToolCallSplitter::Event::Kind::ToolCall);
                        EXPECT_EQ(nlohmann::json::parse(events[0].tool_call.arguments), expected);
                    }
                }
            }
}

/** @test Missing required native arguments cannot be published as executable partial calls. */
TEST(Test__ToolCallParser, QwenRequiredArgumentsFailClosed)
{
    auto tools = qwenTestTools();
    tools[0]["function"]["parameters"]["required"] = {"content", "filePath"};
    for (const std::string parameters : {"", "<parameter=content>123</parameter>",
                                         "<parameter=filePath>/tmp/result.txt</parameter>"})
    {
        const std::string text = "<tool_call><function=write>" + parameters + "</function></tool_call>";
        const auto parsed = parseToolCalls(text, ToolCallFormat::QWEN_3_XML, tools);
        EXPECT_FALSE(parsed.hasToolCalls());
        EXPECT_EQ(parsed.content, text);
        for (size_t split = 0; split <= text.size(); ++split)
        {
            StreamingToolCallSplitter splitter(ToolCallFormat::QWEN_3_XML, tools);
            std::string content;
            for (const auto fragment : {std::string_view(text).substr(0, split),
                                        std::string_view(text).substr(split)})
                for (const auto& event : splitter.process(fragment))
                {
                    EXPECT_EQ(event.kind, StreamingToolCallSplitter::Event::Kind::Content);
                    content += event.content;
                }
            for (const auto& event : splitter.flush())
            {
                EXPECT_EQ(event.kind, StreamingToolCallSplitter::Event::Kind::Content);
                content += event.content;
            }
            EXPECT_EQ(content, text);
        }
    }
}

/** @test Absent schemas and invalid typed JSON cannot manufacture native calls. */
TEST(Test__ToolCallParser, QwenSchemaAdmissionAndMalformedTypedValuesFailClosed)
{
    const std::string unknown = "<tool_call><function=write><parameter=content>123"
        "</parameter></function></tool_call>";
    EXPECT_EQ(parseToolCalls(unknown, ToolCallFormat::QWEN_3_XML).content, unknown);
    const std::string invalid = "<tool_call><function=search_records><parameter=filters>"
        "{not JSON}</parameter></function></tool_call>";
    const auto parsed = parseToolCalls(invalid, ToolCallFormat::QWEN_3_XML, qwenTestTools());
    EXPECT_FALSE(parsed.hasToolCalls());
    EXPECT_EQ(parsed.content, invalid);
    for (const std::string arguments : {
            "<parameter=unknown>123</parameter>",
            "<parameter=content>123</parameter><parameter=content>456</parameter>"})
    {
        const std::string text = "<tool_call><function=write>" + arguments + "</function></tool_call>";
        const auto result = parseToolCalls(text, ToolCallFormat::QWEN_3_XML, qwenTestTools());
        EXPECT_FALSE(result.hasToolCalls());
        EXPECT_EQ(result.content, text);
    }
}

// =============================================================================
// Hermes 2 Pro format
// =============================================================================

TEST(Test__ToolCallParser, Hermes2Pro_SingleToolCall)
{
    std::string text = R"(<tool_call>
{"name": "get_weather", "arguments": {"location": "Paris"}}
</tool_call>)";

    auto result = parseToolCalls(text, ToolCallFormat::HERMES_2_PRO);
    ASSERT_TRUE(result.hasToolCalls());
    ASSERT_EQ(result.tool_calls.size(), 1u);
    EXPECT_EQ(result.tool_calls[0].name, "get_weather");
    EXPECT_EQ(result.tool_calls[0].arguments, R"({"location":"Paris"})");
    EXPECT_FALSE(result.tool_calls[0].id.empty());
    EXPECT_TRUE(result.content.empty());
}

TEST(Test__ToolCallParser, Hermes2Pro_MultipleToolCalls)
{
    std::string text = R"(<tool_call>
{"name": "get_weather", "arguments": {"location": "Paris"}}
</tool_call>
<tool_call>
{"name": "get_time", "arguments": {"timezone": "UTC"}}
</tool_call>)";

    auto result = parseToolCalls(text, ToolCallFormat::HERMES_2_PRO);
    ASSERT_TRUE(result.hasToolCalls());
    ASSERT_EQ(result.tool_calls.size(), 2u);
    EXPECT_EQ(result.tool_calls[0].name, "get_weather");
    EXPECT_EQ(result.tool_calls[1].name, "get_time");
    // Each should get a unique ID
    EXPECT_NE(result.tool_calls[0].id, result.tool_calls[1].id);
}

TEST(Test__ToolCallParser, Hermes2Pro_ToolCallWithContentBefore)
{
    std::string text = R"(Let me check the weather for you.
<tool_call>
{"name": "get_weather", "arguments": {"location": "Paris"}}
</tool_call>)";

    auto result = parseToolCalls(text, ToolCallFormat::HERMES_2_PRO);
    ASSERT_TRUE(result.hasToolCalls());
    ASSERT_EQ(result.tool_calls.size(), 1u);
    EXPECT_EQ(result.tool_calls[0].name, "get_weather");
    // Content before tool call should be preserved
    EXPECT_TRUE(result.content.find("check the weather") != std::string::npos);
}

TEST(Test__ToolCallParser, Hermes2Pro_NoToolCall)
{
    std::string text = "This is a normal response without any tool calls.";

    auto result = parseToolCalls(text, ToolCallFormat::HERMES_2_PRO);
    EXPECT_FALSE(result.hasToolCalls());
    EXPECT_EQ(result.content, text);
}

TEST(Test__ToolCallParser, Hermes2Pro_MalformedJSON)
{
    std::string text = R"(<tool_call>
this is not valid json
</tool_call>)";

    auto result = parseToolCalls(text, ToolCallFormat::HERMES_2_PRO);
    // Should gracefully handle malformed JSON
    EXPECT_FALSE(result.hasToolCalls());
}

TEST(Test__ToolCallParser, Hermes2Pro_MissingArguments)
{
    std::string text = R"(<tool_call>
{"name": "no_args_func"}
</tool_call>)";

    auto result = parseToolCalls(text, ToolCallFormat::HERMES_2_PRO);
    ASSERT_TRUE(result.hasToolCalls());
    ASSERT_EQ(result.tool_calls.size(), 1u);
    EXPECT_EQ(result.tool_calls[0].name, "no_args_func");
    EXPECT_EQ(result.tool_calls[0].arguments, "{}");
}

TEST(Test__ToolCallParser, Hermes2Pro_NestedArguments)
{
    std::string text = R"(<tool_call>
{"name": "create_event", "arguments": {"title": "Meeting", "details": {"time": "3pm", "attendees": ["Alice", "Bob"]}}}
</tool_call>)";

    auto result = parseToolCalls(text, ToolCallFormat::HERMES_2_PRO);
    ASSERT_TRUE(result.hasToolCalls());
    ASSERT_EQ(result.tool_calls.size(), 1u);
    EXPECT_EQ(result.tool_calls[0].name, "create_event");
    // Arguments should be valid JSON
    auto args = nlohmann::json::parse(result.tool_calls[0].arguments);
    EXPECT_EQ(args["title"], "Meeting");
    EXPECT_TRUE(args["details"].is_object());
}

// =============================================================================
// Qwen 3.5/3.8 native function/parameter format
// =============================================================================

/** Reproduce the exact payload emitted by both production models. */
TEST(Test__ToolCallParser, Qwen3NativeProductionPayloadAndSchemaAuthority)
{
    const Qwen35SchemaFactory dense_factory;
    const Qwen35MoESchemaFactory moe_factory;
    ASSERT_EQ(dense_factory.getToolCallFormat(), ToolCallFormat::QWEN_3_XML);
    ASSERT_EQ(moe_factory.getToolCallFormat(), ToolCallFormat::QWEN_3_XML);

    const std::string text = R"(<tool_call>
<function=get_weather>
<parameter=city>
Paris
</parameter>
</function>
</tool_call>)";

    const auto result = parseToolCalls(text, dense_factory.getToolCallFormat(), qwenTestTools());
    ASSERT_TRUE(result.hasToolCalls());
    ASSERT_EQ(result.tool_calls.size(), 1u);
    EXPECT_EQ(result.tool_calls[0].name, "get_weather");
    EXPECT_EQ(nlohmann::json::parse(result.tool_calls[0].arguments),
              nlohmann::json({{"city", "Paris"}}));
    EXPECT_TRUE(result.content.empty());
}

TEST(Test__ToolCallParser, Qwen3NativePreservesTypedAndMultilineParameters)
{
    const std::string text = R"(I will use the requested tool.
<tool_call>
<function=search_records>
<parameter=query>
north
south
</parameter>
<parameter=limit>
3
</parameter>
<parameter=filters>
{"active":true,"tags":["a","b"]}
</parameter>
</function>
</tool_call>)";

    const auto result = parseToolCalls(text, ToolCallFormat::QWEN_3_XML, qwenTestTools());
    ASSERT_TRUE(result.hasToolCalls());
    ASSERT_EQ(result.tool_calls.size(), 1u);
    EXPECT_EQ(result.content, "I will use the requested tool.");
    const auto arguments = nlohmann::json::parse(result.tool_calls[0].arguments);
    EXPECT_EQ(arguments["query"], "north\nsouth");
    EXPECT_EQ(arguments["limit"], 3);
    EXPECT_EQ(arguments["filters"]["active"], true);
    EXPECT_EQ(arguments["filters"]["tags"], nlohmann::json({"a", "b"}));
}

TEST(Test__ToolCallParser, Qwen3NativeSupportsParallelCallsInGenerationOrder)
{
    const std::string text = R"(<tool_call>
<function=get_weather>
<parameter=city>
Paris
</parameter>
</function>
</tool_call>
<tool_call>
<function=get_time>
<parameter=timezone>
Europe/Paris
</parameter>
</function>
</tool_call>)";

    const auto result = parseToolCalls(text, ToolCallFormat::QWEN_3_XML, qwenTestTools());
    ASSERT_EQ(result.tool_calls.size(), 2u);
    EXPECT_EQ(result.tool_calls[0].name, "get_weather");
    EXPECT_EQ(result.tool_calls[1].name, "get_time");
    EXPECT_NE(result.tool_calls[0].id, result.tool_calls[1].id);
}

TEST(Test__ToolCallParser, Qwen3NativeMalformedBlockFailsClosedAsContent)
{
    const std::string text = R"(<tool_call>
<function=get_weather>
<parameter=city>
Paris
</function>
</tool_call>)";

    const auto result = parseToolCalls(text, ToolCallFormat::QWEN_3_XML, qwenTestTools());
    EXPECT_FALSE(result.hasToolCalls());
    EXPECT_EQ(result.content, text);
}

/** Tokenizer boundaries cannot delay safe prose or expose protocol markup. */
TEST(Test__ToolCallParser, StreamingQwen3NativePublishesContentAndCallIncrementally)
{
    StreamingToolCallSplitter splitter(ToolCallFormat::QWEN_3_XML, qwenTestTools());

    auto first = splitter.process("I will check. <tool_");
    ASSERT_EQ(first.size(), 1u);
    EXPECT_EQ(first[0].kind,
              StreamingToolCallSplitter::Event::Kind::Content);
    EXPECT_EQ(first[0].content, "I will check. ");

    auto second = splitter.process(
        "call><function=get_weather><parameter=city>Par");
    EXPECT_TRUE(second.empty());
    auto third = splitter.process(
        "is</parameter></function></tool_call>");
    ASSERT_EQ(third.size(), 1u);
    EXPECT_EQ(third[0].kind,
              StreamingToolCallSplitter::Event::Kind::ToolCall);
    EXPECT_EQ(third[0].tool_call.name, "get_weather");
    EXPECT_EQ(nlohmann::json::parse(third[0].tool_call.arguments),
              nlohmann::json({{"city", "Paris"}}));
    EXPECT_TRUE(splitter.flush().empty());
}

/** A truncated stream must remain inert literal output. */
TEST(Test__ToolCallParser, StreamingIncompleteToolCallFailsClosedOnFlush)
{
    StreamingToolCallSplitter splitter(ToolCallFormat::QWEN_3_XML, qwenTestTools());
    EXPECT_TRUE(splitter.process("<tool_call><function=delete_all>").empty());
    const auto final = splitter.flush();
    ASSERT_EQ(final.size(), 1u);
    EXPECT_EQ(final[0].kind,
              StreamingToolCallSplitter::Event::Kind::Content);
    EXPECT_EQ(final[0].content,
              "<tool_call><function=delete_all>");
}

// =============================================================================
// hasToolCallMarkers
// =============================================================================

TEST(Test__ToolCallParser, HasMarkers_Hermes2Pro)
{
    EXPECT_TRUE(hasToolCallMarkers("<tool_call>\nfoo\n</tool_call>", ToolCallFormat::HERMES_2_PRO));
    EXPECT_FALSE(hasToolCallMarkers("no markers here", ToolCallFormat::HERMES_2_PRO));
    EXPECT_FALSE(hasToolCallMarkers("<tool_call> without end", ToolCallFormat::HERMES_2_PRO));
}

TEST(Test__ToolCallParser, HasMarkers_Qwen3Native)
{
    EXPECT_TRUE(hasToolCallMarkers(
        "<tool_call><function=f></function></tool_call>",
        ToolCallFormat::QWEN_3_XML));
    EXPECT_FALSE(hasToolCallMarkers(
        "<function=f></function>",
        ToolCallFormat::QWEN_3_XML));
}

// =============================================================================
// Generic format (```json blocks)
// =============================================================================

TEST(Test__ToolCallParser, Generic_CodeBlock)
{
    std::string text = R"(Here's the function call:
```json
{"name": "search", "arguments": {"query": "hello"}}
```)";

    auto result = parseToolCalls(text, ToolCallFormat::GENERIC);
    ASSERT_TRUE(result.hasToolCalls());
    ASSERT_EQ(result.tool_calls.size(), 1u);
    EXPECT_EQ(result.tool_calls[0].name, "search");
}

TEST(Test__ToolCallParser, Generic_NoCodeBlock)
{
    std::string text = "Just a plain response.";
    auto result = parseToolCalls(text, ToolCallFormat::GENERIC);
    EXPECT_FALSE(result.hasToolCalls());
}

// =============================================================================
// Format NONE
// =============================================================================

TEST(Test__ToolCallParser, None_AlwaysReturnsNoToolCalls)
{
    std::string text = R"(<tool_call>
{"name": "get_weather", "arguments": {"location": "Paris"}}
</tool_call>)";

    auto result = parseToolCalls(text, ToolCallFormat::NONE);
    EXPECT_FALSE(result.hasToolCalls());
    EXPECT_EQ(result.content, text);
}

// =============================================================================
// ToolCall::toJson
// =============================================================================

TEST(Test__ToolCallParser, ToolCallToJson)
{
    ToolCall tc;
    tc.id = "call_abc123";
    tc.name = "get_weather";
    tc.arguments = R"({"location":"Paris"})";

    auto j = toolCallToJson(tc);
    EXPECT_EQ(j["id"], "call_abc123");
    EXPECT_EQ(j["type"], "function");
    EXPECT_EQ(j["function"]["name"], "get_weather");
    EXPECT_EQ(j["function"]["arguments"], R"({"location":"Paris"})");
}

// =============================================================================
// generateToolCallId
// =============================================================================

TEST(Test__ToolCallParser, GenerateUniqueIds)
{
    auto id1 = generateToolCallId();
    auto id2 = generateToolCallId();
    EXPECT_NE(id1, id2);
    EXPECT_TRUE(id1.substr(0, 5) == "call_");
    EXPECT_TRUE(id2.substr(0, 5) == "call_");
}
