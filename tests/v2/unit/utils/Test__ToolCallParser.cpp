/**
 * @file Test__ToolCallParser.cpp
 * @brief Unit tests for architecture-selected tool-call protocols
 *
 * These tests use model-native text captured from production requests.  They
 * protect the boundary that turns inert generated text into executable OpenAI
 * tool calls, so malformed input must always fail closed as ordinary content.
 */

#include "models/qwen35/Qwen35Schema.h"
#include "models/qwen35moe/Qwen35MoESchema.h"
#include "utils/ToolCallParser.h"
#include "utils/ToolCallTypes.h"
#include <gtest/gtest.h>

using namespace llaminar2;

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

    const auto result = parseToolCalls(text, dense_factory.getToolCallFormat());
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

    const auto result = parseToolCalls(text, ToolCallFormat::QWEN_3_XML);
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

    const auto result = parseToolCalls(text, ToolCallFormat::QWEN_3_XML);
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

    const auto result = parseToolCalls(text, ToolCallFormat::QWEN_3_XML);
    EXPECT_FALSE(result.hasToolCalls());
    EXPECT_EQ(result.content, text);
}

/** Tokenizer boundaries cannot delay safe prose or expose protocol markup. */
TEST(Test__ToolCallParser, StreamingQwen3NativePublishesContentAndCallIncrementally)
{
    StreamingToolCallSplitter splitter(ToolCallFormat::QWEN_3_XML);

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
    StreamingToolCallSplitter splitter(ToolCallFormat::QWEN_3_XML);
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
