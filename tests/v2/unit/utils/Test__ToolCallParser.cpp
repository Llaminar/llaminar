/**
 * @file Test__ToolCallParser.cpp
 * @brief Unit tests for tool call parsing (Hermes 2 Pro, Generic formats)
 */

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
// hasToolCallMarkers
// =============================================================================

TEST(Test__ToolCallParser, HasMarkers_Hermes2Pro)
{
    EXPECT_TRUE(hasToolCallMarkers("<tool_call>\nfoo\n</tool_call>", ToolCallFormat::HERMES_2_PRO));
    EXPECT_FALSE(hasToolCallMarkers("no markers here", ToolCallFormat::HERMES_2_PRO));
    EXPECT_FALSE(hasToolCallMarkers("<tool_call> without end", ToolCallFormat::HERMES_2_PRO));
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
