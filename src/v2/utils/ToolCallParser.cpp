/**
 * @file ToolCallParser.cpp
 * @brief Implementation of tool call parsing from model output
 */

#include "ToolCallParser.h"
#include "Logger.h"
#include <nlohmann/json.hpp>
#include <sstream>
#include <iomanip>

using json = nlohmann::json;

namespace llaminar2
{

    // =========================================================================
    // Hermes 2 Pro format parser
    // =========================================================================
    //
    // Hermes 2 Pro format (used by Qwen 2.5, Qwen 3, Hermes 2/3):
    //   <tool_call>
    //   {"name": "get_weather", "arguments": {"location": "Paris"}}
    //   </tool_call>
    //
    // Multiple tool calls can appear sequentially. Any text outside
    // <tool_call>...</tool_call> blocks is treated as regular content.

    static ToolCallParseResult parseHermes2Pro(const std::string &text)
    {
        ToolCallParseResult result;

        const std::string open_tag = "<tool_call>";
        const std::string close_tag = "</tool_call>";

        size_t pos = 0;
        std::string content_parts;

        while (pos < text.size())
        {
            // Find next <tool_call> tag
            size_t tag_start = text.find(open_tag, pos);

            if (tag_start == std::string::npos)
            {
                // No more tool calls — rest is content
                content_parts += text.substr(pos);
                break;
            }

            // Text before the tag is content
            if (tag_start > pos)
            {
                content_parts += text.substr(pos, tag_start - pos);
            }

            // Find closing tag
            size_t json_start = tag_start + open_tag.size();
            size_t tag_end = text.find(close_tag, json_start);

            if (tag_end == std::string::npos)
            {
                // Unclosed tag — treat rest as content
                LOG_WARN("[ToolCallParser] Unclosed <tool_call> tag at position " << tag_start);
                content_parts += text.substr(tag_start);
                break;
            }

            // Extract JSON between tags
            std::string json_str = text.substr(json_start, tag_end - json_start);

            // Trim whitespace
            size_t first = json_str.find_first_not_of(" \t\n\r");
            size_t last = json_str.find_last_not_of(" \t\n\r");
            if (first != std::string::npos)
                json_str = json_str.substr(first, last - first + 1);

            // Parse JSON
            try
            {
                json tc_json = json::parse(json_str);

                ToolCall tc;
                tc.id = generateToolCallId();

                if (tc_json.contains("name"))
                    tc.name = tc_json["name"].get<std::string>();

                // Arguments can be a string or object
                if (tc_json.contains("arguments"))
                {
                    if (tc_json["arguments"].is_string())
                        tc.arguments = tc_json["arguments"].get<std::string>();
                    else
                        tc.arguments = tc_json["arguments"].dump();
                }

                if (!tc.name.empty())
                {
                    result.tool_calls.push_back(std::move(tc));
                }
                else
                {
                    LOG_WARN("[ToolCallParser] Tool call JSON missing 'name' field");
                }
            }
            catch (const json::parse_error &e)
            {
                LOG_WARN("[ToolCallParser] Failed to parse tool call JSON: " << e.what());
                // Treat the whole block as content
                content_parts += text.substr(tag_start, tag_end + close_tag.size() - tag_start);
            }

            pos = tag_end + close_tag.size();
        }

        // Trim whitespace from content
        size_t first = content_parts.find_first_not_of(" \t\n\r");
        size_t last = content_parts.find_last_not_of(" \t\n\r");
        if (first != std::string::npos)
            result.content = content_parts.substr(first, last - first + 1);

        return result;
    }

    // =========================================================================
    // Generic format parser (fallback)
    // =========================================================================
    //
    // Looks for JSON objects that have "name" and "arguments" fields anywhere
    // in the text. This is a best-effort parser for models without a known
    // format. It tries to find JSON objects in code blocks first, then
    // bare JSON objects.

    static ToolCallParseResult parseGeneric(const std::string &text)
    {
        ToolCallParseResult result;

        // Try to find JSON objects with "name" and "arguments" keys
        // Look for ```json ... ``` code blocks first
        const std::string code_start = "```json";
        const std::string code_end = "```";

        size_t pos = text.find(code_start);
        if (pos != std::string::npos)
        {
            size_t json_begin = pos + code_start.size();
            size_t json_end = text.find(code_end, json_begin);
            if (json_end != std::string::npos)
            {
                std::string json_str = text.substr(json_begin, json_end - json_begin);

                // Trim
                size_t first = json_str.find_first_not_of(" \t\n\r");
                size_t last = json_str.find_last_not_of(" \t\n\r");
                if (first != std::string::npos)
                    json_str = json_str.substr(first, last - first + 1);

                try
                {
                    json parsed = json::parse(json_str);

                    // Could be a single object or array of objects
                    auto extract_tc = [&](const json &obj)
                    {
                        if (obj.contains("name"))
                        {
                            ToolCall tc;
                            tc.id = generateToolCallId();
                            tc.name = obj["name"].get<std::string>();
                            if (obj.contains("arguments"))
                            {
                                if (obj["arguments"].is_string())
                                    tc.arguments = obj["arguments"].get<std::string>();
                                else
                                    tc.arguments = obj["arguments"].dump();
                            }
                            result.tool_calls.push_back(std::move(tc));
                        }
                    };

                    if (parsed.is_array())
                    {
                        for (const auto &item : parsed)
                            extract_tc(item);
                    }
                    else if (parsed.is_object())
                    {
                        extract_tc(parsed);
                    }

                    // Content is text before the code block
                    if (pos > 0)
                    {
                        std::string before = text.substr(0, pos);
                        size_t bf = before.find_first_not_of(" \t\n\r");
                        size_t bl = before.find_last_not_of(" \t\n\r");
                        if (bf != std::string::npos)
                            result.content = before.substr(bf, bl - bf + 1);
                    }

                    return result;
                }
                catch (const json::parse_error &)
                {
                    // Fall through to return as plain content
                }
            }
        }

        // No tool calls found — return as plain content
        result.content = text;
        return result;
    }

    // =========================================================================
    // Public API
    // =========================================================================

    ToolCallParseResult parseToolCalls(const std::string &text, ToolCallFormat format)
    {
        switch (format)
        {
        case ToolCallFormat::HERMES_2_PRO:
            return parseHermes2Pro(text);

        case ToolCallFormat::GENERIC:
            return parseGeneric(text);

        case ToolCallFormat::NONE:
        default:
        {
            // No parsing — return everything as content
            ToolCallParseResult result;
            result.content = text;
            return result;
        }
        }
    }

    bool hasToolCallMarkers(const std::string &text, ToolCallFormat format)
    {
        switch (format)
        {
        case ToolCallFormat::HERMES_2_PRO:
            return text.find("<tool_call>") != std::string::npos &&
                   text.find("</tool_call>") != std::string::npos;

        case ToolCallFormat::GENERIC:
            return text.find("```json") != std::string::npos &&
                   text.find("\"name\"") != std::string::npos;

        case ToolCallFormat::LLAMA_3X:
            return text.find("<|python_tag|>") != std::string::npos;

        case ToolCallFormat::MISTRAL_NEMO:
            return text.find("[TOOL_CALLS]") != std::string::npos;

        default:
            return false;
        }
    }

    std::string generateToolCallId()
    {
        static thread_local std::mt19937 gen{std::random_device{}()};
        std::uniform_int_distribution<uint64_t> dist;
        uint64_t val = dist(gen);

        std::ostringstream ss;
        ss << "call_" << std::hex << std::setfill('0') << std::setw(12) << val;
        return ss.str();
    }

    nlohmann::json toolCallToJson(const ToolCall &tc)
    {
        return {
            {"id", tc.id},
            {"type", "function"},
            {"function", {{"name", tc.name}, {"arguments", tc.arguments}}}};
    }

} // namespace llaminar2
