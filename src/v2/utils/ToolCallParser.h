/**
 * @file ToolCallParser.h
 * @brief Parses tool calls from raw model output text
 *
 * Implements format-specific parsers for extracting structured tool calls
 * from model-generated text. The format is determined per-model via
 * GraphConfig::tool_call_format.
 *
 * Currently supported:
 *   - HERMES_2_PRO: <tool_call>\n{"name":"...","arguments":{...}}\n</tool_call>
 *     (Qwen 2.5, Qwen 3, Hermes 2/3, and many community models)
 *   - GENERIC: Fallback that looks for JSON objects with "name" and "arguments"
 */

#pragma once

#include "ToolCallTypes.h"
#include <string>
#include <random>
#include <nlohmann/json.hpp>

namespace llaminar2
{

    /**
     * @brief Parse tool calls from model output based on the specified format
     *
     * Scans raw model output text for tool call patterns matching the given
     * format. Returns any non-tool-call text as content and extracts tool calls
     * into structured ToolCall objects.
     *
     * @param text Raw model output text
     * @param format The tool call format to parse for
     * @return ToolCallParseResult with content and extracted tool_calls
     */
    ToolCallParseResult parseToolCalls(const std::string &text, ToolCallFormat format);

    /**
     * @brief Check if text contains potential tool call markers for the given format
     *
     * Lightweight check (no JSON parsing) to quickly determine if tool call
     * parsing is needed. Useful for streaming to know when to start buffering.
     *
     * @param text Text to check
     * @param format The tool call format to check for
     * @return true if the text contains markers suggesting tool calls
     */
    bool hasToolCallMarkers(const std::string &text, ToolCallFormat format);

    /**
     * @brief Generate a unique tool call ID (e.g., "call_abc123def456")
     */
    std::string generateToolCallId();

    /**
     * @brief Serialize a ToolCall to OpenAI-format JSON
     */
    nlohmann::json toolCallToJson(const ToolCall &tc);

} // namespace llaminar2
