/**
 * @file ToolCallTypes.h
 * @brief Types for OpenAI-compatible tool/function calling support
 *
 * Defines data structures for tool definitions, tool calls in messages,
 * and tool call format detection/parsing per model family.
 */

#pragma once

#include <string>
#include <vector>

namespace llaminar2
{

    /**
     * @brief Tool call format used by a model for emitting tool invocations
     *
     * Different model families emit tool calls in different text formats.
     * The format determines how we parse raw model output into structured
     * tool_calls objects in the OpenAI response.
     *
     * Each model's graph config can specify which format it uses;
     * HERMES_2_PRO is the default (covers Qwen 2.5, Qwen 3, Hermes, etc.).
     */
    enum class ToolCallFormat
    {
        NONE,          ///< Model does not support tool calling
        HERMES_2_PRO,  ///< <tool_call>\n{"name":"...","arguments":{...}}\n</tool_call>
        LLAMA_3X,      ///< <|python_tag|>{"name":"...","parameters":{...}}
        FUNCTIONARY,   ///< Functionary v3.x format
        MISTRAL_NEMO,  ///< [TOOL_CALLS] format
        DEEPSEEK_R1,   ///< DeepSeek R1 format (with reasoning extraction)
        GENERIC,       ///< Generic fallback (wraps tool calls in JSON code blocks)
    };

    /**
     * @brief Convert ToolCallFormat to human-readable string
     */
    inline const char *toolCallFormatName(ToolCallFormat fmt)
    {
        switch (fmt)
        {
        case ToolCallFormat::NONE:
            return "None";
        case ToolCallFormat::HERMES_2_PRO:
            return "Hermes 2 Pro";
        case ToolCallFormat::LLAMA_3X:
            return "Llama 3.x";
        case ToolCallFormat::FUNCTIONARY:
            return "Functionary";
        case ToolCallFormat::MISTRAL_NEMO:
            return "Mistral Nemo";
        case ToolCallFormat::DEEPSEEK_R1:
            return "DeepSeek R1";
        case ToolCallFormat::GENERIC:
            return "Generic";
        default:
            return "Unknown";
        }
    }

    // =========================================================================
    // Tool definitions (request-side: what tools are available)
    // =========================================================================

    /**
     * @brief A single tool call emitted by the model
     *
     * Corresponds to OpenAI's tool_calls[].function object.
     */
    struct ToolCall
    {
        std::string id;                   ///< Unique tool call ID (e.g., "call_abc123")
        std::string name;                 ///< Function name
        std::string arguments = "{}";     ///< JSON string of function arguments
    };

    /**
     * @brief Result of parsing model output for tool calls
     */
    struct ToolCallParseResult
    {
        std::string content;            ///< Non-tool-call text content (may be empty)
        std::vector<ToolCall> tool_calls; ///< Extracted tool calls (may be empty)

        /// Whether any tool calls were found
        bool hasToolCalls() const { return !tool_calls.empty(); }
    };

} // namespace llaminar2
