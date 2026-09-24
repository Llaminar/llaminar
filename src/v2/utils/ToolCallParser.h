/**
 * @file ToolCallParser.h
 * @brief Parses tool calls from raw model output text
 *
 * Implements strict, format-specific parsers for extracting structured tool
 * calls from model-generated text. The format is selected by the immutable
 * model schema and is never inferred from response text. Malformed blocks are
 * preserved as ordinary content so the server does not manufacture a call the
 * model did not encode completely.
 *
 * Currently supported:
 *   - HERMES_2_PRO: <tool_call>\n{"name":"...","arguments":{...}}\n</tool_call>
 *     (Hermes 2/3 and compatible community models)
 *   - QWEN_3_XML: <tool_call><function=name><parameter=key>value</parameter>
 *     </function></tool_call> (Qwen 3.5/3.8 hybrid families)
 *   - GENERIC: Fallback that looks for JSON objects with "name" and "arguments"
 */

#pragma once

#include "ToolCallTypes.h"
#include <string>
#include <string_view>
#include <vector>
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
     * @brief Incrementally separates ordinary assistant text from tool calls.
     *
     * A complete tool-call delimiter can straddle arbitrary tokenizer pieces.
     * This state machine therefore retains only a possible opening-marker
     * suffix or one active tool block. Ordinary content is released as soon as
     * it cannot become protocol syntax; complete calls are parsed by the same
     * architecture-selected grammar as non-streaming responses.
     *
     * Reasoning text must be split before entering this class. This keeps the
     * OpenAI @c reasoning_content channel independent from callable output and
     * prevents a tool-enabled request from buffering the whole response.
     */
    class StreamingToolCallSplitter
    {
    public:
        /** @brief One ordered piece of the model's public response stream. */
        struct Event
        {
            enum class Kind
            {
                Content, ///< Safe ordinary assistant text.
                ToolCall ///< One complete, validated invocation.
            };

            Kind kind{Kind::Content};
            std::string content;
            ToolCall tool_call;

            /** @brief Construct an ordinary-content event. */
            static Event contentEvent(std::string text);

            /** @brief Construct a validated tool-call event. */
            static Event toolCallEvent(ToolCall call);
        };

        /**
         * @brief Bind the immutable model-native grammar for this stream.
         * @param format Format selected by the admitted model schema.
         */
        explicit StreamingToolCallSplitter(ToolCallFormat format);

        /**
         * @brief Consume one answer-text fragment.
         * @return Ordered content/call events safe to publish immediately.
         */
        std::vector<Event> process(std::string_view text);

        /**
         * @brief End the stream and expose any incomplete marker as content.
         * @return Final ordered events; an incomplete call is never executable.
         */
        std::vector<Event> flush();

    private:
        enum class Phase
        {
            Content,
            ToolBlock
        };

        /** @brief Drain all currently decidable bytes from @ref buffer_. */
        std::vector<Event> drain(bool terminal);

        ToolCallFormat format_{ToolCallFormat::NONE};
        std::string open_marker_;
        std::string close_marker_;
        Phase phase_{Phase::Content};
        std::string buffer_;
    };

    /**
     * @brief Generate a unique tool call ID (e.g., "call_abc123def456")
     */
    std::string generateToolCallId();

    /**
     * @brief Serialize a ToolCall to OpenAI-format JSON
     */
    nlohmann::json toolCallToJson(const ToolCall &tc);

} // namespace llaminar2
