/**
 * @file ToolCallParser.cpp
 * @brief Architecture-selected tool-call protocol parsers
 *
 * The parser is the boundary between model-native text and the OpenAI HTTP
 * contract.  Each implementation accepts one exact grammar and preserves a
 * malformed block as ordinary content.  It must not probe several parsers in
 * sequence: doing so could turn arbitrary generated text into an executable
 * tool invocation.
 */

#include "ToolCallParser.h"
#include "Logger.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cctype>
#include <iomanip>
#include <optional>
#include <sstream>
#include <string_view>

using json = nlohmann::json;

namespace llaminar2
{

    /** @brief Length of the suffix which could grow into @p marker. */
    static size_t partialMarkerSuffixLength(
        std::string_view text,
        std::string_view marker)
    {
        const size_t maximum = std::min(text.size(), marker.size());
        for (size_t length = maximum; length > 0; --length)
        {
            if (text.substr(text.size() - length) == marker.substr(0, length))
                return length;
        }
        return 0;
    }

    /** @brief Return a copy without surrounding ASCII whitespace. */
    static std::string trimAsciiWhitespace(std::string_view value)
    {
        size_t first = 0;
        while (first < value.size() &&
               std::isspace(static_cast<unsigned char>(value[first])))
        {
            ++first;
        }

        size_t last = value.size();
        while (last > first &&
               std::isspace(static_cast<unsigned char>(value[last - 1])))
        {
            --last;
        }
        return std::string(value.substr(first, last - first));
    }

    /** @brief Trim accumulated human-readable content after protocol removal. */
    static void trimParsedContent(ToolCallParseResult &result)
    {
        result.content = trimAsciiWhitespace(result.content);
    }

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

        result.content = std::move(content_parts);
        trimParsedContent(result);

        return result;
    }

    // =========================================================================
    // Qwen 3 native XML-like format parser
    // =========================================================================
    //
    // The Qwen 3.5/3.8 template emits this deliberately small grammar:
    //   <tool_call>
    //   <function=get_weather>
    //   <parameter=city>
    //   Paris
    //   </parameter>
    //   </function>
    //   </tool_call>
    //
    // This is not Hermes JSON despite sharing the outer <tool_call> tag.  The
    // implementation below parses only the installed grammar and rejects a
    // partial function or parameter as a whole so clients never execute a call
    // assembled from incomplete model output.

    /** @brief Whether a Qwen function or parameter name is unambiguous. */
    static bool isQwenToolName(const std::string &name)
    {
        if (name.empty())
            return false;
        for (const unsigned char ch : name)
        {
            if (!(std::isalnum(ch) || ch == '_' || ch == '-' || ch == '.'))
                return false;
        }
        return true;
    }

    /**
     * @brief Decode a native parameter body into one JSON argument value.
     *
     * Structured values and JSON primitives retain their types.  Plain text
     * remains a string, which is the representation used by the Qwen template
     * for ordinary string parameters.
     */
    static json decodeQwenParameterValue(std::string_view body)
    {
        const std::string value = trimAsciiWhitespace(body);
        if (value.empty())
            return "";

        json parsed = json::parse(value, nullptr, false);
        if (!parsed.is_discarded())
            return parsed;
        return value;
    }

    /**
     * @brief Parse the payload of one complete Qwen @c <tool_call> block.
     * @return A call only when every delimiter and identifier is valid.
     */
    static std::optional<ToolCall> parseQwen3XmlPayload(std::string_view payload)
    {
        static constexpr std::string_view function_open = "<function=";
        static constexpr std::string_view function_close = "</function>";
        static constexpr std::string_view parameter_open = "<parameter=";
        static constexpr std::string_view parameter_close = "</parameter>";

        const std::string body = trimAsciiWhitespace(payload);
        if (body.compare(0, function_open.size(), function_open) != 0)
            return std::nullopt;

        const size_t function_name_end = body.find('>', function_open.size());
        if (function_name_end == std::string::npos)
            return std::nullopt;
        const std::string function_name = trimAsciiWhitespace(
            std::string_view(body).substr(
                function_open.size(),
                function_name_end - function_open.size()));
        if (!isQwenToolName(function_name))
            return std::nullopt;

        const size_t function_close_pos = body.rfind(function_close);
        if (function_close_pos == std::string::npos ||
            function_close_pos < function_name_end + 1 ||
            !trimAsciiWhitespace(std::string_view(body).substr(
                                     function_close_pos + function_close.size()))
                 .empty())
        {
            return std::nullopt;
        }

        json arguments = json::object();
        size_t cursor = function_name_end + 1;
        while (cursor < function_close_pos)
        {
            while (cursor < function_close_pos &&
                   std::isspace(static_cast<unsigned char>(body[cursor])))
            {
                ++cursor;
            }
            if (cursor == function_close_pos)
                break;
            if (body.compare(cursor, parameter_open.size(), parameter_open) != 0)
                return std::nullopt;

            const size_t parameter_name_begin = cursor + parameter_open.size();
            const size_t parameter_name_end = body.find('>', parameter_name_begin);
            if (parameter_name_end == std::string::npos ||
                parameter_name_end >= function_close_pos)
            {
                return std::nullopt;
            }
            const std::string parameter_name = trimAsciiWhitespace(
                std::string_view(body).substr(
                    parameter_name_begin,
                    parameter_name_end - parameter_name_begin));
            if (!isQwenToolName(parameter_name) || arguments.contains(parameter_name))
                return std::nullopt;

            const size_t value_begin = parameter_name_end + 1;
            const size_t parameter_close_pos = body.find(parameter_close, value_begin);
            if (parameter_close_pos == std::string::npos ||
                parameter_close_pos > function_close_pos)
            {
                return std::nullopt;
            }
            arguments[parameter_name] = decodeQwenParameterValue(
                std::string_view(body).substr(
                    value_begin,
                    parameter_close_pos - value_begin));
            cursor = parameter_close_pos + parameter_close.size();
        }

        ToolCall call;
        call.id = generateToolCallId();
        call.name = function_name;
        call.arguments = arguments.dump();
        return call;
    }

    /** @brief Extract every complete Qwen tool-call block in generation order. */
    static ToolCallParseResult parseQwen3Xml(const std::string &text)
    {
        static const std::string open_tag = "<tool_call>";
        static const std::string close_tag = "</tool_call>";

        ToolCallParseResult result;
        size_t cursor = 0;
        while (cursor < text.size())
        {
            const size_t block_begin = text.find(open_tag, cursor);
            if (block_begin == std::string::npos)
            {
                result.content += text.substr(cursor);
                break;
            }
            result.content += text.substr(cursor, block_begin - cursor);

            const size_t payload_begin = block_begin + open_tag.size();
            const size_t block_end = text.find(close_tag, payload_begin);
            if (block_end == std::string::npos)
            {
                LOG_WARN("[ToolCallParser] Unclosed Qwen <tool_call> tag at position "
                         << block_begin);
                result.content += text.substr(block_begin);
                break;
            }

            auto call = parseQwen3XmlPayload(
                std::string_view(text).substr(payload_begin, block_end - payload_begin));
            if (call)
            {
                result.tool_calls.push_back(std::move(*call));
            }
            else
            {
                LOG_WARN("[ToolCallParser] Malformed Qwen native tool-call block at position "
                         << block_begin);
                result.content += text.substr(
                    block_begin,
                    block_end + close_tag.size() - block_begin);
            }
            cursor = block_end + close_tag.size();
        }

        trimParsedContent(result);
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

        case ToolCallFormat::QWEN_3_XML:
            return parseQwen3Xml(text);

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
        case ToolCallFormat::QWEN_3_XML:
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

    StreamingToolCallSplitter::Event
    StreamingToolCallSplitter::Event::contentEvent(std::string text)
    {
        Event event;
        event.kind = Kind::Content;
        event.content = std::move(text);
        return event;
    }

    StreamingToolCallSplitter::Event
    StreamingToolCallSplitter::Event::toolCallEvent(ToolCall call)
    {
        Event event;
        event.kind = Kind::ToolCall;
        event.tool_call = std::move(call);
        return event;
    }

    StreamingToolCallSplitter::StreamingToolCallSplitter(ToolCallFormat format)
        : format_(format)
    {
        switch (format_)
        {
        case ToolCallFormat::HERMES_2_PRO:
        case ToolCallFormat::QWEN_3_XML:
            open_marker_ = "<tool_call>";
            close_marker_ = "</tool_call>";
            break;
        case ToolCallFormat::GENERIC:
            open_marker_ = "```json";
            close_marker_ = "```";
            break;
        default:
            // Formats without an installed parser are transparent. The model
            // schema remains the authority; this class never probes another
            // grammar based on generated bytes.
            break;
        }
    }

    std::vector<StreamingToolCallSplitter::Event>
    StreamingToolCallSplitter::process(std::string_view text)
    {
        if (!text.empty())
            buffer_.append(text.data(), text.size());
        return drain(/*terminal=*/false);
    }

    std::vector<StreamingToolCallSplitter::Event>
    StreamingToolCallSplitter::flush()
    {
        return drain(/*terminal=*/true);
    }

    std::vector<StreamingToolCallSplitter::Event>
    StreamingToolCallSplitter::drain(bool terminal)
    {
        std::vector<Event> events;
        if (open_marker_.empty() || close_marker_.empty())
        {
            if (!buffer_.empty())
            {
                events.push_back(Event::contentEvent(std::move(buffer_)));
                buffer_.clear();
            }
            return events;
        }

        while (!buffer_.empty())
        {
            if (phase_ == Phase::Content)
            {
                const size_t marker = buffer_.find(open_marker_);
                if (marker != std::string::npos)
                {
                    if (marker > 0)
                    {
                        events.push_back(Event::contentEvent(
                            buffer_.substr(0, marker)));
                        buffer_.erase(0, marker);
                    }
                    phase_ = Phase::ToolBlock;
                    continue;
                }

                // Keep only a suffix which could become an opening delimiter
                // after the next tokenizer piece. Every earlier byte is known
                // ordinary content and can reach the client immediately.
                const size_t held = terminal
                    ? 0u
                    : partialMarkerSuffixLength(buffer_, open_marker_);
                const size_t safe = buffer_.size() - held;
                if (safe > 0)
                {
                    events.push_back(Event::contentEvent(
                        buffer_.substr(0, safe)));
                    buffer_.erase(0, safe);
                }
                break;
            }

            const size_t close = buffer_.find(
                close_marker_, open_marker_.size());
            if (close == std::string::npos)
            {
                if (terminal)
                {
                    // An incomplete block is literal model output. Executing
                    // a prefix would manufacture arguments the model never
                    // completed.
                    events.push_back(Event::contentEvent(std::move(buffer_)));
                    buffer_.clear();
                    phase_ = Phase::Content;
                }
                break;
            }

            const size_t block_size = close + close_marker_.size();
            const std::string block = buffer_.substr(0, block_size);
            ToolCallParseResult parsed = parseToolCalls(block, format_);
            if (parsed.hasToolCalls() && parsed.content.empty())
            {
                for (auto &call : parsed.tool_calls)
                    events.push_back(Event::toolCallEvent(std::move(call)));
            }
            else
            {
                // Preserve malformed syntax byte-for-byte. The strict parser
                // may trim human-facing content, but a failed protocol block
                // must remain an exact inert response.
                events.push_back(Event::contentEvent(block));
            }
            buffer_.erase(0, block_size);
            phase_ = Phase::Content;
        }
        return events;
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
