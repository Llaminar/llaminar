/**
 * @file ToolCallParser.cpp
 * @brief Architecture-selected tool-call protocol parsers
 *
 * The parser is the boundary between model-native text and the OpenAI HTTP
 * contract.  Each implementation accepts one exact grammar and preserves a
 * malformed block as ordinary content.  It must not probe several parsers in
 * sequence: doing so could turn arbitrary generated text into an executable
 * tool invocation. Native Qwen argument boundaries use the admitted schema
 * and the parameter's newline framing so literal file bytes remain data.
 */

#include "ToolCallParser.h"
#include "Logger.h"
#include "PythonJson.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
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
     * The template puts one framing newline before and after each value. Only
     * those two bytes are syntax; indentation and additional newlines belong
     * to file contents and edit search strings. JSON-looking string values
     * must stay strings, including quotes, numbers, booleans and JSON documents.
     * Non-string parameters are JSON values according to their declared schema.
     * @param body Raw parameter bytes between the native delimiters.
     * @param schema Declared property schema from the admitted function.
     * @return Decoded value, or no value when a typed JSON body is malformed.
     */
    static std::optional<json> decodeQwenParameterValue(std::string_view body,
                                                     const json &schema)
    {
        if (body.starts_with("\r\n"))
            body.remove_prefix(2);
        else if (body.starts_with('\n'))
            body.remove_prefix(1);
        if (body.ends_with("\r\n"))
            body.remove_suffix(2);
        else if (body.ends_with('\n'))
            body.remove_suffix(1);

        if (schema.value("type", json{}) == "string")
            return std::string(body);

        json parsed = json::parse(body, nullptr, false);
        if (parsed.is_discarded())
            return std::nullopt;
        return parsed;
    }

    /** @return An admitted function's parameter schema, without guessing unknown tools. */
    static const json *qwenFunctionParameters(const json &tools, const std::string &name)
    {
        if (!tools.is_array())
            return nullptr;
        for (const auto &tool : tools)
        {
            if (!tool.is_object() || !tool.contains("function"))
                continue;
            const auto &function = tool["function"];
            if (function.is_object() && function.value("name", json{}) == name &&
                function.contains("parameters") && function["parameters"].is_object())
                return &function["parameters"];
        }
        return nullptr;
    }

    /**
     * @brief Find the outer Qwen terminator after the function's closing frontier.
     * @param text Complete or growing native generation bytes.
     * @param begin First payload byte after the outer opening tag.
     * @return Closing-tag position, or npos until its structural predecessor exists.
     *
     * File contents routinely contain literal protocol tags. A tool terminator
     * inside a parameter is data unless it follows the function terminator;
     * streaming and complete-response parsing must use the same boundary rule.
     */
    static size_t qwenToolBlockEnd(std::string_view text, size_t begin)
    {
        constexpr std::string_view function_close = "</function>";
        constexpr std::string_view tool_close = "</tool_call>";
        for (size_t close = text.find(tool_close, begin); close != std::string_view::npos;
             close = text.find(tool_close, close + tool_close.size()))
        {
            size_t predecessor = close;
            while (predecessor > begin && std::isspace(static_cast<unsigned char>(text[predecessor - 1])))
                --predecessor;
            if (predecessor >= begin + function_close.size() &&
                text.substr(predecessor - function_close.size(), function_close.size()) == function_close)
                return close;
        }
        return std::string_view::npos;
    }

    /**
     * @brief Find a framed parameter terminator followed by an admitted element.
     * @param body Native function bytes.
     * @param begin First value byte after the parameter opening tag.
     * @param function_end Authenticated closing position of the containing function.
     * @param properties Object containing the admitted function's argument schemas.
     * @return Structural closing position, or npos for an incomplete parameter.
     *
     * A newline-framed value has a line boundary on either side of its closing
     * tag. Native generations can omit the newline before that tag, especially
     * when asked to write a string without a trailing newline. Entirely inline
     * tag pairs remain file content, even if their names match real arguments.
     * An unknown parameter name cannot establish a grammar boundary inside a
     * value. Compact native parameters retain their unframed delimiters.
     */
    static size_t qwenParameterEnd(std::string_view body, size_t begin,
                                   size_t function_end, const json &properties)
    {
        constexpr std::string_view close_tag = "</parameter>";
        constexpr std::string_view open_tag = "<parameter=";
        const bool newline_framed = body.substr(begin).starts_with('\n') ||
                                    body.substr(begin).starts_with("\r\n");
        for (size_t close = body.find(close_tag, begin); close < function_end;
             close = body.find(close_tag, close + close_tag.size()))
        {
            size_t next = close + close_tag.size();
            bool closes_line = close > begin && body[close - 1] == '\n';
            while (next < function_end && std::isspace(static_cast<unsigned char>(body[next])))
            {
                closes_line |= body[next] == '\n';
                ++next;
            }
            if (newline_framed && !closes_line)
                continue;
            if (next == function_end)
                return close;
            if (!body.substr(next).starts_with(open_tag))
                continue;
            const size_t name_begin = next + open_tag.size();
            const size_t name_end = body.find('>', name_begin);
            if (name_end >= function_end)
                continue;
            const std::string name = trimAsciiWhitespace(body.substr(name_begin, name_end - name_begin));
            if (isQwenToolName(name) && properties.contains(name) && properties[name].is_object())
                return close;
        }
        return std::string_view::npos;
    }

    /**
     * @brief Parse the payload of one complete Qwen @c <tool_call> block.
     * @return A call only when every delimiter and identifier is valid.
     */
    static std::optional<ToolCall> parseQwen3XmlPayload(std::string_view payload,
                                                     const json &tools)
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

        const json *parameters = qwenFunctionParameters(tools, function_name);
        if (!parameters)
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

            const auto properties = parameters->find("properties");
            if (properties == parameters->end() || !properties->is_object() ||
                !properties->contains(parameter_name) || !(*properties)[parameter_name].is_object())
                return std::nullopt;
            const size_t value_begin = parameter_name_end + 1;
            const size_t parameter_close_pos = qwenParameterEnd(
                body, value_begin, function_close_pos, *properties);
            if (parameter_close_pos == std::string::npos ||
                parameter_close_pos > function_close_pos)
            {
                return std::nullopt;
            }
            auto value = decodeQwenParameterValue(
                std::string_view(body).substr(
                    value_begin,
                    parameter_close_pos - value_begin), (*properties)[parameter_name]);
            if (!value)
                return std::nullopt;
            arguments[parameter_name] = std::move(*value);
            cursor = parameter_close_pos + parameter_close.size();
        }

        // A complete outer delimiter does not prove a complete invocation.
        // Keep incomplete model output inert instead of asking the client to
        // execute an argument object that violates its required-key contract.
        if (const auto required = parameters->find("required"); required != parameters->end())
        {
            if (!required->is_array())
                return std::nullopt;
            for (const auto& name : *required)
                if (!name.is_string() || !arguments.contains(name.get_ref<const std::string&>()))
                    return std::nullopt;
        }

        ToolCall call;
        call.id = generateToolCallId();
        call.name = function_name;
        call.arguments = arguments.dump();
        return call;
    }

    /** @brief Extract every complete Qwen tool-call block in generation order. */
    static ToolCallParseResult parseQwen3Xml(const std::string &text, const json &tools)
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
            const size_t block_end = qwenToolBlockEnd(text, payload_begin);
            if (block_end == std::string::npos)
            {
                LOG_WARN("[ToolCallParser] Unclosed Qwen <tool_call> tag at position "
                         << block_begin);
                result.content += text.substr(block_begin);
                break;
            }

            auto call = parseQwen3XmlPayload(
                std::string_view(text).substr(payload_begin, block_end - payload_begin), tools);
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
    // DeepSeek-V4 DSML parser
    // =========================================================================
    //
    // Grammar (vLLM vllm/parser/deepseek_v4.py, the production comparison
    // target; reasoning is split before this parser):
    //
    //   [content]\n\n<｜DSML｜tool_calls>
    //   <｜DSML｜invoke name="f">
    //   <｜DSML｜parameter name="k" string="true|false">v</｜DSML｜parameter>
    //   </｜DSML｜invoke>
    //   </｜DSML｜tool_calls>
    //
    // - The section opens at <｜DSML｜tool_calls>, the observed misspellings
    //   <｜DSML｜toolcalls>/<｜DSML｜tool>, or a bare invoke. The official
    //   encoder writes "\n\n" before the section; that separator belongs to
    //   the protocol, not to the visible content.
    // - An invoke ends at </｜DSML｜invoke> or </｜DSML｜tool_calls>.
    // - Parameters: string="true" keeps the raw text; otherwise (or with the
    //   attribute missing) JSON, falling back to the raw text. A parameter ends
    //   at its close tag or at the next parameter head. Arguments are emitted
    //   with Python json.dumps spelling.
    // - A tool call ends the turn: text between/after invokes is dropped.
    // - Fail closed: a section without one complete invoke stays content, and
    //   an incomplete trailing invoke becomes content after the complete calls.

    namespace
    {
        const std::string kDsml = "｜DSML｜";
        const std::string kDsmlInvokePrefix = "<" + kDsml + "invoke name=\"";
        const std::string kDsmlInvokeEnd = "</" + kDsml + "invoke>";
        const std::string kDsmlToolEnd = "</" + kDsml + "tool_calls>";
        const std::string kDsmlParamStart = "<" + kDsml + "parameter";
        const std::string kDsmlParamClose = "</" + kDsml + "parameter>";
        /** @brief Section openers; the separator-owning spellings start earlier and so win ties. */
        const std::array<std::string, 8> kDsmlOpeners = {
            "\n\n<" + kDsml + "tool_calls>", "\n\n<" + kDsml + "toolcalls>", "\n\n<" + kDsml + "tool>",
            "\n\n" + kDsmlInvokePrefix,
            "<" + kDsml + "tool_calls>", "<" + kDsml + "toolcalls>", "<" + kDsml + "tool>", kDsmlInvokePrefix};

        size_t findDsmlSection(const std::string &text)
        {
            size_t best = std::string::npos;
            for (const auto &opener : kDsmlOpeners)
                best = std::min(best, text.find(opener));
            return best;
        }

        size_t dsmlPartialOpenerLength(const std::string &text)
        {
            size_t held = 0;
            for (const auto &opener : kDsmlOpeners)
                held = std::max(held, partialMarkerSuffixLength(text, opener));
            return held;
        }

        bool isRegexSpace(char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v'; }

        /**
         * @brief Match `\s+name="([^"]+)"(?:\s+string="([^"]*)")?\s*>` at @p i.
         * @return Position after '>' or npos.
         */
        size_t matchDsmlParamHead(const std::string &raw, size_t i, std::string &name,
                                  std::optional<std::string> &is_string)
        {
            const size_t ws = i;
            while (i < raw.size() && isRegexSpace(raw[i])) ++i;
            if (i == ws || raw.compare(i, 6, "name=\"") != 0) return std::string::npos;
            i += 6;
            const size_t name_end = raw.find('"', i);
            if (name_end == std::string::npos || name_end == i) return std::string::npos;
            name = raw.substr(i, name_end - i);
            i = name_end + 1;
            // Optional string attribute (regex backtracks to "absent" when it does not match).
            size_t j = i;
            while (j < raw.size() && isRegexSpace(raw[j])) ++j;
            is_string.reset();
            if (j > i && raw.compare(j, 8, "string=\"") == 0)
            {
                const size_t value_end = raw.find('"', j + 8);
                if (value_end != std::string::npos)
                {
                    is_string = raw.substr(j + 8, value_end - j - 8);
                    i = value_end + 1;
                }
            }
            while (i < raw.size() && isRegexSpace(raw[i])) ++i;
            return i < raw.size() && raw[i] == '>' ? i + 1 : std::string::npos;
        }

        /** @brief Position of the next `<｜DSML｜parameter\s+name=` at or after @p from. */
        size_t findDsmlNextParamHead(const std::string &raw, size_t from)
        {
            for (size_t p = raw.find(kDsmlParamStart, from); p != std::string::npos;
                 p = raw.find(kDsmlParamStart, p + 1))
            {
                size_t i = p + kDsmlParamStart.size();
                const size_t ws = i;
                while (i < raw.size() && isRegexSpace(raw[i])) ++i;
                if (i > ws && raw.compare(i, 5, "name=") == 0) return p;
            }
            return std::string::npos;
        }

        /** @brief _dsml_arg_converter (complete parameters only). */
        nlohmann::ordered_json convertDsmlArguments(const std::string &raw)
        {
            nlohmann::ordered_json params = nlohmann::ordered_json::object();
            size_t pos = 0;
            for (size_t p = raw.find(kDsmlParamStart, pos); p != std::string::npos; p = raw.find(kDsmlParamStart, pos))
            {
                std::string name;
                std::optional<std::string> is_string;
                const size_t value_start = matchDsmlParamHead(raw, p + kDsmlParamStart.size(), name, is_string);
                if (value_start == std::string::npos) { pos = p + 1; continue; }
                const size_t close = raw.find(kDsmlParamClose, value_start);
                const size_t next = findDsmlNextParamHead(raw, value_start);
                const size_t end = std::min(close, next);
                if (end == std::string::npos) { pos = p + 1; continue; }
                const std::string value = raw.substr(value_start, end - value_start);
                if (is_string && *is_string == "true")
                    params[name] = value;
                else
                {
                    auto parsed = nlohmann::ordered_json::parse(value, nullptr, false);
                    params[name] = parsed.is_discarded() ? nlohmann::ordered_json(value) : std::move(parsed);
                }
                pos = end == close ? close + kDsmlParamClose.size() : next;
            }
            return params;
        }

        /** @brief _unwrap_wrapper_args: remove a spurious {"arguments"|"input": {...}} wrapper. */
        std::string unwrapDsmlWrapperArguments(const nlohmann::ordered_json &args, const json &tools,
                                               const std::string &function_name)
        {
            const std::string original = pythonJsonDumps(args);
            if (!tools.is_array() || tools.empty() || function_name.empty() || !args.is_object()) return original;
            const json *properties = nullptr;
            for (const auto &tool : tools)
                if (tool.contains("function") && tool["function"].value("name", "") == function_name &&
                    tool["function"].contains("parameters") && tool["function"]["parameters"].contains("properties") &&
                    tool["function"]["parameters"]["properties"].is_object())
                {
                    properties = &tool["function"]["parameters"]["properties"];
                    break;
                }
            if (!properties || properties->empty()) return original;
            for (const char *wrapper : {"arguments", "input"})
            {
                if (args.size() != 1 || !args.contains(wrapper) || properties->contains(wrapper)) continue;
                nlohmann::ordered_json inner = args[wrapper];
                if (inner.is_string())
                {
                    inner = nlohmann::ordered_json::parse(inner.get<std::string>(), nullptr, false);
                    if (inner.is_discarded()) return original;
                }
                if (!inner.is_object()) continue;
                const bool subset = std::all_of(inner.items().begin(), inner.items().end(),
                                                [&](const auto &item) { return properties->contains(item.key()); });
                if (subset) return pythonJsonDumps(inner);
            }
            return original;
        }
    } // namespace

    std::vector<StreamingToolCallSplitter::Event>
    StreamingToolCallSplitter::drainDeepSeekV4(bool terminal)
    {
        std::vector<Event> events;
        if (phase_ == Phase::Content)
        {
            const size_t section = findDsmlSection(buffer_);
            if (section == std::string::npos)
            {
                const size_t held = terminal ? 0u : dsmlPartialOpenerLength(buffer_);
                if (buffer_.size() > held)
                {
                    events.push_back(Event::contentEvent(buffer_.substr(0, buffer_.size() - held)));
                    buffer_.erase(0, buffer_.size() - held);
                }
                return events;
            }
            if (section > 0) events.push_back(Event::contentEvent(buffer_.substr(0, section)));
            buffer_.erase(0, section);
            phase_ = Phase::ToolBlock;
            dsml_scan_ = 0;
            dsml_emitted_ = false;
        }

        // Tool section: buffer_ retains the whole section so a call-less section
        // can still be returned verbatim. Emit each invoke once it closes.
        while (true)
        {
            const size_t invoke = buffer_.find(kDsmlInvokePrefix, dsml_scan_);
            size_t name_end = std::string::npos, args_end = std::string::npos, end_size = 0;
            if (invoke != std::string::npos)
            {
                name_end = buffer_.find("\">", invoke + kDsmlInvokePrefix.size());
                if (name_end != std::string::npos)
                {
                    const size_t a = buffer_.find(kDsmlInvokeEnd, name_end + 2);
                    const size_t b = buffer_.find(kDsmlToolEnd, name_end + 2);
                    args_end = std::min(a, b);
                    if (args_end != std::string::npos) end_size = args_end == a ? kDsmlInvokeEnd.size() : kDsmlToolEnd.size();
                }
            }
            if (args_end == std::string::npos)
            {
                if (terminal)
                {
                    if (!dsml_emitted_) events.push_back(Event::contentEvent(buffer_));
                    else if (invoke != std::string::npos) events.push_back(Event::contentEvent(buffer_.substr(invoke)));
                    buffer_.clear();
                    phase_ = Phase::Content;
                }
                return events;
            }
            ToolCall call;
            call.id = generateToolCallId();
            call.name = buffer_.substr(invoke + kDsmlInvokePrefix.size(), name_end - invoke - kDsmlInvokePrefix.size());
            call.arguments = unwrapDsmlWrapperArguments(
                convertDsmlArguments(buffer_.substr(name_end + 2, args_end - name_end - 2)), tools_, call.name);
            events.push_back(Event::toolCallEvent(std::move(call)));
            dsml_emitted_ = true;
            dsml_scan_ = args_end + end_size;
        }
    }

    /** @brief Batch DSML parse through the exact streaming state machine. */
    static ToolCallParseResult parseDeepSeekV4Dsml(const std::string &text, const json &tools)
    {
        StreamingToolCallSplitter splitter(ToolCallFormat::DEEPSEEK_V4_DSML, tools);
        ToolCallParseResult result;
        auto fold = [&](std::vector<StreamingToolCallSplitter::Event> events) {
            for (auto &event : events)
                if (event.kind == StreamingToolCallSplitter::Event::Kind::ToolCall)
                    result.tool_calls.push_back(std::move(event.tool_call));
                else
                    result.content += event.content;
        };
        fold(splitter.process(text));
        fold(splitter.flush());
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

    ToolCallParseResult parseToolCalls(const std::string &text, ToolCallFormat format,
                                     const json &tools)
    {
        switch (format)
        {
        case ToolCallFormat::HERMES_2_PRO:
            return parseHermes2Pro(text);

        case ToolCallFormat::QWEN_3_XML:
            return parseQwen3Xml(text, tools);

        case ToolCallFormat::DEEPSEEK_V4_DSML:
            return parseDeepSeekV4Dsml(text, tools);

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

        case ToolCallFormat::DEEPSEEK_V4_DSML:
            return text.find(kDsmlInvokePrefix) != std::string::npos;

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

    StreamingToolCallSplitter::StreamingToolCallSplitter(ToolCallFormat format, json tools)
        : format_(format), tools_(std::move(tools))
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
        if (format_ == ToolCallFormat::DEEPSEEK_V4_DSML)
            return drainDeepSeekV4(terminal);
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

            const size_t close = format_ == ToolCallFormat::QWEN_3_XML
                ? qwenToolBlockEnd(buffer_, open_marker_.size())
                : buffer_.find(close_marker_, open_marker_.size());
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
            ToolCallParseResult parsed = parseToolCalls(block, format_, tools_);
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

    std::vector<std::string> toolSectionReasoningExits(ToolCallFormat format)
    {
        if (format != ToolCallFormat::DEEPSEEK_V4_DSML) return {};
        // vLLM's TOOL_START terminals: the canonical opener and the two
        // misspellings observed in production. A bare invoke is not a reasoning
        // exit there (it is only accepted from CONTENT).
        return {"<" + kDsml + "tool_calls>", "<" + kDsml + "toolcalls>", "<" + kDsml + "tool>"};
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
