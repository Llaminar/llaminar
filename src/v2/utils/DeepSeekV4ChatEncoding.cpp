/**
 * @file DeepSeekV4ChatEncoding.cpp
 * @brief Line-by-line port of vLLM's DeepSeek-V4 encoder (encode_messages and helpers).
 *
 * Pipeline, identical to the reference:
 *   merge_consecutive_assistant_messages -> merge_tool_messages ->
 *   sort_tool_results_by_call_order -> (thinking, no tools) _drop_thinking_messages
 *   -> render_message for every remaining message.
 * Python truthiness is reproduced explicitly: a field is "present" when it is
 * a non-null, non-empty string/array/object. Function names keep the
 * reference names so a reviewer can diff the two implementations.
 */
#include "DeepSeekV4ChatEncoding.h"
#include "PythonJson.h"

#include <algorithm>
#include <map>
#include <stdexcept>

namespace llaminar2::deepseek4
{
    namespace
    {
        using ojson = nlohmann::ordered_json;

        const std::string kUser = "<｜User｜>";
        const std::string kAssistant = "<｜Assistant｜>";
        const std::string kLatestReminder = "<｜latest_reminder｜>";
        const std::string kDsml = kDsmlToken;

        const std::string kEffortHigh =
            "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n"
            "You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.\n"
            "Explicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\n\n";
        const std::string kEffortMax =
            "Reasoning Effort: Beyond maximum — exhaustive, relentless, and uncompromising.\n"
            "You MUST reason with the utmost depth and rigor, leaving absolutely nothing to chance: exhaustively decompose the problem into its most fundamental components, trace every causal chain to its root, and resolve the underlying cause rather than any surface symptom.\n"
            "Do not stop reasoning until you have independently verified the solution from multiple angles and are certain that no assumption remains unchecked and no error remains undiscovered.\n\n";

        /** @brief TOOLS_TEMPLATE with dsml/thinking tokens substituted; schemas go between the halves. */
        const std::string kToolsHeader =
            "## Tools\n\n"
            "You have access to a set of tools to help answer the user's question. You can invoke tools by writing a \"<" + kDsml + "tool_calls>\" block like the following:\n\n"
            "<" + kDsml + "tool_calls>\n"
            "<" + kDsml + "invoke name=\"$TOOL_NAME\">\n"
            "<" + kDsml + "parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</" + kDsml + "parameter>\n"
            "...\n"
            "</" + kDsml + "invoke>\n"
            "<" + kDsml + "invoke name=\"$TOOL_NAME2\">\n"
            "...\n"
            "</" + kDsml + "invoke>\n"
            "</" + kDsml + "tool_calls>\n\n"
            "String parameters should be specified as is and set `string=\"true\"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string=\"false\"`.\n\n"
            "If thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.\n\n"
            "Otherwise, output directly after </think> with tool calls or final response.\n\n"
            "### Available Tool Schemas\n\n";
        const std::string kToolsFooter =
            "\n\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.\n";

        /** @brief Python truthiness of an optional JSON member. */
        bool truthy(const ojson &message, const char *key)
        {
            if (!message.is_object() || !message.contains(key)) return false;
            const auto &v = message[key];
            if (v.is_null()) return false;
            if (v.is_string()) return !v.get_ref<const std::string &>().empty();
            if (v.is_array() || v.is_object()) return !v.empty();
            if (v.is_boolean()) return v.get<bool>();
            return true;
        }

        std::string roleOf(const ojson &message)
        {
            return message.contains("role") && message["role"].is_string() ? message["role"].get<std::string>() : "";
        }

        /** @brief flatten_content_blocks: OpenAI text parts joined by a blank line; strings pass through. */
        std::string flattenContent(const ojson &content)
        {
            if (content.is_null()) return {};
            if (content.is_string()) return content.get<std::string>();
            if (!content.is_array())
                throw std::invalid_argument("DeepSeek-V4 encoder: content must be a string, null or an array of parts");
            std::string out;
            bool first = true;
            for (const auto &block : content)
            {
                if (!first) out += "\n\n";
                first = false;
                if (block.is_string()) { out += block.get<std::string>(); continue; }
                const std::string type = block.is_object() && block.contains("type") && block["type"].is_string()
                                             ? block["type"].get<std::string>() : "";
                if (type == "text")
                    out += block.contains("text") && block["text"].is_string() ? block["text"].get<std::string>() : "";
                else
                    throw std::invalid_argument("DeepSeek-V4 encoder: unsupported content block type '" + type + "'");
            }
            return out;
        }

        std::string contentOf(const ojson &message)
        {
            return message.contains("content") ? flattenContent(message["content"]) : std::string{};
        }

        std::string renderTools(const ojson &tools)
        {
            std::string schemas;
            bool first = true;
            for (const auto &tool : tools)
            {
                if (!tool.contains("function"))
                    throw std::invalid_argument("DeepSeek-V4 encoder: tools must use the OpenAI {\"function\": ...} form");
                if (!first) schemas += "\n";
                first = false;
                schemas += pythonJsonDumps(tool["function"]);
            }
            return kToolsHeader + schemas + kToolsFooter;
        }

        /** @brief encode_arguments_to_dsml for one OpenAI tool call. */
        std::string encodeArguments(const ojson &function)
        {
            ojson arguments = function.contains("arguments") ? function["arguments"] : ojson::object();
            if (arguments.is_string())
            {
                arguments = ojson::parse(arguments.get<std::string>(), nullptr, false);
                if (arguments.is_discarded())
                    throw std::invalid_argument("DeepSeek-V4 encoder: tool-call arguments are not valid JSON");
            }
            if (!arguments.is_object())
                throw std::invalid_argument("DeepSeek-V4 encoder: tool-call arguments must be a JSON object");
            std::string out;
            bool first = true;
            for (const auto &[key, value] : arguments.items())
            {
                if (!first) out += "\n";
                first = false;
                const bool is_string = value.is_string();
                out += "<" + kDsml + "parameter name=\"" + key + "\" string=\"" + (is_string ? "true" : "false") + "\">";
                out += is_string ? value.get<std::string>() : pythonJsonDumps(value);
                out += "</" + kDsml + "parameter>";
            }
            return out;
        }

        std::string toolCallsBlock(const ojson &tool_calls)
        {
            std::string calls;
            bool first = true;
            for (const auto &call : tool_calls)
            {
                if (!call.contains("function") || !call["function"].contains("name"))
                    throw std::invalid_argument("DeepSeek-V4 encoder: assistant tool_calls require function.name");
                const auto &function = call["function"];
                if (!first) calls += "\n";
                first = false;
                calls += "<" + kDsml + "invoke name=\"" + function["name"].get<std::string>() + "\">\n" +
                         encodeArguments(function) + "\n</" + kDsml + "invoke>";
            }
            return "\n\n<" + kDsml + "tool_calls>\n" + calls + "\n</" + kDsml + "tool_calls>";
        }

        /** @brief _merge_assistant_run. */
        ojson mergeAssistantRun(const std::vector<ojson> &run)
        {
            if (run.size() == 1) return run.front();
            ojson combined = run.front();
            std::vector<std::string> contents, reasonings;
            ojson tool_calls = ojson::array();
            for (const auto &m : run)
            {
                if (truthy(m, "content")) contents.push_back(contentOf(m));
                if (truthy(m, "reasoning")) reasonings.push_back(m["reasoning"].get<std::string>());
                if (m.contains("tool_calls") && m["tool_calls"].is_array())
                    for (const auto &call : m["tool_calls"]) tool_calls.push_back(call);
            }
            const auto join = [](const std::vector<std::string> &parts) {
                std::string out;
                for (std::size_t i = 0; i < parts.size(); ++i) out += (i ? "\n\n" : "") + parts[i];
                return out;
            };
            if (!contents.empty()) combined["content"] = join(contents);
            else combined.erase("content");
            if (!reasonings.empty()) combined["reasoning"] = join(reasonings);
            else combined.erase("reasoning");
            if (!tool_calls.empty()) combined["tool_calls"] = tool_calls;
            else combined.erase("tool_calls");
            return combined;
        }

        ojson mergeConsecutiveAssistantMessages(const ojson &messages)
        {
            ojson merged = ojson::array();
            std::vector<ojson> run;
            for (const auto &message : messages)
            {
                if (roleOf(message) == "assistant") { run.push_back(message); continue; }
                if (!run.empty()) { merged.push_back(mergeAssistantRun(run)); run.clear(); }
                merged.push_back(message);
            }
            if (!run.empty()) merged.push_back(mergeAssistantRun(run));
            return merged;
        }

        bool isMergeableUser(const ojson &merged)
        {
            return !merged.empty() && roleOf(merged.back()) == "user" && merged.back().contains("content_blocks");
        }

        /** @brief merge_tool_messages: tool results become tool_result blocks of a user turn. */
        ojson mergeToolMessages(const ojson &messages)
        {
            ojson merged = ojson::array();
            for (const auto &message : messages)
            {
                const auto role = roleOf(message);
                if (role == "tool")
                {
                    ojson block = {{"type", "tool_result"},
                                   {"tool_use_id", message.contains("tool_call_id") && message["tool_call_id"].is_string()
                                                       ? message["tool_call_id"] : ojson("")},
                                   {"content", contentOf(message)}};
                    if (isMergeableUser(merged)) merged.back()["content_blocks"].push_back(block);
                    else merged.push_back({{"role", "user"}, {"content_blocks", ojson::array({block})}});
                }
                else if (role == "user")
                {
                    const std::string text = contentOf(message);
                    ojson block = {{"type", "text"}, {"text", text}};
                    if (isMergeableUser(merged)) merged.back()["content_blocks"].push_back(block);
                    else merged.push_back({{"role", "user"}, {"content", text}, {"content_blocks", ojson::array({block})}});
                }
                else
                {
                    merged.push_back(message);
                }
            }
            return merged;
        }

        /** @brief sort_tool_results_by_call_order (stable; unknown ids sort as call index 0). */
        void sortToolResultsByCallOrder(ojson &messages)
        {
            std::map<std::string, std::size_t> order;
            for (auto &message : messages)
            {
                const auto role = roleOf(message);
                if (role == "assistant" && truthy(message, "tool_calls"))
                {
                    order.clear();
                    std::size_t index = 0;
                    for (const auto &call : message["tool_calls"])
                    {
                        std::string id;
                        if (call.contains("id") && call["id"].is_string()) id = call["id"].get<std::string>();
                        if (id.empty() && call.contains("function") && call["function"].contains("id") &&
                            call["function"]["id"].is_string())
                            id = call["function"]["id"].get<std::string>();
                        if (!id.empty()) order[id] = index;
                        ++index;
                    }
                }
                else if (role == "user" && truthy(message, "content_blocks"))
                {
                    std::vector<ojson> tool_blocks;
                    for (const auto &block : message["content_blocks"])
                        if (block.value("type", "") == "tool_result") tool_blocks.push_back(block);
                    if (tool_blocks.size() <= 1 || order.empty()) continue;
                    const auto rank = [&](const ojson &block) {
                        const auto it = order.find(block.value("tool_use_id", ""));
                        return it == order.end() ? std::size_t{0} : it->second;
                    };
                    std::stable_sort(tool_blocks.begin(), tool_blocks.end(),
                                     [&](const ojson &a, const ojson &b) { return rank(a) < rank(b); });
                    std::size_t next = 0;
                    for (auto &block : message["content_blocks"])
                        if (block.value("type", "") == "tool_result") block = tool_blocks[next++];
                }
            }
        }

        int findLastUserIndex(const ojson &messages)
        {
            for (int i = static_cast<int>(messages.size()) - 1; i >= 0; --i)
            {
                const auto role = roleOf(messages[static_cast<std::size_t>(i)]);
                if (role == "user" || role == "developer") return i;
            }
            return -1;
        }

        /** @brief _drop_thinking_messages. */
        ojson dropThinkingMessages(const ojson &messages)
        {
            const int last_user = findLastUserIndex(messages);
            ojson result = ojson::array();
            for (int i = 0; i < static_cast<int>(messages.size()); ++i)
            {
                const auto &message = messages[static_cast<std::size_t>(i)];
                const auto role = roleOf(message);
                const bool keep_role = role == "user" || role == "system" || role == "tool" ||
                                       role == "latest_reminder" || role == "direct_search_results";
                if (keep_role || i >= last_user) result.push_back(message);
                else if (role == "assistant")
                {
                    ojson copy = message;
                    copy.erase("reasoning");
                    result.push_back(copy);
                }
                // Developer turns (and unknown roles) before the last user turn are dropped.
            }
            return result;
        }

        std::string renderUserBlocks(const ojson &blocks)
        {
            std::string out;
            bool first = true;
            for (const auto &block : blocks)
            {
                if (!first) out += "\n\n";
                first = false;
                const std::string type = block.value("type", "");
                if (type == "text") out += block.value("text", "");
                else if (type == "tool_result") out += "<tool_result>" + block.value("content", "") + "</tool_result>";
                else out += "[Unsupported " + type + "]";
            }
            return out;
        }

        /** @brief render_message for message @p index of the preprocessed conversation. */
        std::string renderMessage(std::size_t index, const ojson &messages, ChatPromptPolicy policy, bool drop_thinking,
                                  int last_user, bool add_generation_prompt)
        {
            const bool thinking = policy.thinking == ChatThinkingMode::Thinking;
            const auto &message = messages[index];
            const auto role = roleOf(message);
            const std::string content = contentOf(message);
            std::string prompt;

            if (index == 0 && thinking)
                prompt += policy.effort == ChatReasoningEffort::High ? kEffortHigh
                        : policy.effort == ChatReasoningEffort::Max  ? kEffortMax : std::string{};

            if (role == "system")
            {
                prompt += content;
                if (truthy(message, "tools")) prompt += "\n\n" + renderTools(message["tools"]);
            }
            else if (role == "developer")
            {
                if (content.empty())
                    throw std::invalid_argument("DeepSeek-V4 encoder: developer messages require content");
                prompt += kUser + content;
                if (truthy(message, "tools")) prompt += "\n\n" + renderTools(message["tools"]);
            }
            else if (role == "user")
            {
                prompt += kUser;
                prompt += truthy(message, "content_blocks") ? renderUserBlocks(message["content_blocks"]) : content;
            }
            else if (role == "latest_reminder")
            {
                prompt += kLatestReminder + content;
            }
            else if (role == "assistant")
            {
                std::string thinking_part;
                if (thinking && (!drop_thinking || static_cast<int>(index) > last_user))
                    thinking_part = (truthy(message, "reasoning") ? message["reasoning"].get<std::string>() : "") +
                                    kThinkEnd;
                prompt += thinking_part + content;
                if (truthy(message, "tool_calls")) prompt += toolCallsBlock(message["tool_calls"]);
                prompt += kEosToken;
            }
            else
            {
                throw std::invalid_argument("DeepSeek-V4 encoder: invalid role '" + role + "'");
            }

            // Transition tokens depend on what follows.
            const bool last = index + 1 == messages.size();
            if (!last)
            {
                const auto next = roleOf(messages[index + 1]);
                if (next != "assistant" && next != "latest_reminder") return prompt;
            }
            const bool system_opens = role == "system" && (last || roleOf(messages[index + 1]) == "assistant");
            if (role == "user" || role == "developer" || system_opens)
            {
                if (!last || add_generation_prompt)
                {
                    prompt += kAssistant;
                    if (!drop_thinking && thinking) prompt += kThinkStart;
                    else if (drop_thinking && thinking && static_cast<int>(index) >= last_user) prompt += kThinkStart;
                    else prompt += kThinkEnd;
                }
            }
            else if (add_generation_prompt && last && role == "assistant")
            {
                prompt += kAssistant + (thinking ? std::string(kThinkStart) : std::string(kThinkEnd));
            }
            return prompt;
        }
    } // namespace

    ChatPromptPolicy resolveChatPromptPolicy(bool thinking_enabled, const std::optional<std::string> &reasoning_effort)
    {
        ChatPromptPolicy policy;
        policy.thinking = thinking_enabled ? ChatThinkingMode::Thinking : ChatThinkingMode::Chat;
        if (!reasoning_effort)
        {
            policy.effort = thinking_enabled ? ChatReasoningEffort::High : ChatReasoningEffort::Low;
            return policy;
        }
        const auto &effort = *reasoning_effort;
        if (effort == "none")
        {
            policy.thinking = ChatThinkingMode::Chat;
            policy.effort = ChatReasoningEffort::Low;
        }
        else if (effort == "max") policy.effort = ChatReasoningEffort::Max;
        else if (effort == "low" || effort == "minimal" || effort == "medium") policy.effort = ChatReasoningEffort::Low;
        else policy.effort = ChatReasoningEffort::High;
        return policy;
    }

    void attachRequestTools(nlohmann::ordered_json &messages, const nlohmann::ordered_json &tools)
    {
        if (!tools.is_array() || tools.empty()) return;
        for (auto &message : messages)
            if (roleOf(message) == "system")
            {
                message["tools"] = tools;
                return;
            }
        messages.insert(messages.begin(), ojson{{"role", "system"}, {"tools", tools}});
    }

    std::string encodeChatMessages(const nlohmann::ordered_json &input, ChatPromptPolicy policy,
                                   bool add_generation_prompt)
    {
        if (!input.is_array())
            throw std::invalid_argument("DeepSeek-V4 encoder: messages must be an array");
        ojson messages = mergeToolMessages(mergeConsecutiveAssistantMessages(input));
        sortToolResultsByCallOrder(messages);

        // Tools anywhere keep every turn's reasoning (the model was trained to
        // see its earlier tool-use deliberation); otherwise only the reasoning
        // after the last user turn survives.
        const bool drop_thinking = std::none_of(messages.begin(), messages.end(),
                                                [](const ojson &m) { return truthy(m, "tools"); });
        if (policy.thinking == ChatThinkingMode::Thinking && drop_thinking)
            messages = dropThinkingMessages(messages);
        const int last_user = findLastUserIndex(messages);

        std::string prompt = kBosToken;
        for (std::size_t i = 0; i < messages.size(); ++i)
            prompt += renderMessage(i, messages, policy, drop_thinking, last_user, add_generation_prompt);
        return prompt;
    }
} // namespace llaminar2::deepseek4
