/**
 * @file ChatTemplate.cpp
 * @brief Chat template detection and formatting implementation
 * @author David Sanftenberg
 * @date 2025
 *
 * Implements chat template detection from GGUF metadata strings and
 * format-specific prompt construction for various LLM families.
 * Primary rendering uses the vendored Jinja2 engine with fallback
 * to hardcoded format implementations.
 */

#include "ChatTemplate.h"
#include "Logger.h"
#include "vendor/jinja/lexer.h"
#include "vendor/jinja/parser.h"
#include "vendor/jinja/runtime.h"
#include "vendor/jinja/value.h"
#include <sstream>
#include <algorithm>
#include <nlohmann/json.hpp>

namespace llaminar2
{

    // ============================================================================
    // Jinja State (opaque, keeps jinja headers out of public API)
    // ============================================================================

    struct ChatTemplate::JinjaState
    {
        jinja::program prog;
        std::string source; // Preserved source for error tracing
    };

    // ============================================================================
    // Utility Functions
    // ============================================================================

    const char *chatTemplateTypeName(ChatTemplateType type)
    {
        switch (type)
        {
        case ChatTemplateType::CHATML:
            return "ChatML";
        case ChatTemplateType::LLAMA3:
            return "LLaMA 3";
        case ChatTemplateType::LLAMA2:
            return "LLaMA 2";
        case ChatTemplateType::MISTRAL_V1:
            return "Mistral v1";
        case ChatTemplateType::MISTRAL_V3:
            return "Mistral v3";
        case ChatTemplateType::MISTRAL_V7:
            return "Mistral v7";
        case ChatTemplateType::PHI3:
            return "Phi-3";
        case ChatTemplateType::PHI4:
            return "Phi-4";
        case ChatTemplateType::GEMMA:
            return "Gemma";
        case ChatTemplateType::DEEPSEEK:
            return "DeepSeek";
        case ChatTemplateType::DEEPSEEK2:
            return "DeepSeek V2";
        case ChatTemplateType::DEEPSEEK3:
            return "DeepSeek V3";
        case ChatTemplateType::ZEPHYR:
            return "Zephyr";
        case ChatTemplateType::VICUNA:
            return "Vicuna";
        case ChatTemplateType::COMMAND_R:
            return "Command-R";
        case ChatTemplateType::UNKNOWN:
        default:
            return "Unknown";
        }
    }

    static std::string trim(const std::string &str)
    {
        size_t start = 0;
        size_t end = str.size();
        while (start < end && std::isspace(static_cast<unsigned char>(str[start])))
        {
            start++;
        }
        while (end > start && std::isspace(static_cast<unsigned char>(str[end - 1])))
        {
            end--;
        }
        return str.substr(start, end - start);
    }

    // ============================================================================
    // ChatTemplate Implementation
    // ============================================================================

    ChatTemplate::ChatTemplate(ChatTemplateType type, const std::string &raw_template,
                               const std::string &bos_token, const std::string &eos_token)
        : type_(type), raw_template_(raw_template),
          bos_token_(bos_token), eos_token_(eos_token)
    {
        // Try to compile the Jinja2 template
        if (!raw_template_.empty())
        {
            jinja_available_ = compileJinja();
        }

        // Auto-detect thinking tags via differential Jinja rendering
        if (jinja_available_)
        {
            detectThinkingTags();
        }
        else
        {
            // Fallback: simple heuristic
            is_thinking_model_ = (raw_template_.find("enable_thinking") != std::string::npos);
            if (is_thinking_model_)
            {
                thinking_start_tag_ = "<think>\n";
                thinking_end_tag_ = "</think>";
            }
        }
    }

    ChatTemplate::~ChatTemplate() = default;

    std::unique_ptr<ChatTemplate> ChatTemplate::create(const std::string &template_str,
                                                       const std::string &bos_token,
                                                       const std::string &eos_token)
    {
        ChatTemplateType type = detectType(template_str);
        LOG_DEBUG("[ChatTemplate] Detected template type: " << chatTemplateTypeName(type)
                                                            << " (jinja will be attempted)");
        return std::unique_ptr<ChatTemplate>(
            new ChatTemplate(type, template_str, bos_token, eos_token));
    }

    std::unique_ptr<ChatTemplate> ChatTemplate::create(ChatTemplateType type)
    {
        return std::unique_ptr<ChatTemplate>(
            new ChatTemplate(type, "", "", ""));
    }

    // ============================================================================
    // Jinja2 Engine Integration
    // ============================================================================

    bool ChatTemplate::compileJinja()
    {
        try
        {
            jinja_state_ = std::make_unique<JinjaState>();
            jinja::lexer lexer;
            auto lexer_res = lexer.tokenize(raw_template_);
            jinja_state_->prog = jinja::parse_from_tokens(lexer_res);
            jinja_state_->source = lexer_res.source;
            LOG_DEBUG("[ChatTemplate] Jinja2 template compiled successfully");
            return true;
        }
        catch (const std::exception &e)
        {
            LOG_WARN("[ChatTemplate] Jinja2 compilation failed: " << e.what()
                                                                  << " — will use fallback renderer");
            jinja_state_.reset();
            return false;
        }
    }

    std::string ChatTemplate::renderJinja(const std::vector<ChatMessage> &messages,
                                          bool add_generation_prompt,
                                          bool enable_thinking,
                                          const std::string &tools_json) const
    {
        if (!jinja_state_)
            return {};

        try
        {
            // Re-parse the template fresh each call to avoid AST mutation bugs
            // in the vendored Jinja engine (execute_impl methods can modify shared
            // AST nodes, causing non-deterministic rendering on subsequent calls).
            jinja::lexer lexer;
            auto lexer_res = lexer.tokenize(raw_template_);
            jinja::program prog = jinja::parse_from_tokens(lexer_res);

            jinja::context ctx(jinja_state_->source);

            // Build the messages array as ordered JSON, including tool-calling fields
            nlohmann::ordered_json msg_array = nlohmann::ordered_json::array();
            for (const auto &msg : messages)
            {
                nlohmann::ordered_json jmsg;
                jmsg["role"] = msg.role;

                // content can be null for assistant messages with tool_calls
                if (msg.hasToolCalls() && msg.content.empty())
                    jmsg["content"] = nullptr;
                else
                    jmsg["content"] = msg.content;

                // Include tool_calls for assistant messages (parse from JSON strings)
                if (!msg.tool_calls.empty())
                {
                    jmsg["tool_calls"] = nlohmann::ordered_json::array();
                    for (const auto &tc_str : msg.tool_calls)
                    {
                        try
                        {
                            jmsg["tool_calls"].push_back(
                                nlohmann::ordered_json::parse(tc_str));
                        }
                        catch (const nlohmann::ordered_json::parse_error &)
                        {
                            // Skip malformed tool call entries
                        }
                    }
                }

                // Include tool_call_id for tool result messages
                if (!msg.tool_call_id.empty())
                    jmsg["tool_call_id"] = msg.tool_call_id;

                // Include name for tool result messages
                if (!msg.name.empty())
                    jmsg["name"] = msg.name;

                msg_array.push_back(jmsg);
            }

            // Build the context object with all standard template variables
            nlohmann::ordered_json inp;
            inp["messages"] = msg_array;
            inp["bos_token"] = bos_token_;
            inp["eos_token"] = eos_token_;
            if (add_generation_prompt)
            {
                inp["add_generation_prompt"] = true;
            }
            inp["enable_thinking"] = enable_thinking;

            // Pass tools array to Jinja context if provided
            if (!tools_json.empty())
            {
                try
                {
                    inp["tools"] = nlohmann::ordered_json::parse(tools_json);
                }
                catch (const nlohmann::ordered_json::parse_error &)
                {
                    // Invalid tools JSON — skip
                }
            }

            // Load context into jinja runtime
            jinja::global_from_json(ctx, inp, false);

            // Execute template
            jinja::runtime runtime(ctx);
            const jinja::value results = runtime.execute(prog);
            auto parts = jinja::runtime::gather_string_parts(results);

            return parts->as_string().str();
        }
        catch (const std::exception &e)
        {
            LOG_WARN("[ChatTemplate] Jinja2 render failed: " << e.what());
            return {};
        }
    }

    /**
     * @brief Render without passing enable_thinking at all (for differential detection)
     *
     * This makes `enable_thinking is defined` evaluate to false.
     */
    std::string ChatTemplate::renderJinjaWithoutThinkingVar(
        const std::vector<ChatMessage> &messages,
        bool add_generation_prompt) const
    {
        if (!jinja_state_)
            return {};

        try
        {
            // Re-parse template fresh (same rationale as renderJinja)
            jinja::lexer lexer;
            auto lexer_res = lexer.tokenize(raw_template_);
            jinja::program prog = jinja::parse_from_tokens(lexer_res);

            jinja::context ctx(jinja_state_->source);

            nlohmann::ordered_json msg_array = nlohmann::ordered_json::array();
            for (const auto &msg : messages)
            {
                nlohmann::ordered_json jmsg;
                jmsg["role"] = msg.role;
                jmsg["content"] = msg.content;
                msg_array.push_back(jmsg);
            }

            nlohmann::ordered_json inp;
            inp["messages"] = msg_array;
            inp["bos_token"] = bos_token_;
            inp["eos_token"] = eos_token_;
            if (add_generation_prompt)
            {
                inp["add_generation_prompt"] = true;
            }
            // Deliberately NOT setting enable_thinking

            jinja::global_from_json(ctx, inp, false);
            jinja::runtime runtime(ctx);
            const jinja::value results = runtime.execute(prog);
            auto parts = jinja::runtime::gather_string_parts(results);
            return parts->as_string().str();
        }
        catch (const std::exception &)
        {
            return {};
        }
    }

    void ChatTemplate::detectThinkingTags()
    {
        // Differential rendering approach (inspired by llama.cpp):
        // Render the generation prompt three ways and compare.
        //
        //   A) enable_thinking=true   (thinking active)
        //   B) enable_thinking=false  (thinking suppressed — templates often
        //                              emit an empty `<think></think>` block
        //                              here, which exposes both tags)
        //   C) enable_thinking unset  (pre-2025 Qwen / non-thinking templates
        //                              behave as if thinking were off)
        //
        // Historically we only compared (A) vs (C). That works for templates
        // whose internal default is "off when unset", but fails silently when
        // a template defaults `enable_thinking=true` in its own namespace
        // (e.g. the community Qwen 3.5 Jinja) — (A) and (C) render identically
        // and the detector incorrectly concludes the template is not a
        // thinking model. Adding the (A) vs (B) comparison fixes that class.
        std::vector<ChatMessage> test_messages = {
            {"user", "Hello"}};

        std::string with_thinking = renderJinja(test_messages, true, true);
        std::string no_thinking = renderJinja(test_messages, true, false);
        std::string without_thinking = renderJinjaWithoutThinkingVar(test_messages, true);

        LOG_DEBUG("[ChatTemplate] detectThinkingTags: with_thinking (" << with_thinking.size()
                                                                        << " chars), no_thinking ("
                                                                        << no_thinking.size()
                                                                        << " chars), without_thinking ("
                                                                        << without_thinking.size() << " chars)");

        if (with_thinking.empty())
        {
            return;
        }

        // Prefer whichever non-thinking rendering actually differs from the
        // thinking one. For modern Qwen 3.x templates this will be
        // no_thinking; for legacy ones it will be without_thinking.
        std::string reference;
        if (!no_thinking.empty() && no_thinking != with_thinking)
        {
            reference = no_thinking;
        }
        else if (!without_thinking.empty() && without_thinking != with_thinking)
        {
            reference = without_thinking;
        }
        else
        {
            // All three renderings are identical — template does not branch on
            // enable_thinking, so it is not a thinking model.
            is_thinking_model_ = false;
            return;
        }

        // Find the common prefix
        size_t common = 0;
        size_t min_len = std::min(with_thinking.size(), reference.size());
        while (common < min_len && with_thinking[common] == reference[common])
        {
            common++;
        }

        // The difference at the end tells us the thinking tags
        std::string thinking_suffix = with_thinking.substr(common);
        std::string non_thinking_suffix = reference.substr(common);

        // Helper: find a closing tag like </think> in a string
        auto find_close_tag = [](const std::string &s) -> std::string
        {
            auto close_pos = s.find("</");
            if (close_pos != std::string::npos)
            {
                auto end_pos = s.find('>', close_pos);
                if (end_pos != std::string::npos)
                {
                    return s.substr(close_pos, end_pos - close_pos + 1);
                }
            }
            return {};
        };

        // Helper: find last opening tag at end of string (e.g., "...<think>" → "<think>")
        auto find_trailing_open_tag = [](const std::string &s) -> std::string
        {
            // Look backward for '<' that isn't a closing tag
            for (size_t i = s.size(); i > 0; --i)
            {
                if (s[i - 1] == '>')
                {
                    // Found end of a tag, find its start
                    auto open = s.rfind('<', i - 1);
                    if (open != std::string::npos)
                    {
                        std::string tag = s.substr(open, i - open);
                        // Skip closing tags
                        if (tag.size() > 2 && tag[1] != '/')
                        {
                            return tag;
                        }
                    }
                    break;
                }
                // Allow trailing whitespace/newlines after the tag
                if (s[i - 1] != '\n' && s[i - 1] != ' ' && s[i - 1] != '\t')
                {
                    break;
                }
            }
            return {};
        };

        // Case 1: thinking_suffix is not empty — start tag is directly in the suffix
        if (!thinking_suffix.empty())
        {
            is_thinking_model_ = true;
            // Strip trailing whitespace to get clean start tag
            std::string stripped = thinking_suffix;
            while (!stripped.empty() && (stripped.back() == '\n' || stripped.back() == ' '))
                stripped.pop_back();
            thinking_start_tag_ = stripped;

            // Try to find end tag in non-thinking suffix
            std::string end_candidate = find_close_tag(non_thinking_suffix);
            if (!end_candidate.empty())
            {
                thinking_end_tag_ = end_candidate;
            }
        }
        // Case 2: thinking_suffix is empty but non_thinking_suffix has content
        // This happens when with_thinking is a prefix of without_thinking
        // e.g., with="...assistant<think>", without="...assistant<think>\n\n</think>"
        // The start tag is at the end of the common prefix, end tag is in the suffix
        else if (!non_thinking_suffix.empty())
        {
            // Look for end tag in the non-thinking suffix
            std::string end_candidate = find_close_tag(non_thinking_suffix);
            if (!end_candidate.empty())
            {
                thinking_end_tag_ = end_candidate;

                // Look for matching start tag at end of common prefix
                std::string common_prefix = with_thinking.substr(0, common);
                std::string start_candidate = find_trailing_open_tag(common_prefix);
                if (!start_candidate.empty())
                {
                    thinking_start_tag_ = start_candidate;
                    is_thinking_model_ = true;
                }
                else
                {
                    // Derive start tag from end tag: </think> → <think>
                    thinking_start_tag_ = "<" + end_candidate.substr(2);
                    is_thinking_model_ = true;
                }
            }
        }

        // If we have start but no end, derive end from start
        if (is_thinking_model_ && thinking_end_tag_.empty() && !thinking_start_tag_.empty())
        {
            std::string stripped = thinking_start_tag_;
            while (!stripped.empty() && (stripped.back() == '\n' || stripped.back() == ' '))
                stripped.pop_back();
            if (stripped.size() > 2 && stripped.front() == '<' && stripped.back() == '>')
            {
                thinking_end_tag_ = "</" + stripped.substr(1);
            }
        }

        if (is_thinking_model_)
        {

            LOG_INFO("[ChatTemplate] Thinking model detected via Jinja differential rendering"
                     << "\n  start_tag: \"" << thinking_start_tag_ << "\""
                     << "\n  end_tag: \"" << thinking_end_tag_ << "\"");
        }
    }

    ChatTemplateType ChatTemplate::detectType(const std::string &tmpl)
    {
        // Helper lambda to check if template contains a substring
        auto contains = [&tmpl](const char *needle) -> bool
        {
            return tmpl.find(needle) != std::string::npos;
        };

        // Detection logic based on llama.cpp's llm_chat_detect_template()
        // Order matters - more specific patterns first

        if (contains("<|im_start|>"))
        {
            // Could be ChatML, Phi-4, or SmolVLM
            if (contains("<|im_sep|>"))
            {
                return ChatTemplateType::PHI4;
            }
            // Default to ChatML for <|im_start|>
            return ChatTemplateType::CHATML;
        }

        if (contains("<|start_header_id|>") && contains("<|end_header_id|>"))
        {
            return ChatTemplateType::LLAMA3;
        }

        if (contains("[INST]"))
        {
            // Mistral or LLaMA 2 family
            // Check for LLaMA 2 first (has <<SYS>> marker)
            if (contains("<<SYS>>"))
            {
                return ChatTemplateType::LLAMA2;
            }
            if (contains("[SYSTEM_PROMPT]"))
            {
                return ChatTemplateType::MISTRAL_V7;
            }
            if (contains("[AVAILABLE_TOOLS]") || contains("\"[INST]\""))
            {
                return ChatTemplateType::MISTRAL_V3;
            }
            if (contains(" [INST]"))
            {
                return ChatTemplateType::MISTRAL_V1;
            }
            // Default to LLaMA 2 for plain [INST]
            return ChatTemplateType::LLAMA2;
        }

        if (contains("<|assistant|>") && contains("<|end|>"))
        {
            return ChatTemplateType::PHI3;
        }

        if (contains("<|user|>") && contains("<|endoftext|>"))
        {
            return ChatTemplateType::ZEPHYR;
        }

        if (contains("<start_of_turn>"))
        {
            return ChatTemplateType::GEMMA;
        }

        if (contains("### Instruction:") && contains("<|EOT|>"))
        {
            return ChatTemplateType::DEEPSEEK;
        }

        if (contains("'Assistant: ' + message['content'] + eos_token"))
        {
            return ChatTemplateType::DEEPSEEK2;
        }

        // DeepSeek V3 uses special Unicode characters
        if (contains("\xEF\xBD\x9C") && contains("Assistant"))
        { // ｜ character
            return ChatTemplateType::DEEPSEEK3;
        }

        if (contains("<|START_OF_TURN_TOKEN|>") && contains("<|USER_TOKEN|>"))
        {
            return ChatTemplateType::COMMAND_R;
        }

        if (contains("USER: ") && contains("ASSISTANT: "))
        {
            return ChatTemplateType::VICUNA;
        }

        LOG_WARN("[ChatTemplate] Could not detect template type, using UNKNOWN");
        return ChatTemplateType::UNKNOWN;
    }

    std::string ChatTemplate::apply(const std::vector<ChatMessage> &messages,
                                    bool add_generation_prompt,
                                    bool enable_thinking,
                                    const std::string &tools_json) const
    {
        // Primary path: Jinja2 rendering
        if (jinja_available_)
        {
            std::string result = renderJinja(messages, add_generation_prompt, enable_thinking, tools_json);
            if (!result.empty())
            {
                return result;
            }
            LOG_WARN("[ChatTemplate] Jinja2 render returned empty, falling back to hardcoded format");
        }

        // Fallback path: hardcoded format implementations (no tools support)
        return applyFallback(messages, add_generation_prompt);
    }

    std::string ChatTemplate::applyFallback(const std::vector<ChatMessage> &messages,
                                            bool add_generation_prompt) const
    {
        switch (type_)
        {
        case ChatTemplateType::CHATML:
            return applyChatML(messages, add_generation_prompt);
        case ChatTemplateType::LLAMA3:
            return applyLlama3(messages, add_generation_prompt);
        case ChatTemplateType::LLAMA2:
            return applyLlama2(messages, add_generation_prompt);
        case ChatTemplateType::MISTRAL_V1:
            return applyMistralV1(messages, add_generation_prompt);
        case ChatTemplateType::MISTRAL_V3:
            return applyMistralV3(messages, add_generation_prompt);
        case ChatTemplateType::MISTRAL_V7:
            return applyMistralV7(messages, add_generation_prompt);
        case ChatTemplateType::PHI3:
            return applyPhi3(messages, add_generation_prompt);
        case ChatTemplateType::PHI4:
            return applyPhi4(messages, add_generation_prompt);
        case ChatTemplateType::GEMMA:
            return applyGemma(messages, add_generation_prompt);
        case ChatTemplateType::DEEPSEEK:
            return applyDeepSeek(messages, add_generation_prompt);
        case ChatTemplateType::DEEPSEEK2:
            return applyDeepSeek2(messages, add_generation_prompt);
        case ChatTemplateType::DEEPSEEK3:
            return applyDeepSeek3(messages, add_generation_prompt);
        case ChatTemplateType::ZEPHYR:
            return applyZephyr(messages, add_generation_prompt);
        case ChatTemplateType::VICUNA:
            return applyVicuna(messages, add_generation_prompt);
        case ChatTemplateType::COMMAND_R:
            return applyCommandR(messages, add_generation_prompt);
        case ChatTemplateType::UNKNOWN:
        default:
            return applyUnknown(messages, add_generation_prompt);
        }
    }

    // ============================================================================
    // Template Application Methods
    // ============================================================================

    std::string ChatTemplate::applyChatML(const std::vector<ChatMessage> &messages,
                                          bool add_ass) const
    {
        // ChatML format: <|im_start|>role\ncontent<|im_end|>\n
        // Used by: Qwen, many instruction-tuned models
        std::stringstream ss;

        for (const auto &msg : messages)
        {
            ss << "<|im_start|>" << msg.role << "\n"
               << msg.content << "<|im_end|>\n";
        }

        if (add_ass)
        {
            ss << "<|im_start|>assistant\n";
            // Thinking models (Qwen3.5, etc.) require a <think> tag in the
            // generation prompt to trigger reasoning. Following llama.cpp,
            // we default to thinking-enabled mode. The response handler
            // strips <think>...</think> from content into reasoning_content.
            if (is_thinking_model_)
            {
                ss << "<think>\n";
            }
        }

        return ss.str();
    }

    std::string ChatTemplate::applyLlama3(const std::vector<ChatMessage> &messages,
                                          bool add_ass) const
    {
        // LLaMA 3 format: <|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|>
        std::stringstream ss;

        for (const auto &msg : messages)
        {
            ss << "<|start_header_id|>" << msg.role << "<|end_header_id|>\n\n"
               << trim(msg.content) << "<|eot_id|>";
        }

        if (add_ass)
        {
            ss << "<|start_header_id|>assistant<|end_header_id|>\n\n";
        }

        return ss.str();
    }

    std::string ChatTemplate::applyLlama2(const std::vector<ChatMessage> &messages,
                                          bool add_ass) const
    {
        // LLaMA 2 format with system message support
        // [INST] <<SYS>>\nsystem\n<</SYS>>\n\nuser [/INST] assistant </s>
        std::stringstream ss;
        bool is_inside_turn = true; // Skip BOS at beginning
        ss << "[INST] ";

        for (const auto &msg : messages)
        {
            if (!is_inside_turn)
            {
                is_inside_turn = true;
                ss << "[INST] ";
            }

            if (msg.role == "system")
            {
                ss << "<<SYS>>\n"
                   << msg.content << "\n<</SYS>>\n\n";
            }
            else if (msg.role == "user")
            {
                ss << msg.content << " [/INST]";
            }
            else if (msg.role == "assistant")
            {
                ss << msg.content << "</s>";
                is_inside_turn = false;
            }
        }

        return ss.str();
    }

    std::string ChatTemplate::applyMistralV1(const std::vector<ChatMessage> &messages,
                                             bool add_ass) const
    {
        // Mistral v1: [INST] content [/INST]
        std::stringstream ss;
        bool is_inside_turn = false;

        for (const auto &msg : messages)
        {
            if (!is_inside_turn)
            {
                ss << " [INST] ";
                is_inside_turn = true;
            }

            if (msg.role == "system")
            {
                ss << msg.content << "\n\n";
            }
            else if (msg.role == "user")
            {
                ss << msg.content << " [/INST]";
            }
            else if (msg.role == "assistant")
            {
                ss << " " << msg.content << "</s>";
                is_inside_turn = false;
            }
        }

        return ss.str();
    }

    std::string ChatTemplate::applyMistralV3(const std::vector<ChatMessage> &messages,
                                             bool add_ass) const
    {
        // Mistral v3: Similar to v1 but with different spacing
        std::stringstream ss;
        bool is_inside_turn = false;

        for (const auto &msg : messages)
        {
            if (!is_inside_turn)
            {
                ss << "[INST] ";
                is_inside_turn = true;
            }

            if (msg.role == "system")
            {
                ss << msg.content << "\n\n";
            }
            else if (msg.role == "user")
            {
                ss << msg.content << "[/INST]";
            }
            else if (msg.role == "assistant")
            {
                ss << " " << trim(msg.content) << "</s>";
                is_inside_turn = false;
            }
        }

        return ss.str();
    }

    std::string ChatTemplate::applyMistralV7(const std::vector<ChatMessage> &messages,
                                             bool add_ass) const
    {
        // Mistral v7: [SYSTEM_PROMPT] content [/SYSTEM_PROMPT] [INST] content [/INST]
        std::stringstream ss;

        for (const auto &msg : messages)
        {
            if (msg.role == "system")
            {
                ss << "[SYSTEM_PROMPT] " << msg.content << "[/SYSTEM_PROMPT]";
            }
            else if (msg.role == "user")
            {
                ss << "[INST] " << msg.content << "[/INST]";
            }
            else if (msg.role == "assistant")
            {
                ss << " " << msg.content << "</s>";
            }
        }

        return ss.str();
    }

    std::string ChatTemplate::applyPhi3(const std::vector<ChatMessage> &messages,
                                        bool add_ass) const
    {
        // Phi-3: <|role|>\ncontent<|end|>\n
        std::stringstream ss;

        for (const auto &msg : messages)
        {
            ss << "<|" << msg.role << "|>\n"
               << msg.content << "<|end|>\n";
        }

        if (add_ass)
        {
            ss << "<|assistant|>\n";
        }

        return ss.str();
    }

    std::string ChatTemplate::applyPhi4(const std::vector<ChatMessage> &messages,
                                        bool add_ass) const
    {
        // Phi-4: <|im_start|>role<|im_sep|>content<|im_end|>
        std::stringstream ss;

        for (const auto &msg : messages)
        {
            ss << "<|im_start|>" << msg.role << "<|im_sep|>"
               << msg.content << "<|im_end|>";
        }

        if (add_ass)
        {
            ss << "<|im_start|>assistant<|im_sep|>";
        }

        return ss.str();
    }

    std::string ChatTemplate::applyGemma(const std::vector<ChatMessage> &messages,
                                         bool add_ass) const
    {
        // Gemma: <start_of_turn>role\ncontent<end_of_turn>\n
        // Note: Gemma uses "model" instead of "assistant"
        std::stringstream ss;
        std::string system_prompt;

        for (const auto &msg : messages)
        {
            if (msg.role == "system")
            {
                // Gemma doesn't have system role, merge with first user message
                system_prompt = trim(msg.content);
                continue;
            }

            std::string role = (msg.role == "assistant") ? "model" : msg.role;
            ss << "<start_of_turn>" << role << "\n";

            if (!system_prompt.empty() && role != "model")
            {
                ss << system_prompt << "\n\n";
                system_prompt.clear();
            }

            ss << trim(msg.content) << "<end_of_turn>\n";
        }

        if (add_ass)
        {
            ss << "<start_of_turn>model\n";
        }

        return ss.str();
    }

    std::string ChatTemplate::applyDeepSeek(const std::vector<ChatMessage> &messages,
                                            bool add_ass) const
    {
        // DeepSeek coder: ### Instruction:\ncontent\n### Response:\ncontent<|EOT|>
        std::stringstream ss;

        for (const auto &msg : messages)
        {
            if (msg.role == "system")
            {
                ss << msg.content;
            }
            else if (msg.role == "user")
            {
                ss << "### Instruction:\n"
                   << msg.content << "\n";
            }
            else if (msg.role == "assistant")
            {
                ss << "### Response:\n"
                   << msg.content << "\n<|EOT|>\n";
            }
        }

        if (add_ass)
        {
            ss << "### Response:\n";
        }

        return ss.str();
    }

    std::string ChatTemplate::applyDeepSeek2(const std::vector<ChatMessage> &messages,
                                             bool add_ass) const
    {
        // DeepSeek V2: User: content\n\nAssistant: content<｜end▁of▁sentence｜>
        std::stringstream ss;
        // UTF-8 encoding for special characters
        const char *eos_marker = "\xEF\xBD\x9C"
                                 "end"
                                 "\xE2\x96\x81"
                                 "of"
                                 "\xE2\x96\x81"
                                 "sentence"
                                 "\xEF\xBD\x9C"; // ｜end▁of▁sentence｜

        for (const auto &msg : messages)
        {
            if (msg.role == "system")
            {
                ss << msg.content << "\n\n";
            }
            else if (msg.role == "user")
            {
                ss << "User: " << msg.content << "\n\n";
            }
            else if (msg.role == "assistant")
            {
                ss << "Assistant: " << msg.content << eos_marker;
            }
        }

        if (add_ass)
        {
            ss << "Assistant:";
        }

        return ss.str();
    }

    std::string ChatTemplate::applyDeepSeek3(const std::vector<ChatMessage> &messages,
                                             bool add_ass) const
    {
        // DeepSeek V3: <｜User｜>content<｜Assistant｜>content<｜end▁of▁sentence｜>
        std::stringstream ss;
        // UTF-8 encoding for special characters
        const char *user_marker = "\xEF\xBD\x9C"
                                  "User"
                                  "\xEF\xBD\x9C"; // ｜User｜
        const char *assistant_marker = "\xEF\xBD\x9C"
                                       "Assistant"
                                       "\xEF\xBD\x9C"; // ｜Assistant｜
        const char *eos_marker = "\xEF\xBD\x9C"
                                 "end"
                                 "\xE2\x96\x81"
                                 "of"
                                 "\xE2\x96\x81"
                                 "sentence"
                                 "\xEF\xBD\x9C"; // ｜end▁of▁sentence｜

        for (const auto &msg : messages)
        {
            if (msg.role == "system")
            {
                ss << msg.content << "\n\n";
            }
            else if (msg.role == "user")
            {
                ss << user_marker << msg.content;
            }
            else if (msg.role == "assistant")
            {
                ss << assistant_marker << msg.content << eos_marker;
            }
        }

        if (add_ass)
        {
            ss << assistant_marker;
        }

        return ss.str();
    }

    std::string ChatTemplate::applyZephyr(const std::vector<ChatMessage> &messages,
                                          bool add_ass) const
    {
        // Zephyr: <|role|>\ncontent<|endoftext|>\n
        std::stringstream ss;

        for (const auto &msg : messages)
        {
            ss << "<|" << msg.role << "|>\n"
               << msg.content << "<|endoftext|>\n";
        }

        if (add_ass)
        {
            ss << "<|assistant|>\n";
        }

        return ss.str();
    }

    std::string ChatTemplate::applyVicuna(const std::vector<ChatMessage> &messages,
                                          bool add_ass) const
    {
        // Vicuna: USER: content\nASSISTANT: content</s>
        std::stringstream ss;

        for (const auto &msg : messages)
        {
            if (msg.role == "system")
            {
                ss << msg.content << "\n\n";
            }
            else if (msg.role == "user")
            {
                ss << "USER: " << msg.content << "\n";
            }
            else if (msg.role == "assistant")
            {
                ss << "ASSISTANT: " << msg.content << "</s>\n";
            }
        }

        if (add_ass)
        {
            ss << "ASSISTANT:";
        }

        return ss.str();
    }

    std::string ChatTemplate::applyCommandR(const std::vector<ChatMessage> &messages,
                                            bool add_ass) const
    {
        // Command-R: <|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>content<|END_OF_TURN_TOKEN|>
        std::stringstream ss;

        for (const auto &msg : messages)
        {
            ss << "<|START_OF_TURN_TOKEN|>";
            if (msg.role == "system")
            {
                ss << "<|SYSTEM_TOKEN|>" << trim(msg.content);
            }
            else if (msg.role == "user")
            {
                ss << "<|USER_TOKEN|>" << trim(msg.content);
            }
            else if (msg.role == "assistant")
            {
                ss << "<|CHATBOT_TOKEN|>" << trim(msg.content);
            }
            ss << "<|END_OF_TURN_TOKEN|>";
        }

        if (add_ass)
        {
            ss << "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>";
        }

        return ss.str();
    }

    std::string ChatTemplate::applyUnknown(const std::vector<ChatMessage> &messages,
                                           bool add_ass) const
    {
        // Fallback: Just concatenate content with basic role prefixes
        std::stringstream ss;

        for (const auto &msg : messages)
        {
            if (msg.role == "system")
            {
                ss << "System: " << msg.content << "\n\n";
            }
            else if (msg.role == "user")
            {
                ss << "User: " << msg.content << "\n\n";
            }
            else if (msg.role == "assistant")
            {
                ss << "Assistant: " << msg.content << "\n\n";
            }
        }

        if (add_ass)
        {
            ss << "Assistant: ";
        }

        return ss.str();
    }

    // ============================================================================
    // ChatParser Implementation
    // ============================================================================

    ChatParser::ChatParser(const ChatTemplate &chat_template)
        : thinking_start_tag_(chat_template.thinkingStartTag()),
          thinking_end_tag_(chat_template.thinkingEndTag())
    {
    }

    ChatParser::ParseResult ChatParser::parse(const std::string &text) const
    {
        ParseResult result;

        if (thinking_end_tag_.empty() || text.empty())
        {
            result.content = text;
            return result;
        }

        // The generation prompt ends with the thinking start tag, so the
        // model output begins immediately inside the thinking block.
        // We search for the end tag to split reasoning from content.
        auto end_pos = text.find(thinking_end_tag_);
        if (end_pos != std::string::npos)
        {
            result.has_reasoning = true;
            result.reasoning_content = text.substr(0, end_pos);

            // Content starts after the end tag
            std::string remainder = text.substr(end_pos + thinking_end_tag_.size());

            // Trim leading whitespace from content
            auto first_non_ws = remainder.find_first_not_of(" \t\n\r");
            if (first_non_ws != std::string::npos)
            {
                result.content = remainder.substr(first_non_ws);
            }
            // If all whitespace after end tag, content is empty
        }
        else
        {
            // No end tag found — model produced thinking but never closed it
            // (e.g. hit max_tokens while still in <think> block).
            // Since the generation prompt ends with the start tag, everything
            // the model wrote is reasoning content, not final content.
            result.has_reasoning = true;
            result.reasoning_content = text;
        }

        return result;
    }

} // namespace llaminar2
