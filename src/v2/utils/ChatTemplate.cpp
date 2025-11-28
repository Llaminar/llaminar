/**
 * @file ChatTemplate.cpp
 * @brief Chat template detection and formatting implementation
 * @author David Sanftenberg
 * @date 2025
 *
 * Implements chat template detection from GGUF metadata strings and
 * format-specific prompt construction for various LLM families.
 */

#include "ChatTemplate.h"
#include "Logger.h"
#include <sstream>
#include <algorithm>

namespace llaminar2
{

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

    ChatTemplate::ChatTemplate(ChatTemplateType type, const std::string &raw_template)
        : type_(type), raw_template_(raw_template)
    {
    }

    std::unique_ptr<ChatTemplate> ChatTemplate::create(const std::string &template_str)
    {
        ChatTemplateType type = detectType(template_str);
        LOG_DEBUG("[ChatTemplate] Detected template type: " << chatTemplateTypeName(type));
        return std::unique_ptr<ChatTemplate>(new ChatTemplate(type, template_str));
    }

    std::unique_ptr<ChatTemplate> ChatTemplate::create(ChatTemplateType type)
    {
        return std::unique_ptr<ChatTemplate>(new ChatTemplate(type, ""));
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

} // namespace llaminar2
