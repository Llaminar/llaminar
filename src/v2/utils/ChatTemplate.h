/**
 * @file ChatTemplate.h
 * @brief Chat template detection and formatting for LLM chat interfaces
 * @author David Sanftenberg
 * @date 2025
 *
 * Provides chat template support for formatting multi-turn conversations
 * according to model-specific formats. Primary rendering uses the vendored
 * Jinja2 template engine (from llama.cpp) for proper template execution.
 * Falls back to hardcoded format implementations if Jinja rendering fails.
 */

#pragma once

#include <string>
#include <vector>
#include <memory>

namespace llaminar2
{

    /**
     * @brief Represents a single message in a chat conversation
     */
    struct ChatMessage
    {
        std::string role;    ///< Role: "system", "user", "assistant"
        std::string content; ///< Message content

        ChatMessage() = default;
        ChatMessage(const std::string &r, const std::string &c)
            : role(r), content(c) {}
    };

    /**
     * @brief Enumeration of supported chat template types
     *
     * These correspond to different model families and their expected
     * prompt formats for instruction-following.
     */
    enum class ChatTemplateType
    {
        CHATML,     ///< ChatML format: <|im_start|>role\ncontent<|im_end|> (Qwen, many others)
        LLAMA3,     ///< LLaMA 3: <|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|>
        LLAMA2,     ///< LLaMA 2: [INST] <<SYS>>system<</SYS>> user [/INST] assistant
        MISTRAL_V1, ///< Mistral v1: [INST] content [/INST]
        MISTRAL_V3, ///< Mistral v3: Similar but with tool support
        MISTRAL_V7, ///< Mistral v7: [SYSTEM_PROMPT]...[/SYSTEM_PROMPT] [INST]...[/INST]
        PHI3,       ///< Phi-3: <|role|>\ncontent<|end|>
        PHI4,       ///< Phi-4: <|im_start|>role<|im_sep|>content<|im_end|>
        GEMMA,      ///< Gemma: <start_of_turn>role\ncontent<end_of_turn>
        DEEPSEEK,   ///< DeepSeek: ### Instruction:\ncontent
        DEEPSEEK2,  ///< DeepSeek V2: User: content\n\nAssistant:
        DEEPSEEK3,  ///< DeepSeek V3: <｜User｜>content<｜Assistant｜>
        ZEPHYR,     ///< Zephyr: <|role|>\ncontent<|endoftext|>
        VICUNA,     ///< Vicuna: USER: content\nASSISTANT:
        COMMAND_R,  ///< Command-R: <|START_OF_TURN_TOKEN|><|USER_TOKEN|>content
        UNKNOWN     ///< Unknown format - will use raw prompt
    };

    /**
     * @brief Converts ChatTemplateType to human-readable string
     */
    const char *chatTemplateTypeName(ChatTemplateType type);

    /**
     * @brief Chat template processor
     *
     * Handles detection of chat template format from GGUF metadata and
     * application of templates to message sequences. Uses the vendored
     * Jinja2 template engine as the primary rendering path, with hardcoded
     * fallback implementations for robustness.
     *
     * Usage:
     * @code
     *   auto tmpl = ChatTemplate::create(gguf_template_string, "<|endoftext|>", "<|endoftext|>");
     *   std::vector<ChatMessage> messages = {
     *       {"system", "You are helpful."},
     *       {"user", "Hello!"}
     *   };
     *   std::string formatted = tmpl->apply(messages, true);
     *   // formatted now contains the properly templated prompt
     * @endcode
     */
    class ChatTemplate
    {
    public:
        /**
         * @brief Create a ChatTemplate from a template string with BOS/EOS tokens
         *
         * The template string is typically the `tokenizer.chat_template`
         * metadata value from a GGUF file (Jinja2 format). BOS/EOS tokens
         * are needed by the Jinja2 engine for template variables.
         *
         * @param template_str The raw template string from GGUF metadata
         * @param bos_token BOS token string (e.g., "<|endoftext|>")
         * @param eos_token EOS token string (e.g., "<|im_end|>")
         * @return Unique pointer to ChatTemplate, never null
         */
        static std::unique_ptr<ChatTemplate> create(const std::string &template_str,
                                                    const std::string &bos_token = "",
                                                    const std::string &eos_token = "");

        /**
         * @brief Create a ChatTemplate with explicit type override
         *
         * @param type The template type to use
         * @return Unique pointer to ChatTemplate
         */
        static std::unique_ptr<ChatTemplate> create(ChatTemplateType type);

        /**
         * @brief Destructor
         */
        ~ChatTemplate();

        /**
         * @brief Apply template to a sequence of messages
         *
         * Tries Jinja2 rendering first, falls back to hardcoded implementation.
         *
         * @param messages Vector of chat messages in conversation order
         * @param add_generation_prompt If true, adds the assistant prompt prefix
         *                              for the model to complete
         * @param enable_thinking If true, enable thinking mode for thinking models
         * @return Formatted prompt string ready for tokenization
         */
        std::string apply(const std::vector<ChatMessage> &messages,
                          bool add_generation_prompt = true,
                          bool enable_thinking = true) const;

        /**
         * @brief Get the detected template type
         */
        ChatTemplateType type() const { return type_; }

        /**
         * @brief Get the raw template string (if available)
         */
        const std::string &rawTemplate() const { return raw_template_; }

        /**
         * @brief Check if this is an unknown/fallback template
         */
        bool isUnknown() const { return type_ == ChatTemplateType::UNKNOWN; }

        /**
         * @brief Check if this is a thinking model (Qwen3.5, etc.)
         *
         * Detected automatically by rendering the template with and without
         * enable_thinking = true and checking if the output differs.
         */
        bool isThinkingModel() const { return is_thinking_model_; }

        /**
         * @brief Check if Jinja2 rendering is available for this template
         */
        bool hasJinjaSupport() const { return jinja_available_; }

        /**
         * @brief Get the detected thinking start tag (e.g., "<think>\n")
         *
         * Empty string if not a thinking model.
         */
        const std::string &thinkingStartTag() const { return thinking_start_tag_; }

        /**
         * @brief Get the detected thinking end tag (e.g., "</think>")
         *
         * Empty string if not a thinking model.
         */
        const std::string &thinkingEndTag() const { return thinking_end_tag_; }

    private:
        /**
         * @brief Private constructor - use create() factory methods
         */
        ChatTemplate(ChatTemplateType type, const std::string &raw_template,
                     const std::string &bos_token, const std::string &eos_token);

        /**
         * @brief Try to compile the Jinja2 template
         * @return true if compilation succeeded
         */
        bool compileJinja();

        /**
         * @brief Render using the Jinja2 engine
         * @return Rendered string, or empty on failure
         */
        std::string renderJinja(const std::vector<ChatMessage> &messages,
                                bool add_generation_prompt,
                                bool enable_thinking) const;

        /**
         * @brief Render without the enable_thinking variable in context
         *
         * Used for differential detection: makes `enable_thinking is defined`
         * evaluate to false, allowing us to detect thinking-model templates.
         */
        std::string renderJinjaWithoutThinkingVar(
            const std::vector<ChatMessage> &messages,
            bool add_generation_prompt) const;

        /**
         * @brief Auto-detect thinking tags by differential Jinja rendering
         *
         * Renders the template with enable_thinking=true and enable_thinking=false,
         * then diffs to find what changed — that's the thinking tags.
         */
        void detectThinkingTags();

        /**
         * @brief Detect template type from Jinja2 template string
         */
        static ChatTemplateType detectType(const std::string &template_str);

        // Template application methods for each format (fallback path)
        std::string applyFallback(const std::vector<ChatMessage> &messages, bool add_ass) const;
        std::string applyChatML(const std::vector<ChatMessage> &messages, bool add_ass) const;
        std::string applyLlama3(const std::vector<ChatMessage> &messages, bool add_ass) const;
        std::string applyLlama2(const std::vector<ChatMessage> &messages, bool add_ass) const;
        std::string applyMistralV1(const std::vector<ChatMessage> &messages, bool add_ass) const;
        std::string applyMistralV3(const std::vector<ChatMessage> &messages, bool add_ass) const;
        std::string applyMistralV7(const std::vector<ChatMessage> &messages, bool add_ass) const;
        std::string applyPhi3(const std::vector<ChatMessage> &messages, bool add_ass) const;
        std::string applyPhi4(const std::vector<ChatMessage> &messages, bool add_ass) const;
        std::string applyGemma(const std::vector<ChatMessage> &messages, bool add_ass) const;
        std::string applyDeepSeek(const std::vector<ChatMessage> &messages, bool add_ass) const;
        std::string applyDeepSeek2(const std::vector<ChatMessage> &messages, bool add_ass) const;
        std::string applyDeepSeek3(const std::vector<ChatMessage> &messages, bool add_ass) const;
        std::string applyZephyr(const std::vector<ChatMessage> &messages, bool add_ass) const;
        std::string applyVicuna(const std::vector<ChatMessage> &messages, bool add_ass) const;
        std::string applyCommandR(const std::vector<ChatMessage> &messages, bool add_ass) const;
        std::string applyUnknown(const std::vector<ChatMessage> &messages, bool add_ass) const;

        ChatTemplateType type_;
        std::string raw_template_;
        std::string bos_token_;
        std::string eos_token_;
        bool is_thinking_model_ = false;
        bool jinja_available_ = false;
        std::string thinking_start_tag_;
        std::string thinking_end_tag_;

        // Opaque pointer to compiled Jinja program (avoids jinja headers in public API)
        struct JinjaState;
        std::unique_ptr<JinjaState> jinja_state_;
    };

    /**
     * @brief Chat output parser for extracting structured content from model output
     *
     * Handles extraction of thinking/reasoning content from model-generated text.
     * Uses tag information from ChatTemplate (auto-detected via Jinja differential
     * rendering) to split output into reasoning and content parts.
     *
     * This is model-agnostic — the tags are detected from the actual template,
     * not hardcoded per model.
     *
     * Usage:
     * @code
     *   ChatParser parser(chat_template);
     *   auto result = parser.parse("reasoning here</think>\n\nactual answer");
     *   // result.reasoning_content == "reasoning here"
     *   // result.content == "actual answer"
     * @endcode
     */
    class ChatParser
    {
    public:
        /**
         * @brief Result of parsing model output
         */
        struct ParseResult
        {
            std::string content;            ///< Main response content
            std::string reasoning_content;  ///< Thinking/reasoning content (empty if none)
            bool has_reasoning = false;     ///< Whether reasoning was found
        };

        /**
         * @brief Construct parser from a ChatTemplate
         *
         * Extracts thinking tag information from the template's auto-detection.
         *
         * @param chat_template The chat template (must outlive the parser)
         */
        explicit ChatParser(const ChatTemplate &chat_template);

        /**
         * @brief Parse model output text into structured parts
         *
         * Extracts reasoning content delimited by thinking tags and returns
         * the remaining content separately.
         *
         * @param text Raw model output text
         * @return ParseResult with separated content and reasoning
         */
        ParseResult parse(const std::string &text) const;

        /**
         * @brief Check if this parser expects thinking content
         */
        bool expectsThinking() const { return !thinking_end_tag_.empty(); }

    private:
        std::string thinking_start_tag_;
        std::string thinking_end_tag_;
    };

} // namespace llaminar2
