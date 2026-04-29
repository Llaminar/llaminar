/**
 * @file Tokenizer.h
 * @brief Token encoding/decoding for LLM text generation
 * @author David Sanftenberg
 * @date 2025
 *
 * Native tokenizer implementation that reads vocabulary directly from
 * GGUF metadata, avoiding duplicate model loading.
 */

#pragma once

#include "ChatTemplate.h"
#include <vector>
#include <string>
#include <memory>
#include <cstdint>
#include <unordered_map>
#include <map>

namespace llaminar2
{
    // Forward declarations
    class ModelContext;
    struct GGUFValue;

    /**
     * @brief Abstract tokenizer interface
     *
     * Provides encode/decode operations for any model architecture.
     * Implementations handle architecture-specific details.
     */
    class ITokenizer
    {
    public:
        virtual ~ITokenizer() = default;

        /**
         * @brief Encode text into tokens
         *
         * @param text Input text
         * @param add_bos Whether to add BOS token at start
         * @param add_eos Whether to add EOS token at end
         * @return Vector of token IDs
         */
        virtual std::vector<int> encode(const std::string &text,
                                        bool add_bos = true,
                                        bool add_eos = false) const = 0;

        /**
         * @brief Decode tokens into text
         *
         * @param tokens Vector of token IDs
         * @param remove_special Whether to skip special tokens (BOS, EOS)
         * @return Decoded text string
         */
        virtual std::string decode(const std::vector<int> &tokens,
                                   bool remove_special = true) const = 0;

        /**
         * @brief Decode a single token into text
         *
         * @param token Token ID
         * @return Decoded text for this token
         */
        virtual std::string decode_token(int token) const = 0;

        /**
         * @brief Get BOS (beginning-of-sequence) token ID
         */
        virtual int bos_token() const = 0;

        /**
         * @brief Get EOS (end-of-sequence) token ID
         */
        virtual int eos_token() const = 0;

        /**
         * @brief Get all stop token IDs
         *
         * Returns all tokens that should terminate generation, including:
         * - EOS token (always)
         * - Chat-specific tokens like <|im_end|> for ChatML models
         *
         * @return Vector of stop token IDs
         */
        virtual std::vector<int> stop_tokens() const = 0;

        /**
         * @brief Check if a token is a stop token
         *
         * @param token_id Token ID to check
         * @return true if token should stop generation
         */
        virtual bool is_stop_token(int token_id) const = 0;

        /**
         * @brief Get vocabulary size
         */
        virtual int vocab_size() const = 0;

        // =====================================================================
        // Chat Template Support
        // =====================================================================

        /**
         * @brief Get the chat template from model metadata
         *
         * Returns the raw chat template string from the GGUF metadata,
         * typically from `tokenizer.chat_template`. Returns empty string
         * if no template is available.
         *
         * @return Raw template string or empty if unavailable
         */
        virtual std::string getChatTemplateString() const = 0;

        /**
         * @brief Get the detected chat template type
         *
         * @return ChatTemplateType enum value (UNKNOWN if not detected)
         */
        virtual ChatTemplateType getChatTemplateType() const = 0;

        /**
         * @brief Check if the tokenizer has a chat template
         *
         * @return true if a chat template is available
         */
        virtual bool hasChatTemplate() const = 0;

        /**
         * @brief Get the chat template (must call hasChatTemplate() first)
         */
        virtual const ChatTemplate &getChatTemplate() const = 0;

        /**
         * @brief Set/override the chat template
         *
         * Allows overriding the model's chat template with a custom one.
         * Useful when the model doesn't have a template or when using
         * a different template format.
         *
         * @param tmpl Chat template to use
         */
        virtual void setChatTemplate(std::unique_ptr<ChatTemplate> tmpl) = 0;

        /**
         * @brief Apply chat template to messages and encode
         *
         * Formats messages using the model's chat template, then tokenizes.
         * This is the main entry point for chat-based inference.
         *
         * @param messages Vector of chat messages (role + content)
         * @param add_generation_prompt Whether to add assistant prompt suffix
         * @param tools Optional JSON array of tool definitions for tool-calling templates
         * @return Encoded token IDs ready for inference
         */
        virtual std::vector<int> encodeChat(
            const std::vector<ChatMessage> &messages,
            bool add_generation_prompt = true,
            const std::string &tools_json = "") const = 0;

        /**
         * @brief Format messages using chat template (without encoding)
         *
         * Useful for debugging or when you need the formatted text.
         *
         * @param messages Vector of chat messages
         * @param add_generation_prompt Whether to add assistant prompt suffix
         * @param tools Optional JSON array of tool definitions for tool-calling templates
         * @return Formatted text string
         */
        virtual std::string applyTemplate(
            const std::vector<ChatMessage> &messages,
            bool add_generation_prompt = true,
            const std::string &tools_json = "") const = 0;
    };

    /**
     * @brief Native BPE tokenizer implementation
     *
     * Reads vocabulary and merge rules directly from GGUF metadata.
     * Implements Byte Pair Encoding (BPE) algorithm for text tokenization.
     *
     * Supports GPT-2 style BPE tokenization used by Qwen, LLaMA, and similar models.
     */
    class BPETokenizer : public ITokenizer
    {
    public:
        /**
         * @brief Create tokenizer from ModelContext
         *
         * Extracts vocabulary from GGUF metadata without re-loading the file.
         *
         * @param model_ctx Model context with loaded GGUF metadata
         * @return Shared pointer to tokenizer, or nullptr on error
         */
        static std::shared_ptr<BPETokenizer> create(std::shared_ptr<ModelContext> model_ctx);

        /**
         * @brief Destructor
         */
        ~BPETokenizer() override = default;

        // ITokenizer implementation
        std::vector<int> encode(const std::string &text,
                                bool add_bos = true,
                                bool add_eos = false) const override;

        std::string decode(const std::vector<int> &tokens,
                           bool remove_special = true) const override;

        std::string decode_token(int token) const override;

        int bos_token() const override { return bos_token_; }
        int eos_token() const override { return eos_token_; }
        std::vector<int> stop_tokens() const override { return stop_tokens_; }
        bool is_stop_token(int token_id) const override;
        int vocab_size() const override { return vocab_.size(); }

        // Chat template implementation
        std::string getChatTemplateString() const override { return chat_template_string_; }
        ChatTemplateType getChatTemplateType() const override;
        bool hasChatTemplate() const override { return chat_template_ != nullptr; }
        const ChatTemplate &getChatTemplate() const override { return *chat_template_; }
        void setChatTemplate(std::unique_ptr<ChatTemplate> tmpl) override;
        std::vector<int> encodeChat(
            const std::vector<ChatMessage> &messages,
            bool add_generation_prompt = true,
            const std::string &tools_json = "") const override;
        std::string applyTemplate(
            const std::vector<ChatMessage> &messages,
            bool add_generation_prompt = true,
            const std::string &tools_json = "") const override;

    private:
        /**
         * @brief Private constructor - use create() factory method
         */
        BPETokenizer() = default;

        /**
         * @brief Initialize tokenizer from GGUF metadata
         */
        bool initializeFromMetadata(std::shared_ptr<ModelContext> model_ctx);

        /**
         * @brief Initialize byte encoder/decoder
         */
        void initializeByteEncoder();

        /**
         * @brief Apply BPE merges to text
         */
        std::vector<int> applyBPE(const std::string &text) const;

        /**
         * @brief Byte-level encoding (for handling any UTF-8)
         */
        std::string bytesToUnicode(const std::string &text) const;
        std::string unicodeToBytes(const std::string &text) const;

        // Vocabulary data
        std::vector<std::string> vocab_;
        std::unordered_map<std::string, int> vocab_map_;
        std::vector<std::pair<std::string, std::string>> merges_;
        std::map<std::pair<std::string, std::string>, int> merge_ranks_;
        std::map<std::pair<int, int>, int> merge_ranks_int_;
        std::vector<int> merge_result_ids_;
        std::vector<int> byte_to_token_id_;

        std::vector<std::string> byte_encoder_;
        std::unordered_map<std::string, int> byte_decoder_;

        int bos_token_ = 0;
        int eos_token_ = 0;
        int pad_token_ = 0;

        // Stop tokens - tokens that should terminate generation
        // Always includes EOS, plus chat-specific tokens like <|im_end|>
        std::vector<int> stop_tokens_;

        // Special tokens (sorted by length descending for greedy matching)
        // e.g., <|im_start|>, <|im_end|>, <|endoftext|>
        std::vector<std::pair<std::string, int>> special_tokens_;

        /**
         * @brief Initialize special tokens from vocabulary
         *
         * Scans vocab for tokens matching <|...|> pattern and stores them
         * sorted by length (longest first) for greedy matching.
         */
        void initializeSpecialTokens(const std::map<std::string, GGUFValue> &metadata);

        /**
         * @brief Initialize stop tokens based on detected chat template
         *
         * Called after special tokens are initialized to identify which
         * tokens should terminate generation.
         */
        void initializeStopTokens();

        /**
         * @brief Encode text with special token handling
         *
         * Splits text on special tokens, applies BPE to non-special segments.
         */
        std::vector<int> encodeWithSpecialTokens(const std::string &text) const;

        // Chat template data
        std::string chat_template_string_;
        std::unique_ptr<ChatTemplate> chat_template_;
    };

    /**
     * @brief Factory function to create appropriate tokenizer for model
     *
     * @param model_ctx Model context with loaded GGUF metadata
     * @return Shared pointer to tokenizer, or nullptr on error
     *
     * Currently creates BPETokenizer for all architectures.
     * In future, could dispatch to architecture-specific implementations
     * if needed (e.g., SentencePiece for some models).
     */
    std::shared_ptr<ITokenizer> createTokenizer(std::shared_ptr<ModelContext> model_ctx);

} // namespace llaminar2
