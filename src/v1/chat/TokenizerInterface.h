#pragma once

#include "ChatMessage.h"
#include "../ModelLoader.h"
#include <string>
#include <vector>
#include <memory>
#include <cstdint>

namespace llaminar
{
    namespace chat
    {

        /**
         * Abstract interface for text tokenization and detokenization
         * Bridges between text input and model token vocabulary
         */
        class TokenizerInterface
        {
        public:
            virtual ~TokenizerInterface() = default;

            /**
             * Convert text string to sequence of token IDs
             * @param text Input text to tokenize
             * @return Vector of token IDs
             */
            virtual std::vector<int32_t> tokenize(const std::string &text) = 0;

            /**
             * Convert sequence of token IDs back to text
             * @param tokens Vector of token IDs
             * @return Reconstructed text string
             */
            virtual std::string detokenize(const std::vector<int32_t> &tokens) = 0;

            /**
             * Apply chat template to format conversation messages
             * @param messages Vector of chat messages
             * @param add_generation_prompt Whether to add generation prompt suffix
             * @return Formatted prompt string
             */
            virtual std::string applyTemplate(const std::vector<ChatMessage> &messages,
                                              bool add_generation_prompt = true) = 0;

            /**
             * Load vocabulary from GGUF model
             * @param model ModelLoader instance with loaded vocabulary
             * @return True if vocabulary loaded successfully
             */
            virtual bool loadVocabulary(const ModelLoader &model) = 0;

            /**
             * Get special token ID by name
             * @param token_name Name of special token (e.g., "<|endoftext|>", "<bos>", "<eos>")
             * @return Token ID or -1 if not found
             */
            virtual int32_t getSpecialToken(const std::string &token_name) = 0;

            /**
             * Get vocabulary size
             * @return Total number of tokens in vocabulary
             */
            virtual size_t getVocabSize() const = 0;

            /**
             * Check if tokenizer is ready for use
             * @return True if vocabulary is loaded and tokenizer is functional
             */
            virtual bool isReady() const = 0;

            /**
             * Get token string representation
             * @param token_id Token ID to look up
             * @return String representation of token
             */
            virtual std::string getTokenString(int32_t token_id) = 0;
        };

        /**
         * Factory function to create appropriate tokenizer implementation
         * @param model ModelLoader with loaded GGUF model
         * @return Unique pointer to tokenizer instance
         */
        std::unique_ptr<TokenizerInterface> createTokenizer(const ModelLoader &model);

    } // namespace chat
} // namespace llaminar