#pragma once

#include "ChatMessage.h"
#include "TokenizerInterface.h"
#include "../ArgumentParser.h"
#include <vector>
#include <memory>
#include <string>
#include <cstdint>

namespace llaminar
{
    namespace chat
    {

        /**
         * Manages conversation state, message history, and context tracking
         * for turn-based chat interactions
         */
        class ChatSession
        {
        public:
            /**
             * Constructor
             * @param tokenizer Tokenizer instance for text processing
             * @param params Configuration parameters
             */
            ChatSession(std::unique_ptr<TokenizerInterface> tokenizer,
                        const LlaminarParams &params);

            /**
             * Destructor
             */
            ~ChatSession() = default;

            /**
             * Add a new message to the conversation
             * @param role Message role ("user", "assistant", "system")
             * @param content Message content
             */
            void addMessage(const std::string &role, const std::string &content);

            /**
             * Add a new message using ChatRole enum
             * @param role Message role enum
             * @param content Message content
             */
            void addMessage(ChatRole role, const std::string &content);

            /**
             * Prepare tokenized prompt from conversation history
             * Applies chat template and manages context length
             * @return Vector of token IDs ready for inference
             */
            std::vector<int32_t> preparePrompt();

            /**
             * Update conversation with assistant response
             * @param response Generated response text
             */
            void updateWithResponse(const std::string &response);

            /**
             * Clear all conversation history
             */
            void clearHistory();

            /**
             * Get conversation history
             * @return Const reference to message history
             */
            const std::vector<ChatMessage> &getHistory() const;

            /**
             * Check if context is approaching limit
             * @return True if context is near maximum size
             */
            bool isContextNearLimit() const;

            /**
             * Get current token count in conversation
             * @return Number of tokens in current context
             */
            size_t getCurrentTokenCount() const;

            /**
             * Get maximum context size
             * @return Maximum allowed tokens in context
             */
            size_t getMaxContextSize() const;

            /**
             * Set system prompt (overwrites existing system message)
             * @param system_prompt System prompt text
             */
            void setSystemPrompt(const std::string &system_prompt);

            /**
             * Check if session has a system prompt
             * @return True if system prompt is set
             */
            bool hasSystemPrompt() const;

            /**
             * Get statistics about the current session
             * @return String with session statistics
             */
            std::string getSessionStats() const;

        private:
            std::vector<ChatMessage> history_;
            std::unique_ptr<TokenizerInterface> tokenizer_;
            const LlaminarParams &params_;
            size_t max_context_tokens_;
            mutable size_t cached_token_count_;
            mutable bool token_count_dirty_;

            /**
             * Recalculate token count if cache is dirty
             */
            void updateTokenCount() const;

            /**
             * Truncate history to fit within context limits
             * Preserves system prompt and recent messages
             */
            void truncateToContext();

            /**
             * Find system prompt message index
             * @return Index of system message or -1 if not found
             */
            int findSystemPromptIndex() const;
        };

    } // namespace chat
} // namespace llaminar