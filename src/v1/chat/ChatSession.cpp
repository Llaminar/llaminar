#include "ChatSession.h"
#include "../Logger.h"
#include <algorithm>
#include <sstream>
#include <numeric>
#include <iomanip>

namespace llaminar
{
    namespace chat
    {

        ChatSession::ChatSession(std::unique_ptr<TokenizerInterface> tokenizer,
                                 const LlaminarParams &params)
            : tokenizer_(std::move(tokenizer)), params_(params), max_context_tokens_(params.ctx_size > 0 ? params.ctx_size : 2048), cached_token_count_(0), token_count_dirty_(true)
        {
            if (!tokenizer_)
            {
                throw std::invalid_argument("Tokenizer cannot be null");
            }

            if (!tokenizer_->isReady())
            {
                throw std::runtime_error("Tokenizer is not ready for use");
            }

            LOG_INFO("ChatSession initialized with max context size: " << max_context_tokens_);

            // Add system prompt if provided
            if (!params.system_prompt.empty())
            {
                setSystemPrompt(params.system_prompt);
            }
        }

        void ChatSession::addMessage(const std::string &role, const std::string &content)
        {
            if (content.empty())
            {
                LOG_WARN("Attempted to add empty message, skipping");
                return;
            }

            // Validate role
            if (role != "user" && role != "assistant" && role != "system")
            {
                LOG_WARN("Unknown message role: " << role << ", using 'user' instead");
            }

            ChatMessage message(role, content);

            // Pre-tokenize the message for context management
            try
            {
                std::string role_content = role + ": " + content;
                message.tokens = tokenizer_->tokenize(role_content);
                LOG_DEBUG("Tokenized message: " << message.tokens.size() << " tokens");
            }
            catch (const std::exception &e)
            {
                LOG_ERROR("Failed to tokenize message: " << e.what());
                // Continue without cached tokens
            }

            history_.push_back(std::move(message));
            token_count_dirty_ = true;

            // Check if we need to truncate
            updateTokenCount();
            if (cached_token_count_ > max_context_tokens_)
            {
                LOG_DEBUG("Context size exceeded, truncating history");
                truncateToContext();
            }
        }

        void ChatSession::addMessage(ChatRole role, const std::string &content)
        {
            addMessage(roleToString(role), content);
        }

        std::vector<int32_t> ChatSession::preparePrompt()
        {
            if (history_.empty())
            {
                LOG_WARN("No messages in conversation history");
                return {};
            }

            try
            {
                // Apply chat template to format conversation
                std::string formatted_prompt = tokenizer_->applyTemplate(history_, true);
                LOG_DEBUG("Formatted prompt length: " << formatted_prompt.length() << " characters");

                // Tokenize the complete formatted prompt
                std::vector<int32_t> prompt_tokens = tokenizer_->tokenize(formatted_prompt);
                LOG_INFO("Prepared prompt with " << prompt_tokens.size() << " tokens");

                // Ensure we don't exceed context limits
                if (prompt_tokens.size() > max_context_tokens_)
                {
                    LOG_WARN("Prompt exceeds context limit, truncating...");
                    // Keep the most recent tokens
                    size_t start_idx = prompt_tokens.size() - max_context_tokens_;
                    prompt_tokens = std::vector<int32_t>(
                        prompt_tokens.begin() + start_idx,
                        prompt_tokens.end());
                    LOG_INFO("Truncated prompt to " << prompt_tokens.size() << " tokens");
                }

                return prompt_tokens;
            }
            catch (const std::exception &e)
            {
                LOG_ERROR("Failed to prepare prompt: " << e.what());
                return {};
            }
        }

        void ChatSession::updateWithResponse(const std::string &response)
        {
            if (response.empty())
            {
                LOG_WARN("Attempted to add empty response");
                return;
            }

            addMessage("assistant", response);
            LOG_DEBUG("Added assistant response with " << response.length() << " characters");
        }

        void ChatSession::clearHistory()
        {
            history_.clear();
            cached_token_count_ = 0;
            token_count_dirty_ = false;
            LOG_INFO("Conversation history cleared");
        }

        const std::vector<ChatMessage> &ChatSession::getHistory() const
        {
            return history_;
        }

        bool ChatSession::isContextNearLimit() const
        {
            updateTokenCount();
            // Consider context "near limit" at 90% of maximum
            return cached_token_count_ >= (max_context_tokens_ * 9 / 10);
        }

        size_t ChatSession::getCurrentTokenCount() const
        {
            updateTokenCount();
            return cached_token_count_;
        }

        size_t ChatSession::getMaxContextSize() const
        {
            return max_context_tokens_;
        }

        void ChatSession::setSystemPrompt(const std::string &system_prompt)
        {
            if (system_prompt.empty())
            {
                return;
            }

            // Remove existing system prompt if any
            int system_idx = findSystemPromptIndex();
            if (system_idx >= 0)
            {
                history_.erase(history_.begin() + system_idx);
            }

            // Insert new system prompt at the beginning
            ChatMessage system_msg("system", system_prompt);
            try
            {
                std::string role_content = "system: " + system_prompt;
                system_msg.tokens = tokenizer_->tokenize(role_content);
            }
            catch (const std::exception &e)
            {
                LOG_ERROR("Failed to tokenize system prompt: " << e.what());
            }

            history_.insert(history_.begin(), std::move(system_msg));
            token_count_dirty_ = true;

            LOG_INFO("System prompt set with " << system_prompt.length() << " characters");
        }

        bool ChatSession::hasSystemPrompt() const
        {
            return findSystemPromptIndex() >= 0;
        }

        std::string ChatSession::getSessionStats() const
        {
            updateTokenCount();
            std::ostringstream stats;
            stats << "Messages: " << history_.size()
                  << ", Tokens: " << cached_token_count_ << "/" << max_context_tokens_
                  << " (" << std::fixed << std::setprecision(1)
                  << (100.0 * cached_token_count_ / max_context_tokens_) << "%)";

            if (hasSystemPrompt())
            {
                stats << ", System prompt: Yes";
            }

            return stats.str();
        }

        void ChatSession::updateTokenCount() const
        {
            if (!token_count_dirty_)
            {
                return;
            }

            cached_token_count_ = 0;
            for (const auto &message : history_)
            {
                if (!message.tokens.empty())
                {
                    cached_token_count_ += message.tokens.size();
                }
                else
                {
                    // Estimate token count if not cached
                    // Rough estimate: 1 token per 4 characters
                    cached_token_count_ += (message.content.length() + message.role.length() + 3) / 4;
                }
            }

            token_count_dirty_ = false;
            LOG_TRACE("Updated token count: " << cached_token_count_);
        }

        void ChatSession::truncateToContext()
        {
            if (history_.empty())
            {
                return;
            }

            // Always preserve system prompt if it exists
            int system_idx = findSystemPromptIndex();
            std::vector<ChatMessage> preserved_messages;

            if (system_idx >= 0)
            {
                preserved_messages.push_back(history_[system_idx]);
            }

            // Calculate target size (leave some room for response)
            size_t target_tokens = max_context_tokens_ * 3 / 4; // Use 75% of context
            size_t current_tokens = 0;

            // Add messages from most recent backwards until we reach target
            for (int i = static_cast<int>(history_.size()) - 1; i >= 0; --i)
            {
                if (i == system_idx)
                {
                    continue; // Already preserved
                }

                const auto &message = history_[i];
                size_t message_tokens = message.tokens.empty()
                                            ? (message.content.length() + message.role.length() + 3) / 4
                                            : message.tokens.size();

                if (current_tokens + message_tokens > target_tokens && !preserved_messages.empty())
                {
                    break; // Would exceed target, stop here
                }

                preserved_messages.insert(preserved_messages.begin() + (system_idx >= 0 ? 1 : 0), message);
                current_tokens += message_tokens;
            }

            size_t removed_count = history_.size() - preserved_messages.size();
            if (removed_count > 0)
            {
                history_ = std::move(preserved_messages);
                token_count_dirty_ = true;
                LOG_INFO("Truncated " << removed_count << " messages to fit context limit");
            }
        }

        int ChatSession::findSystemPromptIndex() const
        {
            for (size_t i = 0; i < history_.size(); ++i)
            {
                if (history_[i].role == "system")
                {
                    return static_cast<int>(i);
                }
            }
            return -1;
        }

    } // namespace chat
} // namespace llaminar