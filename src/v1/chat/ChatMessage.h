#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <chrono>
#include <stdexcept>

namespace llaminar
{
    namespace chat
    {

        /**
         * Represents a single message in a chat conversation
         */
        struct ChatMessage
        {
            std::string role;            // "user", "assistant", "system"
            std::string content;         // Message text content
            int64_t timestamp;           // Unix timestamp when message was created
            std::vector<int32_t> tokens; // Cached tokenization (optional)

            ChatMessage() = default;
            ChatMessage(const std::string &role, const std::string &content)
                : role(role), content(content), timestamp(getCurrentTimestamp()) {}

        private:
            static int64_t getCurrentTimestamp()
            {
                return std::chrono::duration_cast<std::chrono::seconds>(
                           std::chrono::system_clock::now().time_since_epoch())
                    .count();
            }
        };

        /**
         * Chat roles enumeration for type safety
         */
        enum class ChatRole
        {
            SYSTEM,
            USER,
            ASSISTANT
        };

        /**
         * Convert ChatRole enum to string
         */
        inline std::string roleToString(ChatRole role)
        {
            switch (role)
            {
            case ChatRole::SYSTEM:
                return "system";
            case ChatRole::USER:
                return "user";
            case ChatRole::ASSISTANT:
                return "assistant";
            default:
                return "unknown";
            }
        }

        /**
         * Convert string to ChatRole enum
         */
        inline ChatRole stringToRole(const std::string &role)
        {
            if (role == "system")
                return ChatRole::SYSTEM;
            if (role == "user")
                return ChatRole::USER;
            if (role == "assistant")
                return ChatRole::ASSISTANT;
            throw std::invalid_argument("Invalid chat role: " + role);
        }

    } // namespace chat
} // namespace llaminar