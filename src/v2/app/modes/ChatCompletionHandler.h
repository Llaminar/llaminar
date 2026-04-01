/**
 * @file ChatCompletionHandler.h
 * @brief Testable request handler for /v1/chat/completions
 *
 * Encapsulates: JSON parsing → sampling params → prefill → decode → response.
 * All dependencies are injected via interfaces (IOrchestrationRunner, ITokenizer).
 */

#pragma once

#include "utils/Sampler.h"
#include "utils/ChatTemplate.h"
#include <string>
#include <vector>
#include <optional>

namespace llaminar2
{

    class IOrchestrationRunner;
    class ITokenizer;

    /**
     * @brief Parsed and validated chat completion request
     *
     * Extracted from the JSON body before any inference begins.
     * Separating parsing from execution enables independent testing.
     */
    struct ChatCompletionRequest
    {
        std::vector<ChatMessage> messages;
        int max_tokens{128};
        SamplingParams sampling;
    };

    /**
     * @brief Result of a chat completion request
     */
    struct ChatCompletionResponse
    {
        bool ok{false};
        int http_status{500};
        std::string json_body;
    };

    /**
     * @brief Testable handler for /v1/chat/completions
     *
     * Depends only on IOrchestrationRunner and ITokenizer interfaces,
     * enabling full mock-based unit testing.
     */
    class ChatCompletionHandler
    {
    public:
        ChatCompletionHandler(IOrchestrationRunner &runner, ITokenizer &tokenizer);

        /// Parse a JSON string into a validated ChatCompletionRequest.
        /// On failure, returns an error ChatCompletionResponse.
        static std::optional<ChatCompletionRequest> parseRequest(
            const std::string &json_body,
            ChatCompletionResponse &error_out);

        /// Execute inference for a validated request.
        ChatCompletionResponse handleRequest(const ChatCompletionRequest &request);

        /// Convenience: parse + execute in one call.
        ChatCompletionResponse handleRawRequest(const std::string &json_body);

    private:
        IOrchestrationRunner &runner_;
        ITokenizer &tokenizer_;
    };

} // namespace llaminar2
