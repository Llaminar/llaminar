/**
 * @file ChatCompletionHandler.cpp
 * @brief Implementation of ChatCompletionHandler
 */

#include "app/modes/ChatCompletionHandler.h"
#include "execution/runner/IOrchestrationRunner.h"
#include "utils/Tokenizer.h"
#include "utils/Logger.h"
#include "nlohmann/json.hpp"

using json = nlohmann::json;

namespace llaminar2
{

    ChatCompletionHandler::ChatCompletionHandler(
        IOrchestrationRunner &runner, ITokenizer &tokenizer)
        : runner_(runner), tokenizer_(tokenizer)
    {
    }

    // =========================================================================
    // Request parsing (static — no instance state needed)
    // =========================================================================

    std::optional<ChatCompletionRequest> ChatCompletionHandler::parseRequest(
        const std::string &json_body,
        ChatCompletionResponse &error_out)
    {
        json body;
        try
        {
            body = json::parse(json_body);
        }
        catch (const json::parse_error &e)
        {
            error_out.ok = false;
            error_out.http_status = 400;
            json err = {{"error", {{"message", std::string("Invalid JSON: ") + e.what()},
                                   {"type", "invalid_request_error"}}}};
            error_out.json_body = err.dump();
            return std::nullopt;
        }

        // Validate required fields
        if (!body.contains("messages") || !body["messages"].is_array() || body["messages"].empty())
        {
            error_out.ok = false;
            error_out.http_status = 400;
            json err = {{"error", {{"message", "\"messages\" field is required and must be a non-empty array"},
                                   {"type", "invalid_request_error"}}}};
            error_out.json_body = err.dump();
            return std::nullopt;
        }

        // Validate each message
        for (const auto &msg : body["messages"])
        {
            if (!msg.contains("role") || !msg.contains("content"))
            {
                error_out.ok = false;
                error_out.http_status = 400;
                json err = {{"error", {{"message", "Each message must have \"role\" and \"content\" fields"},
                                       {"type", "invalid_request_error"}}}};
                error_out.json_body = err.dump();
                return std::nullopt;
            }
        }

        ChatCompletionRequest request;

        // Extract parameters with OpenAI-compatible defaults
        request.max_tokens = body.value("max_tokens", 128);

        // Sampling parameters — all wired through
        request.sampling.temperature = body.value("temperature", 0.0f);
        request.sampling.top_p = body.value("top_p", 1.0f);
        request.sampling.top_k = body.value("top_k", 0);
        request.sampling.seed = body.value("seed", 0u);

        // Build conversation
        for (const auto &msg : body["messages"])
        {
            request.messages.emplace_back(
                msg["role"].get<std::string>(),
                msg["content"].get<std::string>());
        }

        return request;
    }

    // =========================================================================
    // Inference execution
    // =========================================================================

    ChatCompletionResponse ChatCompletionHandler::handleRequest(
        const ChatCompletionRequest &request)
    {
        ChatCompletionResponse response;

        // Clear KV cache for fresh conversation
        runner_.clearCache();

        // Configure sampling BEFORE any inference
        runner_.setSamplingParams(request.sampling);

        // Encode with chat template
        auto token_ids = tokenizer_.encodeChat(request.messages, /*add_generation_prompt=*/true);
        if (token_ids.empty())
        {
            response.http_status = 500;
            json err = {{"error", {{"message", "Failed to encode conversation with chat template"},
                                   {"type", "server_error"}}}};
            response.json_body = err.dump();
            return response;
        }

        int prompt_tokens = static_cast<int>(token_ids.size());

        // Convert to int32_t
        std::vector<int32_t> input_ids(token_ids.begin(), token_ids.end());

        // Prefill
        if (!runner_.prefill(input_ids))
        {
            response.http_status = 500;
            json err = {{"error", {{"message", std::string("Prefill failed: ") + runner_.lastError()},
                                   {"type", "server_error"}}}};
            response.json_body = err.dump();
            return response;
        }

        // Decode loop
        std::string generated_text;
        int completion_tokens = 0;

        for (int i = 0; i < request.max_tokens; ++i)
        {
            GenerationResult result = runner_.decodeStep();

            if (!result.success())
            {
                response.http_status = 500;
                json err = {{"error", {{"message", std::string("Decode failed: ") + result.error},
                                       {"type", "server_error"}}}};
                response.json_body = err.dump();
                return response;
            }

            if (result.tokens.empty())
                break;

            int32_t next_token = result.tokens[0];
            completion_tokens++;

            if (result.is_complete || tokenizer_.is_stop_token(next_token))
                break;

            generated_text += tokenizer_.decode_token(next_token);
        }

        // Flush GPU timeline
        runner_.flushStageTimeline();

        // Build OpenAI-compatible response
        json json_response = {
            {"id", "chatcmpl-llaminar"},
            {"object", "chat.completion"},
            {"choices", json::array({json{{"index", 0},
                                          {"message", {{"role", "assistant"}, {"content", generated_text}}},
                                          {"finish_reason", "stop"}}})},
            {"usage", {{"prompt_tokens", prompt_tokens},
                       {"completion_tokens", completion_tokens},
                       {"total_tokens", prompt_tokens + completion_tokens}}}};

        response.ok = true;
        response.http_status = 200;
        response.json_body = json_response.dump();
        return response;
    }

    // =========================================================================
    // Convenience: parse + execute
    // =========================================================================

    ChatCompletionResponse ChatCompletionHandler::handleRawRequest(const std::string &json_body)
    {
        ChatCompletionResponse error;
        auto request = parseRequest(json_body, error);
        if (!request)
            return error;

        return handleRequest(*request);
    }

} // namespace llaminar2
