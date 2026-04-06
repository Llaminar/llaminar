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
            json err = {{"error", {{"message", std::string("Invalid JSON: ") + e.what()}, {"type", "invalid_request_error"}}}};
            error_out.json_body = err.dump();
            return std::nullopt;
        }

        // Validate required fields
        if (!body.contains("messages") || !body["messages"].is_array() || body["messages"].empty())
        {
            error_out.ok = false;
            error_out.http_status = 400;
            json err = {{"error", {{"message", "\"messages\" field is required and must be a non-empty array"}, {"type", "invalid_request_error"}}}};
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
                json err = {{"error", {{"message", "Each message must have \"role\" and \"content\" fields"}, {"type", "invalid_request_error"}}}};
                error_out.json_body = err.dump();
                return std::nullopt;
            }
        }

        ChatCompletionRequest request;

        // Extract parameters with OpenAI-compatible defaults
        request.max_tokens = body.value("max_tokens", 128);

        // Sampling parameters — only override SamplingParams defaults if user specified them.
        // Unspecified fields stay at SamplingParams constructor defaults, allowing
        // handleRequest() to detect "no user sampling specified" and apply model defaults.
        if (body.contains("temperature"))
            request.sampling.temperature = body["temperature"].get<float>();
        if (body.contains("top_p"))
            request.sampling.top_p = body["top_p"].get<float>();
        if (body.contains("top_k"))
            request.sampling.top_k = body["top_k"].get<int>();
        if (body.contains("seed"))
            request.sampling.seed = body["seed"].get<unsigned int>();
        if (body.contains("presence_penalty"))
            request.sampling.presence_penalty = body["presence_penalty"].get<float>();
        if (body.contains("frequency_penalty"))
            request.sampling.frequency_penalty = body["frequency_penalty"].get<float>();

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

        // Merge model-recommended defaults for unspecified sampling params.
        // If the user explicitly sent greedy (temp=0, no penalties), respect that.
        // But if they sent NO sampling params at all, use model defaults.
        SamplingParams effective = request.sampling;
        SamplingParams model_defaults = runner_.getRecommendedSamplingParams();
        SamplingParams api_defaults; // constructed with default values

        // Only apply model defaults if user sent exactly the API defaults
        // (meaning they likely didn't specify any sampling params)
        if (effective.temperature == api_defaults.temperature &&
            effective.top_p == api_defaults.top_p &&
            effective.top_k == api_defaults.top_k &&
            effective.presence_penalty == api_defaults.presence_penalty &&
            effective.frequency_penalty == api_defaults.frequency_penalty)
        {
            effective = model_defaults;
            LOG_INFO("[ChatCompletion] No user sampling params specified, using model defaults: "
                     << "temp=" << effective.temperature
                     << " top_p=" << effective.top_p
                     << " top_k=" << effective.top_k
                     << " presence_penalty=" << effective.presence_penalty
                     << " frequency_penalty=" << effective.frequency_penalty);
        }
        else
        {
            LOG_INFO("[ChatCompletion] User sampling params: "
                     << "temp=" << effective.temperature
                     << " top_p=" << effective.top_p
                     << " top_k=" << effective.top_k
                     << " presence_penalty=" << effective.presence_penalty
                     << " frequency_penalty=" << effective.frequency_penalty);
        }

        // Configure sampling BEFORE any inference
        runner_.setSamplingParams(effective);

        // Encode with chat template
        auto token_ids = tokenizer_.encodeChat(request.messages, /*add_generation_prompt=*/true);

        if (token_ids.empty())
        {
            response.http_status = 500;
            json err = {{"error", {{"message", "Failed to encode conversation with chat template"}, {"type", "server_error"}}}};
            response.json_body = err.dump();
            return response;
        }

        int prompt_tokens = static_cast<int>(token_ids.size());
        int max_context = runner_.config().max_seq_len;

        // Validate prompt fits within context window
        if (prompt_tokens > max_context)
        {
            response.http_status = 400;
            json err = {{"error", {{"message", "Prompt (" + std::to_string(prompt_tokens) + " tokens) exceeds context window (" + std::to_string(max_context) + " tokens). "
                                                                                                                                                                "Use -c <size> to increase context length."},
                                   {"type", "invalid_request_error"},
                                   {"param", "messages"}}}};
            response.json_body = err.dump();
            return response;
        }

        // Convert to int32_t
        std::vector<int32_t> input_ids(token_ids.begin(), token_ids.end());

        // Prefill
        if (!runner_.prefill(input_ids))
        {
            response.http_status = 500;
            json err = {{"error", {{"message", std::string("Prefill failed: ") + runner_.lastError()}, {"type", "server_error"}}}};
            response.json_body = err.dump();
            return response;
        }

        // Decode loop
        std::string generated_text;
        int completion_tokens = 0;
        std::string finish_reason = "length"; // Default: hit max_tokens

        for (int i = 0; i < request.max_tokens; ++i)
        {
            GenerationResult result = runner_.decodeStep();

            if (!result.success())
            {
                response.http_status = 500;
                json err = {{"error", {{"message", std::string("Decode failed: ") + result.error}, {"type", "server_error"}}}};
                response.json_body = err.dump();
                return response;
            }

            if (result.tokens.empty())
            {
                finish_reason = "stop";
                break;
            }

            int32_t next_token = result.tokens[0];
            completion_tokens++;

            if (result.is_complete || tokenizer_.is_stop_token(next_token))
            {
                finish_reason = "stop";
                break;
            }

            generated_text += tokenizer_.decode_token(next_token);
        }

        // Flush GPU timeline
        runner_.flushStageTimeline();

        // Post-process output: use ChatParser to extract thinking content
        std::string reasoning_content;
        std::string content = generated_text;
        if (tokenizer_.hasChatTemplate())
        {
            const auto &chat_template = tokenizer_.getChatTemplate();
            ChatParser parser(chat_template);
            if (parser.expectsThinking())
            {
                auto parsed = parser.parse(generated_text);
                content = parsed.content;
                reasoning_content = parsed.reasoning_content;
            }
        }

        // Build OpenAI-compatible response
        json message = {{"role", "assistant"}, {"content", content}};
        if (!reasoning_content.empty())
        {
            message["reasoning_content"] = reasoning_content;
        }

        json json_response = {
            {"id", "chatcmpl-llaminar"},
            {"object", "chat.completion"},
            {"choices", json::array({json{{"index", 0},
                                          {"message", message},
                                          {"finish_reason", finish_reason}}})},
            {"usage", {{"prompt_tokens", prompt_tokens}, {"completion_tokens", completion_tokens}, {"total_tokens", prompt_tokens + completion_tokens}, {"context_window", max_context}, {"context_used", prompt_tokens + completion_tokens}}}};

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
