/**
 * @file ChatCompletionHandler.cpp
 * @brief Implementation of ChatCompletionHandler
 */

#include "app/modes/ChatCompletionHandler.h"
#include "execution/runner/IOrchestrationRunner.h"
#include "utils/Tokenizer.h"
#include "utils/Logger.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <ctime>
#include <random>
#include <sstream>
#include <iomanip>

using json = nlohmann::json;

namespace llaminar2
{

    // =========================================================================
    // StreamingThinkSplitter
    // =========================================================================

    StreamingThinkSplitter::StreamingThinkSplitter(const std::string &end_tag)
        : end_tag_(end_tag), in_thinking_(!end_tag.empty())
    {
    }

    StreamingThinkSplitter::StreamingThinkSplitter()
        : end_tag_(), in_thinking_(false)
    {
    }

    StreamingThinkSplitter::SplitResult StreamingThinkSplitter::process(const std::string &token_text)
    {
        if (!in_thinking_ || end_tag_.empty())
        {
            // Not in thinking mode or no end tag — everything is content
            return {"content", token_text};
        }

        // We're in thinking mode. Check if this token contains the end tag.
        buffer_ += token_text;

        // Check if the buffer contains the end tag
        auto pos = buffer_.find(end_tag_);
        if (pos != std::string::npos)
        {
            // Found the end tag. Everything before it is reasoning, everything after is content.
            in_thinking_ = false;
            std::string reasoning_part = buffer_.substr(0, pos);
            std::string content_part = buffer_.substr(pos + end_tag_.size());

            // Trim leading whitespace from content (the model often puts \n\n after </think>)
            size_t start = content_part.find_first_not_of(" \t\n\r");
            if (start != std::string::npos)
                content_part = content_part.substr(start);
            else
                content_part.clear();

            buffer_.clear();

            // If we have both reasoning and content, we need two chunks.
            // We return reasoning here and buffer the content for the next call.
            if (!content_part.empty())
                buffer_ = content_part; // Will be returned on next process() or flush()

            if (!reasoning_part.empty())
                return {"reasoning_content", reasoning_part};

            // End tag found but no reasoning text before it — check buffered content
            if (!buffer_.empty())
            {
                std::string c = buffer_;
                buffer_.clear();
                return {"content", c};
            }
            return {"content", ""};
        }

        // Check if the buffer could be a partial match for the end tag
        // (the end of the buffer matches a prefix of the end tag)
        bool could_be_partial = false;
        for (size_t len = 1; len < end_tag_.size() && len <= buffer_.size(); ++len)
        {
            if (buffer_.substr(buffer_.size() - len) == end_tag_.substr(0, len))
            {
                could_be_partial = true;
                break;
            }
        }

        if (could_be_partial)
        {
            // Keep buffering — the end tag might span this and the next token
            // But emit any safe prefix that can't be part of the end tag
            // Find the longest suffix that matches a prefix of end_tag_
            size_t match_len = 0;
            for (size_t len = 1; len < end_tag_.size() && len <= buffer_.size(); ++len)
            {
                if (buffer_.substr(buffer_.size() - len) == end_tag_.substr(0, len))
                    match_len = len;
            }

            if (buffer_.size() > match_len)
            {
                std::string safe = buffer_.substr(0, buffer_.size() - match_len);
                buffer_ = buffer_.substr(buffer_.size() - match_len);
                return {"reasoning_content", safe};
            }
            // Entire buffer is a partial match — keep buffering
            return {"reasoning_content", ""};
        }

        // No partial match — emit everything as reasoning
        std::string result = buffer_;
        buffer_.clear();
        return {"reasoning_content", result};
    }

    StreamingThinkSplitter::SplitResult StreamingThinkSplitter::flush()
    {
        if (buffer_.empty())
            return {in_thinking_ ? "reasoning_content" : "content", ""};

        std::string result = buffer_;
        buffer_.clear();

        if (in_thinking_)
            return {"reasoning_content", result};
        return {"content", result};
    }

    // =========================================================================
    // ChatCompletionHandler
    // =========================================================================

    ChatCompletionHandler::ChatCompletionHandler(
        IOrchestrationRunner &runner, ITokenizer &tokenizer,
        const std::string &model_name)
        : runner_(runner), tokenizer_(tokenizer), model_name_(model_name)
    {
    }

    std::string ChatCompletionHandler::generateRequestId()
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<uint64_t> dist;
        uint64_t val = dist(gen);

        std::ostringstream ss;
        ss << "chatcmpl-" << std::hex << std::setfill('0') << std::setw(12) << val;
        return ss.str();
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

        // Extract parameters. If the client does not specify max_tokens, leave it at -1
        // (sentinel) so handleRequest/handleStreamRequest can default it to the remaining
        // context window (max_seq_len - prompt_tokens) after prefill sizing is known.
        if (body.contains("max_tokens"))
            request.max_tokens = body["max_tokens"].get<int>();

        // Streaming and thinking control
        if (body.contains("stream"))
            request.stream = body["stream"].get<bool>();
        if (body.contains("enable_thinking"))
            request.enable_thinking = body["enable_thinking"].get<bool>();

        // Model identifier (optional, echoed back in response)
        if (body.contains("model"))
            request.model = body["model"].get<std::string>();

        // Sampling parameters — track which fields the client explicitly specified.
        // Fields not set by the client will be filled in from model-recommended defaults
        // during setupInference(), per-field (not all-or-nothing).
        if (body.contains("temperature"))
        {
            request.sampling.temperature = body["temperature"].get<float>();
            request.sampling_set.temperature = true;
        }
        if (body.contains("top_p"))
        {
            request.sampling.top_p = body["top_p"].get<float>();
            request.sampling_set.top_p = true;
        }
        if (body.contains("top_k"))
        {
            request.sampling.top_k = body["top_k"].get<int>();
            request.sampling_set.top_k = true;
        }
        if (body.contains("seed"))
        {
            request.sampling.seed = body["seed"].get<unsigned int>();
            request.sampling_set.seed = true;
        }
        if (body.contains("presence_penalty"))
        {
            request.sampling.presence_penalty = body["presence_penalty"].get<float>();
            request.sampling_set.presence_penalty = true;
        }
        if (body.contains("frequency_penalty"))
        {
            request.sampling.frequency_penalty = body["frequency_penalty"].get<float>();
            request.sampling_set.frequency_penalty = true;
        }

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
    // Common inference setup (shared between streaming and non-streaming)
    // =========================================================================

    int ChatCompletionHandler::setupInference(
        const ChatCompletionRequest &request,
        ChatCompletionResponse &error_out,
        std::vector<int32_t> &input_ids)
    {
        // Clear KV cache for fresh conversation
        runner_.clearCache();

        // Per-field merge of model-recommended defaults: the model defaults apply to
        // any field the client did NOT explicitly set. This prevents a client that sets
        // e.g. temperature from accidentally dropping critical knobs like presence_penalty
        // that some models (e.g. Qwen3.5) require to avoid repetition-loop degeneration.
        SamplingParams effective = request.sampling;
        SamplingParams model_defaults = runner_.getRecommendedSamplingParams();
        const auto &set_ = request.sampling_set;

        if (!set_.temperature)
            effective.temperature = model_defaults.temperature;
        if (!set_.top_p)
            effective.top_p = model_defaults.top_p;
        if (!set_.top_k)
            effective.top_k = model_defaults.top_k;
        if (!set_.presence_penalty)
            effective.presence_penalty = model_defaults.presence_penalty;
        if (!set_.frequency_penalty)
            effective.frequency_penalty = model_defaults.frequency_penalty;
        if (!set_.seed)
            effective.seed = model_defaults.seed;

        LOG_INFO("[ChatCompletion] Sampling params (user-set fields marked *): "
                 << "temp=" << effective.temperature << (set_.temperature ? "* " : " ")
                 << "top_p=" << effective.top_p << (set_.top_p ? "* " : " ")
                 << "top_k=" << effective.top_k << (set_.top_k ? "* " : " ")
                 << "presence_penalty=" << effective.presence_penalty << (set_.presence_penalty ? "* " : " ")
                 << "frequency_penalty=" << effective.frequency_penalty << (set_.frequency_penalty ? "*" : ""));

        runner_.setSamplingParams(effective);

        // Encode with chat template
        auto token_ids = tokenizer_.encodeChat(request.messages, /*add_generation_prompt=*/true);

        if (token_ids.empty())
        {
            error_out.http_status = 500;
            json err = {{"error", {{"message", "Failed to encode conversation with chat template"}, {"type", "server_error"}}}};
            error_out.json_body = err.dump();
            return -1;
        }

        int prompt_tokens = static_cast<int>(token_ids.size());
        int max_context = runner_.config().max_seq_len;

        if (prompt_tokens > max_context)
        {
            error_out.http_status = 400;
            json err = {{"error", {{"message", "Prompt (" + std::to_string(prompt_tokens) + " tokens) exceeds context window (" + std::to_string(max_context) + " tokens). "
                                                                                                                                                                "Use -c <size> to increase context length."},
                                   {"type", "invalid_request_error"},
                                   {"param", "messages"}}}};
            error_out.json_body = err.dump();
            return -1;
        }

        input_ids.assign(token_ids.begin(), token_ids.end());

        if (!runner_.prefill(input_ids))
        {
            error_out.http_status = 500;
            json err = {{"error", {{"message", std::string("Prefill failed: ") + runner_.lastError()}, {"type", "server_error"}}}};
            error_out.json_body = err.dump();
            return -1;
        }

        return prompt_tokens;
    }

    // =========================================================================
    // Non-streaming inference
    // =========================================================================

    ChatCompletionResponse ChatCompletionHandler::handleRequest(
        const ChatCompletionRequest &request)
    {
        ChatCompletionResponse response;
        std::vector<int32_t> input_ids;

        int prompt_tokens = setupInference(request, response, input_ids);
        if (prompt_tokens < 0)
            return response;

        int max_context = runner_.config().max_seq_len;

        // Resolve effective max_tokens: if client did not specify a positive value,
        // default to the remaining context window (max_seq_len - prompt_tokens).
        int effective_max_tokens = (request.max_tokens > 0)
                                       ? request.max_tokens
                                       : std::max(1, max_context - prompt_tokens);

        // Decode loop
        std::string generated_text;
        int completion_tokens = 0;
        std::string finish_reason = "length";

        for (int i = 0; i < effective_max_tokens; ++i)
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

        runner_.flushStageTimeline();

        // Post-process output: use ChatParser to extract thinking content
        std::string reasoning_content;
        std::string content = generated_text;
        if (request.enable_thinking && tokenizer_.hasChatTemplate())
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

        // Build response metadata
        std::string request_id = generateRequestId();
        std::string model = request.model.empty() ? model_name_ : request.model;
        int64_t created = static_cast<int64_t>(std::time(nullptr));

        json message = {{"role", "assistant"}, {"content", content}};
        if (!reasoning_content.empty())
        {
            message["reasoning_content"] = reasoning_content;
        }

        json json_response = {
            {"id", request_id},
            {"object", "chat.completion"},
            {"created", created},
            {"model", model},
            {"system_fingerprint", "llaminar-v2"},
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
    // Streaming inference (SSE)
    // =========================================================================

    ChatCompletionResponse ChatCompletionHandler::handleStreamingRequest(
        const ChatCompletionRequest &request,
        const StreamChunkCallback &chunk_cb)
    {
        ChatCompletionResponse response;
        std::vector<int32_t> input_ids;

        int prompt_tokens = setupInference(request, response, input_ids);
        if (prompt_tokens < 0)
            return response;

        // Resolve effective max_tokens: if client did not specify a positive value,
        // default to the remaining context window (max_seq_len - prompt_tokens).
        int max_context = runner_.config().max_seq_len;
        int effective_max_tokens = (request.max_tokens > 0)
                                       ? request.max_tokens
                                       : std::max(1, max_context - prompt_tokens);

        // Generate consistent metadata for all chunks
        std::string request_id = generateRequestId();
        std::string model = request.model.empty() ? model_name_ : request.model;
        int64_t created = static_cast<int64_t>(std::time(nullptr));

        // Helper to build and emit a single SSE chunk
        auto emit_chunk = [&](const json &delta, const char *finish_reason) -> bool
        {
            json choice = {{"index", 0}, {"delta", delta}};
            if (finish_reason)
                choice["finish_reason"] = std::string(finish_reason);
            else
                choice["finish_reason"] = nullptr;

            json chunk = {
                {"id", request_id},
                {"object", "chat.completion.chunk"},
                {"created", created},
                {"model", model},
                {"system_fingerprint", "llaminar-v2"},
                {"choices", json::array({choice})}};

            std::string sse_line = "data: " + chunk.dump() + "\n\n";
            return chunk_cb(sse_line);
        };

        // First chunk: role announcement
        if (!emit_chunk({{"role", "assistant"}}, nullptr))
        {
            response.ok = true;
            response.http_status = 200;
            return response;
        }

        // Set up thinking splitter
        bool use_think_split = request.enable_thinking && tokenizer_.hasChatTemplate();
        StreamingThinkSplitter splitter;
        if (use_think_split)
        {
            const auto &chat_template = tokenizer_.getChatTemplate();
            if (chat_template.isThinkingModel())
            {
                splitter = StreamingThinkSplitter(chat_template.thinkingEndTag());
            }
            else
            {
                use_think_split = false;
            }
        }

        // Decode loop with per-token emission
        int completion_tokens = 0;
        std::string finish_reason = "length";

        for (int i = 0; i < effective_max_tokens; ++i)
        {
            GenerationResult result = runner_.decodeStep();

            if (!result.success())
            {
                // Emit error as a final chunk and terminate
                json error_data = {{"error", result.error}};
                emit_chunk(error_data, "stop");
                chunk_cb("data: [DONE]\n\n");
                response.ok = false;
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

            std::string token_text = tokenizer_.decode_token(next_token);

            if (use_think_split)
            {
                auto split = splitter.process(token_text);
                if (!split.text.empty())
                {
                    json delta;
                    delta[split.field] = split.text;
                    if (!emit_chunk(delta, nullptr))
                        break;
                }

                // Check if the splitter transitioned and has buffered content
                if (!splitter.inThinking())
                {
                    auto flushed = splitter.flush();
                    if (!flushed.text.empty())
                    {
                        json delta;
                        delta[flushed.field] = flushed.text;
                        if (!emit_chunk(delta, nullptr))
                            break;
                    }
                }
            }
            else
            {
                json delta;
                delta["content"] = token_text;
                if (!emit_chunk(delta, nullptr))
                    break;
            }
        }

        // Flush any remaining buffered thinking content
        if (use_think_split)
        {
            auto flushed = splitter.flush();
            if (!flushed.text.empty())
            {
                json delta;
                delta[flushed.field] = flushed.text;
                emit_chunk(delta, nullptr);
            }
        }

        runner_.flushStageTimeline();

        // Final chunk with finish_reason
        emit_chunk(json::object(), finish_reason.c_str());

        // [DONE] sentinel
        chunk_cb("data: [DONE]\n\n");

        response.ok = true;
        response.http_status = 200;
        return response;
    }

    // =========================================================================
    // Convenience: parse + execute (routes to streaming if stream=true)
    // =========================================================================

    ChatCompletionResponse ChatCompletionHandler::handleRawRequest(
        const std::string &json_body,
        const StreamChunkCallback &stream_cb)
    {
        ChatCompletionResponse error;
        auto request = parseRequest(json_body, error);
        if (!request)
            return error;

        if (request->stream && stream_cb)
            return handleStreamingRequest(*request, stream_cb);

        return handleRequest(*request);
    }

} // namespace llaminar2
