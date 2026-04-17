/**
 * @file ChatCompletionHandler.h
 * @brief Testable request handler for /v1/chat/completions
 *
 * Encapsulates: JSON parsing → sampling params → prefill → decode → response.
 * All dependencies are injected via interfaces (IOrchestrationRunner, ITokenizer).
 *
 * Supports both non-streaming and SSE streaming responses with
 * reasoning_content extraction for thinking models.
 */

#pragma once

#include "utils/Sampler.h"
#include "utils/ChatTemplate.h"
#include <string>
#include <vector>
#include <optional>
#include <functional>

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
    /// Per-field overrides: true if the client explicitly specified this field in the request body.
    /// Used to merge with model-recommended defaults (only unspecified fields get the model defaults).
    struct SamplingOverrides
    {
        bool temperature{false};
        bool top_p{false};
        bool top_k{false};
        bool presence_penalty{false};
        bool frequency_penalty{false};
        bool seed{false};
    };

    struct ChatCompletionRequest
    {
        std::vector<ChatMessage> messages;
        /// -1 means "unspecified": the handler will use (context_window - prompt_tokens)
        /// so the model can generate up to the remaining context.
        int max_tokens{-1};
        SamplingParams sampling;
        SamplingOverrides sampling_set;  ///< Per-field "user specified" flags
        bool stream{false};         ///< If true, use SSE streaming response
        bool enable_thinking{true}; ///< If true, enable thinking mode for thinking models
        std::string model;          ///< Model identifier from request (optional)
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

    /// Callback for emitting SSE chunks. Returns false to abort streaming.
    using StreamChunkCallback = std::function<bool(const std::string &sse_line)>;

    /**
     * @brief Tracks thinking state during streaming to split reasoning_content from content
     *
     * During streaming, tokens are emitted one at a time. This class tracks whether
     * we're inside a <think> block and buffers partial end-tag matches to avoid
     * splitting tags across chunks.
     */
    class StreamingThinkSplitter
    {
    public:
        /// Construct with the thinking end tag (e.g., "</think>")
        explicit StreamingThinkSplitter(const std::string &end_tag);

        /// Construct a no-op splitter (no thinking support)
        StreamingThinkSplitter();

        /// Process a token's text. Returns the field name ("reasoning_content" or "content")
        /// and the text to emit. May return empty text if buffering a partial tag match.
        struct SplitResult
        {
            std::string field; ///< "reasoning_content" or "content"
            std::string text;  ///< Text to emit (may be empty if buffering)
        };
        SplitResult process(const std::string &token_text);

        /// Flush any buffered partial tag match (call at end of generation)
        SplitResult flush();

        /// Whether we're currently in the thinking phase
        bool inThinking() const { return in_thinking_; }

    private:
        std::string end_tag_;
        bool in_thinking_{true}; ///< Start in thinking mode (model begins with <think>)
        std::string buffer_;     ///< Buffer for partial end-tag matches
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
        ChatCompletionHandler(IOrchestrationRunner &runner, ITokenizer &tokenizer,
                              const std::string &model_name = "");

        /// Parse a JSON string into a validated ChatCompletionRequest.
        /// On failure, returns an error ChatCompletionResponse.
        static std::optional<ChatCompletionRequest> parseRequest(
            const std::string &json_body,
            ChatCompletionResponse &error_out);

        /// Execute inference for a validated request (non-streaming).
        ChatCompletionResponse handleRequest(const ChatCompletionRequest &request);

        /// Execute inference with SSE streaming. Calls chunk_cb for each SSE line.
        /// Returns a ChatCompletionResponse with ok=true on success (json_body is empty
        /// since data was streamed). On pre-inference errors, returns error response.
        ChatCompletionResponse handleStreamingRequest(
            const ChatCompletionRequest &request,
            const StreamChunkCallback &chunk_cb);

        /// Convenience: parse + execute in one call (routes to streaming if stream=true).
        ChatCompletionResponse handleRawRequest(const std::string &json_body,
                                                const StreamChunkCallback &stream_cb = nullptr);

        /// Generate a unique chat completion ID (e.g., "chatcmpl-a1b2c3d4e5f6")
        static std::string generateRequestId();

    private:
        /// Common setup: clear cache, merge sampling params, encode, prefill.
        /// Returns prompt_tokens on success, or sets error response and returns -1.
        int setupInference(const ChatCompletionRequest &request,
                           ChatCompletionResponse &error_out,
                           std::vector<int32_t> &input_ids);

        IOrchestrationRunner &runner_;
        ITokenizer &tokenizer_;
        std::string model_name_;
    };

} // namespace llaminar2
