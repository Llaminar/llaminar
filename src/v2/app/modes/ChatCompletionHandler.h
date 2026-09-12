/**
 * @file ChatCompletionHandler.h
 * @brief Testable request handler for /v1/chat/completions
 *
 * Encapsulates: JSON parsing → sampling params → prefill → decode → response.
 * All dependencies are injected via interfaces (IOrchestrationRunner, ITokenizer).
 *
 * Supports both non-streaming and SSE streaming responses with
 * reasoning_content extraction for thinking models.
 * Non-streaming callers may request exact prompt/completion token IDs. These
 * are observations of ordinary committed runner output, not model snapshots
 * or a separate inference path, and remain independent of text framing.
 * Optional runtime JSON also projects the placement owner's completed,
 * model-lifetime movement ledger without polling or joining maintenance.
 */

#pragma once

#include "utils/Sampler.h"
#include "utils/ChatTemplate.h"
#include "utils/ToolCallTypes.h"
#include <string>
#include <vector>
#include <optional>
#include <functional>
#include <nlohmann/json.hpp>

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
        bool dry_multiplier{false};
        bool dry_base{false};
        bool dry_allowed_length{false};
        bool dry_penalty_last_n{false};
        bool dry_sequence_breakers{false};
    };

    /** @brief Optional response representation; never changes sampling or execution. */
    enum class CompletionTokenOutput
    {
        TextOnly,       ///< Standard HTTP response with no token-vector storage.
        TextAndIds,     ///< Non-streaming terminal response includes exact token IDs.
    };

    /** @brief Optional completed-request observations; never a live-state probe. */
    enum class CompletionRuntimeOutput
    {
        Omit,    ///< Standard response; no runtime-summary JSON.
        Include, ///< Non-streaming response includes terminal summary and passive lifetime movement evidence.
    };

    /** @brief Validated immutable request policy consumed by the serving handler. */
    struct ChatCompletionRequest
    {
        std::vector<ChatMessage> messages;
        /// -1 means "unspecified": the handler will use (context_window - prompt_tokens)
        /// so the model can generate up to the remaining context.
        int max_tokens{-1};
        SamplingParams sampling;
        SamplingOverrides sampling_set;  ///< Per-field "user specified" flags
        bool stream{false};         ///< If true, use SSE streaming response
        CompletionTokenOutput token_output{CompletionTokenOutput::TextOnly};
                                   ///< Requested terminal representation, independent of model state.
        CompletionRuntimeOutput runtime_output{CompletionRuntimeOutput::Omit};
                                   ///< Optional immutable outcome, independent of logging/PerfStats.
        bool enable_thinking{true}; ///< If true, enable thinking mode for thinking models
        std::string model;          ///< Model identifier from request (optional)

        /// Thinking budget: maximum tokens to spend in thinking mode.
        /// -1 = no budget (unlimited thinking), 0 = disable thinking.
        /// When exhausted, a stop-thinking sequence is injected into the stream.
        int thinking_budget_tokens{-1};

        // =================================================================
        // Tool Calling Fields (OpenAI-compatible)
        // =================================================================

        /// Tool definitions available for the model to call.
        /// JSON array of {type: "function", function: {name, description, parameters}}.
        nlohmann::json tools;

        /// Tool choice control: "none", "auto", "required", or
        /// {"type": "function", "function": {"name": "..."}} to force a specific tool.
        nlohmann::json tool_choice;

        /// Allow model to call multiple tools in a single response.
        bool parallel_tool_calls{false};
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
     * splitting tags across chunks. This is a field-framing authority only:
     * reasoning delimiters never terminate generation, including when the model
     * emits a natural close after a budget-injected close.
     */
    class StreamingThinkSplitter
    {
    public:
        /** @brief Start the reasoning phase with the model's exact closing marker. */
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
        /** @brief Consume one tokenizer piece, retaining a possible partial marker. */
        SplitResult process(const std::string &token_text);

        /// Flush any buffered partial tag match (call at end of generation)
        SplitResult flush();

        /// Whether we're currently in the thinking phase
        bool inThinking() const { return phase_ == Phase::Reasoning; }

    private:
        /** @brief The same reasoning/answer lifecycle governs HTTP and SSE output. */
        enum class Phase
        {
            Reasoning,      ///< Before the first complete close marker.
            AwaitingAnswer, ///< Closed reasoning, but no visible answer yet.
            Content,        ///< Answer text has begun (or reasoning is disabled).
        };

        /**
         * @brief Drain answer text without exposing structural closing markers.
         * @param terminal True only when flushing the final buffered fragment.
         * @return Content with complete structural markers removed, never a stop decision.
         */
        SplitResult drainContent(bool terminal);

        std::string end_tag_;
        Phase phase_{Phase::Reasoning}; ///< Single authority for output transitions.
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

        ChatCompletionResponse handleUnhandledRequestException(
            const char *phase,
            const std::string &detail);

        IOrchestrationRunner &runner_;
        ITokenizer &tokenizer_;
        std::string model_name_;
    };

} // namespace llaminar2
