/**
 * @file ChatCompletionHandler.cpp
 * @brief Implementation of ChatCompletionHandler
 */

#include "app/modes/ChatCompletionHandler.h"
#include "execution/runner/IOrchestrationRunner.h"
#include "utils/Tokenizer.h"
#include "utils/DebugEnv.h"
#include "utils/Logger.h"
#include "utils/ToolCallParser.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <exception>
#include <random>
#include <sstream>
#include <iomanip>

using json = nlohmann::json;

namespace llaminar2
{
    namespace
    {
        std::string dumpJsonForHttp(const json &value)
        {
            // Model/tokenizer output can occasionally contain arbitrary byte
            // fragments (for example an isolated byte-level BPE continuation
            // byte). nlohmann::json's default dump() mode is strict UTF-8 and
            // throws in that case, turning an otherwise successful generation
            // into a 500. HTTP JSON responses must be valid UTF-8, so replace
            // malformed byte sequences with U+FFFD at the serialization
            // boundary instead of rejecting the whole response.
            return value.dump(/*indent=*/-1, /*indent_char=*/' ', /*ensure_ascii=*/false,
                              json::error_handler_t::replace);
        }

        GenerationResult decodeStepWithBudget(IOrchestrationRunner &runner, int max_tokens)
        {
            struct BudgetGuard
            {
                IOrchestrationRunner &runner;
                ~BudgetGuard() { runner.setDecodeStepTokenBudget(0); }
            };

            runner.setDecodeStepTokenBudget(max_tokens);
            BudgetGuard guard{runner};
            return runner.decodeStep();
        }

        class RequestCacheCleanup
        {
        public:
            explicit RequestCacheCleanup(IOrchestrationRunner &runner) : runner_(runner) {}
            ~RequestCacheCleanup() noexcept
            {
                try
                {
                    runner_.clearCache();
                }
                catch (const std::exception &e)
                {
                    LOG_WARN("[ChatCompletion] request cleanup failed: " << e.what());
                }
                catch (...)
                {
                    LOG_WARN("[ChatCompletion] request cleanup failed with unknown exception");
                }
            }

        private:
            IOrchestrationRunner &runner_;
        };

        const char *boolString(bool value)
        {
            return value ? "true" : "false";
        }

        bool hasPrefixCacheSummary(const PrefixRuntimeStateSnapshot &snapshot)
        {
            return snapshot.prefix_cache_config_enabled ||
                   snapshot.prefix_cache_ready ||
                   snapshot.prefix_cache_bypassed ||
                   snapshot.prefix_cache_lookups != 0 ||
                   snapshot.prefix_cache_hits != 0 ||
                   snapshot.prefix_cache_partial_hits != 0 ||
                   snapshot.prefix_cache_misses != 0 ||
                   snapshot.prefix_cache_matched_tokens != 0 ||
                   snapshot.prefix_cache_stores != 0 ||
                   snapshot.prefix_cache_bypasses != 0 ||
                   snapshot.prefix_request.enabled ||
                   snapshot.prefix_request.bypassed ||
                   snapshot.prefix_request.requested_tokens != 0 ||
                   snapshot.prefix_request.matched_tokens != 0;
        }

        bool hasMTPSummary(const PrefixRuntimeStateSnapshot &snapshot)
        {
            return snapshot.mtp_config_enabled ||
                   snapshot.mtp_bypassed ||
                   snapshot.mtp_draft_steps != 0 ||
                   snapshot.mtp_accepted_tokens != 0 ||
                   snapshot.mtp_rejected_tokens != 0 ||
                   snapshot.mtp_rollbacks != 0 ||
                   snapshot.mtp_bypasses != 0 ||
                   snapshot.mtp_verifier_runs != 0 ||
                   snapshot.mtp_verifier_token_count != 0 ||
                   snapshot.mtp_depth_policy_updates != 0 ||
                   snapshot.mtp_request.enabled ||
                   snapshot.mtp_request.bypassed ||
                   snapshot.mtp_request.adaptive_depth_enabled ||
                   snapshot.mtp_request.draft_steps != 0 ||
                   snapshot.mtp_request.accepted_tokens != 0 ||
                   snapshot.mtp_request.rejected_tokens != 0 ||
                   snapshot.mtp_request.rollbacks != 0 ||
                   snapshot.mtp_request.stochastic_accept_tests != 0 ||
                   snapshot.mtp_request.stochastic_residual_samples != 0 ||
                   snapshot.mtp_request.stochastic_terminal_samples != 0;
        }

        bool traceGeneratedTokensEnabled()
        {
            return debugEnv().runtime_debug.trace_generated_tokens;
        }

        std::string previewTokenText(std::string text)
        {
            for (char &ch : text)
            {
                if (ch == '\n')
                    ch = ' ';
                else if (ch == '\r' || ch == '\t')
                    ch = ' ';
            }
            constexpr size_t kMaxPreviewChars = 96;
            if (text.size() > kMaxPreviewChars)
                text = text.substr(0, kMaxPreviewChars - 3) + "...";
            return text;
        }

        void traceGeneratedToken(const char *path,
                                 int completion_index,
                                 int32_t token,
                                 const std::string &text,
                                 bool forced)
        {
            if (!traceGeneratedTokensEnabled())
                return;
            LOG_INFO("[ChatCompletion/token] path="
                     << path
                     << " index=" << completion_index
                     << " token=" << token
                     << " forced=" << boolString(forced)
                     << " text=\"" << previewTokenText(text) << "\"");
        }

        using SteadyClock = std::chrono::steady_clock;

        double elapsedMs(SteadyClock::time_point start)
        {
            return std::chrono::duration<double, std::milli>(
                       SteadyClock::now() - start)
                .count();
        }

        std::string tokenIdPreview(const std::vector<int32_t> &tokens)
        {
            std::ostringstream oss;
            const size_t preview_count = std::min<size_t>(tokens.size(), 48);
            for (size_t i = 0; i < preview_count; ++i)
            {
                if (i)
                    oss << ',';
                oss << tokens[i];
            }
            if (tokens.size() > preview_count)
                oss << ",...";
            return oss.str();
        }

        void logRuntimeStateSummary(IOrchestrationRunner &runner, const char *mode)
        {
            const PrefixRuntimeStateSnapshot snapshot = runner.prefixStateProbe();

            if (hasPrefixCacheSummary(snapshot))
            {
                const auto &request = snapshot.prefix_request;
                const bool bypassed = request.bypassed || snapshot.prefix_cache_bypassed;
                const std::string &bypass_reason =
                    !request.bypass_reason.empty() ? request.bypass_reason
                                                   : snapshot.prefix_cache_bypass_reason;

                std::ostringstream prefix;
                prefix << "enabled=" << boolString(request.enabled || snapshot.prefix_cache_config_enabled)
                       << " ready=" << boolString(snapshot.prefix_cache_ready)
                       << " hit=" << boolString(request.hit)
                       << " partial_hit=" << boolString(request.partial_hit)
                       << " requested_tokens=" << request.requested_tokens
                       << " matched_tokens=" << request.matched_tokens
                       << " matched_blocks=" << request.matched_blocks
                       << " tier=" << request.storage_tier
                       << " lookups=" << snapshot.prefix_cache_lookups
                       << " hits=" << snapshot.prefix_cache_hits
                       << " partial_hits=" << snapshot.prefix_cache_partial_hits
                       << " misses=" << snapshot.prefix_cache_misses;
                if (bypassed)
                {
                    prefix << " bypassed=true";
                    if (!bypass_reason.empty())
                        prefix << " bypass_reason=" << bypass_reason;
                }

                LOG_INFO("[ChatCompletion] Prefix cache summary (" << mode << "): "
                                                                   << prefix.str());
            }

            if (hasMTPSummary(snapshot))
            {
                const auto &request = snapshot.mtp_request;
                const uint64_t accepted = request.accepted_tokens != 0
                                              ? request.accepted_tokens
                                              : snapshot.mtp_accepted_tokens;
                const uint64_t rejected = request.rejected_tokens != 0
                                              ? request.rejected_tokens
                                              : snapshot.mtp_rejected_tokens;
                const uint64_t attempted = accepted + rejected;
                const double acceptance_rate = attempted != 0
                                                   ? static_cast<double>(accepted) /
                                                         static_cast<double>(attempted)
                                                   : request.acceptance_rate;
                const int current_depth = request.current_depth != 0
                                              ? request.current_depth
                                              : snapshot.mtp_current_depth;
                const int min_depth = request.min_depth != 0
                                          ? request.min_depth
                                          : snapshot.mtp_min_depth;
                const int max_depth = request.max_depth != 0
                                          ? request.max_depth
                                          : snapshot.mtp_max_depth;
                const uint64_t depth_updates = request.depth_policy_updates != 0
                                                   ? request.depth_policy_updates
                                                   : snapshot.mtp_depth_policy_updates;
                const bool bypassed = request.bypassed || snapshot.mtp_bypassed;
                const std::string &bypass_reason =
                    !request.bypass_reason.empty() ? request.bypass_reason
                                                   : snapshot.mtp_bypass_reason;

                std::ostringstream mtp;
                mtp << "enabled=" << boolString(request.enabled || snapshot.mtp_config_enabled)
                    << " draft_steps=" << (request.draft_steps != 0
                                               ? request.draft_steps
                                               : snapshot.mtp_draft_steps)
                    << " accepted_tokens=" << accepted
                    << " rejected_tokens=" << rejected
                    << " rollbacks=" << (request.rollbacks != 0
                                             ? request.rollbacks
                                             : snapshot.mtp_rollbacks)
                    << " acceptance=" << std::fixed << std::setprecision(2)
                    << (acceptance_rate * 100.0) << "%"
                    << " verifier_runs=" << snapshot.mtp_verifier_runs
                    << " verifier_tokens=" << snapshot.mtp_verifier_token_count
                    << " verify_mode=" << request.verify_mode
                    << " depth_policy=" << request.depth_policy_mode
                    << " depth=" << current_depth
                    << " [" << min_depth << "," << max_depth << "]"
                    << " depth_updates=" << depth_updates;
                if (!request.last_depth_policy_reason.empty())
                    mtp << " last_depth_reason=" << request.last_depth_policy_reason;
                if (request.stochastic_accept_tests != 0 ||
                    request.stochastic_residual_samples != 0 ||
                    request.stochastic_terminal_samples != 0)
                {
                    mtp << " stochastic_accept_tests=" << request.stochastic_accept_tests
                        << " stochastic_acceptance=" << std::fixed << std::setprecision(2)
                        << (request.stochastic_acceptance_rate * 100.0) << "%"
                        << " stochastic_residual_samples="
                        << request.stochastic_residual_samples
                        << " stochastic_terminal_samples="
                        << request.stochastic_terminal_samples;
                }
                if (bypassed)
                {
                    mtp << " bypassed=true";
                    if (!bypass_reason.empty())
                        mtp << " bypass_reason=" << bypass_reason;
                }

                LOG_INFO("[ChatCompletion] MTP summary (" << mode << "): " << mtp.str());
            }
        }
    }

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

    ChatCompletionResponse ChatCompletionHandler::handleUnhandledRequestException(
        const char *phase,
        const std::string &detail)
    {
        LOG_ERROR("[ChatCompletion] " << phase << " failed with unhandled exception: " << detail);
        try
        {
            runner_.clearCache();
        }
        catch (const std::exception &cleanup_error)
        {
            LOG_ERROR("[ChatCompletion] request cleanup failed: " << cleanup_error.what());
        }
        catch (...)
        {
            LOG_ERROR("[ChatCompletion] request cleanup failed with unknown exception");
        }

        ChatCompletionResponse response;
        response.ok = false;
        response.http_status = 500;
        json err = {{"error", {{"message", std::string("Request failed: ") + detail}, {"type", "server_error"}}}};
        response.json_body = dumpJsonForHttp(err);
        return response;
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
            error_out.json_body = dumpJsonForHttp(err);
            return std::nullopt;
        }

        // Validate required fields
        if (!body.contains("messages") || !body["messages"].is_array() || body["messages"].empty())
        {
            error_out.ok = false;
            error_out.http_status = 400;
            json err = {{"error", {{"message", "\"messages\" field is required and must be a non-empty array"}, {"type", "invalid_request_error"}}}};
            error_out.json_body = dumpJsonForHttp(err);
            return std::nullopt;
        }

        // Validate each message — tool-related messages have relaxed requirements
        for (const auto &msg : body["messages"])
        {
            if (!msg.contains("role"))
            {
                error_out.ok = false;
                error_out.http_status = 400;
                json err = {{"error", {{"message", "Each message must have a \"role\" field"}, {"type", "invalid_request_error"}}}};
                error_out.json_body = dumpJsonForHttp(err);
                return std::nullopt;
            }

            std::string role = msg["role"].get<std::string>();

            // Assistant messages with tool_calls may have null/missing content
            if (role == "assistant" && msg.contains("tool_calls"))
                continue;

            // Tool result messages need tool_call_id but content is required
            if (role == "tool")
            {
                if (!msg.contains("tool_call_id"))
                {
                    error_out.ok = false;
                    error_out.http_status = 400;
                    json err = {{"error", {{"message", "Tool messages must have a \"tool_call_id\" field"}, {"type", "invalid_request_error"}}}};
                    error_out.json_body = dumpJsonForHttp(err);
                    return std::nullopt;
                }
                continue;
            }

            // Standard messages require content
            if (!msg.contains("content"))
            {
                error_out.ok = false;
                error_out.http_status = 400;
                json err = {{"error", {{"message", "Each message must have \"role\" and \"content\" fields"}, {"type", "invalid_request_error"}}}};
                error_out.json_body = dumpJsonForHttp(err);
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

        // DRY penalty parameters
        if (body.contains("dry_multiplier"))
        {
            request.sampling.dry_multiplier = body["dry_multiplier"].get<float>();
            request.sampling_set.dry_multiplier = true;
        }
        if (body.contains("dry_base"))
        {
            request.sampling.dry_base = body["dry_base"].get<float>();
            request.sampling_set.dry_base = true;
        }
        if (body.contains("dry_allowed_length"))
        {
            request.sampling.dry_allowed_length = body["dry_allowed_length"].get<int>();
            request.sampling_set.dry_allowed_length = true;
        }
        if (body.contains("dry_penalty_last_n"))
        {
            request.sampling.dry_penalty_last_n = body["dry_penalty_last_n"].get<int>();
            request.sampling_set.dry_penalty_last_n = true;
        }
        if (body.contains("dry_sequence_breakers"))
        {
            request.sampling.dry_sequence_breakers.clear();
            for (const auto &b : body["dry_sequence_breakers"])
                request.sampling.dry_sequence_breakers.push_back(b.get<std::string>());
            request.sampling_set.dry_sequence_breakers = true;
        }

        // Thinking budget
        if (body.contains("thinking_budget_tokens"))
            request.thinking_budget_tokens = body["thinking_budget_tokens"].get<int>();

        // Tool calling parameters
        if (body.contains("tools") && body["tools"].is_array())
            request.tools = body["tools"];
        if (body.contains("tool_choice"))
            request.tool_choice = body["tool_choice"];
        if (body.contains("parallel_tool_calls"))
            request.parallel_tool_calls = body["parallel_tool_calls"].get<bool>();

        // Build conversation with full tool-calling support
        for (const auto &msg : body["messages"])
        {
            ChatMessage cm;
            cm.role = msg["role"].get<std::string>();

            // Content may be null or missing for assistant messages with tool_calls
            if (msg.contains("content") && !msg["content"].is_null())
                cm.content = msg["content"].get<std::string>();

            // Parse tool_calls from assistant messages (store as serialized JSON strings)
            if (msg.contains("tool_calls") && msg["tool_calls"].is_array())
            {
                for (const auto &tc : msg["tool_calls"])
                    cm.tool_calls.push_back(tc.dump());
            }

            // Parse tool_call_id from tool result messages
            if (msg.contains("tool_call_id"))
                cm.tool_call_id = msg["tool_call_id"].get<std::string>();

            // Parse name from tool result messages
            if (msg.contains("name"))
                cm.name = msg["name"].get<std::string>();

            request.messages.push_back(std::move(cm));
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
        if (!set_.dry_multiplier)
            effective.dry_multiplier = model_defaults.dry_multiplier;
        if (!set_.dry_base)
            effective.dry_base = model_defaults.dry_base;
        if (!set_.dry_allowed_length)
            effective.dry_allowed_length = model_defaults.dry_allowed_length;
        if (!set_.dry_penalty_last_n)
            effective.dry_penalty_last_n = model_defaults.dry_penalty_last_n;
        if (!set_.dry_sequence_breakers)
            effective.dry_sequence_breakers = model_defaults.dry_sequence_breakers;

        LOG_DEBUG("[ChatCompletion] Sampling params (user-set fields marked *): "
                  << "temp=" << effective.temperature << (set_.temperature ? "* " : " ")
                  << "top_p=" << effective.top_p << (set_.top_p ? "* " : " ")
                  << "top_k=" << effective.top_k << (set_.top_k ? "* " : " ")
                  << "presence_penalty=" << effective.presence_penalty << (set_.presence_penalty ? "* " : " ")
                  << "frequency_penalty=" << effective.frequency_penalty << (set_.frequency_penalty ? "*" : ""));

        runner_.setSamplingParams(effective);

        // Encode with chat template (pass tools for tool-aware templates)
        std::string tools_json;
        if (request.tools.is_array() && !request.tools.empty())
            tools_json = request.tools.dump();
        auto token_ids = tokenizer_.encodeChat(request.messages, /*add_generation_prompt=*/true,
                                               tools_json, request.enable_thinking);

        if (token_ids.empty())
        {
            error_out.http_status = 500;
            json err = {{"error", {{"message", "Failed to encode conversation with chat template"}, {"type", "server_error"}}}};
            error_out.json_body = dumpJsonForHttp(err);
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
            error_out.json_body = dumpJsonForHttp(err);
            return -1;
        }

        input_ids.assign(token_ids.begin(), token_ids.end());

        if (traceGeneratedTokensEnabled())
        {
            const auto [min_it, max_it] =
                std::minmax_element(input_ids.begin(), input_ids.end());
            LOG_INFO("[ChatCompletion/trace] setup prompt_tokens="
                     << input_ids.size()
                     << " enable_thinking=" << boolString(request.enable_thinking)
                     << " thinking_budget_tokens=" << request.thinking_budget_tokens
                     << " token_min=" << (min_it != input_ids.end() ? *min_it : -1)
                     << " token_max=" << (max_it != input_ids.end() ? *max_it : -1)
                     << " token_ids=[" << tokenIdPreview(input_ids) << "]");
        }

        const auto prefill_start = SteadyClock::now();
        if (!runner_.prefill(input_ids))
        {
            error_out.http_status = 500;
            json err = {{"error", {{"message", std::string("Prefill failed: ") + runner_.lastError()}, {"type", "server_error"}}}};
            error_out.json_body = dumpJsonForHttp(err);
            return -1;
        }
        if (traceGeneratedTokensEnabled())
        {
            LOG_INFO("[ChatCompletion/trace] prefill_ms="
                     << std::fixed << std::setprecision(3)
                     << elapsedMs(prefill_start));
        }

        return prompt_tokens;
    }

    // =========================================================================
    // Non-streaming inference
    // =========================================================================

    ChatCompletionResponse ChatCompletionHandler::handleRequest(
        const ChatCompletionRequest &request)
    try
    {
        ChatCompletionResponse response;
        RequestCacheCleanup request_cleanup(runner_);
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

        // Thinking budget state
        int thinking_tokens = 0;
        bool in_thinking = true; // Assume we start in thinking mode
        bool thinking_budget_active = (request.thinking_budget_tokens >= 0 && request.enable_thinking);
        std::vector<int32_t> stop_thinking_tokens; // Injected token sequence
        int stop_thinking_idx = 0;                 // Current position in injection
        bool injecting_stop_thinking = false;      // True while forcing stop prompt tokens

        // Pre-tokenize stop-thinking prompt if budget is active
        if (thinking_budget_active)
        {
            std::string stop_prompt = runner_.getStopThinkingPrompt();
            if (!stop_prompt.empty())
            {
                /*
                 * The stop-thinking text is injected into an already-active
                 * assistant generation.  It must be encoded as continuation
                 * text, not as a new sequence, or we would force BOS/EOS
                 * instead of the model-author-recommended phrase.
                 */
                auto encoded = tokenizer_.encode(
                    stop_prompt,
                    /*add_bos=*/false,
                    /*add_eos=*/false);
                stop_thinking_tokens.assign(encoded.begin(), encoded.end());
            }
            if (traceGeneratedTokensEnabled())
            {
                LOG_INFO("[ChatCompletion/trace] stop_thinking_prompt_tokens="
                         << stop_thinking_tokens.size()
                         << " token_ids=[" << tokenIdPreview(stop_thinking_tokens)
                         << "]");
            }
            injecting_stop_thinking =
                request.thinking_budget_tokens == 0 && !stop_thinking_tokens.empty();
            if (injecting_stop_thinking)
                in_thinking = false;
        }

        auto rebalance_error = [&]() -> ChatCompletionResponse
        {
            response.http_status = 500;
            json err = {{"error", {{"message", "MoE rebalance failed"}, {"type", "server_error"}}}};
            response.json_body = dumpJsonForHttp(err);
            return response;
        };

        bool stop_generation = false;
        while (completion_tokens < effective_max_tokens && !stop_generation)
        {
            std::vector<int32_t> step_tokens;
            bool step_complete = false;
            bool step_forced = false;

            // Check if we're injecting stop-thinking tokens.
            if (injecting_stop_thinking &&
                stop_thinking_idx < static_cast<int>(stop_thinking_tokens.size()))
            {
                step_forced = true;
                const int32_t forced_token = stop_thinking_tokens[stop_thinking_idx++];
                const auto forced_start = SteadyClock::now();
                GenerationResult result = runner_.forceDecodeToken(forced_token);
                if (traceGeneratedTokensEnabled())
                {
                    LOG_INFO("[ChatCompletion/trace] forceDecodeToken token="
                             << forced_token
                             << " ms=" << std::fixed << std::setprecision(3)
                             << elapsedMs(forced_start)
                             << " success=" << boolString(result.success()));
                }
                if (!result.success())
                {
                    response.http_status = 500;
                    json err = {{"error", {{"message", std::string("Forced decode failed: ") + result.error}, {"type", "server_error"}}}};
                    response.json_body = dumpJsonForHttp(err);
                    return response;
                }
                step_tokens = result.tokens;
                step_complete = result.is_complete;
                if (stop_thinking_idx >= static_cast<int>(stop_thinking_tokens.size()))
                    injecting_stop_thinking = false;
            }
            else
            {
                const int remaining = effective_max_tokens - completion_tokens;
                const int step_budget = thinking_budget_active ? 1 : remaining;
                const auto decode_start = SteadyClock::now();
                GenerationResult result = decodeStepWithBudget(runner_, step_budget);
                if (traceGeneratedTokensEnabled())
                {
                    LOG_INFO("[ChatCompletion/trace] decodeStep budget="
                             << step_budget
                             << " ms=" << std::fixed << std::setprecision(3)
                             << elapsedMs(decode_start)
                             << " tokens=" << result.tokens.size()
                             << " success=" << boolString(result.success()));
                }

                if (!result.success())
                {
                    response.http_status = 500;
                    json err = {{"error", {{"message", std::string("Decode failed: ") + result.error}, {"type", "server_error"}}}};
                    response.json_body = dumpJsonForHttp(err);
                    return response;
                }

                if (result.tokens.empty())
                {
                    finish_reason = "stop";
                    break;
                }

                step_tokens = result.tokens;
                step_complete = result.is_complete;
            }

            for (size_t token_idx = 0;
                 token_idx < step_tokens.size() && completion_tokens < effective_max_tokens;
                 ++token_idx)
            {
                int32_t next_token = step_tokens[token_idx];
                const bool is_final_returned_token = token_idx + 1 == step_tokens.size();
                if (tokenizer_.is_stop_token(next_token) ||
                    (step_complete && is_final_returned_token))
                {
                    completion_tokens++;
                    finish_reason = "stop";
                    stop_generation = true;
                    break;
                }

                std::string token_text = tokenizer_.decode_token(next_token);

                // Check thinking budget
                if (thinking_budget_active && in_thinking)
                {
                    if (token_text.find("</think>") != std::string::npos)
                    {
                        in_thinking = false;
                    }
                    else
                    {
                        thinking_tokens++;
                        if (thinking_tokens >= request.thinking_budget_tokens &&
                            !stop_thinking_tokens.empty())
                        {
                            // Budget exhausted. Keep the token we just sampled
                            // because decodeStep() has already made it the
                            // authoritative last token. The stop prompt begins
                            // at the next decode position through
                            // forceDecodeToken(), which preserves KV/GDN state.
                            LOG_DEBUG("[ChatCompletion] Thinking budget exhausted ("
                                      << thinking_tokens << " tokens), injecting stop-thinking prompt");
                            injecting_stop_thinking = true;
                            stop_thinking_idx = 0;
                            in_thinking = false;
                        }
                    }
                }

                traceGeneratedToken("nonstream",
                                    completion_tokens,
                                    next_token,
                                    token_text,
                                    step_forced);
                completion_tokens++;
                generated_text += token_text;
            }

            if (!stop_generation && !runner_.maybeApplyMoERebalance())
                return rebalance_error();
        }

        runner_.flushStageTimeline();
        logRuntimeStateSummary(runner_, "non-streaming");

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

        // Parse tool calls from model output (if tools were requested)
        ToolCallParseResult tool_result;
        bool has_tool_calls = false;
        if (request.tools.is_array() && !request.tools.empty())
        {
            ToolCallFormat format = runner_.getToolCallFormat();
            tool_result = parseToolCalls(content, format);
            has_tool_calls = tool_result.hasToolCalls();
            if (has_tool_calls)
            {
                content = tool_result.content;
                finish_reason = "tool_calls";
            }
        }

        // Build response metadata
        std::string request_id = generateRequestId();
        std::string model = request.model.empty() ? model_name_ : request.model;
        int64_t created = static_cast<int64_t>(std::time(nullptr));

        json message = {{"role", "assistant"}};
        if (has_tool_calls)
        {
            // Tool call response: content may be null, tool_calls array present
            message["content"] = content.empty() ? json(nullptr) : json(content);
            json tc_array = json::array();
            for (const auto &tc : tool_result.tool_calls)
                tc_array.push_back(toolCallToJson(tc));
            message["tool_calls"] = tc_array;
        }
        else
        {
            message["content"] = content;
        }

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
        response.json_body = dumpJsonForHttp(json_response);
        return response;
    }
    catch (const std::exception &e)
    {
        return handleUnhandledRequestException("non-streaming request", e.what());
    }
    catch (...)
    {
        return handleUnhandledRequestException("non-streaming request", "unknown exception");
    }

    // =========================================================================
    // Streaming inference (SSE)
    // =========================================================================

    ChatCompletionResponse ChatCompletionHandler::handleStreamingRequest(
        const ChatCompletionRequest &request,
        const StreamChunkCallback &chunk_cb)
    try
    {
        ChatCompletionResponse response;
        RequestCacheCleanup request_cleanup(runner_);
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

            std::string sse_line = "data: " + dumpJsonForHttp(chunk) + "\n\n";
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

        // Tool call state: when tools are provided, accumulate output for post-processing
        bool has_tools = request.tools.is_array() && !request.tools.empty();
        std::string accumulated_text; // Always accumulate for tool call detection

        // Thinking budget state
        int thinking_tokens = 0;
        bool thinking_budget_active = (request.thinking_budget_tokens >= 0 && request.enable_thinking);
        std::vector<int32_t> stop_thinking_tokens;
        int stop_thinking_idx = 0;
        bool injecting_stop_thinking = false;

        if (thinking_budget_active)
        {
            std::string stop_prompt = runner_.getStopThinkingPrompt();
            if (!stop_prompt.empty())
            {
                /*
                 * The stop-thinking text is injected into an already-active
                 * assistant generation.  Encode it as continuation text so the
                 * whole author-recommended phrase, and only that phrase, is
                 * forced into the live decode state.
                 */
                auto encoded = tokenizer_.encode(
                    stop_prompt,
                    /*add_bos=*/false,
                    /*add_eos=*/false);
                stop_thinking_tokens.assign(encoded.begin(), encoded.end());
            }
            if (traceGeneratedTokensEnabled())
            {
                LOG_INFO("[ChatCompletion/trace] stream stop_thinking_prompt_tokens="
                         << stop_thinking_tokens.size()
                         << " token_ids=[" << tokenIdPreview(stop_thinking_tokens)
                         << "]");
            }
            injecting_stop_thinking =
                request.thinking_budget_tokens == 0 && !stop_thinking_tokens.empty();
        }

        auto emit_rebalance_error = [&]() -> ChatCompletionResponse
        {
            json error_data = {{"error", "MoE rebalance failed"}};
            emit_chunk(error_data, "stop");
            chunk_cb("data: [DONE]\n\n");
            response.ok = false;
            response.http_status = 500;
            json err = {{"error", {{"message", "MoE rebalance failed"}, {"type", "server_error"}}}};
            response.json_body = dumpJsonForHttp(err);
            return response;
        };

        bool stop_generation = false;
        while (completion_tokens < effective_max_tokens && !stop_generation)
        {
            std::vector<int32_t> step_tokens;
            bool step_complete = false;
            bool step_forced = false;

            // Check if we're injecting stop-thinking tokens.
            if (injecting_stop_thinking &&
                stop_thinking_idx < static_cast<int>(stop_thinking_tokens.size()))
            {
                step_forced = true;
                const int32_t forced_token = stop_thinking_tokens[stop_thinking_idx++];
                const auto forced_start = SteadyClock::now();
                GenerationResult result = runner_.forceDecodeToken(forced_token);
                if (traceGeneratedTokensEnabled())
                {
                    LOG_INFO("[ChatCompletion/trace] stream forceDecodeToken token="
                             << forced_token
                             << " ms=" << std::fixed << std::setprecision(3)
                             << elapsedMs(forced_start)
                             << " success=" << boolString(result.success()));
                }
                if (!result.success())
                {
                    json error_data = {{"error", result.error}};
                    emit_chunk(error_data, "stop");
                    chunk_cb("data: [DONE]\n\n");
                    response.ok = false;
                    response.http_status = 500;
                    json err = {{"error", {{"message", std::string("Decode failed: ") + result.error}, {"type", "server_error"}}}};
                    response.json_body = dumpJsonForHttp(err);
                    return response;
                }
                step_tokens = result.tokens;
                step_complete = result.is_complete;
                if (stop_thinking_idx >= static_cast<int>(stop_thinking_tokens.size()))
                    injecting_stop_thinking = false;
            }
            else
            {
                const int remaining = effective_max_tokens - completion_tokens;
                const int step_budget = thinking_budget_active ? 1 : remaining;
                const auto decode_start = SteadyClock::now();
                GenerationResult result = decodeStepWithBudget(runner_, step_budget);
                if (traceGeneratedTokensEnabled())
                {
                    LOG_INFO("[ChatCompletion/trace] stream decodeStep budget="
                             << step_budget
                             << " ms=" << std::fixed << std::setprecision(3)
                             << elapsedMs(decode_start)
                             << " tokens=" << result.tokens.size()
                             << " success=" << boolString(result.success()));
                }

                if (!result.success())
                {
                    json error_data = {{"error", result.error}};
                    emit_chunk(error_data, "stop");
                    chunk_cb("data: [DONE]\n\n");
                    response.ok = false;
                    response.http_status = 500;
                    json err = {{"error", {{"message", std::string("Decode failed: ") + result.error}, {"type", "server_error"}}}};
                    response.json_body = dumpJsonForHttp(err);
                    return response;
                }

                if (result.tokens.empty())
                {
                    finish_reason = "stop";
                    break;
                }

                step_tokens = result.tokens;
                step_complete = result.is_complete;
            }

            bool rebalance_applied = false;
            for (size_t token_idx = 0;
                 token_idx < step_tokens.size() && completion_tokens < effective_max_tokens;
                 ++token_idx)
            {
                int32_t next_token = step_tokens[token_idx];
                const bool is_final_returned_token = token_idx + 1 == step_tokens.size();

                completion_tokens++;

                if (tokenizer_.is_stop_token(next_token) ||
                    (step_complete && is_final_returned_token))
                {
                    finish_reason = "stop";
                    stop_generation = true;
                    break;
                }

                if (!rebalance_applied)
                {
                    if (!runner_.maybeApplyMoERebalance())
                        return emit_rebalance_error();
                    rebalance_applied = true;
                }

                std::string token_text = tokenizer_.decode_token(next_token);
                traceGeneratedToken("stream",
                                    completion_tokens - 1,
                                    next_token,
                                    token_text,
                                    step_forced);

                // Check thinking budget (before emitting)
                if (thinking_budget_active && use_think_split && splitter.inThinking() &&
                    !injecting_stop_thinking)
                {
                    thinking_tokens++;
                    if (thinking_tokens >= request.thinking_budget_tokens &&
                        !stop_thinking_tokens.empty())
                    {
                        LOG_DEBUG("[ChatCompletion/stream] Thinking budget exhausted ("
                                  << thinking_tokens << " tokens), injecting stop-thinking prompt");
                        injecting_stop_thinking = true;
                        stop_thinking_idx = 0;
                    }
                }

                if (use_think_split)
                {
                    auto split = splitter.process(token_text);
                    if (!split.text.empty())
                    {
                        accumulated_text += split.text;
                        if (!has_tools)
                        {
                            json delta;
                            delta[split.field] = split.text;
                            if (!emit_chunk(delta, nullptr))
                            {
                                stop_generation = true;
                                break;
                            }
                        }
                    }

                    // Check if the splitter transitioned and has buffered content
                    if (!splitter.inThinking())
                    {
                        auto flushed = splitter.flush();
                        if (!flushed.text.empty())
                        {
                            accumulated_text += flushed.text;
                            if (!has_tools)
                            {
                                json delta;
                                delta[flushed.field] = flushed.text;
                                if (!emit_chunk(delta, nullptr))
                                {
                                    stop_generation = true;
                                    break;
                                }
                            }
                        }
                    }
                }
                else
                {
                    accumulated_text += token_text;
                    if (!has_tools)
                    {
                        json delta;
                        delta["content"] = token_text;
                        if (!emit_chunk(delta, nullptr))
                        {
                            stop_generation = true;
                            break;
                        }
                    }
                }
            }
        }

        // Flush any remaining buffered thinking content
        if (use_think_split)
        {
            auto flushed = splitter.flush();
            if (!flushed.text.empty())
            {
                accumulated_text += flushed.text;
                if (!has_tools)
                {
                    json delta;
                    delta[flushed.field] = flushed.text;
                    emit_chunk(delta, nullptr);
                }
            }
        }

        runner_.flushStageTimeline();
        logRuntimeStateSummary(runner_, "streaming");

        // Post-generation: if tools were provided, parse for tool calls and emit
        if (has_tools)
        {
            ToolCallFormat format = runner_.getToolCallFormat();
            auto tool_result = parseToolCalls(accumulated_text, format);
            if (tool_result.hasToolCalls())
            {
                // Emit tool_calls deltas
                for (size_t ti = 0; ti < tool_result.tool_calls.size(); ++ti)
                {
                    const auto &tc = tool_result.tool_calls[ti];
                    json tc_delta = {
                        {"index", static_cast<int>(ti)},
                        {"id", tc.id},
                        {"type", "function"},
                        {"function", {{"name", tc.name}, {"arguments", tc.arguments}}}};
                    json delta = {{"tool_calls", json::array({tc_delta})}};
                    emit_chunk(delta, nullptr);
                }
                finish_reason = "tool_calls";
            }
            else
            {
                // No tool calls found — emit buffered content as single chunk
                if (!accumulated_text.empty())
                {
                    json delta = {{"content", accumulated_text}};
                    emit_chunk(delta, nullptr);
                }
            }
        }

        // Final chunk with finish_reason
        emit_chunk(json::object(), finish_reason.c_str());

        // [DONE] sentinel
        chunk_cb("data: [DONE]\n\n");

        response.ok = true;
        response.http_status = 200;
        return response;
    }
    catch (const std::exception &e)
    {
        return handleUnhandledRequestException("streaming request", e.what());
    }
    catch (...)
    {
        return handleUnhandledRequestException("streaming request", "unknown exception");
    }

    // =========================================================================
    // Convenience: parse + execute (routes to streaming if stream=true)
    // =========================================================================

    ChatCompletionResponse ChatCompletionHandler::handleRawRequest(
        const std::string &json_body,
        const StreamChunkCallback &stream_cb)
    try
    {
        ChatCompletionResponse error;
        auto request = parseRequest(json_body, error);
        if (!request)
            return error;

        if (request->stream && stream_cb)
            return handleStreamingRequest(*request, stream_cb);

        return handleRequest(*request);
    }
    catch (const std::exception &e)
    {
        return handleUnhandledRequestException("raw request", e.what());
    }
    catch (...)
    {
        return handleUnhandledRequestException("raw request", "unknown exception");
    }

} // namespace llaminar2
