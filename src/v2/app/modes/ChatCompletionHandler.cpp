/**
 * @file ChatCompletionHandler.cpp
 * @brief Production HTTP/SSE inference and shared incremental output parsing.
 *
 * Both response modes consume the same typed reasoning/answer lifecycle.
 * Budget-injected and model-generated close markers have identical framing
 * semantics. Field framing never owns generation termination: only runner/EOS,
 * request token limits, cancellation, and failure can end a response.
 * Parsing retains only a possible marker suffix, avoiding repeated scans of
 * the complete generated transcript during long requests.
 * Opt-in terminal token IDs retain already published runner output, including
 * stop tokens and forced thinking continuations. They never probe live state,
 * re-tokenize generated text, change decode batching, or enable snapshots.
 * Opt-in runtime summaries project the runner's already completed outcome;
 * JSON and logs share one snapshot, without querying GPUs or optional PerfStats.
 * Only opt-in JSON also exports one passive model-lifetime movement ledger;
 * ordinary logs do not copy the growing journal or advance maintenance.
 */

#include "app/modes/ChatCompletionHandler.h"
#include "app/modes/MoEMovementLedgerJson.h"
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

        bool traceGeneratedTokensEnabled()
        {
            return debugEnv().runtime_debug.trace_generated_tokens;
        }

        /**
         * @brief Return the longest suffix of @p text that could be the prefix
         *        of @p marker.
         */
        size_t partialMarkerSuffixLength(
            const std::string &text,
            const std::string &marker)
        {
            size_t match_len = 0;
            for (size_t len = 1; len < marker.size() && len <= text.size(); ++len)
            {
                if (text.substr(text.size() - len) == marker.substr(0, len))
                    match_len = len;
            }
            return match_len;
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

        /**
         * @brief Log completed observations without reading live inference state.
         * @param summary Request authority's validated terminal observations.
         * @param mode HTTP response mode used only as a diagnostic label.
         *
         * The deep prefix probe may synchronize whole devices. It must never
         * be used for routine logging, even at INFO; filtering after probing
         * still performs that work when the message itself is suppressed.
         */
        void logRuntimeStateSummary(const RequestRuntimeSummary &summary, const char *mode)
        {
            if (!Logger::getInstance().shouldLog(LogLevel::INFO))
                return;
            const auto &prefix = summary.prefix_request;
            if (prefix.enabled || prefix.bypassed)
            {
                LOG_INFO("[ChatCompletion] Prefix cache summary (" << mode << "): "
                         << "enabled=" << boolString(prefix.enabled)
                         << " hit=" << boolString(prefix.hit)
                         << " partial_hit=" << boolString(prefix.partial_hit)
                         << " requested_tokens=" << prefix.requested_tokens
                         << " matched_tokens=" << prefix.matched_tokens
                         << " matched_blocks=" << prefix.matched_blocks
                         << " tier=" << prefix.storage_tier
                         << " bypassed=" << boolString(prefix.bypassed)
                         << " bypass_reason=" << prefix.bypass_reason);
            }
            const auto &mtp = summary.mtp_request;
            if (mtp.enabled || mtp.bypassed || mtp.draft_steps != 0u)
            {
                LOG_INFO("[ChatCompletion] MTP summary (" << mode << "): "
                         << "enabled=" << boolString(mtp.enabled)
                         << " draft_steps=" << mtp.draft_steps
                         << " accepted_tokens=" << mtp.accepted_tokens
                         << " rejected_tokens=" << mtp.rejected_tokens
                         << " rollbacks=" << mtp.rollbacks
                         << " acceptance=" << std::fixed << std::setprecision(2)
                         << (mtp.acceptance_rate * 100.0) << "%"
                         << " verifier_runs=" << summary.mtp_verifier_runs
                         << " verifier_tokens=" << summary.mtp_verifier_token_count
                         << " verify_mode=" << mtp.verify_mode
                         << " depth_policy=" << mtp.depth_policy_mode
                         << " depth=" << mtp.current_depth
                         << " [" << mtp.min_depth << "," << mtp.max_depth << "]"
                         << " depth_updates=" << mtp.depth_policy_updates
                         << " last_depth_reason=" << mtp.last_depth_policy_reason
                         << " stochastic_accept_tests=" << mtp.stochastic_accept_tests
                         << " stochastic_acceptance=" << (mtp.stochastic_acceptance_rate * 100.0) << "%"
                         << " stochastic_residual_samples=" << mtp.stochastic_residual_samples
                         << " stochastic_terminal_samples=" << mtp.stochastic_terminal_samples
                         << " bypassed=" << boolString(mtp.bypassed)
                         << " bypass_reason=" << mtp.bypass_reason);
            }
        }

        /**
         * @brief Serialize one immutable terminal observation without recomputing facts.
         * @param summary Runner-owned outcome sampled after the request completed.
         * @return Versioned, optional HTTP extension; no cache contents or live state.
         *
         * Prefix admission epochs explain legitimate movement invalidation. They
         * come from the prefix authority, not inferred from profiling counters.
         */
        json runtimeSummaryJson(const RequestRuntimeSummary &summary)
        {
            const auto &prefix = summary.prefix_request;
            const auto &mtp = summary.mtp_request;
            return {{"schema", 1}, {"prefix_cache", {
                {"enabled", prefix.enabled}, {"bypassed", prefix.bypassed},
                {"bypass_reason", prefix.bypass_reason}, {"hit", prefix.hit},
                {"partial_hit", prefix.partial_hit}, {"requested_tokens", prefix.requested_tokens},
                {"matched_tokens", prefix.matched_tokens}, {"matched_blocks", prefix.matched_blocks},
                {"terminal_logits_restored", prefix.terminal_logits_restored},
                {"terminal_hidden_restored", prefix.terminal_hidden_restored},
                {"mtp_state_restored", prefix.mtp_state_restored},
                {"hybrid_state_restored", prefix.hybrid_state_restored},
                {"storage_tier", prefix.storage_tier},
                {"admission_epoch_earliest", prefix.admission_placement_epochs.earliest()},
                {"admission_epoch_latest", prefix.admission_placement_epochs.latest()},
                {"completion_movement_epoch", prefix.completion_movement_epoch}}},
                {"mtp", {{"enabled", mtp.enabled}, {"bypassed", mtp.bypassed},
                {"bypass_reason", mtp.bypass_reason}, {"verify_mode", mtp.verify_mode},
                {"stochastic_verify", mtp.stochastic_verify},
                {"adaptive_depth_enabled", mtp.adaptive_depth_enabled},
                {"depth_policy_mode", mtp.depth_policy_mode}, {"current_depth", mtp.current_depth},
                {"min_depth", mtp.min_depth}, {"max_depth", mtp.max_depth},
                {"depth_policy_updates", mtp.depth_policy_updates},
                {"last_depth_policy_reason", mtp.last_depth_policy_reason},
                {"draft_steps", mtp.draft_steps}, {"accepted_tokens", mtp.accepted_tokens},
                {"rejected_tokens", mtp.rejected_tokens}, {"rollbacks", mtp.rollbacks},
                {"acceptance_rate", mtp.acceptance_rate},
                {"verifier_runs", summary.mtp_verifier_runs},
                {"verifier_token_count", summary.mtp_verifier_token_count},
                {"stochastic_accept_tests", mtp.stochastic_accept_tests},
                {"stochastic_accepts", mtp.stochastic_accepts},
                {"stochastic_residual_samples", mtp.stochastic_residual_samples},
                {"stochastic_terminal_samples", mtp.stochastic_terminal_samples},
                {"stochastic_acceptance_rate", mtp.stochastic_acceptance_rate}}}};
        }

        bool runChatMoERebalanceMaintenance(
            IOrchestrationRunner &runner,
            uint64_t committed_tokens)
        {
            return runner.maybeApplyMoERebalance(committed_tokens);
        }
    }

    // =========================================================================
    // StreamingThinkSplitter
    // =========================================================================

    StreamingThinkSplitter::StreamingThinkSplitter(const std::string &end_tag)
        : end_tag_(end_tag), phase_(end_tag.empty() ? Phase::Content : Phase::Reasoning)
    {
    }

    StreamingThinkSplitter::StreamingThinkSplitter()
        : end_tag_(), phase_(Phase::Content)
    {
    }

    StreamingThinkSplitter::SplitResult StreamingThinkSplitter::process(const std::string &token_text)
    {
        if (end_tag_.empty())
            return {"content", token_text};
        buffer_ += token_text;
        if (phase_ != Phase::Reasoning)
            return drainContent(/*terminal=*/false);

        const auto pos = buffer_.find(end_tag_);
        if (pos != std::string::npos)
        {
            // A single result cannot emit both fields. Retain the answer tail
            // for the next piece/flush, which uses the same marker validation.
            std::string reasoning_part = buffer_.substr(0, pos);
            buffer_.erase(0, pos + end_tag_.size());
            phase_ = Phase::AwaitingAnswer;
            if (!reasoning_part.empty())
                return {"reasoning_content", reasoning_part};
            return drainContent(/*terminal=*/false);
        }

        const size_t held = partialMarkerSuffixLength(buffer_, end_tag_);
        std::string safe = buffer_.substr(0, buffer_.size() - held);
        buffer_.erase(0, buffer_.size() - held);
        return {"reasoning_content", safe};
    }

    StreamingThinkSplitter::SplitResult StreamingThinkSplitter::drainContent(bool terminal)
    {
        // A model can continue reasoning after the forced budget close before
        // emitting its natural close and final answer. Neither close is EOS.
        // Scan forward once, suppressing delimiters while preserving every
        // ordinary byte; erasing only the consumed prefix also bounds copying.
        std::string content;
        size_t cursor = 0;
        while (cursor < buffer_.size())
        {
            if (phase_ == Phase::AwaitingAnswer)
            {
                const size_t first = buffer_.find_first_not_of(" \t\n\r", cursor);
                cursor = first == std::string::npos ? buffer_.size() : first;
            }
            const size_t marker = buffer_.find(end_tag_, cursor);
            if (marker != std::string::npos)
            {
                if (marker > cursor)
                {
                    content.append(buffer_, cursor, marker - cursor);
                    phase_ = Phase::Content;
                }
                cursor = marker + end_tag_.size();
                continue;
            }

            // Retain only a possible delimiter prefix across tokenizer pieces.
            // End-of-input flush preserves an incomplete literal fragment.
            const size_t held = terminal ? 0u : std::min(
                buffer_.size() - cursor, partialMarkerSuffixLength(buffer_, end_tag_));
            const size_t end = buffer_.size() - held;
            if (end > cursor)
            {
                content.append(buffer_, cursor, end - cursor);
                phase_ = Phase::Content;
            }
            cursor = end;
            break;
        }
        buffer_.erase(0, cursor);
        return {"content", content};
    }

    StreamingThinkSplitter::SplitResult StreamingThinkSplitter::flush()
    {
        if (phase_ != Phase::Reasoning && !end_tag_.empty())
            return drainContent(/*terminal=*/true);
        std::string result = std::move(buffer_);
        buffer_.clear();
        return {inThinking() ? "reasoning_content" : "content", result};
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

        if (body.contains("return_token_ids"))
        {
            if (!body["return_token_ids"].is_boolean())
            {
                error_out.ok = false;
                error_out.http_status = 400;
                error_out.json_body = dumpJsonForHttp({{"error", {
                    {"message", "return_token_ids must be a boolean"},
                    {"type", "invalid_request_error"}}}});
                return std::nullopt;
            }
            request.token_output = body["return_token_ids"].get<bool>()
                ? CompletionTokenOutput::TextAndIds : CompletionTokenOutput::TextOnly;
        }
        if (request.stream && request.token_output == CompletionTokenOutput::TextAndIds)
        {
            error_out.ok = false;
            error_out.http_status = 400;
            error_out.json_body = dumpJsonForHttp({{"error", {
                {"message", "return_token_ids is implemented for non-streaming responses only"},
                {"type", "invalid_request_error"}}}});
            return std::nullopt;
        }

        if (body.contains("return_runtime_summary"))
        {
            if (!body["return_runtime_summary"].is_boolean())
            {
                error_out.http_status = 400;
                error_out.json_body = dumpJsonForHttp({{"error", {
                    {"message", "return_runtime_summary must be a boolean"},
                    {"type", "invalid_request_error"}}}});
                return std::nullopt;
            }
            request.runtime_output = body["return_runtime_summary"].get<bool>()
                ? CompletionRuntimeOutput::Include : CompletionRuntimeOutput::Omit;
        }
        if (request.stream && request.runtime_output == CompletionRuntimeOutput::Include)
        {
            error_out.http_status = 400;
            error_out.json_body = dumpJsonForHttp({{"error", {
                {"message", "return_runtime_summary is implemented for non-streaming responses only"},
                {"type", "invalid_request_error"}}}});
            return std::nullopt;
        }

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
        /*
         * Stop policy is part of request admission, just like sampling policy.
         * In particular, a captured MTP verifier must see ChatML terminators on
         * device before prefill publishes the first decode boundary.  The HTTP
         * response loop may still recognize the terminal token after the final
         * result is materialized, but it is not allowed to become an alternate
         * authority that clips a transaction after later verifier rows have
         * already mutated KV or recurrent state.
         */
        runner_.setStopTokens(tokenizer_.stop_tokens());

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
        std::string content;
        std::string reasoning_content;
        // Allocate only for an explicitly requested terminal observation. The
        // runner already publishes these IDs for ordinary text output; keeping
        // them requires no extra device transfer or speculative-state access.
        std::vector<int32_t> completion_token_ids;
        // Grow with actual output, not an untrusted max_tokens reservation:
        // requests can ask for more tokens than are available before EOS.
        int completion_tokens = 0;
        std::string finish_reason = "length";
        std::string thinking_end_tag;
        if (request.enable_thinking && tokenizer_.hasChatTemplate())
        {
            const auto &chat_template = tokenizer_.getChatTemplate();
            if (chat_template.isThinkingModel())
                thinking_end_tag = chat_template.thinkingEndTag();
        }
        StreamingThinkSplitter splitter(thinking_end_tag);
        const auto append_output = [&](const StreamingThinkSplitter::SplitResult &part)
        {
            (part.field == "reasoning_content" ? reasoning_content : content) += part.text;
        };

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
                if (request.token_output == CompletionTokenOutput::TextAndIds)
                    completion_token_ids.push_back(next_token);
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
                const auto part = splitter.process(token_text);
                append_output(part);
            }

            if (!step_tokens.empty())
            {
                if (!runChatMoERebalanceMaintenance(
                        runner_, step_tokens.size()))
                    return rebalance_error();
            }
        }

        runner_.flushStageTimeline();
        // Read once after the terminal result, before RAII request cleanup.
        // Logging and the opt-in response see the same immutable outcome.
        std::optional<RequestRuntimeSummary> runtime_summary;
        if (request.runtime_output == CompletionRuntimeOutput::Include ||
            Logger::getInstance().shouldLog(LogLevel::INFO))
        {
            runtime_summary = runner_.requestRuntimeSummary();
            logRuntimeStateSummary(*runtime_summary, "non-streaming");
        }

        // HTTP and SSE share marker decisions, including token-split closes;
        // non-streaming only accumulates the same safe fields into one response.
        append_output(splitter.flush());

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

        if (request.token_output == CompletionTokenOutput::TextAndIds)
        {
            // Preserve the tokenizer's actual prompt and the runner's ordered
            // completion, including EOS. Text framing may hide either, so
            // re-encoding the displayed answer would not be equivalent.
            json_response["token_ids"] = {
                {"prompt", input_ids}, {"completion", completion_token_ids}};
        }

        if (request.runtime_output == CompletionRuntimeOutput::Include)
        {
            json_response["runtime_summary"] = runtimeSummaryJson(*runtime_summary);
            // This is an immutable, already-published ledger, not a request
            // reset or maintenance join. Keep it out of routine INFO logging:
            // only an explicit terminal representation pays the copy cost.
            json_response["runtime_summary"]["expert_movement"] =
                moeMovementLedgerJson(runner_.moeOptimizationMovementLedger());
            json_response["runtime_summary"]["expert_movement_topology"] =
                moeMovementTopologyJson(runner_.moeOptimizationMovementTopology());
        }

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
        if (request.runtime_output == CompletionRuntimeOutput::Include)
        {
            // Typed callers share JSON admission; reject before touching state.
            response.http_status = 400;
            response.json_body = dumpJsonForHttp({{"error", {
                {"message", "return_runtime_summary is implemented for non-streaming responses only"},
                {"type", "invalid_request_error"}}}});
            return response;
        }
        if (request.token_output == CompletionTokenOutput::TextAndIds)
        {
            // Direct typed callers must obey the same admission contract as
            // JSON callers, before any request reset or GPU work is submitted.
            response.http_status = 400;
            response.json_body = dumpJsonForHttp({{"error", {
                {"message", "return_token_ids is implemented for non-streaming responses only"},
                {"type", "invalid_request_error"}}}});
            return response;
        }
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

            if (!runChatMoERebalanceMaintenance(
                    runner_, step_tokens.size()))
                return emit_rebalance_error();

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

                std::string token_text = tokenizer_.decode_token(next_token);
                traceGeneratedToken("stream",
                                    completion_tokens - 1,
                                    next_token,
                                    token_text,
                                    step_forced);

                if (use_think_split)
                {
                    auto split = splitter.process(token_text);
                    // Interpret closure before counting budget work. Forced
                    // control tokens are not sampled reasoning, including the
                    // last injected piece that has already ended injection.
                    if (thinking_budget_active && !step_forced && splitter.inThinking())
                    {
                        ++thinking_tokens;
                        if (thinking_tokens >= request.thinking_budget_tokens &&
                            !stop_thinking_tokens.empty())
                        {
                            LOG_DEBUG("[ChatCompletion/stream] Thinking budget exhausted ("
                                      << thinking_tokens << " tokens), injecting stop-thinking prompt");
                            injecting_stop_thinking = true;
                            stop_thinking_idx = 0;
                        }
                    }
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

                    // Drain a buffered answer tail without declaring end-of-input.
                    // A terminal flush here would expose a split closing marker
                    // before its next tokenizer piece can complete the match.
                    if (!splitter.inThinking())
                    {
                        auto flushed = splitter.process("");
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
        if (Logger::getInstance().shouldLog(LogLevel::INFO))
            logRuntimeStateSummary(runner_.requestRuntimeSummary(), "streaming");

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
