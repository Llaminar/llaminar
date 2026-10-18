/**
 * @file HttpRuntimeStats.h
 * @brief Bounded, passive HTTP request accounting and monotonic host timings.
 *
 * One serialized handler publishes its existing immutable terminal summary.
 * Readers copy bounded host data without visiting a runner, synchronizing a
 * device, or retaining prompts, token arrays or request history. HTTP response
 * codes are separate from generation outcomes: SSE errors can follow HTTP 200.
 */
#pragma once

#include "execution/prefix_cache/PrefixCacheStats.h"
#include "app/modes/HttpPrefixCacheStats.h"
#include <nlohmann/json.hpp>
#include <array>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <optional>

namespace llaminar2
{
    /** @brief Arrival at the chat route, before parsing and inference admission. */
    struct HttpRequestArrival
    {
        std::chrono::steady_clock::time_point received = std::chrono::steady_clock::now();
    };

    /** @brief Accumulate completed observations independently of inference ownership. */
    class HttpRuntimeStats
    {
    public:
        using Clock = std::chrono::steady_clock;
        /** @brief Freeze passive cache publishers before concurrent HTTP serving starts. */
        explicit HttpRuntimeStats(PrefixCacheTelemetrySources sources = {})
            : prefix_storage_stats_(std::move(sources)) {}
        /** @brief Separate completed delivery, failed generation and departed consumers. */
        enum class Delivery { Completed, Failed, Disconnected };

        /** @brief Retire exactly one handler call, including early returns and exceptions. */
        class Request
        {
        public:
            /** @brief Count an unfinished scope as failed without querying the runner. */
            ~Request();
            Request(const Request &) = delete;
            Request &operator=(const Request &) = delete;

            /** @brief Begin the prefill call, after template rendering and admission. */
            void prefillStarted(Clock::time_point at = Clock::now());
            /** @brief End successful prefill and begin the host decode/publication interval. */
            void prefillFinished(Clock::time_point at = Clock::now());
            /** @brief Observe the first nonterminal, non-forced model token, including reasoning. */
            void modelTokenObserved(std::optional<Clock::time_point> at = std::nullopt);
            /** @brief Observe the first successful SSE text/reasoning/tool delta, excluding metadata. */
            void outputPublished(std::optional<Clock::time_point> at = std::nullopt);
            /** @brief End decode before terminal diagnostics, cleanup and response formatting. */
            void decodeFinished(Clock::time_point at = Clock::now());
            /**
             * @brief Select one terminal outcome; scope exit publishes it after request cleanup.
             * @param prompt_tokens Authoritative encoded prompt length.
             * @param completion_tokens Committed output count, including EOS.
             * @param summary Runner's existing immutable terminal observation.
             * @param delivery Completion or disconnection; failures use fail().
             * @param at Optional retirement time for deterministic proofs; normally measured after cleanup.
             */
            void complete(int prompt_tokens, int completion_tokens,
                          const RequestRuntimeSummary &summary, Delivery delivery,
                          std::optional<Clock::time_point> at = std::nullopt);
            /**
             * @brief Select an early failure/disconnect without inventing terminal counters.
             * @param status Handler result code, distinct from already-sent SSE HTTP headers.
             * @param delivery Failure or disconnection.
             * @param at Optional retirement time for deterministic proofs; normally measured at scope exit.
             */
            void fail(int status = 500, Delivery delivery = Delivery::Failed,
                      std::optional<Clock::time_point> at = std::nullopt);

        private:
            friend class HttpRuntimeStats;
            /** @brief Admit the sole handler call under the lifetime counter authority. */
            Request(HttpRuntimeStats &owner, HttpRequestArrival arrival, Clock::time_point at);
            HttpRuntimeStats *owner_;
            uint64_t epoch_ = 0, sequence_ = 0;
            std::optional<Delivery> delivery_;
            int handler_status_ = 500, prompt_tokens_ = 0, completion_tokens_ = 0;
            std::optional<RequestRuntimeSummary> runtime_;
            std::optional<Clock::time_point> retirement_at_;
            HttpRequestArrival arrival_;
            Clock::time_point started_;
            std::optional<Clock::time_point> prefill_started_, prefill_ended_, decode_ended_;
            std::optional<Clock::time_point> first_token_, first_output_;
        };

        /** @brief Begin a serialized call; arrival preserves queue time across HTTP workers. */
        [[nodiscard]] Request beginRequest(HttpRequestArrival arrival = {},
                                           std::optional<Clock::time_point> at = std::nullopt);
        /** @brief Count a response's actual HTTP code after its transport has finished. */
        void recordHttpResponse(int status);
        /**
         * @brief Begin a fresh measurement period without interrupting active inference.
         * @return New monotonic epoch; requests started before it cannot publish into its totals.
         */
        uint64_t reset();
        /** @brief Copy bounded observations; undefined rates are null, never NaN. */
        [[nodiscard]] nlohmann::json snapshot() const;

    private:
        /** @brief Constant-memory timing distribution, without retained request samples. */
        struct Distribution
        {
            uint64_t count = 0;
            double total = 0, minimum = 0, maximum = 0;
            /** @brief Accumulate one nonnegative monotonic measurement. */
            void add(double seconds);
            /** @brief Serialize totals and extrema, using null for empty averages. */
            [[nodiscard]] nlohmann::json json() const;
        };
        /** @brief Host timing observations retained for one retired call. */
        struct Timings
        {
            double queue = 0, service = 0, latency = 0;
            std::optional<double> prefill, decode, ttft, first_output;
            /** @brief Serialize explicitly named timing scopes in seconds. */
            [[nodiscard]] nlohmann::json json() const;
        };
        /** @brief Immutable last outcome; absent runtime means incomplete inference. */
        struct Terminal
        {
            uint64_t sequence = 0;
            Clock::time_point retired_at;
            int handler_status = 500;
            Delivery delivery = Delivery::Failed;
            Timings timings;
            int prompt_tokens = 0, completion_tokens = 0;
            std::optional<RequestRuntimeSummary> runtime;
        };
        /** @brief Fixed-size lifetime observations; mutations occur only under mutex_. */
        struct Counters
        {
            uint64_t started = 0, completed = 0, failed = 0, disconnected = 0;
            uint64_t prompt_tokens = 0, completion_tokens = 0;
            uint64_t prefill_prompt_tokens = 0, prefill_uncached_tokens = 0, decode_tokens = 0;
            uint64_t prefix_lookups = 0, prefix_full_hits = 0, prefix_partial_hits = 0;
            uint64_t prefix_misses = 0, prefix_bypassed = 0;
            uint64_t prefix_requested_tokens = 0, prefix_matched_tokens = 0;
            uint64_t mtp_requests = 0, mtp_drafts = 0, mtp_accepted = 0, mtp_rejected = 0;
            uint64_t mtp_depth_updates = 0, mtp_verifier_runs = 0;
            uint64_t mtp_stochastic_tests = 0, mtp_stochastic_accepts = 0;
            Distribution queue, service, latency, prefill, decode, ttft, first_output;
            uint64_t http_responses = 0, http_unclassified = 0;
            std::array<uint64_t, 500> http_status_codes{};
        };
        /** @brief Atomically retire a scope, publishing only validated terminal observations. */
        void retire(const Request &request, Terminal terminal, Clock::time_point at);
        /** @brief Publish a host phase boundary, never a model execution decision. */
        void setPhase(const char *phase);

        const Clock::time_point started_ = Clock::now();
        mutable std::mutex mutex_;
        Counters counters_;
        HttpPrefixCacheStats prefix_storage_stats_; ///< Same mutex and epoch as HTTP counters.
        uint64_t epoch_ = 0, request_sequence_ = 0, active_epoch_ = 0;
        Clock::time_point reset_at_ = started_;
        double reset_unix_seconds_ = std::chrono::duration<double>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        std::optional<Clock::time_point> active_since_;
        const char *active_phase_ = "idle";
        std::optional<Terminal> last_;
    };
}
