/**
 * @file HttpRuntimeStats.cpp
 * @brief Passive lifetime metrics from existing terminal observations and host clocks.
 *
 * Short host critical sections publish complete outcomes. JSON formatting uses
 * a bounded copy outside that lock. Throughput and acceptance are ratios of sums,
 * not averages of request rates. Incomplete inference never invents token totals.
 */
#include "app/modes/HttpRuntimeStats.h"
#include <algorithm>
#include <stdexcept>

namespace llaminar2
{
    namespace
    {
        using json = nlohmann::json;
        /** @brief Undefined denominators are null, including an unmeasurably short interval. */
        json ratio(uint64_t numerator, double denominator)
        {
            return denominator > 0 ? json(static_cast<double>(numerator) / denominator) : json(nullptr);
        }
        /** @brief Preserve the distinction between an absent timing and zero elapsed time. */
        json optionalSeconds(const std::optional<double> &value)
        {
            return value ? json(*value) : json(nullptr);
        }
        /** @brief Reject out-of-order host timestamps instead of publishing negative timings. */
        double seconds(HttpRuntimeStats::Clock::time_point end, HttpRuntimeStats::Clock::time_point start)
        {
            if (end < start) throw std::logic_error("HTTP observation timestamps are out of order");
            return std::chrono::duration<double>(end - start).count();
        }
        /** @brief Logical unmatched prompt tokens; never multiply by TP width or padded work. */
        uint64_t uncached(int prompt, const RequestRuntimeSummary &summary)
        {
            const auto &p = summary.prefix_request;
            return prompt - (p.enabled && !p.bypassed ? p.matched_tokens : 0);
        }
        /** @brief Serialize one actual terminal MTP observation with token-weighted acceptance. */
        json mtpJson(const RequestRuntimeSummary &summary)
        {
            const auto &m = summary.mtp_request;
            const bool measured = m.enabled && !m.bypassed;
            return {{"enabled", m.enabled}, {"bypassed", m.bypassed}, {"bypass_reason", m.bypass_reason},
                {"verify_mode", m.verify_mode}, {"depth_policy", m.depth_policy_mode},
                {"depth", m.current_depth}, {"min_depth", m.min_depth}, {"max_depth", m.max_depth},
                {"depth_updates", m.depth_policy_updates}, {"draft_tokens", m.draft_steps},
                {"accepted_tokens", m.accepted_tokens}, {"rejected_tokens", m.rejected_tokens},
                {"verifier_runs", summary.mtp_verifier_runs},
                {"acceptance_rate", measured ? ratio(m.accepted_tokens,
                    static_cast<double>(m.accepted_tokens) + m.rejected_tokens) : json(nullptr)},
                {"stochastic_acceptance_rate", measured ?
                    ratio(m.stochastic_accepts, m.stochastic_accept_tests) : json(nullptr)}};
        }
    }

    HttpRuntimeStats::Request::Request(HttpRuntimeStats &owner, HttpRequestArrival arrival,
                                       Clock::time_point at)
        : owner_(&owner), arrival_(arrival), started_(at)
    {
        (void)seconds(at, arrival.received);
        std::lock_guard lock(owner.mutex_);
        if (owner.active_since_)
            throw std::logic_error("HTTP inference statistics require serialized handler ownership");
        epoch_ = owner.epoch_;
        sequence_ = ++owner.request_sequence_;
        owner.active_epoch_ = epoch_;
        owner.active_since_ = at;
        owner.active_phase_ = "setup";
        ++owner.counters_.started;
    }

    HttpRuntimeStats::Request::~Request()
    {
        Terminal terminal;
        terminal.handler_status = handler_status_;
        terminal.delivery = delivery_.value_or(Delivery::Failed);
        terminal.prompt_tokens = prompt_tokens_;
        terminal.completion_tokens = completion_tokens_;
        terminal.runtime = std::move(runtime_);
        // The scope is constructed before request-cache cleanup, so its last
        // destructor observes the complete handler lifetime, including cleanup.
        owner_->retire(*this, std::move(terminal), retirement_at_.value_or(Clock::now()));
    }

    void HttpRuntimeStats::Request::prefillStarted(Clock::time_point at)
    {
        if (delivery_ || prefill_started_) throw std::logic_error("HTTP prefill observation already started");
        (void)seconds(at, started_);
        prefill_started_ = at;
        owner_->setPhase("prefill");
    }

    void HttpRuntimeStats::Request::prefillFinished(Clock::time_point at)
    {
        if (delivery_ || !prefill_started_ || prefill_ended_)
            throw std::logic_error("HTTP prefill observation has no unfinished prefill");
        (void)seconds(at, *prefill_started_);
        prefill_ended_ = at;
        owner_->setPhase("decode");
    }

    void HttpRuntimeStats::Request::modelTokenObserved(std::optional<Clock::time_point> at)
    {
        if (delivery_ || !prefill_ended_ || decode_ended_)
            throw std::logic_error("HTTP model token observation is outside decode");
        if (!first_token_) first_token_ = at.value_or(Clock::now());
    }

    void HttpRuntimeStats::Request::outputPublished(std::optional<Clock::time_point> at)
    {
        if (delivery_) throw std::logic_error("HTTP output observation is retired");
        if (!first_output_) first_output_ = at.value_or(Clock::now());
    }

    void HttpRuntimeStats::Request::decodeFinished(Clock::time_point at)
    {
        if (delivery_ || !prefill_ended_ || decode_ended_)
            throw std::logic_error("HTTP decode observation has no unfinished decode");
        (void)seconds(at, *prefill_ended_);
        decode_ended_ = at;
        owner_->setPhase("finishing");
    }

    void HttpRuntimeStats::Request::complete(int prompt_tokens, int completion_tokens,
                                             const RequestRuntimeSummary &summary, Delivery delivery,
                                             std::optional<Clock::time_point> at)
    {
        if (delivery_) throw std::logic_error("HTTP request statistics were already published");
        const auto &p = summary.prefix_request;
        if (delivery == Delivery::Failed || prompt_tokens < 0 || completion_tokens < 0 ||
            p.requested_tokens < 0 || p.matched_tokens < 0 || p.matched_tokens > p.requested_tokens ||
            p.requested_tokens > prompt_tokens || (p.hit && p.partial_hit))
            throw std::invalid_argument("HTTP terminal statistics have inconsistent token counts or outcome");
        // Copy before selecting the outcome; failed allocation leaves this
        // scope unselected and its destructor records one failed request.
        runtime_ = summary;
        prompt_tokens_ = prompt_tokens;
        completion_tokens_ = completion_tokens;
        handler_status_ = 200;
        retirement_at_ = at;
        delivery_ = delivery;
    }

    void HttpRuntimeStats::Request::fail(int status, Delivery delivery,
                                         std::optional<Clock::time_point> at)
    {
        if (delivery_) throw std::logic_error("HTTP request statistics were already published");
        if (delivery == Delivery::Completed)
            throw std::invalid_argument("Incomplete inference cannot publish a successful summary");
        handler_status_ = status;
        retirement_at_ = at;
        delivery_ = delivery;
    }

    HttpRuntimeStats::Request HttpRuntimeStats::beginRequest(HttpRequestArrival arrival,
                                                            std::optional<Clock::time_point> at)
    {
        return Request(*this, arrival, at.value_or(Clock::now()));
    }

    void HttpRuntimeStats::setPhase(const char *phase)
    {
        std::lock_guard lock(mutex_);
        active_phase_ = phase;
    }

    void HttpRuntimeStats::Distribution::add(double value)
    {
        minimum = count ? std::min(minimum, value) : value;
        maximum = count ? std::max(maximum, value) : value;
        total += value;
        ++count;
    }

    nlohmann::json HttpRuntimeStats::Distribution::json() const
    {
        return {{"samples", count}, {"total_seconds", total},
            {"average_seconds", count ? nlohmann::json(total / count) : nlohmann::json(nullptr)},
            {"min_seconds", count ? nlohmann::json(minimum) : nlohmann::json(nullptr)},
            {"max_seconds", count ? nlohmann::json(maximum) : nlohmann::json(nullptr)}};
    }

    nlohmann::json HttpRuntimeStats::Timings::json() const
    {
        return {{"queue_seconds", queue}, {"service_seconds", service}, {"latency_seconds", latency},
            {"prefill_seconds", optionalSeconds(prefill)}, {"decode_seconds", optionalSeconds(decode)},
            {"ttft_seconds", optionalSeconds(ttft)}, {"first_output_seconds", optionalSeconds(first_output)}};
    }

    void HttpRuntimeStats::retire(const Request &request, Terminal terminal, Clock::time_point at)
    {
        auto &t = terminal.timings;
        t.queue = seconds(request.started_, request.arrival_.received);
        t.service = seconds(at, request.started_);
        t.latency = seconds(at, request.arrival_.received);
        if (request.first_token_) t.ttft = seconds(*request.first_token_, request.arrival_.received);
        if (request.first_output_) t.first_output = seconds(*request.first_output_, request.arrival_.received);
        if (request.prefill_ended_)
            t.prefill = seconds(*request.prefill_ended_, *request.prefill_started_);
        if (request.decode_ended_) t.decode = seconds(*request.decode_ended_, *request.prefill_ended_);
        if ((t.ttft && *t.ttft > t.latency) || (t.first_output && *t.first_output > t.latency))
            throw std::logic_error("HTTP first output cannot follow request retirement");

        std::lock_guard lock(mutex_);
        if (!active_since_) throw std::logic_error("HTTP terminal statistics have no active request");
        // Reset does not cancel or inspect an active model. Its old request
        // retires ownership, but cannot repopulate the new measurement epoch.
        if (request.epoch_ != epoch_)
        {
            active_since_.reset();
            active_phase_ = "idle";
            return;
        }
        terminal.sequence = request.sequence_;
        terminal.retired_at = at;
        last_ = std::move(terminal);
        const auto &last = *last_;
        const auto &timings = last.timings;
        auto &c = counters_;
        if (last.delivery == Delivery::Completed) ++c.completed;
        else if (last.delivery == Delivery::Disconnected) ++c.disconnected;
        else ++c.failed;
        c.queue.add(timings.queue);
        c.service.add(timings.service);
        c.latency.add(timings.latency);
        if (timings.ttft) c.ttft.add(*timings.ttft);
        if (timings.first_output) c.first_output.add(*timings.first_output);
        if (last.runtime)
        {
            c.prompt_tokens += last.prompt_tokens;
            c.completion_tokens += last.completion_tokens;
            // Pair the exact token numerator and elapsed denominator. Failed
            // work without terminal usage cannot dilute measured throughput.
            if (timings.prefill)
            {
                c.prefill.add(*timings.prefill);
                c.prefill_prompt_tokens += last.prompt_tokens;
                c.prefill_uncached_tokens += uncached(last.prompt_tokens, *last.runtime);
            }
            if (timings.decode)
            {
                c.decode.add(*timings.decode);
                c.decode_tokens += last.completion_tokens;
            }
            const auto &p = last.runtime->prefix_request;
            if (p.bypassed) ++c.prefix_bypassed;
            else if (p.enabled)
            {
                ++c.prefix_lookups;
                if (p.hit) ++c.prefix_full_hits;
                else if (p.partial_hit) ++c.prefix_partial_hits;
                else ++c.prefix_misses;
                c.prefix_requested_tokens += p.requested_tokens;
                c.prefix_matched_tokens += p.matched_tokens;
            }
            const auto &m = last.runtime->mtp_request;
            if (m.enabled && !m.bypassed)
            {
                ++c.mtp_requests;
                c.mtp_drafts += m.draft_steps;
                c.mtp_accepted += m.accepted_tokens;
                c.mtp_rejected += m.rejected_tokens;
                c.mtp_depth_updates += m.depth_policy_updates;
                c.mtp_verifier_runs += last.runtime->mtp_verifier_runs;
                c.mtp_stochastic_tests += m.stochastic_accept_tests;
                c.mtp_stochastic_accepts += m.stochastic_accepts;
            }
        }
        active_since_.reset();
        active_phase_ = "idle";
    }

    void HttpRuntimeStats::recordHttpResponse(int status)
    {
        std::lock_guard lock(mutex_);
        ++counters_.http_responses;
        if (status >= 100 && status <= 599) ++counters_.http_status_codes[status - 100];
        else ++counters_.http_unclassified;
    }

    uint64_t HttpRuntimeStats::reset()
    {
        std::lock_guard lock(mutex_);
        prefix_storage_stats_.reset();
        counters_ = {};
        last_.reset();
        reset_at_ = Clock::now();
        reset_unix_seconds_ = std::chrono::duration<double>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        return ++epoch_;
    }

    nlohmann::json HttpRuntimeStats::snapshot() const
    {
        Counters c;
        std::optional<Terminal> last;
        std::optional<Clock::time_point> active;
        const char *phase;
        uint64_t epoch, active_epoch, sequence;
        Clock::time_point reset_at;
        double reset_unix;
        json storage;
        {
            std::lock_guard lock(mutex_);
            c = counters_;
            last = last_;
            active = active_since_;
            phase = active_phase_;
            epoch = epoch_;
            active_epoch = active_epoch_;
            sequence = request_sequence_;
            reset_at = reset_at_;
            reset_unix = reset_unix_seconds_;
            storage = prefix_storage_stats_.snapshot();
        }
        const auto now = Clock::now();
        json status_codes = json::object();
        for (size_t i = 0; i < c.http_status_codes.size(); ++i)
            if (c.http_status_codes[i]) status_codes[std::to_string(i + 100)] = c.http_status_codes[i];
        json result{
            {"object", "llaminar.stats"}, {"schema_version", 1}, {"scope", "since_last_reset"},
            {"epoch", epoch}, {"collection_seconds", seconds(now, reset_at)},
            {"last_reset_unix_seconds", reset_unix},
            {"uptime_seconds", seconds(now, started_)},
            {"snapshot_unix_seconds", std::chrono::duration<double>(
                std::chrono::system_clock::now().time_since_epoch()).count()},
            {"requests", {{"started", c.started}, {"active", active ? 1 : 0},
                {"completed", c.completed}, {"failed", c.failed}, {"disconnected", c.disconnected},
                {"active_phase", phase}, {"active_sequence", active ? json(sequence) : json(nullptr)},
                {"active_in_current_epoch", active && active_epoch == epoch ? 1 : 0},
                {"active_seconds", active ? json(seconds(now, *active)) : json(nullptr)}}},
            {"http", {{"scope", "finished_http_responses_all_routes"}, {"responses", c.http_responses},
                {"status_codes", std::move(status_codes)}, {"unclassified", c.http_unclassified}}},
            {"tokens", {{"scope", "requests_with_terminal_summary"},
                {"prompt", c.prompt_tokens}, {"completion", c.completion_tokens}}},
            {"timings", {{"queue", c.queue.json()}, {"service", c.service.json()}, {"latency", c.latency.json()},
                {"prefill", c.prefill.json()}, {"decode", c.decode.json()},
                {"ttft", c.ttft.json()}, {"first_output", c.first_output.json()}}},
            {"throughput", {{"aggregation", "sum_tokens_over_sum_seconds"},
                {"prefill_prompt_tokens", c.prefill_prompt_tokens}, {"prefill_uncached_tokens", c.prefill_uncached_tokens},
                {"decode_tokens", c.decode_tokens},
                {"prefill_tokens_per_second", ratio(c.prefill_uncached_tokens, c.prefill.total)},
                {"effective_prompt_tokens_per_second", ratio(c.prefill_prompt_tokens, c.prefill.total)},
                {"decode_tokens_per_second", ratio(c.decode_tokens, c.decode.total)}}},
            {"prefix_cache", {{"lookups", c.prefix_lookups}, {"full_hits", c.prefix_full_hits},
                {"partial_hits", c.prefix_partial_hits}, {"misses", c.prefix_misses}, {"bypassed", c.prefix_bypassed},
                {"requested_tokens", c.prefix_requested_tokens}, {"matched_tokens", c.prefix_matched_tokens},
                {"request_hit_rate", ratio(c.prefix_full_hits + c.prefix_partial_hits, c.prefix_lookups)},
                {"token_reuse_rate", ratio(c.prefix_matched_tokens, c.prefix_requested_tokens)}}},
            {"mtp", {{"requests", c.mtp_requests}, {"draft_tokens", c.mtp_drafts},
                {"accepted_tokens", c.mtp_accepted}, {"rejected_tokens", c.mtp_rejected},
                {"depth_updates", c.mtp_depth_updates}, {"verifier_runs", c.mtp_verifier_runs},
                {"acceptance_rate", ratio(c.mtp_accepted, static_cast<double>(c.mtp_accepted) + c.mtp_rejected)},
                {"stochastic_accept_tests", c.mtp_stochastic_tests}, {"stochastic_accepts", c.mtp_stochastic_accepts},
                {"stochastic_acceptance_rate", ratio(c.mtp_stochastic_accepts, c.mtp_stochastic_tests)}}},
            {"last_request", nullptr}};
        result["prefix_cache"]["storage"] = std::move(storage);
        if (last)
        {
            json value{{"sequence", last->sequence}, {"age_seconds", seconds(now, last->retired_at)},
                {"handler_status", last->handler_status},
                {"outcome", last->delivery == Delivery::Completed ? "completed" :
                    last->delivery == Delivery::Disconnected ? "disconnected" : "failed"},
                {"timings", last->timings.json()}, {"prompt_tokens", nullptr}, {"completion_tokens", nullptr},
                {"throughput", nullptr}, {"prefix_cache", nullptr}, {"mtp", nullptr}};
            if (last->runtime)
            {
                const auto &p = last->runtime->prefix_request;
                value["prompt_tokens"] = last->prompt_tokens;
                value["completion_tokens"] = last->completion_tokens;
                value["throughput"] = {
                    {"prefill_uncached_tokens", uncached(last->prompt_tokens, *last->runtime)},
                    {"prefill_tokens_per_second", ratio(uncached(last->prompt_tokens, *last->runtime),
                        last->timings.prefill.value_or(0))},
                    {"effective_prompt_tokens_per_second", ratio(last->prompt_tokens, last->timings.prefill.value_or(0))},
                    {"decode_tokens_per_second", ratio(last->completion_tokens, last->timings.decode.value_or(0))}};
                value["prefix_cache"] = {{"enabled", p.enabled}, {"bypassed", p.bypassed},
                    {"bypass_reason", p.bypass_reason}, {"hit", p.hit}, {"partial_hit", p.partial_hit},
                    {"requested_tokens", p.requested_tokens}, {"matched_tokens", p.matched_tokens},
                    {"token_reuse_rate", p.enabled && !p.bypassed ?
                        ratio(p.matched_tokens, p.requested_tokens) : json(nullptr)},
                    {"storage_tier", p.storage_tier}};
                value["mtp"] = mtpJson(*last->runtime);
            }
            result["last_request"] = std::move(value);
        }
        return result;
    }
}
