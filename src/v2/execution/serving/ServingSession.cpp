/**
 * @file ServingSession.cpp
 * @brief Continuous-batching session step logic (see ServingSession.h).
 */

#include "execution/serving/ServingSession.h"
#include "utils/PerfStatsCollector.h"

#include <algorithm>
#include <random>

namespace llaminar2
{
    ServingSession::ServingSession(IServingRowExecutor &rows, int context, int vocab)
        : executor_(rows), context_(context), vocab_(vocab)
    {
    }

    bool ServingSession::begin(int capacity, int slots_to_reset, std::string &error,
                               std::shared_ptr<PhysicalMemorySuballocationLease> decision_storage)
    {
        if (active_)
        {
            error = "Serving batch session is already open";
            return false;
        }
        if (capacity <= 0 || slots_to_reset < capacity || vocab_ <= 0 || context_ <= 0)
        {
            error = "Serving batch capacity, vocabulary or context is invalid";
            return false;
        }
        for (int slot = 0; slot < slots_to_reset; ++slot)
            if (!executor_.resetServingRow(slot))
            {
                error = "Serving batch could not reset slot " + std::to_string(slot);
                return false;
            }
        const std::size_t decision_floats = static_cast<std::size_t>(capacity) * static_cast<std::size_t>(vocab_);
        if (decision_storage &&
            (!decision_storage->valid() || decision_storage->bytes() < decision_floats * sizeof(float)))
        {
            error = "Serving decision storage claim does not cover " + std::to_string(decision_floats * sizeof(float)) +
                    " bytes";
            return false;
        }
        rows_.clear();
        rows_.reserve(static_cast<std::size_t>(capacity));
        slot_busy_.assign(static_cast<std::size_t>(capacity), false);
        // The claim is held for the store's whole lifetime (released in end()).
        decision_storage_ = std::move(decision_storage);
        decision_logits_.assign(decision_floats, 0.0f);
        queue_.clear();
        cancellations_.clear();
        pending_grants_.clear();
        pending_forced_.clear();
        failure_.clear();
        capacity_ = capacity;
        active_ = true;
        return true;
    }

    bool ServingSession::submit(ServingBatchRequest request, std::string &error)
    {
        if (!active_)
            error = "Serving batch session is not open";
        else if (request.prompt.empty() || request.max_new_tokens <= 0)
            error = "Serving request requires a prompt and a positive output budget";
        else if (static_cast<int>(request.prompt.size()) >= context_)
            error = "Serving request prompt exceeds the admitted context";
        else if (!failure_.empty())
            error = "Serving session failed: " + failure_;
        else if (std::any_of(rows_.begin(), rows_.end(), [&](const Row &row) {
                     return row.request.id == request.id;
                 }) || std::any_of(queue_.begin(), queue_.end(), [&](const ServingBatchRequest &queued) {
                     return queued.id == request.id;
                 }))
            error = "Serving request id is already live: " + std::to_string(request.id);
        else
        {
            // Grants, forces and cancellation address this identity, so accepting
            // another live owner would make those operations ambiguous.
            queue_.push_back(std::move(request));
            return true;
        }
        return false;
    }

    void ServingSession::cancel(std::uint64_t id) { cancellations_.push_back(id); }

    void ServingSession::grant(std::uint64_t id, int tokens)
    {
        if (tokens <= 0)
            return;
        for (auto &row : rows_)
            if (row.request.id == id)
            {
                row.grant += tokens;
                return;
            }
        for (const auto &queued : queue_)
            if (queued.id == id)
            {
                // Admission carries the grant: a queued request has no row yet.
                pending_grants_[id] += tokens;
                return;
            }
    }

    void ServingSession::force(std::uint64_t id, int32_t token)
    {
        for (auto &row : rows_)
            if (row.request.id == id)
            {
                row.forced.push_back(token);
                return;
            }
        // Unknown or finished ids are ignored (like grants); only a queued
        // request carries forced tokens into admission.
        for (const auto &queued : queue_)
            if (queued.id == id)
            {
                pending_forced_[id].push_back(token);
                return;
            }
    }

    bool ServingSession::hasRunnableWork() const
    {
        if (!active_)
            return false;
        if (!cancellations_.empty())
            return true;
        if (!queue_.empty() && std::find(slot_busy_.begin(), slot_busy_.end(), false) != slot_busy_.end())
            return true;
        for (const auto &row : rows_)
        {
            const bool may_decide = !row.forced.empty() || !row.request.paced || row.grant > 0;
            if (row.cancelled || row.prompt_consumed < row.request.prompt.size() ||
                ((row.has_logits || row.needs_feed) && may_decide))
                return true;
        }
        return false;
    }

    void ServingSession::end()
    {
        // Closing retires every slot; the next begin() resets all slots again
        // and fails if any reset fails, so no stale state can be admitted.
        for (const auto &row : rows_)
            (void)executor_.resetServingRow(row.slot);
        rows_.clear();
        slot_busy_.clear();
        queue_.clear();
        cancellations_.clear();
        pending_grants_.clear();
        pending_forced_.clear();
        active_ = false;
        capacity_ = 0;
        failure_.clear();
        decision_logits_.clear();
        decision_logits_.shrink_to_fit();
        decision_storage_.reset(); // after the store is freed
    }

    std::vector<ServingBatchEvent> ServingSession::sealAll(const std::string &reason,
                                                           std::vector<ServingBatchEvent> events)
    {
        // Fail every resident and queued request exactly once, reset what can
        // be reset, and admit nothing more.
        failure_ = reason;
        PerfStatsCollector::addCounter("serving", "seals", 1.0, "serving_rows", "cpu");
        PerfStatsCollector::addCounter("serving", "sealed_rows", static_cast<double>(rows_.size() + queue_.size()),
                                       "serving_rows", "cpu");
        auto fail = [&](std::uint64_t id, int prompt_tokens, int completion_tokens) {
            ServingBatchEvent event;
            event.id = id;
            event.finish = ServingFinishReason::Error;
            event.error = failure_;
            event.prompt_tokens = prompt_tokens;
            event.completion_tokens = completion_tokens;
            events.push_back(std::move(event));
        };
        for (const auto &row : rows_)
        {
            fail(row.request.id, static_cast<int>(row.request.prompt.size()), row.completion_tokens);
            (void)executor_.resetServingRow(row.slot);
        }
        for (const auto &queued : queue_)
            fail(queued.id, static_cast<int>(queued.prompt.size()), 0);
        rows_.clear();
        queue_.clear();
        pending_grants_.clear();
        pending_forced_.clear();
        std::fill(slot_busy_.begin(), slot_busy_.end(), false);
        if (recorder_)
            recorder_->stepped(events);
        return events;
    }

    std::vector<ServingBatchEvent> ServingSession::step()
    {
        std::vector<ServingBatchEvent> events;
        if (!active_)
            return events;
        PerfStatsCollector::addCounter("serving", "steps", 1.0, "serving_rows", "cpu");

        // Per-request output of this step, merged into one event per request.
        std::map<std::uint64_t, ServingBatchEvent> step_events;
        auto event_of = [&](const Row &row) -> ServingBatchEvent & {
            auto &event = step_events[row.request.id];
            event.id = row.request.id;
            event.prompt_tokens = static_cast<int>(row.request.prompt.size());
            event.cached_prompt_tokens = row.cached_prompt_tokens;
            event.prefix_cache_enabled = row.prefix_admission.cache_enabled && row.prefix_admission.supported;
            event.prefix_matched_blocks =
                row.cached_prompt_tokens > 0 && row.prefix_admission.block_size > 0
                    ? (row.cached_prompt_tokens + row.prefix_admission.block_size - 1) / row.prefix_admission.block_size
                    : 0;
            event.completion_tokens = row.completion_tokens;
            return event;
        };
        auto publish = [&]() {
            for (auto &[id, event] : step_events)
                events.push_back(std::move(event));
            if (recorder_)
                recorder_->stepped(events);
            return events;
        };

        // 1. Cancellation of a queued request completes without a slot; a
        //    resident request retires at this step's retirement boundary.
        for (const std::uint64_t id : cancellations_)
        {
            for (auto it = queue_.begin(); it != queue_.end(); ++it)
                if (it->id == id)
                {
                    ServingBatchEvent event;
                    event.id = id;
                    event.finish = ServingFinishReason::Cancelled;
                    event.prompt_tokens = static_cast<int>(it->prompt.size());
                    events.push_back(std::move(event));
                    queue_.erase(it);
                    pending_grants_.erase(id);
                    pending_forced_.erase(id);
                    break;
                }
            for (auto &row : rows_)
                if (row.request.id == id)
                    row.cancelled = true;
        }
        cancellations_.clear();

        // 2. Admission: new requests take free slots while resident requests
        //    keep decoding. Each owns an immutable resolved seed
        //    (position-keyed draws), so batching never changes its stream.
        for (int slot = 0; slot < capacity_ && !queue_.empty(); ++slot)
        {
            if (slot_busy_[static_cast<std::size_t>(slot)])
                continue;
            Row row;
            row.request = std::move(queue_.front());
            queue_.pop_front();
            row.slot = slot;
            if (row.request.sampling.seed == 0)
            {
                std::random_device entropy;
                unsigned int seed = 0;
                while (seed == 0)
                    seed = entropy();
                row.request.sampling.seed = seed;
            }
            row.sampler = Sampler(row.request.sampling.seed);
            if (row.request.sampling.repetition_penalty != 1.0f)
                for (const int32_t token : row.request.prompt)
                    row.sampler.record_prompt_token(token);
            if (auto g = pending_grants_.find(row.request.id); g != pending_grants_.end())
            {
                row.grant = g->second;
                pending_grants_.erase(g);
            }
            if (auto f = pending_forced_.find(row.request.id); f != pending_forced_.end())
            {
                row.forced.assign(f->second.begin(), f->second.end());
                pending_forced_.erase(f);
            }
            // Reuse the longest cached prefix in this slot only; at least one
            // prompt token is still computed by the row's own forward.
            const int restored = executor_.restoreServingRowPrefix(slot, row.request.prompt, row.prefix_admission);
            if (restored < 0)
            {
                ServingBatchEvent event;
                event.id = row.request.id;
                event.finish = ServingFinishReason::Error;
                event.error = "Serving prefix restore failed";
                event.prompt_tokens = static_cast<int>(row.request.prompt.size());
                events.push_back(std::move(event));
                if (!executor_.resetServingRow(slot))
                    return sealAll("Serving slot " + std::to_string(slot) +
                                   " could not be reset after a failed prefix restore", events);
                --slot; // the slot is still free for the next queued request
                continue;
            }
            row.prompt_consumed = static_cast<std::size_t>(restored);
            row.cached_tokens = restored;
            row.cached_prompt_tokens = restored;
            if (recorder_)
                recorder_->admitted(row.request, slot, restored);
            PerfStatsCollector::addCounter("serving", "row_admissions", 1.0, "serving_rows", "cpu");
            slot_busy_[static_cast<std::size_t>(slot)] = true;
            rows_.push_back(std::move(row));
        }

        std::vector<ServingFinishReason> finish(rows_.size(), ServingFinishReason::None);
        std::vector<std::string> finish_error(rows_.size());
        for (std::size_t r = 0; r < rows_.size(); ++r)
            if (rows_[r].cancelled)
                finish[r] = ServingFinishReason::Cancelled;

        // 3. Decide the next token from retained logits when allowed: a forced
        //    token wins; otherwise sample when unpaced or granted.
        auto may_decide = [](const Row &row) {
            return !row.forced.empty() || !row.request.paced || row.grant > 0;
        };
        auto decide = [&](std::size_t r) {
            Row &row = rows_[r];
            if (row.request.record_logit_digests)
            {
                // Byte-exact identity of the decision's inputs: any change of
                // arithmetic by batch composition changes the digest.
                std::uint64_t digest = 1469598103934665603ull;
                const auto *bytes = reinterpret_cast<const unsigned char *>(decisionLogits(row.slot));
                for (std::size_t i = 0; i < static_cast<std::size_t>(vocab_) * sizeof(float); ++i)
                    digest = (digest ^ bytes[i]) * 1099511628211ull;
                event_of(row).logit_digests.push_back(digest);
            }
            int32_t token = 0;
            bool sampled = false;
            if (!row.forced.empty())
            {
                token = row.forced.front();
                row.forced.pop_front();
            }
            else
            {
                token = static_cast<int32_t>(sampleServingDecision(row.sampler, decisionLogits(row.slot),
                                                                   static_cast<std::size_t>(vocab_),
                                                                   row.request.sampling, row.cached_tokens));
                sampled = true;
                if (row.request.paced)
                    --row.grant;
            }
            row.has_logits = false;
            if (sampled && std::find(row.request.stop_tokens.begin(), row.request.stop_tokens.end(), token) !=
                               row.request.stop_tokens.end())
            {
                // Like the scalar decodeStep(), the terminal stop token is
                // returned (and counted) so consumers can tell a model EOS
                // from a length boundary; it is never fed back.
                ++row.completion_tokens;
                auto &event = event_of(row);
                event.tokens.push_back(token);
                event.completion_tokens = row.completion_tokens;
                finish[r] = ServingFinishReason::Stop;
                return;
            }
            row.sampler.record_token(token);
            row.last_token = token;
            row.needs_feed = true;
            ++row.completion_tokens;
            auto &event = event_of(row);
            event.tokens.push_back(token);
            event.completion_tokens = row.completion_tokens;
            if (row.completion_tokens >= row.request.max_new_tokens || row.cached_tokens + 1 >= context_)
                finish[r] = ServingFinishReason::Length;
        };
        for (std::size_t r = 0; r < rows_.size(); ++r)
            if (finish[r] == ServingFinishReason::None && rows_[r].has_logits && may_decide(rows_[r]))
                decide(r);

        // 4. Shared forwards: one decode forward over every request feeding its
        //    committed token, then one prefill transaction carrying ONE
        //    request's next prompt chunk (the oldest prefilling request, in
        //    admission order). Decode rows are never padded to a chunk's width,
        //    and prompt chunks of different requests are never aggregated
        //    (design: one chunk per prefill transaction).
        struct Work
        {
            std::size_t r;
            std::vector<int> tokens;
        };
        std::vector<Work> decode_work;
        std::vector<Work> prefill_work;
        for (std::size_t r = 0; r < rows_.size(); ++r)
        {
            if (finish[r] != ServingFinishReason::None)
                continue;
            const Row &row = rows_[r];
            if (row.prompt_consumed < row.request.prompt.size())
            {
                if (!prefill_work.empty())
                    continue; // a later request's chunk waits for a following step
                const std::size_t chunk =
                    std::min<std::size_t>(kPrefillChunk, row.request.prompt.size() - row.prompt_consumed);
                prefill_work.push_back(
                    {r, std::vector<int>(
                            row.request.prompt.begin() + static_cast<std::ptrdiff_t>(row.prompt_consumed),
                            row.request.prompt.begin() + static_cast<std::ptrdiff_t>(row.prompt_consumed + chunk))});
            }
            else if (row.needs_feed && may_decide(row))
            {
                decode_work.push_back({r, {row.last_token}});
            }
            // else blocked: waiting for a grant or a forced token
        }

        auto run_forward = [&](const std::vector<Work> &work, const char *kind) -> bool {
            if (work.empty())
                return true;
            std::vector<int> slots;
            std::vector<std::vector<int>> rows;
            slots.reserve(work.size());
            rows.reserve(work.size());
            for (const auto &item : work)
            {
                slots.push_back(rows_[item.r].slot);
                rows.push_back(item.tokens);
            }
            if (!executor_.forwardServingRows(slots, rows))
                return false;
            // PerfStats proof the serving-row path ran: one count per shared
            // forward and its live rows (decode rows are width one).
            PerfStatsCollector::addCounter("serving", kind, 1.0, "serving_rows", "cpu");
            std::size_t tokens = 0;
            for (const auto &row : rows)
                tokens += row.size();
            PerfStatsCollector::addCounter("serving", "forward_sequences", static_cast<double>(rows.size()),
                                           "serving_rows", "cpu");
            PerfStatsCollector::addCounter("serving", "forward_tokens", static_cast<double>(tokens),
                                           "serving_rows", "cpu");
            if (recorder_)
            {
                std::vector<const float *> row_logits;
                row_logits.reserve(work.size());
                for (std::size_t e = 0; e < work.size(); ++e)
                    row_logits.push_back(executor_.servingRowLogits(static_cast<int>(e)));
                recorder_->forwarded(slots, rows, row_logits, vocab_);
            }
            for (std::size_t e = 0; e < work.size(); ++e)
            {
                const std::size_t r = work[e].r;
                Row &row = rows_[r];
                row.cached_tokens += static_cast<int>(rows[e].size());
                if (row.prompt_consumed < row.request.prompt.size())
                {
                    row.prompt_consumed += rows[e].size();
                    if (row.prompt_consumed < row.request.prompt.size())
                        continue; // still streaming its prompt
                    // The whole prompt is in this slot's KV: archive it.
                    if (!executor_.harvestServingRowPrefix(row.slot, row.prefix_admission, row.request.prompt))
                    {
                        finish[r] = ServingFinishReason::Error;
                        finish_error[r] = "Serving prefix harvest failed";
                        continue;
                    }
                }
                else
                {
                    row.needs_feed = false;
                }
                const float *logits = executor_.servingRowLogits(static_cast<int>(e));
                if (!logits)
                {
                    finish[r] = ServingFinishReason::Error;
                    finish_error[r] = "Serving row produced no logits";
                    continue;
                }
                std::copy(logits, logits + vocab_, decisionLogits(row.slot));
                row.has_logits = true;
                if (may_decide(row))
                    decide(r);
            }
            return true;
        };

        if (!run_forward(decode_work, "decode_forwards") || !run_forward(prefill_work, "prefill_forwards"))
        {
            // A failed shared forward leaves every resident slot's state
            // unknown: fail all resident and queued requests and seal.
            const auto detail = executor_.servingRowError();
            return sealAll(detail.empty() ? std::string("Serving forward failed") : detail, events);
        }

        // 5. Retirement frees each finished request's slot. Every finished
        //    request keeps its real terminal event; a slot that cannot be
        //    emptied may still hold state, so afterwards the session seals
        //    (failing only the still-live requests).
        std::string reset_failure;
        for (std::size_t r = rows_.size(); r-- > 0;)
        {
            if (finish[r] == ServingFinishReason::None)
                continue;
            Row &row = rows_[r];
            auto &event = event_of(row);
            event.finish = finish[r];
            if (finish[r] == ServingFinishReason::Error)
                event.error = finish_error[r];
            if (!executor_.resetServingRow(row.slot) && reset_failure.empty())
                reset_failure = "Serving slot " + std::to_string(row.slot) + " could not be reset";
            slot_busy_[static_cast<std::size_t>(row.slot)] = false;
            rows_.erase(rows_.begin() + static_cast<std::ptrdiff_t>(r));
            PerfStatsCollector::addCounter("serving", "row_retirements", 1.0, "serving_rows", "cpu");
        }
        if (!reset_failure.empty())
        {
            for (auto &[id, event] : step_events)
                events.push_back(std::move(event));
            step_events.clear();
            return sealAll(reset_failure, events);
        }
        return publish();
    }
} // namespace llaminar2
