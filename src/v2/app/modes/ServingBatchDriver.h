/**
 * @file ServingBatchDriver.h
 * @brief Drives the runner-owned continuous-batching session for HTTP serving.
 *
 * The runner is the sole lifecycle authority; this driver is only its thread
 * and transport adapter. One thread exclusively owns the runner session:
 * it applies lane commands (submit/grant/force/cancel), runs shared execution
 * steps while any request can progress, and routes each request's committed
 * tokens to that request's mailbox. Each HTTP request runs its unchanged chat
 * handler on its own thread through a BatchedGenerationLane, which pulls
 * tokens with one-window grants so slow consumers never stall other requests.
 */
#pragma once

#include "app/modes/RequestGenerationLane.h"
#include "execution/runner/IOrchestrationRunner.h"
#include "utils/Logger.h"

#include <atomic>
#include <condition_variable>
#include <deque>
#include <algorithm>
#include <functional>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace llaminar2
{
    /** @brief Token stream of one request, filled by the driver thread. */
    struct ServingMailbox
    {
        std::mutex mutex;
        std::condition_variable cv;
        std::deque<int32_t> tokens;
        ServingFinishReason finish = ServingFinishReason::None;
        std::string error;
        bool admitted = false;          ///< Submission acknowledged.
        int prompt_tokens = 0;          ///< From the latest event.
        int cached_prompt_tokens = 0;   ///< Prefix-cache restored prompt tokens.
        bool prefix_cache_enabled = false;
        std::string admission_error;    ///< Non-empty when admission failed.
    };

    class ServingBatchDriver
    {
    public:
        /**
         * @param runner Initialized runner; owned exclusively by the driver
         *        thread while the driver lives.
         * @param capacity Resident request slots of the session.
         * @param context_tokens Admitted context; bounds each request's output.
         */
        ServingBatchDriver(IOrchestrationRunner &runner, int capacity, int context_tokens)
            : runner_(runner), capacity_(capacity), context_tokens_(context_tokens)
        {
        }

        ~ServingBatchDriver() { stop(); }

        ServingBatchDriver(const ServingBatchDriver &) = delete;
        ServingBatchDriver &operator=(const ServingBatchDriver &) = delete;

        /** @brief Open the session on the driver thread; false with @p error. */
        bool start(std::string &error)
        {
            std::promise<std::string> opened;
            auto result = opened.get_future();
            thread_ = std::thread([this, &opened] {
                if (!runner_.beginServingBatch(capacity_))
                {
                    opened.set_value(runner_.lastError().empty() ? "serving batch rejected"
                                                                 : runner_.lastError());
                    return;
                }
                opened.set_value({});
                loop();
                runner_.endServingBatch();
            });
            error = result.get();
            if (!error.empty())
            {
                thread_.join();
                return false;
            }
            return true;
        }

        void stop()
        {
            {
                std::lock_guard lock(mutex_);
                stopping_ = true;
            }
            cv_.notify_all();
            if (thread_.joinable())
                thread_.join();
            // Unblock any lane still waiting for output.
            std::lock_guard lock(mutex_);
            for (auto &[id, mailbox] : mailboxes_)
            {
                std::lock_guard mailbox_lock(mailbox->mutex);
                if (mailbox->finish == ServingFinishReason::None)
                {
                    mailbox->finish = ServingFinishReason::Error;
                    mailbox->error = "server shutting down";
                }
                mailbox->admitted = true;
                mailbox->cv.notify_all();
            }
        }

        /** @brief Lane factory for the chat handler. */
        std::unique_ptr<IRequestGenerationLane> makeLane();

        // ----- Lane commands (any thread) -----

        std::shared_ptr<ServingMailbox> submit(ServingBatchRequest request)
        {
            auto mailbox = std::make_shared<ServingMailbox>();
            post([this, mailbox, request = std::move(request)]() mutable {
                const std::uint64_t id = request.id;
                const bool ok = runner_.submitServingRequest(std::move(request));
                std::lock_guard lock(mailbox->mutex);
                mailbox->admitted = true;
                if (ok)
                    mailboxes_[id] = mailbox;
                else
                    mailbox->admission_error = runner_.lastError();
                mailbox->cv.notify_all();
            });
            return mailbox;
        }

        void grant(std::uint64_t id, int tokens)
        {
            post([this, id, tokens] { runner_.grantServingTokens(id, tokens); });
        }

        void force(std::uint64_t id, int32_t token)
        {
            post([this, id, token] { runner_.forceServingToken(id, token); });
        }

        void cancel(std::uint64_t id)
        {
            post([this, id] { runner_.cancelServingRequest(id); });
        }

        std::uint64_t nextRequestId() { return next_id_.fetch_add(1) + 1; }

    private:
        void post(std::function<void()> command)
        {
            {
                std::lock_guard lock(mutex_);
                commands_.push_back(std::move(command));
            }
            cv_.notify_all();
        }

        /** @brief Driver thread: apply commands, step while anything can progress. */
        void loop()
        {
            for (;;)
            {
                std::deque<std::function<void()>> commands;
                {
                    std::unique_lock lock(mutex_);
                    cv_.wait(lock, [&] { return stopping_ || !commands_.empty() || runnable_; });
                    if (stopping_)
                        return;
                    commands.swap(commands_);
                }
                for (auto &command : commands)
                    command();
                if (runner_.servingBatchHasRunnableWork())
                {
                    for (const auto &event : runner_.stepServingBatch())
                        dispatch(event);
                }
                runnable_ = runner_.servingBatchHasRunnableWork();
            }
        }

        void dispatch(const ServingBatchEvent &event)
        {
            std::shared_ptr<ServingMailbox> mailbox;
            {
                // mailboxes_ is touched only on this thread after admission.
                auto it = mailboxes_.find(event.id);
                if (it == mailboxes_.end())
                    return;
                mailbox = it->second;
                if (event.finished())
                    mailboxes_.erase(it);
            }
            std::lock_guard lock(mailbox->mutex);
            mailbox->tokens.insert(mailbox->tokens.end(), event.tokens.begin(), event.tokens.end());
            mailbox->prompt_tokens = event.prompt_tokens;
            mailbox->cached_prompt_tokens = event.cached_prompt_tokens;
            mailbox->prefix_cache_enabled = event.prefix_cache_enabled;
            if (event.finished())
            {
                mailbox->finish = event.finish;
                mailbox->error = event.error;
            }
            mailbox->cv.notify_all();
        }

        IOrchestrationRunner &runner_;
        const int capacity_;
        const int context_tokens_;
        std::thread thread_;
        std::mutex mutex_;
        std::condition_variable cv_;
        std::deque<std::function<void()>> commands_;
        bool stopping_ = false;
        bool runnable_ = false; ///< Driver-thread only.
        std::map<std::uint64_t, std::shared_ptr<ServingMailbox>> mailboxes_;
        std::atomic<std::uint64_t> next_id_{0};
    };

    /**
     * @brief Lane of one HTTP request into the shared serving session.
     *
     * Tokens are pulled with one-window grants: decodeStep(n) grants at most n
     * tokens and returns as soon as committed tokens arrive, so the request
     * executes only as fast as its consumer reads.
     */
    class BatchedGenerationLane final : public IRequestGenerationLane
    {
    public:
        BatchedGenerationLane(ServingBatchDriver &driver, int context_tokens)
            : driver_(driver), context_tokens_(context_tokens)
        {
        }

        ~BatchedGenerationLane() override
        {
            if (mailbox_)
            {
                std::lock_guard lock(mailbox_->mutex);
                if (mailbox_->finish != ServingFinishReason::None)
                    return;
            }
            if (id_ != 0)
                driver_.cancel(id_);
        }

        BatchedGenerationLane(const BatchedGenerationLane &) = delete;
        BatchedGenerationLane &operator=(const BatchedGenerationLane &) = delete;

        void beginRequest() override {}
        void setSamplingParams(const SamplingParams &params) override { sampling_ = params; }
        void setStopTokens(const std::vector<int32_t> &tokens) override { stop_tokens_ = tokens; }

        bool prefill(const std::vector<int32_t> &tokens) override
        {
            ServingBatchRequest request;
            request.id = driver_.nextRequestId();
            request.prompt = tokens;
            request.sampling = sampling_;
            request.stop_tokens = stop_tokens_;
            request.max_new_tokens = std::max(1, context_tokens_ - static_cast<int>(tokens.size()));
            request.paced = true;
            id_ = request.id;
            mailbox_ = driver_.submit(std::move(request));
            std::unique_lock lock(mailbox_->mutex);
            mailbox_->cv.wait(lock, [&] { return mailbox_->admitted; });
            if (!mailbox_->admission_error.empty())
            {
                error_ = mailbox_->admission_error;
                return false;
            }
            return true;
        }

        GenerationResult decodeStep(int max_tokens) override
        {
            GenerationResult result;
            if (!mailbox_)
            {
                result.error = "decodeStep before prefill";
                return result;
            }
            max_tokens = std::max(1, max_tokens);
            std::unique_lock lock(mailbox_->mutex);
            if (mailbox_->tokens.empty() && outstanding_ == 0 &&
                mailbox_->finish == ServingFinishReason::None)
            {
                outstanding_ = max_tokens;
                lock.unlock();
                driver_.grant(id_, max_tokens);
                lock.lock();
            }
            mailbox_->cv.wait(lock, [&] {
                return !mailbox_->tokens.empty() || mailbox_->finish != ServingFinishReason::None;
            });
            return drain(lock, max_tokens, result);
        }

        GenerationResult forceDecodeToken(int32_t token) override
        {
            GenerationResult result;
            if (!mailbox_)
            {
                result.error = "forceDecodeToken before prefill";
                return result;
            }
            std::unique_lock lock(mailbox_->mutex);
            lock.unlock();
            driver_.force(id_, token);
            lock.lock();
            mailbox_->cv.wait(lock, [&] {
                return !mailbox_->tokens.empty() || mailbox_->finish != ServingFinishReason::None;
            });
            return drain(lock, 1, result);
        }

        std::string lastError() const override { return error_; }

        /** @brief Prefix reuse of this request (serving rows recompute >= 1 token). */
        RequestRuntimeSummary requestRuntimeSummary() const override
        {
            RequestRuntimeSummary summary;
            if (!mailbox_)
                return summary;
            std::lock_guard lock(mailbox_->mutex);
            auto &prefix = summary.prefix_request;
            prefix.enabled = mailbox_->prefix_cache_enabled;
            prefix.requested_tokens = mailbox_->prompt_tokens;
            prefix.matched_tokens = mailbox_->cached_prompt_tokens;
            prefix.partial_hit = mailbox_->cached_prompt_tokens > 0;
            prefix.storage_tier = mailbox_->cached_prompt_tokens > 0 ? "ram" : "none";
            return summary;
        }

    private:
        GenerationResult &drain(std::unique_lock<std::mutex> &, int max_tokens, GenerationResult &result)
        {
            while (!mailbox_->tokens.empty() && static_cast<int>(result.tokens.size()) < max_tokens)
            {
                result.tokens.push_back(mailbox_->tokens.front());
                mailbox_->tokens.pop_front();
            }
            outstanding_ = std::max(0, outstanding_ - static_cast<int>(result.tokens.size()));
            if (mailbox_->finish == ServingFinishReason::Error && result.tokens.empty())
                result.error = mailbox_->error.empty() ? "serving request failed" : mailbox_->error;
            if (mailbox_->finish == ServingFinishReason::Cancelled && result.tokens.empty())
                result.error = "serving request cancelled";
            // Stop and Length end like the scalar path: the final returned
            // token (stop token, or the last token before the boundary) is
            // terminal.
            if (mailbox_->tokens.empty() && (mailbox_->finish == ServingFinishReason::Stop ||
                                             mailbox_->finish == ServingFinishReason::Length))
                result.is_complete = true;
            return result;
        }

        ServingBatchDriver &driver_;
        const int context_tokens_;
        SamplingParams sampling_;
        std::vector<int32_t> stop_tokens_;
        std::uint64_t id_ = 0;
        std::shared_ptr<ServingMailbox> mailbox_;
        int outstanding_ = 0;
        std::string error_;
    };

    inline std::unique_ptr<IRequestGenerationLane> ServingBatchDriver::makeLane()
    {
        // The session's runner context (config) is immutable after start.
        return std::make_unique<BatchedGenerationLane>(*this, context_tokens_);
    }
} // namespace llaminar2
