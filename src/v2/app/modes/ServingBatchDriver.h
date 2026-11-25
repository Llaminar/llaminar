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
 *
 * Exactly-once outcome: every mailbox handed out by submit() reaches exactly
 * one terminal state — admission failure, a runner finish event, or a driver
 * failure (shutdown, sealed session, exception) — so no lane can wait forever.
 */
#pragma once

#include "app/modes/RequestGenerationLane.h"
#include "app/modes/ServingResponseLeasePool.h"
#include "execution/runner/IOrchestrationRunner.h"
#include "utils/Logger.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
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
        bool admitted = false;          ///< Submission resolved (accepted or failed).
        std::string admission_error;    ///< Non-empty when admission failed.
        int prompt_tokens = 0;          ///< From the latest event.
        int cached_prompt_tokens = 0;   ///< Prefix-cache restored prompt tokens.
        bool prefix_cache_enabled = false;

        /** @brief Resolve as failed unless already terminal; wakes the lane. */
        void fail(const std::string &reason)
        {
            std::lock_guard lock(mutex);
            if (!admitted)
            {
                admitted = true;
                admission_error = reason;
            }
            if (finish == ServingFinishReason::None)
            {
                finish = ServingFinishReason::Error;
                error = reason;
            }
            cv.notify_all();
        }
    };

    /**
     * @brief Cancellation handle of one HTTP request.
     *
     * The transport owns it; the request's lane binds its serving id when it
     * submits. Cancelling before submission makes the lane refuse to submit;
     * cancelling after posts a runner cancellation immediately, whether the
     * request is queued, resident or blocked on its consumer.
     */
    class ServingRequestCancellation
    {
    public:
        void cancel()
        {
            std::function<void()> action;
            {
                std::lock_guard lock(mutex_);
                if (cancelled_)
                    return;
                cancelled_ = true;
                action = std::move(on_cancel_);
            }
            if (action)
                action();
        }

        /** @brief Bind the cancel action; false (unbound) when already cancelled. */
        bool bind(std::function<void()> on_cancel)
        {
            std::lock_guard lock(mutex_);
            if (cancelled_)
                return false;
            on_cancel_ = std::move(on_cancel);
            return true;
        }

        bool cancelled() const
        {
            std::lock_guard lock(mutex_);
            return cancelled_;
        }

    private:
        mutable std::mutex mutex_;
        bool cancelled_ = false;
        std::function<void()> on_cancel_;
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
                std::lock_guard lock(mutex_);
                closed_ = true;
                close_reason_ = "serving session failed to open: " + error;
                return false;
            }
            return true;
        }

        /**
         * @brief Close the session: join the driver thread, then fail every
         *        unresolved mailbox. Idempotent; later commands fail at once.
         */
        void stop()
        {
            {
                std::lock_guard lock(mutex_);
                closed_ = true;
                if (close_reason_.empty())
                    close_reason_ = "server shutting down";
            }
            cv_.notify_all();
            if (thread_.joinable())
                thread_.join();
            failAll(closeReason());
        }

        /** @brief Lane factory for the chat handler (binds the calling thread's cancellation). */
        std::unique_ptr<IRequestGenerationLane> makeLane();

        /**
         * @brief Binds a request's cancellation to lanes created on this thread.
         *
         * The HTTP producer thread installs its request's handle while running
         * the handler, so makeLane() needs no extra handler plumbing.
         */
        /** @brief Transport-owned handles of the request running on this thread. */
        struct RequestContext
        {
            std::shared_ptr<ServingRequestCancellation> cancellation;
            /// Response lease whose execution claim ends when the row finishes.
            std::shared_ptr<ServingResponseLeasePool::Lease> lease;
        };

        class RequestScope
        {
        public:
            explicit RequestScope(std::shared_ptr<ServingRequestCancellation> cancellation,
                                  std::shared_ptr<ServingResponseLeasePool::Lease> lease = {})
                : previous_(current())
            {
                current() = RequestContext{std::move(cancellation), std::move(lease)};
            }
            ~RequestScope() { current() = std::move(previous_); }
            RequestScope(const RequestScope &) = delete;
            RequestScope &operator=(const RequestScope &) = delete;

            static RequestContext &current()
            {
                thread_local RequestContext context;
                return context;
            }

        private:
            RequestContext previous_;
        };

        // ----- Lane commands (any thread) -----

        /** @brief Submit; the returned mailbox always reaches a terminal state. */
        std::shared_ptr<ServingMailbox> submit(ServingBatchRequest request)
        {
            auto mailbox = std::make_shared<ServingMailbox>();
            const std::uint64_t id = request.id;
            const bool posted = post([this, mailbox, id, request = std::move(request)]() mutable {
                // The mailbox stays in unadmitted_ until it is resolved, so a
                // throwing submission is still failed by seal()/stop().
                const bool ok = runner_.submitServingRequest(std::move(request));
                if (ok)
                {
                    mailboxes_[id] = mailbox;
                    std::lock_guard lock(mailbox->mutex);
                    mailbox->admitted = true;
                    mailbox->cv.notify_all();
                }
                else
                {
                    mailbox->fail(runner_.lastError().empty() ? "serving submission rejected"
                                                              : runner_.lastError());
                }
                std::lock_guard lock(mutex_);
                unadmitted_.erase(id);
            }, [&] { unadmitted_[id] = mailbox; });
            if (!posted)
                mailbox->fail(closeReason());
            return mailbox;
        }

        void grant(std::uint64_t id, int tokens)
        {
            (void)post([this, id, tokens] { runner_.grantServingTokens(id, tokens); });
        }

        void force(std::uint64_t id, int32_t token)
        {
            (void)post([this, id, token] { runner_.forceServingToken(id, token); });
        }

        void cancel(std::uint64_t id)
        {
            (void)post([this, id] { runner_.cancelServingRequest(id); });
        }

        std::uint64_t nextRequestId() { return next_id_.fetch_add(1) + 1; }

        /**
         * @brief Run @p access on the driver thread (the runner's owner) and wait.
         *
         * For reads/maintenance of mutable runner state from handler threads.
         * An exception from @p access is rethrown to the caller without
         * sealing the session.
         * @throws std::runtime_error when the driver is closed.
         */
        void call(const std::function<void(IOrchestrationRunner &)> &access)
        {
            auto done = std::make_shared<std::promise<void>>();
            auto result = done->get_future();
            if (!post([this, &access, done] {
                    try
                    {
                        access(runner_);
                        done->set_value();
                    }
                    catch (...)
                    {
                        done->set_exception(std::current_exception());
                    }
                }))
                throw std::runtime_error("serving session closed: " + closeReason());
            // The command alone owns the promise: if stop()/seal() drops it
            // before it runs, the future resolves as a broken promise.
            try
            {
                result.get();
            }
            catch (const std::future_error &)
            {
                throw std::runtime_error("serving session closed: " + closeReason());
            }
        }

        /** @brief Occupancy published by the driver thread after its last step. */
        ServingBatchOccupancy occupancy() const { return {resident_.load(), queued_.load()}; }

        int capacity() const { return capacity_; }

        /** @brief Why serving failed (sealed); empty while healthy or after an orderly stop. */
        std::string failure() const
        {
            std::lock_guard lock(mutex_);
            return failure_;
        }

        /** @brief True once stopped or sealed by a failure. */
        bool closed() const
        {
            std::lock_guard lock(mutex_);
            return closed_;
        }

    private:
        /**
         * @brief Queue a command; false when the driver is closed.
         * @param while_locked Runs under the queue lock on success, so
         *        bookkeeping is atomic with acceptance.
         */
        bool post(std::function<void()> command, const std::function<void()> &while_locked = {})
        {
            {
                std::lock_guard lock(mutex_);
                if (closed_)
                    return false;
                commands_.push_back(std::move(command));
                if (while_locked)
                    while_locked();
            }
            cv_.notify_all();
            return true;
        }

        std::string closeReason() const
        {
            std::lock_guard lock(mutex_);
            return close_reason_;
        }

        /** @brief Seal after a failure: nothing more is accepted or stepped. */
        void seal(const std::string &reason)
        {
            {
                std::lock_guard lock(mutex_);
                if (!closed_)
                {
                    close_reason_ = reason;
                    failure_ = reason;
                }
                closed_ = true;
            }
            LOG_ERROR("[ServingBatchDriver] session sealed: " << reason);
            failAll(reason);
        }

        /** @brief Fail every unresolved mailbox (driver thread joined or sealing on it). */
        void failAll(const std::string &reason)
        {
            std::map<std::uint64_t, std::shared_ptr<ServingMailbox>> pending;
            {
                std::lock_guard lock(mutex_);
                pending.swap(unadmitted_);
                commands_.clear();
            }
            for (auto &[id, mailbox] : pending)
                mailbox->fail(reason);
            for (auto &[id, mailbox] : mailboxes_)
                mailbox->fail(reason);
            mailboxes_.clear();
        }

        /** @brief Driver thread: apply commands, step while anything can progress. */
        void loop()
        {
            for (;;)
            {
                std::deque<std::function<void()>> commands;
                {
                    std::unique_lock lock(mutex_);
                    cv_.wait(lock, [&] { return closed_ || !commands_.empty() || runnable_; });
                    if (closed_)
                        return;
                    commands.swap(commands_);
                }
                try
                {
                    for (auto &command : commands)
                        command();
                    if (runner_.servingBatchHasRunnableWork())
                        for (const auto &event : runner_.stepServingBatch())
                            dispatch(event);
                    // A sealed runner session (failed shared forward) is
                    // fatal for serving: close the driver too, so new
                    // requests and /health report the failure.
                    if (const auto failure = runner_.servingBatchFailure(); !failure.empty())
                    {
                        seal("serving session failed: " + failure);
                        return;
                    }
                    runnable_ = runner_.servingBatchHasRunnableWork();
                    const auto occupancy = runner_.servingBatchOccupancy();
                    resident_.store(occupancy.resident);
                    queued_.store(occupancy.queued);
                }
                catch (const std::exception &error)
                {
                    seal(std::string("serving step failed: ") + error.what());
                    return;
                }
                catch (...)
                {
                    seal("serving step failed with an unknown exception");
                    return;
                }
            }
        }

        void dispatch(const ServingBatchEvent &event)
        {
            auto it = mailboxes_.find(event.id);
            if (it == mailboxes_.end())
                return;
            std::shared_ptr<ServingMailbox> mailbox = it->second;
            if (event.finished())
                mailboxes_.erase(it);
            std::lock_guard lock(mailbox->mutex);
            mailbox->tokens.insert(mailbox->tokens.end(), event.tokens.begin(), event.tokens.end());
            mailbox->prompt_tokens = event.prompt_tokens;
            mailbox->cached_prompt_tokens = event.cached_prompt_tokens;
            mailbox->prefix_cache_enabled = event.prefix_cache_enabled;
            if (event.finished() && mailbox->finish == ServingFinishReason::None)
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
        mutable std::mutex mutex_;
        std::condition_variable cv_;
        std::deque<std::function<void()>> commands_;
        bool closed_ = false;
        std::string close_reason_;
        std::string failure_;
        /** @brief Submitted, not yet applied by the driver thread (mutex_). */
        std::map<std::uint64_t, std::shared_ptr<ServingMailbox>> unadmitted_;
        bool runnable_ = false; ///< Driver-thread only.
        /** @brief Admitted requests (driver thread; read by stop() after join). */
        std::map<std::uint64_t, std::shared_ptr<ServingMailbox>> mailboxes_;
        std::atomic<std::uint64_t> next_id_{0};
        std::atomic<int> resident_{0};
        std::atomic<int> queued_{0};
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
        BatchedGenerationLane(ServingBatchDriver &driver, int context_tokens,
                              ServingBatchDriver::RequestContext context)
            : driver_(driver), context_tokens_(context_tokens),
              cancellation_(std::move(context.cancellation)), lease_(std::move(context.lease))
        {
        }

        /** @brief Retires an unfinished request (cancellation is idempotent). */
        ~BatchedGenerationLane() override
        {
            // The lane is done executing either way; the response lease itself
            // stays with the transport until publication completes.
            if (lease_)
                lease_->executionFinished();
            if (id_ == 0 || !mailbox_)
                return;
            {
                std::lock_guard lock(mailbox_->mutex);
                if (mailbox_->finish != ServingFinishReason::None)
                    return;
            }
            driver_.cancel(id_);
        }

        BatchedGenerationLane(const BatchedGenerationLane &) = delete;
        BatchedGenerationLane &operator=(const BatchedGenerationLane &) = delete;

        void beginRequest() override {}
        void setSamplingParams(const SamplingParams &params) override { sampling_ = params; }
        void setStopTokens(const std::vector<int32_t> &tokens) override { stop_tokens_ = tokens; }

        bool prefill(const std::vector<int32_t> &tokens) override
        {
            if (mailbox_)
            {
                error_ = "prefill called twice for one serving request";
                return false;
            }
            ServingBatchRequest request;
            request.id = driver_.nextRequestId();
            request.prompt = tokens;
            request.sampling = sampling_;
            request.stop_tokens = stop_tokens_;
            request.max_new_tokens = std::max(1, context_tokens_ - static_cast<int>(tokens.size()));
            request.paced = true;
            if (cancellation_ && cancellation_->cancelled())
            {
                error_ = "request cancelled by the client";
                return false;
            }
            id_ = request.id;
            mailbox_ = driver_.submit(std::move(request));
            // A transport close from now on cancels the serving request at
            // once, whether it is queued, resident or waiting on its consumer.
            if (cancellation_ && !cancellation_->bind([driver = &driver_, id = id_] { driver->cancel(id); }))
                driver_.cancel(id_);
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
            return drain(max_tokens, result);
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
            // A forced token must be the very next output. With a live grant
            // or undelivered sampled tokens it would not be: fail fast.
            if (outstanding_ > 0 || !mailbox_->tokens.empty())
            {
                result.error = "forceDecodeToken with an outstanding sampled-token grant";
                return result;
            }
            if (mailbox_->finish != ServingFinishReason::None)
                return drain(1, result);
            lock.unlock();
            driver_.force(id_, token);
            lock.lock();
            mailbox_->cv.wait(lock, [&] {
                return !mailbox_->tokens.empty() || mailbox_->finish != ServingFinishReason::None;
            });
            return drain(1, result);
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
        /** @brief Move up to @p max_tokens delivered tokens out (mailbox lock held). */
        GenerationResult &drain(int max_tokens, GenerationResult &result)
        {
            while (!mailbox_->tokens.empty() && static_cast<int>(result.tokens.size()) < max_tokens)
            {
                result.tokens.push_back(mailbox_->tokens.front());
                mailbox_->tokens.pop_front();
            }
            outstanding_ = mailbox_->finish != ServingFinishReason::None
                               ? 0
                               : std::max(0, outstanding_ - static_cast<int>(result.tokens.size()));
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
            // The serving row retired: its execution claim is free for the
            // next request while this response may still be publishing.
            if (mailbox_->finish != ServingFinishReason::None && lease_)
                lease_->executionFinished();
            return result;
        }

        ServingBatchDriver &driver_;
        const int context_tokens_;
        std::shared_ptr<ServingRequestCancellation> cancellation_;
        std::shared_ptr<ServingResponseLeasePool::Lease> lease_;
        SamplingParams sampling_;
        std::vector<int32_t> stop_tokens_;
        std::uint64_t id_ = 0;
        std::shared_ptr<ServingMailbox> mailbox_;
        int outstanding_ = 0;
        std::string error_;
    };

    inline std::unique_ptr<IRequestGenerationLane> ServingBatchDriver::makeLane()
    {
        return std::make_unique<BatchedGenerationLane>(*this, context_tokens_, RequestScope::current());
    }
} // namespace llaminar2
