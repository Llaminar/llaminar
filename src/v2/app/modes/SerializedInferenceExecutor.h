/**
 * @file SerializedInferenceExecutor.h
 * @brief One stable inference worker with bounded HTTP admission and independent readers.
 *
 * Only admitted inference work enters the stable worker, preserving model/KV
 * ownership and OpenMP locality. HTTP has more workers than admitted inference
 * responses, so queued generations cannot occupy every control-plane worker.
 * Reservations also cover SSE responses before their content provider starts.
 * Shutdown drains submitted work and rejects new admission; it never cancels
 * a GPU transaction or substitutes an inference deadline.
 */
#pragma once

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>

#include "execution/config/ServingCapacity.h" // kServingResponsePoolCeiling

namespace llaminar2
{
    /** @brief Bound network waiters without allowing concurrent model execution. */
    class SerializedInferenceExecutor
    {
        /** @brief Host request ownership only; no model or device state is stored. */
        struct State
        {
            /** @brief Fix admission bounds before exposing the worker. */
            explicit State(std::size_t limit, std::size_t queue_limit)
                : capacity(limit), queue_capacity(queue_limit) {}
            const std::size_t capacity;
            /** @brief Maximum accepted responses that are not currently executing. */
            const std::size_t queue_capacity;
            std::mutex mutex;
            std::condition_variable ready;
            std::deque<std::function<void()>> jobs;
            std::thread::id worker_id;
            std::size_t reserved = 0;
            bool active = false;
            bool stopping = false;
        };

    public:
        /** @brief Maximum accepted generation responses, including queued clients. */
        static constexpr std::size_t kDefaultCapacity = kDefaultServingResponsePool;
        /**
         * @brief Default maximum non-executing accepted responses (queue/preparing).
         *
         * The qualified capacity-one envelope resolves to Q = R - S: with one
         * execution-attached response, at most R - S others may be preparing or
         * queued, preserving the existing "accepted including active and queued
         * <= R" admission semantics.
         */
        static constexpr std::size_t kDefaultQueueCapacity = kDefaultCapacity - 1;

        /** @brief Coherent request-executor counters, independent of GPU execution state. */
        struct Snapshot
        {
            std::size_t capacity;       ///< Response-lease total bound R.
            std::size_t queue_capacity; ///< Queue/preparing bound Q.
            std::size_t admitted;       ///< Accepted responses (attached + queued + preparing).
            std::size_t queued;         ///< Submitted, not yet started.
            bool active;                ///< The single worker is executing a response.
        };

        /** @brief Own one admitted HTTP response and at most one inference submission. */
        class Reservation
        {
            /** @brief Only the executor can construct a charged response lifetime. */
            struct ConstructionKey {};

        public:
            /** @brief Construct under admission ownership after shared storage is allocated. */
            Reservation(ConstructionKey, std::shared_ptr<State> state) noexcept
                : state_(std::move(state)) {}
            /** @brief Release capacity even if an SSE client leaves before publication. */
            ~Reservation()
            {
                std::lock_guard lock(state_->mutex);
                --state_->reserved;
            }
            Reservation(const Reservation &) = delete;
            Reservation &operator=(const Reservation &) = delete;

            /**
             * @brief Execute once on the stable worker and deliver its result/exception.
             * @param work Synchronous inference transaction; captured references outlive this call.
             * @return The transaction's exact result, with no retry or alternate path.
             */
            template <class Work>
            auto run(Work &&work) -> std::invoke_result_t<Work>
            {
                using Result = std::invoke_result_t<Work>;
                auto task = std::make_shared<std::packaged_task<Result()>>(std::forward<Work>(work));
                auto result = task->get_future();
                {
                    std::lock_guard lock(state_->mutex);
                    if (submitted_ || state_->stopping || state_->worker_id == std::this_thread::get_id())
                        throw std::logic_error("HTTP inference reservation is retired, submitted or reentrant");
                    state_->jobs.emplace_back([task] { (*task)(); });
                    submitted_ = true;
                }
                state_->ready.notify_one();
                return result.get();
            }

        private:
            friend class SerializedInferenceExecutor;
            std::shared_ptr<State> state_;
            bool submitted_ = false; ///< Protected by the admission authority's mutex.
        };

        /**
         * @brief Start the sole inference thread with explicit bounded admission.
         * @param capacity Number of accepted generation responses (R), including active and queued.
         * @param queue_capacity Maximum accepted responses that are not currently
         *        executing (Q); zero allows only the response occupying the worker.
         */
        explicit SerializedInferenceExecutor(std::size_t capacity = kDefaultCapacity,
                                             std::size_t queue_capacity = kDefaultQueueCapacity)
            : state_(std::make_shared<State>(checkedCapacity(capacity), checkedQueueCapacity(queue_capacity))),
              worker_([state = state_] { serve(state); }) {}
        /** @brief Drain accepted submissions before model ownership can be released. */
        ~SerializedInferenceExecutor() { shutdown(); }
        SerializedInferenceExecutor(const SerializedInferenceExecutor &) = delete;
        SerializedInferenceExecutor &operator=(const SerializedInferenceExecutor &) = delete;

        /**
         * @brief Reserve one network waiter without blocking an HTTP worker.
         *
         * Admission enforces both typed axes: the response-lease total (R) and
         * the queue/preparing bound (Q). The execution slot occupant (S = 1) is
         * never counted against Q, so an authored Q = 0 still admits the single
         * response occupying the worker while refusing additional accepted
         * responses. With the qualified automatic envelope Q = R - 1 the extra
         * bound coincides exactly with the total bound, preserving the existing
         * "accepted including active and queued <= R" behavior.
         * @return Its shared response lifetime, or null for an explicit overload/shutdown response.
         */
        [[nodiscard]] std::shared_ptr<Reservation> tryReserve()
        {
            std::lock_guard lock(state_->mutex);
            if (state_->stopping || state_->reserved >= state_->capacity ||
                state_->reserved >= 1 + state_->queue_capacity)
                return {};
            auto result = std::make_shared<Reservation>(Reservation::ConstructionKey{}, state_);
            ++state_->reserved;
            return result;
        }

        /** @brief Return the HTTP pool width that leaves capacity for control requests. */
        [[nodiscard]] std::size_t httpWorkerCount() const noexcept { return state_->capacity + 2; }

        /** @brief Read host queue ownership without waiting for a generation. */
        [[nodiscard]] Snapshot snapshot() const
        {
            std::lock_guard lock(state_->mutex);
            return {state_->capacity, state_->queue_capacity, state_->reserved, state_->jobs.size(),
                    state_->active};
        }

        /** @brief Reject new submissions and join all previously submitted work. */
        void shutdown()
        {
            {
                std::lock_guard lock(state_->mutex);
                state_->stopping = true;
            }
            state_->ready.notify_one();
            if (worker_.joinable())
                worker_.join();
        }

    private:
        /** @brief Reject invalid capacity before constructing any worker or pool. */
        static std::size_t checkedCapacity(std::size_t capacity)
        {
            if (capacity == 0 || capacity > static_cast<std::size_t>(kServingResponsePoolCeiling))
                throw std::invalid_argument(
                    "HTTP inference capacity must be between 1 and the installed ceiling (1024)");
            return capacity;
        }

        /** @brief Reject an invalid queue bound before constructing the worker. */
        static std::size_t checkedQueueCapacity(std::size_t queue_capacity)
        {
            if (queue_capacity > static_cast<std::size_t>(kServingResponsePoolCeiling))
                throw std::invalid_argument(
                    "HTTP inference queue capacity must be at most the installed ceiling (1024)");
            return queue_capacity;
        }

        /** @brief Execute FIFO jobs on one thread, retaining ownership through exceptions. */
        static void serve(const std::shared_ptr<State> &state)
        {
            std::unique_lock lock(state->mutex);
            state->worker_id = std::this_thread::get_id();
            for (;;)
            {
                state->ready.wait(lock, [&] { return state->stopping || !state->jobs.empty(); });
                if (state->jobs.empty())
                    return;
                auto work = std::move(state->jobs.front());
                state->jobs.pop_front();
                state->active = true;
                lock.unlock();
                // packaged_task retains every exception for its original HTTP
                // waiter; an inference exception cannot kill or replace this worker.
                work();
                lock.lock();
                state->active = false;
            }
        }

        std::shared_ptr<State> state_;
        std::thread worker_;
    };
}
