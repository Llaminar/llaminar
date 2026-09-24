/**
 * @file TPWorkerPool.h
 * @brief Persistent, generation-safe worker fan-out for tensor-parallel ranks.
 *
 * RankOrchestrator performs the same logical operation on every LocalTP
 * participant. Creating one `std::async` thread per participant and per decode
 * step is expensive, so this pool keeps one worker alive for each participant.
 * Workers retain their runtime thread state and are awakened by `dispatch()`.
 * The caller then uses `collectAll()` as the rank-local completion fence.
 *
 * A dispatch generation owns its callable, result slots, completion count, and
 * collector notification. The pool does not allow the next generation to
 * replace those objects until the current generation has been collected. This
 * invariant matters for very short MTP sidecar operations: under scheduler
 * contention, a worker can finish between dispatch and collector admission.
 * Generation-qualified state prevents that completion from being reset or
 * mistaken for a neighboring operation.
 *
 * Workers intentionally do not select a CUDA or HIP device themselves. Device
 * graph runners submit work through streams that already belong to the correct
 * device context; the rank's dispatched callable also installs the profiler
 * device metadata required by that operation.
 */

#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#include "../../../utils/Logger.h"

namespace llaminar2
{

    /**
     * @brief Fan one rank-local operation out to persistent participant workers.
     *
     * The public protocol is deliberately strict and sequential:
     *
     * 1. `dispatch()` publishes exactly one new generation.
     * 2. Every worker executes that generation once.
     * 3. One or more observers may call `collectAll()` for that generation.
     * 4. A later `dispatch()` is admitted only after the generation is complete
     *    and at least one collector has consumed its result snapshot.
     *
     * RankOrchestrator follows this protocol synchronously. Rejecting an
     * overlapping dispatch turns an ownership bug into an immediate diagnostic
     * instead of silently dropping a participant operation and hanging at a
     * later collective.
     */
    class TPWorkerPool
    {
    public:
        struct WorkerResult
        {
            bool success = false;
            bool completed = false;
            std::exception_ptr exception = nullptr;
            size_t worker_index = 0;
        };

        explicit TPWorkerPool(size_t num_workers)
            : num_workers_(num_workers),
              results_(num_workers)
        {
            if (num_workers_ == 0)
            {
                throw std::invalid_argument(
                    "TPWorkerPool requires at least one participant worker");
            }
            workers_.reserve(num_workers);
            for (size_t i = 0; i < num_workers; ++i)
            {
                workers_.emplace_back([this, i]()
                                      { workerLoop(i); });
            }
            failure_callback_worker_ = std::thread(
                [this]()
                { failureCallbackLoop(); });
            // Note: construction logging handled by caller (RankOrchestrator)
        }

        ~TPWorkerPool()
        {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                shutdown_ = true;
            }
            dispatch_cv_.notify_all();
            failure_callback_cv_.notify_all();

            for (auto &t : workers_)
            {
                if (t.joinable())
                    t.join();
            }
            if (failure_callback_worker_.joinable())
                failure_callback_worker_.join();
        }

        // Non-copyable, non-movable
        TPWorkerPool(const TPWorkerPool &) = delete;
        TPWorkerPool &operator=(const TPWorkerPool &) = delete;

        /**
         * @brief Dispatch work to all workers.
         *
         * The callable receives the worker index (0..N-1) and must return bool.
         * This method wakes all workers and returns immediately.
         *
         * @param fn Callable: bool(size_t worker_index)
         */
        void dispatch(std::function<bool(size_t)> fn)
        {
            if (!fn)
            {
                throw std::invalid_argument(
                    "TPWorkerPool dispatch requires a callable");
            }

            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (generation_ != collected_generation_ ||
                    active_collectors_ != 0)
                {
                    throw std::logic_error(
                        "TPWorkerPool cannot replace a generation while results remain uncollected");
                }

                work_fn_ = std::move(fn);
                generation_++;
                for (size_t index = 0; index < results_.size(); ++index)
                {
                    auto &r = results_[index];
                    r.success = false;
                    r.completed = false;
                    r.exception = nullptr;
                    r.worker_index = index;
                }
                completed_count_.store(0, std::memory_order_release);
                first_failure_index_.store(SIZE_MAX, std::memory_order_release);
            }
            dispatch_cv_.notify_all();
        }

        /**
         * @brief Wait for all workers to complete and return results.
         *
         * If all workers complete within timeout, returns results normally.
         * If timeout expires with incomplete workers, returns partial results
         * (check WorkerResult::completed to distinguish).
         *
         * @param timeout_ms Maximum wait time in milliseconds (0 = wait forever)
         * @return Vector of WorkerResult, one per worker.
         */
        std::vector<WorkerResult> collectAll(int timeout_ms = 0)
        {
            std::unique_lock<std::mutex> lock(mutex_);
            const uint64_t target_generation = generation_;
            if (target_generation == 0)
                return results_;
            ++active_collectors_;

            auto pred = [this, target_generation]()
            {
                return shutdown_ ||
                       completed_generation_ >= target_generation;
            };

            if (timeout_ms > 0)
            {
                collect_cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), pred);
            }
            else
            {
                collect_cv_.wait(lock, pred);
            }

            std::vector<WorkerResult> snapshot = results_;
            if (completed_generation_ >= target_generation)
            {
                collected_generation_ =
                    std::max(collected_generation_, target_generation);
            }
            --active_collectors_;
            return snapshot;
        }

        /**
         * @brief Check if any worker has failed (non-blocking).
         * @return Worker index of first failure, or SIZE_MAX if none.
         */
        size_t firstFailureIndex() const
        {
            return first_failure_index_.load(std::memory_order_acquire);
        }

        /**
         * @brief Get count of completed workers (non-blocking).
         */
        size_t completedCount() const
        {
            return completed_count_.load(std::memory_order_acquire);
        }

        size_t numWorkers() const { return num_workers_; }

        /**
         * @brief Set a callback invoked on first worker failure.
         *
         * When a worker completes with an exception or returns false, this
         * callback is scheduled immediately on the pool's persistent control
         * thread. The failing participant publishes its terminal WorkerResult
         * before scheduling the callback. This ordering is mandatory because
         * NCCL/RCCL abort may block while another participant leaves a collective;
         * executing it on the failing participant would prevent that participant
         * from ever contributing to the completion fence.
         *
         * The callback must be thread-safe. It may block inside the collective
         * backend without occupying any participant worker.
         */
        void setFailureCallback(std::function<void()> cb)
        {
            std::lock_guard<std::mutex> lock(mutex_);
            failure_callback_ = std::move(cb);
        }

    private:
        void workerLoop(size_t index)
        {
            uint64_t last_gen = 0;

            while (true)
            {
                std::function<bool(size_t)> fn;
                uint64_t work_generation = 0;
                {
                    std::unique_lock<std::mutex> lock(mutex_);
                    dispatch_cv_.wait(lock, [this, last_gen]()
                                      { return shutdown_ || generation_ > last_gen; });

                    if (shutdown_)
                        return;

                    last_gen = generation_;
                    work_generation = generation_;
                    fn = work_fn_;
                }

                // Execute the work
                WorkerResult result;
                result.worker_index = index;
                bool schedule_failure_callback = false;
                try
                {
                    result.success = fn(index);
                    if (!result.success)
                    {
                        // Record first failure
                        size_t expected = SIZE_MAX;
                        if (first_failure_index_.compare_exchange_strong(
                                expected, index, std::memory_order_acq_rel))
                        {
                            schedule_failure_callback = true;
                        }
                    }
                }
                catch (const std::exception &e)
                {
                    result.success = false;
                    result.exception = std::current_exception();
                    LOG_ERROR("[TPWorkerPool] Worker " << index
                                                       << " threw exception before collective abort: "
                                                       << e.what());
                    // Record first failure
                    size_t expected = SIZE_MAX;
                    if (first_failure_index_.compare_exchange_strong(
                            expected, index, std::memory_order_acq_rel))
                    {
                        schedule_failure_callback = true;
                    }
                }
                catch (...)
                {
                    result.success = false;
                    result.exception = std::current_exception();
                    LOG_ERROR("[TPWorkerPool] Worker " << index
                                                       << " threw non-std exception before collective abort");
                    // Record first failure
                    size_t expected = SIZE_MAX;
                    if (first_failure_index_.compare_exchange_strong(
                            expected, index, std::memory_order_acq_rel))
                    {
                        schedule_failure_callback = true;
                    }
                }
                result.completed = true;

                bool generation_complete = false;
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    if (work_generation != generation_)
                    {
                        LOG_ERROR(
                            "[TPWorkerPool] Worker " << index
                            << " completed generation " << work_generation
                            << " after generation " << generation_
                            << " replaced it");
                        result.success = false;
                    }

                    results_[index] = std::move(result);
                    const size_t completed =
                        completed_count_.fetch_add(
                            1,
                            std::memory_order_acq_rel) +
                        1;
                    if (completed >= num_workers_)
                    {
                        completed_generation_ = work_generation;
                        generation_complete = true;
                    }
                }

                if (generation_complete)
                {
                    /*
                     * More than one diagnostic observer may wait for the same
                     * generation. Waking all observers is required because no
                     * later worker completion exists to rescue a waiter left
                     * asleep after this generation reaches its terminal count.
                     */
                    collect_cv_.notify_all();
                }
                if (schedule_failure_callback)
                {
                    /*
                     * Publish completion first, then wake the independent abort
                     * coordinator. A collective abort is allowed to block, but a
                     * participant completion can never again be held behind it.
                     */
                    {
                        std::lock_guard<std::mutex> lock(mutex_);
                        failure_callback_generation_ = std::max(
                            failure_callback_generation_,
                            work_generation);
                    }
                    failure_callback_cv_.notify_one();
                }
            }
        }

        /**
         * @brief Execute fatal collective teardown away from participant workers.
         *
         * One persistent thread avoids both a per-failure allocation and the lock
         * inversion that can occur when a thread currently associated with one
         * NCCL/RCCL participant calls communicator abort. Exceptions are fatal:
         * after the first participant failure the collective context is explicitly
         * unusable, so continuing would only conceal asymmetric device state.
         */
        void failureCallbackLoop()
        {
            uint64_t handled_generation = 0;
            while (true)
            {
                std::function<void()> callback;
                {
                    std::unique_lock<std::mutex> lock(mutex_);
                    failure_callback_cv_.wait(
                        lock,
                        [this, handled_generation]()
                        {
                            return shutdown_ ||
                                   failure_callback_generation_ >
                                       handled_generation;
                        });
                    if (shutdown_)
                        return;
                    handled_generation = failure_callback_generation_;
                    callback = failure_callback_;
                }

                if (!callback)
                    continue;
                try
                {
                    callback();
                }
                catch (const std::exception &error)
                {
                    LOG_ERROR("[TPWorkerPool] Fatal collective-abort callback exception: "
                              << error.what());
                    std::terminate();
                }
                catch (...)
                {
                    LOG_ERROR("[TPWorkerPool] Fatal non-standard collective-abort callback exception");
                    std::terminate();
                }
            }
        }

        size_t num_workers_;
        std::vector<std::thread> workers_;
        std::thread failure_callback_worker_;
        std::vector<WorkerResult> results_;

        // Dispatch synchronization
        std::mutex mutex_;
        std::condition_variable dispatch_cv_;
        std::condition_variable failure_callback_cv_;
        std::function<bool(size_t)> work_fn_;
        std::function<void()> failure_callback_;
        uint64_t failure_callback_generation_ = 0;
        uint64_t generation_ = 0;
        uint64_t completed_generation_ = 0;
        uint64_t collected_generation_ = 0;
        size_t active_collectors_ = 0;
        bool shutdown_ = false;

        // Collection state is published under mutex_ with the dispatch state.
        std::condition_variable collect_cv_;
        std::atomic<size_t> completed_count_{0};
        std::atomic<size_t> first_failure_index_{SIZE_MAX};
    };

} // namespace llaminar2
