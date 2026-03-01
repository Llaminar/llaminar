#pragma once

// ============================================================================
// TPWorkerPool — Persistent thread pool for TP device forwarding
// ============================================================================
//
// Replaces std::async(std::launch::async) per decode step with pre-spawned
// worker threads. Benefits:
//   - Eliminates thread creation/destruction overhead (~100-150µs per step)
//   - Worker threads retain HIP/CUDA device context (no per-call setup)
//   - Lower wakeup latency via condition_variable vs pthread_create
//
// Thread lifecycle:
//   1. Construction: spawns N persistent worker threads
//   2. dispatch(): wakes all workers with a callable
//   3. collectAll(): blocks until all workers complete, returns results
//   4. Destruction: signals workers to exit and joins
//
// IMPORTANT: Workers do NOT call hipSetDevice — the DeviceGraphOrchestrator
// uses stream-based GPU operations which are device-agnostic. The HIP runtime
// routes calls to the correct GPU based on the stream's device binding.
// ============================================================================

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace llaminar2
{

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
            workers_.reserve(num_workers);
            for (size_t i = 0; i < num_workers; ++i)
            {
                workers_.emplace_back([this, i]()
                                      { workerLoop(i); });
            }
            // Note: construction logging handled by caller (MultiDeviceOrchestrator)
        }

        ~TPWorkerPool()
        {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                shutdown_ = true;
            }
            dispatch_cv_.notify_all();

            for (auto &t : workers_)
            {
                if (t.joinable())
                    t.join();
            }
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
            {
                std::lock_guard<std::mutex> lock(mutex_);
                work_fn_ = std::move(fn);
                generation_++;
                // Reset results
                for (auto &r : results_)
                {
                    r.success = false;
                    r.completed = false;
                    r.exception = nullptr;
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
            std::unique_lock<std::mutex> lock(collect_mutex_);
            auto pred = [this]()
            { return completed_count_.load(std::memory_order_acquire) >= num_workers_; };

            if (timeout_ms > 0)
            {
                collect_cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), pred);
            }
            else
            {
                collect_cv_.wait(lock, pred);
            }

            return results_;
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
         * callback fires immediately on the failing worker's thread. Use it
         * to abort collective operations (e.g., NCCL/RCCL) so that workers
         * stuck in collectives can unblock and complete.
         *
         * The callback must be thread-safe and non-blocking.
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
                {
                    std::unique_lock<std::mutex> lock(mutex_);
                    dispatch_cv_.wait(lock, [this, last_gen]()
                                      { return shutdown_ || generation_ > last_gen; });

                    if (shutdown_)
                        return;

                    last_gen = generation_;
                    fn = work_fn_;
                }

                // Execute the work
                WorkerResult result;
                result.worker_index = index;
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
                            // First failure — invoke abort callback to unblock stuck workers
                            std::function<void()> cb;
                            {
                                std::lock_guard<std::mutex> lock(mutex_);
                                cb = failure_callback_;
                            }
                            if (cb)
                                cb();
                        }
                    }
                }
                catch (...)
                {
                    result.success = false;
                    result.exception = std::current_exception();
                    // Record first failure
                    size_t expected = SIZE_MAX;
                    if (first_failure_index_.compare_exchange_strong(
                            expected, index, std::memory_order_acq_rel))
                    {
                        // First failure — invoke abort callback to unblock stuck workers
                        std::function<void()> cb;
                        {
                            std::lock_guard<std::mutex> lock(mutex_);
                            cb = failure_callback_;
                        }
                        if (cb)
                            cb();
                    }
                }
                result.completed = true;

                results_[index] = std::move(result);

                // Signal completion
                auto prev = completed_count_.fetch_add(1, std::memory_order_acq_rel);
                if (prev + 1 >= num_workers_)
                {
                    // Last worker — wake the collector
                    collect_cv_.notify_one();
                }
            }
        }

        size_t num_workers_;
        std::vector<std::thread> workers_;
        std::vector<WorkerResult> results_;

        // Dispatch synchronization
        std::mutex mutex_;
        std::condition_variable dispatch_cv_;
        std::function<bool(size_t)> work_fn_;
        std::function<void()> failure_callback_;
        uint64_t generation_ = 0;
        bool shutdown_ = false;

        // Collection synchronization
        std::mutex collect_mutex_;
        std::condition_variable collect_cv_;
        std::atomic<size_t> completed_count_{0};
        std::atomic<size_t> first_failure_index_{SIZE_MAX};
    };

} // namespace llaminar2
