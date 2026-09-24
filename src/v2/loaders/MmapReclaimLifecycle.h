#pragma once

/**
 * @file MmapReclaimLifecycle.h
 * @brief Exactly-once asynchronous lifecycle for model-mapping reclamation.
 *
 * Model mappings become reclaimable only after graph materialization and the
 * first successful prefill have retired every host-side consumer.  Reclaiming
 * a large mapping can spend seconds walking page tables, so that work must not
 * run on an inference authority thread.  This class owns one prestarted worker,
 * accepts one typed submission, and exposes an explicit completion barrier for
 * teardown or a later host allocation that depends on the reclaimed capacity.
 *
 * The worker is created during model setup rather than at the first-prefill
 * boundary.  Submission is therefore only a mutex-protected state transition
 * and notification; it performs no reclamation and creates no thread in the
 * inference path.
 */

#include "../utils/Logger.h"

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

namespace llaminar2
{
    /**
     * @brief Owns one asynchronous, exactly-once mmap reclaim operation.
     *
     * The supplied work function must own the complete safe reclaim sequence,
     * including accelerator host-registration retirement before page advice.
     * The lifecycle joins its worker during destruction, so the function may
     * safely borrow state owned by the containing WeightManager.
     */
    class MmapReclaimLifecycle final
    {
    public:
        /** @brief Observable phases of the one-shot reclaim state machine. */
        enum class State : std::uint8_t
        {
            Idle,     ///< Worker is prepared, but reclaim has not been submitted.
            Running,  ///< The background worker owns the reclaim operation.
            Complete, ///< Reclaim completed successfully, including intentional no-op work.
            Failed,   ///< Reclaim threw; the completion barrier rethrows the failure.
        };

        /** @brief Result of attempting to submit the one-shot operation. */
        enum class Submission : std::uint8_t
        {
            Scheduled,       ///< This caller performed the Idle -> Running transition.
            AlreadyRunning,  ///< Another caller already submitted the operation.
            AlreadyComplete, ///< The operation has already completed successfully.
            AlreadyFailed,   ///< The submitted operation completed with an error.
        };

        /**
         * @brief Typed result returned by the explicit completion barrier.
         *
         * An Idle result means no reclaim boundary was published.  Complete
         * includes a zero-byte result for memory-filesystem mappings whose
         * correct reclaim policy is to retain their process mappings.
         */
        struct Completion
        {
            State state = State::Idle;
            std::size_t advised_bytes = 0;
        };

        using Work = std::function<std::size_t()>;

        /**
         * @brief Construct the lifecycle and prestart its dormant worker.
         *
         * @param work Complete reclaim operation executed after schedule().
         * @throws std::invalid_argument when @p work is empty.
         */
        explicit MmapReclaimLifecycle(Work work)
            : work_(std::move(work))
        {
            if (!work_)
            {
                throw std::invalid_argument(
                    "MmapReclaimLifecycle requires a reclaim operation");
            }

            // Starting during model construction keeps thread creation and its
            // allocator/loader work out of the first-prefill inference boundary.
            worker_ = std::thread([this]() { workerLoop(); });
        }

        /**
         * @brief Join the worker before borrowed owner state can be destroyed.
         *
         * Destruction never abandons an in-flight reclaim.  Failures remain
         * observable through awaitBeforeHostAllocation(); teardown logs them
         * because a destructor cannot safely propagate an exception.
         */
        ~MmapReclaimLifecycle() noexcept
        {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                stop_requested_ = true;
            }
            cv_.notify_all();
            if (worker_.joinable())
            {
                worker_.join();
            }

            State terminal_state = State::Idle;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                terminal_state = state_;
            }
            if (terminal_state == State::Failed)
            {
                LOG_ERROR("[MmapReclaimLifecycle] asynchronous mmap reclaim failed before teardown");
            }
        }

        MmapReclaimLifecycle(const MmapReclaimLifecycle &) = delete;
        MmapReclaimLifecycle &operator=(const MmapReclaimLifecycle &) = delete;
        MmapReclaimLifecycle(MmapReclaimLifecycle &&) = delete;
        MmapReclaimLifecycle &operator=(MmapReclaimLifecycle &&) = delete;

        /**
         * @brief Publish the reclaim boundary without executing work inline.
         *
         * Multiple rank/device lifecycle owners may observe the same first
         * prefill.  Only the first caller transitions Idle -> Running; all later
         * callers receive an idempotent typed result.
         *
         * @return Exact outcome of the submission attempt.
         */
        [[nodiscard]] Submission schedule() noexcept
        {
            std::lock_guard<std::mutex> lock(mutex_);
            switch (state_)
            {
            case State::Idle:
                state_ = State::Running;
                cv_.notify_all();
                return Submission::Scheduled;
            case State::Running:
                return Submission::AlreadyRunning;
            case State::Complete:
                return Submission::AlreadyComplete;
            case State::Failed:
                return Submission::AlreadyFailed;
            }
            std::terminate();
        }

        /**
         * @brief Wait before a host allocation that depends on reclaimed RAM.
         *
         * Ordinary inference must not call this method.  It is the explicit
         * memory-pressure edge for JIT/model admission and is also used by owner
         * teardown.  Calling it before submission returns Idle immediately.
         *
         * @return Terminal state and the number of bytes actually advised.
         * @throws Any exception raised by the background reclaim operation.
         */
        [[nodiscard]] Completion awaitBeforeHostAllocation()
        {
            std::exception_ptr failure;
            Completion completion;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this]() { return state_ != State::Running; });
                completion = Completion{
                    .state = state_,
                    .advised_bytes = advised_bytes_,
                };
                failure = failure_;
            }

            if (failure)
            {
                std::rethrow_exception(failure);
            }
            return completion;
        }

        /** @return Current typed lifecycle state without changing it. */
        [[nodiscard]] State state() const noexcept
        {
            std::lock_guard<std::mutex> lock(mutex_);
            return state_;
        }

        /** @return Stable diagnostic name for a submission outcome. */
        static constexpr const char *toString(Submission submission) noexcept
        {
            switch (submission)
            {
            case Submission::Scheduled:
                return "scheduled";
            case Submission::AlreadyRunning:
                return "already_running";
            case Submission::AlreadyComplete:
                return "already_complete";
            case Submission::AlreadyFailed:
                return "already_failed";
            }
            return "invalid";
        }

    private:
        /**
         * @brief Wait for submission, execute exactly once, and publish terminal state.
         *
         * The exception boundary belongs here so no background exception can
         * escape the thread and terminate the process without a diagnostic.
         */
        void workerLoop() noexcept
        {
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this]() {
                    return state_ == State::Running || stop_requested_;
                });
                if (stop_requested_ && state_ == State::Idle)
                {
                    return;
                }
            }

            std::size_t advised_bytes = 0;
            std::exception_ptr failure;
            try
            {
                advised_bytes = work_();
            }
            catch (...)
            {
                failure = std::current_exception();
            }

            {
                std::lock_guard<std::mutex> lock(mutex_);
                advised_bytes_ = advised_bytes;
                failure_ = failure;
                state_ = failure ? State::Failed : State::Complete;
            }
            cv_.notify_all();
        }

        Work work_; ///< Immutable operation installed before the worker starts.
        mutable std::mutex mutex_; ///< Protects every lifecycle field below.
        std::condition_variable cv_; ///< Submission and completion publication edge.
        State state_ = State::Idle; ///< Sole state-machine authority.
        std::size_t advised_bytes_ = 0; ///< Published successful work result.
        std::exception_ptr failure_; ///< Published failure rethrown by the barrier.
        bool stop_requested_ = false; ///< Lets an unsubmitted worker retire at teardown.
        std::thread worker_; ///< Prestarted one-shot worker; declared last for safe join ordering.
    };
} // namespace llaminar2
