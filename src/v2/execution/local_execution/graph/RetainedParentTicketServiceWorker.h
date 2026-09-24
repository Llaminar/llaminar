/**
 * @file RetainedParentTicketServiceWorker.h
 * @brief Persistent host worker for one retained-parent CPU ticket lane.
 *
 * A heterogeneous retained GPU parent can begin consuming mapped CPU return
 * tickets before the backend call that submits the complete native graph has
 * returned to its caller. Large HIP graphs make that behaviour observable:
 * waiting to start the CPU service until after `hipGraphLaunch()` returns can
 * deadlock graph submission against the first unpublished CPU return.
 *
 * This worker is created during graph setup and reused for every transaction.
 * Its job descriptor is two machine words, so dispatch performs no thread
 * creation or task allocation in the inference path. The caller retains every
 * object named by the descriptor until @ref await completes.
 */

#pragma once

#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>

namespace llaminar2
{
    /**
     * @brief Runs one pre-armed CPU ticket service independently of GPU submission.
     *
     * The state machine is deliberately single-slot: one DeviceGraphExecutor
     * owns one serial graph transaction at a time, and overlapping dispatch is
     * therefore an invalid lifecycle rather than a queueing opportunity.
     * `dispatch()` publishes a borrowed invocation, the persistent thread runs
     * it once, and `await()` consumes the matching generation before another
     * invocation may be installed.
     */
    class RetainedParentTicketServiceWorker final
    {
    public:
        /** @brief Allocation-free task entry used by the persistent worker. */
        using Task = bool (*)(void *context);

        /** @brief Opaque identity for one accepted service invocation. */
        struct Ticket
        {
            std::uint64_t generation = 0u; ///< Strictly increasing worker generation.

            /** @return Whether this ticket identifies a real invocation. */
            [[nodiscard]] constexpr bool valid() const noexcept
            {
                return generation != 0u;
            }
        };

        /** @brief Start the persistent worker in its idle state. */
        RetainedParentTicketServiceWorker();

        /**
         * @brief Join the persistent worker after any accepted invocation retires.
         *
         * Normal owners call @ref await before teardown. If stack unwinding
         * reaches this destructor with a live invocation, destruction still
         * waits for that invocation because abandoning borrowed graph state
         * would be a use-after-free.
         */
        ~RetainedParentTicketServiceWorker();

        RetainedParentTicketServiceWorker(
            const RetainedParentTicketServiceWorker &) = delete;
        RetainedParentTicketServiceWorker &operator=(
            const RetainedParentTicketServiceWorker &) = delete;
        RetainedParentTicketServiceWorker(
            RetainedParentTicketServiceWorker &&) = delete;
        RetainedParentTicketServiceWorker &operator=(
            RetainedParentTicketServiceWorker &&) = delete;

        /**
         * @brief Publish one borrowed service invocation to the idle worker.
         * @param task Non-null allocation-free task entry.
         * @param context Non-null caller-owned state retained through await().
         * @return Exact generation that @ref await must consume.
         * @throws std::logic_error if a prior invocation is still live.
         * @throws std::invalid_argument for an incomplete task descriptor.
         */
        [[nodiscard]] Ticket dispatch(Task task, void *context);

        /**
         * @brief Wait for and consume one exact invocation result.
         * @param ticket Ticket returned by the matching @ref dispatch.
         * @return True only when the task completed successfully.
         * @throws std::logic_error if the ticket is stale or not outstanding.
         */
        [[nodiscard]] bool await(Ticket ticket);

        /** @return Whether no invocation is queued, running, or awaiting collection. */
        [[nodiscard]] bool idle() const noexcept;

    private:
        /** @brief Complete typed lifecycle of the worker's single task slot. */
        enum class State : std::uint8_t
        {
            Idle = 0, ///< No borrowed invocation is installed.
            Queued, ///< Invocation is published but has not begun.
            Running, ///< Persistent thread currently owns the invocation.
            Complete, ///< Result is ready and must be consumed by await().
            Stopping, ///< Destructor has prohibited future dispatch.
        };

        /** @brief Persistent thread loop; all state transitions hold mutex_. */
        void run() noexcept;

        mutable std::mutex mutex_;
        std::condition_variable work_ready_;
        std::condition_variable work_complete_;
        std::thread worker_;
        State state_ = State::Idle;
        Task task_ = nullptr;
        void *context_ = nullptr;
        std::uint64_t generation_ = 0u;
        bool task_result_ = false;
    };
} // namespace llaminar2
