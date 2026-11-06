/**
 * @file RetainedParentTicketServiceWorker.cpp
 * @brief Lifecycle implementation for retained-parent CPU ticket service.
 *
 * Construction completes every lifecycle member before publishing this object
 * to its persistent thread. Dispatch and retirement then transfer the borrowed
 * invocation under the same mutex; teardown joins before any state is released.
 */

#include "RetainedParentTicketServiceWorker.h"

#include "../../../utils/Logger.h"

#include <exception>
#include <stdexcept>

namespace llaminar2
{
    RetainedParentTicketServiceWorker::RetainedParentTicketServiceWorker(pthread_t) noexcept {}

    void* RetainedParentTicketServiceWorker::invoke(void* state) noexcept
    {
        static_cast<RetainedParentTicketServiceWorker*>(state)->run();
        return nullptr;
    }

    RetainedParentTicketServiceWorker::~RetainedParentTicketServiceWorker()
    {
        {
            std::unique_lock lock(mutex_);
            work_complete_.wait(
                lock,
                [this]
                {
                    return state_ != State::Queued &&
                           state_ != State::Running;
                });
            state_ = State::Stopping;
        }
        work_ready_.notify_one();
        if (running_ && ::pthread_join(worker_,nullptr)!=0)
            std::terminate();
        running_=false;
    }

    RetainedParentTicketServiceWorker::Ticket
    RetainedParentTicketServiceWorker::dispatch(Task task, void *context)
    {
        Ticket ticket;
        {
            std::lock_guard lock(mutex_);
            // Withdraw first, including every rejection. No failed dispatch
            // leaves a fixture stack address installed in the original worker.
            auto* borrowed = next_borrowed_entry_;
            next_borrowed_entry_ = nullptr;
            if (borrowed)
                borrowed->outcome.store(BorrowedEntryOutcome::Rejected, std::memory_order_release);
            if (!task || !context)
                throw std::invalid_argument(
                    "Retained-parent ticket service requires a complete task descriptor");
            if (state_ != State::Idle)
                throw std::logic_error("Retained-parent ticket service cannot overlap invocations");
            if (borrowed)
            {
                if (task != borrowed->expected_task || context != borrowed->expected_context ||
                    !borrowed->wrapper_task || !borrowed->wrapper_context)
                    throw std::logic_error("Closed worker wrapper changed its original dispatch task or owner");
                task = borrowed->wrapper_task;
                context = borrowed->wrapper_context;
            }
            ++generation_;
            if (generation_ == 0u)
                ++generation_;
            task_ = task;
            context_ = context;
            task_result_ = false;
            state_ = State::Queued;
            ticket.generation = generation_;
            if (borrowed)
                borrowed->outcome.store(BorrowedEntryOutcome::Dispatched, std::memory_order_release);
        }
        work_ready_.notify_one();
        return ticket;
    }

    bool RetainedParentTicketServiceWorker::await(Ticket ticket)
    {
        if (!ticket.valid())
        {
            throw std::logic_error(
                "Retained-parent ticket service received an invalid ticket");
        }

        std::unique_lock lock(mutex_);
        if (ticket.generation != generation_ || state_ == State::Idle ||
            state_ == State::Stopping)
        {
            throw std::logic_error(
                "Retained-parent ticket service received a stale or unowned ticket");
        }
        work_complete_.wait(
            lock,
            [this, ticket]
            {
                return state_ == State::Complete &&
                       generation_ == ticket.generation;
            });
        const bool result = task_result_;
        task_ = nullptr;
        context_ = nullptr;
        task_result_ = false;
        state_ = State::Idle;
        return result;
    }

    bool RetainedParentTicketServiceWorker::idle() const noexcept
    {
        std::lock_guard lock(mutex_);
        return state_ == State::Idle;
    }

    void RetainedParentTicketServiceWorker::run() noexcept
    {
        for (;;)
        {
            Task task = nullptr;
            void *context = nullptr;
            {
                std::unique_lock lock(mutex_);
                work_ready_.wait(
                    lock,
                    [this]
                    {
                        return state_ == State::Queued ||
                               state_ == State::Stopping;
                    });
                if (state_ == State::Stopping)
                    return;
                task = task_;
                context = context_;
                state_ = State::Running;
            }

            bool result = false;
            try
            {
                result = task(context);
            }
            catch (const std::exception &error)
            {
                LOG_ERROR(
                    "[RetainedParentTicketServiceWorker] Ticket service threw: "
                    << error.what());
            }
            catch (...)
            {
                LOG_ERROR(
                    "[RetainedParentTicketServiceWorker] Ticket service threw a non-standard exception");
            }

            {
                std::lock_guard lock(mutex_);
                task_result_ = result;
                state_ = State::Complete;
            }
            work_complete_.notify_all();
        }
    }
} // namespace llaminar2
