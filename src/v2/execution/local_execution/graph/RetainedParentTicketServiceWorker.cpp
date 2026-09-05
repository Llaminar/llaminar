/**
 * @file RetainedParentTicketServiceWorker.cpp
 * @brief Lifecycle implementation for retained-parent CPU ticket service.
 */

#include "RetainedParentTicketServiceWorker.h"

#include "../../../utils/Logger.h"

#include <exception>
#include <stdexcept>

namespace llaminar2
{
    RetainedParentTicketServiceWorker::RetainedParentTicketServiceWorker()
        : worker_(&RetainedParentTicketServiceWorker::run, this)
    {
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
        if (worker_.joinable())
            worker_.join();
    }

    RetainedParentTicketServiceWorker::Ticket
    RetainedParentTicketServiceWorker::dispatch(Task task, void *context)
    {
        if (!task || !context)
        {
            throw std::invalid_argument(
                "Retained-parent ticket service requires a complete task descriptor");
        }

        Ticket ticket;
        {
            std::lock_guard lock(mutex_);
            if (state_ != State::Idle)
            {
                throw std::logic_error(
                    "Retained-parent ticket service cannot overlap invocations");
            }
            ++generation_;
            if (generation_ == 0u)
                ++generation_;
            task_ = task;
            context_ = context;
            task_result_ = false;
            state_ = State::Queued;
            ticket.generation = generation_;
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
