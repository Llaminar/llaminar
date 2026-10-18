/**
 * @file OrchestrationInitializationLifecycle.h
 * @brief Distinguish admitted metadata from a complete prepared inference lifetime.
 *
 * A successful dry run owns a validated plan, but no prepared model allocations
 * or executable graph. Publishing both outcomes as one initialized boolean
 * makes inference and final prepared-weight retention accept an admission-only
 * runner. This authority publishes one typed completion and rejects promotion
 * or duplicate publication without the shutdown boundary of that lifetime.
 */
#pragma once

#include <atomic>
#include <stdexcept>

namespace llaminar2
{
    /** @brief Atomic publication of one runner's completed initialization kind. */
    class OrchestrationInitializationLifecycle final
    {
    public:
        /** @brief Completed preparation, distinct from temporary setup resources. */
        enum class Completion
        {
            None,
            Admission,
            Inference,
        };

        /** @brief Construct a fresh lifetime with no completed initialization. */
        OrchestrationInitializationLifecycle() = default;

        /**
         * @brief Bind an already constructed runner's completed preparation.
         * @param completion Exact completion supplied by the owning constructor.
         * @throws std::invalid_argument for an unknown completion value.
         */
        explicit OrchestrationInitializationLifecycle(Completion completion)
            : completion_(completion)
        {
            if (completion != Completion::None && completion != Completion::Admission &&
                completion != Completion::Inference)
                throw std::invalid_argument("Unknown orchestration initialization completion");
        }

        /** @return One acquire-ordered lifecycle observation. */
        Completion completion() const noexcept
        {
            return completion_.load(std::memory_order_acquire);
        }

        /** @return Whether admission finished, with or without subsequent graph preparation. */
        bool admissionComplete() const noexcept
        {
            return completion() != Completion::None;
        }

        /** @return Whether inference and final prepared-allocation sealing are legal. */
        bool readyForInference() const noexcept
        {
            return completion() == Completion::Inference;
        }

        /**
         * @brief Publish one fully completed initialization exactly once.
         * @param completed Admission-only or complete inference preparation.
         * @throws std::invalid_argument for an absent or unknown completion kind.
         * @throws std::logic_error when a prior completion has not been retired.
         *
         * In particular, Admission cannot turn into Inference merely because
         * a caller repeats initialize(). A new preparation must begin after
         * shutdown has released its existing plans and collective contexts.
         */
        void publish(Completion completed)
        {
            if (completed != Completion::Admission && completed != Completion::Inference)
                throw std::invalid_argument("Cannot publish absent orchestration initialization");
            auto expected = Completion::None;
            if (!completion_.compare_exchange_strong(expected, completed,
                    std::memory_order_release, std::memory_order_relaxed))
                throw std::logic_error("Orchestration initialization already completed; shutdown is required");
        }

        /**
         * @brief Retire completion after the owning runner has released resources.
         *
         * Request reset never calls this method: retained inference graphs
         * remain ready across requests. This is the terminal shutdown boundary.
         */
        void retire() noexcept
        {
            completion_.store(Completion::None, std::memory_order_release);
        }

    private:
        std::atomic<Completion> completion_{Completion::None};
    };
}
