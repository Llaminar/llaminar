/**
 * @file TerminalArbitration.h
 * @brief Separate execution/delivery outcomes and ordered terminal arbitration (design §10.3).
 *
 * Execution outcome (Completed/Cancelled/Retired) and delivery outcome
 * (Pending/Drained/Abandoned) are INDEPENDENT authorities: a disconnect never
 * rewrites execution history and a slow socket never delays execution
 * completion/slot retirement. A terminal execution outcome is recorded exactly
 * once; any LATE cancellation after a terminal outcome returns AlreadyTerminal
 * (valid late cancellation) instead of rewriting the outcome. Response closure
 * separately requires its own delivery/publication retirement (publication
 * lease in the executor), never this authority.
 */

#pragma once

#include <cstdint>
#include <stdexcept>

namespace llaminar2
{
    /** @brief Execution outcome (design §10.3): one of Completed/Cancelled/Retired. */
    enum class ExecutionOutcome : std::uint8_t
    {
        None,
        Completed,
        Cancelled,
        Retired,
    };

    /** @brief Independent delivery outcome (never rewrites execution history). */
    enum class DeliveryOutcome : std::uint8_t
    {
        Pending,
        Drained,
        Abandoned,
    };

    /** @brief Typed result of a late terminal attempt after a terminal outcome. */
    struct AlreadyTerminalError : std::logic_error
    {
        explicit AlreadyTerminalError(ExecutionOutcome outcome)
            : std::logic_error("Already terminal"), existing(outcome) {}
        ExecutionOutcome existing;
    };

    /** @brief Ordered terminal arbitration: two independent authorities. */
    struct TerminalArbitration
    {
        ExecutionOutcome execution = ExecutionOutcome::None;
        DeliveryOutcome delivery = DeliveryOutcome::Pending;
        bool model_terminal_observed = false;

        void observeModelTerminal() { model_terminal_observed = true; }
        void markDeliveryDrained() { delivery = DeliveryOutcome::Drained; }
        void markDeliveryAbandoned() { delivery = DeliveryOutcome::Abandoned; }

        /** @brief Completion requires the model terminal observation ONLY (not delivery). */
        void complete()
        {
            if (execution != ExecutionOutcome::None)
                throw AlreadyTerminalError(execution);
            if (!model_terminal_observed)
                throw std::logic_error("Completion requires the model terminal observation");
            execution = ExecutionOutcome::Completed;
        }

        /** @brief LATE cancellation after a terminal outcome yields AlreadyTerminal. */
        void cancel()
        {
            if (execution != ExecutionOutcome::None)
                throw AlreadyTerminalError(execution);
            execution = ExecutionOutcome::Cancelled;
        }

        void retire()
        {
            if (execution != ExecutionOutcome::None)
                throw AlreadyTerminalError(execution);
            execution = ExecutionOutcome::Retired;
        }

        [[nodiscard]] bool isTerminal() const noexcept { return execution != ExecutionOutcome::None; }
    };
} // namespace llaminar2
