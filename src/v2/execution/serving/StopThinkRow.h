/**
 * @file StopThinkRow.h
 * @brief Per-request stop/think and forced-sequence state (design §6 stop/think row).
 *
 * Stop matching, reasoning budget and forced-sequence state are device-owned
 * request data; the host never derives them from wall clock. A reasoning
 * budget is bounded and never negative (saturating count of remaining budget
 * units); forced sequence enters/exits exactly once each; terminal stop is
 * observed exactly once. These laws are typed so invalid states cannot be
 * represented.
 */

#pragma once

#include <cstdint>
#include <stdexcept>

namespace llaminar2
{
    /** @brief Stop/think row phases. */
    enum class StopThinkPhase : std::uint8_t
    {
        Normal,
        Thinking,
        ForcedSequence,
        Terminal,
    };

    /** @brief Device-owned per-request stop/think row. */
    struct StopThinkRow
    {
        StopThinkPhase phase = StopThinkPhase::Normal;
        std::uint64_t reasoning_budget_remaining = 0;
        bool stop_matched = false;
        bool forced_sequence_active = false;
        bool depth_learning_skipped = false; ///< Forced-position clip disables MTP learning.
        bool forced_sequence_used = false;   ///< Exactly one forced sequence per transaction.

        /** @brief Enter the thinking phase (budget must be positive; a refusal
         *         must NOT mutate state). */
        void enterThinking()
        {
            if (reasoning_budget_remaining == 0)
                throw std::logic_error("Thinking requires a positive reasoning budget");
            require(StopThinkPhase::Normal, StopThinkPhase::Thinking);
        }

        /** @brief Consume one budget unit (saturates at zero, never negative). */
        void consumeReasoningUnit()
        {
            if (phase != StopThinkPhase::Thinking && phase != StopThinkPhase::ForcedSequence)
                throw std::logic_error("Reasoning units consumed only in think/forced phases");
            if (reasoning_budget_remaining > 0)
                --reasoning_budget_remaining;
        }

        /** @brief Forced sequence enters exactly once per transaction. */
        void enterForcedSequence()
        {
            if (forced_sequence_used)
                throw std::logic_error("Forced sequence used once per transaction");
            require(StopThinkPhase::Thinking, StopThinkPhase::ForcedSequence);
            forced_sequence_active = true;
            forced_sequence_used = true;
        }

        /** @brief Forced sequence exits exactly once. */
        void exitForcedSequence()
        {
            if (phase != StopThinkPhase::ForcedSequence || !forced_sequence_active)
                throw std::logic_error("Forced sequence exit requires an active forced sequence");
            forced_sequence_active = false;
            phase = StopThinkPhase::Thinking;
        }

        /** @brief Owned transaction end: Thinking/ForcedSequence back to Normal. */
        void endThinking()
        {
            if (phase == StopThinkPhase::ForcedSequence)
                throw std::logic_error(
                    "Forced sequence must exit BEFORE ending the transaction "
                    "(no stale forced_sequence_active in Normal)");
            if (phase != StopThinkPhase::Thinking)
                throw std::logic_error("endThinking requires an active think phase");
            phase = StopThinkPhase::Normal;
        }

        /** @brief Owned transaction boundary: resets the per-transaction flags. */
        void beginTransaction()
        {
            if (phase != StopThinkPhase::Normal)
                throw std::logic_error("Transaction boundary requires a clean prior transaction");
            // Owned RESET boundary: the per-transaction flags belong to the NEW
            // transaction and are cleared exactly here.
            forced_sequence_used = false;
            depth_learning_skipped = false;
        }

        /** @brief A forced position clipped the transaction: adaptive depth learning
         *         must be SKIPPED (never counted as poor model acceptance). */
        void noteForcedPositionClipped()
        {
            if (depth_learning_skipped)
                throw std::logic_error("Depth-learning skip recorded twice");
            depth_learning_skipped = true;
        }

        /** @brief Terminal stop observed exactly once (device-owned fact). */
        void observeTerminal()
        {
            if (phase == StopThinkPhase::Terminal || stop_matched)
                throw std::logic_error("Terminal stop observed exactly once");
            stop_matched = true;
            phase = StopThinkPhase::Terminal;
        }

    private:
        void require(StopThinkPhase from, StopThinkPhase to)
        {
            if (phase != from)
                throw std::logic_error("Invalid stop/think transition");
            phase = to;
        }
    };
} // namespace llaminar2
