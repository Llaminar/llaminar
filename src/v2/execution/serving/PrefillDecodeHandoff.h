/**
 * @file PrefillDecodeHandoff.h
 * @brief Typed slot-owned prefill/decode handoff state (design §9.5).
 *
 * Terminal prefill logits and hidden state survive every intervening
 * request's work. The typed handoff distinguishes unsampled ready logits from
 * an already sampled/recorded condition and from a completed/pending first
 * sidecar — replacing runner-global booleans with a state machine whose
 * invalid moves are unrepresentable. Terminal FP32 logits are a physical-memory
 * obligation (sizeof(float) x vocabulary_size per slot), charged through the
 * same BOM authority. Per-slot handoff remains valid until its exact
 * consumer/event retires; a sampled correction is never sampled, emitted or
 * inserted into history twice.
 */

#pragma once

#include "planning/PhysicalMemoryBOM.h"

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace llaminar2
{
    /** @brief Slot-owned handoff phases (design §9.5). */
    enum class HandoffPhase : std::uint8_t
    {
        PrefillPending,       ///< Prefill not finished; no terminal logits yet.
        UnsampledReady,       ///< Terminal logits ready, NOT yet sampled.
        SampledRecorded,      ///< Sampled condition recorded, not emitted/inserted.
        FirstSidecarPending,  ///< First sidecar proposed, completion pending.
        FirstSidecarComplete, ///< First sidecar completed.
        Retired,              ///< Exact consumer/event retired the handoff.
    };

    /** @brief Typed slot-owned handoff; invalid transitions are refused. */
    struct PrefillDecodeHandoff
    {
        HandoffPhase phase = HandoffPhase::PrefillPending;
        std::uint64_t slot = 0; ///< Owning slot (identity only; never part of RNG).

        void publishUnsampled()
        {
            transition(HandoffPhase::PrefillPending, HandoffPhase::UnsampledReady);
        }

        void markSampledRecorded()
        {
            // Sampled/recorded exactly once: only from unsampled-ready.
            transition(HandoffPhase::UnsampledReady, HandoffPhase::SampledRecorded);
        }

        void markFirstSidecarPending()
        {
            transition(HandoffPhase::SampledRecorded, HandoffPhase::FirstSidecarPending);
        }

        void markFirstSidecarComplete()
        {
            transition(HandoffPhase::FirstSidecarPending, HandoffPhase::FirstSidecarComplete);
        }

        /** @brief Exact consumer/event retirement (from any live phase). */
        void retire()
        {
            if (phase == HandoffPhase::Retired)
                throw std::logic_error("Handoff already retired");
            phase = HandoffPhase::Retired;
        }

    private:
        void transition(HandoffPhase from, HandoffPhase to)
        {
            if (phase != from)
                throw std::logic_error("Invalid prefill/decode handoff transition");
            phase = to;
        }
    };

    /**
     * @brief Terminal FP32 logits hosting obligation (design §9.5).
     *
     * @param slots Admitted slots.
     * @param vocabulary_size Admitted vocabulary token count.
     * @return Exact bytes: slots * sizeof(float) * vocabulary_size (before hidden,
     *         sidecar, metadata and event storage).
     */
    [[nodiscard]] inline std::uint64_t terminalLogitsDemand(std::uint64_t slots,
                                                            std::uint64_t vocabulary_size)
    {
        if (slots == 0 || vocabulary_size == 0)
            throw std::invalid_argument("Terminal-logit slots and vocabulary must be positive");
        constexpr std::uint64_t kBytesPerToken = sizeof(float);
        if (vocabulary_size > std::numeric_limits<std::uint64_t>::max() / kBytesPerToken)
            throw std::overflow_error("Terminal-logit demand overflows uint64");
        const auto per_slot = vocabulary_size * kBytesPerToken; // no wrap
        if (slots > std::numeric_limits<std::uint64_t>::max() / per_slot)
            throw std::overflow_error("Terminal-logit demand overflows uint64");
        return slots * per_slot;
    }

    /** @brief Charge terminal-logit banks on the physical-memory BOM authority. */
    inline void addTerminalLogitsDemand(PhysicalMemoryBOMBuilder &builder, std::uint64_t slots,
                                        std::uint64_t vocabulary_size)
    {
        builder.add(PhysicalMemoryOwner::SequenceMetadata,
                    terminalLogitsDemand(slots, vocabulary_size));
    }
} // namespace llaminar2
