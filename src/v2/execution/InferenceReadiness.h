/**
 * @file InferenceReadiness.h
 * @brief Model-agnostic readiness boundary shared by every serving mode.
 *
 * Application modes may observe only whether the initialized runtime is still
 * preparing, ready to accept ordinary requests, or fatally unavailable.  The
 * subsystem, workload, and protocol that produce readiness remain private to
 * the orchestration implementation.
 */

#pragma once

#include <cstdint>
#include <string>

namespace llaminar2
{
    /** Public lifecycle state before the first admitted inference request. */
    enum class InferenceReadinessState : std::uint8_t
    {
        Preparing, ///< Internal production preparation is still in progress.
        Ready,     ///< Ordinary inference requests may be admitted.
        Failed,    ///< Preparation failed and the runner is unusable.
    };

    /** Model-agnostic readiness snapshot exposed to applications. */
    struct InferenceReadiness
    {
        InferenceReadinessState state = InferenceReadinessState::Ready;
        std::string diagnostic; ///< First fatal detail only when Failed.

        /** @return Whether an ordinary inference request may start. */
        [[nodiscard]] bool ready() const noexcept
        {
            return state == InferenceReadinessState::Ready;
        }

        /** @return Whether preparation reached a terminal failure. */
        [[nodiscard]] bool failed() const noexcept
        {
            return state == InferenceReadinessState::Failed;
        }
    };
} // namespace llaminar2
