/**
 * @file MoEOverlayActivationRendezvousDeadline.h
 * @brief Typed no-progress deadlines for mapped ExpertOverlay activation edges.
 *
 * One mapped activation transaction contains many ordered dispatch/return
 * rendezvous.  The canonical collective timeout bounds one missing peer edge;
 * it is not a wall-clock budget for the complete multi-layer transaction.
 * This value type makes that scope explicit and lets deterministic tests drive
 * a synthetic monotonic clock without sleeping.
 */

#pragma once

#include <chrono>
#include <cstdint>
#include <optional>
#include <stdexcept>

namespace llaminar2
{
    /** Exact peer publication whose absence is currently being bounded. */
    enum class MoEOverlayActivationRendezvousKind : std::uint32_t
    {
        DispatchPublication = 1u, ///< One layer's dispatch timeline edge.
        EndpointCompletion = 2u, ///< Both endpoints' terminal Complete stores.
    };

    /**
     * @brief Immutable deadline for exactly one activation rendezvous.
     *
     * Construct a fresh instance whenever the expected protocol edge changes.
     * In particular, consuming layer N and advancing to layer N+1 starts a new
     * deadline.  Expensive local expert compute occurs between rendezvous and
     * is deliberately outside this wait budget.
     */
    class MoEOverlayActivationRendezvousDeadline final
    {
    public:
        using Clock = std::chrono::steady_clock;
        using TimePoint = Clock::time_point;
        using Duration = Clock::duration;

        /**
         * @brief Begin one exact peer-publication wait.
         * @param kind Typed publication expected by the caller.
         * @param timeout Positive no-progress interval for that publication.
         * @param now Monotonic start time; injectable for deterministic tests.
         * @return Immutable rendezvous deadline.
         * @throws std::invalid_argument for an unknown kind or empty timeout.
         */
        [[nodiscard]] static MoEOverlayActivationRendezvousDeadline begin(
            MoEOverlayActivationRendezvousKind kind,
            Duration timeout,
            TimePoint now = Clock::now())
        {
            if ((kind != MoEOverlayActivationRendezvousKind::
                             DispatchPublication &&
                 kind != MoEOverlayActivationRendezvousKind::
                             EndpointCompletion) ||
                timeout <= Duration::zero())
            {
                throw std::invalid_argument(
                    "ExpertOverlay activation rendezvous requires a typed kind and positive timeout");
            }
            return MoEOverlayActivationRendezvousDeadline{
                kind,
                now + timeout};
        }

        /** @return The exact peer publication governed by this deadline. */
        [[nodiscard]] MoEOverlayActivationRendezvousKind kind() const noexcept
        {
            return kind_;
        }

        /**
         * @return Whether the peer may still publish within this rendezvous.
         * @param now Monotonic observation time.
         */
        [[nodiscard]] bool waitingAllowed(
            TimePoint now = Clock::now()) const noexcept
        {
            return now < deadline_;
        }

        /**
         * @return Signed microseconds remaining; non-positive means expired.
         * @param now Monotonic observation time.
         */
        [[nodiscard]] std::int64_t remainingMicroseconds(
            TimePoint now = Clock::now()) const noexcept
        {
            return std::chrono::duration_cast<std::chrono::microseconds>(
                       deadline_ - now)
                .count();
        }

        /**
         * @return Absolute monotonic deadline used as the protocol's earliest
         *         legal timeout observation, or empty if not representable.
         */
        [[nodiscard]] std::optional<std::uint64_t>
        deadlineNanoseconds() const noexcept
        {
            const auto value =
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    deadline_.time_since_epoch())
                    .count();
            if (value <= 0)
                return std::nullopt;
            return static_cast<std::uint64_t>(value);
        }

        /**
         * @brief Materialize terminal timeout evidence only after expiry.
         * @param now Monotonic observation time.
         * @return Absolute observation timestamp, or empty while progress is
         *         still permitted or when the clock value is not representable.
         */
        [[nodiscard]] std::optional<std::uint64_t>
        timeoutObservationNanoseconds(
            TimePoint now = Clock::now()) const noexcept
        {
            if (waitingAllowed(now))
                return std::nullopt;
            const auto value =
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    now.time_since_epoch())
                    .count();
            if (value <= 0)
                return std::nullopt;
            return static_cast<std::uint64_t>(value);
        }

    private:
        /** Constructed only through @ref begin after validating the policy. */
        MoEOverlayActivationRendezvousDeadline(
            MoEOverlayActivationRendezvousKind kind,
            TimePoint deadline) noexcept
            : kind_(kind), deadline_(deadline)
        {
        }

        MoEOverlayActivationRendezvousKind kind_;
        TimePoint deadline_;
    };
} // namespace llaminar2
