/**
 * @file TransferCommandProgressWatch.h
 * @brief Command-local, observational timing for asynchronous transfer diagnostics.
 *
 * A continuously busy queue may contain many healthy short commands. Its age
 * says nothing about one command's latency. Each permanent transfer slot starts
 * a fresh watch when publishing a generation. This watch only throttles logs;
 * it never decides completion, admission, cancellation, or inference ordering.
 */
#pragma once

#include <cstdint>
#include <limits>
#include <optional>

namespace llaminar2
{
    /** @brief Observes one immutable command generation under its owner's lock. */
    class TransferCommandProgressWatch final
    {
    public:
        /**
         * @brief Start a new generation, discarding the previous command's age.
         * @param now_ns Monotonic publication time in nanoseconds; zero is valid.
         */
        void published(std::uint64_t now_ns) noexcept
        {
            publication_ns_ = now_ns;
            last_warning_ns_ = now_ns;
            incomplete_event_queries_ = 0u;
            last_incomplete_event_query_ns_ = now_ns;
        }

        /**
         * @brief Observe a successful native event query that returned not ready.
         * @param now_ns Monotonic time of the worker's bounded progress pass.
         *
         * This is diagnostic history, not a completion decision. Counting exact
         * native observations distinguishes an executing/queued GPU operation
         * from a completion that the host has not serviced recently. A warning
         * must never issue another GPU query merely to collect this evidence.
         */
        void observedIncompleteEvent(std::uint64_t now_ns) noexcept
        {
            if (now_ns < publication_ns_ ||
                now_ns < last_incomplete_event_query_ns_)
                return;
            if (incomplete_event_queries_ !=
                std::numeric_limits<std::uint64_t>::max())
                ++incomplete_event_queries_;
            last_incomplete_event_query_ns_ = now_ns;
        }

        /** @return Native not-ready observations for this generation only. */
        [[nodiscard]] std::uint64_t incompleteEventQueries() const noexcept
        {
            return incomplete_event_queries_;
        }

        /**
         * @brief Report how recently the exact event was observed incomplete.
         * @param now_ns Current monotonic time in nanoseconds.
         * @return Age, or no value if never queried or the clock regressed.
         */
        [[nodiscard]] std::optional<std::uint64_t> incompleteEventQueryAge(
            std::uint64_t now_ns) const noexcept
        {
            if (incomplete_event_queries_ == 0u ||
                now_ns < last_incomplete_event_query_ns_)
                return std::nullopt;
            return now_ns - last_incomplete_event_query_ns_;
        }

        /**
         * @brief Consume a throttled diagnostic for this still-pending command.
         * @param now_ns Current monotonic time in nanoseconds.
         * @return Exact command age when overdue, otherwise no diagnostic.
         *
         * The caller's typed slot lifecycle proves that the command is pending.
         * Subtraction after monotonicity checks avoids timestamp-add overflow.
         */
        [[nodiscard]] std::optional<std::uint64_t> pending(
            std::uint64_t now_ns) noexcept
        {
            constexpr std::uint64_t interval_ns = 5'000'000'000ull;
            if (now_ns < publication_ns_ || now_ns < last_warning_ns_ ||
                now_ns - last_warning_ns_ < interval_ns)
                return std::nullopt;
            last_warning_ns_ = now_ns;
            return now_ns - publication_ns_;
        }

    private:
        std::uint64_t publication_ns_ = 0u; ///< Origin of this command only.
        std::uint64_t last_warning_ns_ = 0u; ///< Per-command log throttle.
        std::uint64_t incomplete_event_queries_ = 0u; ///< Native observations only.
        std::uint64_t last_incomplete_event_query_ns_ = 0u; ///< Last worker poll.
    };
} // namespace llaminar2
