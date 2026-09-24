/**
 * @file MoEOverlayWorkerDrainProtocol.h
 * @brief Monotonic shutdown handshake for asynchronous ExpertOverlay workers.
 *
 * A reusable Boolean cannot distinguish an old idle acknowledgement from the
 * drain requested after a worker has selected new work. This protocol gives
 * each drain request a generation. The owner waits for that exact generation,
 * while the sole worker acknowledges only the generation it observed after it
 * has finished any already-admitted transaction.
 */

#pragma once

#include <atomic>
#include <compare>
#include <cstdint>
#include <exception>
#include <limits>

namespace llaminar2
{
    /**
     * @brief One owner/one worker monotonic drain request protocol.
     *
     * The owner may request multiple generations, although the production
     * service currently requests exactly one during destruction. A stale
     * acknowledgement can satisfy only its own or an older generation; it can
     * never make a later request appear complete.
     */
    class MoEOverlayWorkerDrainProtocol
    {
    public:
        /** Typed identity returned to the owner for one drain request. */
        struct Request
        {
            std::uint64_t generation = 0u;

            /** @return Whether this request names a real drain generation. */
            [[nodiscard]] constexpr bool valid() const noexcept
            {
                return generation != 0u;
            }

            constexpr auto operator<=>(const Request &) const = default;
        };

        /**
         * @brief Publish a new drain generation.
         *
         * This release edge closes admission before the owner starts waiting.
         * Overflow is process-fatal because reusing generation zero or an old
         * generation would make shutdown ordering unprovable.
         */
        [[nodiscard]] Request request() noexcept
        {
            const std::uint64_t prior = requested_.fetch_add(
                1u, std::memory_order_acq_rel);
            if (prior == std::numeric_limits<std::uint64_t>::max())
                std::terminate();
            return Request{prior + 1u};
        }

        /** @return Whether admission has been permanently closed. */
        [[nodiscard]] bool shutdownRequested() const noexcept
        {
            return requested_.load(std::memory_order_acquire) != 0u;
        }

        /** @return The newest owner request visible to the worker. */
        [[nodiscard]] Request currentRequest() const noexcept
        {
            return Request{requested_.load(std::memory_order_acquire)};
        }

        /** @return Whether some published request still needs acknowledgement. */
        [[nodiscard]] bool acknowledgementPending() const noexcept
        {
            return acknowledged_.load(std::memory_order_acquire) <
                requested_.load(std::memory_order_acquire);
        }

        /**
         * @brief Release-publish quiescence for an observed request.
         *
         * The sole worker calls this only after it has either completed the
         * already-admitted transaction or proved that none was admitted.
         */
        void acknowledge(Request observed) noexcept
        {
            if (!observed.valid())
                return;
            std::uint64_t acknowledged = acknowledged_.load(
                std::memory_order_relaxed);
            while (acknowledged < observed.generation &&
                   !acknowledged_.compare_exchange_weak(
                       acknowledged,
                       observed.generation,
                       std::memory_order_release,
                       std::memory_order_relaxed))
            {
                // compare_exchange refreshes acknowledged before retrying.
            }
        }

        /** @return Whether the exact owner request has reached quiescence. */
        [[nodiscard]] bool acknowledged(Request request) const noexcept
        {
            return request.valid() &&
                acknowledged_.load(std::memory_order_acquire) >=
                    request.generation;
        }

    private:
        /** Highest release-published owner request generation. */
        std::atomic<std::uint64_t> requested_{0u};
        /** Highest worker generation proven quiescent. */
        std::atomic<std::uint64_t> acknowledged_{0u};
    };
} // namespace llaminar2
