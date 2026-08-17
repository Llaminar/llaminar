/**
 * @file ExpertTierTransferMeasurement.h
 * @brief Exact, allocation-free timing evidence for one projection movement.
 *
 * ExpertOverlay transfer lanes execute on persistent maintenance streams while
 * inference continues against an immutable residency epoch.  This value type
 * carries the completed lane's observable wall time together with device and
 * host work components.  It contains no pointers, clocks, or backend handles,
 * so setup certification and tests can retain it after a lane is reused.
 */

#pragma once

#include <cstdint>
#include <limits>

namespace llaminar2
{
    /**
     * @brief Timing and byte evidence for one completed projection transfer.
     *
     * `wall_nanoseconds` is the end-to-end lane latency observed by the real
     * event-polled protocol. `device_nanoseconds` is the sum of timing-event
     * intervals containing only submitted kernels and DMA. `host_nanoseconds`
     * covers bounded CPU copies or relay work. `transport_nanoseconds` covers
     * an authenticated network or inter-process data plane. Components may
     * overlap and therefore must not be added to replace the wall measurement.
     */
    struct ExpertTierProjectionTransferMeasurement
    {
        /** Monotonic completed-transfer sequence owned by one lane. */
        std::uint64_t sequence = 0;
        /** Exact descriptor or CPU-native bytes moved by the lane. */
        std::uint64_t bytes = 0;
        /** Observed start-to-ready latency, including real protocol progress. */
        std::uint64_t wall_nanoseconds = 0;
        /** Sum of timing-event GPU work intervals, excluding host poll gaps. */
        std::uint64_t device_nanoseconds = 0;
        /** Sum of bounded host-copy intervals performed by maintenance. */
        std::uint64_t host_nanoseconds = 0;
        /** End-to-end authenticated network/data-plane protocol interval. */
        std::uint64_t transport_nanoseconds = 0;

        /**
         * @return Whether this is a complete, positive observation.
         *
         * A GPU transfer must record positive device work.  CPU-only movement
         * may use the same type with zero device work, so validity is expressed
         * in terms of completion identity, bytes, and end-to-end latency.
         */
        [[nodiscard]] constexpr bool valid() const noexcept
        {
            return sequence != 0 && bytes != 0 && wall_nanoseconds != 0;
        }

        /** @brief Compare every retained measurement field. */
        bool operator==(
            const ExpertTierProjectionTransferMeasurement &) const = default;
    };

    /**
     * @brief Saturating addition used by noexcept telemetry accumulation.
     * @param current Existing non-negative nanosecond or byte total.
     * @param increment New non-negative contribution.
     * @return Exact sum when representable, otherwise UINT64_MAX.
     */
    [[nodiscard]] constexpr std::uint64_t
    saturatingExpertTierMeasurementAdd(
        std::uint64_t current,
        std::uint64_t increment) noexcept
    {
        return increment > std::numeric_limits<std::uint64_t>::max() - current
                   ? std::numeric_limits<std::uint64_t>::max()
                   : current + increment;
    }
} // namespace llaminar2
