/**
 * @file NativeReduceScatterContract.h
 * @brief Device-free range contract for exact-stream native sum/reduce-scatter.
 *
 * LocalTP admission and native enqueue share these checks. Counts name the
 * receive slice, while the input contains one such slice per participant in
 * communicator order. This is buffer geometry, not allocation accounting:
 * PhysicalMemoryAuthority and the caller's arena remain the storage owners.
 */
#pragma once

#include "ICollectiveBackend.h"
#include <cstdint>
#include <limits>

namespace llaminar2
{
    /**
     * @brief Prove representable, aligned, disjoint native device byte ranges.
     * @param send Full rank-major contribution; never dereferenced on the host.
     * @param receive Participant-local result; never dereferenced on the host.
     * @param count Result elements on each participant.
     * @param dtype Exact native element type; unknown enum values are rejected.
     * @param participants Native communicator degree, at least two.
     * @param participant Calling communicator coordinate, not a physical ordinal.
     * @param stream Exact non-null stream; the caller owns stream/device authentication.
     * @return True only for a complete out-of-place operation with safe byte extents.
     *
     * Native in-place scatter has a special offset alias contract. This public
     * operation deliberately requires disjoint banks so publication cannot
     * invalidate another rank slice before its consumer has finished.
     */
    inline bool nativeReduceScatterBuffersValid(
        const void *send, void *receive, std::size_t count, CollectiveDataType dtype,
        int participants, int participant, void *stream) noexcept
    {
        if (!send || !receive || !stream || count == 0 || participants < 2 ||
            participant < 0 || participant >= participants)
            return false;
        std::size_t bytes = 0;
        switch (dtype)
        {
        case CollectiveDataType::FLOAT32:
        case CollectiveDataType::INT32: bytes = 4; break;
        case CollectiveDataType::FLOAT16:
        case CollectiveDataType::BFLOAT16: bytes = 2; break;
        case CollectiveDataType::INT8: bytes = 1; break;
        default: return false;
        }
        const auto maximum = std::numeric_limits<std::uintptr_t>::max();
        if (count > maximum / bytes / static_cast<std::size_t>(participants))
            return false;
        const auto input = reinterpret_cast<std::uintptr_t>(send);
        const auto output = reinterpret_cast<std::uintptr_t>(receive);
        const auto receive_bytes = count * bytes;
        const auto send_bytes = receive_bytes * static_cast<std::size_t>(participants);
        return input % bytes == 0 && output % bytes == 0 &&
            input <= maximum - send_bytes && output <= maximum - receive_bytes &&
            (input + send_bytes <= output || output + receive_bytes <= input);
    }
}
