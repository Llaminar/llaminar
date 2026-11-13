/**
 * @file CapturedTransferEndpointScheduling.h
 * @brief Capture-frozen endpoint launch scheduling, independent of model phase.
 *
 * Both modes execute the original complete channel protocol and communicate
 * exactly its device-published live extent. Only native launch grouping differs.
 * No unknown/automatic mode can select a scheduling policy during replay.
 */
#pragma once
#include <cstdint>
#include <stdexcept>
#include <string_view>

namespace llaminar2
{
    /** @brief Explicit launch grouping for one immutable complete exchange. */
    enum class CapturedTransferEndpointScheduling : std::uint32_t
    {
        Individual = 1, ///< Original ordered endpoint launches.
        Batched = 2, ///< Bounded by-value packets of independent endpoints.
    };

    /** @return Stable capture identity, rejecting undeclared enum values.
     * @param scheduling Explicit frozen launch grouping.
     * @throws std::invalid_argument For a value without a first-class implementation. */
    [[nodiscard]] constexpr std::string_view capturedTransferEndpointSchedulingName(
        CapturedTransferEndpointScheduling scheduling)
    {
        switch (scheduling)
        {
        case CapturedTransferEndpointScheduling::Individual: return "individual";
        case CapturedTransferEndpointScheduling::Batched: return "batched";
        }
        throw std::invalid_argument("Captured exchange requires explicit endpoint scheduling");
    }

    /** @return Semantic message key bound to the exact immutable scheduling.
     * @param semantic_key Caller-authored nonzero graph/message identity.
     * @param scheduling Explicit capture policy, shared by both channel endpoints.
     * @throws std::invalid_argument For a missing semantic key or invalid policy.
     *
     * Hash only immutable metadata, never pointers or live protocol values.
     * This discriminator prevents differently recorded graph families from
     * accidentally accepting one another's otherwise equal semantic message.
     */
    [[nodiscard]] constexpr std::uint64_t capturedTransferEndpointSchedulingKey(
        std::uint64_t semantic_key, CapturedTransferEndpointScheduling scheduling)
    {
        if (!semantic_key) throw std::invalid_argument("Captured exchange needs a semantic message key");
        const auto name = capturedTransferEndpointSchedulingName(scheduling);
        std::uint64_t hash = 14695981039346656037ull;
        for (unsigned byte = 0; byte < 8; ++byte)
        {
            hash ^= (semantic_key >> (byte * 8)) & 255u;
            hash *= 1099511628211ull;
        }
        for (const char c : name) { hash ^= static_cast<unsigned char>(c); hash *= 1099511628211ull; }
        if (!hash) throw std::invalid_argument("Captured scheduling message key cannot be zero");
        return hash;
    }
}
