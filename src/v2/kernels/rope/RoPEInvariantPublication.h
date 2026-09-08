/**
 * @file RoPEInvariantPublication.h
 * @brief Immutable device publication contract for RoPE frequency tables.
 *
 * RoPE inverse frequencies depend only on the rotary width and theta. They are
 * therefore model/graph-family state, not request state. GPU backends publish
 * each exact table into a fixed workspace slot, record a readiness event, and
 * let every graph-local RoPE kernel adopt that same immutable publication.
 * This prevents separately captured graphs from silently overwriting a shared
 * table address while retaining the zero-allocation, device-resident hot path.
 */

#pragma once

#include <cstddef>
#include <cstdint>

namespace llaminar2::rope
{
    /** Number of exact `(rotary_dim, theta)` tables retained per workspace. */
    inline constexpr std::size_t kInvariantPublicationSlots = 16;

    /** Maximum number of FP32 frequencies in one supported RoPE table. */
    inline constexpr int kMaxInverseFrequencyValues = 128;

    /**
     * @brief Shared ownership record for one immutable device frequency table.
     *
     * `ready_event` is backend-owned (`cudaEvent_t` or `hipEvent_t`) and is
     * destroyed by the backend-specific shared-pointer deleter. Keeping the
     * event opaque here lets CUDA and ROCm implement the same lifecycle without
     * introducing either runtime header into the common kernel interface.
     */
    struct RoPEInvariantWorkspacePublication
    {
        void *ready_event = nullptr;             ///< One-time producer completion event.
        float *device_values = nullptr;          ///< Graph-stable inverse-frequency slot.
        int rotary_dim = 0;                      ///< Exact full rotary width represented.
        std::uint32_t theta_bits = 0;             ///< Bit-exact FP32 theta identity.
        std::size_t workspace_slot = 0;           ///< Fixed-stride workspace slot index.
    };
} // namespace llaminar2::rope
