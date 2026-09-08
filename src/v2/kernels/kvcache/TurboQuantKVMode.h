/**
 * @file TurboQuantKVMode.h
 * @brief Shared, capture-time compressed key/value storage policy.
 *
 * Keys use the quadratic-companded AQ8 format because key error perturbs every
 * attention score. Values retain the public Q8_1, TQ4, or TQ8 storage budget.
 * The policy is immutable for the lifetime of a cache so GPU graph capture
 * resolves concrete quantize/dequantize kernels once and never consults host
 * state during replay.
 */

#pragma once

#include "../../execution/config/RuntimeConfig.h"

#include <cstdint>
#include <stdexcept>

namespace llaminar2
{
    /**
     * @brief Physical compressed formats used by one KV cache instance.
     */
    enum class TurboQuantKVMode : uint8_t
    {
        AQ8_K_Q8_1_V, ///< Block-linear Q8_1 values without rotation.
        AQ8_K_TQ4_V, ///< Capacity-oriented asymmetric cache.
        AQ8_K_TQ8_V  ///< Higher-fidelity value cache.
    };

    /**
     * @brief Basis construction used when the first AQ8 key rows are appended.
     *
     * Ordinary prefill averages its retained rows to minimize residual error.
     * Decode-equivalent grouped publication uses the first retained row because
     * serial decode establishes its immutable basis from that same row.
     */
    enum class AttentionKeyAnchorPolicy : uint8_t
    {
        MeanRetained,
        FirstRetained,
    };

    /**
     * @brief Convert a public quantized-cache selector into a physical policy.
     *
     * The selector names the value-storage budget; AQ8 is the invariant key
     * codec for every returned policy.
     *
     * @throws std::invalid_argument for a non-quantized-cache precision.
     */
    inline TurboQuantKVMode turboQuantKVModeFromPrecision(ActivationPrecision precision)
    {
        switch (precision)
        {
        case ActivationPrecision::Q8_1:
            return TurboQuantKVMode::AQ8_K_Q8_1_V;
        case ActivationPrecision::TQ4:
            return TurboQuantKVMode::AQ8_K_TQ4_V;
        case ActivationPrecision::TQ8:
            return TurboQuantKVMode::AQ8_K_TQ8_V;
        default:
            throw std::invalid_argument(
                "turboQuantKVModeFromPrecision requires Q8_1, TQ4, or TQ8");
        }
    }

    /** @brief Value format selected by a physical TurboQuant cache policy. */
    constexpr ActivationPrecision turboQuantValuePrecision(TurboQuantKVMode mode)
    {
        switch (mode)
        {
        case TurboQuantKVMode::AQ8_K_Q8_1_V:
            return ActivationPrecision::Q8_1;
        case TurboQuantKVMode::AQ8_K_TQ8_V:
            return ActivationPrecision::TQ8;
        case TurboQuantKVMode::AQ8_K_TQ4_V:
            return ActivationPrecision::TQ4;
        }
        return ActivationPrecision::Q8_1;
    }

    /** @brief Whether the selected value codec needs codebooks and rotations. */
    constexpr bool turboQuantValueUsesRotation(TurboQuantKVMode mode)
    {
        return mode != TurboQuantKVMode::AQ8_K_Q8_1_V;
    }

    /** @brief Stable diagnostic spelling for PerfStats and error messages. */
    constexpr const char *turboQuantKVModeName(TurboQuantKVMode mode)
    {
        switch (mode)
        {
        case TurboQuantKVMode::AQ8_K_Q8_1_V:
            return "AQ8-K/Q8_1-V";
        case TurboQuantKVMode::AQ8_K_TQ8_V:
            return "AQ8-K/TQ8-V";
        case TurboQuantKVMode::AQ8_K_TQ4_V:
            return "AQ8-K/TQ4-V";
        }
        return "unknown-compressed-kv-mode";
    }
} // namespace llaminar2
