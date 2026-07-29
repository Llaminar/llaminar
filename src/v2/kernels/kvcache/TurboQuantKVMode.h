/**
 * @file TurboQuantKVMode.h
 * @brief Shared, capture-time TurboQuant key/value storage policy.
 *
 * TurboQuant always stores keys with the high-fidelity TQ8 codebook because
 * key error perturbs every attention score. Values may use either TQ4 for
 * maximum capacity or TQ8 for a symmetric, higher-fidelity cache. The policy is
 * immutable for the lifetime of a cache so GPU graph capture resolves concrete
 * quantize/dequantize kernels once and never consults host state during replay.
 */

#pragma once

#include "../../execution/config/RuntimeConfig.h"

#include <cstdint>
#include <stdexcept>

namespace llaminar2
{
    /**
     * @brief Physical TurboQuant formats used by one KV cache instance.
     */
    enum class TurboQuantKVMode : uint8_t
    {
        TQ8_K_TQ4_V, ///< Capacity-oriented asymmetric cache.
        TQ8_K_TQ8_V  ///< Higher-fidelity symmetric cache.
    };

    /**
     * @brief Convert the public cache precision selector into a physical policy.
     *
     * `TQ4` names the value-storage budget and therefore selects TQ8-K/TQ4-V.
     * `TQ8` selects TQ8 for both keys and values.
     *
     * @throws std::invalid_argument for a non-TurboQuant precision.
     */
    inline TurboQuantKVMode turboQuantKVModeFromPrecision(ActivationPrecision precision)
    {
        switch (precision)
        {
        case ActivationPrecision::TQ4:
            return TurboQuantKVMode::TQ8_K_TQ4_V;
        case ActivationPrecision::TQ8:
            return TurboQuantKVMode::TQ8_K_TQ8_V;
        default:
            throw std::invalid_argument(
                "turboQuantKVModeFromPrecision requires TQ4 or TQ8");
        }
    }

    /** @brief Value format selected by a physical TurboQuant cache policy. */
    constexpr ActivationPrecision turboQuantValuePrecision(TurboQuantKVMode mode)
    {
        return mode == TurboQuantKVMode::TQ8_K_TQ8_V
                   ? ActivationPrecision::TQ8
                   : ActivationPrecision::TQ4;
    }

    /** @brief Stable diagnostic spelling for PerfStats and error messages. */
    constexpr const char *turboQuantKVModeName(TurboQuantKVMode mode)
    {
        return mode == TurboQuantKVMode::TQ8_K_TQ8_V
                   ? "TQ8-K/TQ8-V"
                   : "TQ8-K/TQ4-V";
    }
} // namespace llaminar2
