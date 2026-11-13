/**
 * @file CPUAdjustedGDNReadout.h
 * @brief One canonical backward-query update shared by serial and grouped CPU GDN.
 *
 * A readout acts on the undecayed parent, using qhat = decay * (q - eta*k)
 * and eta = beta * dot(k,q). Every dot visits increasing key indices and uses
 * explicit FP32 FMA. This deliberately changes the former updated-state
 * readout's rounding; it is not byte-compatible with that older definition.
 * Fixed-epoch execution applies this primitive newest-to-oldest through its raw
 * history. Checkpoint advancement retains rounded decay/key/correction/update.
 * The complete checkpoint/history state belongs to the CPU cache authority.
 */
#pragma once

#include "kernels/CPUGDNStateLayout.h"
#include <cmath>
#include <string_view>

namespace llaminar2
{
    /** @brief Persisted-state generation for this exact CPU readout definition. */
    inline constexpr std::string_view kCPUGDNReadoutArithmetic = CPUGDNStateLayout::kArithmetic;

    /**
     * @brief Prepare the sole readout definition for a normalized/scaled row.
     * @param query Query after the canonical normalization and head scaling.
     * @param key Canonically normalized key, with key_width live elements.
     * @param decay Rounded decay gate used by state advancement.
     * @param beta Rounded sigmoid write gate used by state advancement.
     * @param key_width Positive key dimension, validated by the owning kernel.
     * @param adjusted Destination with key_width elements; may alias query.
     * @return eta, used by the final FMA with the original value vector.
     *
     * Computing the dot before writing any query element makes in-place
     * preparation safe. No reduction tree or SIMD horizontal sum is permitted:
     * ordinary serial and grouped invocations must round identically.
     */
    inline float prepareAdjustedGDNReadout(
        const float *query, const float *key, float decay, float beta,
        int key_width, float *adjusted)
    {
        float dot = 0.0f;
        for (int j = 0; j < key_width; ++j)
            dot = std::fma(key[j], query[j], dot);
        const float eta = beta * dot;
        for (int j = 0; j < key_width; ++j)
            adjusted[j] = decay * std::fma(-eta, key[j], query[j]);
        return eta;
    }
}
