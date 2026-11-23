/**
 * @file GDNInPlaceShortConv.h
 * @brief Direct publication for aliased short four-tap convolution groups.
 *
 * The small-group owner keeps the three raw history words for its channel in
 * registers. The oldest-to-newest multiply/add order matches the general
 * convolution. Holding raw words across the output store lets the same
 * thread retire its channel's original state even when it is updated in place.
 */
#pragma once

#include <cmath>
#include <cstddef>

namespace llaminar2::gdn
{
    /**
     * @brief Admit the short grouped geometry to the register-history leaf.
     * @param rows Physical rows in one request.
     * @param taps Number of convolution coefficients per channel.
     * @return Whether the existing small-group dispatch can keep raw history in registers.
     */
    [[nodiscard]] constexpr bool usesInPlaceShortConvHistory(int rows, int taps) noexcept
    {
        return taps == 4 && rows > 1 && rows <= 4;
    }

#if defined(__CUDACC__) || defined(__HIPCC__)
    /**
     * @brief Publish an admitted aliased four-tap convolution and retire its raw history.
     * @param input_output Row-major raw source and final result bank.
     * @param weight Four ordered convolution coefficients for every channel.
     * @param bias Optional channel bias.
     * @param initial Optional original three-word state per channel.
     * @param updated Optional final state, which may alias the original state.
     * @param snapshots Optional complete storage for each requested intermediate raw state.
     * @param snapshot_stride Distance between consecutive snapshots in floats.
     * @param snapshots_rows Number of requested snapshots, possibly less than live.
     * @param rows Actual physical rows for the short group.
     * @param live Effective original prefix, checked by the caller.
     * @param channels Channel count and physical row stride.
     * @param channel Exclusive channel owned by this thread.
     * @param silu Whether the convolution also publishes the SiLU result.
     */
    __device__ inline void publishInPlaceShortConvChannel(
        float *input_output, const float *weight, const float *bias,
        const float *initial, float *updated, float *snapshots,
        int snapshot_stride, int snapshots_rows, int rows, int live,
        int channels, int channel, bool silu)
    {
        const auto state_offset = static_cast<std::size_t>(channel) * 3;
        const auto *coefficients = weight + static_cast<std::size_t>(channel) * 4;
        const float w0 = coefficients[0], w1 = coefficients[1], w2 = coefficients[2], w3 = coefficients[3];
        const float channel_bias = bias ? bias[channel] : 0.0f;
        float h0 = initial ? initial[state_offset] : 0.0f;
        float h1 = initial ? initial[state_offset + 1] : 0.0f;
        float h2 = initial ? initial[state_offset + 2] : 0.0f;
        for (int row = 0; row < live; ++row)
        {
            auto &destination = input_output[static_cast<std::size_t>(row) * channels + channel];
            const float raw = destination;
            float sum = 0.0f;
            sum += h0 * w0;
            sum += h1 * w1;
            sum += h2 * w2;
            sum += raw * w3;
            if (bias)
                sum += channel_bias;
            if (silu)
                sum = sum / (1.0f + expf(-sum));
            destination = sum;

            h0 = h1;
            h1 = h2;
            h2 = raw;
            if (snapshots && row < snapshots_rows)
            {
                auto *snapshot = snapshots + static_cast<std::size_t>(row) * snapshot_stride + state_offset;
                snapshot[0] = h0;
                snapshot[1] = h1;
                snapshot[2] = h2;
            }
        }
        for (int row = live; row < rows; ++row)
            input_output[static_cast<std::size_t>(row) * channels + channel] = 0.0f;
        if (updated)
        {
            updated[state_offset] = h0;
            updated[state_offset + 1] = h1;
            updated[state_offset + 2] = h2;
        }
    }
#endif
}
