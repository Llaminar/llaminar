/**
 * @file CPUShortConvolution.cpp
 * @brief CPU implementation of causal depthwise conv1d + SiLU
 *
 * Two execution paths:
 *   Prefill (seq_len > 1): Full causal conv1d with zero-padding, stores tail in conv_state
 *   Decode  (seq_len == 1): Conv1d update using conv_state history
 *
 * Reference: torch_causal_conv1d_update() and F.conv1d() in HuggingFace transformers
 */

#include "CPUShortConvolution.h"
#include "../../../utils/OpenMPUtils.h"

#include <cmath>
#include <cstring>

namespace llaminar2
{

    bool CPUShortConvolution::forward(
        const float *input, const float *weight, const float *bias,
        float *output, float *conv_state,
        int seq_len, int channels, int kernel_size,
        bool apply_silu)
    {
        if (seq_len == 1)
        {
            return executeDecode(input, weight, bias, output, conv_state,
                                 channels, kernel_size, apply_silu);
        }
        else
        {
            return executePrefill(input, weight, bias, output, conv_state,
                                  seq_len, channels, kernel_size, apply_silu);
        }
    }

    bool CPUShortConvolution::executePrefill(
        const float *input, const float *weight, const float *bias,
        float *output, float *conv_state,
        int seq_len, int channels, int kernel_size,
        bool apply_silu)
    {
        // Causal depthwise conv1d + optional SiLU for full sequence
        //
        // Input layout: [seq_len, channels] (row-major, channels is inner dim)
        // Weight layout: [channels, kernel_size]
        // Conv state: [channels, kernel_size - 1]
        //
        // For in-place operation (input == output), process time in reverse order
        // so that each output position only reads from not-yet-overwritten inputs.
        // Causal conv reads positions [t - state_len, ..., t], so writing from
        // t=seq_len-1 down to t=0 ensures all reads see original input values.

        const int state_len = kernel_size - 1;
        const bool in_place = (input == output);

        auto do_work = [&]()
        {
#pragma omp for schedule(static)
            for (int c = 0; c < channels; ++c)
            {
                const float *w = weight + c * kernel_size;
                const float b = bias ? bias[c] : 0.0f;

                // Store tail of input into conv_state BEFORE overwriting (in-place)
                if (conv_state)
                {
                    for (int s = 0; s < state_len; ++s)
                    {
                        const int src_t = seq_len - state_len + s;
                        if (src_t >= 0)
                            conv_state[c * state_len + s] = input[src_t * channels + c];
                        else
                            conv_state[c * state_len + s] = 0.0f;
                    }
                }

                // Process in reverse time order for in-place safety
                if (in_place)
                {
                    for (int t = seq_len - 1; t >= 0; --t)
                    {
                        float sum = b;
                        for (int k = 0; k < kernel_size; ++k)
                        {
                            const int input_t = t - state_len + k;
                            if (input_t >= 0)
                            {
                                sum += w[k] * input[input_t * channels + c];
                            }
                        }
                        if (apply_silu)
                        {
                            const float sig = 1.0f / (1.0f + std::exp(-sum));
                            output[t * channels + c] = sum * sig;
                        }
                        else
                        {
                            output[t * channels + c] = sum;
                        }
                    }
                }
                else
                {
                    for (int t = 0; t < seq_len; ++t)
                    {
                        float sum = b;
                        for (int k = 0; k < kernel_size; ++k)
                        {
                            const int input_t = t - state_len + k;
                            if (input_t >= 0)
                            {
                                sum += w[k] * input[input_t * channels + c];
                            }
                        }
                        if (apply_silu)
                        {
                            const float sig = 1.0f / (1.0f + std::exp(-sum));
                            output[t * channels + c] = sum * sig;
                        }
                        else
                        {
                            output[t * channels + c] = sum;
                        }
                    }
                }
            }
        };
        OMP_WORKSHARE_REGION(do_work);

        return true;
    }

    bool CPUShortConvolution::executeDecode(
        const float *input, const float *weight, const float *bias,
        float *output, float *conv_state,
        int channels, int kernel_size,
        bool apply_silu)
    {
        if (!conv_state)
            return false;

        const int state_len = kernel_size - 1;

        auto do_work = [&]()
        {
#pragma omp for schedule(static)
            for (int c = 0; c < channels; ++c)
            {
                const float *w = weight + c * kernel_size;
                float *state = conv_state + c * state_len;
                const float b = bias ? bias[c] : 0.0f;

                // Compute dot product: [state..., new_input] dot weight
                float sum = b;
                for (int k = 0; k < state_len; ++k)
                {
                    sum += w[k] * state[k];
                }
                sum += w[state_len] * input[c];

                // Shift state left and append new input
                for (int k = 0; k < state_len - 1; ++k)
                {
                    state[k] = state[k + 1];
                }
                if (state_len > 0)
                {
                    state[state_len - 1] = input[c];
                }

                if (apply_silu)
                {
                    const float sig = 1.0f / (1.0f + std::exp(-sum));
                    output[c] = sum * sig;
                }
                else
                {
                    output[c] = sum;
                }
            }
        };
        OMP_WORKSHARE_REGION(do_work);

        return true;
    }

} // namespace llaminar2
