/**
 * @file CPUShortConvolution.h
 * @brief CPU implementation of ITensorShortConvolution
 *
 * Causal depthwise conv1d + SiLU for GDN QKV preprocessing.
 * OpenMP-parallelized across channels.
 */

#pragma once

#include "../../../tensors/TensorKernels.h"

namespace llaminar2
{

    class CPUShortConvolution : public ITensorShortConvolution
    {
    public:
        bool supportsPaddedPrefillRealLength() const override { return true; }

        bool forward(
            const float *input, const float *weight, const float *bias,
            float *output, float *conv_state,
            int seq_len, int channels, int kernel_size,
            bool apply_silu = true) override;

    private:
        bool executePrefill(
            const float *input, const float *weight, const float *bias,
            float *output, float *conv_state,
            int seq_len, int channels, int kernel_size,
            bool apply_silu);

        bool executeDecode(
            const float *input, const float *weight, const float *bias,
            float *output, float *conv_state,
            int channels, int kernel_size,
            bool apply_silu);
    };

} // namespace llaminar2
