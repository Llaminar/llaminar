/**
 * @file CPUShortConvolution.h
 * @brief CPU implementation of ITensorShortConvolution
 *
 * Causal depthwise conv1d + SiLU for GDN QKV preprocessing.
 * OpenMP-parallelized across channels.
 */

#pragma once

#include "../../../tensors/TensorKernels.h"

#include <vector>

namespace llaminar2
{

    class CPUShortConvolution : public ITensorShortConvolution
    {
    public:
        bool supportsPaddedPrefillRealLength() const override { return true; }
        void bindVerifierStateCaptureWorkspace(float *workspace, int rows, int state_size) override;
        void bindSpeculativeStateWorkspace(float *workspace, int state_size) override;
        bool restoreVerifierStateCaptureRow(float *dst_state, int row, void *stream) override;

        bool forward(
            const float *input, const float *weight, const float *bias,
            float *output, float *conv_state,
            int seq_len, int channels, int kernel_size,
            bool apply_silu = true) override;

        bool forwardWithStateSnapshots(
            const float *input, const float *weight, const float *bias,
            float *output, float *conv_state,
            int seq_len, int channels, int kernel_size,
            float *state_snapshots, int snapshot_stride_floats,
            int max_snapshot_rows,
            bool apply_silu = true) override;

        bool restoreStateFromSnapshot(
            float *state, const float *state_snapshots,
            int snapshot_row, int snapshot_stride_floats,
            int state_floats, void *stream = nullptr) override;

    private:
        bool executePrefillPreservingInPlaceTail(
            const float *input, const float *weight, const float *bias,
            float *output, float *conv_state,
            int seq_len, int channels, int kernel_size,
            bool apply_silu);

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

        float *prepareSpeculativeState(float *live_state, int state_floats);

        float *verifier_state_capture_ = nullptr;
        int verifier_state_capture_rows_ = 0;
        int verifier_state_capture_size_ = 0;
        float *speculative_state_work_ = nullptr;
        int speculative_state_work_size_ = 0;
        std::vector<float> owned_speculative_state_work_;
    };

} // namespace llaminar2
