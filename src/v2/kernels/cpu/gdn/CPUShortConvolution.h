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
        bool supportsRequestLiveStateBank(int request_count, int state_size) const override
        {
            return request_count > 0 && state_size > 0;
        }
        void resetGPUState() override;
        void bindVerifierStateCaptureWorkspace(float *workspace, int rows, int state_size) override;
        void bindSpeculativeStateWorkspace(float *workspace, int state_size) override;
        bool restoreVerifierStateCaptureRow(float *dst_state, int row, void *stream) override;
        bool restoreVerifierStateCaptureRows(
            float *dst_state,
            const int *host_row_indices,
            int request_count,
            void *stream) override;

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

        /**
         * @brief Execute unequal CPU requests in one channel-block workshare.
         *
         * Every request owns a persistent convolution-state slot.  Channels
         * are independent, so one OpenMP region distributes
         * `(request, SIMD-channel-block)` pairs while each pair walks its real
         * rows in serial decode order.
         */
        bool forwardBatchedRequestsWithHostSeqLens(
            const float *input, const float *weight, const float *bias,
            float *output, float *conv_state,
            int seq_len, int request_count, int request_seq_len,
            int channels, int kernel_size,
            const int *host_request_seq_lens,
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
        bool ensureRequestStateBank(
            int request_count,
            int state_floats,
            const float *request_zero_state);

        float *verifier_state_capture_ = nullptr;
        int verifier_state_capture_rows_ = 0;
        int verifier_state_capture_size_ = 0;
        float *speculative_state_work_ = nullptr;
        int speculative_state_work_size_ = 0;
        std::vector<float> owned_speculative_state_work_;
        std::vector<float> request_state_bank_;
        int request_state_size_ = 0;
        int request_state_capacity_ = 0;
        std::vector<float> request_input_copy_;
    };

} // namespace llaminar2
