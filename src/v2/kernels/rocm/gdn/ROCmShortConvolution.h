/**
 * @file ROCmShortConvolution.h
 * @brief ROCm/HIP implementation of ITensorShortConvolution
 *
 * Manages GPU-resident conv state internally.
 *
 * Device-pointer design: All input/output pointers passed to forward()
 * are expected to be DEVICE pointers (already on GPU). The stage
 * (ShortConv1dStage) handles coherence via ensureOnDevice() before
 * calling this method.
 */

#pragma once

#include "../../../tensors/TensorKernels.h"
#include "../../../backends/GPUDeviceContextPool.h"
#include "../../../utils/Logger.h"

extern "C"
{
    bool rocmGDN_short_conv1d(
        const float *input, const float *weight, const float *bias,
        float *output, float *conv_state,
        int seq_len, int channels, int kernel_size,
        bool apply_silu,
        float *state_snapshots,
        int snapshot_stride_floats,
        int max_snapshot_rows,
        int device_idx, void *stream);

    bool rocmGDN_short_conv1d_effective(
        const float *input, const float *weight, const float *bias,
        float *output, float *conv_state,
        int seq_len, int channels, int kernel_size,
        bool apply_silu,
        const int *device_effective_seq_len,
        float *state_snapshots,
        int snapshot_stride_floats,
        int max_snapshot_rows,
        int device_idx, void *stream);

    // GPU memory helpers (implemented in ROCmGatedDeltaNetKernels.hip)
    bool rocmGDN_gpu_malloc(float **ptr, size_t count);
    void rocmGDN_gpu_free(float *ptr);
    void rocmGDN_gpu_memset_zero(float *ptr, size_t count);
    void rocmGDN_gpu_memset_zero_async(float *ptr, size_t count, void *stream);
    void rocmGDN_gpu_memcpy(float *dst, const float *src, size_t count);
    void rocmGDN_gpu_memcpy_async(float *dst, const float *src, size_t count, void *stream);
    void rocmGDN_gpu_memcpy_d2h(float *host_dst, const float *device_src, size_t count);
    void rocmGDN_gpu_memcpy_d2h_async(float *host_dst, const float *device_src, size_t count, void *stream);
    void rocmGDN_gpu_set_device(int ordinal);
    void rocmGDN_stream_synchronize(void *stream);
    bool rocmGDN_restore_state_from_capture_metadata(
        float *dst_state,
        const float *state_snapshots,
        const int32_t *device_committed_state_rows,
        int request_index,
        int snapshot_stride_floats,
        int max_snapshot_rows,
        int state_floats,
        int device_idx,
        void *stream);
}

namespace llaminar2
{

    class ROCmShortConvolution : public ITensorShortConvolution
    {
    public:
        explicit ROCmShortConvolution(int device_ordinal)
            : device_ordinal_(device_ordinal) {}

        ~ROCmShortConvolution()
        {
            rocmGDN_gpu_set_device(device_ordinal_);
            rocmGDN_gpu_free(gpu_state_);
            rocmGDN_gpu_free(scratch_);
        }

        void allocateGPUState(int state_size) override { allocateState(state_size); }
        bool allocateGPUScratch(int scratch_size) override { return allocateScratch(scratch_size); }
        void resetGPUState() override { resetState(); }
        void bindVerifierStateCaptureWorkspace(float *workspace, int rows, int state_size) override
        {
            verifier_state_capture_ = workspace;
            verifier_state_capture_rows_ = rows;
            verifier_state_capture_size_ = state_size;
        }

        bool restoreVerifierStateCaptureRow(float *dst_state, int row, void *stream) override
        {
            (void)dst_state;
            if (!gpu_state_ || !verifier_state_capture_ ||
                row < 0 || row >= verifier_state_capture_rows_ ||
                verifier_state_capture_size_ != state_size_)
            {
                return false;
            }

            rocmGDN_gpu_set_device(device_ordinal_);
            const float *src =
                verifier_state_capture_ +
                static_cast<size_t>(row) * static_cast<size_t>(verifier_state_capture_size_);
            if (stream)
                rocmGDN_gpu_memcpy_async(gpu_state_, src, static_cast<size_t>(state_size_), stream);
            else
                rocmGDN_gpu_memcpy(gpu_state_, src, static_cast<size_t>(state_size_));
            return true;
        }

        bool restoreVerifierStateCaptureRowFromDeviceMetadata(
            float *dst_state,
            const int32_t *device_committed_state_rows,
            int request_index,
            void *stream) override
        {
            (void)dst_state;
            if (!gpu_state_ || !verifier_state_capture_ || !device_committed_state_rows ||
                request_index < 0 || !stream ||
                verifier_state_capture_rows_ <= 0 ||
                verifier_state_capture_size_ != state_size_)
            {
                return false;
            }

            rocmGDN_gpu_set_device(device_ordinal_);
            return rocmGDN_restore_state_from_capture_metadata(
                gpu_state_,
                verifier_state_capture_,
                device_committed_state_rows,
                request_index,
                verifier_state_capture_size_,
                verifier_state_capture_rows_,
                state_size_,
                device_ordinal_,
                stream);
        }

        bool supportsPaddedPrefillRealLength() const override { return true; }
        size_t stateBytes() const override
        {
            return state_size_ > 0 ? static_cast<size_t>(state_size_) * sizeof(float) : 0;
        }

        void allocateState(int state_size)
        {
            if (gpu_state_ && state_size_ == state_size)
                return;
            if (gpu_state_)
            {
                rocmGDN_gpu_set_device(device_ordinal_);
                rocmGDN_gpu_free(gpu_state_);
            }
            state_size_ = state_size;
            rocmGDN_gpu_set_device(device_ordinal_);
            if (!rocmGDN_gpu_malloc(&gpu_state_, state_size))
            {
                LOG_ERROR("[ROCmShortConvolution] GPU malloc failed for state");
                gpu_state_ = nullptr;
                return;
            }
            void *stream = GPUDeviceContextPool::instance().getAMDContext(device_ordinal_).defaultStream();
            rocmGDN_gpu_memset_zero_async(gpu_state_, state_size, stream);
            rocmGDN_stream_synchronize(stream);
            LOG_DEBUG("[ROCmShortConvolution] Allocated GPU state: " << state_size << " floats on device " << device_ordinal_);
        }

        void resetState()
        {
            rocmGDN_gpu_set_device(device_ordinal_);
            if (gpu_state_ && state_size_ > 0)
            {
                void *stream = GPUDeviceContextPool::instance().getAMDContext(device_ordinal_).defaultStream();
                rocmGDN_gpu_memset_zero_async(gpu_state_, state_size_, stream);
                rocmGDN_stream_synchronize(stream);
            }
            if (scratch_)
            {
                void *stream = GPUDeviceContextPool::instance().getAMDContext(device_ordinal_).defaultStream();
                rocmGDN_gpu_memset_zero_async(scratch_, scratch_size_, stream);
                rocmGDN_stream_synchronize(stream);
            }
        }

        bool forward(
            const float *input, const float *weight, const float *bias,
            float *output, float *conv_state,
            int seq_len, int channels, int kernel_size,
            bool apply_silu = true) override
        {
            rocmGDN_gpu_set_device(device_ordinal_);
            const int required_state_size = channels * (kernel_size - 1);
            if (!gpu_state_ || state_size_ != required_state_size)
                allocateState(required_state_size);
            if (!gpu_state_)
            {
                LOG_ERROR("[ROCmShortConvolution] Missing GPU convolution state");
                return false;
            }
            float *effective_state = gpu_state_;
            float *effective_output = output;

            // GDN QKV short-conv is commonly in-place. Prefill needs scratch
            // so one timestep cannot clobber another; decode also needs it so
            // the history update stores the raw projection, not the convolved
            // output row.
            const bool needs_scratch = (input == output);
            if (needs_scratch)
            {
                const int required_scratch_size = seq_len * channels;
                if (!scratchPointer() || scratchCapacity() < required_scratch_size)
                {
                    LOG_ERROR("[ROCmShortConvolution] In-place short-conv scratch was not preallocated: need "
                              << required_scratch_size << " floats, have " << scratchCapacity());
                    return false;
                }
                effective_output = scratchPointer();
            }

            // All pointers are device pointers — pass directly to HIP kernel.
            const bool ok = rocmGDN_short_conv1d(
                input, weight, bias, effective_output, effective_state,
                seq_len, channels, kernel_size, apply_silu,
                verifier_state_capture_,
                verifier_state_capture_size_,
                verifier_state_capture_rows_,
                device_ordinal_, stream_);
            if (!ok)
                return false;

            if (needs_scratch)
            {
                const size_t count = static_cast<size_t>(seq_len) * static_cast<size_t>(channels);
                rocmGDN_gpu_memcpy_async(output, scratchPointer(), count, stream_);
            }

            return true;
        }

        bool forwardWithEffectiveSeqLen(
            const float *input, const float *weight, const float *bias,
            float *output, float *conv_state,
            int seq_len, int channels, int kernel_size,
            const int *device_effective_seq_len,
            bool apply_silu = true) override
        {
            rocmGDN_gpu_set_device(device_ordinal_);
            const int required_state_size = channels * (kernel_size - 1);
            if (!gpu_state_ || state_size_ != required_state_size)
                allocateState(required_state_size);
            if (!gpu_state_)
            {
                LOG_ERROR("[ROCmShortConvolution] Missing GPU convolution state");
                return false;
            }

            float *effective_output = output;
            const bool needs_scratch = (input == output);
            if (needs_scratch)
            {
                const int required_scratch_size = seq_len * channels;
                if (!scratchPointer() || scratchCapacity() < required_scratch_size)
                {
                    LOG_ERROR("[ROCmShortConvolution] In-place short-conv scratch was not preallocated: need "
                              << required_scratch_size << " floats, have " << scratchCapacity());
                    return false;
                }
                effective_output = scratchPointer();
            }

            const bool ok = rocmGDN_short_conv1d_effective(
                input, weight, bias, effective_output, gpu_state_,
                seq_len, channels, kernel_size, apply_silu,
                device_effective_seq_len,
                verifier_state_capture_,
                verifier_state_capture_size_,
                verifier_state_capture_rows_,
                device_ordinal_, stream_);
            if (!ok)
                return false;

            if (needs_scratch)
            {
                const size_t count = static_cast<size_t>(seq_len) * static_cast<size_t>(channels);
                rocmGDN_gpu_memcpy_async(output, scratchPointer(), count, stream_);
            }

            return true;
        }

        void setGPUStream(void *stream) override { stream_ = stream; }

        bool exportState(void *dst_host, void *dst_device, void *stream) const override
        {
            if (stateBytes() == 0)
                return true;
            if ((!dst_host && !dst_device) || !gpu_state_)
                return false;

            rocmGDN_gpu_set_device(device_ordinal_);
            if (dst_device)
            {
                auto *dst = static_cast<float *>(dst_device);
                if (stream)
                    rocmGDN_gpu_memcpy_async(dst, gpu_state_, static_cast<size_t>(state_size_), stream);
                else
                    rocmGDN_gpu_memcpy(dst, gpu_state_, static_cast<size_t>(state_size_));
            }
            else
            {
                auto *dst = static_cast<float *>(dst_host);
                if (stream)
                    rocmGDN_gpu_memcpy_d2h_async(dst, gpu_state_, static_cast<size_t>(state_size_), stream);
                else
                    rocmGDN_gpu_memcpy_d2h(dst, gpu_state_, static_cast<size_t>(state_size_));
            }
            return true;
        }

        bool importState(const void *src_host, const void *src_device, void *stream) override
        {
            if (stateBytes() == 0)
                return true;
            const auto *src = static_cast<const float *>(src_host ? src_host : src_device);
            if (!src)
                return false;

            if (!gpu_state_)
                allocateState(state_size_);
            if (!gpu_state_)
                return false;

            rocmGDN_gpu_set_device(device_ordinal_);
            if (stream)
            {
                rocmGDN_gpu_memcpy_async(gpu_state_, src, static_cast<size_t>(state_size_), stream);
            }
            else
            {
                rocmGDN_gpu_memcpy(gpu_state_, src, static_cast<size_t>(state_size_));
            }
            return true;
        }

        void bindScratchWorkspace(float *scratch, int scratch_size) override
        {
            bound_scratch_ = scratch;
            bound_scratch_size_ = scratch_size;
        }

    private:
        int device_ordinal_;
        void *stream_ = nullptr;
        float *gpu_state_ = nullptr;
        int state_size_ = 0;
        float *scratch_ = nullptr;
        int scratch_size_ = 0;
        float *bound_scratch_ = nullptr;
        int bound_scratch_size_ = 0;
        float *verifier_state_capture_ = nullptr;
        int verifier_state_capture_rows_ = 0;
        int verifier_state_capture_size_ = 0;

        float *scratchPointer() const
        {
            return bound_scratch_ ? bound_scratch_ : scratch_;
        }

        int scratchCapacity() const
        {
            return bound_scratch_ ? bound_scratch_size_ : scratch_size_;
        }

        bool allocateScratch(int scratch_size)
        {
            if (bound_scratch_ && bound_scratch_size_ >= scratch_size)
                return true;
            if (scratch_ && scratch_size_ >= scratch_size)
                return true;
            if (scratch_)
            {
                rocmGDN_gpu_set_device(device_ordinal_);
                rocmGDN_gpu_free(scratch_);
                scratch_ = nullptr;
            }

            scratch_size_ = scratch_size;
            rocmGDN_gpu_set_device(device_ordinal_);
            if (!rocmGDN_gpu_malloc(&scratch_, scratch_size_))
            {
                LOG_ERROR("[ROCmShortConvolution] GPU malloc failed for in-place prefill scratch");
                scratch_ = nullptr;
                scratch_size_ = 0;
                return false;
            }
            return true;
        }
    };

} // namespace llaminar2
