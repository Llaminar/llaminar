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
        int device_idx, void *stream);

    bool rocmGDN_short_conv1d_effective(
        const float *input, const float *weight, const float *bias,
        float *output, float *conv_state,
        int seq_len, int channels, int kernel_size,
        bool apply_silu,
        const int *device_effective_seq_len,
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

            // Prefill is parallel over time. When QKV is processed in-place,
            // writing output[t] can clobber input values that another timestep
            // still needs for its causal window. Decode has only one timestep,
            // so it remains safe to write directly in-place.
            const bool needs_scratch = (seq_len > 1 && input == output);
            if (needs_scratch)
            {
                const int required_scratch_size = seq_len * channels;
                if (!scratchPointer() || scratchCapacity() < required_scratch_size)
                {
                    LOG_ERROR("[ROCmShortConvolution] In-place prefill scratch was not preallocated: need "
                              << required_scratch_size << " floats, have " << scratchCapacity());
                    return false;
                }
                effective_output = scratchPointer();
            }

            // All pointers are device pointers — pass directly to HIP kernel.
            const bool ok = rocmGDN_short_conv1d(
                input, weight, bias, effective_output, effective_state,
                seq_len, channels, kernel_size, apply_silu,
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
            const bool needs_scratch = (seq_len > 1 && input == output);
            if (needs_scratch)
            {
                const int required_scratch_size = seq_len * channels;
                if (!scratchPointer() || scratchCapacity() < required_scratch_size)
                {
                    LOG_ERROR("[ROCmShortConvolution] In-place prefill scratch was not preallocated: need "
                              << required_scratch_size << " floats, have " << scratchCapacity());
                    return false;
                }
                effective_output = scratchPointer();
            }

            const bool ok = rocmGDN_short_conv1d_effective(
                input, weight, bias, effective_output, gpu_state_,
                seq_len, channels, kernel_size, apply_silu,
                device_effective_seq_len,
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
            auto *dst = static_cast<float *>(dst_host ? dst_host : dst_device);
            if (!dst || !gpu_state_)
                return false;

            rocmGDN_gpu_set_device(device_ordinal_);
            if (stream)
            {
                rocmGDN_gpu_memcpy_d2h_async(dst, gpu_state_, static_cast<size_t>(state_size_), stream);
                rocmGDN_stream_synchronize(stream);
            }
            else
            {
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
                rocmGDN_stream_synchronize(stream);
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
