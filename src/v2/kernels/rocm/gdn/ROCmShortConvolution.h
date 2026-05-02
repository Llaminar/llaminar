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

    // GPU memory helpers (implemented in ROCmGatedDeltaNetKernels.hip)
    bool rocmGDN_gpu_malloc(float **ptr, size_t count);
    void rocmGDN_gpu_free(float *ptr);
    void rocmGDN_gpu_memset_zero(float *ptr, size_t count);
    void rocmGDN_gpu_memset_zero_async(float *ptr, size_t count, void *stream);
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
        }

        void allocateGPUState(int state_size) override { allocateState(state_size); }
        void resetGPUState() override { resetState(); }

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
            if (gpu_state_ && state_size_ > 0)
            {
                rocmGDN_gpu_set_device(device_ordinal_);
                void *stream = GPUDeviceContextPool::instance().getAMDContext(device_ordinal_).defaultStream();
                rocmGDN_gpu_memset_zero_async(gpu_state_, state_size_, stream);
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

            // All pointers are device pointers — pass directly to HIP kernel.
            return rocmGDN_short_conv1d(
                input, weight, bias, output, effective_state,
                seq_len, channels, kernel_size, apply_silu,
                device_ordinal_, stream_);
        }

        void setGPUStream(void *stream) override { stream_ = stream; }

    private:
        int device_ordinal_;
        void *stream_ = nullptr;
        float *gpu_state_ = nullptr;
        int state_size_ = 0;
    };

} // namespace llaminar2
