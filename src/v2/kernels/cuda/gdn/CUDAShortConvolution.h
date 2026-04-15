/**
 * @file CUDAShortConvolution.h
 * @brief CUDA implementation of ITensorShortConvolution
 *
 * Wraps CUDA kernels for causal depthwise conv1d + SiLU.
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

// Forward declaration of extern "C" kernel wrapper
extern "C"
{
    bool cudaGDN_short_conv1d(
        const float *input, const float *weight, const float *bias,
        float *output, float *conv_state,
        int seq_len, int channels, int kernel_size,
        bool apply_silu,
        int device_idx, void *stream);

    // GPU memory helpers (implemented in CUDAGatedDeltaNetKernels.cu)
    bool cudaGDN_gpu_malloc(float **ptr, size_t count);
    void cudaGDN_gpu_free(float *ptr);
    void cudaGDN_gpu_memset_zero(float *ptr, size_t count);
    void cudaGDN_gpu_memset_zero_async(float *ptr, size_t count, void *stream);
    void cudaGDN_gpu_set_device(int ordinal);
    void cudaGDN_stream_synchronize(void *stream);
}

namespace llaminar2
{

    class CUDAShortConvolution : public ITensorShortConvolution
    {
    public:
        explicit CUDAShortConvolution(int device_ordinal)
            : device_ordinal_(device_ordinal) {}

        ~CUDAShortConvolution()
        {
            cudaGDN_gpu_set_device(device_ordinal_);
            cudaGDN_gpu_free(gpu_state_);
        }

        void allocateGPUState(int state_size) override { allocateState(state_size); }
        void resetGPUState() override { resetState(); }

        /// Allocate GPU conv state [channels * (kernel_size - 1)]
        void allocateState(int state_size)
        {
            if (gpu_state_ && state_size_ == state_size)
                return;
            if (gpu_state_)
            {
                cudaGDN_gpu_set_device(device_ordinal_);
                cudaGDN_gpu_free(gpu_state_);
            }
            state_size_ = state_size;
            cudaGDN_gpu_set_device(device_ordinal_);
            if (!cudaGDN_gpu_malloc(&gpu_state_, state_size))
            {
                LOG_ERROR("[CUDAShortConvolution] GPU malloc failed for state");
                gpu_state_ = nullptr;
                return;
            }
            void *stream = GPUDeviceContextPool::instance().getNvidiaContext(device_ordinal_).defaultStream();
            cudaGDN_gpu_memset_zero_async(gpu_state_, state_size, stream);
            cudaGDN_stream_synchronize(stream);
            LOG_DEBUG("[CUDAShortConvolution] Allocated GPU state: " << state_size << " floats on device " << device_ordinal_);
        }

        void resetState()
        {
            if (gpu_state_ && state_size_ > 0)
            {
                cudaGDN_gpu_set_device(device_ordinal_);
                void *stream = GPUDeviceContextPool::instance().getNvidiaContext(device_ordinal_).defaultStream();
                cudaGDN_gpu_memset_zero_async(gpu_state_, state_size_, stream);
                cudaGDN_stream_synchronize(stream);
            }
        }

        bool forward(
            const float *input, const float *weight, const float *bias,
            float *output, float *conv_state,
            int seq_len, int channels, int kernel_size,
            bool apply_silu = true) override
        {
            cudaGDN_gpu_set_device(device_ordinal_);
            float *effective_state = gpu_state_ ? gpu_state_ : conv_state;

            // All pointers are device pointers — pass directly to CUDA kernel.
            // No H2D/D2H copies, no scratch buffer, no stream synchronization.
            return cudaGDN_short_conv1d(
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
