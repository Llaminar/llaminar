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

    bool cudaGDN_short_conv1d_effective(
        const float *input, const float *weight, const float *bias,
        float *output, float *conv_state,
        int seq_len, int channels, int kernel_size,
        bool apply_silu,
        const int *device_effective_seq_len,
        int device_idx, void *stream);

    // GPU memory helpers (implemented in CUDAGatedDeltaNetKernels.cu)
    bool cudaGDN_gpu_malloc(float **ptr, size_t count);
    void cudaGDN_gpu_free(float *ptr);
    void cudaGDN_gpu_memset_zero(float *ptr, size_t count);
    void cudaGDN_gpu_memset_zero_async(float *ptr, size_t count, void *stream);
    void cudaGDN_gpu_memcpy_async(float *dst, const float *src, size_t count, void *stream);
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
            cudaGDN_gpu_free(scratch_);
        }

        void allocateGPUState(int state_size) override { allocateState(state_size); }
        bool allocateGPUScratch(int scratch_size) override { return allocateScratch(scratch_size); }
        void resetGPUState() override { resetState(); }
        bool supportsPaddedPrefillRealLength() const override { return true; }

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
            float *effective_output = output;

            // Prefill runs one thread per timestep/channel. In-place output can
            // overwrite an input timestep before another thread reads it, so use
            // persistent scratch only for the in-place prefill case.
            const bool needs_scratch = (seq_len > 1 && input == output);
            if (needs_scratch)
            {
                const int required_scratch_size = seq_len * channels;
                if (!scratchPointer() || scratchCapacity() < required_scratch_size)
                {
                    LOG_ERROR("[CUDAShortConvolution] In-place prefill scratch was not preallocated: need "
                              << required_scratch_size << " floats, have " << scratchCapacity());
                    return false;
                }
                effective_output = scratchPointer();
            }

            // All pointers are device pointers — pass directly to CUDA kernel.
            const bool ok = cudaGDN_short_conv1d(
                input, weight, bias, effective_output, effective_state,
                seq_len, channels, kernel_size, apply_silu,
                device_ordinal_, stream_);
            if (!ok)
                return false;

            if (needs_scratch)
            {
                const size_t count = static_cast<size_t>(seq_len) * static_cast<size_t>(channels);
                cudaGDN_gpu_memcpy_async(output, scratchPointer(), count, stream_);
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
            cudaGDN_gpu_set_device(device_ordinal_);
            float *effective_state = gpu_state_ ? gpu_state_ : conv_state;
            float *effective_output = output;

            const bool needs_scratch = (seq_len > 1 && input == output);
            if (needs_scratch)
            {
                const int required_scratch_size = seq_len * channels;
                if (!scratchPointer() || scratchCapacity() < required_scratch_size)
                {
                    LOG_ERROR("[CUDAShortConvolution] In-place prefill scratch was not preallocated: need "
                              << required_scratch_size << " floats, have " << scratchCapacity());
                    return false;
                }
                effective_output = scratchPointer();
            }

            const bool ok = cudaGDN_short_conv1d_effective(
                input, weight, bias, effective_output, effective_state,
                seq_len, channels, kernel_size, apply_silu,
                device_effective_seq_len,
                device_ordinal_, stream_);
            if (!ok)
                return false;

            if (needs_scratch)
            {
                const size_t count = static_cast<size_t>(seq_len) * static_cast<size_t>(channels);
                cudaGDN_gpu_memcpy_async(output, scratchPointer(), count, stream_);
            }

            return true;
        }

        void setGPUStream(void *stream) override { stream_ = stream; }

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
                cudaGDN_gpu_set_device(device_ordinal_);
                cudaGDN_gpu_free(scratch_);
                scratch_ = nullptr;
            }

            scratch_size_ = scratch_size;
            cudaGDN_gpu_set_device(device_ordinal_);
            if (!cudaGDN_gpu_malloc(&scratch_, scratch_size_))
            {
                LOG_ERROR("[CUDAShortConvolution] GPU malloc failed for in-place prefill scratch");
                scratch_ = nullptr;
                scratch_size_ = 0;
                return false;
            }
            return true;
        }
    };

} // namespace llaminar2
