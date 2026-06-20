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
#include "../../../execution/local_execution/graph/GraphCaptureGuard.h"
#include "../../../utils/Logger.h"

#include <algorithm>

// Forward declaration of extern "C" kernel wrapper
extern "C"
{
    bool cudaGDN_short_conv1d(
        const float *input, const float *weight, const float *bias,
        float *output, float *conv_state,
        int seq_len, int channels, int kernel_size,
        bool apply_silu,
        float *state_snapshots,
        int snapshot_stride_floats,
        int max_snapshot_rows,
        int device_idx, void *stream);

    bool cudaGDN_short_conv1d_effective(
        const float *input, const float *weight, const float *bias,
        float *output, float *conv_state,
        int seq_len, int channels, int kernel_size,
        bool apply_silu,
        const int *device_effective_seq_len,
        float *state_snapshots,
        int snapshot_stride_floats,
        int max_snapshot_rows,
        int device_idx, void *stream);

    // GPU memory helpers (implemented in CUDAGatedDeltaNetKernels.cu)
    bool cudaGDN_gpu_malloc(float **ptr, size_t count);
    void cudaGDN_gpu_free(float *ptr);
    void cudaGDN_gpu_memset_zero(float *ptr, size_t count);
    void cudaGDN_gpu_memset_zero_async(float *ptr, size_t count, void *stream);
    void cudaGDN_gpu_memcpy(float *dst, const float *src, size_t count);
    void cudaGDN_gpu_memcpy_async(float *dst, const float *src, size_t count, void *stream);
    void cudaGDN_gpu_memcpy_d2h(float *host_dst, const float *device_src, size_t count);
    void cudaGDN_gpu_memcpy_d2h_async(float *host_dst, const float *device_src, size_t count, void *stream);
    void cudaGDN_gpu_set_device(int ordinal);
    void cudaGDN_stream_synchronize(void *stream);
    bool cudaGDN_gpu_copy_capture_row_from_device_index(
        float *dst,
        const float *capture,
        const int *device_row_index,
        int rows,
        int state_size,
        int device_idx,
        void *stream);
    bool cudaGDN_gpu_copy_capture_rows_from_device_indices(
        float *dst,
        const float *capture,
        const int *device_row_indices,
        int request_count,
        int row_index_stride,
        int rows,
        int state_size,
        int device_idx,
        void *stream);
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
            cudaGDN_gpu_free(request_state_bank_);
            cudaGDN_gpu_free(scratch_);
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
        void bindSpeculativeStateWorkspace(float *workspace, int state_size) override
        {
            speculative_state_work_ = workspace;
            speculative_state_work_size_ = state_size;
        }

        bool restoreVerifierStateCaptureRow(float *dst_state, int row, void *stream) override
        {
            if (!gpu_state_ || !verifier_state_capture_ ||
                row < 0 || row >= verifier_state_capture_rows_ ||
                verifier_state_capture_size_ != state_size_)
            {
                return false;
            }

            cudaGDN_gpu_set_device(device_ordinal_);
            const float *src =
                verifier_state_capture_ +
                static_cast<size_t>(row) * static_cast<size_t>(verifier_state_capture_size_);
            if (stream)
            {
                cudaGDN_gpu_memcpy_async(gpu_state_, src, static_cast<size_t>(state_size_), stream);
                if (dst_state)
                {
                    /*
                     * Keep the hybrid cache's host conv-state mirror aligned
                     * with the device state restored for graph replay.  A graph
                     * rebuild after publication may consult this host vector.
                     */
                    cudaGDN_gpu_memcpy_d2h_async(dst_state, src, static_cast<size_t>(state_size_), stream);
                    cudaGDN_stream_synchronize(stream);
                }
            }
            else
            {
                cudaGDN_gpu_memcpy(gpu_state_, src, static_cast<size_t>(state_size_));
                if (dst_state)
                    cudaGDN_gpu_memcpy_d2h(dst_state, src, static_cast<size_t>(state_size_));
            }
            return true;
        }

        bool restoreVerifierStateCaptureRowFromDeviceIndex(
            float *dst_state,
            const int *device_row_index,
            void *stream) override
        {
            /*
             * Device-indexed MTP publication restores only implementation-owned
             * GPU live state. Keeping the host mirror untouched avoids a D2H
             * sync on the verifier stream and makes replay ownership explicit.
             */
            (void)dst_state;
            if (!gpu_state_ || !verifier_state_capture_ || !device_row_index ||
                !stream ||
                verifier_state_capture_size_ != state_size_)
            {
                return false;
            }

            const bool ok = cudaGDN_gpu_copy_capture_row_from_device_index(
                gpu_state_,
                verifier_state_capture_,
                device_row_index,
                verifier_state_capture_rows_,
                state_size_,
                device_ordinal_,
                stream);
            if (!ok)
                return false;
            return true;
        }

        bool restoreVerifierStateCaptureRowsFromDeviceIndices(
            float *dst_states,
            int dst_state_stride_floats,
            const int *device_row_indices,
            int request_count,
            int row_index_stride,
            void *stream) override
        {
            (void)dst_states;
            (void)dst_state_stride_floats;
            if (!request_state_bank_ ||
                request_state_bank_state_size_ <= 0 ||
                request_state_bank_capacity_ < request_count ||
                !verifier_state_capture_ ||
                !device_row_indices ||
                request_count <= 0 ||
                row_index_stride <= 0 ||
                !stream ||
                verifier_state_capture_size_ != request_state_bank_state_size_)
            {
                return false;
            }

            return cudaGDN_gpu_copy_capture_rows_from_device_indices(
                request_state_bank_,
                verifier_state_capture_,
                device_row_indices,
                request_count,
                row_index_stride,
                verifier_state_capture_rows_,
                request_state_bank_state_size_,
                device_ordinal_,
                stream);
        }

        bool supportsPaddedPrefillRealLength() const override { return true; }
        bool supportsRequestLiveStateBank(int request_count, int state_size) const override
        {
            return request_count > 0 && state_size > 0;
        }
        size_t stateBytes() const override
        {
            return state_size_ > 0 ? static_cast<size_t>(state_size_) * sizeof(float) : 0;
        }

        /// Allocate GPU conv state [channels * (kernel_size - 1)]
        void allocateState(int state_size)
        {
            if (gpu_state_ && state_size_ == state_size)
                return;
            if (isGraphCaptureActive())
            {
                LOG_ERROR("[CUDAShortConvolution] GPU state allocation during graph capture "
                          "(need "
                          << state_size << " floats, have " << state_size_ << ")");
                return;
            }
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
                if (request_state_bank_ && request_state_bank_capacity_ > 0)
                    cudaGDN_gpu_memset_zero_async(
                        request_state_bank_,
                        static_cast<size_t>(request_state_bank_capacity_) *
                            static_cast<size_t>(request_state_bank_state_size_),
                        stream);
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
            const int required_state_size = channels * (kernel_size - 1);
            if (!gpu_state_ || state_size_ != required_state_size)
            {
                if (isGraphCaptureActive())
                {
                    LOG_ERROR("[CUDAShortConvolution::forward] GPU state allocation during graph capture "
                              "(need "
                              << required_state_size << " floats, have " << state_size_ << ")");
                    return false;
                }
                allocateState(required_state_size);
            }
            if (!gpu_state_)
            {
                LOG_ERROR("[CUDAShortConvolution] Missing GPU convolution state");
                return false;
            }
            float *effective_state =
                prepareEffectiveStateForVerifierForward(required_state_size, stream_);
            if (!effective_state)
                return false;
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
                    LOG_ERROR("[CUDAShortConvolution] In-place short-conv scratch was not preallocated: need "
                              << required_scratch_size << " floats, have " << scratchCapacity());
                    return false;
                }
                effective_output = scratchPointer();
            }

            // All pointers are device pointers — pass directly to CUDA kernel.
            const bool ok = cudaGDN_short_conv1d(
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
            const int required_state_size = channels * (kernel_size - 1);
            if (!gpu_state_ || state_size_ != required_state_size)
            {
                if (isGraphCaptureActive())
                {
                    LOG_ERROR("[CUDAShortConvolution::forwardWithEffectiveSeqLen] GPU state allocation during graph capture "
                              "(need "
                              << required_state_size << " floats, have " << state_size_ << ")");
                    return false;
                }
                allocateState(required_state_size);
            }
            if (!gpu_state_)
            {
                LOG_ERROR("[CUDAShortConvolution] Missing GPU convolution state");
                return false;
            }
            float *effective_state =
                prepareEffectiveStateForVerifierForward(required_state_size, stream_);
            if (!effective_state)
                return false;
            float *effective_output = output;

            const bool needs_scratch = (input == output);
            if (needs_scratch)
            {
                const int required_scratch_size = seq_len * channels;
                if (!scratchPointer() || scratchCapacity() < required_scratch_size)
                {
                    LOG_ERROR("[CUDAShortConvolution] In-place short-conv scratch was not preallocated: need "
                              << required_scratch_size << " floats, have " << scratchCapacity());
                    return false;
                }
                effective_output = scratchPointer();
            }

            const bool ok = cudaGDN_short_conv1d_effective(
                input, weight, bias, effective_output, effective_state,
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
                cudaGDN_gpu_memcpy_async(output, scratchPointer(), count, stream_);
            }

            return true;
        }

        bool forwardBatchedRequests(
            const float *input, const float *weight, const float *bias,
            float *output, float *conv_state,
            int seq_len, int request_count, int request_seq_len,
            int channels, int kernel_size,
            bool apply_silu = true) override
        {
            (void)conv_state;
            cudaGDN_gpu_set_device(device_ordinal_);
            const int required_state_size = channels * (kernel_size - 1);
            if (seq_len <= 0 || request_count <= 0 || request_seq_len <= 0 ||
                seq_len != request_count * request_seq_len ||
                required_state_size <= 0)
            {
                LOG_ERROR("[CUDAShortConvolution] Invalid request-batched shape"
                          << " seq_len=" << seq_len
                          << " request_count=" << request_count
                          << " request_seq_len=" << request_seq_len
                          << " state_size=" << required_state_size);
                return false;
            }
            if (!ensureRequestStateBank(request_count, required_state_size))
                return false;

            const bool capture_active =
                verifier_state_capture_ != nullptr &&
                verifier_state_capture_rows_ >= request_count * request_seq_len &&
                verifier_state_capture_size_ == required_state_size;
            if (capture_active &&
                (!stream_ ||
                 !speculative_state_work_ ||
                 speculative_state_work_size_ < request_count * required_state_size))
            {
                LOG_ERROR("[CUDAShortConvolution] Request-batched verifier requires "
                          "one speculative conv-state work slot per request");
                return false;
            }

            float *effective_output = output;
            const bool needs_scratch = (input == output);
            const int flattened_output_floats = seq_len * channels;
            if (needs_scratch)
            {
                if (!scratchPointer() || scratchCapacity() < flattened_output_floats)
                {
                    LOG_ERROR("[CUDAShortConvolution] Request-batched in-place scratch too small: need "
                              << flattened_output_floats << " floats, have "
                              << scratchCapacity());
                    return false;
                }
                effective_output = scratchPointer();
            }

            for (int request = 0; request < request_count; ++request)
            {
                const size_t row_offset =
                    static_cast<size_t>(request) *
                    static_cast<size_t>(request_seq_len) *
                    static_cast<size_t>(channels);
                float *state =
                    request_state_bank_ +
                    static_cast<size_t>(request) *
                        static_cast<size_t>(required_state_size);
                float *snapshots = nullptr;
                int snapshot_rows = 0;
                if (capture_active)
                {
                    float *work_state =
                        speculative_state_work_ +
                        static_cast<size_t>(request) *
                            static_cast<size_t>(required_state_size);
                    cudaGDN_gpu_memcpy_async(
                        work_state,
                        state,
                        static_cast<size_t>(required_state_size),
                        stream_);
                    state = work_state;

                    const int snapshot_base = request * request_seq_len;
                    snapshots =
                        verifier_state_capture_ +
                        static_cast<size_t>(snapshot_base) *
                            static_cast<size_t>(required_state_size);
                    snapshot_rows =
                        std::min(request_seq_len, verifier_state_capture_rows_ - snapshot_base);
                }

                if (!cudaGDN_short_conv1d(
                        input + row_offset,
                        weight,
                        bias,
                        effective_output + row_offset,
                        state,
                        request_seq_len, channels, kernel_size, apply_silu,
                        snapshots,
                        required_state_size,
                        snapshot_rows,
                        device_ordinal_, stream_))
                {
                    return false;
                }
            }

            if (needs_scratch)
            {
                cudaGDN_gpu_memcpy_async(
                    output,
                    scratchPointer(),
                    static_cast<size_t>(flattened_output_floats),
                    stream_);
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

            cudaGDN_gpu_set_device(device_ordinal_);
            if (dst_device)
            {
                auto *dst = static_cast<float *>(dst_device);
                if (stream)
                    cudaGDN_gpu_memcpy_async(dst, gpu_state_, static_cast<size_t>(state_size_), stream);
                else
                    cudaGDN_gpu_memcpy(dst, gpu_state_, static_cast<size_t>(state_size_));
            }
            else
            {
                auto *dst = static_cast<float *>(dst_host);
                if (stream)
                    cudaGDN_gpu_memcpy_d2h_async(dst, gpu_state_, static_cast<size_t>(state_size_), stream);
                else
                    cudaGDN_gpu_memcpy_d2h(dst, gpu_state_, static_cast<size_t>(state_size_));
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

            cudaGDN_gpu_set_device(device_ordinal_);
            if (stream)
            {
                cudaGDN_gpu_memcpy_async(gpu_state_, src, static_cast<size_t>(state_size_), stream);
            }
            else
            {
                cudaGDN_gpu_memcpy(gpu_state_, src, static_cast<size_t>(state_size_));
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
        float *request_state_bank_ = nullptr;
        int request_state_bank_state_size_ = 0;
        int request_state_bank_capacity_ = 0;
        float *scratch_ = nullptr;
        int scratch_size_ = 0;
        float *bound_scratch_ = nullptr;
        int bound_scratch_size_ = 0;
        float *verifier_state_capture_ = nullptr;
        int verifier_state_capture_rows_ = 0;
        int verifier_state_capture_size_ = 0;
        float *speculative_state_work_ = nullptr;
        int speculative_state_work_size_ = 0;

        float *scratchPointer() const
        {
            return bound_scratch_ ? bound_scratch_ : scratch_;
        }

        int scratchCapacity() const
        {
            return bound_scratch_ ? bound_scratch_size_ : scratch_size_;
        }

        float *prepareEffectiveStateForVerifierForward(int required_state_size, void *stream)
        {
            const bool verifier_capture_active =
                verifier_state_capture_ != nullptr &&
                verifier_state_capture_rows_ > 0 &&
                verifier_state_capture_size_ == required_state_size;
            if (!verifier_capture_active)
                return gpu_state_;
            if (!stream)
            {
                LOG_ERROR("[CUDAShortConvolution] Speculative verifier state requires an explicit stream");
                return nullptr;
            }
            if (!speculative_state_work_ ||
                speculative_state_work_size_ < required_state_size)
            {
                LOG_ERROR("[CUDAShortConvolution] Speculative verifier state workspace was not bound: need "
                          << required_state_size << " floats, have "
                          << speculative_state_work_size_);
                return nullptr;
            }

            cudaGDN_gpu_memcpy_async(
                speculative_state_work_,
                gpu_state_,
                static_cast<size_t>(required_state_size),
                stream);
            return speculative_state_work_;
        }

        bool ensureRequestStateBank(int request_count, int required_state_size)
        {
            if (request_count <= 0 || required_state_size <= 0)
                return false;

            if (!gpu_state_ || state_size_ != required_state_size)
            {
                if (isGraphCaptureActive())
                {
                    LOG_ERROR("[CUDAShortConvolution] scalar conv state allocation during graph capture");
                    return false;
                }
                allocateState(required_state_size);
            }
            if (!gpu_state_)
                return false;

            if (request_state_bank_ &&
                request_state_bank_state_size_ == required_state_size &&
                request_state_bank_capacity_ >= request_count)
            {
                return true;
            }

            if (isGraphCaptureActive())
            {
                LOG_ERROR("[CUDAShortConvolution] request conv-state bank allocation during graph capture "
                          "(requests=" << request_count
                          << ", state_size=" << required_state_size << ")");
                return false;
            }

            cudaGDN_gpu_set_device(device_ordinal_);
            if (request_state_bank_)
                cudaGDN_gpu_free(request_state_bank_);
            request_state_bank_ = nullptr;
            request_state_bank_state_size_ = required_state_size;
            request_state_bank_capacity_ = request_count;

            const size_t total_floats =
                static_cast<size_t>(request_count) *
                static_cast<size_t>(required_state_size);
            if (!cudaGDN_gpu_malloc(&request_state_bank_, total_floats))
            {
                LOG_ERROR("[CUDAShortConvolution] GPU malloc failed for request conv-state bank");
                request_state_bank_state_size_ = 0;
                request_state_bank_capacity_ = 0;
                return false;
            }

            void *stream = GPUDeviceContextPool::instance().getNvidiaContext(device_ordinal_).defaultStream();
            cudaGDN_gpu_memset_zero_async(request_state_bank_, total_floats, stream);
            cudaGDN_gpu_memcpy_async(
                request_state_bank_,
                gpu_state_,
                static_cast<size_t>(required_state_size),
                stream);
            cudaGDN_stream_synchronize(stream);
            return true;
        }

        bool allocateScratch(int scratch_size)
        {
            if (bound_scratch_ && bound_scratch_size_ >= scratch_size)
                return true;
            if (scratch_ && scratch_size_ >= scratch_size)
                return true;
            if (scratch_)
            {
                if (isGraphCaptureActive())
                {
                    LOG_ERROR("[CUDAShortConvolution] in-place prefill scratch realloc during graph capture "
                              "(need "
                              << scratch_size << " floats, have " << scratch_size_ << ")");
                    return false;
                }
                cudaGDN_gpu_set_device(device_ordinal_);
                cudaGDN_gpu_free(scratch_);
                scratch_ = nullptr;
            }

            if (isGraphCaptureActive())
            {
                LOG_ERROR("[CUDAShortConvolution] in-place prefill scratch allocation during graph capture "
                          "(need "
                          << scratch_size << " floats)");
                return false;
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
