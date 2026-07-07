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
#include "../../../execution/local_execution/graph/GraphCaptureGuard.h"
#include "../../../utils/Logger.h"

#include <algorithm>

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
    bool rocmGDN_gpu_copy_capture_row_from_device_index(
        float *dst,
        const float *capture,
        const int *device_row_index,
        int rows,
        int state_size,
        int device_idx,
        void *stream);
    bool rocmGDN_gpu_copy_capture_rows_from_device_indices(
        float *dst,
        const float *capture,
        const int *device_row_indices,
        int request_count,
        int row_index_stride,
        int rows,
        int state_size,
        int device_idx,
        void *stream);
    bool rocmGDN_compact_modular_conv_state(
        const float *gathered,
        float *full,
        int degree,
        int qk_channels,
        int local_v_channels,
        int full_v_channels,
        int history_len,
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
            rocmGDN_gpu_free(secondary_gpu_state_);
            rocmGDN_gpu_free(request_state_bank_);
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
        void bindSpeculativeStateWorkspace(float *workspace, int state_size) override
        {
            speculative_state_work_ = workspace;
            speculative_state_work_size_ = state_size;
        }

        bool restoreVerifierStateCaptureRow(float *dst_state, int row, void *stream) override
        {
            if (!selectState(verifier_state_capture_size_) ||
                !verifier_state_capture_ ||
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
            {
                rocmGDN_gpu_memcpy_async(gpu_state_, src, static_cast<size_t>(state_size_), stream);
                if (dst_state)
                {
                    /*
                     * Keep the host conv-state mirror aligned with the HIP
                     * state restored for graph replay.  This makes publication
                     * atomic across graph rebuilds and captured replays.
                     */
                    rocmGDN_gpu_memcpy_d2h_async(dst_state, src, static_cast<size_t>(state_size_), stream);
                    rocmGDN_stream_synchronize(stream);
                }
            }
            else
            {
                rocmGDN_gpu_memcpy(gpu_state_, src, static_cast<size_t>(state_size_));
                if (dst_state)
                    rocmGDN_gpu_memcpy_d2h(dst_state, src, static_cast<size_t>(state_size_));
            }
            return true;
        }

        bool restoreVerifierStateCaptureRowFromDeviceIndex(
            float *dst_state,
            const int *device_row_index,
            void *stream) override
        {
            /*
             * Device-indexed MTP publication is a device-owned state handoff.
             * Host mirror refresh is deliberately excluded so replay can stay
             * ordered on the explicit HIP stream without a D2H synchronization.
             */
            (void)dst_state;
            if (!selectState(verifier_state_capture_size_) ||
                !verifier_state_capture_ || !device_row_index ||
                !stream ||
                verifier_state_capture_size_ != state_size_)
            {
                return false;
            }

            const bool ok = rocmGDN_gpu_copy_capture_row_from_device_index(
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

            return rocmGDN_gpu_copy_capture_rows_from_device_indices(
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

        size_t largestStateBytes() const override
        {
            const int largest_state_size = std::max(state_size_, secondary_state_size_);
            return largest_state_size > 0 ? static_cast<size_t>(largest_state_size) * sizeof(float) : 0;
        }

        void allocateState(int state_size)
        {
            if (selectState(state_size))
                return;
            if (isGraphCaptureActive())
            {
                LOG_ERROR("[ROCmShortConvolution] GPU state allocation during graph capture "
                          "(need "
                          << state_size << " floats, have active=" << state_size_
                          << " secondary=" << secondary_state_size_ << ")");
                return;
            }
            rocmGDN_gpu_set_device(device_ordinal_);
            float *new_state = nullptr;
            if (!rocmGDN_gpu_malloc(&new_state, state_size))
            {
                LOG_ERROR("[ROCmShortConvolution] GPU malloc failed for state");
                return;
            }
            void *stream = GPUDeviceContextPool::instance().getAMDContext(device_ordinal_).defaultStream();
            rocmGDN_gpu_memset_zero_async(new_state, state_size, stream);
            rocmGDN_stream_synchronize(stream);

            if (gpu_state_)
            {
                if (secondary_gpu_state_)
                    rocmGDN_gpu_free(secondary_gpu_state_);
                secondary_gpu_state_ = gpu_state_;
                secondary_state_size_ = state_size_;
            }
            gpu_state_ = new_state;
            state_size_ = state_size;
            LOG_DEBUG("[ROCmShortConvolution] Allocated GPU state: " << state_size << " floats on device " << device_ordinal_);
        }

        void resetState()
        {
            rocmGDN_gpu_set_device(device_ordinal_);
            if (gpu_state_ && state_size_ > 0)
            {
                void *stream = GPUDeviceContextPool::instance().getAMDContext(device_ordinal_).defaultStream();
                rocmGDN_gpu_memset_zero_async(gpu_state_, state_size_, stream);
                if (secondary_gpu_state_ && secondary_state_size_ > 0)
                    rocmGDN_gpu_memset_zero_async(secondary_gpu_state_, secondary_state_size_, stream);
                if (request_state_bank_ && request_state_bank_capacity_ > 0)
                    rocmGDN_gpu_memset_zero_async(
                        request_state_bank_,
                        static_cast<size_t>(request_state_bank_capacity_) *
                            static_cast<size_t>(request_state_bank_state_size_),
                        stream);
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
            if (!ensureActiveState(required_state_size, "ROCmShortConvolution::forward"))
                return false;
            if (!gpu_state_)
            {
                LOG_ERROR("[ROCmShortConvolution] Missing GPU convolution state");
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
            if (!ensureActiveState(required_state_size, "ROCmShortConvolution::forwardWithEffectiveSeqLen"))
                return false;
            if (!gpu_state_)
            {
                LOG_ERROR("[ROCmShortConvolution] Missing GPU convolution state");
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
                    LOG_ERROR("[ROCmShortConvolution] In-place short-conv scratch was not preallocated: need "
                              << required_scratch_size << " floats, have " << scratchCapacity());
                    return false;
                }
                effective_output = scratchPointer();
            }

            const bool ok = rocmGDN_short_conv1d_effective(
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
                rocmGDN_gpu_memcpy_async(output, scratchPointer(), count, stream_);
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
            rocmGDN_gpu_set_device(device_ordinal_);
            const int required_state_size = channels * (kernel_size - 1);
            if (seq_len <= 0 || request_count <= 0 || request_seq_len <= 0 ||
                seq_len != request_count * request_seq_len ||
                required_state_size <= 0)
            {
                LOG_ERROR("[ROCmShortConvolution] Invalid request-batched shape"
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
                LOG_ERROR("[ROCmShortConvolution] Request-batched verifier requires "
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
                    LOG_ERROR("[ROCmShortConvolution] Request-batched in-place scratch too small: need "
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
                    rocmGDN_gpu_memcpy_async(
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

                if (!rocmGDN_short_conv1d(
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
                rocmGDN_gpu_memcpy_async(
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

        bool exportStateForSize(int target_state_size, void *dst_host, void *dst_device, void *stream) override
        {
            if (target_state_size <= 0)
                return true;
            if (!dst_host && !dst_device)
                return false;

            const float *src = nullptr;
            if (gpu_state_ && state_size_ == target_state_size)
                src = gpu_state_;
            else if (secondary_gpu_state_ && secondary_state_size_ == target_state_size)
                src = secondary_gpu_state_;
            if (!src)
                return false;

            rocmGDN_gpu_set_device(device_ordinal_);
            if (dst_device)
            {
                auto *dst = static_cast<float *>(dst_device);
                if (stream)
                    rocmGDN_gpu_memcpy_async(dst, src, static_cast<size_t>(target_state_size), stream);
                else
                    rocmGDN_gpu_memcpy(dst, src, static_cast<size_t>(target_state_size));
            }
            else
            {
                auto *dst = static_cast<float *>(dst_host);
                if (stream)
                    rocmGDN_gpu_memcpy_d2h_async(dst, src, static_cast<size_t>(target_state_size), stream);
                else
                    rocmGDN_gpu_memcpy_d2h(dst, src, static_cast<size_t>(target_state_size));
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

        bool importStateForSize(int target_state_size, const void *src_host, const void *src_device, void *stream) override
        {
            if (target_state_size <= 0)
                return true;
            const auto *src = static_cast<const float *>(src_host ? src_host : src_device);
            if (!src)
                return false;
            if (!ensureActiveState(target_state_size, "ROCmShortConvolution::importStateForSize"))
                return false;

            rocmGDN_gpu_set_device(device_ordinal_);
            if (stream)
                rocmGDN_gpu_memcpy_async(gpu_state_, src, static_cast<size_t>(target_state_size), stream);
            else
                rocmGDN_gpu_memcpy(gpu_state_, src, static_cast<size_t>(target_state_size));
            /*
             * The request-batched bank is derived from the active live state.
             * After an explicit state import, the allocation may still exist
             * with a matching shape but its contents no longer represent the
             * imported prefix/publication state.
             */
            request_state_bank_state_size_ = 0;
            request_state_bank_capacity_ = 0;
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
        float *secondary_gpu_state_ = nullptr;
        int secondary_state_size_ = 0;
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

        bool selectState(int required_state_size)
        {
            if (gpu_state_ && state_size_ == required_state_size)
                return true;
            if (!secondary_gpu_state_ || secondary_state_size_ != required_state_size)
                return false;

            float *old_state = gpu_state_;
            const int old_size = state_size_;
            gpu_state_ = secondary_gpu_state_;
            state_size_ = secondary_state_size_;
            secondary_gpu_state_ = old_state;
            secondary_state_size_ = old_size;
            return true;
        }

        bool ensureActiveState(int required_state_size, const char *caller)
        {
            if (selectState(required_state_size))
                return true;
            if (isGraphCaptureActive())
            {
                LOG_ERROR("["
                          << caller << "] GPU state allocation during graph capture "
                          << "(need " << required_state_size
                          << " floats, have active=" << state_size_
                          << " secondary=" << secondary_state_size_ << ")");
                return false;
            }
            allocateState(required_state_size);
            return selectState(required_state_size);
        }

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
                LOG_ERROR("[ROCmShortConvolution] Speculative verifier state requires an explicit stream");
                return nullptr;
            }
            if (!speculative_state_work_ ||
                speculative_state_work_size_ < required_state_size)
            {
                LOG_ERROR("[ROCmShortConvolution] Speculative verifier state workspace was not bound: need "
                          << required_state_size << " floats, have "
                          << speculative_state_work_size_);
                return nullptr;
            }

            rocmGDN_gpu_memcpy_async(
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

            if (!ensureActiveState(required_state_size, "ROCmShortConvolution::ensureRequestStateBank"))
                return false;
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
                LOG_ERROR("[ROCmShortConvolution] request conv-state bank allocation during graph capture "
                          "(requests=" << request_count
                          << ", state_size=" << required_state_size << ")");
                return false;
            }

            rocmGDN_gpu_set_device(device_ordinal_);
            if (request_state_bank_)
                rocmGDN_gpu_free(request_state_bank_);
            request_state_bank_ = nullptr;
            request_state_bank_state_size_ = required_state_size;
            request_state_bank_capacity_ = request_count;

            const size_t total_floats =
                static_cast<size_t>(request_count) *
                static_cast<size_t>(required_state_size);
            if (!rocmGDN_gpu_malloc(&request_state_bank_, total_floats))
            {
                LOG_ERROR("[ROCmShortConvolution] GPU malloc failed for request conv-state bank");
                request_state_bank_state_size_ = 0;
                request_state_bank_capacity_ = 0;
                return false;
            }

            void *stream = GPUDeviceContextPool::instance().getAMDContext(device_ordinal_).defaultStream();
            rocmGDN_gpu_memset_zero_async(request_state_bank_, total_floats, stream);
            rocmGDN_gpu_memcpy_async(
                request_state_bank_,
                gpu_state_,
                static_cast<size_t>(required_state_size),
                stream);
            rocmGDN_stream_synchronize(stream);
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
