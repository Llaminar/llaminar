/**
 * @file CUDAShortConvolution.h
 * @brief CUDA implementation of ITensorShortConvolution
 *
 * Wraps CUDA kernels for causal depthwise conv1d + SiLU.
 * Consumes cache-owned, persistently bound GPU convolution state.
 *
 * Device-pointer design: All input/output pointers passed to forward()
 * are expected to be DEVICE pointers (already on GPU). The stage
 * (ShortConv1dStage) handles coherence via ensureOnDevice() before
 * calling this method.
 */

#pragma once

#include "../../../tensors/TensorKernels.h"
#include "../../../execution/local_execution/graph/GraphCaptureGuard.h"
#include "../../../utils/Logger.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

// Forward declaration of extern "C" kernel wrapper
extern "C"
{
    bool cudaGDN_short_conv1d(
        const float *input, const float *weight, const float *bias,
        float *output, const float *initial_conv_state, float *updated_conv_state,
        int seq_len, int channels, int kernel_size,
        bool apply_silu,
        float *state_snapshots,
        int snapshot_stride_floats,
        int max_snapshot_rows,
        int device_idx, void *stream);

    bool cudaGDN_short_conv1d_effective(
        const float *input, const float *weight, const float *bias,
        float *output, const float *initial_conv_state, float *updated_conv_state,
        int seq_len, int channels, int kernel_size,
        bool apply_silu,
        const int *device_effective_seq_len,
        float *state_snapshots,
        int snapshot_stride_floats,
        int max_snapshot_rows,
        int device_idx, void *stream);

    bool cudaGDN_short_conv1d_batched(
        const float *input, const float *weight, const float *bias,
        float *output, const float *initial_request_states, float *updated_request_states,
        int request_count, int request_seq_len,
        int channels, int kernel_size,
        bool apply_silu,
        const int *device_request_seq_lens,
        float *state_snapshots,
        int snapshot_stride_floats,
        int max_snapshot_rows,
        int device_idx, void *stream);

    // GPU memory helpers (implemented in CUDAGatedDeltaNetKernels.cu)
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
    bool cudaGDN_gpu_copy_capture_terminal_rows_from_device_lengths(
        float *dst,
        const float *capture,
        const int *device_request_seq_lens,
        int request_count,
        int request_row_width,
        int rows,
        int state_size,
        int device_idx,
        void *stream);
    bool cudaGDN_compact_modular_conv_state(
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

    class CUDAShortConvolution : public ITensorShortConvolution
    {
    public:
        explicit CUDAShortConvolution(int device_ordinal)
            : device_ordinal_(device_ordinal) {}

        ~CUDAShortConvolution() override = default;

        bool bindDeviceState(const GDNDeviceStateBinding &binding) override
        {
            if (device_state_bound_ || !binding.valid())
                return false;

            gpu_state_ = binding.primary_state;
            state_size_ = binding.primary_state_floats;
            secondary_gpu_state_ = binding.secondary_state;
            secondary_state_size_ = binding.secondary_state_floats;
            request_state_bank_ = binding.request_state_bank;
            request_state_bank_floats_ = binding.request_state_bank_floats;
            request_state_bank_capacity_ = binding.request_capacity;
            device_state_bound_ = true;
            return true;
        }

        bool resetGPUState(void *stream) override { return resetState(stream); }
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
            // GPU kernels own the only live state. The generic destination is a
            // CPU-backend concern and is intentionally ignored here.
            (void)dst_state;
            float *live_state = stateForSize(verifier_state_capture_size_);
            if (!live_state || !verifier_state_capture_ ||
                row < 0 || row >= verifier_state_capture_rows_ ||
                verifier_state_capture_size_ <= 0)
            {
                return false;
            }

            cudaGDN_gpu_set_device(device_ordinal_);
            const float *src =
                verifier_state_capture_ +
                static_cast<size_t>(row) * static_cast<size_t>(verifier_state_capture_size_);
            if (stream)
            {
                cudaGDN_gpu_memcpy_async(
                    live_state,
                    src,
                    static_cast<size_t>(verifier_state_capture_size_),
                    stream);
            }
            else
            {
                cudaGDN_gpu_memcpy(
                    live_state,
                    src,
                    static_cast<size_t>(verifier_state_capture_size_));
            }
            return publishLiveStateToRequestZero(
                live_state,
                verifier_state_capture_size_,
                stream);
        }

        bool restoreVerifierStateCaptureRowFromDeviceIndex(
            float *dst_state,
            const int *device_row_index,
            void *stream) override
        {
            // Accepted-row metadata and live recurrent state remain on device.
            (void)dst_state;
            float *live_state = stateForSize(verifier_state_capture_size_);
            if (!live_state || !verifier_state_capture_ || !device_row_index ||
                !stream ||
                verifier_state_capture_size_ <= 0)
            {
                return false;
            }

            const bool ok = cudaGDN_gpu_copy_capture_row_from_device_index(
                live_state,
                verifier_state_capture_,
                device_row_index,
                verifier_state_capture_rows_,
                verifier_state_capture_size_,
                device_ordinal_,
                stream);
            if (!ok)
                return false;
            if (!publishLiveStateToRequestZero(
                    live_state,
                    verifier_state_capture_size_,
                    stream))
            {
                return false;
            }
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
                request_state_bank_capacity_ < request_count ||
                !verifier_state_capture_ ||
                !device_row_indices ||
                request_count <= 0 ||
                row_index_stride <= 0 ||
                !stream ||
                verifier_state_capture_size_ <= 0 ||
                !hasState(verifier_state_capture_size_) ||
                request_state_bank_floats_ <
                    static_cast<size_t>(request_count) *
                        static_cast<size_t>(verifier_state_capture_size_))
            {
                return false;
            }

            if (!cudaGDN_gpu_copy_capture_rows_from_device_indices(
                    request_state_bank_,
                    verifier_state_capture_,
                    device_row_indices,
                    request_count,
                    row_index_stride,
                    verifier_state_capture_rows_,
                    verifier_state_capture_size_,
                    device_ordinal_,
                    stream))
            {
                return false;
            }
            return publishRequestZeroToLiveState(
                verifier_state_capture_size_,
                stream);
        }

        bool restoreVerifierStateCaptureRequestTerminalRows(
            float *dst_states,
            const int *device_request_seq_lens,
            int request_count,
            int request_row_width,
            void *stream) override
        {
            (void)dst_states;
            if (!request_state_bank_ ||
                request_state_bank_capacity_ < request_count ||
                !verifier_state_capture_ ||
                !device_request_seq_lens ||
                !stream ||
                verifier_state_capture_size_ <= 0 ||
                !hasState(verifier_state_capture_size_) ||
                request_state_bank_floats_ <
                    static_cast<size_t>(request_count) *
                        static_cast<size_t>(verifier_state_capture_size_))
            {
                return false;
            }
            if (!cudaGDN_gpu_copy_capture_terminal_rows_from_device_lengths(
                    request_state_bank_,
                    verifier_state_capture_,
                    device_request_seq_lens,
                    request_count,
                    request_row_width,
                    verifier_state_capture_rows_,
                    verifier_state_capture_size_,
                    device_ordinal_,
                    stream))
            {
                return false;
            }
            return publishRequestZeroToLiveState(
                verifier_state_capture_size_,
                stream);
        }

        bool supportsPaddedPrefillRealLength() const override { return true; }
        bool supportsRequestLiveStateBank(int request_count, int state_size) const override
        {
            return device_state_bound_ &&
                   request_count > 0 &&
                   request_count <= request_state_bank_capacity_ &&
                   state_size > 0 &&
                   hasState(state_size) &&
                   request_state_bank_floats_ >=
                       static_cast<size_t>(request_count) *
                           static_cast<size_t>(state_size);
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

        bool resetState(void *stream)
        {
            if (!device_state_bound_ || !stream)
                return false;

            cudaGDN_gpu_set_device(device_ordinal_);
            if (gpu_state_ != request_state_bank_)
                cudaGDN_gpu_memset_zero_async(gpu_state_, state_size_, stream);
            if (secondary_gpu_state_ && secondary_state_size_ > 0 &&
                secondary_gpu_state_ != request_state_bank_)
                cudaGDN_gpu_memset_zero_async(
                    secondary_gpu_state_, secondary_state_size_, stream);
            if (request_state_bank_ && request_state_bank_floats_ > 0)
            {
                cudaGDN_gpu_memset_zero_async(
                    request_state_bank_,
                    request_state_bank_floats_,
                    stream);
            }
            return true;
        }

        bool forward(
            const float *input, const float *weight, const float *bias,
            float *output, float *conv_state,
            int seq_len, int channels, int kernel_size,
            bool apply_silu = true) override
        {
            cudaGDN_gpu_set_device(device_ordinal_);
            const int required_state_size = channels * (kernel_size - 1);
            float *live_state = requireState(
                required_state_size,
                "CUDAShortConvolution::forward");
            if (!live_state)
                return false;
            float *effective_state =
                resolveVerifierStateDestination(
                    required_state_size,
                    stream_);
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
                input, weight, bias, effective_output,
                live_state, effective_state,
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
            float *live_state = requireState(
                required_state_size,
                "CUDAShortConvolution::forwardWithEffectiveSeqLen");
            if (!live_state)
                return false;
            float *effective_state =
                resolveVerifierStateDestination(
                    required_state_size,
                    stream_);
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
                input, weight, bias, effective_output,
                live_state, effective_state,
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
                float *updated_state = state;
                float *snapshots = nullptr;
                int snapshot_rows = 0;
                if (capture_active)
                {
                    updated_state =
                        speculative_state_work_ +
                        static_cast<size_t>(request) *
                            static_cast<size_t>(required_state_size);

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
                        updated_state,
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

        /**
         * @brief Execute one native grouped variable-length request matrix.
         *
         * Live and speculative state banks are contiguous, allowing verifier
         * setup to use one device copy and the kernel launch to encode request
         * identity directly. No request is replayed through the scalar API.
         */
        bool forwardBatchedRequestsWithDeviceSeqLens(
            const float *input, const float *weight, const float *bias,
            float *output, float *conv_state,
            int seq_len, int request_count, int request_seq_len,
            int channels, int kernel_size,
            const int *device_request_seq_lens,
            bool apply_silu = true) override
        {
            (void)conv_state;
            cudaGDN_gpu_set_device(device_ordinal_);
            const int required_state_size = channels * (kernel_size - 1);
            if (!stream_ || !device_request_seq_lens ||
                seq_len <= 0 || request_count <= 0 || request_seq_len <= 0 ||
                seq_len != request_count * request_seq_len ||
                required_state_size <= 0)
            {
                LOG_ERROR("[CUDAShortConvolution] Invalid device-length request-batched shape");
                return false;
            }
            if (!ensureRequestStateBank(request_count, required_state_size))
                return false;

            const bool capture_active =
                verifier_state_capture_ != nullptr &&
                verifier_state_capture_rows_ >= request_count * request_seq_len &&
                verifier_state_capture_size_ == required_state_size;
            float *effective_states = request_state_bank_;
            if (capture_active)
            {
                const int work_floats = request_count * required_state_size;
                if (!speculative_state_work_ ||
                    speculative_state_work_size_ < work_floats)
                {
                    LOG_ERROR("[CUDAShortConvolution] Grouped verifier requires one speculative state slot per request");
                    return false;
                }
                effective_states = speculative_state_work_;
            }

            float *effective_output = output;
            const bool needs_scratch = input == output;
            const int output_floats = seq_len * channels;
            if (needs_scratch)
            {
                if (!scratchPointer() || scratchCapacity() < output_floats)
                {
                    LOG_ERROR("[CUDAShortConvolution] Grouped request scratch is too small: need "
                              << output_floats << " floats, have " << scratchCapacity());
                    return false;
                }
                effective_output = scratchPointer();
            }

            if (!cudaGDN_short_conv1d_batched(
                    input, weight, bias,
                    effective_output,
                    request_state_bank_, effective_states,
                    request_count, request_seq_len,
                    channels, kernel_size, apply_silu,
                    device_request_seq_lens,
                    capture_active ? verifier_state_capture_ : nullptr,
                    required_state_size,
                    capture_active ? verifier_state_capture_rows_ : 0,
                    device_ordinal_, stream_))
            {
                return false;
            }

            if (needs_scratch)
            {
                cudaGDN_gpu_memcpy_async(
                    output,
                    scratchPointer(),
                    static_cast<size_t>(output_floats),
                    stream_);
            }
            return true;
        }

        void bindGPUStream(ExplicitGPUStream stream) override { stream_ = stream.get(); }
        void clearGPUStreamBinding() override { stream_ = nullptr; }

        /**
         * @brief Enqueue a diagnostic copy of the resident request-state bank.
         *
         * Integration tests use this opt-in observation to prove that graph
         * replay published the same convolution history consumed by grouped
         * execution. Runtime execution never reads the host destination.
         */
        bool exportRequestStateBank(
            void *dst_host,
            int request_count,
            int state_size,
            void *stream) const
        {
            if (!dst_host || !stream ||
                !request_state_bank_ ||
                request_count <= 0 || state_size <= 0 ||
                request_count > request_state_bank_capacity_ ||
                !hasState(state_size) ||
                request_state_bank_floats_ <
                    static_cast<size_t>(request_count) *
                        static_cast<size_t>(state_size))
            {
                return false;
            }

            cudaGDN_gpu_set_device(device_ordinal_);
            cudaGDN_gpu_memcpy_d2h_async(
                static_cast<float *>(dst_host),
                request_state_bank_,
                static_cast<size_t>(request_count) *
                    static_cast<size_t>(state_size),
                stream);
            return true;
        }

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

            cudaGDN_gpu_set_device(device_ordinal_);
            if (dst_device)
            {
                auto *dst = static_cast<float *>(dst_device);
                if (stream)
                    cudaGDN_gpu_memcpy_async(dst, src, static_cast<size_t>(target_state_size), stream);
                else
                    cudaGDN_gpu_memcpy(dst, src, static_cast<size_t>(target_state_size));
            }
            else
            {
                auto *dst = static_cast<float *>(dst_host);
                if (stream)
                    cudaGDN_gpu_memcpy_d2h_async(dst, src, static_cast<size_t>(target_state_size), stream);
                else
                    cudaGDN_gpu_memcpy_d2h(dst, src, static_cast<size_t>(target_state_size));
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
            return publishLiveStateToRequestZero(
                gpu_state_,
                state_size_,
                stream);
        }

        bool importStateForSize(int target_state_size, const void *src_host, const void *src_device, void *stream) override
        {
            if (target_state_size <= 0)
                return true;
            const auto *src = static_cast<const float *>(src_host ? src_host : src_device);
            if (!src)
                return false;
            float *target_state = requireState(
                target_state_size,
                "CUDAShortConvolution::importStateForSize");
            if (!target_state)
                return false;

            cudaGDN_gpu_set_device(device_ordinal_);
            if (stream)
            {
                cudaGDN_gpu_memcpy_async(
                    target_state,
                    src,
                    static_cast<size_t>(target_state_size),
                    stream);
            }
            else
            {
                cudaGDN_gpu_memcpy(
                    target_state,
                    src,
                    static_cast<size_t>(target_state_size));
            }
            return publishLiveStateToRequestZero(
                target_state,
                target_state_size,
                stream);
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
        size_t request_state_bank_floats_ = 0;
        int request_state_bank_capacity_ = 0;
        float *bound_scratch_ = nullptr;
        int bound_scratch_size_ = 0;
        float *verifier_state_capture_ = nullptr;
        int verifier_state_capture_rows_ = 0;
        int verifier_state_capture_size_ = 0;
        float *speculative_state_work_ = nullptr;
        int speculative_state_work_size_ = 0;
        bool device_state_bound_ = false;

        /**
         * @brief Resolve one immutable cache-owned state geometry.
         *
         * Captured graph nodes retain their original pointer arguments. A
         * shared mutable "active" pointer would make correctness depend on
         * graph-construction order, so geometry lookup is side-effect free.
         */
        float *stateForSize(int required_state_size) const noexcept
        {
            if (gpu_state_ && state_size_ == required_state_size)
                return gpu_state_;
            if (secondary_gpu_state_ &&
                secondary_state_size_ == required_state_size)
            {
                return secondary_gpu_state_;
            }
            return nullptr;
        }

        bool hasState(int required_state_size) const
        {
            return stateForSize(required_state_size) != nullptr;
        }

        float *requireState(int required_state_size, const char *caller) const
        {
            if (float *state = stateForSize(required_state_size))
                return state;
            LOG_ERROR("["
                      << caller << "] Required cache-owned GPU state was not bound "
                      << "(need " << required_state_size
                      << " floats, have primary=" << state_size_
                      << " secondary=" << secondary_state_size_ << ")");
            return nullptr;
        }

        float *scratchPointer() const
        {
            return bound_scratch_;
        }

        int scratchCapacity() const
        {
            return bound_scratch_size_;
        }

        float *resolveVerifierStateDestination(
            int required_state_size,
            void *stream)
        {
            const bool verifier_capture_active =
                verifier_state_capture_ != nullptr &&
                verifier_state_capture_rows_ > 0 &&
                verifier_state_capture_size_ == required_state_size;
            if (!verifier_capture_active)
                return stateForSize(required_state_size);
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

            return speculative_state_work_;
        }

        /**
         * @brief Publish scalar live state into request zero on its producer stream.
         *
         * Single-geometry bindings alias scalar state to request zero, making
         * publication structurally complete with no operation. Distinct
         * LocalTP geometries enqueue one explicit stream-ordered device copy.
         */
        bool publishLiveStateToRequestZero(
            const float *live_state,
            int live_state_size,
            void *stream) const
        {
            if (!stream || !live_state ||
                live_state_size <= 0 ||
                !request_state_bank_ ||
                request_state_bank_capacity_ <= 0 ||
                request_state_bank_floats_ <
                    static_cast<size_t>(live_state_size))
            {
                LOG_ERROR("[CUDAShortConvolution] Cannot publish live conv state to request zero"
                          << " state_size=" << live_state_size
                          << " request_capacity=" << request_state_bank_capacity_
                          << " request_floats=" << request_state_bank_floats_);
                return false;
            }

            if (live_state == request_state_bank_)
                return true;

            cudaGDN_gpu_set_device(device_ordinal_);
            cudaGDN_gpu_memcpy_async(
                request_state_bank_,
                live_state,
                static_cast<size_t>(live_state_size),
                stream);
            return true;
        }

        /**
         * @brief Publish accepted request zero back to the scalar live owner.
         *
         * Device-resident grouped publication owns the packed request bank,
         * while scalar decode and one-request grouped prefill seed from the
         * scalar bank. Request zero is their shared logical sequence. Copying
         * its accepted bytes on the publication stream keeps those two stable
         * device allocations coherent without mutable host currentness state.
         */
        bool publishRequestZeroToLiveState(
            int live_state_size,
            void *stream) const
        {
            float *live_state = stateForSize(live_state_size);
            if (!stream ||
                !live_state ||
                !request_state_bank_ ||
                live_state_size <= 0 ||
                request_state_bank_floats_ <
                    static_cast<size_t>(live_state_size))
            {
                LOG_ERROR("[CUDAShortConvolution] Cannot publish accepted request-zero conv state"
                          << " state_size=" << live_state_size
                          << " request_floats=" << request_state_bank_floats_);
                return false;
            }

            if (live_state == request_state_bank_)
                return true;

            cudaGDN_gpu_set_device(device_ordinal_);
            cudaGDN_gpu_memcpy_async(
                live_state,
                request_state_bank_,
                static_cast<size_t>(live_state_size),
                stream);
            return true;
        }

        bool ensureRequestStateBank(int request_count, int required_state_size)
        {
            if (request_count <= 0 || required_state_size <= 0)
                return false;

            float *live_state = requireState(
                required_state_size,
                "CUDAShortConvolution::ensureRequestStateBank");
            if (!live_state)
                return false;

            const size_t required_request_floats =
                static_cast<size_t>(request_count) *
                static_cast<size_t>(required_state_size);
            if (!request_state_bank_ ||
                request_count > request_state_bank_capacity_ ||
                required_request_floats > request_state_bank_floats_)
            {
                LOG_ERROR("[CUDAShortConvolution] Cache-owned request conv-state bank is undersized"
                          << " requests=" << request_count
                          << " state_size=" << required_state_size
                          << " request_capacity=" << request_state_bank_capacity_
                          << " available_floats=" << request_state_bank_floats_);
                return false;
            }

            if (request_count == 1)
            {
                if (!stream_)
                {
                    LOG_ERROR("[CUDAShortConvolution] request conv-state publication requires an explicit stream");
                    return false;
                }
                return publishLiveStateToRequestZero(
                    live_state,
                    required_state_size,
                    stream_);
            }
            return true;
        }
    };

} // namespace llaminar2
