/**
 * @file ROCmGatedDeltaNet.h
 * @brief ROCm/HIP implementation of ITensorGatedDeltaNet
 *
 * Consumes cache-owned, persistently bound GPU recurrence state.
 *
 * Device-pointer design: All input/output pointers passed to chunk_forward()
 * and recurrent_step() are expected to be DEVICE pointers (already on GPU).
 * DeviceGraphExecutor and TransferEngine establish storage and producer-event
 * ordering on the stage stream before these methods are called.
 */

#pragma once

#include "../../../tensors/TensorKernels.h"
#include "../../../execution/local_execution/graph/GraphCaptureGuard.h"
#include "../../../utils/Logger.h"

#include <algorithm>

extern "C"
{
    bool rocmGDN_recurrent_step(
        const float *q, const float *k, const float *v,
        const float *alpha, const float *beta_raw,
        const float *A_log, const float *dt_bias,
        float *output, const float *initial_state, float *updated_state,
        int n_heads, int d_k, int d_v,
        bool use_qk_l2norm,
        int device_idx, void *stream);

    bool rocmGDN_chunk_forward(
        const float *Q, const float *K, const float *V,
        const float *alpha, const float *beta_raw,
        const float *A_log, const float *dt_bias,
        float *output, const float *initial_state, float *updated_state,
        int seq_len, int n_heads, int d_k, int d_v,
        bool use_qk_l2norm,
        float *state_snapshots,
        int snapshot_stride_floats,
        int max_snapshot_rows,
        int device_idx, void *stream);

    bool rocmGDN_chunk_forward_effective(
        const float *Q, const float *K, const float *V,
        const float *alpha, const float *beta_raw,
        const float *A_log, const float *dt_bias,
        float *output, const float *initial_state, float *updated_state,
        int seq_len, int n_heads, int d_k, int d_v,
        bool use_qk_l2norm,
        const int *device_effective_seq_len,
        float *state_snapshots,
        int snapshot_stride_floats,
        int max_snapshot_rows,
        int device_idx, void *stream);

    bool rocmGDN_chunk_forward_batched_effective(
        const float *Q, const float *K, const float *V,
        const float *alpha, const float *beta_raw,
        const float *A_log, const float *dt_bias,
        float *output, const float *initial_state, float *updated_state,
        int seq_len, int request_count, int request_seq_len,
        int n_heads, int d_k, int d_v,
        bool use_qk_l2norm,
        const int *device_effective_seq_lens,
        float *state_snapshots,
        int snapshot_stride_floats,
        int max_snapshot_rows,
        int device_idx, void *stream);

    // GPU memory helpers (implemented in ROCmGatedDeltaNetKernels.hip)
    void rocmGDN_gpu_memset_zero(float *ptr, size_t count);
    void rocmGDN_gpu_memset_zero_async(float *ptr, size_t count, void *stream);
    void rocmGDN_gpu_memcpy(float *dst, const float *src, size_t count);
    void rocmGDN_gpu_memcpy_async(float *dst, const float *src, size_t count, void *stream);
    void rocmGDN_gpu_memcpy_d2h(float *host_dst, const float *device_src, size_t count);
    void rocmGDN_gpu_memcpy_d2h_async(float *host_dst, const float *device_src, size_t count, void *stream);
    void rocmGDN_gpu_set_device(int ordinal);
    bool rocmGDN_gpu_publish_capture_row_from_device_index(
        float *live_state,
        float *request_zero_state,
        const float *capture,
        const int *device_row_index,
        int rows,
        int state_size,
        int device_idx,
        void *stream);
    bool rocmGDN_gpu_publish_capture_rows_from_device_indices(
        float *request_states,
        float *request_zero_live_state,
        const float *capture,
        const int *device_row_indices,
        int request_count,
        int row_index_stride,
        int rows,
        int state_size,
        int device_idx,
        void *stream);
    bool rocmGDN_gpu_publish_capture_terminal_rows_from_device_lengths(
        float *request_states,
        float *request_zero_live_state,
        const float *capture,
        const int *device_request_seq_lens,
        int request_count,
        int request_row_width,
        int rows,
        int state_size,
        int device_idx,
        void *stream);
    // QKV deinterleave on device
    bool rocmGDN_deinterleave_qkv(
        const float *merged, float *out_q, float *out_k, float *out_v,
        int seq_len, int n_k_heads, int n_v_heads,
        int d_k, int d_v, int global_v_offset,
        int device_idx, void *stream);
}

namespace llaminar2
{

    class ROCmGatedDeltaNet : public ITensorGatedDeltaNet
    {
    public:
        explicit ROCmGatedDeltaNet(int device_ordinal)
            : device_ordinal_(device_ordinal) {}

        ~ROCmGatedDeltaNet() override = default;

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

            rocmGDN_gpu_set_device(device_ordinal_);
            const float *src =
                verifier_state_capture_ +
                static_cast<size_t>(row) * static_cast<size_t>(verifier_state_capture_size_);
            if (stream)
            {
                rocmGDN_gpu_memcpy_async(
                    live_state,
                    src,
                    static_cast<size_t>(verifier_state_capture_size_),
                    stream);
            }
            else
            {
                rocmGDN_gpu_memcpy(
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

            const bool ok = rocmGDN_gpu_publish_capture_row_from_device_index(
                live_state,
                request_state_bank_,
                verifier_state_capture_,
                device_row_index,
                verifier_state_capture_rows_,
                verifier_state_capture_size_,
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

            if (!rocmGDN_gpu_publish_capture_rows_from_device_indices(
                    request_state_bank_,
                    stateForSize(verifier_state_capture_size_),
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
            return true;
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
            if (!rocmGDN_gpu_publish_capture_terminal_rows_from_device_lengths(
                    request_state_bank_,
                    stateForSize(verifier_state_capture_size_),
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
            return true;
        }

        bool supportsPaddedPrefillRealLength() const override { return true; }
        bool isGPUStateReady(int required_state_size) const override
        {
            return hasState(required_state_size);
        }
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

            rocmGDN_gpu_set_device(device_ordinal_);
            if (gpu_state_ != request_state_bank_)
                rocmGDN_gpu_memset_zero_async(gpu_state_, state_size_, stream);
            if (secondary_gpu_state_ && secondary_state_size_ > 0 &&
                secondary_gpu_state_ != request_state_bank_)
                rocmGDN_gpu_memset_zero_async(
                    secondary_gpu_state_, secondary_state_size_, stream);
            if (request_state_bank_ && request_state_bank_floats_ > 0)
            {
                rocmGDN_gpu_memset_zero_async(
                    request_state_bank_,
                    request_state_bank_floats_,
                    stream);
            }
            return true;
        }

        bool chunk_forward(
            const float *Q, const float *K, const float *V,
            const float *alpha, const float *beta_raw,
            const float *A_log, const float *dt_bias,
            float *output, float *state,
            int seq_len, int n_heads, int d_k, int d_v,
            int chunk_size, bool use_qk_l2norm) override
        {
            (void)chunk_size;
            rocmGDN_gpu_set_device(device_ordinal_);
            const int required_state_size = n_heads * d_k * d_v;
            float *live_state = requireState(
                required_state_size,
                "ROCmGatedDeltaNet::chunk_forward");
            if (!live_state)
                return false;
            float *effective_state =
                resolveVerifierStateDestination(
                    required_state_size,
                    stream_);
            if (!effective_state)
                return false;

            // All pointers are device pointers — pass directly to HIP kernel.
            const bool ok = rocmGDN_chunk_forward(
                Q, K, V, alpha, beta_raw, A_log, dt_bias,
                output, live_state, effective_state,
                seq_len, n_heads, d_k, d_v, use_qk_l2norm,
                verifier_state_capture_,
                verifier_state_capture_size_,
                verifier_state_capture_rows_,
                device_ordinal_, stream_);
            return ok;
        }

        bool chunkForwardWithEffectiveSeqLen(
            const float *Q, const float *K, const float *V,
            const float *alpha, const float *beta_raw,
            const float *A_log, const float *dt_bias,
            float *output, float *state,
            int seq_len, int n_heads, int d_k, int d_v,
            int chunk_size, bool use_qk_l2norm,
            const int *device_effective_seq_len) override
        {
            (void)chunk_size;
            rocmGDN_gpu_set_device(device_ordinal_);
            const int required_state_size = n_heads * d_k * d_v;
            float *live_state = requireState(
                required_state_size,
                "ROCmGatedDeltaNet::chunkForwardWithEffectiveSeqLen");
            if (!live_state)
                return false;

            float *effective_state =
                resolveVerifierStateDestination(
                    required_state_size,
                    stream_);
            if (!effective_state)
                return false;

            const bool ok = rocmGDN_chunk_forward_effective(
                Q, K, V, alpha, beta_raw, A_log, dt_bias,
                output, live_state, effective_state,
                seq_len, n_heads, d_k, d_v, use_qk_l2norm,
                device_effective_seq_len,
                verifier_state_capture_,
                verifier_state_capture_size_,
                verifier_state_capture_rows_,
                device_ordinal_, stream_);
            return ok;
        }

        bool chunkForwardBatchedRequests(
            const float *Q, const float *K, const float *V,
            const float *alpha, const float *beta_raw,
            const float *A_log, const float *dt_bias,
            float *output, float *state,
            int seq_len, int request_count, int request_seq_len,
            int n_heads, int head_dim_k, int head_dim_v,
            int chunk_size, bool use_qk_l2norm) override
        {
            (void)state;
            (void)chunk_size;
            rocmGDN_gpu_set_device(device_ordinal_);
            const int required_state_size = n_heads * head_dim_k * head_dim_v;
            if (seq_len <= 0 || request_count <= 0 || request_seq_len <= 0 ||
                seq_len != request_count * request_seq_len ||
                required_state_size <= 0)
            {
                LOG_ERROR("[ROCmGatedDeltaNet] Invalid request-batched shape"
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
                LOG_ERROR("[ROCmGatedDeltaNet] Request-batched verifier requires "
                          "one speculative recurrence-state work slot per request");
                return false;
            }

            const int qk_stride = n_heads * head_dim_k;
            const int v_stride = n_heads * head_dim_v;
            for (int request = 0; request < request_count; ++request)
            {
                const size_t qk_offset =
                    static_cast<size_t>(request) *
                    static_cast<size_t>(request_seq_len) *
                    static_cast<size_t>(qk_stride);
                const size_t v_offset =
                    static_cast<size_t>(request) *
                    static_cast<size_t>(request_seq_len) *
                    static_cast<size_t>(v_stride);
                float *request_state =
                    request_state_bank_ +
                    static_cast<size_t>(request) *
                        static_cast<size_t>(required_state_size);
                float *updated_request_state = request_state;
                float *snapshots = nullptr;
                int snapshot_rows = 0;
                if (capture_active)
                {
                    updated_request_state =
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

                if (!rocmGDN_chunk_forward(
                        Q + qk_offset,
                        K + qk_offset,
                        V + v_offset,
                        alpha + static_cast<size_t>(request) *
                                    static_cast<size_t>(request_seq_len) *
                                    static_cast<size_t>(n_heads),
                        beta_raw + static_cast<size_t>(request) *
                                       static_cast<size_t>(request_seq_len) *
                                       static_cast<size_t>(n_heads),
                        A_log,
                        dt_bias,
                        output + v_offset,
                        request_state,
                        updated_request_state,
                        request_seq_len, n_heads, head_dim_k, head_dim_v,
                        use_qk_l2norm,
                        snapshots,
                        required_state_size,
                        snapshot_rows,
                        device_ordinal_, stream_))
                {
                    return false;
                }
            }
            return true;
        }

        /**
         * @brief Run variable-length request recurrence as one HIP launch.
         *
         * The grouped launch encodes request identity directly while each
         * block preserves serial timestep order for its request-local state.
         * The kernel reads the canonical request bank and writes its distinct
         * transactional bank directly, so verifier entry needs no seed copy.
         */
        bool chunkForwardBatchedRequestsWithDeviceSeqLens(
            const float *Q, const float *K, const float *V,
            const float *alpha, const float *beta_raw,
            const float *A_log, const float *dt_bias,
            float *output, float *state,
            int seq_len, int request_count, int request_seq_len,
            int n_heads, int head_dim_k, int head_dim_v,
            int chunk_size, bool use_qk_l2norm,
            const int *device_request_seq_lens) override
        {
            (void)state;
            (void)chunk_size;
            rocmGDN_gpu_set_device(device_ordinal_);
            const int required_state_size =
                n_heads * head_dim_k * head_dim_v;
            if (!stream_ || !device_request_seq_lens ||
                seq_len <= 0 || request_count <= 0 || request_seq_len <= 0 ||
                seq_len != request_count * request_seq_len ||
                required_state_size <= 0)
            {
                LOG_ERROR("[ROCmGatedDeltaNet] Invalid device-length request-batched shape");
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
                    LOG_ERROR("[ROCmGatedDeltaNet] Grouped verifier requires one speculative state slot per request");
                    return false;
                }
                effective_states = speculative_state_work_;
            }

            return rocmGDN_chunk_forward_batched_effective(
                Q, K, V, alpha, beta_raw, A_log, dt_bias,
                output, request_state_bank_, effective_states,
                seq_len, request_count, request_seq_len,
                n_heads, head_dim_k, head_dim_v,
                use_qk_l2norm,
                device_request_seq_lens,
                capture_active ? verifier_state_capture_ : nullptr,
                required_state_size,
                capture_active ? verifier_state_capture_rows_ : 0,
                device_ordinal_, stream_);
        }

        bool recurrent_step(
            const float *q, const float *k, const float *v,
            const float *alpha, const float *beta_raw,
            const float *A_log, const float *dt_bias,
            float *output, float *state,
            int n_heads, int d_k, int d_v,
            bool use_qk_l2norm) override
        {
            rocmGDN_gpu_set_device(device_ordinal_);
            const int required_state_size = n_heads * d_k * d_v;
            float *live_state = requireState(
                required_state_size,
                "ROCmGatedDeltaNet::recurrent_step");
            if (!live_state)
                return false;
            float *effective_state =
                resolveVerifierStateDestination(
                    required_state_size,
                    stream_);
            if (!effective_state)
                return false;

            /*
             * Keep ordinary one-token decode on the same row-split chunk
             * kernel used by all-position MTP verification.  Publication
             * restores verifier-captured GDN rows into the live state; if the
             * next decode step uses a mathematically different recurrent-step
             * kernel, the accepted prefix can match immediately and still drift
             * on the following continuation tokens.  CUDA already follows this
             * contract, so ROCm must do the same to make state publication a
             * true replay-equivalent handoff rather than a backend-specific
             * approximation.
             */
            const bool ok = rocmGDN_chunk_forward(
                q, k, v, alpha, beta_raw, A_log, dt_bias,
                output, live_state, effective_state,
                /*seq_len=*/1, n_heads, d_k, d_v, use_qk_l2norm,
                verifier_state_capture_,
                verifier_state_capture_size_,
                verifier_state_capture_rows_,
                device_ordinal_, stream_);
            return ok;
        }

        void bindGPUStream(ExplicitGPUStream stream) override { stream_ = stream.get(); }
        void clearGPUStreamBinding() override { stream_ = nullptr; }

        /**
         * @brief Enqueue a diagnostic copy of the resident request-state bank.
         *
         * This observation API exists for integration parity tests and state
         * diagnostics only; production execution never consumes the host copy.
         * The caller owns @p dst_host and must synchronize @p stream before
         * reading it.  Shape validation prevents a diagnostic from silently
         * reading beyond the persistent request allocation.
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

            rocmGDN_gpu_set_device(device_ordinal_);
            rocmGDN_gpu_memcpy_d2h_async(
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
                "ROCmGatedDeltaNet::importStateForSize");
            if (!target_state)
                return false;

            rocmGDN_gpu_set_device(device_ordinal_);
            if (stream)
            {
                rocmGDN_gpu_memcpy_async(
                    target_state,
                    src,
                    static_cast<size_t>(target_state_size),
                    stream);
            }
            else
            {
                rocmGDN_gpu_memcpy(
                    target_state,
                    src,
                    static_cast<size_t>(target_state_size));
            }
            return publishLiveStateToRequestZero(
                target_state,
                target_state_size,
                stream);
        }

        void bindDeinterleaveWorkspace(float *scratch, size_t scratch_size) override
        {
            bound_deinterleave_scratch_ = scratch;
            bound_deinterleave_scratch_size_ = scratch_size;
        }

        bool deinterleave_qkv_device(
            const float *d_merged_qkv,
            float *&d_q, float *&d_k, float *&d_v,
            int seq_len, int n_k_heads, int n_v_heads,
            int head_dim_k, int head_dim_v, int global_v_head_offset) override
        {
            rocmGDN_gpu_set_device(device_ordinal_);

            size_t q_elems = static_cast<size_t>(seq_len) * n_v_heads * head_dim_k;
            size_t k_elems = q_elems;
            size_t v_elems = static_cast<size_t>(seq_len) * n_v_heads * head_dim_v;
            size_t total = q_elems + k_elems + v_elems;

            float *scratch = bound_deinterleave_scratch_;
            if (scratch)
            {
                if (total > bound_deinterleave_scratch_size_)
                {
                    LOG_ERROR("[ROCmGatedDeltaNet] bound deinterleave workspace too small"
                              << " (requested=" << (total * sizeof(float)) << " bytes"
                              << ", available=" << (bound_deinterleave_scratch_size_ * sizeof(float)) << " bytes"
                              << ", seq_len=" << seq_len
                              << ", n_k_heads=" << n_k_heads
                              << ", n_v_heads=" << n_v_heads
                              << ", head_dim_k=" << head_dim_k
                              << ", head_dim_v=" << head_dim_v << ")");
                    return false;
                }
            }
            else
            {
                LOG_ERROR("[ROCmGatedDeltaNet] deinterleave_qkv_device requires bound graph workspace"
                          << " (requested=" << (total * sizeof(float)) << " bytes"
                          << ", seq_len=" << seq_len
                          << ", n_k_heads=" << n_k_heads
                          << ", n_v_heads=" << n_v_heads
                          << ", head_dim_k=" << head_dim_k
                          << ", head_dim_v=" << head_dim_v << ")");
                return false;
            }

            d_q = scratch;
            d_k = scratch + q_elems;
            d_v = scratch + q_elems + k_elems;

            return rocmGDN_deinterleave_qkv(
                d_merged_qkv, d_q, d_k, d_v,
                seq_len, n_k_heads, n_v_heads,
                head_dim_k, head_dim_v, global_v_head_offset,
                device_ordinal_, stream_);
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
        float *bound_deinterleave_scratch_ = nullptr;
        size_t bound_deinterleave_scratch_size_ = 0;
        float *verifier_state_capture_ = nullptr;
        int verifier_state_capture_rows_ = 0;
        int verifier_state_capture_size_ = 0;
        float *speculative_state_work_ = nullptr;
        int speculative_state_work_size_ = 0;
        bool device_state_bound_ = false;

        bool hasState(int required_state_size) const
        {
            return stateForSize(required_state_size) != nullptr;
        }

        /**
         * @brief Resolve one immutable cache-owned state geometry.
         *
         * HIP graph nodes retain the pointer arguments recorded during
         * capture. Geometry lookup is therefore side-effect free and cannot
         * depend on whichever graph was constructed most recently.
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
                LOG_ERROR("[ROCmGatedDeltaNet] Speculative verifier state requires an explicit stream");
                return nullptr;
            }
            if (!speculative_state_work_ ||
                speculative_state_work_size_ < required_state_size)
            {
                LOG_ERROR("[ROCmGatedDeltaNet] Speculative verifier state workspace was not bound: need "
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
                LOG_ERROR("[ROCmGatedDeltaNet] Cannot publish live recurrence state to request zero"
                          << " state_size=" << live_state_size
                          << " request_capacity=" << request_state_bank_capacity_
                          << " request_floats=" << request_state_bank_floats_);
                return false;
            }

            if (live_state == request_state_bank_)
                return true;

            rocmGDN_gpu_set_device(device_ordinal_);
            rocmGDN_gpu_memcpy_async(
                request_state_bank_,
                live_state,
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
                "ROCmGatedDeltaNet::ensureRequestStateBank");
            if (!live_state)
                return false;

            const size_t required_request_floats =
                static_cast<size_t>(request_count) *
                static_cast<size_t>(required_state_size);
            if (!request_state_bank_ ||
                request_count > request_state_bank_capacity_ ||
                required_request_floats > request_state_bank_floats_)
            {
                LOG_ERROR("[ROCmGatedDeltaNet] Cache-owned request recurrence-state bank is undersized"
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
                    LOG_ERROR("[ROCmGatedDeltaNet] request recurrence-state publication requires an explicit stream");
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
