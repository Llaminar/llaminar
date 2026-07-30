/**
 * @file CUDAGatedDeltaNet.h
 * @brief CUDA implementation of ITensorGatedDeltaNet
 *
 * Wraps CUDA kernels for GDN delta-rule recurrence.
 * Consumes cache-owned, persistently bound GPU recurrence state.
 *
 * Device-pointer design: All input/output pointers passed to chunk_forward()
 * and recurrent_step() are expected to be DEVICE pointers (already on GPU).
 * The stage (GDNRecurrenceStage) handles coherence via ensureOnDevice() /
 * allocateOnDevice() before calling these methods. No H2D/D2H copies are
 * performed here — the CUDA kernels operate directly on device-resident data.
 */

#pragma once

#include "../../../tensors/TensorKernels.h"
#include "../../../interfaces/IWorkspaceConsumer.h"
#include "../../../execution/local_execution/graph/GraphCaptureGuard.h"
#include "../../../utils/DebugEnv.h"
#include "../../../utils/FNV1a.h"
#include "../../../utils/Logger.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

// Forward declarations of extern "C" kernel wrappers
extern "C"
{
    bool cudaGDN_recurrent_step(
        const float *q, const float *k, const float *v,
        const float *alpha, const float *beta_raw,
        const float *A_log, const float *dt_bias,
        float *output, float *state,
        int n_heads, int d_k, int d_v,
        bool use_qk_l2norm,
        int device_idx, void *stream);

    bool cudaGDN_chunk_forward(
        const float *Q, const float *K, const float *V,
        const float *alpha, const float *beta_raw,
        const float *A_log, const float *dt_bias,
        float *output, float *state,
        int seq_len, int n_heads, int d_k, int d_v,
        bool use_qk_l2norm,
        float *state_snapshots,
        int snapshot_stride_floats,
        int max_snapshot_rows,
        int device_idx, void *stream);

    bool cudaGDN_chunk_forward_effective(
        const float *Q, const float *K, const float *V,
        const float *alpha, const float *beta_raw,
        const float *A_log, const float *dt_bias,
        float *output, float *state,
        int seq_len, int n_heads, int d_k, int d_v,
        bool use_qk_l2norm,
        const int *device_effective_seq_len,
        float *state_snapshots,
        int snapshot_stride_floats,
        int max_snapshot_rows,
        int device_idx, void *stream);

    bool cudaGDN_chunk_forward_batched_effective(
        const float *Q, const float *K, const float *V,
        const float *alpha, const float *beta_raw,
        const float *A_log, const float *dt_bias,
        float *output, float *state,
        int seq_len, int request_count, int request_seq_len,
        int n_heads, int d_k, int d_v,
        bool use_qk_l2norm,
        const int *device_effective_seq_lens,
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
    // QKV deinterleave on device
    bool cudaGDN_deinterleave_qkv(
        const float *merged, float *out_q, float *out_k, float *out_v,
        int seq_len, int n_k_heads, int n_v_heads,
        int d_k, int d_v, int global_v_offset,
        int device_idx, void *stream);
}

namespace llaminar2
{

    class CUDAGatedDeltaNet : public ITensorGatedDeltaNet, public IWorkspaceConsumer
    {
    public:
        /// Well-known workspace buffer names for GDN
        static constexpr const char *WS_GDN_DEINTERLEAVE = "gdn_deinterleave_scratch";

        explicit CUDAGatedDeltaNet(int device_ordinal)
            : device_ordinal_(device_ordinal) {}

        ~CUDAGatedDeltaNet() override = default;

        /**
         * @brief Adopt one cache-owned persistent state binding.
         *
         * Rebinding would let a captured kernel silently change addresses, so
         * even an otherwise valid second binding is rejected.
         */
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
            debugLogDeviceIndexedRestoreSamples(
                "CUDAGatedDeltaNet",
                live_state,
                verifier_state_capture_size_,
                stream);
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

        bool isGPUStateReady(int required_state_size) const override
        {
            return hasState(required_state_size);
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

        /// Reset every cache-owned GPU state bank on the producer stream.
        bool resetState(void *stream)
        {
            if (!device_state_bound_ || !stream)
                return false;

            cudaGDN_gpu_set_device(device_ordinal_);
            cudaGDN_gpu_memset_zero_async(gpu_state_, state_size_, stream);
            if (secondary_gpu_state_ && secondary_state_size_ > 0)
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

        bool chunk_forward(
            const float *Q, const float *K, const float *V,
            const float *alpha, const float *beta_raw,
            const float *A_log, const float *dt_bias,
            float *output, float *state,
            int seq_len, int n_heads, int d_k, int d_v,
            int chunk_size, bool use_qk_l2norm) override
        {
            (void)chunk_size;
            cudaGDN_gpu_set_device(device_ordinal_);
            const int required_state_size = n_heads * d_k * d_v;
            float *live_state = requireState(
                required_state_size,
                "CUDAGatedDeltaNet::chunk_forward");
            if (!live_state)
                return false;
            float *effective_state =
                prepareEffectiveStateForVerifierForward(
                    live_state,
                    required_state_size,
                    stream_);
            if (!effective_state)
                return false;

            // All pointers are device pointers — pass directly to CUDA kernel.
            // No H2D/D2H copies, no scratch buffer, no stream synchronization.
            // The stage handles coherence (ensureOnDevice/allocateOnDevice).
            const bool ok = cudaGDN_chunk_forward(
                Q, K, V, alpha, beta_raw, A_log, dt_bias,
                output, effective_state,
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
            cudaGDN_gpu_set_device(device_ordinal_);
            const int required_state_size = n_heads * d_k * d_v;
            float *live_state = requireState(
                required_state_size,
                "CUDAGatedDeltaNet::chunkForwardWithEffectiveSeqLen");
            if (!live_state)
                return false;
            float *effective_state =
                prepareEffectiveStateForVerifierForward(
                    live_state,
                    required_state_size,
                    stream_);
            if (!effective_state)
                return false;

            const bool ok = cudaGDN_chunk_forward_effective(
                Q, K, V, alpha, beta_raw, A_log, dt_bias,
                output, effective_state,
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
            cudaGDN_gpu_set_device(device_ordinal_);
            const int required_state_size = n_heads * head_dim_k * head_dim_v;
            if (seq_len <= 0 || request_count <= 0 || request_seq_len <= 0 ||
                seq_len != request_count * request_seq_len ||
                required_state_size <= 0)
            {
                LOG_ERROR("[CUDAGatedDeltaNet] Invalid request-batched shape"
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
                LOG_ERROR("[CUDAGatedDeltaNet] Request-batched verifier requires "
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
                        request_state,
                        static_cast<size_t>(required_state_size),
                        stream_);
                    request_state = work_state;

                    const int snapshot_base = request * request_seq_len;
                    snapshots =
                        verifier_state_capture_ +
                        static_cast<size_t>(snapshot_base) *
                            static_cast<size_t>(required_state_size);
                    snapshot_rows =
                        std::min(request_seq_len, verifier_state_capture_rows_ - snapshot_base);
                }

                if (!cudaGDN_chunk_forward(
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
         * @brief Run variable-length request recurrence as one grouped launch.
         *
         * The low-level grouped launch encodes request identity directly and
         * retains serial timestep order inside each request block. Verifier
         * setup copies the contiguous live bank once, then the grouped kernel
         * writes the flat request-row snapshot namespace directly.
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
            cudaGDN_gpu_set_device(device_ordinal_);
            const int required_state_size =
                n_heads * head_dim_k * head_dim_v;
            if (!stream_ || !device_request_seq_lens ||
                seq_len <= 0 || request_count <= 0 || request_seq_len <= 0 ||
                seq_len != request_count * request_seq_len ||
                required_state_size <= 0)
            {
                LOG_ERROR("[CUDAGatedDeltaNet] Invalid device-length request-batched shape");
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
                    LOG_ERROR("[CUDAGatedDeltaNet] Grouped verifier requires one speculative state slot per request");
                    return false;
                }
                cudaGDN_gpu_memcpy_async(
                    speculative_state_work_,
                    request_state_bank_,
                    static_cast<size_t>(work_floats),
                    stream_);
                effective_states = speculative_state_work_;
            }

            return cudaGDN_chunk_forward_batched_effective(
                Q, K, V, alpha, beta_raw, A_log, dt_bias,
                output, effective_states,
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
            cudaGDN_gpu_set_device(device_ordinal_);
            const int required_state_size = n_heads * d_k * d_v;
            float *live_state = requireState(
                required_state_size,
                "CUDAGatedDeltaNet::recurrent_step");
            if (!live_state)
                return false;
            float *effective_state =
                prepareEffectiveStateForVerifierForward(
                    live_state,
                    required_state_size,
                    stream_);
            if (!effective_state)
                return false;

            // Keep one-token decode on the same row-split recurrence path as
            // verifier/prefill. The dedicated decode kernel accumulates a
            // slightly different GDN state over long generations, which can
            // flip near-tie Qwen3.6 tokens while the chunk-style verifier path
            // remains aligned with PyTorch.
            const bool ok = cudaGDN_chunk_forward(
                q, k, v, alpha, beta_raw, A_log, dt_bias,
                output, effective_state,
                /*seq_len=*/1, n_heads, d_k, d_v, use_qk_l2norm,
                verifier_state_capture_,
                verifier_state_capture_size_,
                verifier_state_capture_rows_,
                device_ordinal_, stream_);
            return ok;
        }

        void setGPUStream(void *stream) override { stream_ = stream; }

        /**
         * @brief Enqueue a diagnostic copy of the resident request-state bank.
         *
         * Integration tests use this to prove state byte equality before a
         * continuation row can amplify a hidden recurrence mismatch.  Runtime
         * execution remains fully device-owned and never consumes this host
         * observation.  The caller synchronizes the explicit stream.
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
                "CUDAGatedDeltaNet::importStateForSize");
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
            cudaGDN_gpu_set_device(device_ordinal_);

            size_t q_elems = static_cast<size_t>(seq_len) * n_v_heads * head_dim_k;
            size_t k_elems = q_elems;
            size_t v_elems = static_cast<size_t>(seq_len) * n_v_heads * head_dim_v;
            size_t total = q_elems + k_elems + v_elems;

            float *scratch = bound_deinterleave_scratch_;
            if (scratch)
            {
                if (total > bound_deinterleave_scratch_size_)
                {
                    LOG_ERROR("[CUDAGatedDeltaNet] bound deinterleave workspace too small"
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
                LOG_ERROR("[CUDAGatedDeltaNet] deinterleave_qkv_device requires bound graph workspace"
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

            return cudaGDN_deinterleave_qkv(
                d_merged_qkv, d_q, d_k, d_v,
                seq_len, n_k_heads, n_v_heads,
                head_dim_k, head_dim_v, global_v_head_offset,
                device_ordinal_, stream_);
        }

        // =====================================================================
        // IWorkspaceConsumer Interface
        // =====================================================================

        WorkspaceRequirements getWorkspaceRequirements(int m, int n = 0, int k = 0) const override
        {
            WorkspaceRequirements reqs;
            // Deinterleave scratch: estimate based on typical usage (3 × seq × heads × head_dim)
            size_t scratch_bytes = (deinterleave_scratch_size_ > 0)
                                       ? deinterleave_scratch_size_ * sizeof(float)
                                       : static_cast<size_t>(m) * 3 * sizeof(float); // Conservative estimate
            if (scratch_bytes > 0)
                reqs.buffers.push_back({WS_GDN_DEINTERLEAVE, scratch_bytes, 256, false});

            return reqs;
        }

        void bindWorkspace(DeviceWorkspaceManager *workspace) override { workspace_ = workspace; }
        bool hasWorkspace() const override { return workspace_ != nullptr; }
        DeviceWorkspaceManager *getWorkspace() const override { return workspace_; }

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
        size_t deinterleave_scratch_size_ = 0;
        float *bound_deinterleave_scratch_ = nullptr;
        size_t bound_deinterleave_scratch_size_ = 0;
        float *verifier_state_capture_ = nullptr;
        int verifier_state_capture_rows_ = 0;
        int verifier_state_capture_size_ = 0;
        float *speculative_state_work_ = nullptr;
        int speculative_state_work_size_ = 0;
        DeviceWorkspaceManager *workspace_ = nullptr;
        bool device_state_bound_ = false;

        static uint64_t hashFloatBytes(const float *values, size_t count)
        {
            return fnv1a64(values, count * sizeof(float));
        }

        static size_t countNonZeroFloats(const float *values, size_t count)
        {
            size_t nonzero = 0;
            for (size_t i = 0; i < count; ++i)
            {
                if (values[i] != 0.0f)
                    ++nonzero;
            }
            return nonzero;
        }

        static double sumAbsoluteFloats(const float *values, size_t count)
        {
            double sum = 0.0;
            for (size_t i = 0; i < count; ++i)
                sum += static_cast<double>(std::fabs(values[i]));
            return sum;
        }

        void debugLogDeviceIndexedRestoreSamples(
            const char *component,
            const float *live_state,
            int live_state_size,
            void *stream) const
        {
            if (!DebugEnv::isTruthyEnv("LLAMINAR_MTP_PUBLICATION_DIAGNOSTICS") ||
                !stream ||
                !live_state ||
                !verifier_state_capture_ ||
                live_state_size <= 0 ||
                verifier_state_capture_rows_ <= 0 ||
                verifier_state_capture_size_ != live_state_size)
            {
                return;
            }

            const int rows_to_copy = std::min(verifier_state_capture_rows_, 4);
            const size_t row_floats = static_cast<size_t>(live_state_size);
            const size_t vectors = static_cast<size_t>(rows_to_copy + 1);
            std::vector<float> host(vectors * row_floats, 0.0f);

            cudaGDN_gpu_memcpy_d2h_async(
                host.data(),
                live_state,
                row_floats,
                stream);
            for (int row = 0; row < rows_to_copy; ++row)
            {
                const float *src =
                    verifier_state_capture_ +
                    static_cast<size_t>(row) *
                        static_cast<size_t>(verifier_state_capture_size_);
                cudaGDN_gpu_memcpy_d2h_async(
                    host.data() + static_cast<size_t>(row + 1) * row_floats,
                    src,
                    row_floats,
                    stream);
            }
            cudaGDN_stream_synchronize(stream);

            const auto log_summary =
                [&](const char *label, const float *values)
            {
                LOG_INFO("[MTPPublicationDiagnostics] phase=cuda_gdn_restore_sample"
                         << " component=" << component
                         << " kernel=" << static_cast<const void *>(this)
                         << " label=" << label
                         << " state_size=" << live_state_size
                         << " capture_rows=" << verifier_state_capture_rows_
                         << " hash=" << hashFloatBytes(values, row_floats)
                         << " nonzero=" << countNonZeroFloats(values, row_floats)
                         << " sum_abs=" << sumAbsoluteFloats(values, row_floats));
            };

            log_summary("live_after_restore", host.data());
            for (int row = 0; row < rows_to_copy; ++row)
            {
                const std::string label = "capture_row_" + std::to_string(row);
                log_summary(label.c_str(),
                            host.data() + static_cast<size_t>(row + 1) * row_floats);
            }
        }

        bool hasState(int required_state_size) const
        {
            return stateForSize(required_state_size) != nullptr;
        }

        /**
         * @brief Resolve one immutable cache-owned state geometry.
         *
         * CUDA graph nodes retain the pointer value recorded at capture time.
         * Geometry lookup must therefore never mutate a shared "active" host
         * pointer whose value depends on whichever graph happened to build or
         * execute most recently.
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

        float *prepareEffectiveStateForVerifierForward(
            float *live_state,
            int required_state_size,
            void *stream)
        {
            const bool verifier_capture_active =
                verifier_state_capture_ != nullptr &&
                verifier_state_capture_rows_ > 0 &&
                verifier_state_capture_size_ == required_state_size;
            if (!verifier_capture_active)
                return live_state;
            if (!stream)
            {
                LOG_ERROR("[CUDAGatedDeltaNet] Speculative verifier state requires an explicit stream");
                return nullptr;
            }
            if (!speculative_state_work_ ||
                speculative_state_work_size_ < required_state_size)
            {
                LOG_ERROR("[CUDAGatedDeltaNet] Speculative verifier state workspace was not bound: need "
                          << required_state_size << " floats, have "
                          << speculative_state_work_size_);
                return nullptr;
            }

            cudaGDN_gpu_memcpy_async(
                speculative_state_work_,
                live_state,
                static_cast<size_t>(required_state_size),
                stream);
            return speculative_state_work_;
        }

        /**
         * @brief Publish scalar live state into request zero on its producer stream.
         *
         * This copy is executable device work, so it is recorded into CUDA
         * graphs and repeated on every replay. No host-side validity flag is
         * permitted to stand in for publication because replay does not
         * re-enter the C++ wrapper that changed such a flag during capture.
         */
        bool publishLiveStateToRequestZero(
            const float *live_state,
            int live_state_size,
            void *stream) const
        {
            if (!live_state ||
                live_state_size <= 0 ||
                !request_state_bank_ ||
                request_state_bank_capacity_ <= 0 ||
                request_state_bank_floats_ <
                    static_cast<size_t>(live_state_size))
            {
                LOG_ERROR("[CUDAGatedDeltaNet] Cannot publish live recurrence state to request zero"
                          << " state_size=" << live_state_size
                          << " request_capacity=" << request_state_bank_capacity_
                          << " request_floats=" << request_state_bank_floats_);
                return false;
            }

            cudaGDN_gpu_set_device(device_ordinal_);
            if (stream)
            {
                cudaGDN_gpu_memcpy_async(
                    request_state_bank_,
                    live_state,
                    static_cast<size_t>(live_state_size),
                    stream);
            }
            else
            {
                cudaGDN_gpu_memcpy(
                    request_state_bank_,
                    live_state,
                    static_cast<size_t>(live_state_size));
            }
            return true;
        }

        /**
         * @brief Publish accepted request zero back to the scalar live owner.
         *
         * Grouped publication writes every accepted request directly into the
         * packed request bank. Request zero also represents the scalar decode
         * sequence, so its accepted bytes must become the scalar live state on
         * the same stream before a later scalar or one-request grouped graph
         * may consume that owner. Keeping both device owners mutually
         * published removes the need for a host-side currentness flag.
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
                LOG_ERROR("[CUDAGatedDeltaNet] Cannot publish accepted request-zero recurrence state"
                          << " state_size=" << live_state_size
                          << " request_floats=" << request_state_bank_floats_);
                return false;
            }

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
                "CUDAGatedDeltaNet::ensureRequestStateBank");
            if (!live_state)
                return false;

            const size_t required_request_floats =
                static_cast<size_t>(request_count) *
                static_cast<size_t>(required_state_size);
            if (!request_state_bank_ ||
                request_count > request_state_bank_capacity_ ||
                required_request_floats > request_state_bank_floats_)
            {
                LOG_ERROR("[CUDAGatedDeltaNet] Cache-owned request recurrence-state bank is undersized"
                          << " requests=" << request_count
                          << " state_size=" << required_state_size
                          << " request_capacity=" << request_state_bank_capacity_
                          << " available_floats=" << request_state_bank_floats_);
                return false;
            }

            /*
             * A scalar live bank can authoritatively seed only request zero.
             * Multi-request grouped execution owns its packed states directly;
             * accepted-row publication updates those slots in place.
             */
            if (request_count == 1)
            {
                if (!stream_)
                {
                    LOG_ERROR("[CUDAGatedDeltaNet] request recurrence-state publication requires an explicit stream");
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
