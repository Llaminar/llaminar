/**
 * @file CUDAGatedDeltaNet.h
 * @brief CUDA implementation of ITensorGatedDeltaNet
 *
 * Wraps CUDA kernels for GDN delta-rule recurrence.
 * Manages GPU-resident recurrence state internally.
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
#include "../../../backends/GPUDeviceContextPool.h"
#include "../../../execution/local_execution/graph/GraphCaptureGuard.h"
#include "../../../utils/Logger.h"

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
        static constexpr const char *WS_GDN_STATE = "gdn_state";
        static constexpr const char *WS_GDN_DEINTERLEAVE = "gdn_deinterleave_scratch";

        explicit CUDAGatedDeltaNet(int device_ordinal)
            : device_ordinal_(device_ordinal) {}

        ~CUDAGatedDeltaNet()
        {
            cudaGDN_gpu_set_device(device_ordinal_);
            cudaGDN_gpu_free(gpu_state_);
            cudaGDN_gpu_free(deinterleave_scratch_);
        }

        void allocateGPUState(int state_size) override { allocateState(state_size); }
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
                     * The CUDA recurrence state has two mirrors: gpu_state_ is
                     * consumed by captured graph replay, while dst_state points
                     * at the hybrid cache's host mirror used when a later
                     * mutation forces graph setup/rebuild.  Publication must
                     * advance both mirrors to the accepted verifier row or the
                     * next decode can rebuild from stale host state.
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
             * Device-indexed publication is intentionally device-only. The
             * accepted row pointer is GPU-resident, so this method must not
             * refresh dst_state or synchronize for host visibility. Host mirror
             * adoption, when required for diagnostics/export, must be an
             * explicit operation outside the replay hot path.
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

        bool isGPUStateReady(int required_state_size) const override
        {
            return gpu_state_ != nullptr && state_size_ == required_state_size;
        }
        bool supportsPaddedPrefillRealLength() const override { return true; }
        size_t stateBytes() const override
        {
            return state_size_ > 0 ? static_cast<size_t>(state_size_) * sizeof(float) : 0;
        }

        /// Allocate GPU state buffer for the recurrence state [n_heads * d_k * d_v]
        void allocateState(int state_size)
        {
            if (gpu_state_ && state_size_ == state_size)
                return;
            if (isGraphCaptureActive())
            {
                LOG_ERROR("[CUDAGatedDeltaNet] GPU state allocation during graph capture "
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
                LOG_ERROR("[CUDAGatedDeltaNet] GPU malloc failed for state");
                gpu_state_ = nullptr;
                return;
            }
            void *stream = GPUDeviceContextPool::instance().getNvidiaContext(device_ordinal_).defaultStream();
            cudaGDN_gpu_memset_zero_async(gpu_state_, state_size, stream);
            cudaGDN_stream_synchronize(stream);
            LOG_DEBUG("[CUDAGatedDeltaNet] Allocated GPU state: " << state_size << " floats on device " << device_ordinal_);
        }

        /// Reset GPU state to zero
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
            if (!gpu_state_ || state_size_ != required_state_size)
            {
                if (isGraphCaptureActive())
                {
                    LOG_ERROR("[CUDAGatedDeltaNet::chunk_forward] GPU state allocation during graph capture "
                              "(need "
                              << required_state_size << " floats, have " << state_size_ << ")");
                    return false;
                }
                allocateState(required_state_size);
            }
            if (!gpu_state_)
            {
                LOG_ERROR("[CUDAGatedDeltaNet] Missing GPU recurrence state");
                return false;
            }
            float *effective_state =
                prepareEffectiveStateForVerifierForward(required_state_size, stream_);
            if (!effective_state)
                return false;

            // All pointers are device pointers — pass directly to CUDA kernel.
            // No H2D/D2H copies, no scratch buffer, no stream synchronization.
            // The stage handles coherence (ensureOnDevice/allocateOnDevice).
            return cudaGDN_chunk_forward(
                Q, K, V, alpha, beta_raw, A_log, dt_bias,
                output, effective_state,
                seq_len, n_heads, d_k, d_v, use_qk_l2norm,
                verifier_state_capture_,
                verifier_state_capture_size_,
                verifier_state_capture_rows_,
                device_ordinal_, stream_);
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
            if (!gpu_state_ || state_size_ != required_state_size)
            {
                if (isGraphCaptureActive())
                {
                    LOG_ERROR("[CUDAGatedDeltaNet::chunkForwardWithEffectiveSeqLen] GPU state allocation during graph capture "
                              "(need "
                              << required_state_size << " floats, have " << state_size_ << ")");
                    return false;
                }
                allocateState(required_state_size);
            }
            if (!gpu_state_)
            {
                LOG_ERROR("[CUDAGatedDeltaNet] Missing GPU recurrence state");
                return false;
            }
            float *effective_state =
                prepareEffectiveStateForVerifierForward(required_state_size, stream_);
            if (!effective_state)
                return false;

            return cudaGDN_chunk_forward_effective(
                Q, K, V, alpha, beta_raw, A_log, dt_bias,
                output, effective_state,
                seq_len, n_heads, d_k, d_v, use_qk_l2norm,
                device_effective_seq_len,
                verifier_state_capture_,
                verifier_state_capture_size_,
                verifier_state_capture_rows_,
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
            if (!gpu_state_ || state_size_ != required_state_size)
            {
                if (isGraphCaptureActive())
                {
                    LOG_ERROR("[CUDAGatedDeltaNet::recurrent_step] GPU state allocation during graph capture "
                              "(need "
                              << required_state_size << " floats, have " << state_size_ << ")");
                    return false;
                }
                allocateState(required_state_size);
            }
            if (!gpu_state_)
            {
                LOG_ERROR("[CUDAGatedDeltaNet] Missing GPU recurrence state");
                return false;
            }
            float *effective_state =
                prepareEffectiveStateForVerifierForward(required_state_size, stream_);
            if (!effective_state)
                return false;

            // Keep one-token decode on the same row-split recurrence path as
            // verifier/prefill. The dedicated decode kernel accumulates a
            // slightly different GDN state over long generations, which can
            // flip near-tie Qwen3.6 tokens while the chunk-style verifier path
            // remains aligned with PyTorch.
            return cudaGDN_chunk_forward(
                q, k, v, alpha, beta_raw, A_log, dt_bias,
                output, effective_state,
                /*seq_len=*/1, n_heads, d_k, d_v, use_qk_l2norm,
                verifier_state_capture_,
                verifier_state_capture_size_,
                verifier_state_capture_rows_,
                device_ordinal_, stream_);
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
            // State buffer: n_heads * d_k * d_v floats (use state_size_ if known, else estimate from m,n,k)
            size_t state_bytes = (state_size_ > 0)
                                     ? static_cast<size_t>(state_size_) * sizeof(float)
                                     : static_cast<size_t>(m) * static_cast<size_t>(n) * static_cast<size_t>(k) * sizeof(float);
            if (state_bytes > 0)
                reqs.buffers.push_back({WS_GDN_STATE, state_bytes, 256, true});

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
        float *deinterleave_scratch_ = nullptr;
        size_t deinterleave_scratch_size_ = 0;
        float *bound_deinterleave_scratch_ = nullptr;
        size_t bound_deinterleave_scratch_size_ = 0;
        float *verifier_state_capture_ = nullptr;
        int verifier_state_capture_rows_ = 0;
        int verifier_state_capture_size_ = 0;
        float *speculative_state_work_ = nullptr;
        int speculative_state_work_size_ = 0;
        DeviceWorkspaceManager *workspace_ = nullptr;

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
                LOG_ERROR("[CUDAGatedDeltaNet] Speculative verifier state requires an explicit stream");
                return nullptr;
            }
            if (!speculative_state_work_ ||
                speculative_state_work_size_ != required_state_size)
            {
                LOG_ERROR("[CUDAGatedDeltaNet] Speculative verifier state workspace was not bound: need "
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
    };

} // namespace llaminar2
