/**
 * @file ROCmGatedDeltaNet.h
 * @brief ROCm/HIP implementation of ITensorGatedDeltaNet
 *
 * Manages GPU-resident recurrence state internally.
 *
 * Device-pointer design: All input/output pointers passed to chunk_forward()
 * and recurrent_step() are expected to be DEVICE pointers (already on GPU).
 * The stage (GDNRecurrenceStage) handles coherence via ensureOnDevice() /
 * allocateOnDevice() before calling these methods.
 */

#pragma once

#include "../../../tensors/TensorKernels.h"
#include "../../../backends/GPUDeviceContextPool.h"
#include "../../../execution/local_execution/graph/GraphCaptureGuard.h"
#include "../../../utils/Logger.h"

extern "C"
{
    bool rocmGDN_recurrent_step(
        const float *q, const float *k, const float *v,
        const float *alpha, const float *beta_raw,
        const float *A_log, const float *dt_bias,
        float *output, float *state,
        int n_heads, int d_k, int d_v,
        bool use_qk_l2norm,
        int device_idx, void *stream);

    bool rocmGDN_chunk_forward(
        const float *Q, const float *K, const float *V,
        const float *alpha, const float *beta_raw,
        const float *A_log, const float *dt_bias,
        float *output, float *state,
        int seq_len, int n_heads, int d_k, int d_v,
        bool use_qk_l2norm,
        int device_idx, void *stream);

    bool rocmGDN_chunk_forward_effective(
        const float *Q, const float *K, const float *V,
        const float *alpha, const float *beta_raw,
        const float *A_log, const float *dt_bias,
        float *output, float *state,
        int seq_len, int n_heads, int d_k, int d_v,
        bool use_qk_l2norm,
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

        ~ROCmGatedDeltaNet()
        {
            rocmGDN_gpu_set_device(device_ordinal_);
            rocmGDN_gpu_free(gpu_state_);
            rocmGDN_gpu_free(deinterleave_scratch_);
        }

        void allocateGPUState(int state_size) override { allocateState(state_size); }
        void resetGPUState() override { resetState(); }
        bool supportsPaddedPrefillRealLength() const override { return true; }
        bool isGPUStateReady(int required_state_size) const override
        {
            return gpu_state_ != nullptr && state_size_ == required_state_size;
        }
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
                LOG_ERROR("[ROCmGatedDeltaNet] GPU malloc failed for state");
                gpu_state_ = nullptr;
                return;
            }
            void *stream = GPUDeviceContextPool::instance().getAMDContext(device_ordinal_).defaultStream();
            rocmGDN_gpu_memset_zero_async(gpu_state_, state_size, stream);
            rocmGDN_stream_synchronize(stream);
            LOG_DEBUG("[ROCmGatedDeltaNet] Allocated GPU state: " << state_size << " floats on device " << device_ordinal_);
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
            if (deinterleave_scratch_)
            {
                void *stream = GPUDeviceContextPool::instance().getAMDContext(device_ordinal_).defaultStream();
                rocmGDN_gpu_memset_zero_async(deinterleave_scratch_, deinterleave_scratch_size_, stream);
                rocmGDN_stream_synchronize(stream);
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
            rocmGDN_gpu_set_device(device_ordinal_);
            const int required_state_size = n_heads * d_k * d_v;
            if (!gpu_state_ || state_size_ != required_state_size)
            {
                if (isGraphCaptureActive())
                {
                    LOG_ERROR("[ROCmGatedDeltaNet::chunk_forward] GPU state allocation during graph capture "
                              "(need "
                              << required_state_size << " floats, have " << state_size_ << ")");
                    return false;
                }
                allocateState(required_state_size);
            }
            if (!gpu_state_)
            {
                LOG_ERROR("[ROCmGatedDeltaNet] Missing GPU recurrence state");
                return false;
            }
            float *effective_state = gpu_state_;

            // All pointers are device pointers — pass directly to HIP kernel.
            return rocmGDN_chunk_forward(
                Q, K, V, alpha, beta_raw, A_log, dt_bias,
                output, effective_state,
                seq_len, n_heads, d_k, d_v, use_qk_l2norm,
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
            rocmGDN_gpu_set_device(device_ordinal_);
            const int required_state_size = n_heads * d_k * d_v;
            if (!gpu_state_ || state_size_ != required_state_size)
            {
                if (isGraphCaptureActive())
                {
                    LOG_ERROR("[ROCmGatedDeltaNet::chunkForwardWithEffectiveSeqLen] GPU state allocation during graph capture "
                              "(need "
                              << required_state_size << " floats, have " << state_size_ << ")");
                    return false;
                }
                allocateState(required_state_size);
            }
            if (!gpu_state_)
            {
                LOG_ERROR("[ROCmGatedDeltaNet] Missing GPU recurrence state");
                return false;
            }

            return rocmGDN_chunk_forward_effective(
                Q, K, V, alpha, beta_raw, A_log, dt_bias,
                output, gpu_state_,
                seq_len, n_heads, d_k, d_v, use_qk_l2norm,
                device_effective_seq_len,
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
            if (!gpu_state_ || state_size_ != required_state_size)
            {
                if (isGraphCaptureActive())
                {
                    LOG_ERROR("[ROCmGatedDeltaNet::recurrent_step] GPU state allocation during graph capture "
                              "(need "
                              << required_state_size << " floats, have " << state_size_ << ")");
                    return false;
                }
                allocateState(required_state_size);
            }
            if (!gpu_state_)
            {
                LOG_ERROR("[ROCmGatedDeltaNet] Missing GPU recurrence state");
                return false;
            }
            float *effective_state = gpu_state_;

            // All pointers are device pointers — pass directly to HIP kernel.
            return rocmGDN_recurrent_step(
                q, k, v, alpha, beta_raw, A_log, dt_bias,
                output, effective_state,
                n_heads, d_k, d_v, use_qk_l2norm,
                device_ordinal_, stream_);
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
            else if (total > deinterleave_scratch_size_)
            {
                if (isGraphCaptureActive())
                {
                    LOG_ERROR("[ROCmGatedDeltaNet::deinterleave_qkv_device] deinterleave scratch realloc during graph capture "
                              "(need "
                              << total << " floats, have " << deinterleave_scratch_size_ << ")");
                    return false;
                }
                rocmGDN_gpu_free(deinterleave_scratch_);
                if (!rocmGDN_gpu_malloc(&deinterleave_scratch_, total))
                {
                    LOG_ERROR("[ROCmGatedDeltaNet] deinterleave scratch malloc failed"
                              << " (requested=" << (total * sizeof(float)) << " bytes"
                              << ", seq_len=" << seq_len
                              << ", n_k_heads=" << n_k_heads
                              << ", n_v_heads=" << n_v_heads
                              << ", head_dim_k=" << head_dim_k
                              << ", head_dim_v=" << head_dim_v
                              << ", previous=" << (deinterleave_scratch_size_ * sizeof(float))
                              << " bytes)");
                    deinterleave_scratch_ = nullptr;
                    deinterleave_scratch_size_ = 0;
                    return false;
                }
                deinterleave_scratch_size_ = total;
                scratch = deinterleave_scratch_;
            }
            else
            {
                scratch = deinterleave_scratch_;
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
        float *deinterleave_scratch_ = nullptr;
        size_t deinterleave_scratch_size_ = 0;
        float *bound_deinterleave_scratch_ = nullptr;
        size_t bound_deinterleave_scratch_size_ = 0;
    };

} // namespace llaminar2
