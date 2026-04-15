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
#include "../../../backends/GPUDeviceContextPool.h"
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
        int device_idx, void *stream);

    // GPU memory helpers (implemented in CUDAGatedDeltaNetKernels.cu)
    bool cudaGDN_gpu_malloc(float **ptr, size_t count);
    void cudaGDN_gpu_free(float *ptr);
    void cudaGDN_gpu_memset_zero(float *ptr, size_t count);
    void cudaGDN_gpu_memset_zero_async(float *ptr, size_t count, void *stream);
    void cudaGDN_gpu_set_device(int ordinal);
    void cudaGDN_stream_synchronize(void *stream);

    // QKV deinterleave on device
    bool cudaGDN_deinterleave_qkv(
        const float *merged, float *out_q, float *out_k, float *out_v,
        int seq_len, int n_k_heads, int n_v_heads,
        int d_k, int d_v, int global_v_offset,
        int device_idx, void *stream);
}

namespace llaminar2
{

    class CUDAGatedDeltaNet : public ITensorGatedDeltaNet
    {
    public:
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

        /// Allocate GPU state buffer for the recurrence state [n_heads * d_k * d_v]
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
            float *effective_state = gpu_state_ ? gpu_state_ : state;

            // All pointers are device pointers — pass directly to CUDA kernel.
            // No H2D/D2H copies, no scratch buffer, no stream synchronization.
            // The stage handles coherence (ensureOnDevice/allocateOnDevice).
            return cudaGDN_chunk_forward(
                Q, K, V, alpha, beta_raw, A_log, dt_bias,
                output, effective_state,
                seq_len, n_heads, d_k, d_v, use_qk_l2norm,
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
            float *effective_state = gpu_state_ ? gpu_state_ : state;

            // All pointers are device pointers — pass directly to CUDA kernel.
            return cudaGDN_recurrent_step(
                q, k, v, alpha, beta_raw, A_log, dt_bias,
                output, effective_state,
                n_heads, d_k, d_v, use_qk_l2norm,
                device_ordinal_, stream_);
        }

        void setGPUStream(void *stream) override { stream_ = stream; }

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

            // Grow-only scratch allocation
            if (total > deinterleave_scratch_size_)
            {
                cudaGDN_gpu_free(deinterleave_scratch_);
                if (!cudaGDN_gpu_malloc(&deinterleave_scratch_, total))
                {
                    LOG_ERROR("[CUDAGatedDeltaNet] deinterleave scratch malloc failed");
                    deinterleave_scratch_ = nullptr;
                    deinterleave_scratch_size_ = 0;
                    return false;
                }
                deinterleave_scratch_size_ = total;
            }

            d_q = deinterleave_scratch_;
            d_k = deinterleave_scratch_ + q_elems;
            d_v = deinterleave_scratch_ + q_elems + k_elems;

            return cudaGDN_deinterleave_qkv(
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
    };

} // namespace llaminar2
