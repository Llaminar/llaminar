/**
 * @file ROCmRoPEKernelT.cpp
 * @brief ROCm RoPE kernel implementation
 *
 * Implementation of the ROCmRoPEKernelT template specializations.
 * Calls the extern "C" HIP wrapper functions defined in ROCmRoPEKernels.hip.
 *
 * OPTIMIZATIONS (v2):
 * - Pre-computed inverse frequency table cached per (head_dim, freq_base, device)
 * - Workspace-based position_ids buffer (no per-call hipMalloc/hipFree)
 * - Fused Q+K kernel launch to reduce overhead
 *
 * @author Llaminar Team
 * @date 2025-01-17
 */

#include "ROCmRoPEKernelT.h"
#include "../../../tensors/Tensors.h"
#include "../../../utils/Logger.h"
#include "../../../utils/KernelProfiler.h"
#include "../../../execution/WorkspaceDescriptor.h"
#include "../../../execution/DeviceWorkspaceManager.h"
#include <hip/hip_runtime.h>
#include <cmath>
#include <mutex>
#include <map>

// Forward declare extern "C" HIP wrappers (v2 - with inv_freq parameter)
extern "C"
{
    bool hipOps_rope_fp32_v2(
        float *Q,
        float *K,
        const float *inv_freq,
        const int *position_ids,
        int seq_len,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        int device_idx);

    bool hipOps_rope_bf16_v2(
        uint16_t *Q,
        uint16_t *K,
        const float *inv_freq,
        const int *position_ids,
        int seq_len,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        int device_idx);

    bool hipOps_rope_fp16_v2(
        uint16_t *Q,
        uint16_t *K,
        const float *inv_freq,
        const int *position_ids,
        int seq_len,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        int device_idx);

    // Decode-optimized versions (scalar position, no H2D copy)
    bool hipOps_rope_fp32_decode(
        float *Q,
        float *K,
        const float *inv_freq,
        int pos,
        int n_q_heads,
        int n_kv_heads,
        int head_dim,
        int device_idx);

    bool hipOps_rope_bf16_decode(
        uint16_t *Q,
        uint16_t *K,
        const float *inv_freq,
        int pos,
        int n_q_heads,
        int n_kv_heads,
        int head_dim,
        int device_idx);

    bool hipOps_rope_fp16_decode(
        uint16_t *Q,
        uint16_t *K,
        const float *inv_freq,
        int pos,
        int n_q_heads,
        int n_kv_heads,
        int head_dim,
        int device_idx);

    // Contiguous position versions (zero-copy: positions computed on GPU)
    bool hipOps_rope_fp32_contiguous(
        float *Q,
        float *K,
        const float *inv_freq,
        int pos_offset,
        int seq_len,
        int n_q_heads,
        int n_kv_heads,
        int head_dim,
        int device_idx);

    bool hipOps_rope_bf16_contiguous(
        uint16_t *Q,
        uint16_t *K,
        const float *inv_freq,
        int pos_offset,
        int seq_len,
        int n_q_heads,
        int n_kv_heads,
        int head_dim,
        int device_idx);

    bool hipOps_rope_fp16_contiguous(
        uint16_t *Q,
        uint16_t *K,
        const float *inv_freq,
        int pos_offset,
        int seq_len,
        int n_q_heads,
        int n_kv_heads,
        int head_dim,
        int device_idx);
}

// =========================================================================
// Inverse Frequency Cache (shared across all RoPE kernel instances)
// =========================================================================
namespace
{
    struct InvFreqCacheKey
    {
        int head_dim;
        float freq_base;
        int device_idx;

        bool operator<(const InvFreqCacheKey &other) const
        {
            if (head_dim != other.head_dim)
                return head_dim < other.head_dim;
            if (freq_base != other.freq_base)
                return freq_base < other.freq_base;
            return device_idx < other.device_idx;
        }
    };

    struct InvFreqCacheEntry
    {
        float *d_inv_freq = nullptr;
        int half_dim = 0;
    };

    std::map<InvFreqCacheKey, InvFreqCacheEntry> g_inv_freq_cache;
    std::mutex g_inv_freq_mutex;

    /**
     * @brief Get or create inverse frequency table on device
     *
     * Caches the table per (head_dim, freq_base, device_idx) combination.
     * Formula: inv_freq[i] = 1.0 / (freq_base^(2i/head_dim)) for i in [0, head_dim/2)
     */
    float *getOrCreateInvFreq(int head_dim, float freq_base, int device_idx)
    {
        InvFreqCacheKey key{head_dim, freq_base, device_idx};

        std::lock_guard<std::mutex> lock(g_inv_freq_mutex);

        auto it = g_inv_freq_cache.find(key);
        if (it != g_inv_freq_cache.end())
        {
            return it->second.d_inv_freq;
        }

        // Compute on host
        int half_dim = head_dim / 2;
        std::vector<float> h_inv_freq(half_dim);
        for (int i = 0; i < half_dim; ++i)
        {
            h_inv_freq[i] = 1.0f / std::pow(freq_base, 2.0f * i / head_dim);
        }

        // Allocate and copy to device
        hipSetDevice(device_idx);
        float *d_inv_freq = nullptr;
        hipError_t err = hipMalloc(&d_inv_freq, half_dim * sizeof(float));
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmRoPE] Failed to allocate inv_freq cache: " << hipGetErrorString(err));
            return nullptr;
        }

        err = hipMemcpy(d_inv_freq, h_inv_freq.data(), half_dim * sizeof(float), hipMemcpyHostToDevice);
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmRoPE] Failed to copy inv_freq to device: " << hipGetErrorString(err));
            hipFree(d_inv_freq);
            return nullptr;
        }

        LOG_DEBUG("[ROCmRoPE] Created inv_freq cache for head_dim=" << head_dim
                                                                    << ", freq_base=" << freq_base
                                                                    << ", device=" << device_idx);

        g_inv_freq_cache[key] = {d_inv_freq, half_dim};
        return d_inv_freq;
    }

    void clearInvFreqCacheInternal()
    {
        std::lock_guard<std::mutex> lock(g_inv_freq_mutex);
        for (auto &[key, entry] : g_inv_freq_cache)
        {
            if (entry.d_inv_freq)
            {
                hipSetDevice(key.device_idx);
                hipFree(entry.d_inv_freq);
            }
        }
        g_inv_freq_cache.clear();
        LOG_DEBUG("[ROCmRoPE] Cleared inverse frequency cache");
    }

} // anonymous namespace

namespace llaminar2
{
    namespace rocm
    {

        // =========================================================================
        // ROCmRoPEKernelT<FP32> Implementation
        // =========================================================================

        // IWorkspaceConsumer implementation
        WorkspaceRequirements ROCmRoPEKernelT<ActivationPrecision::FP32>::getWorkspaceRequirements(
            int m, int n, int k) const
        {
            (void)n;
            (void)k;

            // Position IDs buffer - m is max sequence length
            size_t pos_ids_bytes = static_cast<size_t>(m) * sizeof(int);

            WorkspaceRequirements reqs;
            reqs.buffers.push_back({
                RoPEWorkspaceBuffers::POSITION_IDS,
                pos_ids_bytes,
                256, // HIP alignment
                true // Required
            });

            return reqs;
        }

        void ROCmRoPEKernelT<ActivationPrecision::FP32>::bindWorkspace(DeviceWorkspaceManager *ws)
        {
            workspace_ = ws;
        }

        bool ROCmRoPEKernelT<ActivationPrecision::FP32>::hasWorkspace() const
        {
            return workspace_ != nullptr;
        }

        DeviceWorkspaceManager *ROCmRoPEKernelT<ActivationPrecision::FP32>::getWorkspace() const
        {
            return workspace_;
        }

        float *ROCmRoPEKernelT<ActivationPrecision::FP32>::getOrCreateInvFreq(int head_dim, float freq_base, int device_idx)
        {
            return ::getOrCreateInvFreq(head_dim, freq_base, device_idx);
        }

        void ROCmRoPEKernelT<ActivationPrecision::FP32>::clearInvFreqCache()
        {
            ::clearInvFreqCacheInternal();
        }

        bool ROCmRoPEKernelT<ActivationPrecision::FP32>::apply(
            float *data, float *output,
            const int *pos_ids,
            int batch_size, int seq_len, int head_dim, int num_heads,
            float theta_base, bool interleaved,
            const MPIContext *mpi_ctx,
            int device_idx)
        {
            (void)output;
            (void)batch_size;
            (void)interleaved; // TODO: support interleaved layout
            (void)mpi_ctx;
            int dev = (device_idx >= 0) ? device_idx : device_idx_;

            // Get cached inverse frequency table
            float *d_inv_freq = getOrCreateInvFreq(head_dim, theta_base, dev);
            if (!d_inv_freq)
            {
                LOG_ERROR("[ROCmRoPEKernelT<FP32>] Failed to get inverse frequency table");
                return false;
            }

            return hipOps_rope_fp32_v2(data, nullptr, d_inv_freq, pos_ids, seq_len, num_heads, num_heads, head_dim, dev);
        }

        bool ROCmRoPEKernelT<ActivationPrecision::FP32>::apply_typed(
            float *Q,
            float *K,
            const int *position_ids,
            int seq_len,
            int n_heads,
            int n_kv_heads,
            int head_dim,
            float rope_theta,
            int device_idx)
        {
            int dev = (device_idx >= 0) ? device_idx : device_idx_;

            // Get cached inverse frequency table
            float *d_inv_freq = getOrCreateInvFreq(head_dim, rope_theta, dev);
            if (!d_inv_freq)
            {
                LOG_ERROR("[ROCmRoPEKernelT<FP32>] Failed to get inverse frequency table");
                return false;
            }

            return hipOps_rope_fp32_v2(Q, K, d_inv_freq, position_ids, seq_len, n_heads, n_kv_heads, head_dim, dev);
        }

        bool ROCmRoPEKernelT<ActivationPrecision::FP32>::apply_tensor(
            TensorBase *Q,
            TensorBase *K,
            const int *position_ids,
            int seq_len,
            int n_heads,
            int n_kv_heads,
            int head_dim,
            float rope_theta,
            const MPIContext *mpi_ctx,
            int device_idx,
            int pos_offset)
        {
            KERNEL_PROFILE_SCOPE(KernelType::ROPE);
            (void)mpi_ctx;

            if (!Q || Q->native_type() != TensorType::FP32)
            {
                LOG_ERROR("[ROCmRoPEKernelT<FP32>] Q must be FP32Tensor");
                return false;
            }

            auto *q_fp32 = dynamic_cast<FP32Tensor *>(Q);
            FP32Tensor *k_fp32 = nullptr;
            if (K)
            {
                if (K->native_type() != TensorType::FP32)
                {
                    LOG_ERROR("[ROCmRoPEKernelT<FP32>] K must be FP32Tensor");
                    return false;
                }
                k_fp32 = dynamic_cast<FP32Tensor *>(K);
            }

            int dev = (device_idx >= 0) ? device_idx : device_idx_;
            hipSetDevice(dev);

            // Get GPU pointers - coherence is handled by GraphExecutor
            float *d_Q = static_cast<float *>(q_fp32->gpu_data_ptr());
            float *d_K = k_fp32 ? static_cast<float *>(k_fp32->gpu_data_ptr()) : nullptr;

            if (!d_Q)
            {
                LOG_ERROR("[ROCmRoPEKernelT<FP32>] Q GPU pointer is null");
                return false;
            }

            // Get cached inverse frequency table
            float *d_inv_freq = getOrCreateInvFreq(head_dim, rope_theta, dev);
            if (!d_inv_freq)
            {
                LOG_ERROR("[ROCmRoPEKernelT<FP32>] Failed to get inverse frequency table");
                return false;
            }

            // ZERO-COPY PATH: If position_ids is nullptr, use contiguous kernel
            // Positions computed on-the-fly on GPU: pos = pos_offset + seq_idx
            if (position_ids == nullptr)
            {
                return hipOps_rope_fp32_contiguous(d_Q, d_K, d_inv_freq, pos_offset, seq_len, n_heads, n_kv_heads, head_dim, dev);
            }

            // DECODE OPTIMIZATION: For seq_len=1, use scalar position to avoid H2D copy
            if (seq_len == 1)
            {
                int pos = position_ids[0];
                return hipOps_rope_fp32_decode(d_Q, d_K, d_inv_freq, pos, n_heads, n_kv_heads, head_dim, dev);
            }

            // NON-CONTIGUOUS PATH: Need to copy position_ids array (rare: batched with padding)
            int *d_position_ids = nullptr;
            size_t pos_bytes = seq_len * sizeof(int);
            bool using_workspace = false;

            if (workspace_)
            {
                d_position_ids = static_cast<int *>(workspace_->getBuffer(RoPEWorkspaceBuffers::POSITION_IDS));
                if (d_position_ids)
                {
                    using_workspace = true;
                }
            }

            if (!d_position_ids)
            {
                // Fallback: allocate temporary (slower path)
                hipError_t err = hipMalloc(&d_position_ids, pos_bytes);
                if (err != hipSuccess)
                {
                    LOG_ERROR("[ROCmRoPEKernelT<FP32>] Failed to allocate position_ids on GPU: " << hipGetErrorString(err));
                    return false;
                }
            }

            // Copy position_ids to device
            hipError_t err = hipMemcpy(d_position_ids, position_ids, pos_bytes, hipMemcpyHostToDevice);
            if (err != hipSuccess)
            {
                LOG_ERROR("[ROCmRoPEKernelT<FP32>] Failed to copy position_ids to GPU: " << hipGetErrorString(err));
                if (!using_workspace)
                    hipFree(d_position_ids);
                return false;
            }

            // Call the optimized kernel
            bool ok = hipOps_rope_fp32_v2(d_Q, d_K, d_inv_freq, d_position_ids, seq_len, n_heads, n_kv_heads, head_dim, dev);

            // Free temporary allocation if not using workspace
            if (!using_workspace && d_position_ids)
            {
                hipFree(d_position_ids);
            }

            return ok;
        }

        // =========================================================================
        // ROCmRoPEKernelT<BF16> Implementation
        // =========================================================================

        // IWorkspaceConsumer implementation
        WorkspaceRequirements ROCmRoPEKernelT<ActivationPrecision::BF16>::getWorkspaceRequirements(
            int m, int n, int k) const
        {
            (void)n;
            (void)k;

            size_t pos_ids_bytes = static_cast<size_t>(m) * sizeof(int);

            WorkspaceRequirements reqs;
            reqs.buffers.push_back({RoPEWorkspaceBuffers::POSITION_IDS,
                                    pos_ids_bytes,
                                    256,
                                    true});

            return reqs;
        }

        void ROCmRoPEKernelT<ActivationPrecision::BF16>::bindWorkspace(DeviceWorkspaceManager *ws)
        {
            workspace_ = ws;
        }

        bool ROCmRoPEKernelT<ActivationPrecision::BF16>::hasWorkspace() const
        {
            return workspace_ != nullptr;
        }

        DeviceWorkspaceManager *ROCmRoPEKernelT<ActivationPrecision::BF16>::getWorkspace() const
        {
            return workspace_;
        }

        float *ROCmRoPEKernelT<ActivationPrecision::BF16>::getOrCreateInvFreq(int head_dim, float freq_base, int device_idx)
        {
            return ::getOrCreateInvFreq(head_dim, freq_base, device_idx);
        }

        void ROCmRoPEKernelT<ActivationPrecision::BF16>::clearInvFreqCache()
        {
            ::clearInvFreqCacheInternal();
        }

        bool ROCmRoPEKernelT<ActivationPrecision::BF16>::apply_bf16(
            uint16_t *data, uint16_t *output,
            const int *pos_ids,
            int batch_size, int seq_len, int head_dim, int num_heads,
            float theta_base, int device_idx)
        {
            (void)output;
            (void)batch_size;
            int dev = (device_idx >= 0) ? device_idx : device_idx_;

            float *d_inv_freq = getOrCreateInvFreq(head_dim, theta_base, dev);
            if (!d_inv_freq)
            {
                LOG_ERROR("[ROCmRoPEKernelT<BF16>] Failed to get inverse frequency table");
                return false;
            }

            return hipOps_rope_bf16_v2(data, nullptr, d_inv_freq, pos_ids, seq_len, num_heads, num_heads, head_dim, dev);
        }

        bool ROCmRoPEKernelT<ActivationPrecision::BF16>::apply_typed(
            uint16_t *Q,
            uint16_t *K,
            const int *position_ids,
            int seq_len,
            int n_heads,
            int n_kv_heads,
            int head_dim,
            float rope_theta,
            int device_idx)
        {
            int dev = (device_idx >= 0) ? device_idx : device_idx_;

            float *d_inv_freq = getOrCreateInvFreq(head_dim, rope_theta, dev);
            if (!d_inv_freq)
            {
                LOG_ERROR("[ROCmRoPEKernelT<BF16>] Failed to get inverse frequency table");
                return false;
            }

            return hipOps_rope_bf16_v2(Q, K, d_inv_freq, position_ids, seq_len, n_heads, n_kv_heads, head_dim, dev);
        }

        bool ROCmRoPEKernelT<ActivationPrecision::BF16>::apply_tensor(
            TensorBase *Q,
            TensorBase *K,
            const int *position_ids,
            int seq_len,
            int n_heads,
            int n_kv_heads,
            int head_dim,
            float rope_theta,
            const MPIContext *mpi_ctx,
            int device_idx,
            int pos_offset)
        {
            KERNEL_PROFILE_SCOPE(KernelType::ROPE);
            (void)mpi_ctx;

            if (!Q || Q->native_type() != TensorType::BF16)
            {
                LOG_ERROR("[ROCmRoPEKernelT<BF16>] Q must be BF16Tensor");
                return false;
            }

            auto *q_bf16 = dynamic_cast<BF16Tensor *>(Q);
            BF16Tensor *k_bf16 = nullptr;
            if (K)
            {
                if (K->native_type() != TensorType::BF16)
                {
                    LOG_ERROR("[ROCmRoPEKernelT<BF16>] K must be BF16Tensor");
                    return false;
                }
                k_bf16 = dynamic_cast<BF16Tensor *>(K);
            }

            int dev = (device_idx >= 0) ? device_idx : device_idx_;
            hipSetDevice(dev);

            uint16_t *d_Q = static_cast<uint16_t *>(q_bf16->gpu_data_ptr());
            uint16_t *d_K = k_bf16 ? static_cast<uint16_t *>(k_bf16->gpu_data_ptr()) : nullptr;

            if (!d_Q)
            {
                LOG_ERROR("[ROCmRoPEKernelT<BF16>] Q GPU pointer is null");
                return false;
            }

            float *d_inv_freq = getOrCreateInvFreq(head_dim, rope_theta, dev);
            if (!d_inv_freq)
            {
                LOG_ERROR("[ROCmRoPEKernelT<BF16>] Failed to get inverse frequency table");
                return false;
            }

            // ZERO-COPY PATH: If position_ids is nullptr, use contiguous kernel
            if (position_ids == nullptr)
            {
                return hipOps_rope_bf16_contiguous(d_Q, d_K, d_inv_freq, pos_offset, seq_len, n_heads, n_kv_heads, head_dim, dev);
            }

            // DECODE OPTIMIZATION: For seq_len=1, use scalar position to avoid H2D copy
            if (seq_len == 1)
            {
                int pos = position_ids[0];
                return hipOps_rope_bf16_decode(d_Q, d_K, d_inv_freq, pos, n_heads, n_kv_heads, head_dim, dev);
            }

            // NON-CONTIGUOUS PATH: Need to copy position_ids array
            int *d_position_ids = nullptr;
            size_t pos_bytes = seq_len * sizeof(int);
            bool using_workspace = false;

            if (workspace_)
            {
                d_position_ids = static_cast<int *>(workspace_->getBuffer(RoPEWorkspaceBuffers::POSITION_IDS));
                if (d_position_ids)
                {
                    using_workspace = true;
                }
            }

            if (!d_position_ids)
            {
                hipError_t err = hipMalloc(&d_position_ids, pos_bytes);
                if (err != hipSuccess)
                {
                    LOG_ERROR("[ROCmRoPEKernelT<BF16>] Failed to allocate position_ids on GPU: " << hipGetErrorString(err));
                    return false;
                }
            }

            hipError_t err = hipMemcpy(d_position_ids, position_ids, pos_bytes, hipMemcpyHostToDevice);
            if (err != hipSuccess)
            {
                LOG_ERROR("[ROCmRoPEKernelT<BF16>] Failed to copy position_ids to GPU: " << hipGetErrorString(err));
                if (!using_workspace)
                    hipFree(d_position_ids);
                return false;
            }

            bool ok = hipOps_rope_bf16_v2(d_Q, d_K, d_inv_freq, d_position_ids, seq_len, n_heads, n_kv_heads, head_dim, dev);

            if (!using_workspace && d_position_ids)
            {
                hipFree(d_position_ids);
            }

            return ok;
        }

        // =========================================================================
        // ROCmRoPEKernelT<FP16> Implementation
        // =========================================================================

        // IWorkspaceConsumer implementation
        WorkspaceRequirements ROCmRoPEKernelT<ActivationPrecision::FP16>::getWorkspaceRequirements(
            int m, int n, int k) const
        {
            (void)n;
            (void)k;

            size_t pos_ids_bytes = static_cast<size_t>(m) * sizeof(int);

            WorkspaceRequirements reqs;
            reqs.buffers.push_back({RoPEWorkspaceBuffers::POSITION_IDS,
                                    pos_ids_bytes,
                                    256,
                                    true});

            return reqs;
        }

        void ROCmRoPEKernelT<ActivationPrecision::FP16>::bindWorkspace(DeviceWorkspaceManager *ws)
        {
            workspace_ = ws;
        }

        bool ROCmRoPEKernelT<ActivationPrecision::FP16>::hasWorkspace() const
        {
            return workspace_ != nullptr;
        }

        DeviceWorkspaceManager *ROCmRoPEKernelT<ActivationPrecision::FP16>::getWorkspace() const
        {
            return workspace_;
        }

        float *ROCmRoPEKernelT<ActivationPrecision::FP16>::getOrCreateInvFreq(int head_dim, float freq_base, int device_idx)
        {
            return ::getOrCreateInvFreq(head_dim, freq_base, device_idx);
        }

        void ROCmRoPEKernelT<ActivationPrecision::FP16>::clearInvFreqCache()
        {
            ::clearInvFreqCacheInternal();
        }

        bool ROCmRoPEKernelT<ActivationPrecision::FP16>::apply_fp16(
            uint16_t *data, uint16_t *output,
            const int *pos_ids,
            int batch_size, int seq_len, int head_dim, int num_heads,
            float theta_base, int device_idx)
        {
            (void)output;
            (void)batch_size;
            int dev = (device_idx >= 0) ? device_idx : device_idx_;

            float *d_inv_freq = getOrCreateInvFreq(head_dim, theta_base, dev);
            if (!d_inv_freq)
            {
                LOG_ERROR("[ROCmRoPEKernelT<FP16>] Failed to get inverse frequency table");
                return false;
            }

            return hipOps_rope_fp16_v2(data, nullptr, d_inv_freq, pos_ids, seq_len, num_heads, num_heads, head_dim, dev);
        }

        bool ROCmRoPEKernelT<ActivationPrecision::FP16>::apply_typed(
            uint16_t *Q,
            uint16_t *K,
            const int *position_ids,
            int seq_len,
            int n_heads,
            int n_kv_heads,
            int head_dim,
            float rope_theta,
            int device_idx)
        {
            int dev = (device_idx >= 0) ? device_idx : device_idx_;

            float *d_inv_freq = getOrCreateInvFreq(head_dim, rope_theta, dev);
            if (!d_inv_freq)
            {
                LOG_ERROR("[ROCmRoPEKernelT<FP16>] Failed to get inverse frequency table");
                return false;
            }

            return hipOps_rope_fp16_v2(Q, K, d_inv_freq, position_ids, seq_len, n_heads, n_kv_heads, head_dim, dev);
        }

        bool ROCmRoPEKernelT<ActivationPrecision::FP16>::apply_tensor(
            TensorBase *Q,
            TensorBase *K,
            const int *position_ids,
            int seq_len,
            int n_heads,
            int n_kv_heads,
            int head_dim,
            float rope_theta,
            const MPIContext *mpi_ctx,
            int device_idx,
            int pos_offset)
        {
            KERNEL_PROFILE_SCOPE(KernelType::ROPE);
            (void)mpi_ctx;

            if (!Q || Q->native_type() != TensorType::FP16)
            {
                LOG_ERROR("[ROCmRoPEKernelT<FP16>] Q must be FP16Tensor");
                return false;
            }

            auto *q_fp16 = dynamic_cast<FP16Tensor *>(Q);
            FP16Tensor *k_fp16 = nullptr;
            if (K)
            {
                if (K->native_type() != TensorType::FP16)
                {
                    LOG_ERROR("[ROCmRoPEKernelT<FP16>] K must be FP16Tensor");
                    return false;
                }
                k_fp16 = dynamic_cast<FP16Tensor *>(K);
            }

            int dev = (device_idx >= 0) ? device_idx : device_idx_;
            hipSetDevice(dev);

            uint16_t *d_Q = static_cast<uint16_t *>(q_fp16->gpu_data_ptr());
            uint16_t *d_K = k_fp16 ? static_cast<uint16_t *>(k_fp16->gpu_data_ptr()) : nullptr;

            if (!d_Q)
            {
                LOG_ERROR("[ROCmRoPEKernelT<FP16>] Q GPU pointer is null");
                return false;
            }

            float *d_inv_freq = getOrCreateInvFreq(head_dim, rope_theta, dev);
            if (!d_inv_freq)
            {
                LOG_ERROR("[ROCmRoPEKernelT<FP16>] Failed to get inverse frequency table");
                return false;
            }

            // ZERO-COPY PATH: If position_ids is nullptr, use contiguous kernel
            if (position_ids == nullptr)
            {
                return hipOps_rope_fp16_contiguous(d_Q, d_K, d_inv_freq, pos_offset, seq_len, n_heads, n_kv_heads, head_dim, dev);
            }

            // DECODE OPTIMIZATION: For seq_len=1, use scalar position to avoid H2D copy
            if (seq_len == 1)
            {
                int pos = position_ids[0];
                return hipOps_rope_fp16_decode(d_Q, d_K, d_inv_freq, pos, n_heads, n_kv_heads, head_dim, dev);
            }

            // NON-CONTIGUOUS PATH: Need to copy position_ids array
            int *d_position_ids = nullptr;
            size_t pos_bytes = seq_len * sizeof(int);
            bool using_workspace = false;

            if (workspace_)
            {
                d_position_ids = static_cast<int *>(workspace_->getBuffer(RoPEWorkspaceBuffers::POSITION_IDS));
                if (d_position_ids)
                {
                    using_workspace = true;
                }
            }

            if (!d_position_ids)
            {
                hipError_t err = hipMalloc(&d_position_ids, pos_bytes);
                if (err != hipSuccess)
                {
                    LOG_ERROR("[ROCmRoPEKernelT<FP16>] Failed to allocate position_ids on GPU: " << hipGetErrorString(err));
                    return false;
                }
            }

            hipError_t err = hipMemcpy(d_position_ids, position_ids, pos_bytes, hipMemcpyHostToDevice);
            if (err != hipSuccess)
            {
                LOG_ERROR("[ROCmRoPEKernelT<FP16>] Failed to copy position_ids to GPU: " << hipGetErrorString(err));
                if (!using_workspace)
                    hipFree(d_position_ids);
                return false;
            }

            bool ok = hipOps_rope_fp16_v2(d_Q, d_K, d_inv_freq, d_position_ids, seq_len, n_heads, n_kv_heads, head_dim, dev);

            if (!using_workspace && d_position_ids)
            {
                hipFree(d_position_ids);
            }

            return ok;
        }

    } // namespace rocm
} // namespace llaminar2
