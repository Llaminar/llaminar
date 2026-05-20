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
#include "../../../utils/ROCmKernelProfiler.h"
#include "../../../execution/local_execution/device/WorkspaceDescriptor.h"
#include "../../../execution/local_execution/device/DeviceWorkspaceManager.h"
#include "../ROCmKernelBase.h"
#include "../../../backends/rocm/HipDeviceGuard.h"
#include "../../../kernels/rope/RoPEDeviceParams.h"
#include <hip/hip_runtime.h>

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
        int rotary_dim,
        int device_idx, void *stream);

    bool hipOps_rope_bf16_v2(
        uint16_t *Q,
        uint16_t *K,
        const float *inv_freq,
        const int *position_ids,
        int seq_len,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        int rotary_dim,
        int device_idx, void *stream);

    bool hipOps_rope_fp16_v2(
        uint16_t *Q,
        uint16_t *K,
        const float *inv_freq,
        const int *position_ids,
        int seq_len,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        int rotary_dim,
        int device_idx, void *stream);

    // Decode-optimized versions (scalar position, no H2D copy)
    bool hipOps_rope_fp32_decode(
        float *Q,
        float *K,
        const float *inv_freq,
        int pos,
        int n_q_heads,
        int n_kv_heads,
        int head_dim, int rotary_dim, int device_idx, void *stream);

    bool hipOps_rope_bf16_decode(
        uint16_t *Q,
        uint16_t *K,
        const float *inv_freq,
        int pos,
        int n_q_heads,
        int n_kv_heads,
        int head_dim, int rotary_dim, int device_idx, void *stream);

    bool hipOps_rope_fp16_decode(
        uint16_t *Q,
        uint16_t *K,
        const float *inv_freq,
        int pos,
        int n_q_heads,
        int n_kv_heads,
        int head_dim, int rotary_dim, int device_idx, void *stream);

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
        int rotary_dim,
        int device_idx, void *stream,
        const llaminar2::rope::RoPEDeviceParams *device_params = nullptr);

    bool hipOps_rope_bf16_contiguous(
        uint16_t *Q,
        uint16_t *K,
        const float *inv_freq,
        int pos_offset,
        int seq_len,
        int n_q_heads,
        int n_kv_heads,
        int head_dim,
        int rotary_dim,
        int device_idx, void *stream,
        const llaminar2::rope::RoPEDeviceParams *device_params = nullptr);

    bool hipOps_rope_fp16_contiguous(
        uint16_t *Q,
        uint16_t *K,
        const float *inv_freq,
        int pos_offset,
        int seq_len,
        int n_q_heads,
        int n_kv_heads,
        int head_dim,
        int rotary_dim,
        int device_idx, void *stream,
        const llaminar2::rope::RoPEDeviceParams *device_params = nullptr);

    // Workspace-aware inverse frequency population
    bool hipOps_rope_populate_inv_freq(
        float *d_inv_freq,
        int head_dim,
        float freq_base,
        int device_idx, void *stream);
}

namespace llaminar2
{
    namespace rocm
    {

        // =========================================================================
        // ROCmRoPEKernelT<FP32> Implementation
        // =========================================================================

        ROCmRoPEKernelT<ActivationPrecision::FP32>::~ROCmRoPEKernelT()
        {
            if (h_device_params_)
            {
                hipHostFree(h_device_params_);
                h_device_params_ = nullptr;
            }
        }

        void ROCmRoPEKernelT<ActivationPrecision::FP32>::setDynamicPosOffset(int pos_offset)
        {
            if (!h_device_params_)
            {
                hipHostMalloc(&h_device_params_, sizeof(rope::RoPEDeviceParams), hipHostMallocDefault);
            }
            if (h_device_params_)
            {
                h_device_params_->pos_offset = pos_offset;
            }
        }

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
            // Inverse frequency table - allocated for worst-case head_dim
            reqs.buffers.push_back({
                RoPEWorkspaceBuffers::INV_FREQ,
                static_cast<size_t>(MAX_HALF_DIM) * sizeof(float),
                256, // HIP alignment
                true // Required
            });
            // Device params buffer for graph capture
            reqs.buffers.push_back({RoPEWorkspaceBuffers::DEVICE_PARAMS,
                                    sizeof(rope::RoPEDeviceParams), 256, true});

            return reqs;
        }

        void ROCmRoPEKernelT<ActivationPrecision::FP32>::bindWorkspace(DeviceWorkspaceManager *ws)
        {
            // Only reset inv_freq state when workspace ACTUALLY changes.
            // The graph is rebuilt every forward() call with new stage objects,
            // but the workspace pointer stays the same. Resetting unconditionally
            // forces a synchronous hipMemcpy every RoPE call, which blocks until
            // all pending GPU work (GEMM) completes — causing ~4ms overhead per call.
            if (workspace_ != ws)
            {
                inv_freq_initialized_ = false;
            }
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

        bool ROCmRoPEKernelT<ActivationPrecision::FP32>::apply_typed(
            float *Q,
            float *K,
            const int *position_ids,
            int seq_len,
            int n_heads,
            int n_kv_heads,
            int head_dim,
            float rope_theta,
            int device_idx,
            int rotary_dim)
        {
            int dev = (device_idx >= 0) ? device_idx : device_idx_;
            const int eff_rotary = (rotary_dim > 0 && rotary_dim < head_dim) ? rotary_dim : head_dim;

            if (!validateROCmWorkspaceBinding(workspace_, dev, "ROCmRoPEKernelT<FP32>"))
            {
                return false;
            }

            float *d_inv_freq = static_cast<float *>(workspace_->getBuffer(RoPEWorkspaceBuffers::INV_FREQ));
            if (!d_inv_freq)
            {
                LOG_ERROR("[ROCmRoPEKernelT<FP32>] INV_FREQ buffer not allocated in workspace");
                return false;
            }

            // Initialize inv_freq if needed
            if (!inv_freq_initialized_ || inv_freq_head_dim_ != eff_rotary || inv_freq_theta_ != rope_theta)
            {
                if (!hipOps_rope_populate_inv_freq(d_inv_freq, eff_rotary, rope_theta, dev, gpu_stream_))
                {
                    LOG_ERROR("[ROCmRoPEKernelT<FP32>] Failed to populate inv_freq");
                    return false;
                }
                inv_freq_initialized_ = true;
                inv_freq_head_dim_ = eff_rotary;
                inv_freq_theta_ = rope_theta;
            }

            // Copy position_ids from host to device workspace buffer
            // (position_ids is a host pointer; HIP kernel expects device pointer)
            int *d_position_ids = static_cast<int *>(workspace_->getBuffer(RoPEWorkspaceBuffers::POSITION_IDS));
            if (!d_position_ids)
            {
                LOG_ERROR("[ROCmRoPEKernelT<FP32>] POSITION_IDS buffer not allocated in workspace");
                return false;
            }

            size_t pos_bytes = static_cast<size_t>(seq_len) * sizeof(int);
            hipError_t err = hipMemcpyAsync(d_position_ids, position_ids, pos_bytes, hipMemcpyHostToDevice,
                                            static_cast<hipStream_t>(gpu_stream_));
            if (err != hipSuccess)
            {
                LOG_ERROR("[ROCmRoPEKernelT<FP32>] Failed to copy position_ids to GPU: " << hipGetErrorString(err));
                return false;
            }

            return hipOps_rope_fp32_v2(Q, K, d_inv_freq, d_position_ids, seq_len, n_heads, n_kv_heads, head_dim, eff_rotary, dev, gpu_stream_);
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
            const IMPIContext *mpi_ctx,
            int device_idx,
            int pos_offset,
            int rotary_dim)
        {
            ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::ROPE, static_cast<hipStream_t>(gpu_stream_));
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
            HipDeviceGuard::setDevice(dev);

            // Get GPU pointers - coherence is handled by DeviceGraphExecutor
            float *d_Q = static_cast<float *>(q_fp32->gpu_data_ptr());
            float *d_K = k_fp32 ? static_cast<float *>(k_fp32->gpu_data_ptr()) : nullptr;

            if (!d_Q)
            {
                LOG_ERROR("[ROCmRoPEKernelT<FP32>] Q GPU pointer is null");
                return false;
            }

            if (!validateROCmWorkspaceBinding(workspace_, dev, "ROCmRoPEKernelT<FP32>"))
            {
                return false;
            }

            // Effective rotary dimension: 0 means full rotation (=head_dim)
            const int eff_rotary = (rotary_dim > 0 && rotary_dim < head_dim) ? rotary_dim : head_dim;

            float *d_inv_freq = static_cast<float *>(workspace_->getBuffer(RoPEWorkspaceBuffers::INV_FREQ));
            if (!d_inv_freq)
            {
                LOG_ERROR("[ROCmRoPEKernelT<FP32>] INV_FREQ buffer not allocated in workspace");
                return false;
            }

            // Initialize inv_freq if needed (GPU-compute, fully async — no pipeline drain)
            if (!inv_freq_initialized_ || inv_freq_head_dim_ != eff_rotary || inv_freq_theta_ != rope_theta)
            {
                if (!hipOps_rope_populate_inv_freq(d_inv_freq, eff_rotary, rope_theta, dev, gpu_stream_))
                {
                    LOG_ERROR("[ROCmRoPEKernelT<FP32>] Failed to populate inv_freq");
                    return false;
                }
                inv_freq_initialized_ = true;
                inv_freq_head_dim_ = eff_rotary;
                inv_freq_theta_ = rope_theta;
            }

            const bool force_device_positions = (gpu_stream_ != nullptr && position_ids != nullptr);

            // DECODE OPTIMIZATION: For seq_len=1, use scalar position to avoid H2D copy.
            // GRAPH CAPTURE: Skip decode fast path when gpu_stream_ is set — scalar pos
            // would be frozen in the captured graph. Fall through to contiguous path which
            // uses device_params (H2D memcpy captured, re-reads from pinned memory on replay).
            if (seq_len == 1 && !force_device_positions && !gpu_stream_)
            {
                int pos = position_ids ? position_ids[0] : pos_offset;
                return hipOps_rope_fp32_decode(d_Q, d_K, d_inv_freq, pos, n_heads, n_kv_heads, head_dim, eff_rotary, dev, gpu_stream_);
            }

            // CONTIGUOUS DETECTION: Check if positions are sequential (pos_offset, pos_offset+1, ...)
            // If so, use the zero-copy contiguous kernel which computes positions on-the-fly on GPU.
            {
                bool is_contiguous = (position_ids == nullptr) && !force_device_positions;
                if (!is_contiguous && position_ids)
                {
                    is_contiguous = true;
                    for (int i = 0; i < seq_len; ++i)
                    {
                        if (position_ids[i] != pos_offset + i)
                        {
                            is_contiguous = false;
                            break;
                        }
                    }
                }
                if (is_contiguous)
                {
                    // For graph capture: use device params buffer so pos_offset can change between replays.
                    // The H2D memcpy is captured in the graph; on replay it re-reads from pinned h_device_params_.
                    const rope::RoPEDeviceParams *d_params = nullptr;
                    if (gpu_stream_ && workspace_)
                    {
                        // Lazy-allocate pinned host buffer for graph-captured H2D
                        if (!h_device_params_)
                        {
                            hipHostMalloc(&h_device_params_, sizeof(rope::RoPEDeviceParams), hipHostMallocDefault);
                        }
                        h_device_params_->pos_offset = pos_offset;

                        d_params = static_cast<const rope::RoPEDeviceParams *>(
                            workspace_->getBuffer(RoPEWorkspaceBuffers::DEVICE_PARAMS));
                        if (d_params)
                        {
                            hipMemcpyAsync(const_cast<rope::RoPEDeviceParams *>(d_params),
                                           h_device_params_,
                                           sizeof(rope::RoPEDeviceParams),
                                           hipMemcpyHostToDevice,
                                           static_cast<hipStream_t>(gpu_stream_));
                        }
                    }
                    return hipOps_rope_fp32_contiguous(d_Q, d_K, d_inv_freq, pos_offset, seq_len,
                                                       n_heads, n_kv_heads, head_dim, eff_rotary, dev, gpu_stream_, d_params);
                }
            }

            // NON-CONTIGUOUS PATH: Need to copy position_ids array (rare: batched with padding/reordering)
            int *d_position_ids = static_cast<int *>(workspace_->getBuffer(RoPEWorkspaceBuffers::POSITION_IDS));
            if (!d_position_ids)
            {
                LOG_ERROR("[ROCmRoPEKernelT<FP32>] POSITION_IDS buffer not allocated in workspace");
                return false;
            }

            // Copy position_ids to device (async to avoid pipeline drain)
            size_t pos_bytes = seq_len * sizeof(int);
            hipError_t err = hipMemcpyAsync(d_position_ids, position_ids, pos_bytes, hipMemcpyHostToDevice, static_cast<hipStream_t>(gpu_stream_));
            if (err != hipSuccess)
            {
                LOG_ERROR("[ROCmRoPEKernelT<FP32>] Failed to copy position_ids to GPU: " << hipGetErrorString(err));
                return false;
            }

            // Call the optimized kernel
            return hipOps_rope_fp32_v2(d_Q, d_K, d_inv_freq, d_position_ids, seq_len, n_heads, n_kv_heads, head_dim, eff_rotary, dev, gpu_stream_);
        }

        // =========================================================================
        // ROCmRoPEKernelT<BF16> Implementation
        // =========================================================================

        ROCmRoPEKernelT<ActivationPrecision::BF16>::~ROCmRoPEKernelT()
        {
            if (h_device_params_)
            {
                hipHostFree(h_device_params_);
                h_device_params_ = nullptr;
            }
        }

        void ROCmRoPEKernelT<ActivationPrecision::BF16>::setDynamicPosOffset(int pos_offset)
        {
            if (!h_device_params_)
            {
                hipHostMalloc(&h_device_params_, sizeof(rope::RoPEDeviceParams), hipHostMallocDefault);
            }
            if (h_device_params_)
            {
                h_device_params_->pos_offset = pos_offset;
            }
        }

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
            // Inverse frequency table - allocated for worst-case head_dim
            reqs.buffers.push_back({RoPEWorkspaceBuffers::INV_FREQ,
                                    static_cast<size_t>(MAX_HALF_DIM) * sizeof(float),
                                    256,
                                    true});
            // Device params buffer for graph capture
            reqs.buffers.push_back({RoPEWorkspaceBuffers::DEVICE_PARAMS,
                                    sizeof(rope::RoPEDeviceParams), 256, true});

            return reqs;
        }

        void ROCmRoPEKernelT<ActivationPrecision::BF16>::bindWorkspace(DeviceWorkspaceManager *ws)
        {
            // Only reset inv_freq state when workspace ACTUALLY changes.
            // See FP32 bindWorkspace() for detailed explanation.
            if (workspace_ != ws)
            {
                inv_freq_initialized_ = false;
            }
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

        bool ROCmRoPEKernelT<ActivationPrecision::BF16>::apply_typed(
            uint16_t *Q,
            uint16_t *K,
            const int *position_ids,
            int seq_len,
            int n_heads,
            int n_kv_heads,
            int head_dim,
            float rope_theta,
            int device_idx,
            int rotary_dim)
        {
            int dev = (device_idx >= 0) ? device_idx : device_idx_;
            const int eff_rotary = (rotary_dim > 0 && rotary_dim < head_dim) ? rotary_dim : head_dim;

            if (!validateROCmWorkspaceBinding(workspace_, dev, "ROCmRoPEKernelT<BF16>"))
            {
                return false;
            }

            float *d_inv_freq = static_cast<float *>(workspace_->getBuffer(RoPEWorkspaceBuffers::INV_FREQ));
            if (!d_inv_freq)
            {
                LOG_ERROR("[ROCmRoPEKernelT<BF16>] INV_FREQ buffer not allocated in workspace");
                return false;
            }

            // Initialize inv_freq if needed
            if (!inv_freq_initialized_ || inv_freq_head_dim_ != eff_rotary || inv_freq_theta_ != rope_theta)
            {
                if (!hipOps_rope_populate_inv_freq(d_inv_freq, eff_rotary, rope_theta, dev, gpu_stream_))
                {
                    LOG_ERROR("[ROCmRoPEKernelT<BF16>] Failed to populate inv_freq");
                    return false;
                }
                inv_freq_initialized_ = true;
                inv_freq_head_dim_ = eff_rotary;
                inv_freq_theta_ = rope_theta;
            }

            // Copy position_ids from host to device workspace buffer
            // (position_ids is a host pointer; HIP kernel expects device pointer)
            int *d_position_ids = static_cast<int *>(workspace_->getBuffer(RoPEWorkspaceBuffers::POSITION_IDS));
            if (!d_position_ids)
            {
                LOG_ERROR("[ROCmRoPEKernelT<BF16>] POSITION_IDS buffer not allocated in workspace");
                return false;
            }

            size_t pos_bytes = static_cast<size_t>(seq_len) * sizeof(int);
            hipError_t err = hipMemcpyAsync(d_position_ids, position_ids, pos_bytes, hipMemcpyHostToDevice,
                                            static_cast<hipStream_t>(gpu_stream_));
            if (err != hipSuccess)
            {
                LOG_ERROR("[ROCmRoPEKernelT<BF16>] Failed to copy position_ids to GPU: " << hipGetErrorString(err));
                return false;
            }

            return hipOps_rope_bf16_v2(Q, K, d_inv_freq, d_position_ids, seq_len, n_heads, n_kv_heads, head_dim, eff_rotary, dev, gpu_stream_);
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
            const IMPIContext *mpi_ctx,
            int device_idx,
            int pos_offset,
            int rotary_dim)
        {
            ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::ROPE, static_cast<hipStream_t>(gpu_stream_));
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
            HipDeviceGuard::setDevice(dev);

            uint16_t *d_Q = static_cast<uint16_t *>(q_bf16->gpu_data_ptr());
            uint16_t *d_K = k_bf16 ? static_cast<uint16_t *>(k_bf16->gpu_data_ptr()) : nullptr;

            if (!d_Q)
            {
                LOG_ERROR("[ROCmRoPEKernelT<BF16>] Q GPU pointer is null");
                return false;
            }

            if (!validateROCmWorkspaceBinding(workspace_, dev, "ROCmRoPEKernelT<BF16>"))
            {
                return false;
            }

            // Effective rotary dimension: 0 means full rotation (=head_dim)
            const int eff_rotary = (rotary_dim > 0 && rotary_dim < head_dim) ? rotary_dim : head_dim;

            float *d_inv_freq = static_cast<float *>(workspace_->getBuffer(RoPEWorkspaceBuffers::INV_FREQ));
            if (!d_inv_freq)
            {
                LOG_ERROR("[ROCmRoPEKernelT<BF16>] INV_FREQ buffer not allocated in workspace");
                return false;
            }

            // Initialize inv_freq if needed
            if (!inv_freq_initialized_ || inv_freq_head_dim_ != eff_rotary || inv_freq_theta_ != rope_theta)
            {
                if (!hipOps_rope_populate_inv_freq(d_inv_freq, eff_rotary, rope_theta, dev, gpu_stream_))
                {
                    LOG_ERROR("[ROCmRoPEKernelT<BF16>] Failed to populate inv_freq");
                    return false;
                }
                inv_freq_initialized_ = true;
                inv_freq_head_dim_ = eff_rotary;
                inv_freq_theta_ = rope_theta;
            }

            // DECODE OPTIMIZATION: For seq_len=1, use scalar position to avoid H2D copy
            const bool force_device_positions = (gpu_stream_ != nullptr && position_ids != nullptr);

            // GRAPH CAPTURE: Skip decode fast path when gpu_stream_ is set — scalar pos frozen in graph.
            if (seq_len == 1 && !force_device_positions && !gpu_stream_)
            {
                int pos = position_ids ? position_ids[0] : pos_offset;
                return hipOps_rope_bf16_decode(d_Q, d_K, d_inv_freq, pos, n_heads, n_kv_heads, head_dim, eff_rotary, dev, gpu_stream_);
            }

            // CONTIGUOUS DETECTION: Avoid synchronous hipMemcpy pipeline drain
            {
                bool is_contiguous = (position_ids == nullptr) && !force_device_positions;
                if (!is_contiguous && position_ids)
                {
                    is_contiguous = true;
                    for (int i = 0; i < seq_len; ++i)
                    {
                        if (position_ids[i] != pos_offset + i)
                        {
                            is_contiguous = false;
                            break;
                        }
                    }
                }
                if (is_contiguous)
                {
                    // For graph capture: use device params buffer so pos_offset can change between replays.
                    const rope::RoPEDeviceParams *d_params = nullptr;
                    if (gpu_stream_ && workspace_)
                    {
                        if (!h_device_params_)
                        {
                            hipHostMalloc(&h_device_params_, sizeof(rope::RoPEDeviceParams), hipHostMallocDefault);
                        }
                        h_device_params_->pos_offset = pos_offset;

                        d_params = static_cast<const rope::RoPEDeviceParams *>(
                            workspace_->getBuffer(RoPEWorkspaceBuffers::DEVICE_PARAMS));
                        if (d_params)
                        {
                            hipMemcpyAsync(const_cast<rope::RoPEDeviceParams *>(d_params),
                                           h_device_params_,
                                           sizeof(rope::RoPEDeviceParams),
                                           hipMemcpyHostToDevice,
                                           static_cast<hipStream_t>(gpu_stream_));
                        }
                    }
                    return hipOps_rope_bf16_contiguous(d_Q, d_K, d_inv_freq, pos_offset, seq_len,
                                                       n_heads, n_kv_heads, head_dim, eff_rotary, dev, gpu_stream_, d_params);
                }
            }

            // NON-CONTIGUOUS PATH: Need to copy position_ids array
            // Workspace is already verified above, just get the buffer
            int *d_position_ids = static_cast<int *>(workspace_->getBuffer(RoPEWorkspaceBuffers::POSITION_IDS));
            if (!d_position_ids)
            {
                LOG_ERROR("[ROCmRoPEKernelT<BF16>] POSITION_IDS buffer not allocated in workspace");
                return false;
            }

            size_t pos_bytes = seq_len * sizeof(int);
            hipError_t err = hipMemcpyAsync(d_position_ids, position_ids, pos_bytes, hipMemcpyHostToDevice, static_cast<hipStream_t>(gpu_stream_));
            if (err != hipSuccess)
            {
                LOG_ERROR("[ROCmRoPEKernelT<BF16>] Failed to copy position_ids to GPU: " << hipGetErrorString(err));
                return false;
            }

            return hipOps_rope_bf16_v2(d_Q, d_K, d_inv_freq, d_position_ids, seq_len, n_heads, n_kv_heads, head_dim, eff_rotary, dev, gpu_stream_);
        }

        // =========================================================================
        // ROCmRoPEKernelT<FP16> Implementation
        // =========================================================================

        ROCmRoPEKernelT<ActivationPrecision::FP16>::~ROCmRoPEKernelT()
        {
            if (h_device_params_)
            {
                hipHostFree(h_device_params_);
                h_device_params_ = nullptr;
            }
        }

        void ROCmRoPEKernelT<ActivationPrecision::FP16>::setDynamicPosOffset(int pos_offset)
        {
            if (!h_device_params_)
            {
                hipHostMalloc(&h_device_params_, sizeof(rope::RoPEDeviceParams), hipHostMallocDefault);
            }
            if (h_device_params_)
            {
                h_device_params_->pos_offset = pos_offset;
            }
        }

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
            // Inverse frequency table - allocated for worst-case head_dim
            reqs.buffers.push_back({RoPEWorkspaceBuffers::INV_FREQ,
                                    static_cast<size_t>(MAX_HALF_DIM) * sizeof(float),
                                    256,
                                    true});
            // Device params buffer for graph capture
            reqs.buffers.push_back({RoPEWorkspaceBuffers::DEVICE_PARAMS,
                                    sizeof(rope::RoPEDeviceParams), 256, true});

            return reqs;
        }

        void ROCmRoPEKernelT<ActivationPrecision::FP16>::bindWorkspace(DeviceWorkspaceManager *ws)
        {
            // Only reset inv_freq state when workspace ACTUALLY changes.
            // See FP32 bindWorkspace() for detailed explanation.
            if (workspace_ != ws)
            {
                inv_freq_initialized_ = false;
            }
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

        bool ROCmRoPEKernelT<ActivationPrecision::FP16>::apply_typed(
            uint16_t *Q,
            uint16_t *K,
            const int *position_ids,
            int seq_len,
            int n_heads,
            int n_kv_heads,
            int head_dim,
            float rope_theta,
            int device_idx,
            int rotary_dim)
        {
            int dev = (device_idx >= 0) ? device_idx : device_idx_;
            const int eff_rotary = (rotary_dim > 0 && rotary_dim < head_dim) ? rotary_dim : head_dim;

            if (!validateROCmWorkspaceBinding(workspace_, dev, "ROCmRoPEKernelT<FP16>"))
            {
                return false;
            }

            float *d_inv_freq = static_cast<float *>(workspace_->getBuffer(RoPEWorkspaceBuffers::INV_FREQ));
            if (!d_inv_freq)
            {
                LOG_ERROR("[ROCmRoPEKernelT<FP16>] INV_FREQ buffer not allocated in workspace");
                return false;
            }

            // Initialize inv_freq if needed
            if (!inv_freq_initialized_ || inv_freq_head_dim_ != eff_rotary || inv_freq_theta_ != rope_theta)
            {
                if (!hipOps_rope_populate_inv_freq(d_inv_freq, eff_rotary, rope_theta, dev, gpu_stream_))
                {
                    LOG_ERROR("[ROCmRoPEKernelT<FP16>] Failed to populate inv_freq");
                    return false;
                }
                inv_freq_initialized_ = true;
                inv_freq_head_dim_ = eff_rotary;
                inv_freq_theta_ = rope_theta;
            }

            // Copy position_ids from host to device workspace buffer
            // (position_ids is a host pointer; HIP kernel expects device pointer)
            int *d_position_ids = static_cast<int *>(workspace_->getBuffer(RoPEWorkspaceBuffers::POSITION_IDS));
            if (!d_position_ids)
            {
                LOG_ERROR("[ROCmRoPEKernelT<FP16>] POSITION_IDS buffer not allocated in workspace");
                return false;
            }

            size_t pos_bytes = static_cast<size_t>(seq_len) * sizeof(int);
            hipError_t err = hipMemcpyAsync(d_position_ids, position_ids, pos_bytes, hipMemcpyHostToDevice,
                                            static_cast<hipStream_t>(gpu_stream_));
            if (err != hipSuccess)
            {
                LOG_ERROR("[ROCmRoPEKernelT<FP16>] Failed to copy position_ids to GPU: " << hipGetErrorString(err));
                return false;
            }

            return hipOps_rope_fp16_v2(Q, K, d_inv_freq, d_position_ids, seq_len, n_heads, n_kv_heads, head_dim, eff_rotary, dev, gpu_stream_);
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
            const IMPIContext *mpi_ctx,
            int device_idx,
            int pos_offset,
            int rotary_dim)
        {
            ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::ROPE, static_cast<hipStream_t>(gpu_stream_));
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
            HipDeviceGuard::setDevice(dev);

            uint16_t *d_Q = static_cast<uint16_t *>(q_fp16->gpu_data_ptr());
            uint16_t *d_K = k_fp16 ? static_cast<uint16_t *>(k_fp16->gpu_data_ptr()) : nullptr;

            if (!d_Q)
            {
                LOG_ERROR("[ROCmRoPEKernelT<FP16>] Q GPU pointer is null");
                return false;
            }

            if (!validateROCmWorkspaceBinding(workspace_, dev, "ROCmRoPEKernelT<FP16>"))
            {
                return false;
            }

            // Effective rotary dimension: 0 means full rotation (=head_dim)
            const int eff_rotary = (rotary_dim > 0 && rotary_dim < head_dim) ? rotary_dim : head_dim;

            float *d_inv_freq = static_cast<float *>(workspace_->getBuffer(RoPEWorkspaceBuffers::INV_FREQ));
            if (!d_inv_freq)
            {
                LOG_ERROR("[ROCmRoPEKernelT<FP16>] INV_FREQ buffer not allocated in workspace");
                return false;
            }

            // Initialize inv_freq if needed
            if (!inv_freq_initialized_ || inv_freq_head_dim_ != eff_rotary || inv_freq_theta_ != rope_theta)
            {
                if (!hipOps_rope_populate_inv_freq(d_inv_freq, eff_rotary, rope_theta, dev, gpu_stream_))
                {
                    LOG_ERROR("[ROCmRoPEKernelT<FP16>] Failed to populate inv_freq");
                    return false;
                }
                inv_freq_initialized_ = true;
                inv_freq_head_dim_ = eff_rotary;
                inv_freq_theta_ = rope_theta;
            }

            // DECODE OPTIMIZATION: For seq_len=1, use scalar position to avoid H2D copy
            const bool force_device_positions = (gpu_stream_ != nullptr && position_ids != nullptr);

            // GRAPH CAPTURE: Skip decode fast path when gpu_stream_ is set — scalar pos frozen in graph.
            if (seq_len == 1 && !force_device_positions && !gpu_stream_)
            {
                int pos = position_ids ? position_ids[0] : pos_offset;
                return hipOps_rope_fp16_decode(d_Q, d_K, d_inv_freq, pos, n_heads, n_kv_heads, head_dim, eff_rotary, dev, gpu_stream_);
            }

            // CONTIGUOUS DETECTION: Avoid synchronous hipMemcpy pipeline drain
            {
                bool is_contiguous = (position_ids == nullptr) && !force_device_positions;
                if (!is_contiguous && position_ids)
                {
                    is_contiguous = true;
                    for (int i = 0; i < seq_len; ++i)
                    {
                        if (position_ids[i] != pos_offset + i)
                        {
                            is_contiguous = false;
                            break;
                        }
                    }
                }
                if (is_contiguous)
                {
                    // For graph capture: use device params buffer so pos_offset can change between replays.
                    const rope::RoPEDeviceParams *d_params = nullptr;
                    if (gpu_stream_ && workspace_)
                    {
                        if (!h_device_params_)
                        {
                            hipHostMalloc(&h_device_params_, sizeof(rope::RoPEDeviceParams), hipHostMallocDefault);
                        }
                        h_device_params_->pos_offset = pos_offset;

                        d_params = static_cast<const rope::RoPEDeviceParams *>(
                            workspace_->getBuffer(RoPEWorkspaceBuffers::DEVICE_PARAMS));
                        if (d_params)
                        {
                            hipMemcpyAsync(const_cast<rope::RoPEDeviceParams *>(d_params),
                                           h_device_params_,
                                           sizeof(rope::RoPEDeviceParams),
                                           hipMemcpyHostToDevice,
                                           static_cast<hipStream_t>(gpu_stream_));
                        }
                    }
                    return hipOps_rope_fp16_contiguous(d_Q, d_K, d_inv_freq, pos_offset, seq_len,
                                                       n_heads, n_kv_heads, head_dim, eff_rotary, dev, gpu_stream_, d_params);
                }
            }

            // NON-CONTIGUOUS PATH: Need to copy position_ids array
            // Workspace is already verified above, just get the buffer
            int *d_position_ids = static_cast<int *>(workspace_->getBuffer(RoPEWorkspaceBuffers::POSITION_IDS));
            if (!d_position_ids)
            {
                LOG_ERROR("[ROCmRoPEKernelT<FP16>] POSITION_IDS buffer not allocated in workspace");
                return false;
            }

            size_t pos_bytes = seq_len * sizeof(int);
            hipError_t err = hipMemcpyAsync(d_position_ids, position_ids, pos_bytes, hipMemcpyHostToDevice, static_cast<hipStream_t>(gpu_stream_));
            if (err != hipSuccess)
            {
                LOG_ERROR("[ROCmRoPEKernelT<FP16>] Failed to copy position_ids to GPU: " << hipGetErrorString(err));
                return false;
            }

            return hipOps_rope_fp16_v2(d_Q, d_K, d_inv_freq, d_position_ids, seq_len, n_heads, n_kv_heads, head_dim, eff_rotary, dev, gpu_stream_);
        }

    } // namespace rocm
} // namespace llaminar2
