/**
 * @file CUDAFlashAttentionKernelT.cpp
 * @brief C++ implementation of CUDA Flash Attention kernel methods
 *
 * Implements ITensorAttention interface by delegating to CUDA kernels
 * defined in CUDAFlashAttentionKernels.cu.
 *
 * @author David Sanftenberg
 */

#include "CUDAFlashAttentionKernelT.h"
#include "../../../backends/IWorkerGPUContext.h"
#include "../../../execution/local_execution/device/DeviceWorkspaceManager.h"
#include "../../../execution/local_execution/device/WorkspaceDescriptor.h"
#include "../../../tensors/BlockStructures.h"
#include "../../../tensors/FP16Utils.h"
#include "../../../utils/Logger.h"
#include "../../../utils/CUDAKernelProfiler.h"
#include "../../attention/AttentionDeviceParams.h"
#include <cuda_runtime_api.h>
#include <cstring>
#include <stdexcept>
#include <vector>

// Extern "C" declarations for CUDA kernel wrappers
extern "C"
{
    // FA2-style pipelined prefill with WMMA (Ampere SM >= 8.0)
    // Supports head_dim=64 (6 consumer warps) and head_dim=128 (4 consumer warps)
    int cudaFlashAttn_prefill_fa2(
        const float *Q, const float *K, const float *V, float *O,
        int batch_size, int seq_len, int kv_len,
        int n_heads, int n_kv_heads, int head_dim,
        bool causal, int window_size, int position_offset,
        const llaminar2::attention::AttentionDeviceParams *device_params,
        const float *mask,
        void *stream);

    // Flash Decoding for single-token decode with split-K parallelism
    int cudaFlashAttn_decode_fp32(
        const float *Q, const float *K_cache, const float *V_cache, float *O,
        float *O_partial, float *m_partial, float *l_partial,
        int batch_size, int kv_len,
        int n_heads, int n_kv_heads, int head_dim,
        int num_splits,
        const llaminar2::attention::AttentionDeviceParams *device_params,
        void *stream);

    int cudaFlashAttn_allocWorkspace(
        void **partial_output, void **partial_m, void **partial_l,
        int batch_size, int n_heads, int head_dim, int num_splits);

    void cudaFlashAttn_freeWorkspace(void *partial_output, void *partial_m, void *partial_l);

    int cudaFlashAttn_setDevice(int device_idx);
    int cudaFlashAttn_synchronize();
}

namespace llaminar2
{
    namespace cuda
    {
        // Default number of splits for Flash Decoding
        constexpr int DEFAULT_NUM_SPLITS = 8;

        // =====================================================================
        // FP32 Specialization Implementation
        // =====================================================================

        CUDAFlashAttentionKernelT<ActivationPrecision::FP32>::CUDAFlashAttentionKernelT(int device_idx)
            : device_idx_(device_idx), stream_(nullptr),
              partial_output_buf_(nullptr), partial_m_buf_(nullptr), partial_l_buf_(nullptr),
              workspace_size_(0), max_splits_(0), workspace_(nullptr), device_ctx_(nullptr)
        {
            if (device_idx < 0)
            {
                throw std::runtime_error(
                    "[CUDAFlashAttentionKernelT<FP32>] Invalid device_idx=" + std::to_string(device_idx) +
                    " — caller must pass explicit device ordinal");
            }
            LOG_DEBUG("[CUDAFlashAttentionKernelT<FP32>] Created for device " << device_idx);
        }

        CUDAFlashAttentionKernelT<ActivationPrecision::FP32>::CUDAFlashAttentionKernelT(
            IWorkerGPUContext *ctx)
            : stream_(nullptr),
              partial_output_buf_(nullptr), partial_m_buf_(nullptr), partial_l_buf_(nullptr),
              workspace_size_(0), max_splits_(0), workspace_(nullptr), device_ctx_(nullptr)
        {
            if (!ctx)
            {
                throw std::runtime_error(
                    "[CUDAFlashAttentionKernelT<FP32>] Device context is null");
            }

            if (!ctx->isInitialized())
            {
                throw std::runtime_error(
                    "[CUDAFlashAttentionKernelT<FP32>] Device context is not initialized");
            }

            setDeviceContext(ctx);
            device_idx_ = ctx->deviceOrdinal();

            // Get stream from context
            stream_ = ctx->defaultStream();

            LOG_DEBUG("[CUDAFlashAttentionKernelT<FP32>] Created for device " << device_idx_
                                                                              << " using device context");
        }

        CUDAFlashAttentionKernelT<ActivationPrecision::FP32>::~CUDAFlashAttentionKernelT()
        {
            freeWorkspace();
            if (h_attn_params_)
            {
                cudaFreeHost(h_attn_params_);
                h_attn_params_ = nullptr;
            }
        }

        CUDAFlashAttentionKernelT<ActivationPrecision::FP32>::CUDAFlashAttentionKernelT(
            CUDAFlashAttentionKernelT &&other) noexcept
            : device_idx_(other.device_idx_), stream_(other.stream_),
              partial_output_buf_(other.partial_output_buf_),
              partial_m_buf_(other.partial_m_buf_),
              partial_l_buf_(other.partial_l_buf_),
              workspace_size_(other.workspace_size_),
              max_splits_(other.max_splits_),
              workspace_(other.workspace_),
              device_ctx_(other.device_ctx_),
              h_attn_params_(other.h_attn_params_)
        {
            other.stream_ = nullptr;
            other.partial_output_buf_ = nullptr;
            other.partial_m_buf_ = nullptr;
            other.partial_l_buf_ = nullptr;
            other.workspace_size_ = 0;
            other.max_splits_ = 0;
            other.workspace_ = nullptr;
            other.device_ctx_ = nullptr;
            other.h_attn_params_ = nullptr;
        }

        CUDAFlashAttentionKernelT<ActivationPrecision::FP32> &
        CUDAFlashAttentionKernelT<ActivationPrecision::FP32>::operator=(
            CUDAFlashAttentionKernelT &&other) noexcept
        {
            if (this != &other)
            {
                freeWorkspace();
                if (h_attn_params_)
                {
                    cudaFreeHost(h_attn_params_);
                    h_attn_params_ = nullptr;
                }
                device_idx_ = other.device_idx_;
                stream_ = other.stream_;
                partial_output_buf_ = other.partial_output_buf_;
                partial_m_buf_ = other.partial_m_buf_;
                partial_l_buf_ = other.partial_l_buf_;
                workspace_size_ = other.workspace_size_;
                max_splits_ = other.max_splits_;
                workspace_ = other.workspace_;
                device_ctx_ = other.device_ctx_;
                h_attn_params_ = other.h_attn_params_;

                other.stream_ = nullptr;
                other.partial_output_buf_ = nullptr;
                other.partial_m_buf_ = nullptr;
                other.partial_l_buf_ = nullptr;
                other.workspace_size_ = 0;
                other.max_splits_ = 0;
                other.workspace_ = nullptr;
                other.device_ctx_ = nullptr;
                other.h_attn_params_ = nullptr;
            }
            return *this;
        }

        void CUDAFlashAttentionKernelT<ActivationPrecision::FP32>::allocateWorkspace(
            int n_heads, int head_dim, int num_splits)
        {
            if (num_splits <= max_splits_ && partial_output_buf_ != nullptr)
            {
                return; // Already have enough workspace
            }

            freeWorkspace();

            // Allocate for batch_size=1 (resize if needed)
            int batch_size = 1;
            if (cudaFlashAttn_allocWorkspace(
                    &partial_output_buf_, &partial_m_buf_, &partial_l_buf_,
                    batch_size, n_heads, head_dim, num_splits) != 0)
            {
                LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>] Failed to allocate workspace");
                partial_output_buf_ = nullptr;
                partial_m_buf_ = nullptr;
                partial_l_buf_ = nullptr;
                return;
            }

            max_splits_ = num_splits;
            workspace_size_ = static_cast<size_t>(batch_size) * n_heads * num_splits * head_dim * sizeof(float);
            LOG_DEBUG("[CUDAFlashAttentionKernelT<FP32>] Allocated workspace: " << workspace_size_ << " bytes");
        }

        void CUDAFlashAttentionKernelT<ActivationPrecision::FP32>::freeWorkspace()
        {
            if (partial_output_buf_ || partial_m_buf_ || partial_l_buf_)
            {
                cudaFlashAttn_freeWorkspace(partial_output_buf_, partial_m_buf_, partial_l_buf_);
                partial_output_buf_ = nullptr;
                partial_m_buf_ = nullptr;
                partial_l_buf_ = nullptr;
                workspace_size_ = 0;
                max_splits_ = 0;
            }
        }

        bool CUDAFlashAttentionKernelT<ActivationPrecision::FP32>::compute(
            const float *Q, const float *K, const float *V, float *output,
            int seq_len, int n_heads, int n_kv_heads, int head_dim,
            bool causal, int window_size,
            TensorBase *workspace_scores,
            TensorBase *workspace_buffer,
            TensorBase *workspace_context,
            TensorBase *workspace_mask,
            bool use_bf16,
            const MPIContext *mpi_ctx,
            int device_idx)
        {
            (void)workspace_scores;
            (void)workspace_buffer;
            (void)workspace_context;
            (void)use_bf16;
            (void)mpi_ctx;

            int dev = (device_idx >= 0) ? device_idx : device_idx_;

            const float *mask_ptr = nullptr;
            if (workspace_mask)
            {
                mask_ptr = static_cast<const float *>(workspace_mask->gpu_data_ptr());
            }
            return apply_typed(Q, K, V, output,
                               1, seq_len, seq_len, // batch=1, kv_len=seq_len
                               n_heads, n_kv_heads, head_dim,
                               causal, window_size, 0, dev, nullptr, mask_ptr);
        }

        bool CUDAFlashAttentionKernelT<ActivationPrecision::FP32>::compute_batch(
            const float *Q, const float *K, const float *V, float *output,
            int batch_size, int seq_len, int n_heads, int n_kv_heads, int head_dim,
            bool causal, int window_size,
            TensorBase *workspace_scores,
            TensorBase *workspace_buffer,
            TensorBase *workspace_context,
            TensorBase *workspace_mask,
            bool use_bf16,
            const MPIContext *mpi_ctx,
            int device_idx)
        {
            (void)workspace_scores;
            (void)workspace_buffer;
            (void)workspace_context;
            (void)use_bf16;
            (void)mpi_ctx;

            int dev = (device_idx >= 0) ? device_idx : device_idx_;
            const float *mask_ptr = nullptr;
            if (workspace_mask)
            {
                mask_ptr = static_cast<const float *>(workspace_mask->gpu_data_ptr());
            }
            return apply_typed(Q, K, V, output,
                               batch_size, seq_len, seq_len,
                               n_heads, n_kv_heads, head_dim,
                               causal, window_size, 0, dev, nullptr, mask_ptr);
        }

        bool CUDAFlashAttentionKernelT<ActivationPrecision::FP32>::compute_decode(
            const float *Q, const float *K, const float *V, float *output,
            int seq_len, int kv_len, int n_heads, int n_kv_heads, int head_dim,
            bool causal, int position_offset)
        {
            return apply_typed(Q, K, V, output,
                               1, seq_len, kv_len,
                               n_heads, n_kv_heads, head_dim,
                               causal, -1, position_offset, device_idx_, nullptr, nullptr);
        }

        bool CUDAFlashAttentionKernelT<ActivationPrecision::FP32>::apply_typed(
            const float *Q, const float *K, const float *V, float *output,
            int batch_size, int seq_len, int kv_len,
            int n_heads, int n_kv_heads, int head_dim,
            bool causal, int window_size, int position_offset,
            int device_idx,
            const attention::AttentionDeviceParams *device_params,
            const float *mask)
        {
            if (!Q || !K || !V || !output)
            {
                LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>] Null pointer input");
                return false;
            }

            if (seq_len <= 0 || kv_len <= 0 || n_heads <= 0 || head_dim <= 0)
            {
                LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>] Invalid dimensions: "
                          << "seq_len=" << seq_len << " kv_len=" << kv_len
                          << " n_heads=" << n_heads << " head_dim=" << head_dim);
                return false;
            }

            // GQA validation
            if (n_heads % n_kv_heads != 0)
            {
                LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>] n_heads must be divisible by n_kv_heads");
                return false;
            }

            // Set device — skip when running under graph capture (stream_ is non-null).
            // cudaSetDevice() is NOT graph-capturable and corrupts the capture on CUDA.
            // The device was already set before the capture scope began.
            if (!stream_)
            {
                if (cudaFlashAttn_setDevice(device_idx) != 0)
                {
                    LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>] Failed to set device " << device_idx);
                    return false;
                }
            }

            int result;

            // Choose algorithm based on seq_len
            if (seq_len == 1)
            {
                // Flash Decoding for single-token decode
                // Always use this path for seq_len=1, even for short KV
                int num_splits = DEFAULT_NUM_SPLITS;
                if (kv_len <= 64)
                    num_splits = 1; // No splitting for very short KV
                else if (kv_len < 128)
                    num_splits = 2;
                else if (kv_len < 256)
                    num_splits = 4;

                allocateWorkspace(n_heads, head_dim, num_splits);

                if (!partial_output_buf_ || !partial_m_buf_ || !partial_l_buf_)
                {
                    LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>] Workspace allocation failed");
                    return false;
                }

                LOG_DEBUG("[CUDAFlashAttentionKernelT<FP32>] Using Flash Decoding: kv_len=" << kv_len
                                                                                            << " num_splits=" << num_splits);

                {
                    CUDA_KERNEL_PROFILE_SCOPE(CUDAKernelType::FLASH_ATTN_DECODE);
                    result = cudaFlashAttn_decode_fp32(
                        Q, K, V, output,
                        static_cast<float *>(partial_output_buf_),
                        static_cast<float *>(partial_m_buf_),
                        static_cast<float *>(partial_l_buf_),
                        batch_size, kv_len,
                        n_heads, n_kv_heads, head_dim,
                        num_splits, device_params, stream_);
                }
            }
            else
            {
                // Flash Attention 2 (pipelined) for prefill or short KV
                // Uses pipelined cp.async + WMMA on Ampere SM >= 8.0
                LOG_DEBUG("[CUDAFlashAttentionKernelT<FP32>] Using Flash Attention 2 (pipelined WMMA): "
                          << "batch=" << batch_size << " seq_len=" << seq_len << " kv_len=" << kv_len);

                {
                    CUDA_KERNEL_PROFILE_SCOPE(CUDAKernelType::FLASH_ATTN_PREFILL);
                    result = cudaFlashAttn_prefill_fa2(
                        Q, K, V, output,
                        batch_size, seq_len, kv_len,
                        n_heads, n_kv_heads, head_dim,
                        causal, window_size, position_offset,
                        device_params, mask,
                        stream_);
                }
            }

            // Note: No cudaFlashAttn_synchronize() - rely on CUDA stream ordering
            // This enables GPU pipeline parallelism

            if (result != 0)
            {
                LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>] Kernel execution failed");
                return false;
            }

            return true;
        }

        bool CUDAFlashAttentionKernelT<ActivationPrecision::FP32>::compute_tensor(
            const ITensor *Q,
            const ITensor *K,
            const ITensor *V,
            ITensor *output,
            int batch_size,
            int seq_len,
            int kv_len,
            int n_heads,
            int n_kv_heads,
            int head_dim,
            bool causal,
            int window_size,
            ITensor *workspace_scores,
            ITensor *workspace_mask,
            const MPIContext *mpi_ctx,
            int device_idx,
            int head_start,
            int local_n_heads,
            int local_n_kv_heads)
        {
            (void)workspace_scores;
            (void)mpi_ctx;
            (void)head_start;
            (void)local_n_heads;
            (void)local_n_kv_heads;

            if (!Q || !K || !V || !output)
            {
                LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>::compute_tensor] Null tensor provided");
                return false;
            }

            // Verify tensor types match FP32
            if (Q->native_type() != TensorType::FP32)
            {
                LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>::compute_tensor] Q tensor type mismatch: expected FP32, got "
                          << Q->dtype_name());
                return false;
            }

            // Extract GPU pointers - tensors should already be coherent on device
            const float *Q_ptr = static_cast<const float *>(Q->gpu_data_ptr());
            const float *K_ptr = static_cast<const float *>(K->gpu_data_ptr());
            const float *V_ptr = static_cast<const float *>(V->gpu_data_ptr());
            float *output_ptr = static_cast<float *>(output->gpu_data_ptr());

            thread_local std::vector<float> k_host_fp32;
            thread_local std::vector<float> v_host_fp32;

            if (K->native_type() != TensorType::FP32 || V->native_type() != TensorType::FP32)
            {
                if (K->native_type() != V->native_type())
                {
                    LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>::compute_tensor] Mixed K/V types not supported: K="
                              << K->dtype_name() << " V=" << V->dtype_name());
                    return false;
                }

                if (!workspace_)
                {
                    LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>::compute_tensor] Mixed-precision KV requires workspace-bound conversion buffers");
                    return false;
                }

                const size_t rows = static_cast<size_t>(batch_size) * static_cast<size_t>(kv_len);
                const size_t logical_cols = static_cast<size_t>(n_kv_heads) * static_cast<size_t>(head_dim);
                const size_t logical_elements = rows * logical_cols;
                k_host_fp32.resize(logical_elements);
                v_host_fp32.resize(logical_elements);

                if (K->native_type() == TensorType::FP16)
                {
                    thread_local std::vector<uint16_t> k_half;
                    thread_local std::vector<uint16_t> v_half;
                    k_half.resize(logical_elements);
                    v_half.resize(logical_elements);

                    cudaError_t k_err = cudaMemcpy(k_half.data(), K->gpu_data_ptr(), logical_elements * sizeof(uint16_t), cudaMemcpyDeviceToHost);
                    cudaError_t v_err = cudaMemcpy(v_half.data(), V->gpu_data_ptr(), logical_elements * sizeof(uint16_t), cudaMemcpyDeviceToHost);
                    if (k_err != cudaSuccess || v_err != cudaSuccess)
                    {
                        LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>::compute_tensor] Failed D2H copy for FP16 KV conversion"
                                  << " k_err=" << cudaGetErrorString(k_err)
                                  << " v_err=" << cudaGetErrorString(v_err));
                        return false;
                    }

                    for (size_t i = 0; i < logical_elements; ++i)
                    {
                        k_host_fp32[i] = fp16_to_fp32(k_half[i]);
                        v_host_fp32[i] = fp16_to_fp32(v_half[i]);
                    }
                }
                else if (K->native_type() == TensorType::Q8_1)
                {
                    const size_t blocks_per_row = (logical_cols + Q8_1Block::BLOCK_SIZE - 1) / Q8_1Block::BLOCK_SIZE;
                    const size_t block_count = rows * blocks_per_row;

                    thread_local std::vector<Q8_1Block> k_blocks;
                    thread_local std::vector<Q8_1Block> v_blocks;
                    k_blocks.resize(block_count);
                    v_blocks.resize(block_count);
                    if (cudaMemcpy(k_blocks.data(), K->gpu_data_ptr(), block_count * sizeof(Q8_1Block), cudaMemcpyDeviceToHost) != cudaSuccess ||
                        cudaMemcpy(v_blocks.data(), V->gpu_data_ptr(), block_count * sizeof(Q8_1Block), cudaMemcpyDeviceToHost) != cudaSuccess)
                    {
                        LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>::compute_tensor] Failed D2H copy for Q8_1 KV conversion");
                        return false;
                    }

                    for (size_t r = 0; r < rows; ++r)
                    {
                        const size_t row_base = r * logical_cols;
                        for (size_t b = 0; b < blocks_per_row; ++b)
                        {
                            const Q8_1Block &kb = k_blocks[r * blocks_per_row + b];
                            const Q8_1Block &vb = v_blocks[r * blocks_per_row + b];
                            const float k_scale = fp16_to_fp32(kb.d);
                            const float v_scale = fp16_to_fp32(vb.d);
                            const size_t col_base = b * Q8_1Block::BLOCK_SIZE;
                            for (size_t i = 0; i < Q8_1Block::BLOCK_SIZE; ++i)
                            {
                                const size_t col = col_base + i;
                                if (col >= logical_cols)
                                    break;
                                k_host_fp32[row_base + col] = k_scale * static_cast<float>(kb.qs[i]);
                                v_host_fp32[row_base + col] = v_scale * static_cast<float>(vb.qs[i]);
                            }
                        }
                    }
                }
                else
                {
                    LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>::compute_tensor] Unsupported KV tensor type for FP32 attention: "
                              << K->dtype_name());
                    return false;
                }

                float *d_k_tmp = static_cast<float *>(workspace_->getBuffer(AttentionWorkspaceBuffers::K_TMP_FP32));
                float *d_v_tmp = static_cast<float *>(workspace_->getBuffer(AttentionWorkspaceBuffers::V_TMP_FP32));
                if (!d_k_tmp || !d_v_tmp)
                {
                    LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>::compute_tensor] Missing workspace conversion buffers: "
                              << AttentionWorkspaceBuffers::K_TMP_FP32 << " / "
                              << AttentionWorkspaceBuffers::V_TMP_FP32);
                    return false;
                }

                if (cudaMemcpy(d_k_tmp, k_host_fp32.data(), logical_elements * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess ||
                    cudaMemcpy(d_v_tmp, v_host_fp32.data(), logical_elements * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
                {
                    LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>::compute_tensor] Failed H2D copy for temporary FP32 KV buffers");
                    return false;
                }

                K_ptr = d_k_tmp;
                V_ptr = d_v_tmp;
            }

            if (!Q_ptr || !K_ptr || !V_ptr || !output_ptr)
            {
                LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>::compute_tensor] GPU data pointer is null. "
                          << "Ensure tensors are coherent on device (ensureOnDevice).");
                return false;
            }

            // Wire device_params for graph-capture replay: H2D memcpy from pinned host
            // memory is captured as a graph node. On replay, the memcpy re-reads
            // the updated pinned values (kv_len, position_offset, mask_stride).
            const attention::AttentionDeviceParams *d_attn_params = nullptr;
            if (stream_ && workspace_)
            {
                // Lazy-allocate pinned host buffer
                if (!h_attn_params_)
                {
                    cudaError_t err = cudaMallocHost(reinterpret_cast<void **>(&h_attn_params_),
                                                     sizeof(attention::AttentionDeviceParams));
                    if (err != cudaSuccess)
                    {
                        LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>::compute_tensor] "
                                  "cudaMallocHost failed for h_attn_params_: "
                                  << cudaGetErrorString(err));
                        h_attn_params_ = nullptr;
                    }
                }
                if (h_attn_params_)
                {
                    h_attn_params_->kv_len = kv_len;
                    h_attn_params_->position_offset = 0;
                    h_attn_params_->mask_stride = kv_len;

                    void *d_buf = workspace_->getBuffer(AttentionWorkspaceBuffers::DEVICE_PARAMS);
                    if (d_buf)
                    {
                        cudaMemcpyAsync(d_buf, h_attn_params_,
                                        sizeof(attention::AttentionDeviceParams),
                                        cudaMemcpyHostToDevice,
                                        static_cast<cudaStream_t>(stream_));
                        d_attn_params = static_cast<const attention::AttentionDeviceParams *>(d_buf);
                    }
                }
            }

            const float *mask_ptr = nullptr;
            if (workspace_mask)
            {
                mask_ptr = static_cast<const float *>(workspace_mask->gpu_data_ptr());
            }

            LOG_DEBUG("[CUDAFlashAttentionKernelT<FP32>::compute_tensor] batch=" << batch_size
                                                                                 << " seq_len=" << seq_len << " kv_len=" << kv_len
                                                                                 << " n_heads=" << n_heads << " n_kv_heads=" << n_kv_heads
                                                                                 << " head_dim=" << head_dim << " causal=" << causal);

            int dev = (device_idx >= 0) ? device_idx : device_idx_;

            // Dispatch to apply_typed which handles algorithm selection
            if (kv_len != seq_len)
            {
                // Decode path (different Q and KV lengths)
                return apply_typed(Q_ptr, K_ptr, V_ptr, output_ptr,
                                   batch_size, seq_len, kv_len,
                                   n_heads, n_kv_heads, head_dim,
                                   causal, window_size, 0, dev,
                                   d_attn_params, mask_ptr);
            }
            else
            {
                // Prefill path
                return apply_typed(Q_ptr, K_ptr, V_ptr, output_ptr,
                                   batch_size, seq_len, seq_len,
                                   n_heads, n_kv_heads, head_dim,
                                   causal, window_size, 0, dev,
                                   d_attn_params, mask_ptr);
            }
        }

        // =====================================================================
        // FP32 IWorkspaceConsumer Interface Implementation
        // =====================================================================

        WorkspaceRequirements CUDAFlashAttentionKernelT<ActivationPrecision::FP32>::getWorkspaceRequirements(
            int m, int n, int k) const
        {
            WorkspaceRequirements reqs;

            // Default parameters for Flash Decoding workspace sizing
            // Conservative estimates for maximum expected configuration
            const int batch_size = (m > 0) ? m : 1;
            const int n_heads = (n > 0) ? n : 128;     // Max expected heads
            const int head_dim = (k > 0) ? k : 128;    // Max expected head dim
            const int num_splits = DEFAULT_NUM_SPLITS; // 8 splits
            const int max_kv_len = 4096;               // decode workspace bound

            // Conservative conversion buffer sizing for mixed-precision KV
            // Assume n_kv_heads <= n_heads and allocate with n_heads for safety.
            size_t kv_convert_bytes = static_cast<size_t>(batch_size) *
                                      static_cast<size_t>(max_kv_len) *
                                      static_cast<size_t>(n_heads) *
                                      static_cast<size_t>(head_dim) *
                                      sizeof(float);

            // partial_output: [batch × n_heads × num_splits × head_dim] FP32
            size_t partial_output_bytes = static_cast<size_t>(batch_size) * n_heads * num_splits * head_dim * sizeof(float);

            // partial_m: [batch × n_heads × num_splits] FP32 (max scores per split)
            size_t partial_m_bytes = static_cast<size_t>(batch_size) * n_heads * num_splits * sizeof(float);

            // partial_l: [batch × n_heads × num_splits] FP32 (logsumexp per split)
            size_t partial_l_bytes = static_cast<size_t>(batch_size) * n_heads * num_splits * sizeof(float);

            reqs.buffers.push_back({AttentionWorkspaceBuffers::PARTIAL_OUTPUT, partial_output_bytes, 256, true});
            reqs.buffers.push_back({AttentionWorkspaceBuffers::PARTIAL_M, partial_m_bytes, 256, true});
            reqs.buffers.push_back({AttentionWorkspaceBuffers::PARTIAL_L, partial_l_bytes, 256, true});
            reqs.buffers.push_back({AttentionWorkspaceBuffers::DEVICE_PARAMS,
                                    sizeof(attention::AttentionDeviceParams),
                                    256,
                                    true});
            reqs.buffers.push_back({AttentionWorkspaceBuffers::K_TMP_FP32, kv_convert_bytes, 256, true});
            reqs.buffers.push_back({AttentionWorkspaceBuffers::V_TMP_FP32, kv_convert_bytes, 256, true});

            LOG_DEBUG("[CUDAFlashAttentionKernelT<FP32>::getWorkspaceRequirements] "
                      << "batch=" << batch_size << " n_heads=" << n_heads << " head_dim=" << head_dim
                      << " num_splits=" << num_splits
                      << " max_kv_len=" << max_kv_len
                      << " => partial_output=" << (partial_output_bytes / 1024) << "KB"
                      << ", partial_m=" << partial_m_bytes << "B"
                      << ", partial_l=" << partial_l_bytes << "B"
                      << ", kv_convert(each)=" << (kv_convert_bytes / (1024 * 1024)) << "MB");

            return reqs;
        }

        void CUDAFlashAttentionKernelT<ActivationPrecision::FP32>::bindWorkspace(
            DeviceWorkspaceManager *workspace)
        {
            workspace_ = workspace;
            if (workspace)
            {
                LOG_DEBUG("[CUDAFlashAttentionKernelT<FP32>] Bound workspace manager, entering managed mode");
            }
            else
            {
                LOG_DEBUG("[CUDAFlashAttentionKernelT<FP32>] Unbound workspace, returning to legacy mode");
            }
        }

        bool CUDAFlashAttentionKernelT<ActivationPrecision::FP32>::hasWorkspace() const
        {
            return workspace_ != nullptr;
        }

        DeviceWorkspaceManager *CUDAFlashAttentionKernelT<ActivationPrecision::FP32>::getWorkspace() const
        {
            return workspace_;
        }

        void CUDAFlashAttentionKernelT<ActivationPrecision::FP32>::setDynamicAttnParams(
            int kv_len, int position_offset)
        {
            // Lazy-allocate pinned host buffer if not yet created.
            if (!h_attn_params_)
            {
                cudaError_t err = cudaMallocHost(reinterpret_cast<void **>(&h_attn_params_),
                                                 sizeof(attention::AttentionDeviceParams));
                if (err != cudaSuccess)
                    h_attn_params_ = nullptr;
            }
            if (h_attn_params_)
            {
                h_attn_params_->kv_len = kv_len;
                h_attn_params_->position_offset = position_offset;
                h_attn_params_->mask_stride = kv_len;

                // Explicit H2D copy OUTSIDE graph capture scope.
                // Ensures device buffer has current attention params before
                // CUDA graph replay (see RoPE setDynamicPosOffset rationale).
                if (stream_ && workspace_)
                {
                    void *d_buf = workspace_->getBuffer(AttentionWorkspaceBuffers::DEVICE_PARAMS);
                    if (d_buf)
                    {
                        cudaMemcpyAsync(d_buf, h_attn_params_,
                                        sizeof(attention::AttentionDeviceParams),
                                        cudaMemcpyHostToDevice,
                                        static_cast<cudaStream_t>(stream_));
                    }
                }
            }
        }

        // =====================================================================
        // FP16 Specialization Implementation (stub - delegates to FP32)
        // =====================================================================

        CUDAFlashAttentionKernelT<ActivationPrecision::FP16>::CUDAFlashAttentionKernelT(int device_idx)
            : device_idx_(device_idx), stream_(nullptr),
              partial_output_buf_(nullptr), partial_m_buf_(nullptr), partial_l_buf_(nullptr),
              workspace_size_(0), max_splits_(0), workspace_(nullptr), device_ctx_(nullptr)
        {
            if (device_idx < 0)
            {
                throw std::runtime_error(
                    "[CUDAFlashAttentionKernelT<FP16>] Invalid device_idx=" + std::to_string(device_idx) +
                    " — caller must pass explicit device ordinal");
            }
            LOG_DEBUG("[CUDAFlashAttentionKernelT<FP16>] Created for device " << device_idx);
        }

        CUDAFlashAttentionKernelT<ActivationPrecision::FP16>::CUDAFlashAttentionKernelT(
            IWorkerGPUContext *ctx)
            : stream_(nullptr),
              partial_output_buf_(nullptr), partial_m_buf_(nullptr), partial_l_buf_(nullptr),
              workspace_size_(0), max_splits_(0), workspace_(nullptr), device_ctx_(nullptr)
        {
            if (!ctx)
            {
                throw std::runtime_error(
                    "[CUDAFlashAttentionKernelT<FP16>] Device context is null");
            }

            if (!ctx->isInitialized())
            {
                throw std::runtime_error(
                    "[CUDAFlashAttentionKernelT<FP16>] Device context is not initialized");
            }

            setDeviceContext(ctx);
            device_idx_ = ctx->deviceOrdinal();
            stream_ = ctx->defaultStream();

            LOG_DEBUG("[CUDAFlashAttentionKernelT<FP16>] Created for device " << device_idx_
                                                                              << " using device context");
        }

        CUDAFlashAttentionKernelT<ActivationPrecision::FP16>::~CUDAFlashAttentionKernelT()
        {
            freeWorkspace();
            if (h_attn_params_)
            {
                cudaFreeHost(h_attn_params_);
                h_attn_params_ = nullptr;
            }
        }

        CUDAFlashAttentionKernelT<ActivationPrecision::FP16>::CUDAFlashAttentionKernelT(
            CUDAFlashAttentionKernelT &&other) noexcept
            : device_idx_(other.device_idx_), stream_(other.stream_),
              partial_output_buf_(other.partial_output_buf_),
              partial_m_buf_(other.partial_m_buf_),
              partial_l_buf_(other.partial_l_buf_),
              workspace_size_(other.workspace_size_),
              max_splits_(other.max_splits_),
              workspace_(other.workspace_),
              device_ctx_(other.device_ctx_),
              h_attn_params_(other.h_attn_params_)
        {
            other.stream_ = nullptr;
            other.partial_output_buf_ = nullptr;
            other.partial_m_buf_ = nullptr;
            other.partial_l_buf_ = nullptr;
            other.workspace_ = nullptr;
            other.device_ctx_ = nullptr;
            other.h_attn_params_ = nullptr;
        }

        CUDAFlashAttentionKernelT<ActivationPrecision::FP16> &
        CUDAFlashAttentionKernelT<ActivationPrecision::FP16>::operator=(
            CUDAFlashAttentionKernelT &&other) noexcept
        {
            if (this != &other)
            {
                freeWorkspace();
                if (h_attn_params_)
                {
                    cudaFreeHost(h_attn_params_);
                    h_attn_params_ = nullptr;
                }
                device_idx_ = other.device_idx_;
                stream_ = other.stream_;
                partial_output_buf_ = other.partial_output_buf_;
                partial_m_buf_ = other.partial_m_buf_;
                partial_l_buf_ = other.partial_l_buf_;
                workspace_size_ = other.workspace_size_;
                max_splits_ = other.max_splits_;
                workspace_ = other.workspace_;
                device_ctx_ = other.device_ctx_;
                h_attn_params_ = other.h_attn_params_;
                other.stream_ = nullptr;
                other.partial_output_buf_ = nullptr;
                other.partial_m_buf_ = nullptr;
                other.partial_l_buf_ = nullptr;
                other.workspace_ = nullptr;
                other.device_ctx_ = nullptr;
                other.h_attn_params_ = nullptr;
            }
            return *this;
        }

        void CUDAFlashAttentionKernelT<ActivationPrecision::FP16>::allocateWorkspace(
            int n_heads, int head_dim, int num_splits)
        {
            // TODO: Implement FP16 workspace
            (void)n_heads;
            (void)head_dim;
            (void)num_splits;
        }

        void CUDAFlashAttentionKernelT<ActivationPrecision::FP16>::freeWorkspace()
        {
            if (partial_output_buf_ || partial_m_buf_ || partial_l_buf_)
            {
                cudaFlashAttn_freeWorkspace(partial_output_buf_, partial_m_buf_, partial_l_buf_);
                partial_output_buf_ = nullptr;
                partial_m_buf_ = nullptr;
                partial_l_buf_ = nullptr;
            }
        }

        bool CUDAFlashAttentionKernelT<ActivationPrecision::FP16>::compute(
            const float *Q, const float *K, const float *V, float *output,
            int seq_len, int n_heads, int n_kv_heads, int head_dim,
            bool causal, int window_size,
            TensorBase *workspace_scores,
            TensorBase *workspace_buffer,
            TensorBase *workspace_context,
            TensorBase *workspace_mask,
            bool use_bf16,
            const MPIContext *mpi_ctx,
            int device_idx)
        {
            // TODO: Native FP16 implementation
            // For now, use FP32 path
            LOG_WARN("[CUDAFlashAttentionKernelT<FP16>] FP16 not yet implemented, using FP32");
            CUDAFlashAttentionKernelT<ActivationPrecision::FP32> fp32_kernel(device_idx_);
            return fp32_kernel.compute(Q, K, V, output, seq_len, n_heads, n_kv_heads, head_dim,
                                       causal, window_size, workspace_scores, workspace_buffer,
                                       workspace_context, workspace_mask, use_bf16, mpi_ctx, device_idx);
        }

        bool CUDAFlashAttentionKernelT<ActivationPrecision::FP16>::compute_batch(
            const float *Q, const float *K, const float *V, float *output,
            int batch_size, int seq_len, int n_heads, int n_kv_heads, int head_dim,
            bool causal, int window_size,
            TensorBase *workspace_scores,
            TensorBase *workspace_buffer,
            TensorBase *workspace_context,
            TensorBase *workspace_mask,
            bool use_bf16,
            const MPIContext *mpi_ctx,
            int device_idx)
        {
            LOG_WARN("[CUDAFlashAttentionKernelT<FP16>] FP16 not yet implemented, using FP32");
            CUDAFlashAttentionKernelT<ActivationPrecision::FP32> fp32_kernel(device_idx_);
            return fp32_kernel.compute_batch(Q, K, V, output, batch_size, seq_len, n_heads, n_kv_heads, head_dim,
                                             causal, window_size, workspace_scores, workspace_buffer,
                                             workspace_context, workspace_mask, use_bf16, mpi_ctx, device_idx);
        }

        bool CUDAFlashAttentionKernelT<ActivationPrecision::FP16>::compute_decode(
            const float *Q, const float *K, const float *V, float *output,
            int seq_len, int kv_len, int n_heads, int n_kv_heads, int head_dim,
            bool causal, int position_offset)
        {
            LOG_WARN("[CUDAFlashAttentionKernelT<FP16>] FP16 not yet implemented, using FP32");
            CUDAFlashAttentionKernelT<ActivationPrecision::FP32> fp32_kernel(device_idx_);
            return fp32_kernel.compute_decode(Q, K, V, output, seq_len, kv_len, n_heads, n_kv_heads, head_dim,
                                              causal, position_offset);
        }

        bool CUDAFlashAttentionKernelT<ActivationPrecision::FP16>::apply_typed(
            const uint16_t *Q, const uint16_t *K, const uint16_t *V, uint16_t *output,
            int batch_size, int seq_len, int kv_len,
            int n_heads, int n_kv_heads, int head_dim,
            bool causal, int window_size, int position_offset,
            int device_idx,
            const attention::AttentionDeviceParams *device_params,
            const float *mask)
        {
            (void)device_params;
            (void)mask;
            // TODO: Native FP16 path
            (void)Q;
            (void)K;
            (void)V;
            (void)output;
            (void)batch_size;
            (void)seq_len;
            (void)kv_len;
            (void)n_heads;
            (void)n_kv_heads;
            (void)head_dim;
            (void)causal;
            (void)window_size;
            (void)position_offset;
            (void)device_idx;
            LOG_ERROR("[CUDAFlashAttentionKernelT<FP16>] apply_typed not yet implemented");
            return false;
        }

        bool CUDAFlashAttentionKernelT<ActivationPrecision::FP16>::compute_tensor(
            const ITensor *Q,
            const ITensor *K,
            const ITensor *V,
            ITensor *output,
            int batch_size,
            int seq_len,
            int kv_len,
            int n_heads,
            int n_kv_heads,
            int head_dim,
            bool causal,
            int window_size,
            ITensor *workspace_scores,
            ITensor *workspace_mask,
            const MPIContext *mpi_ctx,
            int device_idx,
            int head_start,
            int local_n_heads,
            int local_n_kv_heads)
        {
            // FP16 compute_tensor: delegate to FP32 for now
            LOG_WARN("[CUDAFlashAttentionKernelT<FP16>] FP16 compute_tensor not yet implemented, using FP32");
            CUDAFlashAttentionKernelT<ActivationPrecision::FP32> fp32_kernel(device_idx_);
            return fp32_kernel.compute_tensor(Q, K, V, output, batch_size, seq_len, kv_len,
                                              n_heads, n_kv_heads, head_dim, causal, window_size,
                                              workspace_scores, workspace_mask, mpi_ctx, device_idx,
                                              head_start, local_n_heads, local_n_kv_heads);
        }

        // =====================================================================
        // FP16 IWorkspaceConsumer Interface Implementation
        // =====================================================================

        WorkspaceRequirements CUDAFlashAttentionKernelT<ActivationPrecision::FP16>::getWorkspaceRequirements(
            int m, int n, int k) const
        {
            // Delegate to FP32 implementation (same workspace requirements)
            CUDAFlashAttentionKernelT<ActivationPrecision::FP32> fp32_kernel(device_idx_);
            return fp32_kernel.getWorkspaceRequirements(m, n, k);
        }

        void CUDAFlashAttentionKernelT<ActivationPrecision::FP16>::bindWorkspace(
            DeviceWorkspaceManager *workspace)
        {
            workspace_ = workspace;
            if (workspace)
            {
                LOG_DEBUG("[CUDAFlashAttentionKernelT<FP16>] Bound workspace manager, entering managed mode");
            }
            else
            {
                LOG_DEBUG("[CUDAFlashAttentionKernelT<FP16>] Unbound workspace, returning to legacy mode");
            }
        }

        bool CUDAFlashAttentionKernelT<ActivationPrecision::FP16>::hasWorkspace() const
        {
            return workspace_ != nullptr;
        }

        DeviceWorkspaceManager *CUDAFlashAttentionKernelT<ActivationPrecision::FP16>::getWorkspace() const
        {
            return workspace_;
        }

        void CUDAFlashAttentionKernelT<ActivationPrecision::FP16>::setDynamicAttnParams(
            int kv_len, int position_offset)
        {
            if (!h_attn_params_)
            {
                cudaError_t err = cudaMallocHost(reinterpret_cast<void **>(&h_attn_params_),
                                                 sizeof(attention::AttentionDeviceParams));
                if (err != cudaSuccess)
                    h_attn_params_ = nullptr;
            }
            if (h_attn_params_)
            {
                h_attn_params_->kv_len = kv_len;
                h_attn_params_->position_offset = position_offset;
                h_attn_params_->mask_stride = kv_len;

                if (stream_ && workspace_)
                {
                    void *d_buf = workspace_->getBuffer(AttentionWorkspaceBuffers::DEVICE_PARAMS);
                    if (d_buf)
                    {
                        cudaMemcpyAsync(d_buf, h_attn_params_,
                                        sizeof(attention::AttentionDeviceParams),
                                        cudaMemcpyHostToDevice,
                                        static_cast<cudaStream_t>(stream_));
                    }
                }
            }
        }

        // =====================================================================
        // BF16 Specialization Implementation (stub - delegates to FP32)
        // =====================================================================

        CUDAFlashAttentionKernelT<ActivationPrecision::BF16>::CUDAFlashAttentionKernelT(int device_idx)
            : device_idx_(device_idx), stream_(nullptr),
              partial_output_buf_(nullptr), partial_m_buf_(nullptr), partial_l_buf_(nullptr),
              workspace_size_(0), max_splits_(0), workspace_(nullptr), device_ctx_(nullptr)
        {
            if (device_idx < 0)
            {
                throw std::runtime_error(
                    "[CUDAFlashAttentionKernelT<BF16>] Invalid device_idx=" + std::to_string(device_idx) +
                    " — caller must pass explicit device ordinal");
            }
            LOG_DEBUG("[CUDAFlashAttentionKernelT<BF16>] Created for device " << device_idx);
        }

        CUDAFlashAttentionKernelT<ActivationPrecision::BF16>::CUDAFlashAttentionKernelT(
            IWorkerGPUContext *ctx)
            : stream_(nullptr),
              partial_output_buf_(nullptr), partial_m_buf_(nullptr), partial_l_buf_(nullptr),
              workspace_size_(0), max_splits_(0), workspace_(nullptr), device_ctx_(nullptr)
        {
            if (!ctx)
            {
                throw std::runtime_error(
                    "[CUDAFlashAttentionKernelT<BF16>] Device context is null");
            }

            if (!ctx->isInitialized())
            {
                throw std::runtime_error(
                    "[CUDAFlashAttentionKernelT<BF16>] Device context is not initialized");
            }

            setDeviceContext(ctx);
            device_idx_ = ctx->deviceOrdinal();
            stream_ = ctx->defaultStream();

            LOG_DEBUG("[CUDAFlashAttentionKernelT<BF16>] Created for device " << device_idx_
                                                                              << " using device context");
        }

        CUDAFlashAttentionKernelT<ActivationPrecision::BF16>::~CUDAFlashAttentionKernelT()
        {
            freeWorkspace();
            if (h_attn_params_)
            {
                cudaFreeHost(h_attn_params_);
                h_attn_params_ = nullptr;
            }
        }

        CUDAFlashAttentionKernelT<ActivationPrecision::BF16>::CUDAFlashAttentionKernelT(
            CUDAFlashAttentionKernelT &&other) noexcept
            : device_idx_(other.device_idx_), stream_(other.stream_),
              partial_output_buf_(other.partial_output_buf_),
              partial_m_buf_(other.partial_m_buf_),
              partial_l_buf_(other.partial_l_buf_),
              workspace_size_(other.workspace_size_),
              max_splits_(other.max_splits_),
              workspace_(other.workspace_),
              device_ctx_(other.device_ctx_),
              h_attn_params_(other.h_attn_params_)
        {
            other.stream_ = nullptr;
            other.partial_output_buf_ = nullptr;
            other.partial_m_buf_ = nullptr;
            other.partial_l_buf_ = nullptr;
            other.workspace_ = nullptr;
            other.device_ctx_ = nullptr;
            other.h_attn_params_ = nullptr;
        }

        CUDAFlashAttentionKernelT<ActivationPrecision::BF16> &
        CUDAFlashAttentionKernelT<ActivationPrecision::BF16>::operator=(
            CUDAFlashAttentionKernelT &&other) noexcept
        {
            if (this != &other)
            {
                freeWorkspace();
                if (h_attn_params_)
                {
                    cudaFreeHost(h_attn_params_);
                    h_attn_params_ = nullptr;
                }
                device_idx_ = other.device_idx_;
                stream_ = other.stream_;
                partial_output_buf_ = other.partial_output_buf_;
                partial_m_buf_ = other.partial_m_buf_;
                partial_l_buf_ = other.partial_l_buf_;
                workspace_size_ = other.workspace_size_;
                max_splits_ = other.max_splits_;
                workspace_ = other.workspace_;
                device_ctx_ = other.device_ctx_;
                h_attn_params_ = other.h_attn_params_;
                other.stream_ = nullptr;
                other.partial_output_buf_ = nullptr;
                other.partial_m_buf_ = nullptr;
                other.partial_l_buf_ = nullptr;
                other.workspace_ = nullptr;
                other.device_ctx_ = nullptr;
                other.h_attn_params_ = nullptr;
            }
            return *this;
        }

        void CUDAFlashAttentionKernelT<ActivationPrecision::BF16>::allocateWorkspace(
            int n_heads, int head_dim, int num_splits)
        {
            // TODO: Implement BF16 workspace
            (void)n_heads;
            (void)head_dim;
            (void)num_splits;
        }

        void CUDAFlashAttentionKernelT<ActivationPrecision::BF16>::freeWorkspace()
        {
            if (partial_output_buf_ || partial_m_buf_ || partial_l_buf_)
            {
                cudaFlashAttn_freeWorkspace(partial_output_buf_, partial_m_buf_, partial_l_buf_);
                partial_output_buf_ = nullptr;
                partial_m_buf_ = nullptr;
                partial_l_buf_ = nullptr;
            }
        }

        bool CUDAFlashAttentionKernelT<ActivationPrecision::BF16>::compute(
            const float *Q, const float *K, const float *V, float *output,
            int seq_len, int n_heads, int n_kv_heads, int head_dim,
            bool causal, int window_size,
            TensorBase *workspace_scores,
            TensorBase *workspace_buffer,
            TensorBase *workspace_context,
            TensorBase *workspace_mask,
            bool use_bf16,
            const MPIContext *mpi_ctx,
            int device_idx)
        {
            LOG_WARN("[CUDAFlashAttentionKernelT<BF16>] BF16 not yet implemented, using FP32");
            CUDAFlashAttentionKernelT<ActivationPrecision::FP32> fp32_kernel(device_idx_);
            return fp32_kernel.compute(Q, K, V, output, seq_len, n_heads, n_kv_heads, head_dim,
                                       causal, window_size, workspace_scores, workspace_buffer,
                                       workspace_context, workspace_mask, use_bf16, mpi_ctx, device_idx);
        }

        bool CUDAFlashAttentionKernelT<ActivationPrecision::BF16>::compute_batch(
            const float *Q, const float *K, const float *V, float *output,
            int batch_size, int seq_len, int n_heads, int n_kv_heads, int head_dim,
            bool causal, int window_size,
            TensorBase *workspace_scores,
            TensorBase *workspace_buffer,
            TensorBase *workspace_context,
            TensorBase *workspace_mask,
            bool use_bf16,
            const MPIContext *mpi_ctx,
            int device_idx)
        {
            LOG_WARN("[CUDAFlashAttentionKernelT<BF16>] BF16 not yet implemented, using FP32");
            CUDAFlashAttentionKernelT<ActivationPrecision::FP32> fp32_kernel(device_idx_);
            return fp32_kernel.compute_batch(Q, K, V, output, batch_size, seq_len, n_heads, n_kv_heads, head_dim,
                                             causal, window_size, workspace_scores, workspace_buffer,
                                             workspace_context, workspace_mask, use_bf16, mpi_ctx, device_idx);
        }

        bool CUDAFlashAttentionKernelT<ActivationPrecision::BF16>::compute_decode(
            const float *Q, const float *K, const float *V, float *output,
            int seq_len, int kv_len, int n_heads, int n_kv_heads, int head_dim,
            bool causal, int position_offset)
        {
            LOG_WARN("[CUDAFlashAttentionKernelT<BF16>] BF16 not yet implemented, using FP32");
            CUDAFlashAttentionKernelT<ActivationPrecision::FP32> fp32_kernel(device_idx_);
            return fp32_kernel.compute_decode(Q, K, V, output, seq_len, kv_len, n_heads, n_kv_heads, head_dim,
                                              causal, position_offset);
        }

        bool CUDAFlashAttentionKernelT<ActivationPrecision::BF16>::apply_typed(
            const uint16_t *Q, const uint16_t *K, const uint16_t *V, uint16_t *output,
            int batch_size, int seq_len, int kv_len,
            int n_heads, int n_kv_heads, int head_dim,
            bool causal, int window_size, int position_offset,
            int device_idx,
            const attention::AttentionDeviceParams *device_params,
            const float *mask)
        {
            // TODO: Native BF16 path
            (void)device_params;
            (void)mask;
            (void)Q;
            (void)K;
            (void)V;
            (void)output;
            (void)batch_size;
            (void)seq_len;
            (void)kv_len;
            (void)n_heads;
            (void)n_kv_heads;
            (void)head_dim;
            (void)causal;
            (void)window_size;
            (void)position_offset;
            (void)device_idx;
            LOG_ERROR("[CUDAFlashAttentionKernelT<BF16>] apply_typed not yet implemented");
            return false;
        }

        bool CUDAFlashAttentionKernelT<ActivationPrecision::BF16>::compute_tensor(
            const ITensor *Q,
            const ITensor *K,
            const ITensor *V,
            ITensor *output,
            int batch_size,
            int seq_len,
            int kv_len,
            int n_heads,
            int n_kv_heads,
            int head_dim,
            bool causal,
            int window_size,
            ITensor *workspace_scores,
            ITensor *workspace_mask,
            const MPIContext *mpi_ctx,
            int device_idx,
            int head_start,
            int local_n_heads,
            int local_n_kv_heads)
        {
            // BF16 compute_tensor: delegate to FP32 for now
            LOG_WARN("[CUDAFlashAttentionKernelT<BF16>] BF16 compute_tensor not yet implemented, using FP32");
            CUDAFlashAttentionKernelT<ActivationPrecision::FP32> fp32_kernel(device_idx_);
            return fp32_kernel.compute_tensor(Q, K, V, output, batch_size, seq_len, kv_len,
                                              n_heads, n_kv_heads, head_dim, causal, window_size,
                                              workspace_scores, workspace_mask, mpi_ctx, device_idx,
                                              head_start, local_n_heads, local_n_kv_heads);
        }

        // =====================================================================
        // BF16 IWorkspaceConsumer Interface Implementation
        // =====================================================================

        WorkspaceRequirements CUDAFlashAttentionKernelT<ActivationPrecision::BF16>::getWorkspaceRequirements(
            int m, int n, int k) const
        {
            // Delegate to FP32 implementation (same workspace requirements)
            CUDAFlashAttentionKernelT<ActivationPrecision::FP32> fp32_kernel(device_idx_);
            return fp32_kernel.getWorkspaceRequirements(m, n, k);
        }

        void CUDAFlashAttentionKernelT<ActivationPrecision::BF16>::bindWorkspace(
            DeviceWorkspaceManager *workspace)
        {
            workspace_ = workspace;
            if (workspace)
            {
                LOG_DEBUG("[CUDAFlashAttentionKernelT<BF16>] Bound workspace manager, entering managed mode");
            }
            else
            {
                LOG_DEBUG("[CUDAFlashAttentionKernelT<BF16>] Unbound workspace, returning to legacy mode");
            }
        }

        bool CUDAFlashAttentionKernelT<ActivationPrecision::BF16>::hasWorkspace() const
        {
            return workspace_ != nullptr;
        }

        DeviceWorkspaceManager *CUDAFlashAttentionKernelT<ActivationPrecision::BF16>::getWorkspace() const
        {
            return workspace_;
        }

        void CUDAFlashAttentionKernelT<ActivationPrecision::BF16>::setDynamicAttnParams(
            int kv_len, int position_offset)
        {
            if (!h_attn_params_)
            {
                cudaError_t err = cudaMallocHost(reinterpret_cast<void **>(&h_attn_params_),
                                                 sizeof(attention::AttentionDeviceParams));
                if (err != cudaSuccess)
                    h_attn_params_ = nullptr;
            }
            if (h_attn_params_)
            {
                h_attn_params_->kv_len = kv_len;
                h_attn_params_->position_offset = position_offset;
                h_attn_params_->mask_stride = kv_len;

                if (stream_ && workspace_)
                {
                    void *d_buf = workspace_->getBuffer(AttentionWorkspaceBuffers::DEVICE_PARAMS);
                    if (d_buf)
                    {
                        cudaMemcpyAsync(d_buf, h_attn_params_,
                                        sizeof(attention::AttentionDeviceParams),
                                        cudaMemcpyHostToDevice,
                                        static_cast<cudaStream_t>(stream_));
                    }
                }
            }
        }

    } // namespace cuda
} // namespace llaminar2
