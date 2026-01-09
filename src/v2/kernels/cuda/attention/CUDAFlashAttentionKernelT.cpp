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
#include "../../../utils/Logger.h"
#include <cstring>

// Extern "C" declarations for CUDA kernel wrappers
extern "C"
{
    // FA3-style pipelined prefill (SM >= 8.0)
    // Supports head_dim=64 (6 consumer warps) and head_dim=128 (4 consumer warps)
    int cudaFlashAttn_prefill_fa3(
        const float *Q, const float *K, const float *V, float *O,
        int batch_size, int seq_len, int kv_len,
        int n_heads, int n_kv_heads, int head_dim,
        bool causal, int window_size, int position_offset,
        void *stream);

    // Flash Decoding for single-token decode with split-K parallelism
    int cudaFlashAttn_decode_fp32(
        const float *Q, const float *K_cache, const float *V_cache, float *O,
        float *O_partial, float *m_partial, float *l_partial,
        int batch_size, int kv_len,
        int n_heads, int n_kv_heads, int head_dim,
        int num_splits,
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
              workspace_size_(0), max_splits_(0)
        {
            LOG_DEBUG("[CUDAFlashAttentionKernelT<FP32>] Created for device " << device_idx);
        }

        CUDAFlashAttentionKernelT<ActivationPrecision::FP32>::~CUDAFlashAttentionKernelT()
        {
            freeWorkspace();
        }

        CUDAFlashAttentionKernelT<ActivationPrecision::FP32>::CUDAFlashAttentionKernelT(
            CUDAFlashAttentionKernelT &&other) noexcept
            : device_idx_(other.device_idx_), stream_(other.stream_),
              partial_output_buf_(other.partial_output_buf_),
              partial_m_buf_(other.partial_m_buf_),
              partial_l_buf_(other.partial_l_buf_),
              workspace_size_(other.workspace_size_),
              max_splits_(other.max_splits_)
        {
            other.stream_ = nullptr;
            other.partial_output_buf_ = nullptr;
            other.partial_m_buf_ = nullptr;
            other.partial_l_buf_ = nullptr;
            other.workspace_size_ = 0;
            other.max_splits_ = 0;
        }

        CUDAFlashAttentionKernelT<ActivationPrecision::FP32> &
        CUDAFlashAttentionKernelT<ActivationPrecision::FP32>::operator=(
            CUDAFlashAttentionKernelT &&other) noexcept
        {
            if (this != &other)
            {
                freeWorkspace();
                device_idx_ = other.device_idx_;
                stream_ = other.stream_;
                partial_output_buf_ = other.partial_output_buf_;
                partial_m_buf_ = other.partial_m_buf_;
                partial_l_buf_ = other.partial_l_buf_;
                workspace_size_ = other.workspace_size_;
                max_splits_ = other.max_splits_;

                other.stream_ = nullptr;
                other.partial_output_buf_ = nullptr;
                other.partial_m_buf_ = nullptr;
                other.partial_l_buf_ = nullptr;
                other.workspace_size_ = 0;
                other.max_splits_ = 0;
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
            (void)workspace_mask;
            (void)use_bf16;
            (void)mpi_ctx;

            int dev = (device_idx >= 0) ? device_idx : device_idx_;
            return apply_typed(Q, K, V, output,
                               1, seq_len, seq_len, // batch=1, kv_len=seq_len
                               n_heads, n_kv_heads, head_dim,
                               causal, window_size, 0, dev);
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
            (void)workspace_mask;
            (void)use_bf16;
            (void)mpi_ctx;

            int dev = (device_idx >= 0) ? device_idx : device_idx_;
            return apply_typed(Q, K, V, output,
                               batch_size, seq_len, seq_len,
                               n_heads, n_kv_heads, head_dim,
                               causal, window_size, 0, dev);
        }

        bool CUDAFlashAttentionKernelT<ActivationPrecision::FP32>::compute_decode(
            const float *Q, const float *K, const float *V, float *output,
            int seq_len, int kv_len, int n_heads, int n_kv_heads, int head_dim,
            bool causal, int position_offset)
        {
            return apply_typed(Q, K, V, output,
                               1, seq_len, kv_len,
                               n_heads, n_kv_heads, head_dim,
                               causal, -1, position_offset, device_idx_);
        }

        bool CUDAFlashAttentionKernelT<ActivationPrecision::FP32>::apply_typed(
            const float *Q, const float *K, const float *V, float *output,
            int batch_size, int seq_len, int kv_len,
            int n_heads, int n_kv_heads, int head_dim,
            bool causal, int window_size, int position_offset,
            int device_idx)
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

            // Set device
            if (cudaFlashAttn_setDevice(device_idx) != 0)
            {
                LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>] Failed to set device " << device_idx);
                return false;
            }

            int result;

            // Choose algorithm based on seq_len
            if (seq_len == 1)
            {
                // Flash Decoding for single-token decode
                // Always use this path for seq_len=1, even for short KV
                int num_splits = DEFAULT_NUM_SPLITS;
                if (kv_len <= 64)
                    num_splits = 1;  // No splitting for very short KV
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

                result = cudaFlashAttn_decode_fp32(
                    Q, K, V, output,
                    static_cast<float *>(partial_output_buf_),
                    static_cast<float *>(partial_m_buf_),
                    static_cast<float *>(partial_l_buf_),
                    batch_size, kv_len,
                    n_heads, n_kv_heads, head_dim,
                    num_splits, stream_);
            }
            else
            {
                // Flash Attention 3 for prefill or short KV
                // Uses pipelined cp.async on SM >= 8.0, falls back to WMMA/scalar on older GPUs
                LOG_DEBUG("[CUDAFlashAttentionKernelT<FP32>] Using Flash Attention 3 (pipelined): "
                          << "batch=" << batch_size << " seq_len=" << seq_len << " kv_len=" << kv_len);

                result = cudaFlashAttn_prefill_fa3(
                    Q, K, V, output,
                    batch_size, seq_len, kv_len,
                    n_heads, n_kv_heads, head_dim,
                    causal, window_size, position_offset,
                    stream_);
            }

            // Synchronize to ensure completion (for testing; remove for async)
            cudaFlashAttn_synchronize();

            if (result != 0)
            {
                LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>] Kernel execution failed");
                return false;
            }

            return true;
        }

        // =====================================================================
        // FP16 Specialization Implementation (stub - delegates to FP32)
        // =====================================================================

        CUDAFlashAttentionKernelT<ActivationPrecision::FP16>::CUDAFlashAttentionKernelT(int device_idx)
            : device_idx_(device_idx), stream_(nullptr),
              partial_output_buf_(nullptr), partial_m_buf_(nullptr), partial_l_buf_(nullptr),
              workspace_size_(0), max_splits_(0)
        {
            LOG_DEBUG("[CUDAFlashAttentionKernelT<FP16>] Created for device " << device_idx);
        }

        CUDAFlashAttentionKernelT<ActivationPrecision::FP16>::~CUDAFlashAttentionKernelT()
        {
            freeWorkspace();
        }

        CUDAFlashAttentionKernelT<ActivationPrecision::FP16>::CUDAFlashAttentionKernelT(
            CUDAFlashAttentionKernelT &&other) noexcept
            : device_idx_(other.device_idx_), stream_(other.stream_),
              partial_output_buf_(other.partial_output_buf_),
              partial_m_buf_(other.partial_m_buf_),
              partial_l_buf_(other.partial_l_buf_),
              workspace_size_(other.workspace_size_),
              max_splits_(other.max_splits_)
        {
            other.stream_ = nullptr;
            other.partial_output_buf_ = nullptr;
            other.partial_m_buf_ = nullptr;
            other.partial_l_buf_ = nullptr;
        }

        CUDAFlashAttentionKernelT<ActivationPrecision::FP16> &
        CUDAFlashAttentionKernelT<ActivationPrecision::FP16>::operator=(
            CUDAFlashAttentionKernelT &&other) noexcept
        {
            if (this != &other)
            {
                freeWorkspace();
                device_idx_ = other.device_idx_;
                stream_ = other.stream_;
                partial_output_buf_ = other.partial_output_buf_;
                partial_m_buf_ = other.partial_m_buf_;
                partial_l_buf_ = other.partial_l_buf_;
                workspace_size_ = other.workspace_size_;
                max_splits_ = other.max_splits_;
                other.stream_ = nullptr;
                other.partial_output_buf_ = nullptr;
                other.partial_m_buf_ = nullptr;
                other.partial_l_buf_ = nullptr;
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
            int device_idx)
        {
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

        // =====================================================================
        // BF16 Specialization Implementation (stub - delegates to FP32)
        // =====================================================================

        CUDAFlashAttentionKernelT<ActivationPrecision::BF16>::CUDAFlashAttentionKernelT(int device_idx)
            : device_idx_(device_idx), stream_(nullptr),
              partial_output_buf_(nullptr), partial_m_buf_(nullptr), partial_l_buf_(nullptr),
              workspace_size_(0), max_splits_(0)
        {
            LOG_DEBUG("[CUDAFlashAttentionKernelT<BF16>] Created for device " << device_idx);
        }

        CUDAFlashAttentionKernelT<ActivationPrecision::BF16>::~CUDAFlashAttentionKernelT()
        {
            freeWorkspace();
        }

        CUDAFlashAttentionKernelT<ActivationPrecision::BF16>::CUDAFlashAttentionKernelT(
            CUDAFlashAttentionKernelT &&other) noexcept
            : device_idx_(other.device_idx_), stream_(other.stream_),
              partial_output_buf_(other.partial_output_buf_),
              partial_m_buf_(other.partial_m_buf_),
              partial_l_buf_(other.partial_l_buf_),
              workspace_size_(other.workspace_size_),
              max_splits_(other.max_splits_)
        {
            other.stream_ = nullptr;
            other.partial_output_buf_ = nullptr;
            other.partial_m_buf_ = nullptr;
            other.partial_l_buf_ = nullptr;
        }

        CUDAFlashAttentionKernelT<ActivationPrecision::BF16> &
        CUDAFlashAttentionKernelT<ActivationPrecision::BF16>::operator=(
            CUDAFlashAttentionKernelT &&other) noexcept
        {
            if (this != &other)
            {
                freeWorkspace();
                device_idx_ = other.device_idx_;
                stream_ = other.stream_;
                partial_output_buf_ = other.partial_output_buf_;
                partial_m_buf_ = other.partial_m_buf_;
                partial_l_buf_ = other.partial_l_buf_;
                workspace_size_ = other.workspace_size_;
                max_splits_ = other.max_splits_;
                other.stream_ = nullptr;
                other.partial_output_buf_ = nullptr;
                other.partial_m_buf_ = nullptr;
                other.partial_l_buf_ = nullptr;
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
            int device_idx)
        {
            // TODO: Native BF16 path
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

    } // namespace cuda
} // namespace llaminar2
