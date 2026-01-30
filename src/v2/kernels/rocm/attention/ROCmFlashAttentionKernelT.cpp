/**
 * @file ROCmFlashAttentionKernelT.cpp
 * @brief C++ implementation of ROCm Flash Attention kernel methods
 *
 * Implements ITensorAttention interface by delegating to HIP kernels
 * defined in ROCmFlashAttentionKernels.hip.
 *
 * Target Architecture: AMD MI50 (gfx906 / Vega 20)
 *
 * @author David Sanftenberg
 */

#include "ROCmFlashAttentionKernelT.h"
#include "../../../execution/local_execution/device/DeviceWorkspaceManager.h"
#include "../../../execution/local_execution/device/WorkspaceDescriptor.h"
#include "../../../utils/Logger.h"
#include "../../../utils/KernelProfiler.h"
#include <cstring>

// Extern "C" declarations for HIP kernel wrappers
extern "C"
{
    // Flash Attention 2 prefill kernel for MI50
    int hipFlashAttn_prefill_fa2(
        const float *Q, const float *K, const float *V, float *O,
        int batch_size, int seq_len, int kv_len,
        int n_heads, int n_kv_heads, int head_dim,
        bool causal, int window_size, int position_offset,
        void *stream);

    // Flash Decoding for single-token decode with split-K parallelism
    int hipFlashAttn_decode_fp32(
        const float *Q, const float *K_cache, const float *V_cache, float *O,
        float *O_partial, float *m_partial, float *l_partial,
        int batch_size, int kv_len,
        int n_heads, int n_kv_heads, int head_dim,
        int num_splits,
        void *stream);

    int hipFlashAttn_allocWorkspace(
        void **partial_output, void **partial_m, void **partial_l,
        int batch_size, int n_heads, int head_dim, int num_splits);

    void hipFlashAttn_freeWorkspace(void *partial_output, void *partial_m, void *partial_l);

    int hipFlashAttn_setDevice(int device_idx);
    // hipFlashAttn_synchronize() removed - caller manages coherence via events
}

namespace llaminar2
{
    namespace rocm
    {
        // Default number of splits for Flash Decoding
        constexpr int DEFAULT_NUM_SPLITS = 8;

        // =====================================================================
        // FP32 Specialization Implementation
        // =====================================================================

        ROCmFlashAttentionKernelT<ActivationPrecision::FP32>::ROCmFlashAttentionKernelT(int device_idx)
            : device_idx_(device_idx), stream_(nullptr),
              partial_output_buf_(nullptr), partial_m_buf_(nullptr), partial_l_buf_(nullptr),
              workspace_size_(0), max_splits_(0)
        {
            LOG_DEBUG("[ROCmFlashAttentionKernelT<FP32>] Created for device " << device_idx);
        }

        ROCmFlashAttentionKernelT<ActivationPrecision::FP32>::~ROCmFlashAttentionKernelT()
        {
            freeWorkspace();
        }

        ROCmFlashAttentionKernelT<ActivationPrecision::FP32>::ROCmFlashAttentionKernelT(
            ROCmFlashAttentionKernelT &&other) noexcept
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

        ROCmFlashAttentionKernelT<ActivationPrecision::FP32> &
        ROCmFlashAttentionKernelT<ActivationPrecision::FP32>::operator=(
            ROCmFlashAttentionKernelT &&other) noexcept
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

        void ROCmFlashAttentionKernelT<ActivationPrecision::FP32>::allocateWorkspace(
            int n_heads, int head_dim, int num_splits)
        {
            // Workspace is now REQUIRED - no legacy allocation path
            if (!hasWorkspace())
            {
                LOG_ERROR("[ROCmFlashAttentionKernelT<FP32>] Workspace not bound - hot-path allocation disabled. "
                          "Call bindWorkspace() before allocateWorkspace()");
                partial_output_buf_ = nullptr;
                partial_m_buf_ = nullptr;
                partial_l_buf_ = nullptr;
                return;
            }

            // Use pre-allocated buffers from workspace manager
            partial_output_buf_ = workspace_->getBuffer(AttentionWorkspaceBuffers::PARTIAL_OUTPUT);
            partial_m_buf_ = workspace_->getBuffer(AttentionWorkspaceBuffers::PARTIAL_M);
            partial_l_buf_ = workspace_->getBuffer(AttentionWorkspaceBuffers::PARTIAL_L);
            max_splits_ = num_splits;

            LOG_DEBUG("[ROCmFlashAttentionKernelT<FP32>] Using managed workspace buffers");
        }

        void ROCmFlashAttentionKernelT<ActivationPrecision::FP32>::freeWorkspace()
        {
            // Workspace buffers are managed externally, just clear pointers
            partial_output_buf_ = nullptr;
            partial_m_buf_ = nullptr;
            partial_l_buf_ = nullptr;
            workspace_size_ = 0;
            max_splits_ = 0;
        }

        bool ROCmFlashAttentionKernelT<ActivationPrecision::FP32>::compute(
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

        bool ROCmFlashAttentionKernelT<ActivationPrecision::FP32>::compute_batch(
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

        bool ROCmFlashAttentionKernelT<ActivationPrecision::FP32>::compute_decode(
            const float *Q, const float *K, const float *V, float *output,
            int seq_len, int kv_len, int n_heads, int n_kv_heads, int head_dim,
            bool causal, int position_offset, int device_idx)
        {
            int dev = (device_idx >= 0) ? device_idx : device_idx_;
            return apply_typed(Q, K, V, output,
                               1, seq_len, kv_len,
                               n_heads, n_kv_heads, head_dim,
                               causal, -1, position_offset, dev);
        }

        bool ROCmFlashAttentionKernelT<ActivationPrecision::FP32>::apply_typed(
            const float *Q, const float *K, const float *V, float *output,
            int batch_size, int seq_len, int kv_len,
            int n_heads, int n_kv_heads, int head_dim,
            bool causal, int window_size, int position_offset,
            int device_idx)
        {
            KERNEL_PROFILE_SCOPE(KernelType::ATTENTION);

            if (!Q || !K || !V || !output)
            {
                LOG_ERROR("[ROCmFlashAttentionKernelT<FP32>] Null pointer input");
                return false;
            }

            if (seq_len <= 0 || kv_len <= 0 || n_heads <= 0 || head_dim <= 0)
            {
                LOG_ERROR("[ROCmFlashAttentionKernelT<FP32>] Invalid dimensions: "
                          << "seq_len=" << seq_len << " kv_len=" << kv_len
                          << " n_heads=" << n_heads << " head_dim=" << head_dim);
                return false;
            }

            // GQA validation
            if (n_heads % n_kv_heads != 0)
            {
                LOG_ERROR("[ROCmFlashAttentionKernelT<FP32>] n_heads must be divisible by n_kv_heads");
                return false;
            }

            // Head dim validation for MI50
            if (head_dim > 128)
            {
                LOG_ERROR("[ROCmFlashAttentionKernelT<FP32>] head_dim=" << head_dim
                                                                        << " exceeds MI50 kernel limit (128)");
                return false;
            }

            // Set device
            if (hipFlashAttn_setDevice(device_idx) != 0)
            {
                LOG_ERROR("[ROCmFlashAttentionKernelT<FP32>] Failed to set device " << device_idx);
                return false;
            }

            int result;

            // Choose algorithm based on seq_len
            if (seq_len == 1)
            {
                // Flash Decoding for single-token decode
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
                    LOG_ERROR("[ROCmFlashAttentionKernelT<FP32>] Workspace allocation failed");
                    return false;
                }

                LOG_DEBUG("[ROCmFlashAttentionKernelT<FP32>] Using Flash Decoding: kv_len=" << kv_len
                                                                                            << " num_splits=" << num_splits);

                result = hipFlashAttn_decode_fp32(
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
                // Flash Attention 2 for prefill
                LOG_DEBUG("[ROCmFlashAttentionKernelT<FP32>] Using Flash Attention 2 (MI50): "
                          << "batch=" << batch_size << " seq_len=" << seq_len << " kv_len=" << kv_len);

                result = hipFlashAttn_prefill_fa2(
                    Q, K, V, output,
                    batch_size, seq_len, kv_len,
                    n_heads, n_kv_heads, head_dim,
                    causal, window_size, position_offset,
                    stream_);
            }

            if (result != 0)
            {
                LOG_ERROR("[ROCmFlashAttentionKernelT<FP32>] Kernel execution failed with code " << result);
                return false;
            }

            // Removed hipFlashAttn_synchronize() - caller manages coherence via events
            return true;
        }

        bool ROCmFlashAttentionKernelT<ActivationPrecision::FP32>::compute_tensor(
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
            (void)workspace_mask;
            (void)mpi_ctx;
            (void)head_start;
            (void)local_n_heads;
            (void)local_n_kv_heads;

            if (!Q || !K || !V || !output)
            {
                LOG_ERROR("[ROCmFlashAttentionKernelT<FP32>::compute_tensor] Null tensor");
                return false;
            }

            // Check if tensors have FP32 type
            if (Q->native_type() != TensorType::FP32 || K->native_type() != TensorType::FP32 ||
                V->native_type() != TensorType::FP32 || output->native_type() != TensorType::FP32)
            {
                LOG_ERROR("[ROCmFlashAttentionKernelT<FP32>::compute_tensor] Expected FP32 tensors, got Q="
                          << Q->dtype_name() << " K=" << K->dtype_name()
                          << " V=" << V->dtype_name() << " O=" << output->dtype_name());
                return false;
            }

            // Extract GPU pointers - tensors should already be coherent on device
            const float *Q_ptr = static_cast<const float *>(Q->gpu_data_ptr());
            const float *K_ptr = static_cast<const float *>(K->gpu_data_ptr());
            const float *V_ptr = static_cast<const float *>(V->gpu_data_ptr());
            float *O_ptr = static_cast<float *>(output->gpu_data_ptr());

            if (!Q_ptr || !K_ptr || !V_ptr || !O_ptr)
            {
                LOG_ERROR("[ROCmFlashAttentionKernelT<FP32>::compute_tensor] GPU data pointer is null. "
                          << "Ensure tensors are coherent on device (ensureOnDevice).");
                return false;
            }

            int dev = (device_idx >= 0) ? device_idx : device_idx_;

            LOG_DEBUG("[ROCmFlashAttentionKernelT<FP32>::compute_tensor] batch=" << batch_size
                                                                                 << " seq_len=" << seq_len << " kv_len=" << kv_len
                                                                                 << " n_heads=" << n_heads << " n_kv_heads=" << n_kv_heads
                                                                                 << " head_dim=" << head_dim << " causal=" << causal
                                                                                 << " device_idx=" << dev);

            // Determine if this is decode (seq_len=1) or prefill
            if (seq_len == 1 && kv_len > 1)
            {
                return compute_decode(Q_ptr, K_ptr, V_ptr, O_ptr,
                                      seq_len, kv_len, n_heads, n_kv_heads, head_dim,
                                      causal, 0, dev);
            }
            else
            {
                return apply_typed(Q_ptr, K_ptr, V_ptr, O_ptr,
                                   batch_size, seq_len, kv_len,
                                   n_heads, n_kv_heads, head_dim,
                                   causal, window_size, 0, dev);
            }
        }

        // =====================================================================
        // FP32 IWorkspaceConsumer Interface Implementation
        // =====================================================================

        WorkspaceRequirements ROCmFlashAttentionKernelT<ActivationPrecision::FP32>::getWorkspaceRequirements(
            int m, int n, int k) const
        {
            WorkspaceRequirements reqs;

            // Default parameters for Flash Decoding workspace sizing
            // Conservative estimates for maximum expected configuration
            const int batch_size = (m > 0) ? m : 1;
            const int n_heads = (n > 0) ? n : 128;     // Max expected heads
            const int head_dim = (k > 0) ? k : 128;    // Max expected head dim
            const int num_splits = DEFAULT_NUM_SPLITS; // 8 splits

            // partial_output: [batch × n_heads × num_splits × head_dim] FP32
            size_t partial_output_bytes = static_cast<size_t>(batch_size) * n_heads * num_splits * head_dim * sizeof(float);

            // partial_m: [batch × n_heads × num_splits] FP32 (max scores per split)
            size_t partial_m_bytes = static_cast<size_t>(batch_size) * n_heads * num_splits * sizeof(float);

            // partial_l: [batch × n_heads × num_splits] FP32 (logsumexp per split)
            size_t partial_l_bytes = static_cast<size_t>(batch_size) * n_heads * num_splits * sizeof(float);

            reqs.buffers.push_back({AttentionWorkspaceBuffers::PARTIAL_OUTPUT, partial_output_bytes, 256, true});
            reqs.buffers.push_back({AttentionWorkspaceBuffers::PARTIAL_M, partial_m_bytes, 256, true});
            reqs.buffers.push_back({AttentionWorkspaceBuffers::PARTIAL_L, partial_l_bytes, 256, true});

            LOG_DEBUG("[ROCmFlashAttentionKernelT<FP32>::getWorkspaceRequirements] "
                      << "batch=" << batch_size << " n_heads=" << n_heads << " head_dim=" << head_dim
                      << " num_splits=" << num_splits
                      << " => partial_output=" << (partial_output_bytes / 1024) << "KB"
                      << ", partial_m=" << partial_m_bytes << "B"
                      << ", partial_l=" << partial_l_bytes << "B");

            return reqs;
        }

        void ROCmFlashAttentionKernelT<ActivationPrecision::FP32>::bindWorkspace(
            DeviceWorkspaceManager *workspace)
        {
            workspace_ = workspace;
            if (workspace)
            {
                LOG_DEBUG("[ROCmFlashAttentionKernelT<FP32>] Bound workspace manager, entering managed mode");
            }
            else
            {
                LOG_DEBUG("[ROCmFlashAttentionKernelT<FP32>] Unbound workspace, returning to legacy mode");
            }
        }

        bool ROCmFlashAttentionKernelT<ActivationPrecision::FP32>::hasWorkspace() const
        {
            return workspace_ != nullptr;
        }

        DeviceWorkspaceManager *ROCmFlashAttentionKernelT<ActivationPrecision::FP32>::getWorkspace() const
        {
            return workspace_;
        }

        // =====================================================================
        // FP16 Specialization Implementation
        // =====================================================================

        ROCmFlashAttentionKernelT<ActivationPrecision::FP16>::ROCmFlashAttentionKernelT(int device_idx)
            : device_idx_(device_idx), stream_(nullptr),
              partial_output_buf_(nullptr), partial_m_buf_(nullptr), partial_l_buf_(nullptr),
              workspace_size_(0), max_splits_(0)
        {
            LOG_DEBUG("[ROCmFlashAttentionKernelT<FP16>] Created for device " << device_idx);
        }

        ROCmFlashAttentionKernelT<ActivationPrecision::FP16>::~ROCmFlashAttentionKernelT()
        {
            freeWorkspace();
        }

        ROCmFlashAttentionKernelT<ActivationPrecision::FP16>::ROCmFlashAttentionKernelT(
            ROCmFlashAttentionKernelT &&other) noexcept
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

        ROCmFlashAttentionKernelT<ActivationPrecision::FP16> &
        ROCmFlashAttentionKernelT<ActivationPrecision::FP16>::operator=(
            ROCmFlashAttentionKernelT &&other) noexcept
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

        void ROCmFlashAttentionKernelT<ActivationPrecision::FP16>::allocateWorkspace(
            int n_heads, int head_dim, int num_splits)
        {
            // Workspace is now REQUIRED - no legacy allocation path
            if (!hasWorkspace())
            {
                LOG_ERROR("[ROCmFlashAttentionKernelT<FP16>] Workspace not bound - hot-path allocation disabled. "
                          "Call bindWorkspace() before allocateWorkspace()");
                partial_output_buf_ = nullptr;
                partial_m_buf_ = nullptr;
                partial_l_buf_ = nullptr;
                return;
            }

            partial_output_buf_ = workspace_->getBuffer(AttentionWorkspaceBuffers::PARTIAL_OUTPUT);
            partial_m_buf_ = workspace_->getBuffer(AttentionWorkspaceBuffers::PARTIAL_M);
            partial_l_buf_ = workspace_->getBuffer(AttentionWorkspaceBuffers::PARTIAL_L);
            max_splits_ = num_splits;
            LOG_DEBUG("[ROCmFlashAttentionKernelT<FP16>] Using managed workspace buffers");
        }

        void ROCmFlashAttentionKernelT<ActivationPrecision::FP16>::freeWorkspace()
        {
            // Workspace buffers are managed externally, just clear pointers
            partial_output_buf_ = nullptr;
            partial_m_buf_ = nullptr;
            partial_l_buf_ = nullptr;
            workspace_size_ = 0;
            max_splits_ = 0;
        }

        bool ROCmFlashAttentionKernelT<ActivationPrecision::FP16>::compute(
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
            // FP16 kernel with FP32 interface - convert internally
            // For now, delegate to FP32 implementation
            // TODO: Implement native FP16 path with FP32->FP16 conversion kernels
            LOG_WARN("[ROCmFlashAttentionKernelT<FP16>] Using FP32 fallback - native FP16 not yet implemented");

            (void)workspace_scores;
            (void)workspace_buffer;
            (void)workspace_context;
            (void)workspace_mask;
            (void)use_bf16;
            (void)mpi_ctx;

            int dev = (device_idx >= 0) ? device_idx : device_idx_;

            // Fall back to FP32 kernel
            return hipFlashAttn_prefill_fa2(
                       Q, K, V, output,
                       1, seq_len, seq_len,
                       n_heads, n_kv_heads, head_dim,
                       causal, window_size, 0,
                       stream_) == 0;
        }

        bool ROCmFlashAttentionKernelT<ActivationPrecision::FP16>::compute_batch(
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
            (void)device_idx;

            return hipFlashAttn_prefill_fa2(
                       Q, K, V, output,
                       batch_size, seq_len, seq_len,
                       n_heads, n_kv_heads, head_dim,
                       causal, window_size, 0,
                       stream_) == 0;
        }

        bool ROCmFlashAttentionKernelT<ActivationPrecision::FP16>::compute_decode(
            const float *Q, const float *K, const float *V, float *output,
            int seq_len, int kv_len, int n_heads, int n_kv_heads, int head_dim,
            bool causal, int position_offset, int device_idx)
        {
            (void)causal;
            (void)position_offset;

            int dev = (device_idx >= 0) ? device_idx : device_idx_;
            if (hipFlashAttn_setDevice(dev) != 0)
            {
                LOG_ERROR("[ROCmFlashAttentionKernelT<FP16>::compute_decode] Failed to set device " << dev);
                return false;
            }

            int num_splits = DEFAULT_NUM_SPLITS;
            if (kv_len <= 64)
                num_splits = 1;
            else if (kv_len < 128)
                num_splits = 2;
            else if (kv_len < 256)
                num_splits = 4;

            allocateWorkspace(n_heads, head_dim, num_splits);

            return hipFlashAttn_decode_fp32(
                       Q, K, V, output,
                       static_cast<float *>(partial_output_buf_),
                       static_cast<float *>(partial_m_buf_),
                       static_cast<float *>(partial_l_buf_),
                       1, kv_len,
                       n_heads, n_kv_heads, head_dim,
                       num_splits, stream_) == 0;
        }
        bool ROCmFlashAttentionKernelT<ActivationPrecision::FP16>::apply_typed(
            const uint16_t *Q, const uint16_t *K, const uint16_t *V, uint16_t *output,
            int batch_size, int seq_len, int kv_len,
            int n_heads, int n_kv_heads, int head_dim,
            bool causal, int window_size, int position_offset,
            int device_idx)
        {
            // TODO: Implement native FP16 kernel
            // For now, this would require conversion - not implemented
            LOG_ERROR("[ROCmFlashAttentionKernelT<FP16>::apply_typed] Native FP16 not implemented");
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
            return false;
        }

        bool ROCmFlashAttentionKernelT<ActivationPrecision::FP16>::compute_tensor(
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
            // TODO: Implement FP16 tensor path
            LOG_ERROR("[ROCmFlashAttentionKernelT<FP16>::compute_tensor] Not implemented");
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
            (void)workspace_scores;
            (void)workspace_mask;
            (void)mpi_ctx;
            (void)device_idx;
            (void)head_start;
            (void)local_n_heads;
            (void)local_n_kv_heads;
            return false;
        }

        // =====================================================================
        // FP16 IWorkspaceConsumer Interface Implementation
        // =====================================================================

        WorkspaceRequirements ROCmFlashAttentionKernelT<ActivationPrecision::FP16>::getWorkspaceRequirements(
            int m, int n, int k) const
        {
            WorkspaceRequirements reqs;

            const int batch_size = (m > 0) ? m : 1;
            const int n_heads = (n > 0) ? n : 128;
            const int head_dim = (k > 0) ? k : 128;
            const int num_splits = DEFAULT_NUM_SPLITS;

            // FP16 uses FP32 workspace for numerical stability
            size_t partial_output_bytes = static_cast<size_t>(batch_size) * n_heads * num_splits * head_dim * sizeof(float);
            size_t partial_m_bytes = static_cast<size_t>(batch_size) * n_heads * num_splits * sizeof(float);
            size_t partial_l_bytes = static_cast<size_t>(batch_size) * n_heads * num_splits * sizeof(float);

            reqs.buffers.push_back({AttentionWorkspaceBuffers::PARTIAL_OUTPUT, partial_output_bytes, 256, true});
            reqs.buffers.push_back({AttentionWorkspaceBuffers::PARTIAL_M, partial_m_bytes, 256, true});
            reqs.buffers.push_back({AttentionWorkspaceBuffers::PARTIAL_L, partial_l_bytes, 256, true});

            LOG_DEBUG("[ROCmFlashAttentionKernelT<FP16>::getWorkspaceRequirements] "
                      << "partial_output=" << (partial_output_bytes / 1024) << "KB");

            return reqs;
        }

        void ROCmFlashAttentionKernelT<ActivationPrecision::FP16>::bindWorkspace(
            DeviceWorkspaceManager *workspace)
        {
            workspace_ = workspace;
            if (workspace)
            {
                LOG_DEBUG("[ROCmFlashAttentionKernelT<FP16>] Bound workspace manager, entering managed mode");
            }
            else
            {
                LOG_DEBUG("[ROCmFlashAttentionKernelT<FP16>] Unbound workspace, returning to legacy mode");
            }
        }

        bool ROCmFlashAttentionKernelT<ActivationPrecision::FP16>::hasWorkspace() const
        {
            return workspace_ != nullptr;
        }

        DeviceWorkspaceManager *ROCmFlashAttentionKernelT<ActivationPrecision::FP16>::getWorkspace() const
        {
            return workspace_;
        }

        // =====================================================================
        // BF16 Specialization Implementation
        // =====================================================================

        ROCmFlashAttentionKernelT<ActivationPrecision::BF16>::ROCmFlashAttentionKernelT(int device_idx)
            : device_idx_(device_idx), stream_(nullptr),
              partial_output_buf_(nullptr), partial_m_buf_(nullptr), partial_l_buf_(nullptr),
              workspace_size_(0), max_splits_(0)
        {
            // Note: MI50 has limited BF16 support - may fall back to FP32
            LOG_DEBUG("[ROCmFlashAttentionKernelT<BF16>] Created for device " << device_idx
                                                                              << " (Note: MI50 has limited BF16 support)");
        }

        ROCmFlashAttentionKernelT<ActivationPrecision::BF16>::~ROCmFlashAttentionKernelT()
        {
            freeWorkspace();
        }

        ROCmFlashAttentionKernelT<ActivationPrecision::BF16>::ROCmFlashAttentionKernelT(
            ROCmFlashAttentionKernelT &&other) noexcept
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

        ROCmFlashAttentionKernelT<ActivationPrecision::BF16> &
        ROCmFlashAttentionKernelT<ActivationPrecision::BF16>::operator=(
            ROCmFlashAttentionKernelT &&other) noexcept
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

        void ROCmFlashAttentionKernelT<ActivationPrecision::BF16>::allocateWorkspace(
            int n_heads, int head_dim, int num_splits)
        {
            // Workspace is now REQUIRED - no legacy allocation path
            if (!hasWorkspace())
            {
                LOG_ERROR("[ROCmFlashAttentionKernelT<BF16>] Workspace not bound - hot-path allocation disabled. "
                          "Call bindWorkspace() before allocateWorkspace()");
                partial_output_buf_ = nullptr;
                partial_m_buf_ = nullptr;
                partial_l_buf_ = nullptr;
                return;
            }

            partial_output_buf_ = workspace_->getBuffer(AttentionWorkspaceBuffers::PARTIAL_OUTPUT);
            partial_m_buf_ = workspace_->getBuffer(AttentionWorkspaceBuffers::PARTIAL_M);
            partial_l_buf_ = workspace_->getBuffer(AttentionWorkspaceBuffers::PARTIAL_L);
            max_splits_ = num_splits;
            LOG_DEBUG("[ROCmFlashAttentionKernelT<BF16>] Using managed workspace buffers");
        }

        void ROCmFlashAttentionKernelT<ActivationPrecision::BF16>::freeWorkspace()
        {
            // Workspace buffers are managed externally, just clear pointers
            partial_output_buf_ = nullptr;
            partial_m_buf_ = nullptr;
            partial_l_buf_ = nullptr;
            workspace_size_ = 0;
            max_splits_ = 0;
        }

        bool ROCmFlashAttentionKernelT<ActivationPrecision::BF16>::compute(
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
            // MI50 doesn't have native BF16 support - fall back to FP32
            LOG_WARN("[ROCmFlashAttentionKernelT<BF16>] MI50 lacks BF16 support - using FP32 fallback");

            (void)workspace_scores;
            (void)workspace_buffer;
            (void)workspace_context;
            (void)workspace_mask;
            (void)use_bf16;
            (void)mpi_ctx;

            return hipFlashAttn_prefill_fa2(
                       Q, K, V, output,
                       1, seq_len, seq_len,
                       n_heads, n_kv_heads, head_dim,
                       causal, window_size, 0,
                       stream_) == 0;
        }

        bool ROCmFlashAttentionKernelT<ActivationPrecision::BF16>::compute_batch(
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
            (void)device_idx;

            return hipFlashAttn_prefill_fa2(
                       Q, K, V, output,
                       batch_size, seq_len, seq_len,
                       n_heads, n_kv_heads, head_dim,
                       causal, window_size, 0,
                       stream_) == 0;
        }

        bool ROCmFlashAttentionKernelT<ActivationPrecision::BF16>::compute_decode(
            const float *Q, const float *K, const float *V, float *output,
            int seq_len, int kv_len, int n_heads, int n_kv_heads, int head_dim,
            bool causal, int position_offset, int device_idx)
        {
            (void)causal;
            (void)position_offset;

            int dev = (device_idx >= 0) ? device_idx : device_idx_;
            if (hipFlashAttn_setDevice(dev) != 0)
            {
                LOG_ERROR("[ROCmFlashAttentionKernelT<BF16>::compute_decode] Failed to set device " << dev);
                return false;
            }

            int num_splits = DEFAULT_NUM_SPLITS;
            if (kv_len <= 64)
                num_splits = 1;
            else if (kv_len < 128)
                num_splits = 2;
            else if (kv_len < 256)
                num_splits = 4;

            allocateWorkspace(n_heads, head_dim, num_splits);

            return hipFlashAttn_decode_fp32(
                       Q, K, V, output,
                       static_cast<float *>(partial_output_buf_),
                       static_cast<float *>(partial_m_buf_),
                       static_cast<float *>(partial_l_buf_),
                       1, kv_len,
                       n_heads, n_kv_heads, head_dim,
                       num_splits, stream_) == 0;
        }

        bool ROCmFlashAttentionKernelT<ActivationPrecision::BF16>::apply_typed(
            const uint16_t *Q, const uint16_t *K, const uint16_t *V, uint16_t *output,
            int batch_size, int seq_len, int kv_len,
            int n_heads, int n_kv_heads, int head_dim,
            bool causal, int window_size, int position_offset,
            int device_idx)
        {
            LOG_ERROR("[ROCmFlashAttentionKernelT<BF16>::apply_typed] Native BF16 not supported on MI50");
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
            return false;
        }

        bool ROCmFlashAttentionKernelT<ActivationPrecision::BF16>::compute_tensor(
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
            LOG_ERROR("[ROCmFlashAttentionKernelT<BF16>::compute_tensor] Not implemented for MI50");
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
            (void)workspace_scores;
            (void)workspace_mask;
            (void)mpi_ctx;
            (void)device_idx;
            (void)head_start;
            (void)local_n_heads;
            (void)local_n_kv_heads;
            return false;
        }

        // =====================================================================
        // BF16 IWorkspaceConsumer Interface Implementation
        // =====================================================================

        WorkspaceRequirements ROCmFlashAttentionKernelT<ActivationPrecision::BF16>::getWorkspaceRequirements(
            int m, int n, int k) const
        {
            WorkspaceRequirements reqs;

            const int batch_size = (m > 0) ? m : 1;
            const int n_heads = (n > 0) ? n : 128;
            const int head_dim = (k > 0) ? k : 128;
            const int num_splits = DEFAULT_NUM_SPLITS;

            // BF16 on MI50 falls back to FP32, so workspace is FP32
            size_t partial_output_bytes = static_cast<size_t>(batch_size) * n_heads * num_splits * head_dim * sizeof(float);
            size_t partial_m_bytes = static_cast<size_t>(batch_size) * n_heads * num_splits * sizeof(float);
            size_t partial_l_bytes = static_cast<size_t>(batch_size) * n_heads * num_splits * sizeof(float);

            reqs.buffers.push_back({AttentionWorkspaceBuffers::PARTIAL_OUTPUT, partial_output_bytes, 256, true});
            reqs.buffers.push_back({AttentionWorkspaceBuffers::PARTIAL_M, partial_m_bytes, 256, true});
            reqs.buffers.push_back({AttentionWorkspaceBuffers::PARTIAL_L, partial_l_bytes, 256, true});

            LOG_DEBUG("[ROCmFlashAttentionKernelT<BF16>::getWorkspaceRequirements] "
                      << "partial_output=" << (partial_output_bytes / 1024) << "KB");

            return reqs;
        }

        void ROCmFlashAttentionKernelT<ActivationPrecision::BF16>::bindWorkspace(
            DeviceWorkspaceManager *workspace)
        {
            workspace_ = workspace;
            if (workspace)
            {
                LOG_DEBUG("[ROCmFlashAttentionKernelT<BF16>] Bound workspace manager, entering managed mode");
            }
            else
            {
                LOG_DEBUG("[ROCmFlashAttentionKernelT<BF16>] Unbound workspace, returning to legacy mode");
            }
        }

        bool ROCmFlashAttentionKernelT<ActivationPrecision::BF16>::hasWorkspace() const
        {
            return workspace_ != nullptr;
        }

        DeviceWorkspaceManager *ROCmFlashAttentionKernelT<ActivationPrecision::BF16>::getWorkspace() const
        {
            return workspace_;
        }

    } // namespace rocm
} // namespace llaminar2
