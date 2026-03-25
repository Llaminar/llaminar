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
#include "../../../utils/Logger.h"
#include "../../../utils/CUDAKernelProfiler.h"
#include "../../attention/AttentionDeviceParams.h"
#include <cuda_runtime_api.h>
#include <cstring>
#include <stdexcept>

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
        void *stream,
        int device_idx);

    // FA2 with FP16 K/V inputs — skips FP16→FP32→FP16 round-trip when KV cache is FP16
    int cudaFlashAttn_prefill_fa2_fp16kv(
        const float *Q, const void *K_fp16, const void *V_fp16, float *O,
        int batch_size, int seq_len, int kv_len,
        int n_heads, int n_kv_heads, int head_dim,
        bool causal, int window_size, int position_offset,
        const llaminar2::attention::AttentionDeviceParams *device_params,
        const float *mask,
        void *stream,
        int device_idx);

    // cuBLAS-based unfused attention for prefill with FP16 KV
    int cudaFlashAttn_prefill_cublas_fp16kv(
        const float *Q, const void *K_fp16, const void *V_fp16, float *O,
        int batch_size, int seq_len, int kv_len,
        int n_heads, int n_kv_heads, int head_dim,
        bool causal, int position_offset,
        void *stream,
        int device_idx);

    // Flash Decoding for single-token decode with split-K parallelism
    int cudaFlashAttn_decode_fp32(
        const float *Q, const float *K_cache, const float *V_cache, float *O,
        float *O_partial, float *m_partial, float *l_partial,
        int batch_size, int kv_len,
        int n_heads, int n_kv_heads, int head_dim,
        int num_splits,
        const llaminar2::attention::AttentionDeviceParams *device_params,
        void *stream,
        int device_idx);

    // Flash Decoding with FP16 KV cache — reads half directly, no conversion
    int cudaFlashAttn_decode_fp16kv(
        const float *Q, const void *K_cache_fp16, const void *V_cache_fp16, float *O,
        float *O_partial, float *m_partial, float *l_partial,
        int batch_size, int kv_len,
        int n_heads, int n_kv_heads, int head_dim,
        int num_splits,
        const llaminar2::attention::AttentionDeviceParams *device_params,
        void *stream,
        int device_idx);

    int cudaFlashAttn_allocWorkspace(
        void **partial_output, void **partial_m, void **partial_l,
        int batch_size, int n_heads, int head_dim, int num_splits);

    void cudaFlashAttn_freeWorkspace(void *partial_output, void *partial_m, void *partial_l);

    int cudaFlashAttn_setDevice(int device_idx);
    int cudaFlashAttn_synchronize();

    // GPU-side KV cache conversion
    int cudaFlashAttn_convert_fp16_to_fp32(const void *src, float *dst, int count, void *stream);
    int cudaFlashAttn_dequant_q8_1_to_fp32(const void *src, float *dst, int rows, int cols, void *stream);

    // Dynamic versions for CUDA graph replay (read kv_len from device_params at runtime)
    int cudaFlashAttn_convert_fp16_to_fp32_dynamic(
        const void *src, float *dst,
        int cols_per_row, int max_kv_len,
        const void *device_params, void *stream);
    int cudaFlashAttn_dequant_q8_1_to_fp32_dynamic(
        const void *src, float *dst,
        int cols, int max_kv_len,
        const void *device_params, void *stream);
}

namespace llaminar2
{
    namespace cuda
    {
        // Maximum number of splits for Flash Decoding
        constexpr int MAX_NUM_SPLITS = 32;

        // Minimum KV positions per split to avoid excessive overhead
        constexpr int MIN_KV_PER_SPLIT = 16;

        /**
         * @brief Compute optimal num_splits for Flash Decoding based on GPU SM count.
         *
         * The grid is (n_heads, num_splits, batch). With __launch_bounds__(256, 4) and
         * 61 regs/thread, max concurrent blocks = num_sms × 4. We pick num_splits such
         * that total grid fits within this capacity (avoiding an inefficient tail wave).
         *
         * For Qwen2.5-7B (28 heads) on RTX 3090 (82 SMs): 328/28 = 11 splits = 308 blocks.
         */
        static int computeNumSplitsForDevice(int kv_len, int n_heads, int device_idx)
        {
            if (kv_len <= 1)
                return 1;

            // Get SM count (cached per device via static array)
            static int sm_count_cache[8] = {0};
            int num_sms = sm_count_cache[device_idx & 7];
            if (num_sms == 0)
            {
                cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device_idx);
                if (num_sms <= 0)
                    num_sms = 82; // Conservative fallback
                sm_count_cache[device_idx & 7] = num_sms;
            }

            // Max concurrent blocks from register pressure (61 regs → 4 blocks/SM)
            constexpr int MAX_BLOCKS_PER_SM = 4;
            int max_concurrent = MAX_BLOCKS_PER_SM * num_sms;

            // Floor division: stay at or below capacity to avoid tail-wave inefficiency
            int desired_splits = max_concurrent / n_heads;

            // Clamp: each split needs enough work (MIN_KV_PER_SPLIT positions)
            int max_splits_by_kv = kv_len / MIN_KV_PER_SPLIT;
            if (max_splits_by_kv < 1)
                max_splits_by_kv = 1;

            int num_splits = desired_splits;
            if (num_splits > max_splits_by_kv)
                num_splits = max_splits_by_kv;
            if (num_splits > MAX_NUM_SPLITS)
                num_splits = MAX_NUM_SPLITS;
            if (num_splits < 1)
                num_splits = 1;

            return num_splits;
        }

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

            // Always set the active device — required for workspace allocation (cudaMalloc)
            // and cudaFuncSetAttribute calls which operate on the current device, not the stream's device.
            // In multi-GPU TP mode, the thread-local device may be wrong.
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
                int num_splits = computeNumSplitsForDevice(kv_len, n_heads, device_idx);

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
                        num_splits, device_params, stream_,
                        device_idx);
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
                        stream_,
                        device_idx);
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

            // === EARLY device_params setup ===
            // Wire device_params for graph-capture replay: H2D memcpy from pinned host
            // memory is captured as a graph node. On replay, the memcpy re-reads
            // the updated pinned values (kv_len, position_offset, mask_stride).
            //
            // IMPORTANT: This MUST be done BEFORE KV conversion so that the dynamic
            // conversion kernels (used during graph capture) can read the current
            // kv_len from device_params. Without this ordering, the captured graph
            // would freeze the conversion element count, causing stale data on replay.
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

            // === Mixed-precision KV handling (FP16→direct or Q8_1→FP32) ===
            // For FP16 KV: both prefill (seq_len > 1) and decode (seq_len == 1)
            // use direct FP16 kernel variants, eliminating FP16→FP32 conversion.
            bool use_fp16kv_direct = false;
            const void *K_fp16_ptr = nullptr;
            const void *V_fp16_ptr = nullptr;

            if (K->native_type() != TensorType::FP32 || V->native_type() != TensorType::FP32)
            {
                if (K->native_type() != V->native_type())
                {
                    LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>::compute_tensor] Mixed K/V types not supported: K="
                              << K->dtype_name() << " V=" << V->dtype_name());
                    return false;
                }

                if (K->native_type() == TensorType::FP16)
                {
                    // FAST PATH: FP16 KV → pass FP16 pointers directly to kernel
                    // Works for both prefill (FA2 FP16KV) and decode (flash_decoding_fp16kv)
                    use_fp16kv_direct = true;
                    K_fp16_ptr = K->gpu_data_ptr();
                    V_fp16_ptr = V->gpu_data_ptr();
                    LOG_DEBUG("[CUDAFlashAttentionKernelT<FP32>::compute_tensor] FP16 KV direct path (no conversion)");
                }
                else
                {
                    // CONVERSION PATH: FP16 decode or Q8_1 → convert to FP32 workspace
                    if (!workspace_)
                    {
                        LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>::compute_tensor] Mixed-precision KV requires workspace-bound conversion buffers");
                        return false;
                    }

                    const size_t rows = static_cast<size_t>(batch_size) * static_cast<size_t>(kv_len);
                    const size_t logical_cols = static_cast<size_t>(n_kv_heads) * static_cast<size_t>(head_dim);
                    const size_t logical_elements = rows * logical_cols;

                    float *d_k_tmp = static_cast<float *>(workspace_->getBuffer(AttentionWorkspaceBuffers::K_TMP_FP32));
                    float *d_v_tmp = static_cast<float *>(workspace_->getBuffer(AttentionWorkspaceBuffers::V_TMP_FP32));
                    if (!d_k_tmp || !d_v_tmp)
                    {
                        LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>::compute_tensor] Missing workspace conversion buffers: "
                                  << AttentionWorkspaceBuffers::K_TMP_FP32 << " / "
                                  << AttentionWorkspaceBuffers::V_TMP_FP32);
                        return false;
                    }

                    if (K->native_type() == TensorType::FP16)
                    {
                        int k_ret, v_ret;
                        if (d_attn_params)
                        {
                            constexpr int MAX_KV_LEN = 4096;
                            k_ret = cudaFlashAttn_convert_fp16_to_fp32_dynamic(
                                K->gpu_data_ptr(), d_k_tmp,
                                static_cast<int>(logical_cols), MAX_KV_LEN,
                                d_attn_params, stream_);
                            v_ret = cudaFlashAttn_convert_fp16_to_fp32_dynamic(
                                V->gpu_data_ptr(), d_v_tmp,
                                static_cast<int>(logical_cols), MAX_KV_LEN,
                                d_attn_params, stream_);
                        }
                        else
                        {
                            k_ret = cudaFlashAttn_convert_fp16_to_fp32(
                                K->gpu_data_ptr(), d_k_tmp,
                                static_cast<int>(logical_elements), stream_);
                            v_ret = cudaFlashAttn_convert_fp16_to_fp32(
                                V->gpu_data_ptr(), d_v_tmp,
                                static_cast<int>(logical_elements), stream_);
                        }
                        if (k_ret != 0 || v_ret != 0)
                        {
                            LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>::compute_tensor] GPU FP16→FP32 conversion failed"
                                      << " k_ret=" << k_ret << " v_ret=" << v_ret
                                      << " dynamic=" << (d_attn_params != nullptr));
                            return false;
                        }
                    }
                    else if (K->native_type() == TensorType::Q8_1)
                    {
                        int k_ret, v_ret;
                        if (d_attn_params)
                        {
                            constexpr int MAX_KV_LEN = 4096;
                            k_ret = cudaFlashAttn_dequant_q8_1_to_fp32_dynamic(
                                K->gpu_data_ptr(), d_k_tmp,
                                static_cast<int>(logical_cols), MAX_KV_LEN,
                                d_attn_params, stream_);
                            v_ret = cudaFlashAttn_dequant_q8_1_to_fp32_dynamic(
                                V->gpu_data_ptr(), d_v_tmp,
                                static_cast<int>(logical_cols), MAX_KV_LEN,
                                d_attn_params, stream_);
                        }
                        else
                        {
                            k_ret = cudaFlashAttn_dequant_q8_1_to_fp32(
                                K->gpu_data_ptr(), d_k_tmp,
                                static_cast<int>(rows), static_cast<int>(logical_cols), stream_);
                            v_ret = cudaFlashAttn_dequant_q8_1_to_fp32(
                                V->gpu_data_ptr(), d_v_tmp,
                                static_cast<int>(rows), static_cast<int>(logical_cols), stream_);
                        }
                        if (k_ret != 0 || v_ret != 0)
                        {
                            LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>::compute_tensor] GPU Q8_1→FP32 dequantization failed"
                                      << " k_ret=" << k_ret << " v_ret=" << v_ret
                                      << " dynamic=" << (d_attn_params != nullptr));
                            return false;
                        }
                    }
                    else
                    {
                        LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>::compute_tensor] Unsupported KV tensor type for FP32 attention: "
                                  << K->dtype_name());
                        return false;
                    }

                    K_ptr = d_k_tmp;
                    V_ptr = d_v_tmp;
                }
            }

            if (!Q_ptr || !K_ptr || !V_ptr || !output_ptr)
            {
                LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>::compute_tensor] GPU data pointer is null. "
                          << "Ensure tensors are coherent on device (ensureOnDevice).");
                return false;
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

            // Dispatch: FP16 KV direct path (prefill and decode) or standard apply_typed
            if (use_fp16kv_direct)
            {
                // FP16 KV direct: skip FP32 conversion, pass FP16 pointers to kernel
                if (cudaFlashAttn_setDevice(dev) != 0)
                {
                    LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>] Failed to set device " << dev);
                    return false;
                }

                if (seq_len == 1)
                {
                    // DECODE: Flash Decoding with FP16 KV — no conversion needed
                    int num_splits = computeNumSplitsForDevice(kv_len, n_heads, dev);

                    allocateWorkspace(n_heads, head_dim, num_splits);

                    if (!partial_output_buf_ || !partial_m_buf_ || !partial_l_buf_)
                    {
                        LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>] Workspace allocation failed for FP16KV decode");
                        return false;
                    }

                    int result;
                    {
                        CUDA_KERNEL_PROFILE_SCOPE(CUDAKernelType::FLASH_ATTN_DECODE);
                        result = cudaFlashAttn_decode_fp16kv(
                            Q_ptr, K_fp16_ptr, V_fp16_ptr, output_ptr,
                            static_cast<float *>(partial_output_buf_),
                            static_cast<float *>(partial_m_buf_),
                            static_cast<float *>(partial_l_buf_),
                            batch_size, kv_len,
                            n_heads, n_kv_heads, head_dim,
                            num_splits, d_attn_params, stream_, dev);
                    }
                    if (result != 0)
                    {
                        LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>] Flash Decoding FP16KV failed");
                        return false;
                    }
                    return true;
                }

                // PREFILL: FA2 with FP16 KV

                // Check for cuBLAS attention override (env: LLAMINAR_CUBLAS_ATTN=1)
                static const int use_cublas_attn = []()
                {
                    const char *env = std::getenv("LLAMINAR_CUBLAS_ATTN");
                    return (env && atoi(env) == 1) ? 1 : 0;
                }();

                if (use_cublas_attn && n_heads % n_kv_heads == 0)
                {
                    int result;
                    {
                        CUDA_KERNEL_PROFILE_SCOPE(CUDAKernelType::FLASH_ATTN_PREFILL);
                        result = cudaFlashAttn_prefill_cublas_fp16kv(
                            Q_ptr, K_fp16_ptr, V_fp16_ptr, output_ptr,
                            batch_size, seq_len, kv_len,
                            n_heads, n_kv_heads, head_dim,
                            causal, 0, // position_offset
                            stream_, dev);
                    }
                    if (result != 0)
                    {
                        LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>] cuBLAS attention failed");
                        return false;
                    }
                    return true;
                }

                int result;
                {
                    CUDA_KERNEL_PROFILE_SCOPE(CUDAKernelType::FLASH_ATTN_PREFILL);
                    result = cudaFlashAttn_prefill_fa2_fp16kv(
                        Q_ptr, K_fp16_ptr, V_fp16_ptr, output_ptr,
                        batch_size, seq_len, kv_len,
                        n_heads, n_kv_heads, head_dim,
                        causal, window_size, 0,
                        d_attn_params, mask_ptr,
                        stream_, dev);
                }
                if (result != 0)
                {
                    LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>] FA2 FP16KV kernel failed");
                    return false;
                }
                return true;
            }

            // Standard path: K/V are FP32 (either native or converted)
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
            const int num_splits = MAX_NUM_SPLITS; // Conservative: allocate for max possible splits
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
