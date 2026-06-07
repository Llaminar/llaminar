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
#include "../../../utils/DebugEnv.h"
#include "../../attention/AttentionDeviceParams.h"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cstring>
#include <stdexcept>

// Extern "C" declarations for CUDA kernel wrappers
extern "C"
{
    // FA2-style pipelined prefill with WMMA (Ampere SM >= 8.0)
    // Supports head_dim=64 (6 consumer warps), head_dim=128 (4 consumer warps), and head_dim=256 (2 consumer warps)
    int cudaFlashAttn_prefill_fa2(
        const float *Q, const float *K, const float *V, float *O,
        int batch_size, int seq_len, int kv_len,
        int n_heads, int n_kv_heads, int head_dim,
        bool causal, int window_size, int position_offset,
        const llaminar2::attention::AttentionDeviceParams *device_params,
        const float *mask,
        void *stream,
        int device_idx,
        int head_start,
        int gqa_n_rep);

    // FA2 with FP16 K/V inputs — skips FP16→FP32→FP16 round-trip when KV cache is FP16
    int cudaFlashAttn_prefill_fa2_fp16kv(
        const float *Q, const void *K_fp16, const void *V_fp16, float *O,
        int batch_size, int seq_len, int kv_len,
        int n_heads, int n_kv_heads, int head_dim,
        bool causal, int window_size, int position_offset,
        const llaminar2::attention::AttentionDeviceParams *device_params,
        const float *mask,
        void *stream,
        int device_idx,
        int head_start,
        int gqa_n_rep);

    // Flash Decoding for single-token decode with split-K parallelism
    int cudaFlashAttn_decode_fp32(
        const float *Q, const float *K_cache, const float *V_cache, float *O,
        float *O_partial, float *m_partial, float *l_partial,
        int batch_size, int kv_len,
        int n_heads, int n_kv_heads, int head_dim,
        int num_splits,
        const llaminar2::attention::AttentionDeviceParams *device_params,
        void *stream,
        int device_idx,
        int head_start,
        int gqa_n_rep);

    // Flash Decoding with FP16 KV cache — reads half directly, no conversion
    int cudaFlashAttn_decode_fp16kv(
        const float *Q, const void *K_cache_fp16, const void *V_cache_fp16, float *O,
        float *O_partial, float *m_partial, float *l_partial,
        int batch_size, int kv_len,
        int n_heads, int n_kv_heads, int head_dim,
        int num_splits,
        const llaminar2::attention::AttentionDeviceParams *device_params,
        void *stream,
        int device_idx,
        int head_start,
        int gqa_n_rep);

    // Flash Decoding with Q8_1 KV cache — fused inline dequant, no workspace
    int cudaFlashAttn_decode_q8_1(
        const float *Q, const void *K_cache_q8, const void *V_cache_q8, float *O,
        float *O_partial, float *m_partial, float *l_partial,
        int batch_size, int kv_len,
        int n_heads, int n_kv_heads, int head_dim,
        int num_splits,
        const llaminar2::attention::AttentionDeviceParams *device_params,
        void *stream,
        int device_idx,
        int head_start,
        int gqa_n_rep);

    // Flash Decoding with TQ8 K / TQ4 V — fused rotation + centroid attention
    int cudaFlashAttn_decode_tqkv(
        const float *Q, float *O,
        float *O_partial, float *m_partial, float *l_partial,
        const void *K_cache, const void *V_cache,
        const float *rotation, const float *rotation_t,
        int batch_size, int kv_count,
        int n_heads, int n_kv_heads, int head_dim,
        int num_splits,
        int max_seq_len, int tail,
        int k_block_size, int v_block_size,
        const llaminar2::attention::AttentionDeviceParams *device_params,
        void *stream,
        int device_idx,
        int head_start,
        int gqa_n_rep);

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

        constexpr int MAX_SMALL_DECODE_ROWS = 4;

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
            if (debugEnv().gemm.deterministic)
                return 1;

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

        static int sanitizeSmallDecodeQueryRows(int query_rows)
        {
            return (query_rows > 1 && query_rows <= MAX_SMALL_DECODE_ROWS) ? query_rows : 1;
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
                h_attn_params_capacity_ = 0;
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
              h_attn_params_(other.h_attn_params_),
              h_attn_params_capacity_(other.h_attn_params_capacity_),
              dynamic_attn_kv_len_(other.dynamic_attn_kv_len_),
              dynamic_attn_position_offset_(other.dynamic_attn_position_offset_),
              dynamic_attn_query_rows_(other.dynamic_attn_query_rows_),
              dynamic_attn_param_rows_(other.dynamic_attn_param_rows_),
              dynamic_attn_host_valid_(other.dynamic_attn_host_valid_),
              dynamic_attn_device_valid_(other.dynamic_attn_device_valid_)
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
            other.h_attn_params_capacity_ = 0;
            other.dynamic_attn_host_valid_ = false;
            other.dynamic_attn_device_valid_ = false;
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
                    h_attn_params_capacity_ = 0;
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
                h_attn_params_capacity_ = other.h_attn_params_capacity_;
                dynamic_attn_kv_len_ = other.dynamic_attn_kv_len_;
                dynamic_attn_position_offset_ = other.dynamic_attn_position_offset_;
                dynamic_attn_query_rows_ = other.dynamic_attn_query_rows_;
                dynamic_attn_param_rows_ = other.dynamic_attn_param_rows_;
                dynamic_attn_host_valid_ = other.dynamic_attn_host_valid_;
                dynamic_attn_device_valid_ = other.dynamic_attn_device_valid_;

                other.stream_ = nullptr;
                other.partial_output_buf_ = nullptr;
                other.partial_m_buf_ = nullptr;
                other.partial_l_buf_ = nullptr;
                other.workspace_size_ = 0;
                other.max_splits_ = 0;
                other.workspace_ = nullptr;
                other.device_ctx_ = nullptr;
                other.h_attn_params_ = nullptr;
                other.h_attn_params_capacity_ = 0;
                other.dynamic_attn_host_valid_ = false;
                other.dynamic_attn_device_valid_ = false;
            }
            return *this;
        }

        bool CUDAFlashAttentionKernelT<ActivationPrecision::FP32>::allocateWorkspace(
            int n_heads, int head_dim, int num_splits)
        {
            if (num_splits <= max_splits_ && partial_output_buf_ != nullptr)
            {
                return true; // Already have enough workspace
            }

            freeWorkspace();

            if (!workspace_)
            {
                LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>] Flash decode requires bound graph workspace");
                return false;
            }

            partial_output_buf_ = workspace_->getBuffer(AttentionWorkspaceBuffers::PARTIAL_OUTPUT);
            partial_m_buf_ = workspace_->getBuffer(AttentionWorkspaceBuffers::PARTIAL_M);
            partial_l_buf_ = workspace_->getBuffer(AttentionWorkspaceBuffers::PARTIAL_L);

            const size_t required_partial_output =
                static_cast<size_t>(std::max(1, n_heads)) *
                static_cast<size_t>(std::max(1, num_splits)) *
                static_cast<size_t>(std::max(1, head_dim)) * sizeof(float);
            const size_t required_partial_meta =
                static_cast<size_t>(std::max(1, n_heads)) *
                static_cast<size_t>(std::max(1, num_splits)) * sizeof(float);
            if (!partial_output_buf_ || !partial_m_buf_ || !partial_l_buf_ ||
                workspace_->getBufferSize(AttentionWorkspaceBuffers::PARTIAL_OUTPUT) < required_partial_output ||
                workspace_->getBufferSize(AttentionWorkspaceBuffers::PARTIAL_M) < required_partial_meta ||
                workspace_->getBufferSize(AttentionWorkspaceBuffers::PARTIAL_L) < required_partial_meta)
            {
                LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>] Bound attention workspace is too small: "
                          << "required partial_output=" << required_partial_output
                          << " partial_m/l=" << required_partial_meta
                          << " bytes for n_heads=" << n_heads
                          << " head_dim=" << head_dim
                          << " num_splits=" << num_splits
                          << "; available partial_output="
                          << workspace_->getBufferSize(AttentionWorkspaceBuffers::PARTIAL_OUTPUT)
                          << " partial_m="
                          << workspace_->getBufferSize(AttentionWorkspaceBuffers::PARTIAL_M)
                          << " partial_l="
                          << workspace_->getBufferSize(AttentionWorkspaceBuffers::PARTIAL_L));
                partial_output_buf_ = nullptr;
                partial_m_buf_ = nullptr;
                partial_l_buf_ = nullptr;
                return false;
            }

            max_splits_ = num_splits;
            workspace_size_ = required_partial_output + required_partial_meta + required_partial_meta;
            LOG_DEBUG("[CUDAFlashAttentionKernelT<FP32>] Using managed attention workspace: "
                      << workspace_size_ << " bytes");
            return true;
        }

        void CUDAFlashAttentionKernelT<ActivationPrecision::FP32>::freeWorkspace()
        {
            partial_output_buf_ = nullptr;
            partial_m_buf_ = nullptr;
            partial_l_buf_ = nullptr;
            workspace_size_ = 0;
            max_splits_ = 0;
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
            const IMPIContext *mpi_ctx,
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
            const IMPIContext *mpi_ctx,
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
            const float *mask,
            int head_start,
            int gqa_n_rep)
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

            // Always set the active device — required for workspace binding checks and
            // cudaFuncSetAttribute calls which operate on the current device, not the stream's device.
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

                if (!allocateWorkspace(n_heads, head_dim, num_splits))
                {
                    LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>] Workspace binding failed");
                    return false;
                }

                LOG_DEBUG("[CUDAFlashAttentionKernelT<FP32>] Using Flash Decoding: kv_len=" << kv_len
                                                                                            << " num_splits=" << num_splits);

                {
                    CUDA_KERNEL_PROFILE_SCOPE_STREAM(CUDAKernelType::FLASH_ATTN_DECODE, stream_);
                    result = cudaFlashAttn_decode_fp32(
                        Q, K, V, output,
                        static_cast<float *>(partial_output_buf_),
                        static_cast<float *>(partial_m_buf_),
                        static_cast<float *>(partial_l_buf_),
                        batch_size, kv_len,
                        n_heads, n_kv_heads, head_dim,
                        num_splits, device_params, stream_,
                        device_idx,
                        head_start, gqa_n_rep);
                }
            }
            else
            {
                // Flash Attention 2 (pipelined) for prefill or short KV
                // Uses pipelined cp.async + WMMA on Ampere SM >= 8.0
                LOG_DEBUG("[CUDAFlashAttentionKernelT<FP32>] Using Flash Attention 2 (pipelined WMMA): "
                          << "batch=" << batch_size << " seq_len=" << seq_len << " kv_len=" << kv_len);

                {
                    CUDA_KERNEL_PROFILE_SCOPE_STREAM(CUDAKernelType::FLASH_ATTN_PREFILL, stream_);
                    result = cudaFlashAttn_prefill_fa2(
                        Q, K, V, output,
                        batch_size, seq_len, kv_len,
                        n_heads, n_kv_heads, head_dim,
                        causal, window_size, position_offset,
                        device_params, mask,
                        stream_,
                        device_idx,
                        head_start, gqa_n_rep);
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
            const IMPIContext *mpi_ctx,
            int device_idx,
            int head_start,
            int local_n_heads,
            int local_n_kv_heads,
            int gqa_n_rep)
        {
            (void)workspace_scores;
            (void)mpi_ctx;
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

            const bool use_small_fp16kv_decode =
                K->native_type() == TensorType::FP16 &&
                V->native_type() == TensorType::FP16 &&
                batch_size == 1 &&
                causal &&
                seq_len > 1 &&
                seq_len <= MAX_SMALL_DECODE_ROWS &&
                kv_len > seq_len;
            const int expected_query_rows =
                use_small_fp16kv_decode ? seq_len : 1;

            // Wire device_params for graph-capture replay. The tiny H2D copy is
            // issued before beginCapture()/graph launch via
            // setDynamicAttnParams()/prepareDynamicAttnParams(), never from the
            // captured stage body. Capturing this memcpy leaves graph replay
            // dependent on host-side pinned storage lifetime and can make
            // cudaGraphLaunch fail with an illegal memory access.
            const attention::AttentionDeviceParams *d_attn_params = nullptr;
            if (stream_ && workspace_)
            {
                const int dynamic_position_offset =
                    (causal && kv_len > seq_len) ? (kv_len - seq_len) : 0;

                cudaStreamCaptureStatus cap_status = cudaStreamCaptureStatusNone;
                const cudaError_t cap_err =
                    cudaStreamIsCapturing(static_cast<cudaStream_t>(stream_), &cap_status);
                if (cap_err != cudaSuccess)
                {
                    LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>::compute_tensor] cudaStreamIsCapturing failed before attention-param use: "
                              << cudaGetErrorString(cap_err));
                    return false;
                }

                if (cap_status == cudaStreamCaptureStatusActive)
                {
                    if (!dynamic_attn_host_valid_ ||
                        !dynamic_attn_device_valid_ ||
                        dynamic_attn_param_rows_ < expected_query_rows ||
                        dynamic_attn_query_rows_ != expected_query_rows)
                    {
                        LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>::compute_tensor] "
                                  "Attention device params were not ready before CUDA graph capture"
                                  << " captured_body(kv_len=" << kv_len
                                  << ", pos=" << dynamic_position_offset
                                  << ", rows=" << expected_query_rows << ")"
                                  << " prepared(kv_len=" << dynamic_attn_kv_len_
                                  << ", pos=" << dynamic_attn_position_offset_
                                  << ", rows=" << dynamic_attn_query_rows_
                                  << ", param_rows=" << dynamic_attn_param_rows_
                                  << ") host_valid=" << dynamic_attn_host_valid_
                                  << " device_valid=" << dynamic_attn_device_valid_
                                  << " workspace=" << (workspace_ != nullptr)
                                  << " stream=" << stream_);
                        return false;
                    }
                }
                else
                {
                    setDynamicAttnParams(kv_len, dynamic_position_offset, expected_query_rows);
                    if (!dynamicAttnParamsReady(kv_len, dynamic_position_offset, expected_query_rows))
                    {
                        LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>::compute_tensor] Attention device params were not ready on the explicit stream");
                        return false;
                    }
                }

                if (!dynamic_attn_device_valid_)
                {
                    LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>::compute_tensor] Attention device params were not ready on the explicit stream");
                    return false;
                }

                void *d_buf = workspace_->getBuffer(AttentionWorkspaceBuffers::DEVICE_PARAMS);
                if (!d_buf)
                {
                    LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>::compute_tensor] Missing workspace buffer "
                              << AttentionWorkspaceBuffers::DEVICE_PARAMS);
                    return false;
                }
                d_attn_params = static_cast<const attention::AttentionDeviceParams *>(d_buf);
            }

            // === Mixed-precision KV handling (FP16→direct or Q8_1→fused/FP32) ===
            // For FP16 KV: both prefill (seq_len > 1) and decode (seq_len == 1)
            // use direct FP16 kernel variants, eliminating FP16→FP32 conversion.
            // For Q8_1 KV decode: fused inline dequant kernel (no workspace),
            // For Q8_1 KV prefill: dequant to FP32 workspace (no Q8_1 prefill kernel).
            bool use_fp16kv_direct = false;
            bool use_q8kv_direct = false;
            const void *K_fp16_ptr = nullptr;
            const void *V_fp16_ptr = nullptr;
            const void *K_q8_ptr = nullptr;
            const void *V_q8_ptr = nullptr;

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
                else if (K->native_type() == TensorType::Q8_1 && seq_len == 1)
                {
                    // FAST PATH: Q8_1 KV decode → fused inline dequant kernel
                    // Reads Q8_1 blocks directly in the attention inner loop,
                    // eliminating dequant kernel + FP32 workspace buffer.
                    use_q8kv_direct = true;
                    K_q8_ptr = K->gpu_data_ptr();
                    V_q8_ptr = V->gpu_data_ptr();
                    LOG_DEBUG("[CUDAFlashAttentionKernelT<FP32>::compute_tensor] Q8_1 KV fused decode path (no workspace)");
                }
                else
                {
                    // CONVERSION PATH: Q8_1 prefill → convert to FP32 workspace
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

                    if (!allocateWorkspace(n_heads, head_dim, num_splits))
                    {
                        LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>] Workspace binding failed for FP16KV decode");
                        return false;
                    }

                    int result;
                    {
                        CUDA_KERNEL_PROFILE_SCOPE_STREAM(CUDAKernelType::FLASH_ATTN_DECODE, stream_);
                        result = cudaFlashAttn_decode_fp16kv(
                            Q_ptr, K_fp16_ptr, V_fp16_ptr, output_ptr,
                            static_cast<float *>(partial_output_buf_),
                            static_cast<float *>(partial_m_buf_),
                            static_cast<float *>(partial_l_buf_),
                            batch_size, kv_len,
                            n_heads, n_kv_heads, head_dim,
                            num_splits, d_attn_params, stream_, dev,
                            head_start, gqa_n_rep);
                    }
                    if (result != 0)
                    {
                        LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>] Flash Decoding FP16KV failed");
                        return false;
                    }
                    return true;
                }

                if (use_small_fp16kv_decode)
                {
                    if (!stream_)
                    {
                        LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>] Small-M FP16KV decode requires an explicit non-null CUDA stream");
                        return false;
                    }

                    if (!workspace_)
                    {
                        LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>] Small-M FP16KV decode requires a bound workspace");
                        return false;
                    }

                    float *O_partial = static_cast<float *>(workspace_->getBuffer(AttentionWorkspaceBuffers::PARTIAL_OUTPUT));
                    float *m_partial = static_cast<float *>(workspace_->getBuffer(AttentionWorkspaceBuffers::PARTIAL_M));
                    float *l_partial = static_cast<float *>(workspace_->getBuffer(AttentionWorkspaceBuffers::PARTIAL_L));

                    if (!O_partial || !m_partial || !l_partial)
                    {
                        LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>] Workspace buffers missing for small-M FP16KV decode");
                        return false;
                    }

                    const size_t row_stride =
                        static_cast<size_t>(n_heads) * static_cast<size_t>(head_dim);
                    for (int row = 0; row < seq_len; ++row)
                    {
                        const int row_kv_len = std::max(1, kv_len - (seq_len - 1 - row));
                        const int num_splits = computeNumSplitsForDevice(row_kv_len, n_heads, dev);
                        const attention::AttentionDeviceParams *row_params =
                            d_attn_params ? (d_attn_params + row) : nullptr;

                        int result;
                        {
                            CUDA_KERNEL_PROFILE_SCOPE_STREAM(CUDAKernelType::FLASH_ATTN_DECODE, stream_);
                            result = cudaFlashAttn_decode_fp16kv(
                                Q_ptr + static_cast<size_t>(row) * row_stride,
                                K_fp16_ptr, V_fp16_ptr,
                                output_ptr + static_cast<size_t>(row) * row_stride,
                                O_partial, m_partial, l_partial,
                                1, row_kv_len,
                                n_heads, n_kv_heads, head_dim,
                                num_splits, row_params, stream_, dev,
                                head_start, gqa_n_rep);
                        }
                        if (result != 0)
                        {
                            LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>] Small-M FP16KV decode failed"
                                      << " row=" << row
                                      << " row_kv_len=" << row_kv_len
                                      << " num_splits=" << num_splits);
                            return false;
                        }
                    }
                    return true;
                }

                // PREFILL: FA2 with FP16 KV

                int result;
                {
                    CUDA_KERNEL_PROFILE_SCOPE_STREAM(CUDAKernelType::FLASH_ATTN_PREFILL, stream_);
                    result = cudaFlashAttn_prefill_fa2_fp16kv(
                        Q_ptr, K_fp16_ptr, V_fp16_ptr, output_ptr,
                        batch_size, seq_len, kv_len,
                        n_heads, n_kv_heads, head_dim,
                        causal, window_size, 0,
                        d_attn_params, mask_ptr,
                        stream_, dev,
                        head_start, gqa_n_rep);
                }
                if (result != 0)
                {
                    LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>] FA2 FP16KV kernel failed");
                    return false;
                }
                return true;
            }

            // Q8_1 KV fused decode: inline dequant in attention kernel, no workspace
            if (use_q8kv_direct)
            {
                if (cudaFlashAttn_setDevice(dev) != 0)
                {
                    LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>] Failed to set device " << dev);
                    return false;
                }

                // seq_len == 1 is guaranteed by the flag setup above
                int num_splits = computeNumSplitsForDevice(kv_len, n_heads, dev);

                if (!allocateWorkspace(n_heads, head_dim, num_splits))
                {
                    LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>] Workspace binding failed for Q8_1 fused decode");
                    return false;
                }

                int result;
                {
                    CUDA_KERNEL_PROFILE_SCOPE_STREAM(CUDAKernelType::FLASH_ATTN_DECODE, stream_);
                    result = cudaFlashAttn_decode_q8_1(
                        Q_ptr, K_q8_ptr, V_q8_ptr, output_ptr,
                        static_cast<float *>(partial_output_buf_),
                        static_cast<float *>(partial_m_buf_),
                        static_cast<float *>(partial_l_buf_),
                        batch_size, kv_len,
                        n_heads, n_kv_heads, head_dim,
                        num_splits, d_attn_params, stream_, dev,
                        head_start, gqa_n_rep);
                }
                if (result != 0)
                {
                    LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>] Flash Decoding Q8_1 fused failed");
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
                                   d_attn_params, mask_ptr,
                                   head_start, gqa_n_rep);
            }
            else
            {
                // Prefill path
                return apply_typed(Q_ptr, K_ptr, V_ptr, output_ptr,
                                   batch_size, seq_len, seq_len,
                                   n_heads, n_kv_heads, head_dim,
                                   causal, window_size, 0, dev,
                                   d_attn_params, mask_ptr,
                                   head_start, gqa_n_rep);
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
            const int n_heads = (n > 0) ? n : 128;  // Max expected heads
            const int head_dim = (k > 0) ? k : 128; // Max expected head dim
            const int num_splits = MAX_NUM_SPLITS;  // Conservative: allocate for max possible splits
            const int max_kv_len = 4096;            // decode workspace bound

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
                                    sizeof(attention::AttentionDeviceParams) *
                                        static_cast<size_t>(MAX_SMALL_DECODE_ROWS),
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
            freeWorkspace();
            workspace_ = workspace;
            if (workspace)
            {
                LOG_DEBUG("[CUDAFlashAttentionKernelT<FP32>] Bound workspace manager");
            }
            else
            {
                LOG_DEBUG("[CUDAFlashAttentionKernelT<FP32>] Unbound workspace manager");
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

        bool CUDAFlashAttentionKernelT<ActivationPrecision::FP32>::ensureHostAttnParamsCapacity(
            int capacity)
        {
            capacity = std::max(1, capacity);
            if (h_attn_params_ && h_attn_params_capacity_ >= capacity)
                return true;

            cudaStreamCaptureStatus cap_status = cudaStreamCaptureStatusNone;
            if (stream_)
                cudaStreamIsCapturing(static_cast<cudaStream_t>(stream_), &cap_status);
            if (cap_status == cudaStreamCaptureStatusActive)
            {
                LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>] "
                          "attention device params were not allocated before graph capture");
                return false;
            }

            if (h_attn_params_)
            {
                cudaFreeHost(h_attn_params_);
                h_attn_params_ = nullptr;
                h_attn_params_capacity_ = 0;
            }

            cudaError_t err = cudaMallocHost(reinterpret_cast<void **>(&h_attn_params_),
                                             sizeof(attention::AttentionDeviceParams) *
                                                 static_cast<size_t>(capacity));
            if (err != cudaSuccess)
            {
                LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>] cudaMallocHost failed for "
                          << capacity << " attention param row(s): "
                          << cudaGetErrorString(err));
                h_attn_params_ = nullptr;
                h_attn_params_capacity_ = 0;
                return false;
            }

            h_attn_params_capacity_ = capacity;
            dynamic_attn_device_valid_ = false;
            return true;
        }

        bool CUDAFlashAttentionKernelT<ActivationPrecision::FP32>::uploadDynamicAttnParams(
            void *stream)
        {
            if (!dynamic_attn_host_valid_ || !h_attn_params_)
            {
                LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>] Cannot upload attention params before host values are prepared");
                dynamic_attn_device_valid_ = false;
                return false;
            }
            if (!stream)
            {
                LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>] Cannot upload attention params on a null/default CUDA stream");
                dynamic_attn_device_valid_ = false;
                return false;
            }
            if (!workspace_)
            {
                LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>] Cannot upload attention params without a bound workspace");
                dynamic_attn_device_valid_ = false;
                return false;
            }

            cudaStreamCaptureStatus cap_status = cudaStreamCaptureStatusNone;
            const cudaError_t cap_err =
                cudaStreamIsCapturing(static_cast<cudaStream_t>(stream), &cap_status);
            if (cap_err != cudaSuccess)
            {
                LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>] cudaStreamIsCapturing failed before attention-param upload: "
                          << cudaGetErrorString(cap_err));
                dynamic_attn_device_valid_ = false;
                return false;
            }
            if (cap_status == cudaStreamCaptureStatusActive)
            {
                LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>] Refusing to record attention-param H2D inside CUDA graph capture");
                dynamic_attn_device_valid_ = false;
                return false;
            }

            void *d_buf = workspace_->getBuffer(AttentionWorkspaceBuffers::DEVICE_PARAMS);
            if (!d_buf)
            {
                LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>] Missing workspace buffer "
                          << AttentionWorkspaceBuffers::DEVICE_PARAMS
                          << " for attention params");
                dynamic_attn_device_valid_ = false;
                return false;
            }

            const cudaError_t copy_err =
                cudaMemcpyAsync(d_buf,
                                h_attn_params_,
                                sizeof(attention::AttentionDeviceParams) *
                                    static_cast<size_t>(dynamic_attn_param_rows_),
                                cudaMemcpyHostToDevice,
                                static_cast<cudaStream_t>(stream));
            if (copy_err != cudaSuccess)
            {
                LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>] cudaMemcpyAsync failed for attention params: "
                          << cudaGetErrorString(copy_err));
                dynamic_attn_device_valid_ = false;
                return false;
            }

            dynamic_attn_device_valid_ = true;
            return true;
        }

        bool CUDAFlashAttentionKernelT<ActivationPrecision::FP32>::dynamicAttnParamsReady(
            int kv_len, int position_offset, int query_rows) const
        {
            const int sanitized_query_rows = sanitizeSmallDecodeQueryRows(query_rows);
            const int param_rows = sanitized_query_rows;
            return dynamic_attn_host_valid_ &&
                   dynamic_attn_device_valid_ &&
                   dynamic_attn_kv_len_ == kv_len &&
                   dynamic_attn_position_offset_ == position_offset &&
                   dynamic_attn_query_rows_ == sanitized_query_rows &&
                   dynamic_attn_param_rows_ == param_rows;
        }

        void CUDAFlashAttentionKernelT<ActivationPrecision::FP32>::setDynamicAttnParams(
            int kv_len, int position_offset)
        {
            setDynamicAttnParams(kv_len, position_offset, 1);
        }

        void CUDAFlashAttentionKernelT<ActivationPrecision::FP32>::setDynamicAttnParams(
            int kv_len, int position_offset, int query_rows)
        {
            const int sanitized_query_rows = sanitizeSmallDecodeQueryRows(query_rows);
            const int param_rows = sanitized_query_rows;
            const bool same_params =
                dynamic_attn_host_valid_ &&
                dynamic_attn_kv_len_ == kv_len &&
                dynamic_attn_position_offset_ == position_offset &&
                dynamic_attn_query_rows_ == sanitized_query_rows &&
                dynamic_attn_param_rows_ == param_rows;

            if (!ensureHostAttnParamsCapacity(param_rows))
            {
                dynamic_attn_host_valid_ = false;
                dynamic_attn_device_valid_ = false;
                return;
            }

            if (!same_params)
                dynamic_attn_device_valid_ = false;

            for (int row = 0; row < param_rows; ++row)
            {
                const int row_kv_len = std::max(1, kv_len - (param_rows - 1 - row));
                h_attn_params_[row].kv_len = row_kv_len;
                h_attn_params_[row].position_offset = position_offset + row;
                h_attn_params_[row].mask_stride = kv_len;
            }

            dynamic_attn_kv_len_ = kv_len;
            dynamic_attn_position_offset_ = position_offset;
            dynamic_attn_query_rows_ = sanitized_query_rows;
            dynamic_attn_param_rows_ = param_rows;
            dynamic_attn_host_valid_ = true;

            if (!stream_ || !workspace_)
                return;

            cudaStreamCaptureStatus cap_status = cudaStreamCaptureStatusNone;
            const cudaError_t cap_err =
                cudaStreamIsCapturing(static_cast<cudaStream_t>(stream_), &cap_status);
            if (cap_err != cudaSuccess)
            {
                LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>] cudaStreamIsCapturing failed while preparing attention params: "
                          << cudaGetErrorString(cap_err));
                dynamic_attn_device_valid_ = false;
                return;
            }
            if (cap_status == cudaStreamCaptureStatusActive)
            {
                if (!dynamic_attn_device_valid_)
                {
                    LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>] Attention params changed during CUDA graph capture; "
                              "they must be uploaded before beginCapture()");
                }
                return;
            }

            if (!dynamic_attn_device_valid_)
                uploadDynamicAttnParams(stream_);
        }

        bool CUDAFlashAttentionKernelT<ActivationPrecision::FP32>::prepareDynamicAttnParams(
            int kv_len, int position_offset, int query_rows, void *stream)
        {
            if (!stream)
            {
                LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>] prepareDynamicAttnParams requires an explicit non-null CUDA stream");
                dynamic_attn_device_valid_ = false;
                return false;
            }
            setGPUStream(stream);
            setDynamicAttnParams(kv_len, position_offset, query_rows);
            const bool ready = dynamicAttnParamsReady(kv_len, position_offset, query_rows);
            if (!ready)
            {
                LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>] Dynamic attention params not ready after prepare"
                          << " requested(kv_len=" << kv_len
                          << ", pos=" << position_offset
                          << ", rows=" << sanitizeSmallDecodeQueryRows(query_rows) << ")"
                          << " prepared(kv_len=" << dynamic_attn_kv_len_
                          << ", pos=" << dynamic_attn_position_offset_
                          << ", rows=" << dynamic_attn_query_rows_
                          << ", param_rows=" << dynamic_attn_param_rows_
                          << ") host_valid=" << dynamic_attn_host_valid_
                          << " device_valid=" << dynamic_attn_device_valid_
                          << " workspace=" << (workspace_ != nullptr)
                          << " stream=" << stream_);
                return false;
            }
            return true;
        }

        // =====================================================================
        // Fused TQ KV Decode — rotation trick + centroid attention
        // =====================================================================

        bool CUDAFlashAttentionKernelT<ActivationPrecision::FP32>::compute_tensor_tq_decode(
            const ITensor *Q,
            ITensor *output,
            const void *K_cache,
            const void *V_cache,
            const float *rotation,
            const float *rotation_t,
            int batch_size,
            int kv_count,
            int n_heads,
            int n_kv_heads,
            int head_dim,
            int max_seq_len,
            int tail,
            int k_block_size,
            int v_block_size,
            int head_start,
            int gqa_n_rep)
        {
            const int num_splits = debugEnv().gemm.deterministic
                                       ? 1
                                       : std::max(1, std::min(kv_count / 64, 32));

            // Ensure workspace is allocated
            if (workspace_)
            {
                void *O_partial = workspace_->getBuffer(AttentionWorkspaceBuffers::PARTIAL_OUTPUT);
                void *m_partial = workspace_->getBuffer(AttentionWorkspaceBuffers::PARTIAL_M);
                void *l_partial = workspace_->getBuffer(AttentionWorkspaceBuffers::PARTIAL_L);

                if (!O_partial || !m_partial || !l_partial)
                {
                    LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>::compute_tensor_tq_decode] Workspace buffers not allocated");
                    return false;
                }

                const attention::AttentionDeviceParams *d_params = nullptr;
                if (h_attn_params_)
                {
                    void *d_buf = workspace_->getBuffer(AttentionWorkspaceBuffers::DEVICE_PARAMS);
                    if (d_buf)
                        d_params = static_cast<const attention::AttentionDeviceParams *>(d_buf);
                }

                int result = cudaFlashAttn_decode_tqkv(
                    static_cast<const float *>(Q->gpu_data_ptr()),
                    static_cast<float *>(output->gpu_data_ptr()),
                    static_cast<float *>(O_partial),
                    static_cast<float *>(m_partial),
                    static_cast<float *>(l_partial),
                    K_cache, V_cache,
                    rotation, rotation_t,
                    batch_size, kv_count,
                    n_heads, n_kv_heads, head_dim,
                    num_splits,
                    max_seq_len, tail,
                    k_block_size, v_block_size,
                    d_params,
                    stream_, device_idx_,
                    head_start, gqa_n_rep);

                if (result != 0)
                {
                    LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>::compute_tensor_tq_decode] Fused TQ kernel failed");
                    return false;
                }
                return true;
            }

            LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>::compute_tensor_tq_decode] "
                      "Fused TQ decode requires bound graph workspace");
            return false;
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
            partial_output_buf_ = nullptr;
            partial_m_buf_ = nullptr;
            partial_l_buf_ = nullptr;
            workspace_size_ = 0;
            max_splits_ = 0;
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
            const IMPIContext *mpi_ctx,
            int device_idx)
        {
            // TODO: Native FP16 implementation
            // For now, use FP32 path
            LOG_WARN("[CUDAFlashAttentionKernelT<FP16>] FP16 not yet implemented, using FP32");
            CUDAFlashAttentionKernelT<ActivationPrecision::FP32> fp32_kernel(device_idx_);
            fp32_kernel.setGPUStream(stream_);
            fp32_kernel.bindWorkspace(workspace_);
            fp32_kernel.setDeviceContext(device_ctx_);
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
            const IMPIContext *mpi_ctx,
            int device_idx)
        {
            LOG_WARN("[CUDAFlashAttentionKernelT<FP16>] FP16 not yet implemented, using FP32");
            CUDAFlashAttentionKernelT<ActivationPrecision::FP32> fp32_kernel(device_idx_);
            fp32_kernel.setGPUStream(stream_);
            fp32_kernel.bindWorkspace(workspace_);
            fp32_kernel.setDeviceContext(device_ctx_);
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
            fp32_kernel.setGPUStream(stream_);
            fp32_kernel.bindWorkspace(workspace_);
            fp32_kernel.setDeviceContext(device_ctx_);
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
            const IMPIContext *mpi_ctx,
            int device_idx,
            int head_start,
            int local_n_heads,
            int local_n_kv_heads,
            int gqa_n_rep)
        {
            // FP16 compute_tensor: delegate to FP32 for now
            LOG_WARN("[CUDAFlashAttentionKernelT<FP16>] FP16 compute_tensor not yet implemented, using FP32");
            CUDAFlashAttentionKernelT<ActivationPrecision::FP32> fp32_kernel(device_idx_);
            fp32_kernel.setGPUStream(stream_);
            fp32_kernel.bindWorkspace(workspace_);
            fp32_kernel.setDeviceContext(device_ctx_);
            return fp32_kernel.compute_tensor(Q, K, V, output, batch_size, seq_len, kv_len,
                                              n_heads, n_kv_heads, head_dim, causal, window_size,
                                              workspace_scores, workspace_mask, mpi_ctx, device_idx,
                                              head_start, local_n_heads, local_n_kv_heads, gqa_n_rep);
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
            freeWorkspace();
            workspace_ = workspace;
            if (workspace)
            {
                LOG_DEBUG("[CUDAFlashAttentionKernelT<FP16>] Bound workspace manager");
            }
            else
            {
                LOG_DEBUG("[CUDAFlashAttentionKernelT<FP16>] Unbound workspace manager");
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
            partial_output_buf_ = nullptr;
            partial_m_buf_ = nullptr;
            partial_l_buf_ = nullptr;
            workspace_size_ = 0;
            max_splits_ = 0;
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
            const IMPIContext *mpi_ctx,
            int device_idx)
        {
            LOG_WARN("[CUDAFlashAttentionKernelT<BF16>] BF16 not yet implemented, using FP32");
            CUDAFlashAttentionKernelT<ActivationPrecision::FP32> fp32_kernel(device_idx_);
            fp32_kernel.setGPUStream(stream_);
            fp32_kernel.bindWorkspace(workspace_);
            fp32_kernel.setDeviceContext(device_ctx_);
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
            const IMPIContext *mpi_ctx,
            int device_idx)
        {
            LOG_WARN("[CUDAFlashAttentionKernelT<BF16>] BF16 not yet implemented, using FP32");
            CUDAFlashAttentionKernelT<ActivationPrecision::FP32> fp32_kernel(device_idx_);
            fp32_kernel.setGPUStream(stream_);
            fp32_kernel.bindWorkspace(workspace_);
            fp32_kernel.setDeviceContext(device_ctx_);
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
            fp32_kernel.setGPUStream(stream_);
            fp32_kernel.bindWorkspace(workspace_);
            fp32_kernel.setDeviceContext(device_ctx_);
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
            const IMPIContext *mpi_ctx,
            int device_idx,
            int head_start,
            int local_n_heads,
            int local_n_kv_heads,
            int gqa_n_rep)
        {
            // BF16 compute_tensor: delegate to FP32 for now
            LOG_WARN("[CUDAFlashAttentionKernelT<BF16>] BF16 compute_tensor not yet implemented, using FP32");
            CUDAFlashAttentionKernelT<ActivationPrecision::FP32> fp32_kernel(device_idx_);
            fp32_kernel.setGPUStream(stream_);
            fp32_kernel.bindWorkspace(workspace_);
            fp32_kernel.setDeviceContext(device_ctx_);
            return fp32_kernel.compute_tensor(Q, K, V, output, batch_size, seq_len, kv_len,
                                              n_heads, n_kv_heads, head_dim, causal, window_size,
                                              workspace_scores, workspace_mask, mpi_ctx, device_idx,
                                              head_start, local_n_heads, local_n_kv_heads, gqa_n_rep);
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
            freeWorkspace();
            workspace_ = workspace;
            if (workspace)
            {
                LOG_DEBUG("[CUDAFlashAttentionKernelT<BF16>] Bound workspace manager");
            }
            else
            {
                LOG_DEBUG("[CUDAFlashAttentionKernelT<BF16>] Unbound workspace manager");
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
