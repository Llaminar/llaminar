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
#include "../../../backends/IWorkerGPUContext.h"
#include "../../../execution/local_execution/device/DeviceWorkspaceManager.h"
#include "../../../execution/local_execution/device/WorkspaceDescriptor.h"
#include "../ROCmKernelBase.h"
#include "../../../utils/Logger.h"
#include "../../../utils/ROCmKernelProfiler.h"
#include "../../../utils/DebugEnv.h"
#include "../../attention/AttentionDeviceParams.h"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstring>
#include <stdexcept>

// Extern "C" declarations for HIP kernel wrappers
extern "C"
{
    // Flash Attention 2 prefill kernel for MI50
    int hipFlashAttn_prefill_fa2(
        const float *Q, const float *K, const float *V, float *O,
        int batch_size, int seq_len, int kv_len,
        int n_heads, int n_kv_heads, int head_dim,
        bool causal, int window_size, int position_offset,
        const llaminar2::attention::AttentionDeviceParams *device_params,
        const float *mask,
        void *stream,
        int head_start, int gqa_n_rep);

    // Flash Attention 2 prefill with native FP16 KV cache (avoids FP32 conversion)
    int hipFlashAttn_prefill_fa2_fp16(
        const float *Q, const void *K, const void *V, float *O,
        int batch_size, int seq_len, int kv_len,
        int n_heads, int n_kv_heads, int head_dim,
        bool causal, int window_size, int position_offset,
        const llaminar2::attention::AttentionDeviceParams *device_params,
        const float *mask,
        void *stream,
        int head_start, int gqa_n_rep);

    // Flash Attention 2 prefill with native Q8_1 KV cache (inline dequant)
    int hipFlashAttn_prefill_fa2_q8_1(
        const float *Q, const void *K, const void *V, float *O,
        int batch_size, int seq_len, int kv_len,
        int n_heads, int n_kv_heads, int head_dim,
        bool causal, int window_size, int position_offset,
        const llaminar2::attention::AttentionDeviceParams *device_params,
        const float *mask,
        void *stream,
        int head_start, int gqa_n_rep);

    // Flash Decoding for single-token decode with split-K parallelism
    int hipFlashAttn_decode_fp32(
        const float *Q, const float *K_cache, const float *V_cache, float *O,
        float *O_partial, float *m_partial, float *l_partial,
        int batch_size, int kv_len,
        int n_heads, int n_kv_heads, int head_dim,
        int num_splits,
        const llaminar2::attention::AttentionDeviceParams *device_params,
        void *stream,
        int head_start, int gqa_n_rep);

    // Flash Decoding with native FP16 KV cache (avoids FP32 conversion)
    int hipFlashAttn_decode_fp16(
        const float *Q, const void *K_cache, const void *V_cache, float *O,
        float *O_partial, float *m_partial, float *l_partial,
        int batch_size, int kv_len,
        int n_heads, int n_kv_heads, int head_dim,
        int num_splits,
        const llaminar2::attention::AttentionDeviceParams *device_params,
        void *stream,
        int head_start, int gqa_n_rep);

    // Flash Decoding with native Q8_1 KV cache (inline dequant)
    int hipFlashAttn_decode_q8_1(
        const float *Q, const void *K_cache, const void *V_cache, float *O,
        float *O_partial, float *m_partial, float *l_partial,
        int batch_size, int kv_len,
        int n_heads, int n_kv_heads, int head_dim,
        int num_splits,
        const llaminar2::attention::AttentionDeviceParams *device_params,
        void *stream,
        int head_start, int gqa_n_rep);

    int hipFlashAttn_allocWorkspace(
        void **partial_output, void **partial_m, void **partial_l,
        int batch_size, int n_heads, int head_dim, int num_splits);

    void hipFlashAttn_freeWorkspace(void *partial_output, void *partial_m, void *partial_l);

    int hipFlashAttn_setDevice(int device_idx);
    // hipFlashAttn_synchronize() removed - caller manages coherence via events

    // GPU-side tensor type conversion (avoids catastrophic CPU D2H → convert → H2D round-trip)
    bool hip_convert_tensor_to_fp32(
        const void *d_src,
        llaminar2::TensorType src_type,
        float *d_dst,
        int count,
        int rows,
        int cols,
        hipStream_t stream);
}

namespace llaminar2
{
    namespace rocm
    {
        // Default number of splits for Flash Decoding
        constexpr int DEFAULT_NUM_SPLITS = 8;
        constexpr int MAX_SMALL_DECODE_ROWS = 2;

        static int selectFlashDecodeNumSplits(int kv_len)
        {
            if (debugEnv().gemm.deterministic)
                return 1;

            if (kv_len <= 64)
                return 1;
            if (kv_len < 128)
                return 2;
            if (kv_len < 256)
                return 4;
            return DEFAULT_NUM_SPLITS;
        }

        // =====================================================================
        // FP32 Specialization Implementation
        // =====================================================================

        ROCmFlashAttentionKernelT<ActivationPrecision::FP32>::ROCmFlashAttentionKernelT(int device_idx)
            : device_idx_(device_idx), stream_(nullptr),
              partial_output_buf_(nullptr), partial_m_buf_(nullptr), partial_l_buf_(nullptr),
              workspace_size_(0), max_splits_(0), workspace_(nullptr), device_ctx_(nullptr),
              h_attn_params_(nullptr), h_attn_params_capacity_(0), small_decode_rows_(0)
        {
            if (device_idx < 0)
            {
                throw std::runtime_error(
                    "[ROCmFlashAttentionKernelT<FP32>] Invalid device_idx=" + std::to_string(device_idx) +
                    " — caller must pass explicit device ordinal");
            }
            LOG_DEBUG("[ROCmFlashAttentionKernelT<FP32>] Created for device " << device_idx);
        }

        ROCmFlashAttentionKernelT<ActivationPrecision::FP32>::ROCmFlashAttentionKernelT(
            IWorkerGPUContext *ctx)
            : stream_(nullptr),
              partial_output_buf_(nullptr), partial_m_buf_(nullptr), partial_l_buf_(nullptr),
              workspace_size_(0), max_splits_(0), workspace_(nullptr), device_ctx_(nullptr),
              h_attn_params_(nullptr), h_attn_params_capacity_(0), small_decode_rows_(0)
        {
            if (!ctx)
            {
                throw std::runtime_error(
                    "[ROCmFlashAttentionKernelT<FP32>] Device context is null");
            }

            if (!ctx->isInitialized())
            {
                throw std::runtime_error(
                    "[ROCmFlashAttentionKernelT<FP32>] Device context is not initialized");
            }

            setDeviceContext(ctx);
            device_idx_ = ctx->deviceOrdinal();

            // Get stream from context
            stream_ = ctx->defaultStream();

            LOG_DEBUG("[ROCmFlashAttentionKernelT<FP32>] Created for device " << device_idx_
                                                                              << " using device context");
        }

        ROCmFlashAttentionKernelT<ActivationPrecision::FP32>::~ROCmFlashAttentionKernelT()
        {
            freeWorkspace();
            if (h_attn_params_)
            {
                hipHostFree(h_attn_params_);
                h_attn_params_ = nullptr;
                h_attn_params_capacity_ = 0;
            }
        }

        ROCmFlashAttentionKernelT<ActivationPrecision::FP32>::ROCmFlashAttentionKernelT(
            ROCmFlashAttentionKernelT &&other) noexcept
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
              small_decode_rows_(other.small_decode_rows_)
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
            other.small_decode_rows_ = 0;
        }

        ROCmFlashAttentionKernelT<ActivationPrecision::FP32> &
        ROCmFlashAttentionKernelT<ActivationPrecision::FP32>::operator=(
            ROCmFlashAttentionKernelT &&other) noexcept
        {
            if (this != &other)
            {
                freeWorkspace();
                if (h_attn_params_)
                {
                    hipHostFree(h_attn_params_);
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
                small_decode_rows_ = other.small_decode_rows_;

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
                other.small_decode_rows_ = 0;
            }
            return *this;
        }

        bool ROCmFlashAttentionKernelT<ActivationPrecision::FP32>::ensureHostAttnParamsCapacity(
            int capacity)
        {
            capacity = std::max(1, capacity);
            if (h_attn_params_ && h_attn_params_capacity_ >= capacity)
                return true;

            hipStreamCaptureStatus cap_status = hipStreamCaptureStatusNone;
            if (stream_)
                hipStreamIsCapturing(static_cast<hipStream_t>(stream_), &cap_status);
            if (cap_status == hipStreamCaptureStatusActive)
            {
                LOG_ERROR("[ROCmFlashAttentionKernelT<FP32>] "
                          "attention device params were not allocated before graph capture");
                return false;
            }

            if (h_attn_params_)
            {
                hipHostFree(h_attn_params_);
                h_attn_params_ = nullptr;
                h_attn_params_capacity_ = 0;
            }

            hipError_t err = hipHostMalloc(&h_attn_params_,
                                           sizeof(attention::AttentionDeviceParams) *
                                               static_cast<size_t>(capacity),
                                           hipHostMallocDefault);
            if (err != hipSuccess)
            {
                LOG_ERROR("[ROCmFlashAttentionKernelT<FP32>] hipHostMalloc failed for "
                          << capacity << " attention param row(s): "
                          << hipGetErrorString(err));
                h_attn_params_ = nullptr;
                h_attn_params_capacity_ = 0;
                return false;
            }

            h_attn_params_capacity_ = capacity;
            dynamic_attn_device_valid_ = false;
            return true;
        }

        bool ROCmFlashAttentionKernelT<ActivationPrecision::FP32>::uploadDynamicAttnParams(
            void *stream)
        {
            if (!dynamic_attn_host_valid_ || !h_attn_params_)
            {
                LOG_ERROR("[ROCmFlashAttentionKernelT<FP32>] Cannot upload attention params before host values are prepared");
                dynamic_attn_device_valid_ = false;
                return false;
            }
            if (!stream)
            {
                LOG_ERROR("[ROCmFlashAttentionKernelT<FP32>] Cannot upload attention params on a null/default HIP stream");
                dynamic_attn_device_valid_ = false;
                return false;
            }
            if (!workspace_)
            {
                LOG_ERROR("[ROCmFlashAttentionKernelT<FP32>] Cannot upload attention params without a bound workspace");
                dynamic_attn_device_valid_ = false;
                return false;
            }

            hipStreamCaptureStatus cap_status = hipStreamCaptureStatusNone;
            const hipError_t cap_err =
                hipStreamIsCapturing(static_cast<hipStream_t>(stream), &cap_status);
            if (cap_err != hipSuccess)
            {
                LOG_ERROR("[ROCmFlashAttentionKernelT<FP32>] hipStreamIsCapturing failed before attention-param upload: "
                          << hipGetErrorString(cap_err));
                dynamic_attn_device_valid_ = false;
                return false;
            }
            if (cap_status == hipStreamCaptureStatusActive)
            {
                LOG_ERROR("[ROCmFlashAttentionKernelT<FP32>] Refusing to record attention-param H2D inside HIP graph capture");
                dynamic_attn_device_valid_ = false;
                return false;
            }

            void *d_buf = workspace_->getBuffer(AttentionWorkspaceBuffers::DEVICE_PARAMS);
            if (!d_buf)
            {
                LOG_ERROR("[ROCmFlashAttentionKernelT<FP32>] Missing workspace buffer "
                          << AttentionWorkspaceBuffers::DEVICE_PARAMS
                          << " for attention params");
                dynamic_attn_device_valid_ = false;
                return false;
            }

            const hipError_t copy_err =
                hipMemcpyAsync(d_buf,
                               h_attn_params_,
                               sizeof(attention::AttentionDeviceParams) *
                                   static_cast<size_t>(dynamic_attn_param_rows_),
                               hipMemcpyHostToDevice,
                               static_cast<hipStream_t>(stream));
            if (copy_err != hipSuccess)
            {
                LOG_ERROR("[ROCmFlashAttentionKernelT<FP32>] hipMemcpyAsync failed for attention params: "
                          << hipGetErrorString(copy_err));
                dynamic_attn_device_valid_ = false;
                return false;
            }

            dynamic_attn_device_valid_ = true;
            return true;
        }

        bool ROCmFlashAttentionKernelT<ActivationPrecision::FP32>::dynamicAttnParamsReady(
            int kv_len, int position_offset, int query_rows) const
        {
            const int sanitized_query_rows =
                (query_rows > 1 && query_rows <= MAX_SMALL_DECODE_ROWS) ? query_rows : 1;
            const int param_rows = std::max(1, sanitized_query_rows);
            return dynamic_attn_host_valid_ &&
                   dynamic_attn_device_valid_ &&
                   dynamic_attn_kv_len_ == kv_len &&
                   dynamic_attn_position_offset_ == position_offset &&
                   dynamic_attn_query_rows_ == sanitized_query_rows &&
                   dynamic_attn_param_rows_ == param_rows;
        }

        void ROCmFlashAttentionKernelT<ActivationPrecision::FP32>::allocateWorkspace(
            int n_heads, int head_dim, int num_splits)
        {
            // Workspace is now REQUIRED - no legacy allocation path
            if (!validateROCmWorkspaceBinding(workspace_, device_idx_, "ROCmFlashAttentionKernelT<FP32>"))
            {
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

            const size_t required_partial_output =
                static_cast<size_t>(std::max(1, n_heads)) *
                static_cast<size_t>(std::max(1, num_splits)) *
                static_cast<size_t>(std::max(1, head_dim)) * sizeof(float);
            const size_t required_partial_meta =
                static_cast<size_t>(std::max(1, n_heads)) *
                static_cast<size_t>(std::max(1, num_splits)) * sizeof(float);
            if (workspace_->getBufferSize(AttentionWorkspaceBuffers::PARTIAL_OUTPUT) < required_partial_output ||
                workspace_->getBufferSize(AttentionWorkspaceBuffers::PARTIAL_M) < required_partial_meta ||
                workspace_->getBufferSize(AttentionWorkspaceBuffers::PARTIAL_L) < required_partial_meta)
            {
                LOG_ERROR("[ROCmFlashAttentionKernelT<FP32>] Bound attention workspace is too small: "
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
                return;
            }

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

        bool ROCmFlashAttentionKernelT<ActivationPrecision::FP32>::compute_batch(
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

        bool ROCmFlashAttentionKernelT<ActivationPrecision::FP32>::compute_decode(
            const float *Q, const float *K, const float *V, float *output,
            int seq_len, int kv_len, int n_heads, int n_kv_heads, int head_dim,
            bool causal, int position_offset, int device_idx)
        {
            int dev = (device_idx >= 0) ? device_idx : device_idx_;
            return apply_typed(Q, K, V, output,
                               1, seq_len, kv_len,
                               n_heads, n_kv_heads, head_dim,
                               causal, -1, position_offset, dev, nullptr, nullptr);
        }

        bool ROCmFlashAttentionKernelT<ActivationPrecision::FP32>::apply_typed(
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
            if (head_dim > 256)
            {
                LOG_ERROR("[ROCmFlashAttentionKernelT<FP32>] head_dim=" << head_dim
                                                                        << " exceeds MI50 kernel limit (256)");
                return false;
            }

            // Set device (skip during stream capture — device was set before capture began)
            {
                hipStreamCaptureStatus cap_status = hipStreamCaptureStatusNone;
                if (stream_)
                    hipStreamIsCapturing(static_cast<hipStream_t>(stream_), &cap_status);
                if (cap_status != hipStreamCaptureStatusActive)
                {
                    if (hipFlashAttn_setDevice(device_idx) != 0)
                    {
                        LOG_ERROR("[ROCmFlashAttentionKernelT<FP32>] Failed to set device " << device_idx);
                        return false;
                    }
                }
            }

            int result;

            // Choose algorithm based on seq_len. Split-K flash decode remains
            // the default decode path; LLAMINAR_ROCM_FA_DECODE_VIA_PREFILL is
            // an explicit diagnostic/reference mode only.
            const bool decode_via_prefill = debugEnv().rocm.fa_decode_via_prefill;
            if (seq_len == 1 && !decode_via_prefill)
            {
                // Flash Decoding for single-token decode
                const int num_splits = selectFlashDecodeNumSplits(kv_len);

                allocateWorkspace(n_heads, head_dim, num_splits);

                if (!partial_output_buf_ || !partial_m_buf_ || !partial_l_buf_)
                {
                    LOG_ERROR("[ROCmFlashAttentionKernelT<FP32>] Workspace allocation failed");
                    return false;
                }

                LOG_DEBUG("[ROCmFlashAttentionKernelT<FP32>] Using Flash Decoding: kv_len=" << kv_len
                                                                                            << " num_splits=" << num_splits);

                {
                    ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::FLASH_ATTN_DECODE, static_cast<hipStream_t>(stream_));
                    result = hipFlashAttn_decode_fp32(
                        Q, K, V, output,
                        static_cast<float *>(partial_output_buf_),
                        static_cast<float *>(partial_m_buf_),
                        static_cast<float *>(partial_l_buf_),
                        batch_size, kv_len,
                        n_heads, n_kv_heads, head_dim,
                        num_splits, device_params, stream_,
                        head_start, gqa_n_rep);
                }
            }
            else
            {
                // Flash Attention 2 for prefill. When explicitly requested via
                // LLAMINAR_ROCM_FA_DECODE_VIA_PREFILL=1, this branch is also a
                // diagnostic/reference path for single-token decode.
                LOG_DEBUG("[ROCmFlashAttentionKernelT<FP32>] Using Flash Attention 2 (MI50): "
                          << "batch=" << batch_size << " seq_len=" << seq_len << " kv_len=" << kv_len);

                {
                    ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::FLASH_ATTN_PREFILL, static_cast<hipStream_t>(stream_));
                    result = hipFlashAttn_prefill_fa2(
                        Q, K, V, output,
                        batch_size, seq_len, kv_len,
                        n_heads, n_kv_heads, head_dim,
                        causal, window_size, position_offset,
                        device_params, mask,
                        stream_,
                        head_start, gqa_n_rep);
                }
            }

            if (result != 0)
            {
                LOG_ERROR("[ROCmFlashAttentionKernelT<FP32>] Kernel execution failed with code " << result);
                return false;
            }

            // Removed hipFlashAttn_synchronize() - caller manages coherence via events
            return true;
        }

        void ROCmFlashAttentionKernelT<ActivationPrecision::FP32>::setDynamicAttnParams(
            int kv_len, int position_offset)
        {
            setDynamicAttnParams(kv_len, position_offset, 1);
        }

        void ROCmFlashAttentionKernelT<ActivationPrecision::FP32>::setDynamicAttnParams(
            int kv_len, int position_offset, int query_rows)
        {
            const int sanitized_query_rows =
                (query_rows > 1 && query_rows <= MAX_SMALL_DECODE_ROWS) ? query_rows : 1;
            small_decode_rows_ = (sanitized_query_rows > 1) ? sanitized_query_rows : 0;
            const int param_rows = std::max(1, sanitized_query_rows);
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
            {
                dynamic_attn_device_valid_ = false;
            }

            if (small_decode_rows_ > 1)
            {
                for (int row = 0; row < small_decode_rows_; ++row)
                {
                    const int row_kv_len = std::max(1, kv_len - (small_decode_rows_ - 1 - row));
                    h_attn_params_[row].kv_len = row_kv_len;
                    h_attn_params_[row].position_offset = std::max(0, row_kv_len - 1);
                    h_attn_params_[row].mask_stride = row_kv_len;
                }
            }
            else
            {
                h_attn_params_->kv_len = kv_len;
                h_attn_params_->position_offset = position_offset;
                h_attn_params_->mask_stride = kv_len;
            }

            dynamic_attn_kv_len_ = kv_len;
            dynamic_attn_position_offset_ = position_offset;
            dynamic_attn_query_rows_ = sanitized_query_rows;
            dynamic_attn_param_rows_ = param_rows;
            dynamic_attn_host_valid_ = true;

            if (!stream_ || !workspace_)
            {
                return;
            }

            hipStreamCaptureStatus cap_status = hipStreamCaptureStatusNone;
            const hipError_t cap_err =
                hipStreamIsCapturing(static_cast<hipStream_t>(stream_), &cap_status);
            if (cap_err != hipSuccess)
            {
                LOG_ERROR("[ROCmFlashAttentionKernelT<FP32>] hipStreamIsCapturing failed while preparing attention params: "
                          << hipGetErrorString(cap_err));
                dynamic_attn_device_valid_ = false;
                return;
            }
            if (cap_status == hipStreamCaptureStatusActive)
            {
                if (!dynamic_attn_device_valid_)
                {
                    LOG_ERROR("[ROCmFlashAttentionKernelT<FP32>] Attention params changed during HIP graph capture; "
                              "they must be uploaded before beginCapture()");
                }
                return;
            }

            if (!dynamic_attn_device_valid_)
            {
                uploadDynamicAttnParams(stream_);
            }
        }

        bool ROCmFlashAttentionKernelT<ActivationPrecision::FP32>::prepareDynamicAttnParams(
            int kv_len, int position_offset, int query_rows, void *stream)
        {
            if (!stream)
            {
                LOG_ERROR("[ROCmFlashAttentionKernelT<FP32>] prepareDynamicAttnParams requires an explicit non-null HIP stream");
                dynamic_attn_device_valid_ = false;
                return false;
            }
            setGPUStream(stream);
            setDynamicAttnParams(kv_len, position_offset, query_rows);
            const bool ready = dynamicAttnParamsReady(kv_len, position_offset, query_rows);
            if (!ready)
            {
                const int sanitized_query_rows =
                    (query_rows > 1 && query_rows <= MAX_SMALL_DECODE_ROWS) ? query_rows : 1;
                LOG_ERROR("[ROCmFlashAttentionKernelT<FP32>] Dynamic attention params not ready after prepare"
                          << " requested(kv_len=" << kv_len
                          << ", pos=" << position_offset
                          << ", rows=" << sanitized_query_rows << ")"
                          << " actual(kv_len=" << dynamic_attn_kv_len_
                          << ", pos=" << dynamic_attn_position_offset_
                          << ", rows=" << dynamic_attn_query_rows_
                          << ", param_rows=" << dynamic_attn_param_rows_
                          << ") host_valid=" << dynamic_attn_host_valid_
                          << " device_valid=" << dynamic_attn_device_valid_
                          << " workspace=" << (workspace_ != nullptr)
                          << " stream=" << stream_);
            }
            return ready;
        }

        void ROCmFlashAttentionKernelT<ActivationPrecision::FP32>::resetDynamicState()
        {
            small_decode_rows_ = 0;
            dynamic_attn_kv_len_ = 0;
            dynamic_attn_position_offset_ = 0;
            dynamic_attn_query_rows_ = 1;
            dynamic_attn_param_rows_ = 1;
            dynamic_attn_host_valid_ = false;
            dynamic_attn_device_valid_ = false;
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
                LOG_ERROR("[ROCmFlashAttentionKernelT<FP32>::compute_tensor] Null tensor");
                return false;
            }

            // Q and output must be FP32 for this specialization
            if (Q->native_type() != TensorType::FP32 || output->native_type() != TensorType::FP32)
            {
                LOG_ERROR("[ROCmFlashAttentionKernelT<FP32>::compute_tensor] Expected FP32 Q/output tensors, got Q="
                          << Q->dtype_name() << " K=" << K->dtype_name()
                          << " V=" << V->dtype_name() << " O=" << output->dtype_name());
                return false;
            }

            // Extract GPU pointers - tensors should already be coherent on device
            const float *Q_ptr = static_cast<const float *>(Q->gpu_data_ptr());
            const float *K_ptr = static_cast<const float *>(K->gpu_data_ptr());
            const float *V_ptr = static_cast<const float *>(V->gpu_data_ptr());
            float *O_ptr = static_cast<float *>(output->gpu_data_ptr());

            // Track whether native KV dispatch is possible (FP16/Q8_1 → no conversion).
            // LLAMINAR_ROCM_FA_DISABLE_NATIVE_KV is a diagnostic escape hatch for
            // long-context decode parity triage: it forces the existing FP32
            // conversion path so native FP16/Q8_1 flash decode can be isolated.
            bool use_native_kv = false;
            TensorType kv_native_type = TensorType::FP32;
            const auto &rocm_env = debugEnv().rocm;
            const bool disable_native_kv = rocm_env.fa_disable_native_kv;
            const bool decode_via_prefill = rocm_env.fa_decode_via_prefill;

            if (K->native_type() != TensorType::FP32 || V->native_type() != TensorType::FP32)
            {
                if (K->native_type() != V->native_type())
                {
                    LOG_ERROR("[ROCmFlashAttentionKernelT<FP32>::compute_tensor] Mixed K/V types not supported: K="
                              << K->dtype_name() << " V=" << V->dtype_name());
                    return false;
                }

                kv_native_type = K->native_type();

                // For prefill (head_dim >= 64) or decode, dispatch to
                // native FP16/Q8_1 kernel — eliminates FP32 conversion pipeline entirely
                if (!disable_native_kv &&
                    head_dim >= 64 &&
                    (kv_native_type == TensorType::FP16 ||
                     (kv_native_type == TensorType::Q8_1 && head_dim % 32 == 0)))
                {
                    use_native_kv = true;
                    LOG_DEBUG("[ROCmFlashAttentionKernelT<FP32>::compute_tensor] Native "
                              << K->dtype_name() << " KV path — skipping FP32 conversion");
                }
                else
                {
                    // Fallback: FP32 workspace conversion (decode path, or head_dim < 64)
                    if (!workspace_)
                    {
                        LOG_ERROR("[ROCmFlashAttentionKernelT<FP32>::compute_tensor] Mixed-precision KV requires workspace-bound conversion buffers");
                        return false;
                    }

                    const size_t rows = static_cast<size_t>(batch_size) * static_cast<size_t>(kv_len);
                    const size_t logical_cols = static_cast<size_t>(n_kv_heads) * static_cast<size_t>(head_dim);
                    const size_t logical_elements = rows * logical_cols;

                    float *d_k_tmp = static_cast<float *>(workspace_->getBuffer(AttentionWorkspaceBuffers::K_TMP_FP32));
                    float *d_v_tmp = static_cast<float *>(workspace_->getBuffer(AttentionWorkspaceBuffers::V_TMP_FP32));
                    if (!d_k_tmp || !d_v_tmp)
                    {
                        LOG_ERROR("[ROCmFlashAttentionKernelT<FP32>::compute_tensor] Missing workspace conversion buffers: "
                                  << AttentionWorkspaceBuffers::K_TMP_FP32 << " / "
                                  << AttentionWorkspaceBuffers::V_TMP_FP32);
                        return false;
                    }

                    const auto hip_stream = static_cast<hipStream_t>(stream_);
                    const bool k_ok = hip_convert_tensor_to_fp32(
                        K->gpu_data_ptr(), K->native_type(), d_k_tmp,
                        static_cast<int>(logical_elements),
                        static_cast<int>(rows), static_cast<int>(logical_cols),
                        hip_stream);
                    const bool v_ok = hip_convert_tensor_to_fp32(
                        V->gpu_data_ptr(), V->native_type(), d_v_tmp,
                        static_cast<int>(logical_elements),
                        static_cast<int>(rows), static_cast<int>(logical_cols),
                        hip_stream);
                    if (!k_ok || !v_ok)
                    {
                        LOG_ERROR("[ROCmFlashAttentionKernelT<FP32>::compute_tensor] GPU-side KV conversion failed for type "
                                  << K->dtype_name());
                        return false;
                    }

                    K_ptr = d_k_tmp;
                    V_ptr = d_v_tmp;
                }
            }

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

            // Wire device_params for graph-capture replay. The tiny H2D copy is
            // deliberately issued before beginCapture()/graph launch through
            // setDynamicAttnParams()/prepareDynamicAttnParams(), never from the
            // captured stage body. Capturing this memcpy made ROCm attention
            // graph replay crash intermittently inside hipGraphLaunch.
            const attention::AttentionDeviceParams *d_attn_params = nullptr;
            if (stream_ && workspace_)
            {
                const bool small_native_decode =
                    use_native_kv &&
                    batch_size == 1 &&
                    causal &&
                    !decode_via_prefill &&
                    seq_len > 1 &&
                    seq_len <= MAX_SMALL_DECODE_ROWS &&
                    kv_len > seq_len;
                const int query_rows_for_params = small_native_decode ? seq_len : 1;
                const int dynamic_position_offset =
                    (causal && kv_len > seq_len) ? (kv_len - seq_len) : 0;

                hipStreamCaptureStatus cap_status = hipStreamCaptureStatusNone;
                const hipError_t cap_err =
                    hipStreamIsCapturing(static_cast<hipStream_t>(stream_), &cap_status);
                if (cap_err != hipSuccess)
                {
                    LOG_ERROR("[ROCmFlashAttentionKernelT<FP32>::compute_tensor] hipStreamIsCapturing failed before attention-param use: "
                              << hipGetErrorString(cap_err));
                    return false;
                }

                if (cap_status == hipStreamCaptureStatusActive)
                {
                    if (!dynamic_attn_host_valid_ ||
                        !dynamic_attn_device_valid_ ||
                        dynamic_attn_param_rows_ < 1 ||
                        dynamic_attn_query_rows_ != query_rows_for_params)
                    {
                        LOG_ERROR("[ROCmFlashAttentionKernelT<FP32>::compute_tensor] "
                                  "Attention device params were not ready before HIP graph capture"
                                  << " captured_body(kv_len=" << kv_len
                                  << ", pos=" << dynamic_position_offset
                                  << ", rows=" << query_rows_for_params << ")"
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
                    setDynamicAttnParams(kv_len, dynamic_position_offset, query_rows_for_params);
                    if (!dynamicAttnParamsReady(kv_len, dynamic_position_offset, query_rows_for_params))
                    {
                        LOG_ERROR("[ROCmFlashAttentionKernelT<FP32>::compute_tensor] Attention device params were not ready on the explicit stream");
                        return false;
                    }
                }

                if (!dynamic_attn_device_valid_)
                {
                    LOG_ERROR("[ROCmFlashAttentionKernelT<FP32>::compute_tensor] Attention device params were not ready on the explicit stream");
                    return false;
                }

                void *d_buf = workspace_->getBuffer(AttentionWorkspaceBuffers::DEVICE_PARAMS);
                if (!d_buf)
                {
                    LOG_ERROR("[ROCmFlashAttentionKernelT<FP32>::compute_tensor] Missing workspace buffer "
                              << AttentionWorkspaceBuffers::DEVICE_PARAMS);
                    return false;
                }
                d_attn_params = static_cast<const attention::AttentionDeviceParams *>(d_buf);
            }

            const float *mask_ptr = nullptr;
            if (workspace_mask)
            {
                mask_ptr = static_cast<const float *>(workspace_mask->gpu_data_ptr());
            }

            // Native KV dispatch: call typed kernel directly, skip FP32 conversion
            if (use_native_kv)
            {
                // Set HIP device context for multi-device TP — required before
                // any hipMalloc, stream creation, or hipBLAS calls in downstream
                // helpers (get_stream_pool, get_dequant_buffer, ensure_scores_buffer).
                if (hipFlashAttn_setDevice(dev) != 0)
                {
                    LOG_ERROR("[ROCmFlashAttentionKernelT<FP32>::compute_tensor] "
                              "Failed to set device "
                              << dev << " for native KV dispatch");
                    return false;
                }

                int result;
                if (small_decode_rows_ > 1)
                {
                    const int max_num_splits = selectFlashDecodeNumSplits(kv_len);

                    allocateWorkspace(n_heads, head_dim, max_num_splits);

                    if (!partial_output_buf_ || !partial_m_buf_ || !partial_l_buf_)
                    {
                        LOG_ERROR("[ROCmFlashAttentionKernelT<FP32>::compute_tensor] Workspace allocation failed for small native decode");
                        return false;
                    }

                    result = 0;
                    ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::FLASH_ATTN_DECODE,
                                                     static_cast<hipStream_t>(stream_));
                    for (int row = 0; row < small_decode_rows_; ++row)
                    {
                        const int row_kv_len = std::max(1, kv_len - (small_decode_rows_ - 1 - row));
                        const int row_num_splits = selectFlashDecodeNumSplits(row_kv_len);
                        const float *row_q =
                            Q_ptr + static_cast<size_t>(row) * static_cast<size_t>(n_heads) *
                                        static_cast<size_t>(head_dim);
                        float *row_o =
                            O_ptr + static_cast<size_t>(row) * static_cast<size_t>(n_heads) *
                                        static_cast<size_t>(head_dim);
                        const attention::AttentionDeviceParams *row_params =
                            d_attn_params ? (d_attn_params + row) : nullptr;

                        if (kv_native_type == TensorType::FP16)
                        {
                            result = hipFlashAttn_decode_fp16(
                                row_q, K->gpu_data_ptr(), V->gpu_data_ptr(), row_o,
                                static_cast<float *>(partial_output_buf_),
                                static_cast<float *>(partial_m_buf_),
                                static_cast<float *>(partial_l_buf_),
                                1, row_kv_len,
                                n_heads, n_kv_heads, head_dim,
                                row_num_splits, row_params, stream_,
                                head_start, gqa_n_rep);
                        }
                        else
                        {
                            result = hipFlashAttn_decode_q8_1(
                                row_q, K->gpu_data_ptr(), V->gpu_data_ptr(), row_o,
                                static_cast<float *>(partial_output_buf_),
                                static_cast<float *>(partial_m_buf_),
                                static_cast<float *>(partial_l_buf_),
                                1, row_kv_len,
                                n_heads, n_kv_heads, head_dim,
                                row_num_splits, row_params, stream_,
                                head_start, gqa_n_rep);
                        }
                        if (result != 0)
                            break;
                    }
                }
                else if (seq_len == 1 && !decode_via_prefill)
                {
                    // Flash Decoding with native KV cache
                    const int num_splits = selectFlashDecodeNumSplits(kv_len);

                    allocateWorkspace(n_heads, head_dim, num_splits);

                    if (!partial_output_buf_ || !partial_m_buf_ || !partial_l_buf_)
                    {
                        LOG_ERROR("[ROCmFlashAttentionKernelT<FP32>::compute_tensor] Workspace allocation failed for native decode");
                        return false;
                    }

                    {
                        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::FLASH_ATTN_DECODE,
                                                         static_cast<hipStream_t>(stream_));
                        if (kv_native_type == TensorType::FP16)
                        {
                            result = hipFlashAttn_decode_fp16(
                                Q_ptr, K->gpu_data_ptr(), V->gpu_data_ptr(), O_ptr,
                                static_cast<float *>(partial_output_buf_),
                                static_cast<float *>(partial_m_buf_),
                                static_cast<float *>(partial_l_buf_),
                                batch_size, kv_len,
                                n_heads, n_kv_heads, head_dim,
                                num_splits, d_attn_params, stream_,
                                head_start, gqa_n_rep);
                        }
                        else
                        {
                            result = hipFlashAttn_decode_q8_1(
                                Q_ptr, K->gpu_data_ptr(), V->gpu_data_ptr(), O_ptr,
                                static_cast<float *>(partial_output_buf_),
                                static_cast<float *>(partial_m_buf_),
                                static_cast<float *>(partial_l_buf_),
                                batch_size, kv_len,
                                n_heads, n_kv_heads, head_dim,
                                num_splits, d_attn_params, stream_,
                                head_start, gqa_n_rep);
                        }
                    }
                }
                else
                {
                    // Prefill with native KV. With
                    // LLAMINAR_ROCM_FA_DECODE_VIA_PREFILL=1, this path is also
                    // an explicit diagnostic/reference path for single-token
                    // decode.
                    ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::FLASH_ATTN_PREFILL,
                                                     static_cast<hipStream_t>(stream_));
                    if (kv_native_type == TensorType::FP16)
                    {
                        result = hipFlashAttn_prefill_fa2_fp16(
                            Q_ptr, K->gpu_data_ptr(), V->gpu_data_ptr(), O_ptr,
                            batch_size, seq_len, kv_len,
                            n_heads, n_kv_heads, head_dim,
                            causal, window_size, 0,
                            d_attn_params, mask_ptr, stream_,
                            head_start, gqa_n_rep);
                    }
                    else
                    {
                        result = hipFlashAttn_prefill_fa2_q8_1(
                            Q_ptr, K->gpu_data_ptr(), V->gpu_data_ptr(), O_ptr,
                            batch_size, seq_len, kv_len,
                            n_heads, n_kv_heads, head_dim,
                            causal, window_size, 0,
                            d_attn_params, mask_ptr, stream_,
                            head_start, gqa_n_rep);
                    }
                }
                if (result != 0)
                {
                    LOG_ERROR("[ROCmFlashAttentionKernelT<FP32>::compute_tensor] Native "
                              << K->dtype_name() << " KV kernel failed: " << result);
                    return false;
                }
                return true;
            }

            return apply_typed(Q_ptr, K_ptr, V_ptr, O_ptr,
                               batch_size, seq_len, kv_len,
                               n_heads, n_kv_heads, head_dim,
                               causal, window_size, 0, dev,
                               d_attn_params, mask_ptr,
                               head_start, gqa_n_rep);
        }
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
                                    sizeof(attention::AttentionDeviceParams) *
                                        static_cast<size_t>(MAX_SMALL_DECODE_ROWS),
                                    256,
                                    true});
            reqs.buffers.push_back({AttentionWorkspaceBuffers::K_TMP_FP32, kv_convert_bytes, 256, true});
            reqs.buffers.push_back({AttentionWorkspaceBuffers::V_TMP_FP32, kv_convert_bytes, 256, true});

            LOG_DEBUG("[ROCmFlashAttentionKernelT<FP32>::getWorkspaceRequirements] "
                      << "batch=" << batch_size << " n_heads=" << n_heads << " head_dim=" << head_dim
                      << " num_splits=" << num_splits
                      << " max_kv_len=" << max_kv_len
                      << " => partial_output=" << (partial_output_bytes / 1024) << "KB"
                      << ", partial_m=" << partial_m_bytes << "B"
                      << ", partial_l=" << partial_l_bytes << "B"
                      << ", kv_convert(each)=" << (kv_convert_bytes / (1024 * 1024)) << "MB");

            return reqs;
        }

        void ROCmFlashAttentionKernelT<ActivationPrecision::FP32>::bindWorkspace(
            DeviceWorkspaceManager *workspace)
        {
            workspace_ = workspace;
            dynamic_attn_device_valid_ = false;
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
              workspace_size_(0), max_splits_(0), workspace_(nullptr), device_ctx_(nullptr)
        {
            if (device_idx < 0)
            {
                throw std::runtime_error(
                    "[ROCmFlashAttentionKernelT<FP16>] Invalid device_idx=" + std::to_string(device_idx) +
                    " — caller must pass explicit device ordinal");
            }
            LOG_DEBUG("[ROCmFlashAttentionKernelT<FP16>] Created for device " << device_idx);
        }

        ROCmFlashAttentionKernelT<ActivationPrecision::FP16>::ROCmFlashAttentionKernelT(
            IWorkerGPUContext *ctx)
            : stream_(nullptr),
              partial_output_buf_(nullptr), partial_m_buf_(nullptr), partial_l_buf_(nullptr),
              workspace_size_(0), max_splits_(0), workspace_(nullptr), device_ctx_(nullptr)
        {
            if (!ctx)
            {
                throw std::runtime_error(
                    "[ROCmFlashAttentionKernelT<FP16>] Device context is null");
            }

            if (!ctx->isInitialized())
            {
                throw std::runtime_error(
                    "[ROCmFlashAttentionKernelT<FP16>] Device context is not initialized");
            }

            setDeviceContext(ctx);
            device_idx_ = ctx->deviceOrdinal();
            stream_ = ctx->defaultStream();

            LOG_DEBUG("[ROCmFlashAttentionKernelT<FP16>] Created for device " << device_idx_
                                                                              << " using device context");
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
              max_splits_(other.max_splits_),
              workspace_(other.workspace_),
              device_ctx_(other.device_ctx_)
        {
            other.stream_ = nullptr;
            other.partial_output_buf_ = nullptr;
            other.partial_m_buf_ = nullptr;
            other.partial_l_buf_ = nullptr;
            other.workspace_size_ = 0;
            other.max_splits_ = 0;
            other.workspace_ = nullptr;
            other.device_ctx_ = nullptr;
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
                workspace_ = other.workspace_;
                device_ctx_ = other.device_ctx_;

                other.stream_ = nullptr;
                other.partial_output_buf_ = nullptr;
                other.partial_m_buf_ = nullptr;
                other.partial_l_buf_ = nullptr;
                other.workspace_size_ = 0;
                other.max_splits_ = 0;
                other.workspace_ = nullptr;
                other.device_ctx_ = nullptr;
            }
            return *this;
        }

        void ROCmFlashAttentionKernelT<ActivationPrecision::FP16>::allocateWorkspace(
            int n_heads, int head_dim, int num_splits)
        {
            // Workspace is now REQUIRED - no legacy allocation path
            if (!validateROCmWorkspaceBinding(workspace_, device_idx_, "ROCmFlashAttentionKernelT<FP16>"))
            {
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
            const IMPIContext *mpi_ctx,
            int device_idx)
        {
            // FP16 kernel with FP32 interface - convert internally
            // For now, delegate to FP32 implementation
            // TODO: Implement native FP16 path with FP32->FP16 conversion kernels
            LOG_WARN("[ROCmFlashAttentionKernelT<FP16>] Using FP32 fallback - native FP16 not yet implemented");

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

            // Fall back to FP32 kernel
            return hipFlashAttn_prefill_fa2(
                       Q, K, V, output,
                       1, seq_len, seq_len,
                       n_heads, n_kv_heads, head_dim,
                       causal, window_size, 0,
                       nullptr, mask_ptr,
                       stream_, 0, 0) == 0;
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

            return hipFlashAttn_prefill_fa2(
                       Q, K, V, output,
                       batch_size, seq_len, seq_len,
                       n_heads, n_kv_heads, head_dim,
                       causal, window_size, 0,
                       nullptr, mask_ptr,
                       stream_, 0, 0) == 0;
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

            const int num_splits = selectFlashDecodeNumSplits(kv_len);

            allocateWorkspace(n_heads, head_dim, num_splits);

            return hipFlashAttn_decode_fp32(
                       Q, K, V, output,
                       static_cast<float *>(partial_output_buf_),
                       static_cast<float *>(partial_m_buf_),
                       static_cast<float *>(partial_l_buf_),
                       1, kv_len,
                       n_heads, n_kv_heads, head_dim,
                       num_splits, nullptr, stream_, 0, 0) == 0;
        }
        bool ROCmFlashAttentionKernelT<ActivationPrecision::FP16>::apply_typed(
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
            const IMPIContext *mpi_ctx,
            int device_idx,
            int head_start,
            int local_n_heads,
            int local_n_kv_heads,
            int gqa_n_rep)
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
            (void)gqa_n_rep;
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
              workspace_size_(0), max_splits_(0), workspace_(nullptr), device_ctx_(nullptr)
        {
            if (device_idx < 0)
            {
                throw std::runtime_error(
                    "[ROCmFlashAttentionKernelT<BF16>] Invalid device_idx=" + std::to_string(device_idx) +
                    " — caller must pass explicit device ordinal");
            }
            // Note: MI50 has limited BF16 support - may fall back to FP32
            LOG_DEBUG("[ROCmFlashAttentionKernelT<BF16>] Created for device " << device_idx
                                                                              << " (Note: MI50 has limited BF16 support)");
        }

        ROCmFlashAttentionKernelT<ActivationPrecision::BF16>::ROCmFlashAttentionKernelT(IWorkerGPUContext *ctx)
            : stream_(nullptr),
              partial_output_buf_(nullptr), partial_m_buf_(nullptr), partial_l_buf_(nullptr),
              workspace_size_(0), max_splits_(0), workspace_(nullptr), device_ctx_(nullptr)
        {
            if (!ctx)
            {
                throw std::runtime_error("[ROCmFlashAttentionKernelT<BF16>] Device context is null");
            }
            if (!ctx->isInitialized())
            {
                throw std::runtime_error("[ROCmFlashAttentionKernelT<BF16>] Device context is not initialized");
            }
            setDeviceContext(ctx);
            device_idx_ = ctx->deviceOrdinal();
            stream_ = ctx->defaultStream();
            // Note: MI50 has limited BF16 support - may fall back to FP32
            LOG_DEBUG("[ROCmFlashAttentionKernelT<BF16>] Created for device " << device_idx_
                                                                              << " using device context (Note: MI50 has limited BF16 support)");
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
              max_splits_(other.max_splits_),
              workspace_(other.workspace_),
              device_ctx_(other.device_ctx_)
        {
            other.stream_ = nullptr;
            other.partial_output_buf_ = nullptr;
            other.partial_m_buf_ = nullptr;
            other.partial_l_buf_ = nullptr;
            other.workspace_size_ = 0;
            other.max_splits_ = 0;
            other.workspace_ = nullptr;
            other.device_ctx_ = nullptr;
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
                workspace_ = other.workspace_;
                device_ctx_ = other.device_ctx_;

                other.stream_ = nullptr;
                other.partial_output_buf_ = nullptr;
                other.partial_m_buf_ = nullptr;
                other.partial_l_buf_ = nullptr;
                other.workspace_size_ = 0;
                other.max_splits_ = 0;
                other.workspace_ = nullptr;
                other.device_ctx_ = nullptr;
            }
            return *this;
        }

        void ROCmFlashAttentionKernelT<ActivationPrecision::BF16>::allocateWorkspace(
            int n_heads, int head_dim, int num_splits)
        {
            // Workspace is now REQUIRED - no legacy allocation path
            if (!validateROCmWorkspaceBinding(workspace_, device_idx_, "ROCmFlashAttentionKernelT<BF16>"))
            {
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
            const IMPIContext *mpi_ctx,
            int device_idx)
        {
            // MI50 doesn't have native BF16 support - fall back to FP32
            LOG_WARN("[ROCmFlashAttentionKernelT<BF16>] MI50 lacks BF16 support - using FP32 fallback");

            (void)workspace_scores;
            (void)workspace_buffer;
            (void)workspace_context;
            (void)use_bf16;
            (void)mpi_ctx;

            const float *mask_ptr = nullptr;
            if (workspace_mask)
            {
                mask_ptr = static_cast<const float *>(workspace_mask->gpu_data_ptr());
            }

            return hipFlashAttn_prefill_fa2(
                       Q, K, V, output,
                       1, seq_len, seq_len,
                       n_heads, n_kv_heads, head_dim,
                       causal, window_size, 0,
                       nullptr, mask_ptr,
                       stream_, 0, 0) == 0;
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

            return hipFlashAttn_prefill_fa2(
                       Q, K, V, output,
                       batch_size, seq_len, seq_len,
                       n_heads, n_kv_heads, head_dim,
                       causal, window_size, 0,
                       nullptr, mask_ptr,
                       stream_, 0, 0) == 0;
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

            const int num_splits = selectFlashDecodeNumSplits(kv_len);

            allocateWorkspace(n_heads, head_dim, num_splits);

            return hipFlashAttn_decode_fp32(
                       Q, K, V, output,
                       static_cast<float *>(partial_output_buf_),
                       static_cast<float *>(partial_m_buf_),
                       static_cast<float *>(partial_l_buf_),
                       1, kv_len,
                       n_heads, n_kv_heads, head_dim,
                       num_splits, nullptr, stream_, 0, 0) == 0;
        }

        bool ROCmFlashAttentionKernelT<ActivationPrecision::BF16>::apply_typed(
            const uint16_t *Q, const uint16_t *K, const uint16_t *V, uint16_t *output,
            int batch_size, int seq_len, int kv_len,
            int n_heads, int n_kv_heads, int head_dim,
            bool causal, int window_size, int position_offset,
            int device_idx,
            const attention::AttentionDeviceParams *device_params,
            const float *mask)
        {
            LOG_ERROR("[ROCmFlashAttentionKernelT<BF16>::apply_typed] Native BF16 not supported on MI50");
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
            const IMPIContext *mpi_ctx,
            int device_idx,
            int head_start,
            int local_n_heads,
            int local_n_kv_heads,
            int gqa_n_rep)
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
            (void)gqa_n_rep;
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
