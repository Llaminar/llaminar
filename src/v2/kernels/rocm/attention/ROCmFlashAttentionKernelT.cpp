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
#include <array>
#include <atomic>
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

    /** @brief Device-resident prefill with native BF16 K/V cache storage. */
    int hipFlashAttn_prefill_fa2_bf16(
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

    /** @brief Split-K decode with native BF16 K/V cache storage. */
    int hipFlashAttn_decode_bf16(
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

    int hipFlashAttn_decode_fp32_grouped_verifier_rows(
        const float *Q, const void *K_cache, const void *V_cache, float *O,
        float *O_partial, float *m_partial, float *l_partial,
        int verifier_rows, int max_kv_len,
        int n_heads, int n_kv_heads, int head_dim,
        int max_num_splits,
        const llaminar2::attention::AttentionDeviceParams *device_params,
        void *stream,
        int head_start, int gqa_n_rep);

    int hipFlashAttn_decode_fp16_grouped_verifier_rows(
        const float *Q, const void *K_cache, const void *V_cache, float *O,
        float *O_partial, float *m_partial, float *l_partial,
        int verifier_rows, int max_kv_len,
        int n_heads, int n_kv_heads, int head_dim,
        int max_num_splits,
        const llaminar2::attention::AttentionDeviceParams *device_params,
        void *stream,
        int head_start, int gqa_n_rep);

    /** @brief Grouped shared-cache verifier decode over native BF16 K/V. */
    int hipFlashAttn_decode_bf16_grouped_verifier_rows(
        const float *Q, const void *K_cache, const void *V_cache, float *O,
        float *O_partial, float *m_partial, float *l_partial,
        int verifier_rows, int max_kv_len,
        int n_heads, int n_kv_heads, int head_dim,
        int max_num_splits,
        const llaminar2::attention::AttentionDeviceParams *device_params,
        void *stream,
        int head_start, int gqa_n_rep);

    int hipFlashAttn_decode_q8_1_grouped_verifier_rows(
        const float *Q, const void *K_cache, const void *V_cache, float *O,
        float *O_partial, float *m_partial, float *l_partial,
        int verifier_rows, int max_kv_len,
        int n_heads, int n_kv_heads, int head_dim,
        int max_num_splits,
        const llaminar2::attention::AttentionDeviceParams *device_params,
        void *stream,
        int head_start, int gqa_n_rep);

    int hipFlashAttn_decode_fp32_grouped_request_rows(
        const float *Q, const void *K_cache, const void *V_cache, float *O,
        float *O_partial, float *m_partial, float *l_partial,
        int request_count, int query_rows, int max_kv_len,
        int n_heads, int n_kv_heads, int head_dim,
        int max_num_splits,
        const llaminar2::attention::AttentionDeviceParams *device_params,
        void *stream,
        int head_start, int gqa_n_rep);

    int hipFlashAttn_decode_fp16_grouped_request_rows(
        const float *Q, const void *K_cache, const void *V_cache, float *O,
        float *O_partial, float *m_partial, float *l_partial,
        int request_count, int query_rows, int max_kv_len,
        int n_heads, int n_kv_heads, int head_dim,
        int max_num_splits,
        const llaminar2::attention::AttentionDeviceParams *device_params,
        void *stream,
        int head_start, int gqa_n_rep);

    /** @brief Grouped independent-request decode over native BF16 K/V. */
    int hipFlashAttn_decode_bf16_grouped_request_rows(
        const float *Q, const void *K_cache, const void *V_cache, float *O,
        float *O_partial, float *m_partial, float *l_partial,
        int request_count, int query_rows, int max_kv_len,
        int n_heads, int n_kv_heads, int head_dim,
        int max_num_splits,
        const llaminar2::attention::AttentionDeviceParams *device_params,
        void *stream,
        int head_start, int gqa_n_rep);

    int hipFlashAttn_decode_q8_1_grouped_request_rows(
        const float *Q, const void *K_cache, const void *V_cache, float *O,
        float *O_partial, float *m_partial, float *l_partial,
        int request_count, int query_rows, int max_kv_len,
        int n_heads, int n_kv_heads, int head_dim,
        int max_num_splits,
        const llaminar2::attention::AttentionDeviceParams *device_params,
        void *stream,
        int head_start, int gqa_n_rep);

    int hipFlashAttn_prepare_device_params_from_count(
        void *device_params,
        const int *post_append_cached_tokens,
        int seq_len,
        int query_rows,
        int kv_stride,
        const int *active_query_rows_device,
        void *stream);

    int hipFlashAttn_prepare_device_params_from_geometry(
        void *device_params,
        int kv_len,
        int kv_stride,
        int position_offset,
        int query_rows,
        void *stream);

    int hipFlashAttn_prepare_device_params_from_request_counts(
        void *device_params,
        const int *post_append_cached_tokens,
        int request_count,
        int query_rows,
        int kv_stride,
        void *stream);

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
        constexpr int MAX_FLASH_DECODE_SPLITS = 32;
        constexpr int MIN_KV_ROWS_PER_SPLIT = 16;
        constexpr int TARGET_RESIDENT_BLOCKS_PER_CU = 7;
        /*
         * MTP target verification runs `draft_count + 1` continuation rows.
         * Draft depths through fifteen therefore require M=2..16 to remain on
         * the continuation/decode path. Treating any of those rows as prefill
         * changes causal semantics and can make ROCm accept a token CUDA/CPU
         * reject.
         */
        constexpr int MAX_SMALL_DECODE_ROWS =
            attention::kMaxGroupedVerifierAttentionRows;

        /**
         * @brief Select a graph-stable physical split envelope.
         *
         * Production callers pass the KV cache capacity, query-head count, and
         * device ordinal. The captured grid therefore remains invariant while
         * the HIP kernel activates the appropriate split and wavefront prefix
         * from device-resident sequence metadata.
         *
         * gfx906 can retain seven 36-VGPR wavefronts per SIMD for the native
         * FP16 decode kernel. One 256-thread workgroup contributes one
         * wavefront to each of the CU's four SIMDs, so seven resident
         * workgroups per CU are available before VGPR pressure becomes the
         * limiter. The physical envelope is rounded upward to a power of two
         * so the device policy can select 1/2/4/8/16/32 active split planes
         * without recapturing the graph.
         */
        static int selectFlashDecodeSplitEnvelope(
            int kv_capacity,
            int n_heads,
            int device_idx)
        {
            if (debugEnv().gemm.deterministic)
                return 1;

            if (kv_capacity <= 0 || n_heads <= 0 || device_idx < 0)
            {
                throw std::invalid_argument(
                    "[ROCmFlashAttentionKernelT] Stable decode envelope "
                    "requires positive KV capacity/query-head count and an "
                    "explicit device ordinal");
            }

            if (kv_capacity <= 64)
                return 1;

            constexpr size_t kMaxCachedDevices = 64;
            if (static_cast<size_t>(device_idx) >= kMaxCachedDevices)
            {
                throw std::out_of_range(
                    "[ROCmFlashAttentionKernelT] Device ordinal exceeds the "
                    "fixed launch-property cache");
            }

            static std::array<std::atomic<int>, kMaxCachedDevices> cu_cache{};
            int num_cus =
                cu_cache[static_cast<size_t>(device_idx)].load(
                    std::memory_order_relaxed);
            if (num_cus == 0)
            {
                const hipError_t status = hipDeviceGetAttribute(
                    &num_cus,
                    hipDeviceAttributeMultiprocessorCount,
                    device_idx);
                if (status != hipSuccess || num_cus <= 0)
                {
                    throw std::runtime_error(
                        "[ROCmFlashAttentionKernelT] Cannot construct a stable "
                        "decode launch envelope because HIP did not publish a "
                        "positive CU count for device " +
                        std::to_string(device_idx) +
                        " (status=" +
                        std::to_string(static_cast<int>(status)) + ")");
                }
                cu_cache[static_cast<size_t>(device_idx)].store(
                    num_cus,
                    std::memory_order_relaxed);
            }

            const int occupancy_splits =
                (TARGET_RESIDENT_BLOCKS_PER_CU * num_cus + n_heads - 1) /
                n_heads;
            const int capacity_splits =
                std::max(1, kv_capacity / MIN_KV_ROWS_PER_SPLIT);
            const int desired_splits = std::clamp(
                std::min(occupancy_splits, capacity_splits),
                1,
                MAX_FLASH_DECODE_SPLITS);

            int envelope = 1;
            while (envelope < desired_splits &&
                   envelope < MAX_FLASH_DECODE_SPLITS)
            {
                envelope <<= 1;
            }
            return std::clamp(envelope, 1, MAX_FLASH_DECODE_SPLITS);
        }

        // =====================================================================
        // FP32 Specialization Implementation
        // =====================================================================

        ROCmFlashAttentionKernelT<ActivationPrecision::FP32>::ROCmFlashAttentionKernelT(int device_idx)
            : device_idx_(device_idx), stream_(nullptr),
              partial_output_buf_(nullptr), partial_m_buf_(nullptr), partial_l_buf_(nullptr),
              workspace_size_(0), max_splits_(0), workspace_(nullptr), device_ctx_(nullptr),
              small_decode_rows_(0)
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
              small_decode_rows_(0)
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

            LOG_DEBUG("[ROCmFlashAttentionKernelT<FP32>] Created for device " << device_idx_
                                                                              << "; awaiting explicit execution stream");
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
              max_splits_(other.max_splits_),
              workspace_(other.workspace_),
              device_ctx_(other.device_ctx_),
              small_decode_rows_(other.small_decode_rows_),
              dynamic_attn_kv_len_(other.dynamic_attn_kv_len_),
              dynamic_attn_kv_stride_(other.dynamic_attn_kv_stride_),
              dynamic_attn_position_offset_(other.dynamic_attn_position_offset_),
              dynamic_attn_query_rows_(other.dynamic_attn_query_rows_),
              dynamic_attn_param_rows_(other.dynamic_attn_param_rows_),
              dynamic_attn_device_valid_(other.dynamic_attn_device_valid_),
              dynamic_attn_device_derived_(other.dynamic_attn_device_derived_)
        {
            other.stream_ = nullptr;
            other.partial_output_buf_ = nullptr;
            other.partial_m_buf_ = nullptr;
            other.partial_l_buf_ = nullptr;
            other.workspace_size_ = 0;
            other.max_splits_ = 0;
            other.workspace_ = nullptr;
            other.device_ctx_ = nullptr;
            other.small_decode_rows_ = 0;
            other.dynamic_attn_device_valid_ = false;
            other.dynamic_attn_device_derived_ = false;
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
                workspace_ = other.workspace_;
                device_ctx_ = other.device_ctx_;
                small_decode_rows_ = other.small_decode_rows_;
                dynamic_attn_kv_len_ = other.dynamic_attn_kv_len_;
                dynamic_attn_kv_stride_ = other.dynamic_attn_kv_stride_;
                dynamic_attn_position_offset_ = other.dynamic_attn_position_offset_;
                dynamic_attn_query_rows_ = other.dynamic_attn_query_rows_;
                dynamic_attn_param_rows_ = other.dynamic_attn_param_rows_;
                dynamic_attn_device_valid_ = other.dynamic_attn_device_valid_;
                dynamic_attn_device_derived_ = other.dynamic_attn_device_derived_;

                other.stream_ = nullptr;
                other.partial_output_buf_ = nullptr;
                other.partial_m_buf_ = nullptr;
                other.partial_l_buf_ = nullptr;
                other.workspace_size_ = 0;
                other.max_splits_ = 0;
                other.workspace_ = nullptr;
                other.device_ctx_ = nullptr;
                other.small_decode_rows_ = 0;
                other.dynamic_attn_device_valid_ = false;
                other.dynamic_attn_device_derived_ = false;
            }
            return *this;
        }

        bool ROCmFlashAttentionKernelT<ActivationPrecision::FP32>::writeDynamicAttnParams(
            int kv_len,
            int kv_stride,
            int position_offset,
            int query_rows,
            void *stream)
        {
            if (!stream)
            {
                LOG_ERROR("[ROCmFlashAttentionKernelT<FP32>] Cannot write attention params on a null/default HIP stream");
                dynamic_attn_device_valid_ = false;
                return false;
            }
            if (!workspace_)
            {
                LOG_ERROR("[ROCmFlashAttentionKernelT<FP32>] Cannot write attention params without a bound workspace");
                dynamic_attn_device_valid_ = false;
                return false;
            }

            void *d_buf = workspace_->getBuffer(AttentionWorkspaceBuffers::DEVICE_PARAMS);
            const size_t required_bytes =
                sizeof(attention::AttentionDeviceParams) *
                static_cast<size_t>(query_rows);
            if (!d_buf ||
                workspace_->getBufferSize(AttentionWorkspaceBuffers::DEVICE_PARAMS) <
                    required_bytes)
            {
                LOG_ERROR("[ROCmFlashAttentionKernelT<FP32>] Missing workspace buffer "
                          << AttentionWorkspaceBuffers::DEVICE_PARAMS
                          << " for " << query_rows << " attention param row(s)");
                dynamic_attn_device_valid_ = false;
                return false;
            }

            if (hipFlashAttn_prepare_device_params_from_geometry(
                    d_buf,
                    kv_len,
                    kv_stride,
                    position_offset,
                    query_rows,
                    stream) != 0)
            {
                LOG_ERROR("[ROCmFlashAttentionKernelT<FP32>] Device attention-param writer failed"
                          << " kv_len=" << kv_len
                          << " kv_stride=" << kv_stride
                          << " position_offset=" << position_offset
                          << " query_rows=" << query_rows);
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
            return dynamic_attn_device_valid_ &&
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

            LOG_TRACE("[ROCmFlashAttentionKernelT<FP32>] Using managed workspace buffers");
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
                    (void)hipStreamIsCapturing(static_cast<hipStream_t>(stream_), &cap_status);
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
                const int launch_kv_capacity =
                    dynamic_attn_kv_stride_ > 0 ? dynamic_attn_kv_stride_ : kv_len;
                const int num_splits =
                    selectFlashDecodeSplitEnvelope(
                        launch_kv_capacity,
                        n_heads,
                        device_idx);

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
            /*
             * Live sequence state may select a smaller active prefix, but it
             * must not shrink the cache-capacity envelope already bound to a
             * graph.  Only resetDynamicState() starts a new capacity lifetime.
             */
            const int resolved_kv_stride = std::max(
                kv_len,
                dynamic_attn_kv_stride_ > 0
                    ? dynamic_attn_kv_stride_
                    : kv_len);
            const bool same_params =
                !dynamic_attn_device_derived_ &&
                dynamic_attn_device_valid_ &&
                dynamic_attn_kv_len_ == kv_len &&
                dynamic_attn_kv_stride_ == resolved_kv_stride &&
                dynamic_attn_position_offset_ == position_offset &&
                dynamic_attn_query_rows_ == sanitized_query_rows &&
                dynamic_attn_param_rows_ == param_rows;

            if (same_params)
                return;

            dynamic_attn_kv_len_ = kv_len;
            dynamic_attn_kv_stride_ = resolved_kv_stride;
            dynamic_attn_position_offset_ = position_offset;
            dynamic_attn_query_rows_ = sanitized_query_rows;
            dynamic_attn_param_rows_ = param_rows;
            dynamic_attn_device_valid_ = false;
            dynamic_attn_device_derived_ = false;

            if (!stream_ || !workspace_)
                return;

            (void)writeDynamicAttnParams(
                kv_len,
                resolved_kv_stride,
                position_offset,
                sanitized_query_rows,
                stream_);
        }

        bool ROCmFlashAttentionKernelT<ActivationPrecision::FP32>::prepareDynamicAttnParams(
            int kv_len,
            int position_offset,
            int query_rows,
            void *stream,
            int kv_stride)
        {
            if (!stream)
            {
                LOG_ERROR("[ROCmFlashAttentionKernelT<FP32>] prepareDynamicAttnParams requires an explicit non-null HIP stream");
                dynamic_attn_device_valid_ = false;
                return false;
            }
            setGPUStream(stream);
            const int resolved_kv_stride =
                std::max(kv_len, kv_stride > 0 ? kv_stride : kv_len);
            const int sanitized_query_rows =
                (query_rows > 1 && query_rows <= MAX_SMALL_DECODE_ROWS)
                    ? query_rows
                    : 1;
            const int param_rows = std::max(1, sanitized_query_rows);
            const bool same_params =
                !dynamic_attn_device_derived_ &&
                dynamic_attn_device_valid_ &&
                dynamic_attn_kv_len_ == kv_len &&
                dynamic_attn_kv_stride_ == resolved_kv_stride &&
                dynamic_attn_position_offset_ == position_offset &&
                dynamic_attn_query_rows_ == sanitized_query_rows &&
                dynamic_attn_param_rows_ == param_rows;
            if (!same_params)
            {
                small_decode_rows_ =
                    sanitized_query_rows > 1 ? sanitized_query_rows : 0;
                dynamic_attn_kv_len_ = kv_len;
                dynamic_attn_kv_stride_ = resolved_kv_stride;
                dynamic_attn_position_offset_ = position_offset;
                dynamic_attn_query_rows_ = sanitized_query_rows;
                dynamic_attn_param_rows_ = param_rows;
                dynamic_attn_device_valid_ = false;
                dynamic_attn_device_derived_ = false;
                if (!writeDynamicAttnParams(
                        kv_len,
                        resolved_kv_stride,
                        position_offset,
                        sanitized_query_rows,
                        stream))
                {
                    return false;
                }
            }
            const bool ready =
                dynamicAttnParamsReady(kv_len, position_offset, query_rows) &&
                dynamic_attn_kv_stride_ == resolved_kv_stride;
            if (!ready)
            {
                LOG_ERROR("[ROCmFlashAttentionKernelT<FP32>] Dynamic attention params not ready after prepare"
                          << " requested(kv_len=" << kv_len
                          << ", kv_stride=" << resolved_kv_stride
                          << ", pos=" << position_offset
                          << ", rows=" << sanitized_query_rows << ")"
                          << " actual(kv_len=" << dynamic_attn_kv_len_
                          << ", kv_stride=" << dynamic_attn_kv_stride_
                          << ", pos=" << dynamic_attn_position_offset_
                          << ", rows=" << dynamic_attn_query_rows_
                          << ", param_rows=" << dynamic_attn_param_rows_
                          << ") device_valid=" << dynamic_attn_device_valid_
                          << " workspace=" << (workspace_ != nullptr)
                          << " stream=" << stream_);
            }
            return ready;
        }

        bool ROCmFlashAttentionKernelT<ActivationPrecision::FP32>::prepareDynamicAttnParamsFromDeviceSequenceState(
            const int *post_append_cached_tokens_device,
            int seq_len,
            int query_rows,
            void *stream,
            int kv_stride,
            const int *active_query_rows_device,
            const attention::AttentionPrefillCaptureGeometry &prefill_capture)
        {
            (void)prefill_capture;
            const int sanitized_query_rows =
                (query_rows > 1 && query_rows <= MAX_SMALL_DECODE_ROWS) ? query_rows : 1;
            if (!post_append_cached_tokens_device || seq_len <= 0 ||
                kv_stride <= 0 || !stream)
            {
                LOG_ERROR("[ROCmFlashAttentionKernelT<FP32>] Device-derived attention params require count pointer, positive seq_len/cache capacity, and explicit stream");
                dynamic_attn_device_valid_ = false;
                dynamic_attn_device_derived_ = false;
                return false;
            }
            if (!workspace_)
            {
                LOG_ERROR("[ROCmFlashAttentionKernelT<FP32>] Device-derived attention params require a bound workspace");
                dynamic_attn_device_valid_ = false;
                dynamic_attn_device_derived_ = false;
                return false;
            }

            void *d_buf = workspace_->getBuffer(AttentionWorkspaceBuffers::DEVICE_PARAMS);
            if (!d_buf)
            {
                LOG_ERROR("[ROCmFlashAttentionKernelT<FP32>] Missing workspace buffer "
                          << AttentionWorkspaceBuffers::DEVICE_PARAMS
                          << " for device-derived attention params");
                dynamic_attn_device_valid_ = false;
                dynamic_attn_device_derived_ = false;
                return false;
            }

            setGPUStream(stream);
            const int rc = hipFlashAttn_prepare_device_params_from_count(
                d_buf,
                post_append_cached_tokens_device,
                seq_len,
                sanitized_query_rows,
                kv_stride,
                active_query_rows_device,
                stream);
            if (rc != 0)
            {
                LOG_ERROR("[ROCmFlashAttentionKernelT<FP32>] Failed to derive attention params from device KV count");
                dynamic_attn_device_valid_ = false;
                dynamic_attn_device_derived_ = false;
                return false;
            }

            small_decode_rows_ = (sanitized_query_rows > 1) ? sanitized_query_rows : 0;
            dynamic_attn_kv_len_ = 0;
            dynamic_attn_kv_stride_ = kv_stride;
            dynamic_attn_position_offset_ = 0;
            dynamic_attn_query_rows_ = sanitized_query_rows;
            dynamic_attn_param_rows_ = sanitized_query_rows;
            dynamic_attn_device_valid_ = true;
            dynamic_attn_device_derived_ = true;
            return true;
        }

        void ROCmFlashAttentionKernelT<ActivationPrecision::FP32>::resetDynamicState()
        {
            small_decode_rows_ = 0;
            dynamic_attn_kv_len_ = 0;
            dynamic_attn_kv_stride_ = 0;
            dynamic_attn_position_offset_ = 0;
            dynamic_attn_query_rows_ = 1;
            dynamic_attn_param_rows_ = 1;
            dynamic_attn_device_valid_ = false;
            dynamic_attn_device_derived_ = false;
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
            int gqa_n_rep,
            const attention::AttentionExecutionPolicy &execution_policy)
        {
            (void)workspace_scores;
            (void)mpi_ctx;
            (void)local_n_heads;
            (void)local_n_kv_heads;

            if (execution_policy.prefill_parallel_axis !=
                attention::AttentionPrefillParallelAxis::QuerySequence)
            {
                LOG_ERROR("[ROCmFlashAttentionKernelT<FP32>::compute_tensor] Unsupported declared prefill parallel axis: "
                          << attention::attentionPrefillParallelAxisName(
                                 execution_policy.prefill_parallel_axis));
                return false;
            }

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
                     kv_native_type == TensorType::BF16 ||
                     (kv_native_type == TensorType::Q8_1 && head_dim % 32 == 0)))
                {
                    use_native_kv = true;
                    LOG_TRACE("[ROCmFlashAttentionKernelT<FP32>::compute_tensor] Native "
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

            LOG_TRACE("[ROCmFlashAttentionKernelT<FP32>::compute_tensor] batch=" << batch_size
                                                                                 << " seq_len=" << seq_len << " kv_len=" << kv_len
                                                                                 << " n_heads=" << n_heads << " n_kv_heads=" << n_kv_heads
                                                                                 << " head_dim=" << head_dim << " causal=" << causal
                                                                                 << " device_idx=" << dev);

            // Wire the device-owned parameter block for graph-capture replay.
            // Resident cache execution derives it from the live device count;
            // explicit-geometry callers populate the same block with a tiny
            // stream-ordered kernel. Neither path has a host parameter mirror.
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

                if (dynamic_attn_device_derived_)
                {
                    if (!dynamic_attn_device_valid_ ||
                        dynamic_attn_param_rows_ < query_rows_for_params ||
                        dynamic_attn_query_rows_ != query_rows_for_params)
                    {
                        LOG_ERROR("[ROCmFlashAttentionKernelT<FP32>::compute_tensor] "
                                  "Device-derived attention params were not ready"
                                  << " rows=" << query_rows_for_params
                                  << " prepared_rows=" << dynamic_attn_query_rows_
                                  << " param_rows=" << dynamic_attn_param_rows_
                                  << " device_valid=" << dynamic_attn_device_valid_);
                        return false;
                    }
                }
                else if (cap_status == hipStreamCaptureStatusActive)
                {
                    if (!dynamic_attn_device_valid_ ||
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
                                  << ") device_valid=" << dynamic_attn_device_valid_
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
                // Set the HIP device context before launching native cache
                // kernels in a multi-device TP worker.
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
                    /*
                     * Enter the named grouped contract instead of hiding a
                     * row-replay loop inside compute_tensor().  The method
                     * below consumes this already-prepared device parameter
                     * block and launches one phase grid plus one reduction.
                     */
                    return compute_verifier_rows_decode_equivalent(
                        Q,
                        K,
                        V,
                        output,
                        small_decode_rows_,
                        kv_len,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        causal,
                        window_size,
                        mpi_ctx,
                        dev,
                        head_start,
                        gqa_n_rep);
                }
                else if (seq_len == 1 && !decode_via_prefill)
                {
                    // Flash Decoding with native KV cache
                    const int launch_kv_capacity =
                        dynamic_attn_kv_stride_ > 0 ? dynamic_attn_kv_stride_ : kv_len;
                    const int num_splits =
                        selectFlashDecodeSplitEnvelope(
                            launch_kv_capacity,
                            n_heads,
                            dev);

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
                        else if (kv_native_type == TensorType::BF16)
                        {
                            result = hipFlashAttn_decode_bf16(
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
                    else if (kv_native_type == TensorType::BF16)
                    {
                        result = hipFlashAttn_prefill_fa2_bf16(
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

        bool ROCmFlashAttentionKernelT<ActivationPrecision::FP32>::compute_verifier_rows_decode_equivalent(
            const ITensor *Q,
            const ITensor *K,
            const ITensor *V,
            ITensor *output,
            int verifier_rows,
            int kv_len,
            int n_heads,
            int n_kv_heads,
            int head_dim,
            bool causal,
            int window_size,
            const IMPIContext *mpi_ctx,
            int device_idx,
            int head_start,
            int gqa_n_rep)
        {
            (void)window_size;
            (void)mpi_ctx;
            if (!Q || !K || !V || !output)
            {
                LOG_ERROR("[ROCmFlashAttentionKernelT<FP32>::compute_verifier_rows_decode_equivalent] Null tensor");
                return false;
            }
            if (verifier_rows < 2 || verifier_rows > MAX_SMALL_DECODE_ROWS ||
                kv_len <= verifier_rows || !causal)
            {
                LOG_ERROR("[ROCmFlashAttentionKernelT<FP32>::compute_verifier_rows_decode_equivalent] Invalid verifier span"
                          << " rows=" << verifier_rows
                          << " kv_len=" << kv_len
                          << " causal=" << causal);
                return false;
            }
            if (!stream_ || !workspace_)
            {
                LOG_ERROR("[ROCmFlashAttentionKernelT<FP32>::compute_verifier_rows_decode_equivalent] "
                          "requires explicit HIP stream and bound workspace");
                return false;
            }
            if (Q->native_type() != TensorType::FP32 ||
                output->native_type() != TensorType::FP32)
            {
                LOG_ERROR("[ROCmFlashAttentionKernelT<FP32>::compute_verifier_rows_decode_equivalent] "
                          "requires FP32 Q/output, got Q=" << Q->dtype_name()
                                                           << " O=" << output->dtype_name());
                return false;
            }
            if (K->native_type() != V->native_type())
            {
                LOG_ERROR("[ROCmFlashAttentionKernelT<FP32>::compute_verifier_rows_decode_equivalent] "
                          "mixed K/V cache types are unsupported: K=" << K->dtype_name()
                                                                      << " V=" << V->dtype_name());
                return false;
            }
            if (K->native_type() != TensorType::FP32 &&
                K->native_type() != TensorType::BF16 &&
                K->native_type() != TensorType::FP16 &&
                K->native_type() != TensorType::Q8_1)
            {
                LOG_ERROR("[ROCmFlashAttentionKernelT<FP32>::compute_verifier_rows_decode_equivalent] "
                          "unsupported grouped K/V cache type " << K->dtype_name());
                return false;
            }
            if (Q->rows() < static_cast<size_t>(verifier_rows) ||
                output->rows() < static_cast<size_t>(verifier_rows) ||
                K->rows() < static_cast<size_t>(kv_len) ||
                V->rows() < static_cast<size_t>(kv_len))
            {
                LOG_ERROR("[ROCmFlashAttentionKernelT<FP32>::compute_verifier_rows_decode_equivalent] "
                          "undersized grouped tensor geometry"
                          << " verifier_rows=" << verifier_rows
                          << " kv_len=" << kv_len
                          << " Q_rows=" << Q->rows()
                          << " K_rows=" << K->rows()
                          << " V_rows=" << V->rows()
                          << " O_rows=" << output->rows());
                return false;
            }

            /*
             * The small-M ROCm path keys off the prepared row-local attention
             * params.  Prepare them here so the graph stage has a single
             * explicit decode-equivalence contract to call.
             */
            const int position_offset = kv_len - verifier_rows;
            if (!dynamic_attn_device_derived_ &&
                !prepareDynamicAttnParams(kv_len, position_offset, verifier_rows, stream_))
            {
                LOG_ERROR("[ROCmFlashAttentionKernelT<FP32>::compute_verifier_rows_decode_equivalent] "
                          "failed preparing dynamic params");
                return false;
            }

            const float *q_ptr = static_cast<const float *>(Q->gpu_data_ptr());
            const void *k_ptr = K->gpu_data_ptr();
            const void *v_ptr = V->gpu_data_ptr();
            float *output_ptr = static_cast<float *>(output->gpu_data_ptr());
            if (!q_ptr || !k_ptr || !v_ptr || !output_ptr)
            {
                LOG_ERROR("[ROCmFlashAttentionKernelT<FP32>::compute_verifier_rows_decode_equivalent] Missing GPU data pointer");
                return false;
            }

            const int dev = device_idx >= 0 ? device_idx : device_idx_;
            if (hipFlashAttn_setDevice(dev) != 0)
            {
                LOG_ERROR("[ROCmFlashAttentionKernelT<FP32>::compute_verifier_rows_decode_equivalent] Failed to set ROCm device "
                          << dev);
                return false;
            }

            const int launch_kv_capacity =
                dynamic_attn_kv_stride_ > 0 ? dynamic_attn_kv_stride_ : kv_len;
            const int max_num_splits =
                selectFlashDecodeSplitEnvelope(
                    launch_kv_capacity,
                    n_heads,
                    dev);
            allocateWorkspace(n_heads, head_dim, max_num_splits);
            if (!partial_output_buf_ || !partial_m_buf_ || !partial_l_buf_)
            {
                LOG_ERROR("[ROCmFlashAttentionKernelT<FP32>::compute_verifier_rows_decode_equivalent] Workspace binding failed");
                return false;
            }

            const size_t partial_rows =
                static_cast<size_t>(verifier_rows) *
                static_cast<size_t>(n_heads) *
                static_cast<size_t>(max_num_splits);
            const size_t required_partial_output =
                partial_rows * static_cast<size_t>(head_dim) * sizeof(float);
            const size_t required_partial_meta = partial_rows * sizeof(float);
            if (workspace_->getBufferSize(AttentionWorkspaceBuffers::PARTIAL_OUTPUT) <
                    required_partial_output ||
                workspace_->getBufferSize(AttentionWorkspaceBuffers::PARTIAL_M) <
                    required_partial_meta ||
                workspace_->getBufferSize(AttentionWorkspaceBuffers::PARTIAL_L) <
                    required_partial_meta)
            {
                LOG_ERROR("[ROCmFlashAttentionKernelT<FP32>::compute_verifier_rows_decode_equivalent] "
                          "Grouped verifier workspace is too small"
                          << " required_output=" << required_partial_output
                          << " required_meta=" << required_partial_meta
                          << " rows=" << verifier_rows);
                return false;
            }

            void *device_params_buffer =
                workspace_->getBuffer(AttentionWorkspaceBuffers::DEVICE_PARAMS);
            if (!device_params_buffer ||
                workspace_->getBufferSize(AttentionWorkspaceBuffers::DEVICE_PARAMS) <
                    static_cast<size_t>(verifier_rows) *
                        sizeof(attention::AttentionDeviceParams))
            {
                LOG_ERROR("[ROCmFlashAttentionKernelT<FP32>::compute_verifier_rows_decode_equivalent] Missing grouped device params workspace");
                return false;
            }
            const auto *device_params =
                static_cast<const attention::AttentionDeviceParams *>(
                    device_params_buffer);

            TensorType execution_kv_type = K->native_type();
            const bool convert_to_fp32 =
                debugEnv().rocm.fa_disable_native_kv &&
                execution_kv_type != TensorType::FP32;
            if (convert_to_fp32)
            {
                float *k_tmp = static_cast<float *>(
                    workspace_->getBuffer(AttentionWorkspaceBuffers::K_TMP_FP32));
                float *v_tmp = static_cast<float *>(
                    workspace_->getBuffer(AttentionWorkspaceBuffers::V_TMP_FP32));
                const size_t logical_elements =
                    static_cast<size_t>(kv_len) *
                    static_cast<size_t>(n_kv_heads) *
                    static_cast<size_t>(head_dim);
                const size_t required_bytes = logical_elements * sizeof(float);
                if (!k_tmp || !v_tmp ||
                    workspace_->getBufferSize(AttentionWorkspaceBuffers::K_TMP_FP32) <
                        required_bytes ||
                    workspace_->getBufferSize(AttentionWorkspaceBuffers::V_TMP_FP32) <
                        required_bytes ||
                    !hip_convert_tensor_to_fp32(
                        k_ptr,
                        execution_kv_type,
                        k_tmp,
                        static_cast<int>(logical_elements),
                        kv_len,
                        n_kv_heads * head_dim,
                        static_cast<hipStream_t>(stream_)) ||
                    !hip_convert_tensor_to_fp32(
                        v_ptr,
                        execution_kv_type,
                        v_tmp,
                        static_cast<int>(logical_elements),
                        kv_len,
                        n_kv_heads * head_dim,
                        static_cast<hipStream_t>(stream_)))
                {
                    LOG_ERROR("[ROCmFlashAttentionKernelT<FP32>::compute_verifier_rows_decode_equivalent] Grouped K/V conversion failed");
                    return false;
                }
                k_ptr = k_tmp;
                v_ptr = v_tmp;
                execution_kv_type = TensorType::FP32;
            }

            int result = -1;
            {
                ROCM_KERNEL_PROFILE_SCOPE_STREAM(
                    ROCmKernelType::FLASH_ATTN_DECODE,
                    static_cast<hipStream_t>(stream_));
                if (execution_kv_type == TensorType::FP16)
                {
                    result = hipFlashAttn_decode_fp16_grouped_verifier_rows(
                        q_ptr, k_ptr, v_ptr, output_ptr,
                        static_cast<float *>(partial_output_buf_),
                        static_cast<float *>(partial_m_buf_),
                        static_cast<float *>(partial_l_buf_),
                        verifier_rows, kv_len,
                        n_heads, n_kv_heads, head_dim,
                        max_num_splits, device_params, stream_,
                        head_start, gqa_n_rep);
                }
                else if (execution_kv_type == TensorType::BF16)
                {
                    result = hipFlashAttn_decode_bf16_grouped_verifier_rows(
                        q_ptr, k_ptr, v_ptr, output_ptr,
                        static_cast<float *>(partial_output_buf_),
                        static_cast<float *>(partial_m_buf_),
                        static_cast<float *>(partial_l_buf_),
                        verifier_rows, kv_len,
                        n_heads, n_kv_heads, head_dim,
                        max_num_splits, device_params, stream_,
                        head_start, gqa_n_rep);
                }
                else if (execution_kv_type == TensorType::Q8_1)
                {
                    result = hipFlashAttn_decode_q8_1_grouped_verifier_rows(
                        q_ptr, k_ptr, v_ptr, output_ptr,
                        static_cast<float *>(partial_output_buf_),
                        static_cast<float *>(partial_m_buf_),
                        static_cast<float *>(partial_l_buf_),
                        verifier_rows, kv_len,
                        n_heads, n_kv_heads, head_dim,
                        max_num_splits, device_params, stream_,
                        head_start, gqa_n_rep);
                }
                else
                {
                    result = hipFlashAttn_decode_fp32_grouped_verifier_rows(
                        q_ptr, k_ptr, v_ptr, output_ptr,
                        static_cast<float *>(partial_output_buf_),
                        static_cast<float *>(partial_m_buf_),
                        static_cast<float *>(partial_l_buf_),
                        verifier_rows, kv_len,
                        n_heads, n_kv_heads, head_dim,
                        max_num_splits, device_params, stream_,
                        head_start, gqa_n_rep);
                }
            }
            if (result != 0)
            {
                LOG_ERROR("[ROCmFlashAttentionKernelT<FP32>::compute_verifier_rows_decode_equivalent] Grouped verifier decode failed"
                          << " rows=" << verifier_rows
                          << " kv_len=" << kv_len
                          << " max_num_splits=" << max_num_splits
                          << " kv_type=" << K->dtype_name());
                return false;
            }
            return true;
        }

        bool ROCmFlashAttentionKernelT<ActivationPrecision::FP32>::compute_device_request_batch_decode_equivalent(
            const ITensor *Q,
            const ITensor *K,
            const ITensor *V,
            const int *post_append_cached_tokens_device,
            ITensor *output,
            int request_count,
            int query_rows,
            int max_kv_len,
            int n_heads,
            int n_kv_heads,
            int head_dim,
            bool causal,
            int window_size,
            const IMPIContext *mpi_ctx,
            int device_idx,
            int head_start,
            int gqa_n_rep)
        {
            (void)window_size;
            (void)mpi_ctx;
            const int total_rows = request_count * query_rows;
            if (!Q || !K || !V || !output ||
                !post_append_cached_tokens_device)
            {
                LOG_ERROR("[ROCmFlashAttentionKernelT<FP32>::compute_device_request_batch_decode_equivalent] Null request-batch input");
                return false;
            }
            if (request_count < 2 || query_rows <= 0 ||
                total_rows > MAX_SMALL_DECODE_ROWS || max_kv_len <= 0 ||
                n_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0 || !causal)
            {
                LOG_ERROR("[ROCmFlashAttentionKernelT<FP32>::compute_device_request_batch_decode_equivalent] Invalid grouped request geometry"
                          << " requests=" << request_count
                          << " query_rows=" << query_rows
                          << " max_kv_len=" << max_kv_len
                          << " n_heads=" << n_heads
                          << " n_kv_heads=" << n_kv_heads
                          << " head_dim=" << head_dim
                          << " causal=" << causal);
                return false;
            }
            if (!stream_ || !workspace_)
            {
                LOG_ERROR("[ROCmFlashAttentionKernelT<FP32>::compute_device_request_batch_decode_equivalent] requires an explicit HIP stream and bound workspace");
                return false;
            }
            if (Q->native_type() != TensorType::FP32 ||
                output->native_type() != TensorType::FP32 ||
                K->native_type() != V->native_type())
            {
                LOG_ERROR("[ROCmFlashAttentionKernelT<FP32>::compute_device_request_batch_decode_equivalent] Invalid Q/output or asymmetric K/V types"
                          << " Q=" << Q->dtype_name()
                          << " K=" << K->dtype_name()
                          << " V=" << V->dtype_name()
                          << " O=" << output->dtype_name());
                return false;
            }
            if (K->native_type() != TensorType::FP32 &&
                K->native_type() != TensorType::BF16 &&
                K->native_type() != TensorType::FP16 &&
                K->native_type() != TensorType::Q8_1)
            {
                LOG_ERROR("[ROCmFlashAttentionKernelT<FP32>::compute_device_request_batch_decode_equivalent] Unsupported fixed-stride K/V type "
                          << K->dtype_name());
                return false;
            }

            const size_t required_kv_rows =
                static_cast<size_t>(request_count) *
                static_cast<size_t>(max_kv_len);
            if (K->rows() < required_kv_rows || V->rows() < required_kv_rows ||
                Q->rows() < static_cast<size_t>(total_rows) ||
                output->rows() < static_cast<size_t>(total_rows))
            {
                LOG_ERROR("[ROCmFlashAttentionKernelT<FP32>::compute_device_request_batch_decode_equivalent] Fixed-stride tensor geometry is undersized"
                          << " required_kv_rows=" << required_kv_rows
                          << " K_rows=" << K->rows()
                          << " V_rows=" << V->rows()
                          << " total_rows=" << total_rows
                          << " Q_rows=" << Q->rows()
                          << " O_rows=" << output->rows());
                return false;
            }

            const float *q_ptr = static_cast<const float *>(Q->gpu_data_ptr());
            const void *k_ptr = K->gpu_data_ptr();
            const void *v_ptr = V->gpu_data_ptr();
            float *output_ptr = static_cast<float *>(output->gpu_data_ptr());
            if (!q_ptr || !k_ptr || !v_ptr || !output_ptr)
            {
                LOG_ERROR("[ROCmFlashAttentionKernelT<FP32>::compute_device_request_batch_decode_equivalent] Missing GPU data pointer");
                return false;
            }

            const int dev = device_idx >= 0 ? device_idx : device_idx_;
            if (hipFlashAttn_setDevice(dev) != 0)
            {
                LOG_ERROR("[ROCmFlashAttentionKernelT<FP32>::compute_device_request_batch_decode_equivalent] Failed to set ROCm device "
                          << dev);
                return false;
            }

            const int max_num_splits =
                selectFlashDecodeSplitEnvelope(
                    max_kv_len,
                    n_heads,
                    dev);
            allocateWorkspace(n_heads, head_dim, max_num_splits);
            if (!partial_output_buf_ || !partial_m_buf_ || !partial_l_buf_)
            {
                LOG_ERROR("[ROCmFlashAttentionKernelT<FP32>::compute_device_request_batch_decode_equivalent] Workspace binding failed");
                return false;
            }

            const size_t partial_rows =
                static_cast<size_t>(total_rows) *
                static_cast<size_t>(n_heads) *
                static_cast<size_t>(max_num_splits);
            const size_t required_partial_output =
                partial_rows * static_cast<size_t>(head_dim) * sizeof(float);
            const size_t required_partial_meta = partial_rows * sizeof(float);
            if (workspace_->getBufferSize(AttentionWorkspaceBuffers::PARTIAL_OUTPUT) <
                    required_partial_output ||
                workspace_->getBufferSize(AttentionWorkspaceBuffers::PARTIAL_M) <
                    required_partial_meta ||
                workspace_->getBufferSize(AttentionWorkspaceBuffers::PARTIAL_L) <
                    required_partial_meta)
            {
                LOG_ERROR("[ROCmFlashAttentionKernelT<FP32>::compute_device_request_batch_decode_equivalent] Grouped request workspace is too small"
                          << " required_output=" << required_partial_output
                          << " required_meta=" << required_partial_meta
                          << " rows=" << total_rows);
                return false;
            }

            void *device_params_buffer =
                workspace_->getBuffer(AttentionWorkspaceBuffers::DEVICE_PARAMS);
            if (!device_params_buffer ||
                workspace_->getBufferSize(AttentionWorkspaceBuffers::DEVICE_PARAMS) <
                    static_cast<size_t>(total_rows) *
                        sizeof(attention::AttentionDeviceParams))
            {
                LOG_ERROR("[ROCmFlashAttentionKernelT<FP32>::compute_device_request_batch_decode_equivalent] Missing grouped device params workspace");
                return false;
            }
            if (hipFlashAttn_prepare_device_params_from_request_counts(
                    device_params_buffer,
                    post_append_cached_tokens_device,
                    request_count,
                    query_rows,
                    max_kv_len,
                    stream_) != 0)
            {
                LOG_ERROR("[ROCmFlashAttentionKernelT<FP32>::compute_device_request_batch_decode_equivalent] Failed to derive request-row params");
                return false;
            }
            const auto *device_params =
                static_cast<const attention::AttentionDeviceParams *>(
                    device_params_buffer);

            dynamic_attn_kv_len_ = 0;
            dynamic_attn_kv_stride_ = max_kv_len;
            dynamic_attn_position_offset_ = 0;
            dynamic_attn_query_rows_ = total_rows;
            dynamic_attn_param_rows_ = total_rows;
            dynamic_attn_device_valid_ = true;
            dynamic_attn_device_derived_ = true;

            TensorType execution_kv_type = K->native_type();
            const bool convert_to_fp32 =
                debugEnv().rocm.fa_disable_native_kv &&
                execution_kv_type != TensorType::FP32;
            if (convert_to_fp32)
            {
                float *k_tmp = static_cast<float *>(
                    workspace_->getBuffer(AttentionWorkspaceBuffers::K_TMP_FP32));
                float *v_tmp = static_cast<float *>(
                    workspace_->getBuffer(AttentionWorkspaceBuffers::V_TMP_FP32));
                const size_t logical_elements =
                    required_kv_rows *
                    static_cast<size_t>(n_kv_heads) *
                    static_cast<size_t>(head_dim);
                const size_t required_bytes = logical_elements * sizeof(float);
                if (!k_tmp || !v_tmp ||
                    workspace_->getBufferSize(AttentionWorkspaceBuffers::K_TMP_FP32) <
                        required_bytes ||
                    workspace_->getBufferSize(AttentionWorkspaceBuffers::V_TMP_FP32) <
                        required_bytes ||
                    !hip_convert_tensor_to_fp32(
                        k_ptr,
                        execution_kv_type,
                        k_tmp,
                        static_cast<int>(logical_elements),
                        static_cast<int>(required_kv_rows),
                        n_kv_heads * head_dim,
                        static_cast<hipStream_t>(stream_)) ||
                    !hip_convert_tensor_to_fp32(
                        v_ptr,
                        execution_kv_type,
                        v_tmp,
                        static_cast<int>(logical_elements),
                        static_cast<int>(required_kv_rows),
                        n_kv_heads * head_dim,
                        static_cast<hipStream_t>(stream_)))
                {
                    LOG_ERROR("[ROCmFlashAttentionKernelT<FP32>::compute_device_request_batch_decode_equivalent] Grouped K/V conversion failed");
                    return false;
                }
                k_ptr = k_tmp;
                v_ptr = v_tmp;
                execution_kv_type = TensorType::FP32;
            }

            int result = -1;
            {
                ROCM_KERNEL_PROFILE_SCOPE_STREAM(
                    ROCmKernelType::FLASH_ATTN_DECODE,
                    static_cast<hipStream_t>(stream_));
                if (execution_kv_type == TensorType::FP16)
                {
                    result = hipFlashAttn_decode_fp16_grouped_request_rows(
                        q_ptr, k_ptr, v_ptr, output_ptr,
                        static_cast<float *>(partial_output_buf_),
                        static_cast<float *>(partial_m_buf_),
                        static_cast<float *>(partial_l_buf_),
                        request_count, query_rows, max_kv_len,
                        n_heads, n_kv_heads, head_dim,
                        max_num_splits, device_params, stream_,
                        head_start, gqa_n_rep);
                }
                else if (execution_kv_type == TensorType::BF16)
                {
                    result = hipFlashAttn_decode_bf16_grouped_request_rows(
                        q_ptr, k_ptr, v_ptr, output_ptr,
                        static_cast<float *>(partial_output_buf_),
                        static_cast<float *>(partial_m_buf_),
                        static_cast<float *>(partial_l_buf_),
                        request_count, query_rows, max_kv_len,
                        n_heads, n_kv_heads, head_dim,
                        max_num_splits, device_params, stream_,
                        head_start, gqa_n_rep);
                }
                else if (execution_kv_type == TensorType::Q8_1)
                {
                    result = hipFlashAttn_decode_q8_1_grouped_request_rows(
                        q_ptr, k_ptr, v_ptr, output_ptr,
                        static_cast<float *>(partial_output_buf_),
                        static_cast<float *>(partial_m_buf_),
                        static_cast<float *>(partial_l_buf_),
                        request_count, query_rows, max_kv_len,
                        n_heads, n_kv_heads, head_dim,
                        max_num_splits, device_params, stream_,
                        head_start, gqa_n_rep);
                }
                else
                {
                    result = hipFlashAttn_decode_fp32_grouped_request_rows(
                        q_ptr, k_ptr, v_ptr, output_ptr,
                        static_cast<float *>(partial_output_buf_),
                        static_cast<float *>(partial_m_buf_),
                        static_cast<float *>(partial_l_buf_),
                        request_count, query_rows, max_kv_len,
                        n_heads, n_kv_heads, head_dim,
                        max_num_splits, device_params, stream_,
                        head_start, gqa_n_rep);
                }
            }
            if (result != 0)
            {
                LOG_ERROR("[ROCmFlashAttentionKernelT<FP32>::compute_device_request_batch_decode_equivalent] Grouped request decode failed"
                          << " requests=" << request_count
                          << " query_rows=" << query_rows
                          << " max_kv_len=" << max_kv_len
                          << " max_num_splits=" << max_num_splits
                          << " kv_type=" << K->dtype_name());
                return false;
            }
            return true;
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
            const int num_splits = MAX_FLASH_DECODE_SPLITS;
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

            LOG_TRACE("[ROCmFlashAttentionKernelT<FP32>::getWorkspaceRequirements] "
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
                LOG_TRACE("[ROCmFlashAttentionKernelT<FP32>] Bound workspace manager, entering managed mode");
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

            LOG_DEBUG("[ROCmFlashAttentionKernelT<FP16>] Created for device " << device_idx_
                                                                              << "; awaiting explicit execution stream");
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

            const int num_splits =
                selectFlashDecodeSplitEnvelope(kv_len, n_heads, dev);

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
            int gqa_n_rep,
            const attention::AttentionExecutionPolicy &execution_policy)
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
            (void)execution_policy;
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
            const int num_splits = MAX_FLASH_DECODE_SPLITS;

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
            // Note: MI50 has limited BF16 support - may fall back to FP32
            LOG_DEBUG("[ROCmFlashAttentionKernelT<BF16>] Created for device " << device_idx_
                                                                              << "; awaiting explicit execution stream"
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

            const int num_splits =
                selectFlashDecodeSplitEnvelope(kv_len, n_heads, dev);

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
            int gqa_n_rep,
            const attention::AttentionExecutionPolicy &execution_policy)
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
            (void)execution_policy;
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
            const int num_splits = MAX_FLASH_DECODE_SPLITS;

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
