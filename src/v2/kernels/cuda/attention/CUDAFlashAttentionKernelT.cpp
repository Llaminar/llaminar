/**
 * @file CUDAFlashAttentionKernelT.cpp
 * @brief C++ implementation of CUDA Flash Attention kernel methods
 *
 * Implements ITensorAttention interface by delegating to CUDA kernels
 * defined in CUDAFlashAttentionKernels.cu. Decode uses a capacity-stable KV
 * split envelope and the same ordered scalar/grouped reduction in every
 * determinism mode; determinism does not serialize a whole attention head.
 *
 * @author David Sanftenberg
 */

#include "CUDAFlashAttentionKernelT.h"
#include "CUDAFlashAttentionLaunchPolicy.h"
#include "../../../backends/cuda/CUDAGraphCapture.h"
#include "../../../backends/IWorkerGPUContext.h"
#include "../../../execution/local_execution/device/DeviceWorkspaceManager.h"
#include "../../../execution/local_execution/device/WorkspaceDescriptor.h"
#include "../../../utils/Logger.h"
#include "../../../utils/CUDAKernelProfiler.h"
#include "../../../utils/DebugEnv.h"
#include "../../../utils/PerfStatsCollector.h"
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

    /**
     * @brief Launch a deterministic K/V-context transaction with FP32 K/V.
     */
    int cudaFlashAttn_prefill_fa2_context_parallel(
        const float *Q, const float *K, const float *V, float *O,
        float *O_partial, float *m_partial, float *l_partial,
        int batch_size, int seq_len, int kv_capacity,
        int n_heads, int n_kv_heads, int head_dim,
        bool causal, int window_size, int position_offset,
        const llaminar2::attention::AttentionDeviceParams *device_params,
        const float *mask,
        int context_partition_size, int max_context_partitions,
        int context_partition_slots,
        int device_direct_partition_limit,
        int reducer_dimension_warps,
        void *capture_conditional,
        void *stream, int device_idx, int head_start, int gqa_n_rep);

    /**
     * @brief Launch a deterministic K/V-context transaction with FP16 K/V.
     */
    int cudaFlashAttn_prefill_fa2_fp16kv_context_parallel(
        const float *Q, const void *K_fp16, const void *V_fp16, float *O,
        float *O_partial, float *m_partial, float *l_partial,
        int batch_size, int seq_len, int kv_capacity,
        int n_heads, int n_kv_heads, int head_dim,
        bool causal, int window_size, int position_offset,
        const llaminar2::attention::AttentionDeviceParams *device_params,
        const float *mask,
        int context_partition_size, int max_context_partitions,
        int context_partition_slots,
        int device_direct_partition_limit,
        int reducer_dimension_warps,
        void *capture_conditional,
        void *stream, int device_idx, int head_start, int gqa_n_rep);

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

    /**
     * @brief Launch all compact verifier rows through one FP16-KV decode grid.
     *
     * Unlike an ordinary decode batch, every verifier row reads the same cache
     * allocation but has its own device-resident logical KV length. The kernel
     * preserves the exact split sizing and reduction order of independent M=1
     * decode while amortizing launch overhead across runtime M=2..16.
     */
    int cudaFlashAttn_decode_fp16kv_grouped_verifier_rows(
        const float *Q, const void *K_cache_fp16, const void *V_cache_fp16, float *O,
        float *O_partial, float *m_partial, float *l_partial,
        int verifier_rows, int max_kv_len,
        int n_heads, int n_kv_heads, int head_dim,
        int max_num_splits,
        const llaminar2::attention::AttentionDeviceParams *device_params,
        void *stream,
        int device_idx,
        int head_start,
        int gqa_n_rep);

    /**
     * @brief Launch independent request banks through one row-local FP16 decode.
     */
    int cudaFlashAttn_decode_fp16kv_grouped_request_rows(
        const float *Q, const void *K_cache_fp16, const void *V_cache_fp16, float *O,
        float *O_partial, float *m_partial, float *l_partial,
        int request_count, int query_rows, int max_kv_len,
        int n_heads, int n_kv_heads, int head_dim,
        int max_num_splits,
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

    int cudaFlashAttn_prepare_device_params_from_count(
        void *device_params,
        const int *post_append_cached_tokens,
        int seq_len,
        int query_rows,
        int kv_stride,
        const int *active_query_rows_device,
        const int *ring_head_device,
        int ring_capacity,
        unsigned long long prefill_branch_condition,
        int direct_kv_limit,
        void *stream);

    int cudaFlashAttn_prepare_device_params_from_geometry(
        void *device_params,
        int kv_len,
        int kv_stride,
        int position_offset,
        int query_rows,
        unsigned long long prefill_branch_condition,
        int direct_kv_limit,
        void *stream);

    int cudaFlashAttn_prepare_device_params_from_request_counts(
        void *device_params,
        const int *post_append_cached_tokens,
        int request_count,
        int query_rows,
        int kv_stride,
        void *stream);

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
        constexpr int MAX_NUM_SPLITS =
            attention_workspace::kMaximumDecodeSplits;

        constexpr int MAX_SMALL_DECODE_ROWS =
            attention::kMaxGroupedVerifierAttentionRows;

        // Minimum KV positions per split to avoid excessive overhead
        constexpr int MIN_KV_PER_SPLIT = 16;

        /**
         * @brief Resolve one CUDA prefill graph plan from immutable geometry.
         *
         * CUDA device properties are setup-time inputs. The returned mode is
         * embedded in graph capture and never re-evaluated from a live sequence
         * length. Returning an invalid plan is a fatal configuration result at
         * the caller; this helper never substitutes query execution for an
         * invalid context request.
         */
        static fa2_policy::FA2PrefillParallelPlan resolvePrefillPlanForDevice(
            int batch_size,
            int query_rows,
            int local_query_heads,
            int head_dim,
            int kv_capacity,
            int device_idx,
            const attention::AttentionExecutionPolicy &execution_policy)
        {
            cudaDeviceProp properties{};
            const cudaError_t status =
                cudaGetDeviceProperties(&properties, device_idx);
            if (status != cudaSuccess)
            {
                LOG_ERROR("[CUDAFlashAttentionKernelT] Failed to query CUDA prefill capture geometry for device "
                          << device_idx << ": " << cudaGetErrorString(status));
                return {};
            }

            return fa2_policy::selectFA2PrefillParallelPlan({
                .batch_size = batch_size,
                .query_rows = query_rows,
                .local_query_heads = local_query_heads,
                .head_dim = head_dim,
                .kv_capacity = kv_capacity,
                .sm_count = properties.multiProcessorCount,
                .requested_axis =
                    execution_policy.prefill_parallel_axis,
            });
        }

        /**
         * @brief Publish one successfully submitted FA2 physical-plan choice.
         *
         * Captured production replay does not return through this host method,
         * so this setup-owned inventory has no replay-path synchronization or
         * contention cost. The `gpu_graph_inventory` domain survives benchmark
         * warmup reset and lets E2E prove that the intended physical mode was
         * actually launched instead of merely being declared by model policy.
         */
        static void recordFA2ParallelPlanSelection(
            const fa2_policy::FA2PrefillParallelPlan &plan,
            const attention::AttentionExecutionPolicy &execution_policy,
            int batch_size,
            int query_rows,
            int local_query_heads,
            int head_dim,
            int kv_capacity,
            int device_idx,
            const char *kv_storage)
        {
            if (!PerfStatsCollector::isDomainEnabled("gpu_graph_inventory"))
                return;

            PerfStatsCollector::addCounter(
                "gpu_graph_inventory",
                "cuda_fa2_parallel_plan_selections",
                1.0,
                "capture_setup",
                "cuda:" + std::to_string(device_idx),
                {
                    {"requested_axis",
                     attention::attentionPrefillParallelAxisName(
                         execution_policy.prefill_parallel_axis)},
                    {"selected_axis",
                     fa2_policy::fa2PrefillPhysicalModeName(plan.mode)},
                    {"batch_size", std::to_string(batch_size)},
                    {"query_rows", std::to_string(query_rows)},
                    {"local_query_heads", std::to_string(local_query_heads)},
                    {"head_dim", std::to_string(head_dim)},
                    {"kv_capacity", std::to_string(kv_capacity)},
                    {"query_warp_groups",
                     std::to_string(plan.query_warp_groups)},
                    {"query_grid_blocks",
                     std::to_string(plan.query_grid_blocks)},
                    {"context_partitions",
                     std::to_string(plan.max_context_partitions)},
                    {"context_partition_slots",
                     std::to_string(plan.context_partition_slots)},
                    {"device_direct_partition_limit",
                     std::to_string(plan.device_direct_partition_limit)},
                    {"reducer_dimension_warps",
                     std::to_string(plan.reducer_dimension_warps)},
                    {"kv_storage", kv_storage ? kv_storage : "unknown"},
                });
        }

        /**
         * @brief Select the fixed split envelope for one captured decode graph.
         *
         * The physical grid must not depend on the live KV length: doing so
         * changes graph topology as decode advances and forces recapture.  The
         * caller therefore supplies the stable cache capacity.  Device kernels
         * derive the active split prefix from the live device-owned KV count.
         * Scalar and grouped rows use this same envelope and ordered reducer;
         * the deterministic CLI policy must not replace it with one split.
         *
         * The grid is `(n_heads, split_envelope, batch)`. With
         * `__launch_bounds__(256, 4)` and 61 registers per thread, the selected
         * envelope fills the resident block capacity without an avoidable tail
         * wave. The 32-split architectural bound matches the preallocated graph
         * workspace.
         *
         * For Qwen2.5-7B (28 heads) on RTX 3090 (82 SMs): 328/28 = 11 splits = 308 blocks.
         */
        static int computeDecodeSplitEnvelopeForDevice(
            int kv_capacity,
            int n_heads,
            int device_idx)
        {
            if (kv_capacity <= 1 || n_heads <= 0)
                return 1;

            // Get SM count (cached per device via static array)
            static int sm_count_cache[8] = {0};
            int num_sms = sm_count_cache[device_idx & 7];
            if (num_sms == 0)
            {
                const cudaError_t status = cudaDeviceGetAttribute(
                    &num_sms,
                    cudaDevAttrMultiProcessorCount,
                    device_idx);
                if (status != cudaSuccess || num_sms <= 0)
                {
                    throw std::runtime_error(
                        "[CUDAFlashAttentionKernelT] Cannot construct a stable "
                        "decode launch envelope because CUDA did not publish a "
                        "positive SM count for device " +
                        std::to_string(device_idx) +
                        " (status=" +
                        std::to_string(static_cast<int>(status)) + ")");
                }
                sm_count_cache[device_idx & 7] = num_sms;
            }

            // Max concurrent blocks from register pressure (61 regs → 4 blocks/SM)
            constexpr int MAX_BLOCKS_PER_SM = 4;
            int max_concurrent = MAX_BLOCKS_PER_SM * num_sms;

            // Floor division: stay at or below capacity to avoid tail-wave inefficiency
            int desired_splits = max_concurrent / n_heads;

            // Clamp: each split needs enough work (MIN_KV_PER_SPLIT positions)
            int max_splits_by_kv = kv_capacity / MIN_KV_PER_SPLIT;
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

            LOG_DEBUG("[CUDAFlashAttentionKernelT<FP32>] Created for device " << device_idx_
                                                                              << "; awaiting explicit execution stream");
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
              max_splits_(other.max_splits_),
              workspace_(other.workspace_),
              device_ctx_(other.device_ctx_),
              dynamic_attn_kv_len_(other.dynamic_attn_kv_len_),
              dynamic_attn_kv_stride_(other.dynamic_attn_kv_stride_),
              dynamic_attn_position_offset_(other.dynamic_attn_position_offset_),
              dynamic_attn_query_rows_(other.dynamic_attn_query_rows_),
              dynamic_attn_param_rows_(other.dynamic_attn_param_rows_),
              dynamic_attn_device_valid_(other.dynamic_attn_device_valid_),
              dynamic_attn_device_derived_(other.dynamic_attn_device_derived_),
              pending_prefill_conditional_(
                  std::move(other.pending_prefill_conditional_))
        {
            other.stream_ = nullptr;
            other.partial_output_buf_ = nullptr;
            other.partial_m_buf_ = nullptr;
            other.partial_l_buf_ = nullptr;
            other.workspace_size_ = 0;
            other.max_splits_ = 0;
            other.workspace_ = nullptr;
            other.device_ctx_ = nullptr;
            other.dynamic_attn_device_valid_ = false;
            other.dynamic_attn_device_derived_ = false;
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
                workspace_ = other.workspace_;
                device_ctx_ = other.device_ctx_;
                dynamic_attn_kv_len_ = other.dynamic_attn_kv_len_;
                dynamic_attn_kv_stride_ = other.dynamic_attn_kv_stride_;
                dynamic_attn_position_offset_ = other.dynamic_attn_position_offset_;
                dynamic_attn_query_rows_ = other.dynamic_attn_query_rows_;
                dynamic_attn_param_rows_ = other.dynamic_attn_param_rows_;
                dynamic_attn_device_valid_ = other.dynamic_attn_device_valid_;
                dynamic_attn_device_derived_ = other.dynamic_attn_device_derived_;
                pending_prefill_conditional_ =
                    std::move(other.pending_prefill_conditional_);

                other.stream_ = nullptr;
                other.partial_output_buf_ = nullptr;
                other.partial_m_buf_ = nullptr;
                other.partial_l_buf_ = nullptr;
                other.workspace_size_ = 0;
                other.max_splits_ = 0;
                other.workspace_ = nullptr;
                other.device_ctx_ = nullptr;
                other.dynamic_attn_device_valid_ = false;
                other.dynamic_attn_device_derived_ = false;
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
            /*
             * A decode call without an explicit historical position consumes
             * the newest query rows in the supplied KV span. Treating the
             * default as absolute position zero silently truncated causal
             * attention to the first key for every long-context raw-pointer
             * caller. The negative sentinel keeps explicit positions intact
             * while making the ordinary append-at-tail contract total for
             * both M=1 decode and grouped verifier rows.
             */
            const int resolved_position_offset =
                position_offset >= 0 ? position_offset : kv_len - seq_len;
            return apply_typed(Q, K, V, output,
                               1, seq_len, kv_len,
                               n_heads, n_kv_heads, head_dim,
                               causal, -1, resolved_position_offset,
                               device_idx_, nullptr, nullptr);
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
                const int launch_kv_capacity =
                    dynamic_attn_kv_stride_ > 0 ? dynamic_attn_kv_stride_ : kv_len;
                const int num_splits = computeDecodeSplitEnvelopeForDevice(
                    launch_kv_capacity,
                    n_heads,
                    device_idx);

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
            int gqa_n_rep,
            const attention::AttentionExecutionPolicy &execution_policy,
            const attention::AttentionKVLogicalView &kv_logical_view)
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
            if (!kv_logical_view.isContiguous())
            {
                LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>::compute_tensor] "
                          "circular CPU-style KV descriptors are invalid for gathered CUDA views");
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

            const bool row_local_decode_params_ready =
                dynamic_attn_device_valid_ &&
                dynamic_attn_query_rows_ == seq_len &&
                dynamic_attn_param_rows_ >= seq_len;
            const bool use_small_fp16kv_decode =
                K->native_type() == TensorType::FP16 &&
                V->native_type() == TensorType::FP16 &&
                batch_size == 1 &&
                causal &&
                seq_len > 1 &&
                seq_len <= MAX_SMALL_DECODE_ROWS &&
                kv_len > seq_len &&
                row_local_decode_params_ready;
            const int expected_query_rows =
                use_small_fp16kv_decode ? seq_len : 1;

            // Wire the device-owned parameter block for graph-capture replay.
            // Resident cache execution derives it from the live device count;
            // explicit-geometry callers populate the same block with a tiny
            // stream-ordered kernel. Neither path has a host parameter mirror.
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

                if (dynamic_attn_device_derived_)
                {
                    if (!dynamic_attn_device_valid_ ||
                        dynamic_attn_param_rows_ < expected_query_rows ||
                        dynamic_attn_query_rows_ != expected_query_rows)
                    {
                        LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>::compute_tensor] "
                                  "Device-derived attention params were not ready"
                                  << " rows=" << expected_query_rows
                                  << " prepared_rows=" << dynamic_attn_query_rows_
                                  << " param_rows=" << dynamic_attn_param_rows_
                                  << " device_valid=" << dynamic_attn_device_valid_);
                        return false;
                    }
                }
                else if (cap_status == cudaStreamCaptureStatusActive)
                {
                    if (!dynamic_attn_device_valid_ ||
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
                                  << ") device_valid=" << dynamic_attn_device_valid_
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
                    LOG_TRACE("[CUDAFlashAttentionKernelT<FP32>::compute_tensor] FP16 KV direct path (no conversion)");
                }
                else if (K->native_type() == TensorType::Q8_1 && seq_len == 1)
                {
                    // FAST PATH: Q8_1 KV decode → fused inline dequant kernel
                    // Reads Q8_1 blocks directly in the attention inner loop,
                    // eliminating dequant kernel + FP32 workspace buffer.
                    use_q8kv_direct = true;
                    K_q8_ptr = K->gpu_data_ptr();
                    V_q8_ptr = V->gpu_data_ptr();
                    LOG_TRACE("[CUDAFlashAttentionKernelT<FP32>::compute_tensor] Q8_1 KV fused decode path (no workspace)");
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

            LOG_TRACE("[CUDAFlashAttentionKernelT<FP32>::compute_tensor] batch=" << batch_size
                                                                                 << " seq_len=" << seq_len << " kv_len=" << kv_len
                                                                                 << " n_heads=" << n_heads << " n_kv_heads=" << n_kv_heads
                                                                                 << " head_dim=" << head_dim << " causal=" << causal);

            int dev = (device_idx >= 0) ? device_idx : device_idx_;
            const int launch_kv_capacity =
                dynamic_attn_kv_stride_ > 0
                    ? dynamic_attn_kv_stride_
                    : kv_len;
            const fa2_policy::FA2PrefillParallelPlan prefill_plan =
                resolvePrefillPlanForDevice(
                    batch_size,
                    seq_len,
                    n_heads,
                    head_dim,
                    launch_kv_capacity,
                    dev,
                    execution_policy);
            if (!prefill_plan.valid)
            {
                LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>::compute_tensor] Invalid declared CUDA prefill capture plan"
                          << " requested="
                          << attention::attentionPrefillParallelAxisName(
                                 execution_policy.prefill_parallel_axis)
                          << " batch=" << batch_size
                          << " rows=" << seq_len
                          << " heads=" << n_heads
                          << " head_dim=" << head_dim
                          << " kv_capacity=" << launch_kv_capacity
                          << " device=" << dev);
                return false;
            }
            std::unique_ptr<CUDAActiveCaptureConditional>
                active_prefill_conditional;
            if (pending_prefill_conditional_ &&
                !prefill_plan.usesDeviceAdaptiveParallelism())
            {
                LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>::compute_tensor] Device-param producer opened an adaptive transaction for a non-adaptive FA2 plan");
                pending_prefill_conditional_.reset();
                return false;
            }
            if (pending_prefill_conditional_)
            {
                /*
                 * Move the single-use construction token into this call. Every
                 * return path now destroys it, including workspace, conversion,
                 * or launch failures before branch commitment. A failed capture
                 * cannot leave stale builder state attached to the reusable
                 * kernel object and poison an unrelated later request.
                 */
                active_prefill_conditional =
                    std::move(pending_prefill_conditional_);
            }

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
                    const int launch_kv_capacity =
                        dynamic_attn_kv_stride_ > 0 ? dynamic_attn_kv_stride_ : kv_len;
                    const int num_splits = computeDecodeSplitEnvelopeForDevice(
                        launch_kv_capacity,
                        n_heads,
                        dev);

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
                    LOG_TRACE("[CUDAFlashAttentionKernelT<FP32>] Small-M FP16KV decode path"
                              << " rows=" << seq_len
                              << " kv_len=" << kv_len
                              << " n_heads=" << n_heads
                              << " n_kv_heads=" << n_kv_heads
                              << " head_dim=" << head_dim
                              << " device_params=" << (d_attn_params != nullptr));
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

                    /*
                     * Launch one phase-1 grid and one reduction grid for the
                     * complete verifier span. Each grid-z row reads its own
                     * AttentionDeviceParams entry but shares the same K/V cache.
                     * The launcher derives that row's split count with the same
                     * policy used by serial decode, so block partitioning and
                     * FP32 merge order remain byte-identical to M=1.
                     */
                    const int launch_kv_capacity =
                        dynamic_attn_kv_stride_ > 0 ? dynamic_attn_kv_stride_ : kv_len;
                    const int max_num_splits =
                        computeDecodeSplitEnvelopeForDevice(
                            launch_kv_capacity,
                            n_heads,
                            dev);
                    int result;
                    {
                        CUDA_KERNEL_PROFILE_SCOPE_STREAM(
                            CUDAKernelType::FLASH_ATTN_DECODE,
                            stream_);
                        result = cudaFlashAttn_decode_fp16kv_grouped_verifier_rows(
                            Q_ptr,
                            K_fp16_ptr,
                            V_fp16_ptr,
                            output_ptr,
                            O_partial,
                            m_partial,
                            l_partial,
                            seq_len,
                            kv_len,
                            n_heads,
                            n_kv_heads,
                            head_dim,
                            max_num_splits,
                            d_attn_params,
                            stream_,
                            dev,
                            head_start,
                            gqa_n_rep);
                    }
                    if (result != 0)
                    {
                        LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>] Grouped small-M FP16KV decode failed"
                                  << " rows=" << seq_len
                                  << " max_kv_len=" << kv_len
                                  << " max_num_splits=" << max_num_splits);
                        return false;
                    }
                    return true;
                }

                // PREFILL: one capture-stable query or context transaction.
                LOG_TRACE("[CUDAFlashAttentionKernelT<FP32>] FA2 FP16KV prefill path"
                          << " parallel_axis="
                          << (prefill_plan.usesContextParallelism()
                                  ? "key_value_context"
                                  : "query_sequence")
                          << " seq_len=" << seq_len
                          << " kv_len=" << kv_len
                          << " kv_capacity=" << launch_kv_capacity
                          << " n_heads=" << n_heads
                          << " n_kv_heads=" << n_kv_heads
                          << " head_dim=" << head_dim
                          << " causal=" << causal
                          << " device_params=" << (d_attn_params != nullptr));

                int result = -1;
                {
                    CUDA_KERNEL_PROFILE_SCOPE_STREAM(
                        CUDAKernelType::FLASH_ATTN_PREFILL,
                        stream_);
                    if (prefill_plan.usesContextParallelism())
                    {
                        if (!stream_ || !workspace_ || !d_attn_params)
                        {
                            LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>] Context-parallel FP16KV prefill requires an exact stream, bound workspace, and device-owned attention params");
                            return false;
                        }

                        float *O_partial = static_cast<float *>(
                            workspace_->getBuffer(
                                AttentionWorkspaceBuffers::PARTIAL_OUTPUT));
                        float *m_partial = static_cast<float *>(
                            workspace_->getBuffer(
                                AttentionWorkspaceBuffers::PARTIAL_M));
                        float *l_partial = static_cast<float *>(
                            workspace_->getBuffer(
                                AttentionWorkspaceBuffers::PARTIAL_L));
                        const bool capacity_valid =
                            O_partial && m_partial && l_partial &&
                            workspace_->getBufferSize(
                                AttentionWorkspaceBuffers::PARTIAL_OUTPUT) >=
                                prefill_plan.partial_output_bytes &&
                            workspace_->getBufferSize(
                                AttentionWorkspaceBuffers::PARTIAL_M) >=
                                prefill_plan.partial_m_bytes &&
                            workspace_->getBufferSize(
                                AttentionWorkspaceBuffers::PARTIAL_L) >=
                                prefill_plan.partial_l_bytes;
                        if (!capacity_valid)
                        {
                            LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>] Context-parallel FP16KV workspace does not match the captured plan"
                                      << " required_output="
                                      << prefill_plan.partial_output_bytes
                                      << " required_scalar="
                                      << prefill_plan.partial_m_bytes);
                            return false;
                        }

                        result =
                            cudaFlashAttn_prefill_fa2_fp16kv_context_parallel(
                                Q_ptr,
                                K_fp16_ptr,
                                V_fp16_ptr,
                                output_ptr,
                                O_partial,
                                m_partial,
                                l_partial,
                                batch_size,
                                seq_len,
                                launch_kv_capacity,
                                n_heads,
                                n_kv_heads,
                                head_dim,
                                causal,
                                window_size,
                                /*position_offset=*/0,
                                d_attn_params,
                                mask_ptr,
                                fa2_policy::kFA2CanonicalContextPartitionKeys,
                                prefill_plan.max_context_partitions,
                                prefill_plan.context_partition_slots,
                                prefill_plan.device_direct_partition_limit,
                                prefill_plan.reducer_dimension_warps,
                                active_prefill_conditional.get(),
                                stream_,
                                dev,
                                head_start,
                                gqa_n_rep);
                    }
                    else
                    {
                        result = cudaFlashAttn_prefill_fa2_fp16kv(
                            Q_ptr, K_fp16_ptr, V_fp16_ptr, output_ptr,
                            batch_size, seq_len, kv_len,
                            n_heads, n_kv_heads, head_dim,
                            causal, window_size, 0,
                            d_attn_params, mask_ptr,
                            stream_, dev,
                            head_start, gqa_n_rep);
                    }
                }
                if (result != 0)
                {
                    LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>] FA2 FP16KV prefill transaction failed"
                              << " parallel_axis="
                              << (prefill_plan.usesContextParallelism()
                                      ? "key_value_context"
                                      : "query_sequence"));
                    return false;
                }
                recordFA2ParallelPlanSelection(
                    prefill_plan,
                    execution_policy,
                    batch_size,
                    seq_len,
                    n_heads,
                    head_dim,
                    launch_kv_capacity,
                    dev,
                    "fp16");
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
                const int launch_kv_capacity =
                    dynamic_attn_kv_stride_ > 0 ? dynamic_attn_kv_stride_ : kv_len;
                const int num_splits = computeDecodeSplitEnvelopeForDevice(
                    launch_kv_capacity,
                    n_heads,
                    dev);

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
            if (prefill_plan.usesContextParallelism())
            {
                if (!stream_ || !workspace_ || !d_attn_params)
                {
                    LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>] Context-parallel FP32 prefill requires an exact stream, bound workspace, and device-owned attention params");
                    return false;
                }

                float *O_partial = static_cast<float *>(
                    workspace_->getBuffer(
                        AttentionWorkspaceBuffers::PARTIAL_OUTPUT));
                float *m_partial = static_cast<float *>(
                    workspace_->getBuffer(
                        AttentionWorkspaceBuffers::PARTIAL_M));
                float *l_partial = static_cast<float *>(
                    workspace_->getBuffer(
                        AttentionWorkspaceBuffers::PARTIAL_L));
                const bool capacity_valid =
                    O_partial && m_partial && l_partial &&
                    workspace_->getBufferSize(
                        AttentionWorkspaceBuffers::PARTIAL_OUTPUT) >=
                        prefill_plan.partial_output_bytes &&
                    workspace_->getBufferSize(
                        AttentionWorkspaceBuffers::PARTIAL_M) >=
                        prefill_plan.partial_m_bytes &&
                    workspace_->getBufferSize(
                        AttentionWorkspaceBuffers::PARTIAL_L) >=
                        prefill_plan.partial_l_bytes;
                if (!capacity_valid)
                {
                    LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>] Context-parallel FP32 workspace does not match the captured plan"
                              << " required_output="
                              << prefill_plan.partial_output_bytes
                              << " required_scalar="
                              << prefill_plan.partial_m_bytes);
                    return false;
                }

                int result = -1;
                {
                    CUDA_KERNEL_PROFILE_SCOPE_STREAM(
                        CUDAKernelType::FLASH_ATTN_PREFILL,
                        stream_);
                    result = cudaFlashAttn_prefill_fa2_context_parallel(
                        Q_ptr,
                        K_ptr,
                        V_ptr,
                        output_ptr,
                        O_partial,
                        m_partial,
                        l_partial,
                        batch_size,
                        seq_len,
                        launch_kv_capacity,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        causal,
                        window_size,
                        /*position_offset=*/0,
                        d_attn_params,
                        mask_ptr,
                        fa2_policy::kFA2CanonicalContextPartitionKeys,
                        prefill_plan.max_context_partitions,
                        prefill_plan.context_partition_slots,
                        prefill_plan.device_direct_partition_limit,
                        prefill_plan.reducer_dimension_warps,
                        active_prefill_conditional.get(),
                        stream_,
                        dev,
                        head_start,
                        gqa_n_rep);
                }
                if (result != 0)
                {
                    LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>] FA2 FP32 context transaction failed");
                    return false;
                }
                recordFA2ParallelPlanSelection(
                    prefill_plan,
                    execution_policy,
                    batch_size,
                    seq_len,
                    n_heads,
                    head_dim,
                    launch_kv_capacity,
                    dev,
                    "fp32");
                return true;
            }

            bool direct_success = false;
            if (kv_len != seq_len)
            {
                // Decode path (different Q and KV lengths)
                direct_success = apply_typed(
                    Q_ptr, K_ptr, V_ptr, output_ptr,
                    batch_size, seq_len, kv_len,
                    n_heads, n_kv_heads, head_dim,
                    causal, window_size, 0, dev,
                    d_attn_params, mask_ptr,
                    head_start, gqa_n_rep);
            }
            else
            {
                // Prefill path
                direct_success = apply_typed(
                    Q_ptr, K_ptr, V_ptr, output_ptr,
                    batch_size, seq_len, seq_len,
                    n_heads, n_kv_heads, head_dim,
                    causal, window_size, 0, dev,
                    d_attn_params, mask_ptr,
                    head_start, gqa_n_rep);
            }
            if (direct_success && seq_len > 1)
            {
                recordFA2ParallelPlanSelection(
                    prefill_plan,
                    execution_policy,
                    batch_size,
                    seq_len,
                    n_heads,
                    head_dim,
                    launch_kv_capacity,
                    dev,
                    "fp32");
            }
            return direct_success;
        }

        bool CUDAFlashAttentionKernelT<ActivationPrecision::FP32>::compute_verifier_rows_decode_equivalent(
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
            int gqa_n_rep,
            const attention::AttentionKVLogicalView &kv_logical_view,
            const attention::AttentionExecutionPolicy &execution_policy)
        {
            (void)execution_policy;
            if (!Q || !K || !V || !output)
            {
                LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>::compute_verifier_rows_decode_equivalent] Null tensor");
                return false;
            }
            if (!kv_logical_view.isContiguous())
            {
                LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>::compute_verifier_rows_decode_equivalent] "
                          "circular CPU-style KV descriptors are invalid for gathered CUDA views");
                return false;
            }
            if (verifier_rows < 2 || verifier_rows > MAX_SMALL_DECODE_ROWS ||
                kv_len <= verifier_rows || !causal)
            {
                LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>::compute_verifier_rows_decode_equivalent] Invalid verifier span"
                          << " rows=" << verifier_rows
                          << " kv_len=" << kv_len
                          << " causal=" << causal);
                return false;
            }
            if (!stream_ || !workspace_)
            {
                LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>::compute_verifier_rows_decode_equivalent] "
                          "requires explicit CUDA stream and bound workspace");
                return false;
            }
            if (Q->native_type() != TensorType::FP32 || output->native_type() != TensorType::FP32)
            {
                LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>::compute_verifier_rows_decode_equivalent] "
                          "requires FP32 Q/output, got Q=" << Q->dtype_name()
                                                           << " O=" << output->dtype_name());
                return false;
            }
            if (K->native_type() != TensorType::FP16 || V->native_type() != TensorType::FP16)
            {
                LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>::compute_verifier_rows_decode_equivalent] "
                          "currently proven only for FP16 KV, got K=" << K->dtype_name()
                                                                      << " V=" << V->dtype_name());
                return false;
            }

            /*
             * The M-row verifier path decides whether small-M decode is legal
             * before compute_tensor() can lazily upload params, so prepare the
             * row-local param block at this named contract boundary.
             */
            const int position_offset = kv_len - verifier_rows;
            if (!dynamic_attn_device_derived_ &&
                !prepareDynamicAttnParams(kv_len, position_offset, verifier_rows, stream_))
            {
                LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>::compute_verifier_rows_decode_equivalent] "
                          "failed preparing dynamic params");
                return false;
            }

            return compute_tensor(Q, K, V, output,
                                  /*batch_size=*/1,
                                  verifier_rows,
                                  kv_len,
                                  n_heads,
                                  n_kv_heads,
                                  head_dim,
                                  causal,
                                  window_size,
                                  nullptr,
                                  nullptr,
                                  mpi_ctx,
                                  device_idx,
                                  head_start,
                                  n_heads,
                                  n_kv_heads,
                                  gqa_n_rep,
                                  /*execution_policy=*/{},
                                  kv_logical_view);
        }

        bool CUDAFlashAttentionKernelT<ActivationPrecision::FP32>::compute_device_request_batch_decode_equivalent(
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
                LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>::compute_device_request_batch_decode_equivalent] Null request-batch input");
                return false;
            }
            if (request_count < 2 || query_rows <= 0 ||
                total_rows > MAX_SMALL_DECODE_ROWS || max_kv_len <= 0 ||
                n_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0 || !causal)
            {
                LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>::compute_device_request_batch_decode_equivalent] Invalid grouped request geometry"
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
                LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>::compute_device_request_batch_decode_equivalent] "
                          "requires an explicit CUDA stream and bound workspace");
                return false;
            }
            if (Q->native_type() != TensorType::FP32 ||
                output->native_type() != TensorType::FP32 ||
                K->native_type() != TensorType::FP16 ||
                V->native_type() != TensorType::FP16)
            {
                LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>::compute_device_request_batch_decode_equivalent] "
                          "requires FP32 Q/output and FP16 fixed-stride K/V"
                          << " Q=" << Q->dtype_name()
                          << " K=" << K->dtype_name()
                          << " V=" << V->dtype_name()
                          << " O=" << output->dtype_name());
                return false;
            }

            const size_t required_kv_rows =
                static_cast<size_t>(request_count) *
                static_cast<size_t>(max_kv_len);
            if (K->rows() < required_kv_rows || V->rows() < required_kv_rows ||
                Q->rows() < static_cast<size_t>(total_rows) ||
                output->rows() < static_cast<size_t>(total_rows))
            {
                LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>::compute_device_request_batch_decode_equivalent] "
                          "fixed-stride tensor geometry is undersized"
                          << " required_kv_rows=" << required_kv_rows
                          << " K_rows=" << K->rows()
                          << " V_rows=" << V->rows()
                          << " required_query_rows=" << total_rows
                          << " Q_rows=" << Q->rows()
                          << " O_rows=" << output->rows());
                return false;
            }

            const float *Q_ptr = static_cast<const float *>(Q->gpu_data_ptr());
            const void *K_ptr = K->gpu_data_ptr();
            const void *V_ptr = V->gpu_data_ptr();
            float *output_ptr = static_cast<float *>(output->gpu_data_ptr());
            if (!Q_ptr || !K_ptr || !V_ptr || !output_ptr)
            {
                LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>::compute_device_request_batch_decode_equivalent] Missing GPU data pointer");
                return false;
            }

            const int dev = device_idx >= 0 ? device_idx : device_idx_;
            if (cudaFlashAttn_setDevice(dev) != 0)
            {
                LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>::compute_device_request_batch_decode_equivalent] Failed to set CUDA device "
                          << dev);
                return false;
            }

            const int max_num_splits =
                computeDecodeSplitEnvelopeForDevice(max_kv_len, n_heads, dev);
            if (!allocateWorkspace(n_heads, head_dim, max_num_splits))
            {
                LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>::compute_device_request_batch_decode_equivalent] Workspace binding failed");
                return false;
            }

            /*
             * allocateWorkspace() validates one scalar row because that is the
             * generic kernel contract.  The grouped request path owns several
             * z-planes, so validate the complete fixed workspace explicitly
             * before launching.  The graph arena is still allocated once by
             * AttentionComputeStage; this check performs no allocation.
             */
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
                LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>::compute_device_request_batch_decode_equivalent] "
                          "Grouped request workspace is too small"
                          << " required_output=" << required_partial_output
                          << " required_meta=" << required_partial_meta
                          << " rows=" << total_rows);
                return false;
            }

            void *device_params =
                workspace_->getBuffer(AttentionWorkspaceBuffers::DEVICE_PARAMS);
            if (!device_params ||
                workspace_->getBufferSize(AttentionWorkspaceBuffers::DEVICE_PARAMS) <
                    static_cast<size_t>(total_rows) *
                        sizeof(attention::AttentionDeviceParams))
            {
                LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>::compute_device_request_batch_decode_equivalent] Missing grouped device params workspace");
                return false;
            }

            if (cudaFlashAttn_prepare_device_params_from_request_counts(
                    device_params,
                    post_append_cached_tokens_device,
                    request_count,
                    query_rows,
                    max_kv_len,
                    stream_) != 0)
            {
                LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>::compute_device_request_batch_decode_equivalent] Failed to derive request-row params");
                return false;
            }

            dynamic_attn_kv_len_ = 0;
            dynamic_attn_kv_stride_ = max_kv_len;
            dynamic_attn_position_offset_ = 0;
            dynamic_attn_query_rows_ = total_rows;
            dynamic_attn_param_rows_ = total_rows;
            dynamic_attn_device_valid_ = true;
            dynamic_attn_device_derived_ = true;

            int result = -1;
            {
                CUDA_KERNEL_PROFILE_SCOPE_STREAM(
                    CUDAKernelType::FLASH_ATTN_DECODE,
                    stream_);
                result = cudaFlashAttn_decode_fp16kv_grouped_request_rows(
                    Q_ptr,
                    K_ptr,
                    V_ptr,
                    output_ptr,
                    static_cast<float *>(partial_output_buf_),
                    static_cast<float *>(partial_m_buf_),
                    static_cast<float *>(partial_l_buf_),
                    request_count,
                    query_rows,
                    max_kv_len,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    max_num_splits,
                    static_cast<const attention::AttentionDeviceParams *>(
                        device_params),
                    stream_,
                    dev,
                    head_start,
                    gqa_n_rep);
            }
            if (result != 0)
            {
                LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>::compute_device_request_batch_decode_equivalent] Grouped request decode failed"
                          << " requests=" << request_count
                          << " query_rows=" << query_rows
                          << " max_kv_len=" << max_kv_len
                          << " max_num_splits=" << max_num_splits);
                return false;
            }
            return true;
        }

        // =====================================================================
        // FP32 IWorkspaceConsumer Interface Implementation
        // =====================================================================

        WorkspaceRequirements CUDAFlashAttentionKernelT<ActivationPrecision::FP32>::getWorkspaceRequirements(
            int m, int n, int k) const
        {
            const int batch_size = (m > 0) ? m : 1;
            const int n_heads = (n > 0) ? n : 128;  // Max expected heads
            const int head_dim = (k > 0) ? k : 128; // Max expected head dim
            constexpr int max_kv_len = 4096;
            const attention_workspace::Geometry geometry{
                .compact_query_rows = batch_size,
                .request_count = batch_size,
                .local_query_heads = n_heads,
                // The kernel interface has no independent KV-head argument.
                // The owning stage replaces this conservative upper bound with
                // its exact GQA request geometry before arena publication.
                .local_kv_heads = n_heads,
                .head_dim = head_dim,
                .context_rows = max_kv_len,
                .decode_splits = MAX_NUM_SPLITS,
                .include_device_params = true,
                .include_fp32_kv_conversion = true,
            };
            WorkspaceRequirements reqs =
                attention_workspace::requirements(geometry);
            const auto *partial_output =
                reqs.find(AttentionWorkspaceBuffers::PARTIAL_OUTPUT);
            const auto *partial_m =
                reqs.find(AttentionWorkspaceBuffers::PARTIAL_M);
            const auto *partial_l =
                reqs.find(AttentionWorkspaceBuffers::PARTIAL_L);
            const size_t kv_convert_bytes =
                attention_workspace::fp32KVConversionBufferBytes(geometry);

            LOG_TRACE("[CUDAFlashAttentionKernelT<FP32>::getWorkspaceRequirements] "
                      << "batch=" << batch_size << " n_heads=" << n_heads << " head_dim=" << head_dim
                      << " num_splits=" << MAX_NUM_SPLITS
                      << " max_kv_len=" << max_kv_len
                      << " => partial_output=" << (partial_output->size_bytes / 1024) << "KB"
                      << ", partial_m=" << partial_m->size_bytes << "B"
                      << ", partial_l=" << partial_l->size_bytes << "B"
                      << ", kv_convert(each)=" << (kv_convert_bytes / (1024 * 1024)) << "MB");

            return reqs;
        }

        void CUDAFlashAttentionKernelT<ActivationPrecision::FP32>::bindWorkspace(
            DeviceWorkspaceManager *workspace)
        {
            freeWorkspace();
            workspace_ = workspace;
            dynamic_attn_device_valid_ = false;
            if (workspace)
            {
                LOG_TRACE("[CUDAFlashAttentionKernelT<FP32>] Bound workspace manager");
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

        bool CUDAFlashAttentionKernelT<ActivationPrecision::FP32>::writeDynamicAttnParams(
            int kv_len,
            int kv_stride,
            int position_offset,
            int query_rows,
            void *stream)
        {
            if (!stream)
            {
                LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>] Cannot write attention params on a null/default CUDA stream");
                dynamic_attn_device_valid_ = false;
                return false;
            }
            if (!workspace_)
            {
                LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>] Cannot write attention params without a bound workspace");
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
                LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>] Missing workspace buffer "
                          << AttentionWorkspaceBuffers::DEVICE_PARAMS
                          << " for " << query_rows << " attention param row(s)");
                dynamic_attn_device_valid_ = false;
                return false;
            }

            if (cudaFlashAttn_prepare_device_params_from_geometry(
                    d_buf,
                    kv_len,
                    kv_stride,
                    position_offset,
                    query_rows,
                    /*prefill_branch_condition=*/0,
                    /*direct_kv_limit=*/0,
                    stream) != 0)
            {
                LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>] Device attention-param writer failed"
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

        bool CUDAFlashAttentionKernelT<ActivationPrecision::FP32>::dynamicAttnParamsReady(
            int kv_len, int position_offset, int query_rows) const
        {
            const int sanitized_query_rows = sanitizeSmallDecodeQueryRows(query_rows);
            const int param_rows = sanitized_query_rows;
            return dynamic_attn_device_valid_ &&
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
            /*
             * A live-length update must never narrow the immutable cache
             * capacity that selected the captured launch envelope.  The
             * capacity is reset only at the explicit kernel/request lifecycle
             * boundary.  Preserving it here also keeps uncaptured diagnostics
             * byte-comparable with graph replay.
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

        bool CUDAFlashAttentionKernelT<ActivationPrecision::FP32>::prepareDynamicAttnParams(
            int kv_len,
            int position_offset,
            int query_rows,
            void *stream,
            int kv_stride)
        {
            if (!stream)
            {
                LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>] prepareDynamicAttnParams requires an explicit non-null CUDA stream");
                dynamic_attn_device_valid_ = false;
                return false;
            }
            setGPUStream(stream);
            const int resolved_kv_stride =
                std::max(kv_len, kv_stride > 0 ? kv_stride : kv_len);
            const int sanitized_query_rows =
                sanitizeSmallDecodeQueryRows(query_rows);
            const bool same_params =
                !dynamic_attn_device_derived_ &&
                dynamic_attn_device_valid_ &&
                dynamic_attn_kv_len_ == kv_len &&
                dynamic_attn_kv_stride_ == resolved_kv_stride &&
                dynamic_attn_position_offset_ == position_offset &&
                dynamic_attn_query_rows_ == sanitized_query_rows &&
                dynamic_attn_param_rows_ == sanitized_query_rows;
            if (!same_params)
            {
                dynamic_attn_kv_len_ = kv_len;
                dynamic_attn_kv_stride_ = resolved_kv_stride;
                dynamic_attn_position_offset_ = position_offset;
                dynamic_attn_query_rows_ = sanitized_query_rows;
                dynamic_attn_param_rows_ = sanitized_query_rows;
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
                LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>] Dynamic attention params not ready after prepare"
                          << " requested(kv_len=" << kv_len
                          << ", kv_stride=" << resolved_kv_stride
                          << ", pos=" << position_offset
                          << ", rows=" << sanitizeSmallDecodeQueryRows(query_rows) << ")"
                          << " prepared(kv_len=" << dynamic_attn_kv_len_
                          << ", kv_stride=" << dynamic_attn_kv_stride_
                          << ", pos=" << dynamic_attn_position_offset_
                          << ", rows=" << dynamic_attn_query_rows_
                          << ", param_rows=" << dynamic_attn_param_rows_
                          << ") device_valid=" << dynamic_attn_device_valid_
                          << " workspace=" << (workspace_ != nullptr)
                          << " stream=" << stream_);
                return false;
            }
            return true;
        }

        bool CUDAFlashAttentionKernelT<ActivationPrecision::FP32>::prepareDynamicAttnParamsFromDeviceSequenceState(
            const int *post_append_cached_tokens_device,
            int seq_len,
            int query_rows,
            void *stream,
            int kv_stride,
            const int *active_query_rows_device,
            const attention::AttentionPrefillCaptureGeometry &prefill_capture,
            const int *device_ring_head,
            int ring_capacity)
        {
            const int sanitized_query_rows = sanitizeSmallDecodeQueryRows(query_rows);
            if (!post_append_cached_tokens_device || seq_len <= 0 ||
                kv_stride <= 0 || !stream ||
                ((device_ring_head == nullptr) != (ring_capacity == 0)) ||
                (ring_capacity > 0 && ring_capacity != kv_stride))
            {
                LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>] Device-derived attention params require count pointer, coherent ring geometry, positive seq_len/cache capacity, and explicit stream");
                dynamic_attn_device_valid_ = false;
                dynamic_attn_device_derived_ = false;
                return false;
            }
            if (!workspace_)
            {
                LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>] Device-derived attention params require a bound workspace");
                dynamic_attn_device_valid_ = false;
                dynamic_attn_device_derived_ = false;
                return false;
            }

            void *d_buf = workspace_->getBuffer(AttentionWorkspaceBuffers::DEVICE_PARAMS);
            if (!d_buf)
            {
                LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>] Missing workspace buffer "
                          << AttentionWorkspaceBuffers::DEVICE_PARAMS
                          << " for device-derived attention params");
                dynamic_attn_device_valid_ = false;
                dynamic_attn_device_derived_ = false;
                return false;
            }

            setGPUStream(stream);
            if (pending_prefill_conditional_)
            {
                LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>] Previous adaptive prefill transaction was not consumed before a new parameter publication");
                dynamic_attn_device_valid_ = false;
                dynamic_attn_device_derived_ = false;
                return false;
            }
            if (!prefill_capture.empty() && !prefill_capture.valid())
            {
                LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>] Partially specified prefill capture geometry cannot publish device attention params");
                dynamic_attn_device_valid_ = false;
                dynamic_attn_device_derived_ = false;
                return false;
            }
            if (prefill_capture.valid() &&
                prefill_capture.kv_capacity != kv_stride)
            {
                LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>] Prefill capture capacity disagrees with attention-param stride"
                          << " capture_capacity=" << prefill_capture.kv_capacity
                          << " kv_stride=" << kv_stride);
                dynamic_attn_device_valid_ = false;
                dynamic_attn_device_derived_ = false;
                return false;
            }

            cudaStreamCaptureStatus capture_status =
                cudaStreamCaptureStatusNone;
            const cudaError_t capture_query_status = cudaStreamIsCapturing(
                static_cast<cudaStream_t>(stream),
                &capture_status);
            if (capture_query_status != cudaSuccess)
            {
                LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>] Cannot query capture state before device attention-param publication: "
                          << cudaGetErrorString(capture_query_status));
                dynamic_attn_device_valid_ = false;
                dynamic_attn_device_derived_ = false;
                return false;
            }

            unsigned long long prefill_branch_condition = 0;
            int direct_kv_limit = 0;
            if (capture_status == cudaStreamCaptureStatusActive &&
                prefill_capture.valid())
            {
                const fa2_policy::FA2PrefillParallelPlan capture_plan =
                    resolvePrefillPlanForDevice(
                        prefill_capture.batch_size,
                        prefill_capture.query_rows,
                        prefill_capture.local_query_heads,
                        prefill_capture.head_dim,
                        prefill_capture.kv_capacity,
                        device_idx_,
                        prefill_capture.execution_policy);
                if (!capture_plan.valid)
                {
                    LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>] Invalid prefill plan before device attention-param publication");
                    dynamic_attn_device_valid_ = false;
                    dynamic_attn_device_derived_ = false;
                    return false;
                }

                if (capture_plan.usesDeviceAdaptiveParallelism())
                {
#if CUDART_VERSION >= 12030
                    pending_prefill_conditional_ =
                        std::make_unique<CUDAActiveCaptureConditional>(
                            static_cast<cudaStream_t>(stream),
                            CUDAActiveCaptureConditionalKind::IfOnly);
                    if (!pending_prefill_conditional_->ready())
                    {
                        LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>] Cannot begin adaptive prefill transaction: "
                                  << pending_prefill_conditional_->error());
                        pending_prefill_conditional_.reset();
                        dynamic_attn_device_valid_ = false;
                        dynamic_attn_device_derived_ = false;
                        return false;
                    }
                    direct_kv_limit =
                        fa2_policy::kFA2CanonicalContextPartitionKeys *
                        capture_plan.device_direct_partition_limit;
                    const bool published =
                        pending_prefill_conditional_->publishPredicate(
                            [&](cudaGraphConditionalHandle condition)
                            {
                                prefill_branch_condition =
                                    static_cast<unsigned long long>(condition);
                                return cudaFlashAttn_prepare_device_params_from_count(
                                           d_buf,
                                           post_append_cached_tokens_device,
                                           seq_len,
                                           sanitized_query_rows,
                                           kv_stride,
                                           active_query_rows_device,
                                           device_ring_head,
                                           ring_capacity,
                                           prefill_branch_condition,
                                           direct_kv_limit,
                                           stream) == 0;
                            });
                    if (!published)
                    {
                        LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>] Device attention-param predicate publication failed: "
                                  << pending_prefill_conditional_->error());
                        pending_prefill_conditional_.reset();
                        dynamic_attn_device_valid_ = false;
                        dynamic_attn_device_derived_ = false;
                        return false;
                    }
#else
                    LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>] Adaptive prefill requires CUDA 12.3 native conditional graphs");
                    dynamic_attn_device_valid_ = false;
                    dynamic_attn_device_derived_ = false;
                    return false;
#endif
                }
            }

            const int rc = prefill_branch_condition != 0
                               ? 0
                               : cudaFlashAttn_prepare_device_params_from_count(
                                     d_buf,
                                     post_append_cached_tokens_device,
                                     seq_len,
                                     sanitized_query_rows,
                                     kv_stride,
                                     active_query_rows_device,
                                     device_ring_head,
                                     ring_capacity,
                                     /*prefill_branch_condition=*/0,
                                     /*direct_kv_limit=*/0,
                                     stream);
            if (rc != 0)
            {
                LOG_ERROR("[CUDAFlashAttentionKernelT<FP32>] Failed to derive attention params from device KV count");
                dynamic_attn_device_valid_ = false;
                dynamic_attn_device_derived_ = false;
                return false;
            }

            dynamic_attn_kv_len_ = 0;
            dynamic_attn_kv_stride_ = kv_stride;
            dynamic_attn_position_offset_ = 0;
            dynamic_attn_query_rows_ = sanitized_query_rows;
            dynamic_attn_param_rows_ = sanitized_query_rows;
            dynamic_attn_device_valid_ = true;
            dynamic_attn_device_derived_ = true;
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
            const int num_splits =
                computeDecodeSplitEnvelopeForDevice(
                    max_seq_len,
                    n_heads,
                    device_idx_);

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
                if (dynamic_attn_device_valid_)
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

            LOG_DEBUG("[CUDAFlashAttentionKernelT<FP16>] Created for device " << device_idx_
                                                                              << "; awaiting explicit execution stream");
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
              max_splits_(other.max_splits_),
              workspace_(other.workspace_),
              device_ctx_(other.device_ctx_)
        {
            other.stream_ = nullptr;
            other.partial_output_buf_ = nullptr;
            other.partial_m_buf_ = nullptr;
            other.partial_l_buf_ = nullptr;
            other.workspace_ = nullptr;
            other.device_ctx_ = nullptr;
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
                workspace_ = other.workspace_;
                device_ctx_ = other.device_ctx_;
                other.stream_ = nullptr;
                other.partial_output_buf_ = nullptr;
                other.partial_m_buf_ = nullptr;
                other.partial_l_buf_ = nullptr;
                other.workspace_ = nullptr;
                other.device_ctx_ = nullptr;
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
            int gqa_n_rep,
            const attention::AttentionExecutionPolicy &execution_policy,
            const attention::AttentionKVLogicalView &kv_logical_view)
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
                                              head_start, local_n_heads, local_n_kv_heads, gqa_n_rep,
                                              execution_policy, kv_logical_view);
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
                LOG_TRACE("[CUDAFlashAttentionKernelT<FP16>] Bound workspace manager");
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
            if (!stream_ || !workspace_)
                return;

            void *device_params =
                workspace_->getBuffer(AttentionWorkspaceBuffers::DEVICE_PARAMS);
            if (!device_params)
            {
                LOG_ERROR("[CUDAFlashAttentionKernelT<FP16>] Missing device attention-param workspace");
                return;
            }

            if (cudaFlashAttn_prepare_device_params_from_geometry(
                    device_params, kv_len, kv_len, position_offset, 1,
                    /*prefill_branch_condition=*/0,
                    /*direct_kv_limit=*/0,
                    stream_) != 0)
            {
                LOG_ERROR("[CUDAFlashAttentionKernelT<FP16>] Device attention-param writer failed");
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

            LOG_DEBUG("[CUDAFlashAttentionKernelT<BF16>] Created for device " << device_idx_
                                                                              << "; awaiting explicit execution stream");
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
              max_splits_(other.max_splits_),
              workspace_(other.workspace_),
              device_ctx_(other.device_ctx_)
        {
            other.stream_ = nullptr;
            other.partial_output_buf_ = nullptr;
            other.partial_m_buf_ = nullptr;
            other.partial_l_buf_ = nullptr;
            other.workspace_ = nullptr;
            other.device_ctx_ = nullptr;
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
                workspace_ = other.workspace_;
                device_ctx_ = other.device_ctx_;
                other.stream_ = nullptr;
                other.partial_output_buf_ = nullptr;
                other.partial_m_buf_ = nullptr;
                other.partial_l_buf_ = nullptr;
                other.workspace_ = nullptr;
                other.device_ctx_ = nullptr;
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
            int gqa_n_rep,
            const attention::AttentionExecutionPolicy &execution_policy,
            const attention::AttentionKVLogicalView &kv_logical_view)
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
                                              head_start, local_n_heads, local_n_kv_heads, gqa_n_rep,
                                              execution_policy, kv_logical_view);
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
                LOG_TRACE("[CUDAFlashAttentionKernelT<BF16>] Bound workspace manager");
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
            if (!stream_ || !workspace_)
                return;

            void *device_params =
                workspace_->getBuffer(AttentionWorkspaceBuffers::DEVICE_PARAMS);
            if (!device_params)
            {
                LOG_ERROR("[CUDAFlashAttentionKernelT<BF16>] Missing device attention-param workspace");
                return;
            }

            if (cudaFlashAttn_prepare_device_params_from_geometry(
                    device_params, kv_len, kv_len, position_offset, 1,
                    /*prefill_branch_condition=*/0,
                    /*direct_kv_limit=*/0,
                    stream_) != 0)
            {
                LOG_ERROR("[CUDAFlashAttentionKernelT<BF16>] Device attention-param writer failed");
            }
        }

    } // namespace cuda
} // namespace llaminar2
