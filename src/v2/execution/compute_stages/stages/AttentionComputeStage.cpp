/**
 * @file AttentionComputeStage.cpp
 * @brief Implementation of AttentionComputeStage
 */

#include "AttentionComputeStage.h"
#include "../ComputeStageUtils.h"
#include "../../../utils/DebugEnv.h"
#include "../../../tensors/Tensors.h"
#include "../../../tensors/GpuTensorView.h"
#include "../../../tensors/TensorKernels.h"
#include "../../../tensors/TQ8Tensor.h"
#include "../../../tensors/TQ4Tensor.h"
#include "../../../tensors/FP16Utils.h"
#include "../../../tensors/SIMDHelpers.h"
#include "../../../transfer/TransferEngine.h"
#include "../../../backends/BackendManager.h"
#include "../../../execution/local_execution/graph/GraphCaptureGuard.h"
#include "../../../kernels/attention/AttentionDeviceParams.h"
#include "../../../kernels/cpu/CPUKVCache.h"
#include "../../../kernels/cpu/turboquant/TurboQuantContext.h"
#include "../../../kernels/cpu/rotation/ActivationRotation.h"
#include "../../../utils/Logger.h"
#include "../../../kernels/KernelFactory.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <stdexcept>

#if defined(HAVE_CUDA)
#include "../../../kernels/cuda/kvcache/CUDARingKVCacheTQ.h"
#include "../../../kernels/cuda/attention/CUDAFlashAttentionKernelT.h"
#endif

namespace llaminar2
{

    namespace
    {
        /**
         * @brief True when a verifier Q tensor is laid out as [head][row][dim].
         *
         * CPU HybridQ16 RoPE intentionally stores Q in head-major form because
         * the integer attention kernel consumes one head block at a time.  The
         * grouped verifier, however, must replay each logical token as an M=1
         * decode row, whose Q tensor is [row][head * dim].  Treating row `i`
         * as contiguous in the head-major buffer mixes future verifier rows
         * into the current row and breaks decode equivalence.
         */
        bool isHeadMajorVerifierQ(const TensorBase *q,
                                  int seq_len,
                                  int n_heads,
                                  int head_dim)
        {
            if (!q || q->native_type() != TensorType::Q16_1 ||
                seq_len <= 1 || n_heads <= 0 || head_dim <= 0)
            {
                return false;
            }

            const auto &shape = q->shape();
            return shape.size() >= 2 &&
                   shape[0] == static_cast<size_t>(n_heads * seq_len) &&
                   shape[1] == static_cast<size_t>(head_dim);
        }

        /**
         * @brief Copy one logical verifier Q row into a one-token FP32 tensor.
         *
         * Most attention Q tensors are row-major and can be copied in one
         * contiguous slice.  HybridQ16 is the exception: RoPE writes head-major
         * Q for the CPU integer attention kernel, so the verifier must gather
         * the requested row across all heads before invoking the serial M=1
         * attention path.
         */
        bool copyHostVerifierQRow(const TensorBase *src,
                                  int row,
                                  int seq_len,
                                  int n_heads,
                                  int head_dim,
                                  FP32Tensor *dst)
        {
            if (!src || !dst || row < 0 || row >= seq_len)
            {
                LOG_ERROR("[AttentionComputeStage] Invalid verifier Q row copy request");
                return false;
            }

            const float *src_data = src->data();
            float *dst_data = dst->mutable_data();
            if (!src_data || !dst_data)
            {
                LOG_ERROR("[AttentionComputeStage] Verifier Q row copy requires host-visible FP32 data");
                return false;
            }

            const int q_dim = n_heads * head_dim;
            if (isHeadMajorVerifierQ(src, seq_len, n_heads, head_dim))
            {
                for (int h = 0; h < n_heads; ++h)
                {
                    const size_t src_offset =
                        (static_cast<size_t>(h) * static_cast<size_t>(seq_len) +
                         static_cast<size_t>(row)) *
                        static_cast<size_t>(head_dim);
                    const size_t dst_offset =
                        static_cast<size_t>(h) * static_cast<size_t>(head_dim);
                    std::copy_n(src_data + src_offset, head_dim, dst_data + dst_offset);
                }
                return true;
            }

            const auto &shape = src->shape();
            if (shape.size() < 2 ||
                shape[0] < static_cast<size_t>(seq_len) ||
                shape[1] < static_cast<size_t>(q_dim))
            {
                LOG_ERROR("[AttentionComputeStage] Verifier Q tensor shape ["
                          << (shape.empty() ? 0 : shape[0]) << ","
                          << (shape.size() > 1 ? shape[1] : 0)
                          << "] cannot provide row-major q_dim=" << q_dim);
                return false;
            }

            std::copy_n(src_data + static_cast<size_t>(row) * static_cast<size_t>(q_dim),
                        q_dim,
                        dst_data);
            return true;
        }

        bool copyAttentionFP32DeviceRow(
            TensorBase *dst,
            int dst_row,
            int dst_cols,
            const TensorBase *src,
            int src_row,
            int src_cols,
            int copy_cols,
            DeviceId device,
            void *stream,
            const char *label)
        {
            if (!stream)
            {
                LOG_ERROR("[AttentionComputeStage] " << label
                                                      << " requires an explicit GPU stream");
                return false;
            }

            IBackend *backend = getBackendFor(device);
            if (!backend)
            {
                LOG_ERROR("[AttentionComputeStage] No backend for " << device.to_string()
                                                                     << " while copying " << label);
                return false;
            }

            auto *dst_ptr = static_cast<float *>(dst ? dst->gpu_data_ptr() : nullptr);
            const auto *src_ptr = static_cast<const float *>(src ? src->gpu_data_ptr() : nullptr);
            if (!dst_ptr || !src_ptr)
            {
                LOG_ERROR("[AttentionComputeStage] Null device pointer while copying "
                          << label << " dst=" << static_cast<void *>(dst_ptr)
                          << " src=" << static_cast<const void *>(src_ptr));
                return false;
            }

            const size_t dst_offset = static_cast<size_t>(dst_row) * static_cast<size_t>(dst_cols);
            const size_t src_offset = static_cast<size_t>(src_row) * static_cast<size_t>(src_cols);
            const size_t bytes = static_cast<size_t>(copy_cols) * sizeof(float);
            const bool ok = backend->deviceCopyAsync(
                dst_ptr + dst_offset,
                src_ptr + src_offset,
                bytes,
                device.gpu_ordinal(),
                stream);
            if (!ok)
            {
                LOG_ERROR("[AttentionComputeStage] Device row copy failed for " << label
                                                                                << " bytes=" << bytes);
                return false;
            }
            return true;
        }
    } // namespace

    // =============================================================================
    // AttentionComputeStage Implementation
    // =============================================================================

    AttentionComputeStage::AttentionComputeStage(Params params)
        : IComputeStage(params.device_id),
          params_(std::move(params)),
          cpu_grouped_k_views_(static_cast<size_t>(std::max(0, params_.batch_size)), nullptr),
          cpu_grouped_v_views_(static_cast<size_t>(std::max(0, params_.batch_size)), nullptr),
          cpu_grouped_kv_lens_(static_cast<size_t>(std::max(0, params_.batch_size)), 0)
    {
        if (params_.device_id.is_gpu() &&
            params_.kv_cache &&
            debugEnv().attention.debug_effective_kv_snapshot &&
            debugEnv().attention.debugEffectiveKVSnapshotLayerSelected(
                params_.layer_idx))
        {
            const int request_count = std::max(1, params_.batch_size);
            debug_device_kv_count_views_.reserve(
                static_cast<size_t>(request_count));
            debug_device_kv_head_views_.reserve(
                static_cast<size_t>(request_count));
            debug_device_kv_count_names_.reserve(
                static_cast<size_t>(request_count));
            debug_device_kv_head_names_.reserve(
                static_cast<size_t>(request_count));

            for (int request = 0; request < request_count; ++request)
            {
                const int *count =
                    params_.kv_cache->deviceCachedTokenCountPtr(
                        params_.layer_idx,
                        request);
                const int *head =
                    params_.kv_cache->deviceRingHeadPtr(
                        params_.layer_idx,
                        request);
                if (!count || !head)
                {
                    throw std::runtime_error(
                        "effective-KV diagnostics require canonical device "
                        "count and ring-head pointers for every request");
                }

                debug_device_kv_count_names_.push_back(
                    "device_kv_count_request_" +
                    std::to_string(request));
                debug_device_kv_head_names_.push_back(
                    "device_kv_head_request_" +
                    std::to_string(request));
                debug_device_kv_count_views_.push_back(
                    std::make_unique<GpuTensorView>(
                        const_cast<int *>(count),
                        /*rows=*/1,
                        /*cols=*/1,
                        TensorType::INT32,
                        params_.device_id));
                debug_device_kv_head_views_.push_back(
                    std::make_unique<GpuTensorView>(
                        const_cast<int *>(head),
                        /*rows=*/1,
                        /*cols=*/1,
                        TensorType::INT32,
                        params_.device_id));
            }
        }
    }

    int AttentionComputeStage::dynamicAttentionParamRows(int logical_seq_len, int kv_len) const
    {
        if (!params_.kv_cache)
            return 1;

        /*
         * Every GPU grouped verifier call below is row-local and consumes one
         * AttentionDeviceParams record per logical row. Cache storage format
         * cannot change that launch contract: native FP16/BF16/FP32/Q8 caches
         * and RoPE-on-read converted views all enter the same grouped API.
         * Returning one row for any format leaves rows 1..M uninitialized and
         * silently turns a valid grouped launch into zero output.
         */
        const bool gpu_grouped_verifier =
            params_.device_id.is_gpu() &&
            params_.batch_size == 1 &&
            params_.causal &&
            logical_seq_len > 1 &&
            logical_seq_len <= attention::kMaxGroupedVerifierAttentionRows &&
            kv_len > logical_seq_len;
        return gpu_grouped_verifier ? logical_seq_len : 1;
    }

    void AttentionComputeStage::updateDynamicParams(int pos_offset, int seq_len)
    {
        params_.position_offset = pos_offset;
        if (!params_.kv_cache || params_.layer_idx < 0)
        {
            dynamic_pre_append_cached_tokens_ = -1;
            dynamic_logical_seq_len_ = 0;
            dynamic_post_append_kv_len_ = 0;
            return;
        }

        const bool padded_prefill_replay =
            prefill_replay_params_set_ &&
            prefill_effective_seq_len_ > 0 &&
            prefill_bucket_seq_len_ > 0 &&
            prefill_bucket_seq_len_ == seq_len &&
            prefill_effective_seq_len_ < seq_len;
        const int logical_seq_len = padded_prefill_replay
                                        ? prefill_effective_seq_len_
                                        : seq_len;

        /*
         * `pos_offset` is the admitted request range, supplied before capture
         * or replay. It is sufficient for fixed host launch geometry and does
         * not require observing canonical GPU sequence state. The attention
         * kernel below still derives its live row lengths from the device count
         * written by the preceding captured append.
         */
        const int cache_capacity = std::max(1, params_.kv_cache->max_seq_len());
        const int pre_append_cached_tokens =
            std::clamp(pos_offset, 0, cache_capacity);
        dynamic_pre_append_cached_tokens_ = pre_append_cached_tokens;
        dynamic_logical_seq_len_ = logical_seq_len;
        dynamic_post_append_kv_len_ = std::clamp(
            pre_append_cached_tokens + logical_seq_len,
            1,
            cache_capacity);

        if (!cached_kernel_)
        {
            return;
        }

        // Propagate the current stage stream so every device-side parameter
        // writer is ordered with the attention graph that consumes it.
        bindStageStream(cached_kernel_);

        const int kv_len = dynamic_post_append_kv_len_;
        const int logical_pos_offset = std::max(0, kv_len - logical_seq_len);
        const int query_rows_for_params =
            dynamicAttentionParamRows(logical_seq_len, kv_len);

        /*
         * GPU prefill graphs are reusable across request lifetimes. A graph
         * captured for the first prompt chunk (pre_append_cached_tokens == 0)
         * can later replay as a suffix after prefix restore. Always recording
         * the device-count derivation node makes the graph topology invariant:
         * the KV append kernel owns the device sequence state, and attention
         * consumes that post-append state on the same stream.
         */
        const bool will_derive_from_device_count =
            params_.device_id.is_gpu() &&
            params_.kv_cache->deviceCachedTokenCountPtr(params_.layer_idx, 0) != nullptr;
        if (!will_derive_from_device_count &&
            !cached_kernel_->prepareDynamicAttnParams(
                kv_len,
                logical_pos_offset,
                query_rows_for_params,
                gpuStream(),
                cache_capacity))
        {
            const std::string msg =
                "[AttentionComputeStage] Failed to prepare dynamic attention params for layer " +
                std::to_string(params_.layer_idx) +
                " on " + params_.device_id.toString();
            LOG_ERROR(msg);
            throw std::runtime_error(msg);
        }

    }

    bool AttentionComputeStage::supportsDeviceResidentDynamicPositionReplay() const
    {
        if (!params_.device_id.is_gpu() ||
            !params_.kv_cache ||
            params_.layer_idx < 0)
        {
            return false;
        }

        return params_.kv_cache->deviceCachedTokenCountPtr(params_.layer_idx, 0) != nullptr;
    }

    void AttentionComputeStage::updatePrefillReplayParams(const PrefillReplayParams &replay)
    {
        prefill_bucket_seq_len_ = replay.bucket_seq_len > 0 ? replay.bucket_seq_len : params_.seq_len;
        const int real_seq_len = replay.real_seq_len > 0 ? replay.real_seq_len : params_.seq_len;
        prefill_effective_seq_len_ = std::clamp(real_seq_len, 1, std::max(1, params_.seq_len));
        prefill_replay_params_set_ = true;
    }

    // =============================================================================
    // Kernel Caching (for Workspace Binding)
    // =============================================================================

    ITensorAttention *AttentionComputeStage::getOrCreateKernel()
    {
        // Need Q tensor to determine kernel type
        if (!params_.Q)
        {
            LOG_ERROR("[AttentionComputeStage::getOrCreateKernel] Q tensor not set");
            return nullptr;
        }

        auto *kernel = getOrRefreshKernelByTensorType(
            cached_kernel_,
            cached_kernel_tensor_type_,
            params_.Q,
            [&]()
            {
                return llaminar::v2::kernels::KernelFactory::getOrCreateAttention(params_.Q, params_.device_id);
            });
        if (!kernel)
        {
            LOG_ERROR("[AttentionComputeStage::getOrCreateKernel] Failed to create attention kernel for "
                      << params_.Q->dtype_name());
            return nullptr;
        }

        LOG_TRACE("[AttentionComputeStage::getOrCreateKernel] Created and cached attention kernel for "
                  << params_.Q->dtype_name() << " on " << params_.device_id.to_string());

        return kernel;
    }

    IWorkspaceConsumer *AttentionComputeStage::getKernelAsWorkspaceConsumer()
    {
        auto *kernel = getOrCreateKernel();
        if (!kernel)
        {
            return nullptr;
        }
        return dynamic_cast<IWorkspaceConsumer *>(kernel);
    }

    WorkspaceRequirements AttentionComputeStage::getWorkspaceRequirements(int m, int n, int k) const
    {
        auto *self = const_cast<AttentionComputeStage *>(this);
        auto *consumer = self->getKernelAsWorkspaceConsumer();
        if (!consumer)
        {
            LOG_TRACE("[AttentionComputeStage] No attention kernel available for workspace requirements");
            return WorkspaceRequirements{};
        }

        /*
         * The generic workspace allocator visits the ordinary one-row decode
         * graph.  That same graph is later replayed for an MTP verifier group,
         * however, and split attention indexes its partial-output/M/L buffers
         * by every query row.  Reserve the complete production verifier span
         * even when the allocator's model-level M hint is one.  Stage-local
         * head dimensions remain authoritative because TP/MoE graph hints can
         * describe a smaller shard than the concrete attention stage.
         */
        const attention::AttentionWorkspaceCardinality
            workspace_cardinality =
                attention::planAttentionWorkspaceCardinality(
                    m,
                    params_.batch_size);
        const int workspace_partial_rows =
            workspace_cardinality.compact_query_rows;
        const int workspace_heads = std::max(n, params_.n_heads);
        const int workspace_head_dim = std::max(k, params_.head_dim);

        auto reqs = consumer->getWorkspaceRequirements(
            workspace_partial_rows,
            workspace_heads,
            workspace_head_dim);

        /*
         * Mixed-precision KV conversion has a different index space from the
         * split-attention partials: it stores one cache per independent request,
         * not one cache per verifier query row.  Restore these two descriptors
         * to the real request count after asking the backend for conservatively
         * grouped partial buffers.  This preserves correctness without paying
         * four complete 4096-token KV conversion buffers for one request.
         */
        const int workspace_kv_heads = std::max(1, params_.n_kv_heads);
        const size_t kv_convert_bytes =
                                        static_cast<size_t>(
                                            workspace_cardinality
                                                .request_count) *
                                        4096ULL *
                                        static_cast<size_t>(workspace_kv_heads) *
                                        static_cast<size_t>(workspace_head_dim) *
                                        sizeof(float);
        for (auto &buffer : reqs.buffers)
        {
            if (buffer.name == "attn_k_tmp_fp32" ||
                buffer.name == "attn_v_tmp_fp32")
            {
                buffer.size_bytes = kv_convert_bytes;
            }
        }

        return reqs;
    }

    bool AttentionComputeStage::execute(IDeviceContext *ctx)
    {
        const bool gpu_stage = params_.device_id.is_gpu();
        if (gpu_stage)
            (void)requireGPUStream();
        if (gpu_stage && params_.kv_cache && !params_.read_kv_from_cache)
        {
            LOG_ERROR("[AttentionComputeStage] GPU attention with a KV cache requires the device-owned post-append cache path"
                      << " layer=" << params_.layer_idx
                      << " device=" << params_.device_id.toString());
            return false;
        }

        const bool padded_prefill_replay =
            prefill_replay_params_set_ &&
            prefill_effective_seq_len_ > 0 &&
            prefill_bucket_seq_len_ > 0 &&
            prefill_bucket_seq_len_ == params_.seq_len &&
            prefill_effective_seq_len_ < params_.seq_len;
        const int logical_seq_len = padded_prefill_replay
                                        ? prefill_effective_seq_len_
                                        : params_.seq_len;
        const bool has_current_dynamic_sequence_state =
            dynamic_pre_append_cached_tokens_ >= 0 &&
            dynamic_logical_seq_len_ == logical_seq_len &&
            dynamic_post_append_kv_len_ >= logical_seq_len;

        // The scalar horizon controls only host-side launch geometry. GPU
        // kernels consume the canonical post-append device count below; CPU
        // caches retain their ordinary synchronous sequence-state interface.
        int effective_kv_len = params_.kv_len;
        if (params_.kv_cache && params_.layer_idx >= 0)
        {
            if (has_current_dynamic_sequence_state)
            {
                effective_kv_len = dynamic_post_append_kv_len_;
            }
            else
            {
                if (gpu_stage)
                {
                    effective_kv_len = std::clamp(
                        std::max(0, params_.position_offset) + logical_seq_len,
                        std::max(1, logical_seq_len),
                        std::max(1, params_.kv_cache->max_seq_len()));
                }
                else
                {
                    effective_kv_len = params_.kv_cache->get_cached_tokens(
                        params_.layer_idx,
                        0);
                    if (effective_kv_len == 0)
                    {
                        effective_kv_len = params_.seq_len;
                    }
                }
            }
            LOG_TRACE("[AttentionComputeStage] Dynamic kv_len from cache: " << effective_kv_len
                                                                            << " (static was: " << params_.kv_len << ")");
        }
        const int effective_kv_stride =
            gpu_stage && params_.kv_cache
                ? std::max(1, params_.kv_cache->max_seq_len())
                : std::max(1, effective_kv_len);
        // Read K/V from cache at execution time when requested.
        // This allows GPU prefill to use the FP16 tensors in the KV cache
        // (populated by KVCacheAppendStage) instead of the Q8_1 projection
        // buffers, eliminating the Q8_1→FP32→FP16 triple conversion.
        ITensor *effective_K = params_.K;
        ITensor *effective_V = params_.V;
        bool cpu_grouped_request_cache = false;
        if (params_.kv_cache && params_.layer_idx >= 0)
        {
            // Always override K/V from KV cache when:
            // 1. read_kv_from_cache is set (GPU optimization for type conversion), OR
            // 2. We're in decode mode (effective_kv_len > seq_len) - the graph may have
            //    been built during prefill (cached_tokens=0) with K/V wired to activation
            //    scratch buffers. During decode, those buffers only hold the current token's
            //    projection. The full KV history lives in the cache (populated by
            //    KVCacheAppendStage which runs before this stage).
            // 3. apply_rope_to_k is set (rope_on_read mode) - K is stored pre-RoPE
            //    in the cache, so we must read through get_kv_converted() which
            //    fuses RoPE into the read path. This applies to BOTH prefill and decode.
            if (params_.read_kv_from_cache || effective_kv_len > params_.seq_len || params_.apply_rope_to_k)
            {
                ITensor *cache_k = nullptr;
                ITensor *cache_v = nullptr;

                // For CPU, use get_kv_converted<FP32>() which handles:
                // - Incremental dequant for all cache precisions (FP16, BF16, Q8_1, TQ4, split TQ)
                // - Optional fused RoPE-on-read (when apply_rope_to_k is set)
                // - Lazy shadow buffer management inside the cache
                // This covers both decode (full KV history) and prefill with rope_on_read
                // (K stored pre-RoPE, needs RoPE applied during read).
                const bool is_cpu_path = !gpu_stage;
                if (is_cpu_path && (effective_kv_len > params_.seq_len || params_.apply_rope_to_k))
                {
                    if (params_.batch_size > 1)
                    {
                        if (cpu_grouped_k_views_.size() !=
                                static_cast<size_t>(params_.batch_size) ||
                            cpu_grouped_v_views_.size() !=
                                static_cast<size_t>(params_.batch_size) ||
                            cpu_grouped_kv_lens_.size() !=
                                static_cast<size_t>(params_.batch_size))
                        {
                            LOG_ERROR("[AttentionComputeStage] CPU grouped KV descriptor capacity drifted from graph batch size");
                            return false;
                        }

                        IKVCache::KVReadParams read_params;
                        if (params_.apply_rope_to_k)
                        {
                            read_params.rope_theta = params_.rope_theta;
                            read_params.position_start = 0;
                            read_params.rope_dim = static_cast<int>(
                                params_.partial_rotary_factor * params_.head_dim);
                        }
                        read_params.n_kv_heads = params_.n_kv_heads;
                        read_params.head_dim = params_.head_dim;
                        read_params.turboquant_ctx = params_.turboquant_ctx;

                        int max_request_kv_len = 0;
                        for (int request = 0; request < params_.batch_size; ++request)
                        {
                            ITensor *request_k = nullptr;
                            ITensor *request_v = nullptr;
                            int request_kv_len = 0;
                            if (!params_.kv_cache->get_kv_converted(
                                    params_.layer_idx,
                                    request,
                                    ActivationPrecision::FP32,
                                    &request_k,
                                    &request_v,
                                    &request_kv_len,
                                    &read_params) ||
                                !request_k || !request_v ||
                                request_kv_len < params_.seq_len)
                            {
                                LOG_ERROR("[AttentionComputeStage] CPU grouped cache conversion failed"
                                          << " layer=" << params_.layer_idx
                                          << " request=" << request
                                          << " kv_len=" << request_kv_len);
                                return false;
                            }
                            cpu_grouped_k_views_[static_cast<size_t>(request)] = request_k;
                            cpu_grouped_v_views_[static_cast<size_t>(request)] = request_v;
                            cpu_grouped_kv_lens_[static_cast<size_t>(request)] = request_kv_len;
                            max_request_kv_len = std::max(max_request_kv_len, request_kv_len);
                        }

                        effective_K = const_cast<ITensor *>(cpu_grouped_k_views_.front());
                        effective_V = const_cast<ITensor *>(cpu_grouped_v_views_.front());
                        effective_kv_len = max_request_kv_len;
                        cpu_grouped_request_cache = true;
                    }

                    // =====================================================================
                    // Fused TQ attention path: pass raw TQ8/TQ4 tensors directly to the
                    // attention kernel, eliminating FP32 shadow buffers entirely.
                    //
                    // Conditions for fused path:
                    //   - Decode mode (effective_kv_len > seq_len)
                    //   - TQ cache format (TQ8-K / TQ4-V)
                    //   - NOT rope_on_read (RoPE already applied at write time)
                    //   - TurboQuantContext available (for rotation matrices)
                    //
                    // When active, the kernel exploits rotation orthogonality:
                    //   dot(Q, dequant(K)) = (norm/D) · dot(Π·Q, centroids(K))
                    // reducing per-position cost from O(D²) to O(D).
                    // =====================================================================
                    const bool is_tq_decode = (!cpu_grouped_request_cache &&
                                               effective_kv_len > params_.seq_len &&
                                               !params_.apply_rope_to_k &&
                                               params_.turboquant_ctx);
                    if (is_tq_decode)
                    {
                        ITensor *raw_k = params_.kv_cache->get_k(params_.layer_idx, 0);
                        ITensor *raw_v = params_.kv_cache->get_v(params_.layer_idx, 0);

                        if (raw_k && raw_v &&
                            raw_k->native_type() == TensorType::TQ8 &&
                            raw_v->native_type() == TensorType::TQ4)
                        {
                            // Set per-layer TQ context on tensors (needed by kernel for rotation)
                            const auto &layer_ctx = params_.turboquant_ctx->for_layer(params_.layer_idx);
                            auto *k_tq8 = dynamic_cast<TQ8Tensor *>(raw_k);
                            auto *v_tq4 = dynamic_cast<TQ4Tensor *>(raw_v);
                            if (k_tq8 && v_tq4)
                            {
                                k_tq8->set_turboquant_context(&layer_ctx);
                                v_tq4->set_turboquant_context(&layer_ctx);
                                effective_K = raw_k;
                                effective_V = raw_v;
                                LOG_TRACE("[AttentionComputeStage] Fused TQ path: passing raw TQ8/TQ4 tensors for layer "
                                          << params_.layer_idx << " kv_len=" << effective_kv_len);
                            }
                        }
                    }

                    // =====================================================================
                    // Fused Q16_1/Q8_1 attention path: pass raw quantized tensors directly
                    // to the attention kernel, eliminating FP32 shadow buffers.
                    //
                    // Q16_1: VNNI int16 QK dot product (VPDPWSSD) + int16 V accumulation
                    // Q8_1: int8→float inline dequant in attention's inner loop
                    //
                    // Conditions: decode mode, NOT rope_on_read, matching cache format
                    // =====================================================================
                    if (!cpu_grouped_request_cache &&
                        effective_K == params_.K && // not overridden by fused TQ path
                        effective_kv_len > params_.seq_len &&
                        !params_.apply_rope_to_k)
                    {
                        const auto kp = params_.kv_cache->k_precision();
                        const auto vp = params_.kv_cache->v_precision();
                        if ((kp == ActivationPrecision::Q16_1 && vp == ActivationPrecision::Q16_1) ||
                            (kp == ActivationPrecision::Q8_1 && vp == ActivationPrecision::Q8_1))
                        {
                            ITensor *raw_k = params_.kv_cache->get_k(params_.layer_idx, 0);
                            ITensor *raw_v = params_.kv_cache->get_v(params_.layer_idx, 0);
                            if (raw_k && raw_v)
                            {
                                effective_K = raw_k;
                                effective_V = raw_v;
                                LOG_TRACE("[AttentionComputeStage] Fused " << activationPrecisionToString(kp)
                                                                           << " path: passing raw tensors for layer "
                                                                           << params_.layer_idx << " kv_len=" << effective_kv_len);
                            }
                        }
                    }

                    // Converted-cache path: dequantize through get_kv_converted
                    // when fused raw attention cannot consume the cache format.
                    if (!cpu_grouped_request_cache &&
                        effective_K == params_.K) // not overridden by any fused path above
                    {
                        IKVCache::KVReadParams read_params;
                        if (params_.apply_rope_to_k)
                        {
                            read_params.rope_theta = params_.rope_theta;
                            read_params.position_start = 0; // Cache rows are stored in position order
                            read_params.rope_dim = static_cast<int>(
                                params_.partial_rotary_factor * params_.head_dim);
                        }
                        read_params.n_kv_heads = params_.n_kv_heads;
                        read_params.head_dim = params_.head_dim;
                        read_params.turboquant_ctx = params_.turboquant_ctx;
                        read_params.gpu_stream = gpuStream();

                        int kv_len_out = 0;
                        if (params_.kv_cache->get_kv_converted(
                                params_.layer_idx, 0,
                                ActivationPrecision::FP32,
                                &cache_k, &cache_v, &kv_len_out,
                                &read_params))
                        {
                            effective_K = cache_k;
                            effective_V = cache_v;
                            if (kv_len_out <= 0)
                            {
                                LOG_ERROR("[AttentionComputeStage] get_kv_converted returned invalid kv_len="
                                          << kv_len_out << " for layer " << params_.layer_idx);
                                return false;
                            }
                            effective_kv_len = kv_len_out;
                            LOG_TRACE("[AttentionComputeStage] Using cache get_kv<FP32> ("
                                      << cache_k->dtype_name() << ") for layer " << params_.layer_idx
                                      << " kv_len=" << kv_len_out);
                        }
                        else
                        {
                            LOG_ERROR("[AttentionComputeStage] get_kv_converted failed for layer "
                                      << params_.layer_idx << "; failing instead of substituting raw cache tensors");
                            return false;
                        }
                    } // end converted-cache get_kv_converted
                }
                else
                {
                    /*
                     * All GPU attention cache reads use one request-batched,
                     * graph-capturable operation, including batch size one.
                     * The cache reads canonical device head/count values and
                     * emits a fixed-stride view; no host-sized tensor wrapper,
                     * scalar snapshot path, or stale-shadow validity decision
                     * participates in production execution.
                     */
                    bool cache_read_ok = false;
                    if (params_.apply_rope_to_k)
                    {
                        IKVCache::KVReadParams read_params;
                        read_params.rope_theta = params_.rope_theta;
                        read_params.position_start = 0;
                        read_params.n_kv_heads = params_.n_kv_heads;
                        read_params.head_dim = params_.head_dim;
                        read_params.rope_dim = static_cast<int>(
                            params_.partial_rotary_factor * params_.head_dim);
                        read_params.turboquant_ctx = params_.turboquant_ctx;
                        read_params.gpu_stream = gpuStream();
                        cache_read_ok =
                            params_.kv_cache->get_kv_batched_converted_device_view(
                                params_.layer_idx,
                                /*first_seq_idx=*/0,
                                params_.batch_size,
                                ActivationPrecision::FP16,
                                &cache_k,
                                &cache_v,
                                read_params);
                    }
                    else
                    {
                        cache_read_ok = params_.kv_cache->get_kv_batched_device_view(
                            params_.layer_idx,
                            /*first_seq_idx=*/0,
                            params_.batch_size,
                            &cache_k,
                            &cache_v,
                            gpuStream());
                    }
                    if (!cache_read_ok || !cache_k || !cache_v)
                    {
                        LOG_ERROR("[AttentionComputeStage] Device-owned grouped KV read failed"
                                  << " layer=" << params_.layer_idx
                                  << " batch=" << params_.batch_size
                                  << " max_kv_len=" << effective_kv_len
                                  << " converted=" << params_.apply_rope_to_k
                                  << " device=" << params_.device_id.to_string());
                        return false;
                    }
                    effective_K = cache_k;
                    effective_V = cache_v;
                    LOG_TRACE("[AttentionComputeStage] Using device-owned grouped cache view"
                              << " layer=" << params_.layer_idx
                              << " batch=" << params_.batch_size
                              << " max_kv_len=" << effective_kv_len
                              << " type=" << cache_k->dtype_name());
                }
            }
        }

        // Detect attention mode if auto-detection enabled
        AttentionMode mode = params_.attention_mode;
        if (params_.auto_detect_mode)
        {
            mode = detect_attention_mode(params_.batch_size, params_.seq_len, effective_kv_len);
        }

        const bool is_decode_mode = (mode == AttentionMode::DECODE ||
                                     (params_.seq_len < effective_kv_len && params_.batch_size == 1));

        LOG_TRACE("[AttentionComputeStage] Execute: batch=" << params_.batch_size
                                                            << " seq_len=" << params_.seq_len
                                                            << " kv_len=" << effective_kv_len
                                                            << " n_heads=" << params_.n_heads
                                                            << " n_kv_heads=" << params_.n_kv_heads
                                                            << " head_dim=" << params_.head_dim
                                                            << " position_offset=" << params_.position_offset
                                                            << " mode=" << attention_mode_name(mode)
                                                            << " Q_type=" << (params_.Q ? params_.Q->dtype_name() : "null")
                                                            << " K_type=" << (effective_K ? effective_K->dtype_name() : "null")
                                                            << " V_type=" << (effective_V ? effective_V->dtype_name() : "null")
                                                            << " output=" << (void *)params_.output);

        // Validate inputs
        if (!ensureRequiredPointers("AttentionComputeStage", {
                                                                 {"Q", params_.Q},
                                                                 {"K", effective_K},
                                                                 {"V", effective_V},
                                                                 {"output", params_.output},
                                                             }))
        {
            return false;
        }

        if (params_.seq_len <= 0 || effective_kv_len <= 0 ||
            params_.n_heads <= 0 || params_.n_kv_heads <= 0 || params_.head_dim <= 0)
        {
            LOG_ERROR("[AttentionComputeStage] Invalid dimensions");
            return false;
        }

        if (params_.n_heads % params_.n_kv_heads != 0)
        {
            LOG_ERROR("[AttentionComputeStage] n_heads (" << params_.n_heads
                                                          << ") must be divisible by n_kv_heads (" << params_.n_kv_heads << ")");
            return false;
        }

        // Use cached kernel (enables workspace binding for GPU kernels)
        auto *kernel = getOrCreateKernel();
        if (!kernel)
        {
            LOG_ERROR("[AttentionComputeStage] Failed to get attention kernel");
            return false;
        }
        bindStageStream(kernel);

        /*
         * Independent GPU requests use separate KV banks and separate live
         * device counts.  They must never enter the ordinary batch path, whose
         * single AttentionDeviceParams row describes only equal-length batches.
         * The explicit request contract below supports compact per-request rows
         * up to the same sixteen-row bound used by grouped MTP verification.
         */
        /*
         * Small independent request batches always use the row-local resident
         * kernel, including the first prompt block where every cache count is
         * equal to the query-row count.  The same captured graph may replay
         * after prefix restore or a later append has grown those device-owned
         * counts.  Selecting ordinary prefill from the host launch horizon and
         * grouped decode only after observing a larger host KV length would
         * make the arithmetic regime depend on stale host state: graph replay
         * would keep prefill reduction order while an isolated request entered
         * the serial-decode-equivalent grouped path.
         *
         * The grouped request kernel derives every row's visible prefix and
         * launch policy from canonical cache counts on the execution stream.
         * Its fixed maximum-stride grid is valid for both initial prefill and
         * grown-prefix replay, making the regime choice graph-stable and fully
         * device-owned.
         */
        const bool gpu_grouped_request_decode =
            gpu_stage &&
            params_.kv_cache &&
            params_.layer_idx >= 0 &&
            params_.batch_size > 1 &&
            logical_seq_len > 0 &&
            params_.batch_size * logical_seq_len <=
                attention::kMaxGroupedVerifierAttentionRows &&
            params_.causal;

        if (gpu_stage && params_.kv_cache && params_.layer_idx >= 0)
        {
            const int *device_cached_tokens =
                params_.kv_cache->deviceCachedTokenCountPtr(params_.layer_idx, 0);
            const bool needs_device_sequence_params =
                has_current_dynamic_sequence_state ||
                effective_kv_len > logical_seq_len;
            if (device_cached_tokens && needs_device_sequence_params)
            {
                if (!gpu_grouped_request_decode)
                {
                    const int query_rows_for_params =
                        dynamicAttentionParamRows(logical_seq_len, effective_kv_len);
                    if (!kernel->prepareDynamicAttnParamsFromDeviceSequenceState(
                            device_cached_tokens,
                            logical_seq_len,
                            query_rows_for_params,
                            gpuStream(),
                            effective_kv_stride,
                            params_.active_query_rows_device))
                    {
                        LOG_ERROR("[AttentionComputeStage] Failed to derive dynamic attention params from device KV state for layer "
                                  << params_.layer_idx << " on " << params_.device_id.toString());
                        return false;
                    }
                }
            }
            else
            {
                /*
                 * KernelFactory may reuse one attention kernel object across
                 * graph signatures and request lifetimes. An earlier grouped
                 * verifier can therefore leave device-derived M=2..16 params on
                 * the object. Ordinary prefill must explicitly replace that
                 * ownership with its one-row static launch metadata; otherwise
                 * compute_tensor() can consume the stale verifier mode even
                 * though this stage never requested device sequence state.
                 *
                 * During capture, updateDynamicParams() has already prepared
                 * this identical row before beginCapture(), so this call only
                 * validates the existing device buffer and records no duplicate
                 * parameter writer.
                 */
                const int query_rows_for_params =
                    dynamicAttentionParamRows(logical_seq_len, effective_kv_len);
                const int logical_position_offset =
                    std::max(0, effective_kv_len - logical_seq_len);
                if (!kernel->prepareDynamicAttnParams(
                        effective_kv_len,
                        logical_position_offset,
                        query_rows_for_params,
                        gpuStream(),
                        effective_kv_stride))
                {
                    LOG_ERROR("[AttentionComputeStage] Failed to establish static attention params for layer "
                              << params_.layer_idx << " on "
                              << params_.device_id.toString());
                    return false;
                }
            }
        }

        // Get device index using proper ordinal for GPU devices (0-based), not legacy index
        int device_idx = params_.device_id.toKernelDeviceIndex();

        // Decode continuations use the kernel's causal position offset instead
        // of a materialized additive mask. A per-step mask tensor changes shape
        // as KV length grows and would require H2D upload during graph capture.
        ITensor *mask_to_use = params_.workspace_mask;

        if (params_.causal && is_decode_mode)
        {
            mask_to_use = nullptr;
        }

        // Dispatch to kernel's compute method. Decode kernels get the logical
        // query start through setDynamicAttnParams() or canonical kv_len-seq_len
        // geometry.
        const bool kernel_causal = params_.causal;

        // Device-agnostic unified path using compute_tensor()
        // The kernel factory creates the appropriate kernel (CPU or GPU) based on dev_type,
        // and compute_tensor() handles type dispatch internally.
        // Since compute_tensor() now takes ITensor*, we can pass Q/K/V directly without casting.
        // This allows GPU tensor wrappers (like GpuTensorView from CUDA KV cache) to work.

        if (!ensureRequiredPointers("AttentionComputeStage", {
                                                                 {"Q", params_.Q},
                                                                 {"K", effective_K},
                                                                 {"V", effective_V},
                                                                 {"output", params_.output},
                                                             }))
        {
            return false;
        }

        // Device coherence is now handled automatically by DeviceGraphExecutor at stage boundaries
        // based on the stage's coherencePolicy() (FULL by default)

        LOG_TRACE("[AttentionComputeStage] Executing kernel: Q_type=" << params_.Q->dtype_name()
                                                                      << " device=" << params_.device_id.to_string()
                                                                      << " device_idx=" << device_idx);

        // =====================================================================
        // DEBUG: Dump effective K/V to binary files for Python analysis
        // Enable with LLAMINAR_DUMP_EFFECTIVE_KV=1
        // Dumps layer 0 data (or all layers with LLAMINAR_DUMP_EFFECTIVE_KV_ALL=1)
        // =====================================================================
        {
            const auto &env = debugEnv().attention;
            const bool dump_enabled = env.dump_effective_kv;
            const bool dump_all = env.dump_effective_kv_all;

            if (dump_enabled && (dump_all || params_.layer_idx == 0) && is_decode_mode)
            {
                static int dump_iteration = 0;
                const std::string dump_dir = "/tmp/effective_kv_dump/layer" +
                                             std::to_string(params_.layer_idx) +
                                             "_iter" + std::to_string(dump_iteration);
                std::filesystem::create_directories(dump_dir);

                // Write metadata
                {
                    std::ofstream meta(dump_dir + "/meta.txt");
                    meta << "layer=" << params_.layer_idx << "\n"
                         << "iteration=" << dump_iteration << "\n"
                         << "seq_len=" << params_.seq_len << "\n"
                         << "kv_len=" << effective_kv_len << "\n"
                         << "n_heads=" << params_.n_heads << "\n"
                         << "n_kv_heads=" << params_.n_kv_heads << "\n"
                         << "head_dim=" << params_.head_dim << "\n"
                         << "batch_size=" << params_.batch_size << "\n"
                         << "mode=" << attention_mode_name(mode) << "\n"
                         << "Q_type=" << (params_.Q ? params_.Q->dtype_name() : "null") << "\n"
                         << "K_type=" << (effective_K ? effective_K->dtype_name() : "null") << "\n"
                         << "V_type=" << (effective_V ? effective_V->dtype_name() : "null") << "\n"
                         << "K_is_converted=" << (effective_K && dynamic_cast<FP32Tensor *>(effective_K) ? 1 : 0) << "\n"
                         << "V_is_converted=" << (effective_V && dynamic_cast<FP32Tensor *>(effective_V) ? 1 : 0) << "\n"
                         << "K_ptr=" << (void *)effective_K << "\n"
                         << "V_ptr=" << (void *)effective_V << "\n";
                }

                // Dump Q (FP32)
                if (auto *q_fp32 = dynamic_cast<FP32Tensor *>(params_.Q))
                {
                    const size_t q_elems = static_cast<size_t>(params_.seq_len) *
                                           params_.n_heads * params_.head_dim;
                    std::ofstream f(dump_dir + "/Q.bin", std::ios::binary);
                    f.write(reinterpret_cast<const char *>(q_fp32->data()),
                            q_elems * sizeof(float));
                }

                // Dump effective K (should be FP32 after dequant)
                if (auto *k_fp32 = dynamic_cast<FP32Tensor *>(effective_K))
                {
                    const size_t k_elems = static_cast<size_t>(effective_kv_len) *
                                           params_.n_kv_heads * params_.head_dim;
                    std::ofstream f(dump_dir + "/K_effective.bin", std::ios::binary);
                    f.write(reinterpret_cast<const char *>(k_fp32->data()),
                            k_elems * sizeof(float));
                }

                // Dump effective V (should be FP32 after dequant)
                if (auto *v_fp32 = dynamic_cast<FP32Tensor *>(effective_V))
                {
                    const size_t v_elems = static_cast<size_t>(effective_kv_len) *
                                           params_.n_kv_heads * params_.head_dim;
                    std::ofstream f(dump_dir + "/V_effective.bin", std::ios::binary);
                    f.write(reinterpret_cast<const char *>(v_fp32->data()),
                            v_elems * sizeof(float));
                }

                // Dump raw TQ4 K cache if available
                if (params_.kv_cache && params_.layer_idx >= 0)
                {
                    ITensor *cache_k = params_.kv_cache->get_k(params_.layer_idx, 0);
                    if (auto *k_tq4 = dynamic_cast<TQ4Tensor *>(cache_k))
                    {
                        const size_t raw_bytes = static_cast<size_t>(effective_kv_len) *
                                                 k_tq4->blocks_per_row() * k_tq4->block_bytes();
                        std::ofstream f(dump_dir + "/K_cache_tq4.bin", std::ios::binary);
                        f.write(reinterpret_cast<const char *>(k_tq4->typed_data()),
                                raw_bytes);
                        std::ofstream m(dump_dir + "/K_cache_meta.txt");
                        m << "blocks_per_row=" << k_tq4->blocks_per_row() << "\n"
                          << "block_bytes=" << k_tq4->block_bytes() << "\n"
                          << "head_dim=" << k_tq4->head_dim() << "\n"
                          << "rows=" << effective_kv_len << "\n";
                    }
                }

                LOG_DEBUG("[AttentionComputeStage] Dumped effective K/V to " << dump_dir);
                if (params_.layer_idx == 0)
                    dump_iteration++;
            }
        }

        // =====================================================================
        // Fused TQ GPU decode: rotation trick + centroid attention
        // Bypasses normal compute_tensor() since TQ requires ring buffer
        // metadata, rotation matrices, and codebook access.
        // NOTE: Currently slower than dequant+flash path. Enable with
        // LLAMINAR_ENABLE_FUSED_TQ_ATTN=1 for testing.
        // =====================================================================
#if defined(HAVE_CUDA)
        if (is_decode_mode && params_.device_id.is_gpu() && params_.kv_cache &&
            debugEnv().attention.enable_fused_tq_attention)
        {
            const auto kp = params_.kv_cache->k_precision();
            const auto vp = params_.kv_cache->v_precision();
            if (kp == ActivationPrecision::TQ8 && vp == ActivationPrecision::TQ4)
            {
                auto *tq_cache = dynamic_cast<CUDARingKVCacheTQ *>(params_.kv_cache);
                auto *cuda_kernel = dynamic_cast<cuda::CUDAFlashAttentionKernelT<ActivationPrecision::FP32> *>(kernel);
                if (tq_cache && cuda_kernel)
                {
                    const auto &rots = tq_cache->rotations();
                    const size_t layer_offset = static_cast<size_t>(params_.layer_idx) *
                                                rots.n_kv_heads * rots.head_dim * rots.head_dim;
                    const float *R = rots.d_rotations + layer_offset;
                    const float *Rt = rots.d_rotations_t + layer_offset;

                    bool tq_success = cuda_kernel->compute_tensor_tq_decode(
                        params_.Q, params_.output,
                        tq_cache->raw_k_cache(params_.layer_idx, 0),
                        tq_cache->raw_v_cache(params_.layer_idx, 0),
                        R, Rt,
                        params_.batch_size,
                        effective_kv_len,
                        params_.n_heads,
                        params_.n_kv_heads,
                        params_.head_dim,
                        tq_cache->max_seq_len(),
                        tq_cache->ring_tail(params_.layer_idx, 0),
                        static_cast<int>(tq_cache->k_block_size()),
                        static_cast<int>(tq_cache->v_block_size()));

                    if (!tq_success)
                    {
                        LOG_ERROR("[AttentionComputeStage] Fused TQ GPU decode kernel failed");
                        return false;
                    }
                    LOG_TRACE("[AttentionComputeStage] Fused TQ GPU decode for layer " << params_.layer_idx);
                    return true;
                }
            }
        }
#endif

        // =================================================================
        // KV rotation: rotate Q before attention, inverse-rotate output after.
        // When K/V were rotated by ActivationRotation before Q16_1 quantization
        // in KVCacheAppendStage, Q must be rotated by the same matrix so that
        // Q@R @ (K@R)^T = Q@K^T (rotation cancels in the score). The output
        // (sum of alpha_i * V_i@R) must then be inverse-rotated to recover
        // the correct unrotated attention result.
        //
        // During PREFILL, effective_K/V are the raw projection outputs (unrotated),
        // since the cache override is only triggered for decode (kv_len > seq_len).
        // In this case we must also rotate K/V to match the rotated Q.
        // During DECODE, effective_K/V come from the cache (already rotated),
        // so only Q rotation is needed.
        // =================================================================
        const auto *kv_rot = params_.kv_rotation;
        const int q_dim = params_.n_heads * params_.head_dim;
        const int kv_dim = params_.n_kv_heads * params_.head_dim;
        const bool projections_need_rotation = kv_rot &&
                                               (effective_K == params_.K) &&
                                               (effective_V == params_.V);

        if (kv_rot)
        {
            float *q_fp32 = params_.Q->mutable_data();
            if (q_fp32)
            {
                kv_rot->rotate_rows_inplace(
                    q_fp32,
                    params_.batch_size * params_.seq_len,
                    q_dim);
            }

            // Prefill path: K/V are original projections, not from cache.
            // Rotate them so Q_rot @ K_rot^T = Q @ K^T and V_rot gives R*context.
            if (projections_need_rotation)
            {
                float *k_fp32 = effective_K->mutable_data();
                float *v_fp32 = effective_V->mutable_data();
                if (k_fp32)
                    kv_rot->rotate_rows_inplace(k_fp32, params_.seq_len, kv_dim);
                if (v_fp32)
                    kv_rot->rotate_rows_inplace(v_fp32, params_.seq_len, kv_dim);
            }
        }

        if (debugEnv().attention.debug_effective_kv_snapshot &&
            debugEnv().attention.debugEffectiveKVSnapshotLayerSelected(params_.layer_idx))
        {
            /*
             * GPU request gathers are request-major fixed-stride tensors.  A
             * diagnostic snapshot must retain every request bank; recording
             * only `effective_kv_len` rows silently captured request zero and
             * made unequal-length batch failures impossible to localize.
             */
            const size_t effective_request_count =
                gpu_stage ? static_cast<size_t>(std::max(1, params_.batch_size))
                          : size_t{1};
            /*
             * GPU resident views retain one max-sequence-capacity bank per
             * request. Snapshot copy nodes are part of the captured graph, so
             * their byte count must use that immutable physical stride rather
             * than the host logical length observed during capture. Consumers
             * still use device-owned live counts; inactive capacity is copied
             * only for diagnostics and never enters attention arithmetic.
             */
            const size_t diagnostic_kv_stride =
                gpu_stage ? static_cast<size_t>(effective_kv_stride)
                          : static_cast<size_t>(effective_kv_len);
            const size_t k_rows =
                effective_request_count * diagnostic_kv_stride;
            const size_t v_rows =
                effective_request_count * diagnostic_kv_stride;
            // CPU get_kv_converted() shadows are flat [max_seq_len * kv_dim]
            // tensors, while ROCm cache views are [kv_len, kv_dim]. Compare
            // the logical attention layout consumed by the kernel.
            const size_t logical_kv_cols = static_cast<size_t>(params_.n_kv_heads * params_.head_dim);
            const size_t k_cols = effective_K ? logical_kv_cols : 0;
            const size_t v_cols = effective_V ? logical_kv_cols : 0;
            debug_effective_k_tensor_ = effective_K;
            debug_effective_v_tensor_ = effective_V;
            debug_effective_k_rows_ = (effective_K && k_rows > 0 && k_cols > 0) ? k_rows : 0;
            debug_effective_k_cols_ = (effective_K && k_rows > 0 && k_cols > 0) ? k_cols : 0;
            debug_effective_v_rows_ = (effective_V && v_rows > 0 && v_cols > 0) ? v_rows : 0;
            debug_effective_v_cols_ = (effective_V && v_rows > 0 && v_cols > 0) ? v_cols : 0;

            // Snapshot callbacks reuse dump info first built before execute()
            // for coherence. Force post-execute getDumpInfo() to include these
            // just-resolved tensor-backed diagnostic outputs.
            invalidateDumpInfoCache();
        }

        const bool small_verifier_decode =
            is_decode_mode &&
            params_.batch_size == 1 &&
            params_.causal &&
            params_.seq_len > 1 &&
            params_.seq_len <= attention::kMaxGroupedVerifierAttentionRows &&
            effective_kv_len > params_.seq_len;
        bool success = false;
        if (cpu_grouped_request_cache)
        {
            LOG_TRACE("[AttentionComputeStage] Using grouped CPU request-cache attention"
                      << " layer=" << params_.layer_idx
                      << " requests=" << params_.batch_size
                      << " query_rows=" << params_.seq_len);
            success = kernel->compute_request_batch_decode_equivalent(
                params_.Q,
                cpu_grouped_k_views_.data(),
                cpu_grouped_v_views_.data(),
                cpu_grouped_kv_lens_.data(),
                params_.output,
                params_.batch_size,
                params_.seq_len,
                params_.n_heads,
                params_.n_kv_heads,
                params_.head_dim,
                kernel_causal,
                params_.window_size,
                params_.mpi_ctx,
                device_idx,
                params_.head_start,
                params_.gqa_n_rep);
            if (!success)
            {
                LOG_ERROR("[AttentionComputeStage] Backend lacks grouped request-cache decode attention"
                          << " layer=" << params_.layer_idx
                          << " requests=" << params_.batch_size
                          << " device=" << params_.device_id.to_string());
            }
        }
        else if (gpu_grouped_request_decode)
        {
            const int *device_request_counts =
                params_.kv_cache->deviceCachedTokenCountPtr(
                    params_.layer_idx,
                    /*seq_idx=*/0);
            if (!device_request_counts)
            {
                LOG_ERROR("[AttentionComputeStage] Grouped GPU request attention requires canonical device KV counts"
                          << " layer=" << params_.layer_idx
                          << " requests=" << params_.batch_size
                          << " device=" << params_.device_id.to_string());
                return false;
            }

            LOG_TRACE("[AttentionComputeStage] Using device-owned grouped GPU request-cache attention"
                      << " layer=" << params_.layer_idx
                      << " requests=" << params_.batch_size
                      << " query_rows=" << logical_seq_len
                      << " logical_kv_len=" << effective_kv_len
                      << " resident_kv_stride=" << effective_kv_stride);
            success = kernel->compute_device_request_batch_decode_equivalent(
                params_.Q,
                effective_K,
                effective_V,
                device_request_counts,
                params_.output,
                params_.batch_size,
                logical_seq_len,
                effective_kv_stride,
                params_.n_heads,
                params_.n_kv_heads,
                params_.head_dim,
                kernel_causal,
                params_.window_size,
                params_.mpi_ctx,
                device_idx,
                params_.head_start,
                params_.gqa_n_rep);
            if (!success)
            {
                LOG_ERROR("[AttentionComputeStage] Backend lacks grouped device request-cache decode attention"
                          << " layer=" << params_.layer_idx
                          << " requests=" << params_.batch_size
                          << " query_rows=" << logical_seq_len
                          << " device=" << params_.device_id.to_string());
            }
        }
        else if (small_verifier_decode)
        {
            LOG_TRACE("[AttentionComputeStage] Using grouped decode-equivalent verifier attention"
                      << " layer=" << params_.layer_idx
                      << " rows=" << params_.seq_len
                      << " effective_kv_len=" << effective_kv_len
                      << " q_type=" << (params_.Q ? params_.Q->dtype_name() : "null"));
            success = kernel->compute_verifier_rows_decode_equivalent(
                params_.Q,
                effective_K,
                effective_V,
                params_.output,
                params_.seq_len,
                effective_kv_len,
                params_.n_heads,
                params_.n_kv_heads,
                params_.head_dim,
                kernel_causal,
                params_.window_size,
                params_.mpi_ctx,
                device_idx,
                params_.head_start,
                params_.gqa_n_rep);
            if (!success)
            {
                LOG_ERROR("[AttentionComputeStage] Backend lacks grouped decode-equivalent verifier attention"
                          << " layer=" << params_.layer_idx
                          << " rows=" << params_.seq_len
                          << " effective_kv_len=" << effective_kv_len
                          << " device=" << params_.device_id.to_string());
            }
        }
        else
        {
            success = kernel->compute_tensor(
                params_.Q, effective_K, effective_V, params_.output,
                params_.batch_size,
                params_.seq_len,
                effective_kv_len,
                params_.n_heads,
                params_.n_kv_heads,
                params_.head_dim,
                kernel_causal, // Pass false for decode (we built the mask explicitly)
                params_.window_size,
                params_.workspace_scores,
                mask_to_use, // Use our decode mask if we built one
                params_.mpi_ctx,
                device_idx,
                params_.head_start,
                -1, // local_n_heads (n_heads is already local)
                -1, // local_n_kv_heads (n_kv_heads is already local)
                params_.gqa_n_rep);
        }

        if (!success)
        {
            LOG_ERROR("[AttentionComputeStage] Attention kernel execution failed");
            return false;
        }

        if (kv_rot)
        {
            // Inverse-rotate attention output to undo the V rotation.
            // output = sum(alpha_i * V_i@R) → R^T * output = sum(alpha_i * V_i)
            float *out_fp32 = params_.output->mutable_data();
            if (out_fp32)
            {
                kv_rot->inverse_rotate_rows_inplace(
                    out_fp32,
                    params_.batch_size * params_.seq_len,
                    q_dim);
            }

            // Q, K, V do NOT need inverse-rotation:
            // - Q is overwritten by next layer's QKV projection
            // - K/V during prefill: KVCacheAppend already ran (depends on rope/proj),
            //   and no downstream stage reads these buffers after attention
        }

        return true;
    }

    size_t AttentionComputeStage::estimatedFlops() const
    {
        // Attention FLOPs:
        // Q @ K^T: 2 * batch * n_heads * seq_len * kv_len * head_dim
        // softmax: ~4 * batch * n_heads * seq_len * kv_len
        // scores @ V: 2 * batch * n_heads * seq_len * kv_len * head_dim
        const size_t qk_flops = 2ULL * params_.batch_size * params_.n_heads *
                                params_.seq_len * params_.kv_len * params_.head_dim;
        const size_t softmax_flops = 4ULL * params_.batch_size * params_.n_heads *
                                     params_.seq_len * params_.kv_len;
        const size_t sv_flops = qk_flops;
        return qk_flops + softmax_flops + sv_flops;
    }

    size_t AttentionComputeStage::estimatedMemoryBytes() const
    {
        // Workspace for attention scores: n_heads * seq_len * kv_len
        return static_cast<size_t>(params_.n_heads) * params_.seq_len * params_.kv_len * sizeof(float);
    }

    bool AttentionComputeStage::supportsBackend(ComputeBackendType backend) const
    {
        switch (backend)
        {
        case ComputeBackendType::CPU:

            return true;
#if defined(HAVE_CUDA)
        case ComputeBackendType::GPU_CUDA:
            return true;
#endif
#if defined(HAVE_ROCM)
        case ComputeBackendType::GPU_ROCM:
            return true;
#endif
        default:
            return false;
        }
    }

    StageDumpInfo AttentionComputeStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;

        // Input: Q, K, V tensors
        // Q shape: [batch_size * seq_len, n_heads * head_dim]
        // K/V shape: [batch_size * kv_len, n_kv_heads * head_dim]
        const size_t total_q_tokens = static_cast<size_t>(params_.batch_size * params_.seq_len);
        size_t total_kv_tokens = static_cast<size_t>(params_.batch_size * params_.kv_len);
        const ITensor *dump_K = params_.K;
        const ITensor *dump_V = params_.V;

        // Decode graph construction may cache GpuTensorView pointers while the
        // KV cache still has N rows. The immediately preceding kv_append stage
        // can replace those wrappers with N+1-row views, leaving params_.K/V
        // stale before execute() has a chance to re-query the cache. Snapshot
        // and validation metadata must therefore refresh cache-backed K/V views
        // here, mirroring the execution path's live-cache read.
        if (isGraphCaptureActive() && debug_effective_k_tensor_ &&
            debug_effective_v_tensor_)
        {
            /*
             * Snapshot capture records this metadata while the GPU stream is
             * inside beginCapture()/endCapture(). The exact cache-owned views
             * were resolved by execute() and remain allocation-stable, so use
             * them directly. Reading a host token count here would violate the
             * fully device-owned replay contract and can invalidate capture.
             */
            dump_K = debug_effective_k_tensor_;
            dump_V = debug_effective_v_tensor_;
            total_kv_tokens = debug_effective_k_rows_;
        }
        else if (params_.kv_cache && params_.layer_idx >= 0 &&
                 !isGraphCaptureActive())
        {
            const int cached_tokens = params_.kv_cache->get_cached_tokens(params_.layer_idx, 0);
            const bool should_read_cache = cached_tokens > 0 &&
                                           (params_.read_kv_from_cache ||
                                            cached_tokens > params_.seq_len ||
                                            params_.apply_rope_to_k);

            if (should_read_cache)
            {
                ITensor *cache_k = nullptr;
                ITensor *cache_v = nullptr;
                int cache_kv_len = 0;
                if (params_.kv_cache->get_kv(params_.layer_idx, 0, &cache_k, &cache_v, &cache_kv_len) &&
                    cache_k && cache_v && cache_kv_len > 0)
                {
                    dump_K = cache_k;
                    dump_V = cache_v;
                    total_kv_tokens = static_cast<size_t>(params_.batch_size * cache_kv_len);
                }
                else if (cache_kv_len == 0)
                {
                    dump_K = nullptr;
                    dump_V = nullptr;
                    total_kv_tokens = 0;
                }
            }
        }

        if (params_.Q)
        {
            info.addInput("Q", params_.Q, total_q_tokens, params_.n_heads * params_.head_dim);
        }
        if (dump_K)
        {
            info.addInput("K", dump_K, total_kv_tokens, params_.n_kv_heads * params_.head_dim);
        }
        if (dump_V)
        {
            info.addInput("V", dump_V, total_kv_tokens, params_.n_kv_heads * params_.head_dim);
        }

        // Output: attention context
        // Output shape: [batch_size * seq_len, n_heads * head_dim]
        if (params_.output)
        {
            LOG_DEBUG("[AttentionComputeStage::getDumpInfo] output=" << (void *)params_.output
                                                                     << " type=" << params_.output->dtype_name()
                                                                     << " batch_size=" << params_.batch_size
                                                                     << " seq_len=" << params_.seq_len
                                                                     << " total_tokens=" << total_q_tokens
                                                                     << " n_heads*head_dim=" << (params_.n_heads * params_.head_dim));
            // Use ITensor* overload to enable coherence tracking
            info.addOutput("output", params_.output, total_q_tokens, params_.n_heads * params_.head_dim);
        }
        else
        {
            LOG_DEBUG("[AttentionComputeStage::getDumpInfo] output is NULL");
        }

        if (debugEnv().attention.debug_effective_kv_snapshot &&
            debugEnv().attention.debugEffectiveKVSnapshotLayerSelected(params_.layer_idx))
        {
            /*
             * GPU cache/projection buffers are allocated at request capacity,
             * and the captured D2D snapshot node must retain one immutable byte
             * count across warmup, capture, and replay. The live logical token
             * count still governs attention arithmetic through device-owned
             * params; it must not resize diagnostic graph nodes. CPU snapshots
             * remain synchronous and may describe only their logical rows.
             */
            const size_t expected_kv_rows =
                params_.device_id.is_gpu() && params_.kv_cache
                    ? static_cast<size_t>(std::max(1, params_.batch_size)) *
                          static_cast<size_t>(std::max(1, params_.kv_cache->max_seq_len()))
                    : (total_kv_tokens > 0
                           ? total_kv_tokens
                           : static_cast<size_t>(params_.seq_len > 0 ? params_.seq_len : 0));
            const size_t expected_kv_cols = static_cast<size_t>(params_.n_kv_heads * params_.head_dim);
            /*
             * execute() resolves the exact cache-owned view consumed by the
             * attention kernel.  Keep that pointer for both eager and captured
             * snapshot publication.  Falling back to dump_K/dump_V after a
             * captured replay selects sequence zero's raw cache view; labeling
             * that storage as `request_count * max_kv_len` rows makes every
             * later request appear zero even though the grouped kernel consumed
             * the correctly gathered request-major tensor.
             */
            /*
             * GPU effective-K/V diagnostics exist only after execute() resolves
             * the production cache view. Omitting an unresolved diagnostic from
             * pre-execution metadata is intentional: the executor requires a
             * finalized warmup manifest before capture and hard-fails if the
             * post-execute outputs disappear or change. CPU execution remains
             * synchronous and may describe its directly selected dump tensors.
             */
            const ITensor *effective_k_tensor =
                debug_effective_k_tensor_
                    ? debug_effective_k_tensor_
                    : (params_.device_id.is_gpu() ? nullptr : dump_K);
            const ITensor *effective_v_tensor =
                debug_effective_v_tensor_
                    ? debug_effective_v_tensor_
                    : (params_.device_id.is_gpu() ? nullptr : dump_V);
            const size_t effective_k_rows = debug_effective_k_rows_ > 0 ? debug_effective_k_rows_ : expected_kv_rows;
            const size_t effective_v_rows = debug_effective_v_rows_ > 0 ? debug_effective_v_rows_ : expected_kv_rows;
            const size_t effective_k_cols = debug_effective_k_cols_ > 0 ? debug_effective_k_cols_ : expected_kv_cols;
            const size_t effective_v_cols = debug_effective_v_cols_ > 0 ? debug_effective_v_cols_ : expected_kv_cols;
            if (effective_k_tensor && effective_k_rows > 0 && effective_k_cols > 0)
            {
                info.addOutput("effective_k", effective_k_tensor, effective_k_rows, effective_k_cols);
            }
            if (effective_v_tensor && effective_v_rows > 0 && effective_v_cols > 0)
            {
                info.addOutput("effective_v", effective_v_tensor, effective_v_rows, effective_v_cols);
            }
            for (size_t request = 0;
                 request < debug_device_kv_count_views_.size();
                 ++request)
            {
                info.addOutput(
                    debug_device_kv_count_names_[request].c_str(),
                    debug_device_kv_count_views_[request].get(),
                    /*rows=*/1,
                    /*cols=*/1);
                info.addOutput(
                    debug_device_kv_head_names_[request].c_str(),
                    debug_device_kv_head_views_[request].get(),
                    /*rows=*/1,
                    /*cols=*/1);
            }
        }

        // Scalars capture all necessary info for debugging
        info.addScalarInt("batch_size", params_.batch_size);
        info.addScalarInt("seq_len", params_.seq_len);
        info.addScalarInt("kv_len", params_.kv_len);
        info.addScalarInt("layer_idx", params_.layer_idx);
        info.addScalarInt("n_heads", params_.n_heads);
        info.addScalarInt("n_kv_heads", params_.n_kv_heads);
        info.addScalarInt("head_dim", params_.head_dim);
        info.addScalarBool("causal", params_.causal);
        info.addScalarInt("window_size", params_.window_size);
        info.addScalarInt("device_id", params_.device_id.toKernelDeviceIndex());

        // Add attention mode info (as int - PREFILL=0, DECODE=1, BATCHED_DECODE=2, CHUNKED_PREFILL=3)
        AttentionMode mode = params_.auto_detect_mode
                                 ? detect_attention_mode(params_.batch_size, params_.seq_len, params_.kv_len)
                                 : params_.attention_mode;
        info.addScalarInt("attention_mode", static_cast<int>(mode));
        info.addScalarBool("auto_detect_mode", params_.auto_detect_mode);

        return info;
    }

    StageBufferRequirements AttentionComputeStage::getBufferRequirements() const
    {
        StageBufferRequirements reqs;

        // Input: Q (query)
        if (params_.Q)
        {
            const size_t q_rows = static_cast<size_t>(params_.batch_size * params_.seq_len);
            const size_t q_cols = static_cast<size_t>(params_.n_heads * params_.head_dim);
            BufferTensorType buf_type = toBufferTensorType(params_.Q->native_type());
            reqs.addInput("Q", {q_rows, q_cols}, buf_type);
        }

        // Input: K (key - may have different kv_len than Q's seq_len)
        if (params_.K)
        {
            const size_t k_rows = static_cast<size_t>(params_.batch_size * params_.kv_len);
            const size_t k_cols = static_cast<size_t>(params_.n_kv_heads * params_.head_dim);
            BufferTensorType buf_type = toBufferTensorType(params_.K->native_type());
            reqs.addInput("K", {k_rows, k_cols}, buf_type);
        }

        // Input: V (value)
        if (params_.V)
        {
            const size_t v_rows = static_cast<size_t>(params_.batch_size * params_.kv_len);
            const size_t v_cols = static_cast<size_t>(params_.n_kv_heads * params_.head_dim);
            BufferTensorType buf_type = toBufferTensorType(params_.V->native_type());
            reqs.addInput("V", {v_rows, v_cols}, buf_type);
        }

        // Output: attention output
        if (params_.output)
        {
            const size_t out_rows = static_cast<size_t>(params_.batch_size * params_.seq_len);
            const size_t out_cols = static_cast<size_t>(params_.n_heads * params_.head_dim);
            BufferTensorType buf_type = toBufferTensorType(params_.output->native_type());
            reqs.addOutput("output", {out_rows, out_cols}, buf_type);
        }

        // Scratch: workspace buffers (if pre-allocated)
        if (params_.workspace_scores)
        {
            reqs.addScratch("workspace_scores", params_.workspace_scores->shape(),
                            toBufferTensorType(params_.workspace_scores->native_type()));
        }
        if (params_.workspace_context)
        {
            reqs.addScratch("workspace_context", params_.workspace_context->shape(),
                            toBufferTensorType(params_.workspace_context->native_type()));
        }

        return reqs;
    }

    StageBufferContract AttentionComputeStage::bufferContract() const
    {
        if (!params_.q_buffer_id || !params_.output_buffer_id)
            return {};

        auto contract = StageBufferContract::build()
                            .addInput(*params_.q_buffer_id)
                            .addOutput(*params_.output_buffer_id);

        if (params_.workspace_scores_buffer_id)
            contract.addInOut(*params_.workspace_scores_buffer_id);
        if (params_.workspace_context_buffer_id)
            contract.addInOut(*params_.workspace_context_buffer_id);

        return contract;
    }

} // namespace llaminar2
