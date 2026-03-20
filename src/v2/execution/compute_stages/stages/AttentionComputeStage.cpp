/**
 * @file AttentionComputeStage.cpp
 * @brief Implementation of AttentionComputeStage
 */

#include "AttentionComputeStage.h"
#include "../ComputeStageUtils.h"
#include "../../../utils/DebugEnv.h"
#include "../../../tensors/Tensors.h"
#include "../../../tensors/TensorKernels.h"
#include "../../../kernels/cpu/CPUKVCache.h"
#include "../../../utils/Logger.h"
#include "../../../kernels/KernelFactory.h"
#include <limits>

namespace llaminar2
{

    // =============================================================================
    // AttentionComputeStage Implementation
    // =============================================================================

    AttentionComputeStage::AttentionComputeStage(Params params)
        : IComputeStage(params.device_id), params_(std::move(params))
    {
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

        LOG_DEBUG("[AttentionComputeStage::getOrCreateKernel] Created and cached attention kernel for "
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

    bool AttentionComputeStage::execute(IDeviceContext *ctx)
    {
        // Dynamic kv_len: query from KV cache at execution time if available
        // This enables declarative graph construction where the stage runs after
        // KVCacheAppendStage has already appended tokens
        int effective_kv_len = params_.kv_len;
        if (params_.kv_cache && params_.layer_idx >= 0)
        {
            effective_kv_len = params_.kv_cache->get_cached_tokens(params_.layer_idx, 0);
            if (effective_kv_len == 0)
            {
                effective_kv_len = params_.seq_len; // Prefill case
            }
            LOG_TRACE("[AttentionComputeStage] Dynamic kv_len from cache: " << effective_kv_len
                                                                            << " (static was: " << params_.kv_len << ")");
        }

        // Read K/V from cache at execution time when requested.
        // This allows GPU prefill to use the FP16 tensors in the KV cache
        // (populated by KVCacheAppendStage) instead of the Q8_1 projection
        // buffers, eliminating the Q8_1→FP32→FP16 triple conversion.
        ITensor *effective_K = params_.K;
        ITensor *effective_V = params_.V;
        if (params_.kv_cache && params_.layer_idx >= 0)
        {
            // Always override K/V from KV cache when:
            // 1. read_kv_from_cache is set (GPU optimization for type conversion), OR
            // 2. We're in decode mode (effective_kv_len > seq_len) - the graph may have
            //    been built during prefill (cached_tokens=0) with K/V wired to activation
            //    scratch buffers. During decode, those buffers only hold the current token's
            //    projection. The full KV history lives in the cache (populated by
            //    KVCacheAppendStage which runs before this stage).
            if (params_.read_kv_from_cache || effective_kv_len > params_.seq_len)
            {
                ITensor *cache_k = params_.kv_cache->get_k(params_.layer_idx, 0);
                ITensor *cache_v = params_.kv_cache->get_v(params_.layer_idx, 0);
                if (cache_k && cache_v)
                {
                    effective_K = cache_k;
                    effective_V = cache_v;
                    LOG_TRACE("[AttentionComputeStage] Using cache K/V ("
                              << cache_k->dtype_name() << ") for layer " << params_.layer_idx);
                }
                else
                {
                    LOG_TRACE("[AttentionComputeStage] Cache K/V not available yet, using wired tensors");
                }
            }
        }

        // Detect attention mode if auto-detection enabled
        AttentionMode mode = params_.attention_mode;
        if (params_.auto_detect_mode)
        {
            mode = detect_attention_mode(params_.batch_size, params_.seq_len, effective_kv_len);
        }

        LOG_DEBUG("[AttentionComputeStage] Execute: batch=" << params_.batch_size
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

        // Get device index using proper ordinal for GPU devices (0-based), not legacy index
        int device_idx = params_.device_id.toKernelDeviceIndex();

        // Build proper causal mask for decode mode
        // In decode mode (seq_len < kv_len), we need to account for position offset
        // The kernel's internal causal mask assumes m=0 for decode (query position 0),
        // but decode tokens should be able to attend to all cached positions [0, kv_len-1]
        //
        // Key insight: For decode with seq_len=1, the query is at position (kv_len-1),
        // so it should attend to ALL kv_len positions. The kernel's "n > m" check would
        // only allow attending to position 0, which is wrong.
        std::unique_ptr<FP32Tensor> decode_mask;
        ITensor *mask_to_use = params_.workspace_mask;

        const bool is_decode_mode = (mode == AttentionMode::DECODE ||
                                     (params_.seq_len < effective_kv_len && params_.batch_size == 1));

        if (params_.causal && is_decode_mode)
        {
            if (gpuStream() == nullptr)
            {
                // Direct execution on default stream: safe to use temporary CPU mask
                // (will survive until kernel returns synchronously)
                const int base_pos = (params_.position_offset > 0)
                                         ? params_.position_offset
                                         : (effective_kv_len - params_.seq_len);

                decode_mask = std::make_unique<FP32Tensor>(
                    std::vector<size_t>{static_cast<size_t>(params_.seq_len * effective_kv_len)});
                float *mask_data = decode_mask->mutable_data();

                for (int q = 0; q < params_.seq_len; ++q)
                {
                    const int q_pos = base_pos + q; // Global position of this query
                    for (int k = 0; k < effective_kv_len; ++k)
                    {
                        // Causal: Query at position q_pos can attend to K positions [0, q_pos]
                        mask_data[q * effective_kv_len + k] = (k <= q_pos)
                                                                  ? 0.0f
                                                                  : -std::numeric_limits<float>::infinity();
                    }
                }

                mask_to_use = decode_mask.get();
                LOG_DEBUG("[AttentionComputeStage] Built decode causal mask: base_pos=" << base_pos
                                                                                        << " seq_len=" << params_.seq_len << " kv_len=" << effective_kv_len);
            }
            else
            {
                // Non-default stream (graph capture or manual replay on capture stream):
                // Cannot use temporary CPU mask — it would be destroyed after execute(),
                // leaving captured graph with a dangling pointer.
                // For decode (seq_len=1), causal=false + mask=nullptr = attend to all
                // positions, which is correct since the single query token should
                // attend to the entire KV cache.
                mask_to_use = nullptr;
                LOG_DEBUG("[AttentionComputeStage] Decode on non-default stream: using nullptr mask (attend-to-all)");
            }
        }

        // Dispatch to kernel's compute method
        // IMPORTANT: For decode with explicit mask, we pass causal=false to avoid
        // double-masking (kernel would apply "n > m" on top of our mask)
        const bool kernel_causal = params_.causal && !is_decode_mode;

        // Device-agnostic unified path using compute_tensor()
        // The kernel factory creates the appropriate kernel (CPU or GPU) based on dev_type,
        // and compute_tensor() handles type dispatch internally.
        // Since compute_tensor() now takes ITensor*, we can pass Q/K/V directly without casting.
        // This allows GPU tensor wrappers (like GpuTensorView from CUDA KV cache) to work.

        if (!ensureRequiredPointers("AttentionComputeStage", {
                                                                 {"Q", params_.Q},
                                                                 {"K", params_.K},
                                                                 {"V", params_.V},
                                                                 {"output", params_.output},
                                                             }))
        {
            return false;
        }

        // Device coherence is now handled automatically by DeviceGraphExecutor at stage boundaries
        // based on the stage's coherencePolicy() (FULL by default)

        LOG_DEBUG("[AttentionComputeStage] Executing kernel: Q_type=" << params_.Q->dtype_name()
                                                                      << " device=" << params_.device_id.to_string()
                                                                      << " device_idx=" << device_idx);

        bool success = kernel->compute_tensor(
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
            device_idx);

        // Device coherence (mark_device_dirty) is now handled automatically by DeviceGraphExecutor
        // at stage boundaries based on the stage's coherencePolicy() (FULL by default)

        if (!success)
        {
            LOG_ERROR("[AttentionComputeStage] Kernel compute_tensor() failed");
            return false;
        }

        LOG_DEBUG("[AttentionComputeStage] Execute complete (mode=" << attention_mode_name(mode) << ")");
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
        const size_t total_kv_tokens = static_cast<size_t>(params_.batch_size * params_.kv_len);

        if (params_.Q)
        {
            info.addInput("Q", params_.Q, total_q_tokens, params_.n_heads * params_.head_dim);
        }
        if (params_.K)
        {
            info.addInput("K", params_.K, total_kv_tokens, params_.n_kv_heads * params_.head_dim);
        }
        if (params_.V)
        {
            info.addInput("V", params_.V, total_kv_tokens, params_.n_kv_heads * params_.head_dim);
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

        // Scalars capture all necessary info for debugging
        info.addScalarInt("batch_size", params_.batch_size);
        info.addScalarInt("seq_len", params_.seq_len);
        info.addScalarInt("kv_len", params_.kv_len);
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
