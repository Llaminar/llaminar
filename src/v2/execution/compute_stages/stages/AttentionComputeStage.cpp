/**
 * @file AttentionComputeStage.cpp
 * @brief Implementation of AttentionComputeStage
 */

#include "AttentionComputeStage.h"
#include "../ComputeStageUtils.h"
#include "../../../utils/DebugEnv.h"
#include "../../../tensors/Tensors.h"
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
        : params_(std::move(params)) {}

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
                                                            << " K_type=" << (params_.K ? params_.K->dtype_name() : "null")
                                                            << " output=" << (void *)params_.output);

        // Validate inputs
        if (!params_.Q || !params_.K || !params_.V || !params_.output)
        {
            LOG_ERROR("[AttentionComputeStage] Null tensor pointers");
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

        // Determine device type: check tensor location FIRST, then fall back to context
        using DeviceType = llaminar::v2::kernels::DeviceType;
        DeviceType dev_type = DeviceType::CPU;
        const bool is_gpu_tensor = params_.Q->is_on_gpu();

        if (is_gpu_tensor)
        {
#if defined(HAVE_CUDA)
            dev_type = DeviceType::CUDA;
#elif defined(HAVE_ROCM)
            dev_type = DeviceType::ROCm;
#endif
        }
        else if (ctx)
        {
            auto *gpu_ctx = dynamic_cast<IGPUDeviceContext *>(ctx);
            if (gpu_ctx)
            {
#if defined(HAVE_CUDA)
                dev_type = DeviceType::CUDA;
#elif defined(HAVE_ROCM)
                dev_type = DeviceType::ROCm;
#endif
            }
        }

        // Create attention kernel via KernelFactory using ITensor* overload (supports both CPU and GPU)
        auto kernel = llaminar::v2::kernels::KernelFactory::createAttention(params_.Q, dev_type);
        if (!kernel)
        {
            LOG_ERROR("[AttentionComputeStage] Failed to create attention kernel for tensor type "
                      << params_.Q->dtype_name());
            return false;
        }

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
        TensorBase *mask_to_use = asTensorBase(params_.workspace_mask, "workspace_mask");

        const bool is_decode_mode = (mode == AttentionMode::DECODE ||
                                     (params_.seq_len < effective_kv_len && params_.batch_size == 1));

        if (params_.causal && is_decode_mode)
        {
            // Build decode-specific causal mask
            // For decode: seq_len=1 (or small), kv_len = full cache length
            // Query at position i (within seq_len) corresponds to global position (base_pos + i)
            // where base_pos = position_offset if provided, else (kv_len - seq_len)
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

        // Dispatch to kernel's compute method
        // IMPORTANT: For decode with explicit mask, we pass causal=false to avoid
        // double-masking (kernel would apply "n > m" on top of our mask)
        const bool kernel_causal = params_.causal && !is_decode_mode;

        bool success = false;

        if (is_gpu_tensor)
        {
            // GPU path: use active_data_ptr() which returns GPU pointer when tensor is on GPU
            const float *Q_ptr = static_cast<const float *>(params_.Q->active_data_ptr());
            const float *K_ptr = static_cast<const float *>(params_.K->active_data_ptr());
            const float *V_ptr = static_cast<const float *>(params_.V->active_data_ptr());
            float *output_ptr = static_cast<float *>(params_.output->active_mutable_data_ptr());

            LOG_DEBUG("[AttentionComputeStage] GPU path: Q=" << (void *)Q_ptr
                                                             << " K=" << (void *)K_ptr
                                                             << " V=" << (void *)V_ptr
                                                             << " output=" << (void *)output_ptr);

            // For decode mode (seq_len != kv_len), use compute_decode()
            // For prefill mode (seq_len == kv_len), use compute()
            if (is_decode_mode)
            {
                // Use compute_decode() which properly handles different seq_len and kv_len
                success = kernel->compute_decode(
                    Q_ptr, K_ptr, V_ptr, output_ptr,
                    params_.seq_len,
                    effective_kv_len,
                    params_.n_heads,
                    params_.n_kv_heads,
                    params_.head_dim,
                    kernel_causal,
                    params_.position_offset);
            }
            else
            {
                // Prefill: seq_len == kv_len, use standard compute()
                success = kernel->compute(
                    Q_ptr, K_ptr, V_ptr, output_ptr,
                    params_.seq_len,
                    params_.n_heads,
                    params_.n_kv_heads,
                    params_.head_dim,
                    kernel_causal,
                    params_.window_size,
                    nullptr, nullptr, nullptr, nullptr, // workspaces not used for GPU flash attention
                    false,
                    params_.mpi_ctx,
                    device_idx);
            }
        }
        else
        {
            // CPU path: use TensorBase* via compute_tensor()
            auto *Q_base = asTensorBase(params_.Q, "Q");
            auto *K_base = asTensorBase(params_.K, "K");
            auto *V_base = asTensorBase(params_.V, "V");
            auto *output_base = asTensorBase(params_.output, "output");

            if (!Q_base || !K_base || !V_base || !output_base)
            {
                LOG_ERROR("[AttentionComputeStage] CPU path requires TensorBase tensors");
                return false;
            }

            // Cast workspace_scores to TensorBase* (can be null)
            auto *workspace_scores_base = asTensorBase(params_.workspace_scores, "workspace_scores");

            success = kernel->compute_tensor(
                Q_base, K_base, V_base, output_base,
                params_.batch_size,
                params_.seq_len,
                effective_kv_len,
                params_.n_heads,
                params_.n_kv_heads,
                params_.head_dim,
                kernel_causal, // Pass false for decode (we built the mask explicitly)
                params_.window_size,
                workspace_scores_base,
                mask_to_use, // Use our decode mask if we built one
                params_.mpi_ctx,
                device_idx);
        }

        if (!success)
        {
            LOG_ERROR("[AttentionComputeStage] Kernel compute_tensor() failed");
            return false;
        }

        // Debug: dump first few output values
        if (params_.output)
        {
            const float *out_data = getSafeFp32Data(params_.output);
            if (out_data)
            {
                LOG_DEBUG("[AttentionComputeStage] output=" << (void *)params_.output
                                                            << " Q_type=" << (params_.Q ? params_.Q->dtype_name() : "null")
                                                            << " out[0:8]=" << out_data[0] << "," << out_data[1] << "," << out_data[2] << "," << out_data[3]
                                                            << "," << out_data[4] << "," << out_data[5] << "," << out_data[6] << "," << out_data[7]);
            }
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

    StageDumpInfo AttentionComputeStage::getDumpInfo() const
    {
        StageDumpInfo info;

        // Output: attention context
        // Output shape: [batch_size * seq_len, n_heads * head_dim]
        if (params_.output)
        {
            const float *out_data = getSafeFp32Data(params_.output);
            const size_t total_tokens = static_cast<size_t>(params_.batch_size * params_.seq_len);
            LOG_DEBUG("[AttentionComputeStage::getDumpInfo] output=" << (void *)params_.output
                                                                     << " type=" << params_.output->dtype_name()
                                                                     << " out_data=" << (void *)out_data
                                                                     << " batch_size=" << params_.batch_size
                                                                     << " seq_len=" << params_.seq_len
                                                                     << " total_tokens=" << total_tokens
                                                                     << " n_heads*head_dim=" << (params_.n_heads * params_.head_dim));
            if (out_data)
            {
                info.addOutput("output",
                               out_data,
                               total_tokens,
                               params_.n_heads * params_.head_dim);
            }
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

} // namespace llaminar2
