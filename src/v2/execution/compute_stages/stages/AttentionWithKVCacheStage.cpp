/**
 * @file AttentionWithKVCacheStage.cpp
 * @brief Implementation of AttentionWithKVCacheStage
 */

#include "AttentionWithKVCacheStage.h"
#include "../../../utils/DebugEnv.h"
#include "../../../tensors/Tensors.h"
#include "../../../utils/Logger.h"
#include "../../../kernels/KernelFactory.h"
#include "../../../kernels/cpu/CPUKVCache.h"
#include <limits>

namespace llaminar2
{

    // =============================================================================
    // AttentionWithKVCacheStage Implementation
    // =============================================================================

    AttentionWithKVCacheStage::AttentionWithKVCacheStage(Params params)
        : IComputeStage(params.device_id)
        , params_(std::move(params))
    {
    }

    ITensorAttention *AttentionWithKVCacheStage::getOrCreateKernel(const ITensor *tensor)
    {
        if (!tensor)
        {
            LOG_ERROR("[AttentionWithKVCacheStage::getOrCreateKernel] Null tensor");
            return nullptr;
        }

        return getOrRefreshKernelByTensorType(
            cached_kernel_,
            cached_kernel_tensor_type_,
            tensor,
            [&]()
            {
                return llaminar::v2::kernels::KernelFactory::getOrCreateAttention(tensor, params_.device_id);
            });
    }

    AttentionWithKVCacheStage::Mode AttentionWithKVCacheStage::effectiveMode() const
    {
        if (params_.mode != Mode::AUTO)
        {
            return params_.mode;
        }

        // Auto-detect based on seq_len and cache state
        if (params_.batch_size > 1 && params_.sequence_lengths != nullptr)
        {
            return Mode::BATCHED;
        }

        // Single sequence: check if decode (single token) vs prefill
        if (params_.seq_len == 1)
        {
            // Single token with cache -> decode mode
            if (params_.kv_cache != nullptr)
            {
                int cached = params_.kv_cache->get_cached_tokens(params_.layer_idx, 0);
                if (cached > 0)
                {
                    return Mode::DECODE;
                }
            }
        }

        return Mode::PREFILL;
    }

    bool AttentionWithKVCacheStage::execute(IDeviceContext *ctx)
    {
        if (!ensureContext(ctx, "AttentionWithKVCacheStage"))
        {
            return false;
        }

            if (!ensureRequiredPointers("AttentionWithKVCacheStage", {
                                                                  {"Q", params_.Q},
                                                                  {"K", params_.K},
                                                                  {"V", params_.V},
                                                                  {"output", params_.output},
                                                              }))
            {
            return false;
        }

        Mode mode = effectiveMode();
        LOG_DEBUG("[AttentionWithKVCacheStage] Execute mode=" << static_cast<int>(mode)
                                                              << " seq_len=" << params_.seq_len << " batch=" << params_.batch_size
                                                              << " layer=" << params_.layer_idx << " heads=" << params_.n_heads
                                                              << " kv_heads=" << params_.n_kv_heads << " head_dim=" << params_.head_dim);

        switch (mode)
        {
        case Mode::PREFILL:
            return executePrefill(ctx);
        case Mode::DECODE:
            return executeDecode(ctx);
        case Mode::BATCHED:
            return executeBatched(ctx);
        case Mode::AUTO:
            // Should never reach here
            LOG_ERROR("[AttentionWithKVCacheStage] AUTO mode not resolved");
            return false;
        }

        return false;
    }

    bool AttentionWithKVCacheStage::executePrefill(IDeviceContext *ctx)
    {
        (void)ctx; // Device context for future GPU support

        LOG_DEBUG("[AttentionWithKVCacheStage::executePrefill] layer=" << params_.layer_idx
                                                                       << " seq_len=" << params_.seq_len);

        // Cast ITensor* to TensorBase* for CPU operations
        auto *Q_base = requireTensorBasePtr(params_.Q, "Q");
        auto *K_base = requireTensorBasePtr(params_.K, "K");
        auto *V_base = requireTensorBasePtr(params_.V, "V");
        auto *output_base = requireTensorBasePtr(params_.output, "output");
        if (!Q_base || !K_base || !V_base || !output_base)
        {
            LOG_ERROR("[AttentionWithKVCacheStage::executePrefill] GPU tensors not yet supported");
            return false;
        }

        // Step 1: Append new K/V to cache (if cache provided)
        if (params_.kv_cache)
        {
            bool append_ok = params_.kv_cache->append(
                params_.layer_idx, 0, K_base, V_base, params_.seq_len);
            if (!append_ok)
            {
                LOG_ERROR("[AttentionWithKVCacheStage] Failed to append K/V to cache");
                return false;
            }
            LOG_DEBUG("[AttentionWithKVCacheStage] Appended " << params_.seq_len
                                                              << " tokens to layer " << params_.layer_idx << " cache");
        }

        // Step 2: Get full K/V from cache (includes just-appended tokens)
        TensorBase *K_full = K_base;
        TensorBase *V_full = V_base;
        int kv_len = params_.seq_len;

        if (params_.kv_cache)
        {
            // Use cached K/V (will include all tokens including just-appended)
            K_full = dynamic_cast<TensorBase *>(params_.kv_cache->get_k(params_.layer_idx, 0));
            V_full = dynamic_cast<TensorBase *>(params_.kv_cache->get_v(params_.layer_idx, 0));
            kv_len = params_.kv_cache->get_cached_tokens(params_.layer_idx, 0);
            LOG_DEBUG("[AttentionWithKVCacheStage] Using cached K/V with " << kv_len << " tokens");
        }

        // Step 3: Create tensor views with actual seq_len dimensions
        // This is critical because activation buffers are allocated at max_seq_len
        // but MpiAttentionOrchestrator validates tensor shapes match
        int q_dim = params_.n_heads * params_.head_dim;
        int kv_dim = params_.n_kv_heads * params_.head_dim;

        auto Q_view = Q_base->create_view({static_cast<size_t>(params_.seq_len), static_cast<size_t>(q_dim)});
        auto K_view = K_full->create_view({static_cast<size_t>(kv_len), static_cast<size_t>(kv_dim)});
        auto V_view = V_full->create_view({static_cast<size_t>(kv_len), static_cast<size_t>(kv_dim)});
        auto out_view = output_base->create_view({static_cast<size_t>(params_.seq_len), static_cast<size_t>(q_dim)});

        if (!Q_view || !K_view || !V_view || !out_view)
        {
            LOG_ERROR("[AttentionWithKVCacheStage] Failed to create tensor views");
            return false;
        }

        // Step 4: Get attention kernel via device-scoped KernelFactory cache
        auto *attention_kernel = getOrCreateKernel(Q_view.get());

        if (!attention_kernel)
        {
            LOG_ERROR("[AttentionWithKVCacheStage::executePrefill] Failed to create attention kernel");
            return false;
        }
        bindStageStream(attention_kernel);

        // Step 5: Allocate workspace for attention scores [n_heads, seq_len, kv_len]
        FP32Tensor scores_workspace({static_cast<size_t>(params_.n_heads * params_.seq_len * kv_len)});

        // Step 6: Build causal mask if needed
        std::unique_ptr<FP32Tensor> mask_tensor;
        if (params_.causal)
        {
            mask_tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(params_.seq_len * kv_len)});
            float *mask_data = mask_tensor->mutable_data();
            for (int q = 0; q < params_.seq_len; ++q)
            {
                for (int k = 0; k < kv_len; ++k)
                {
                    // Causal: Q at position q can attend to K at positions [0, q]
                    mask_data[q * kv_len + k] = (k <= q) ? 0.0f : -std::numeric_limits<float>::infinity();
                }
            }
        }

        // Step 7: Dispatch via compute_tensor
        bool success = attention_kernel->compute_tensor(
            Q_view.get(), K_view.get(), V_view.get(), out_view.get(),
            params_.batch_size,
            params_.seq_len, // Query sequence length
            kv_len,          // Key/value sequence length
            params_.n_heads,
            params_.n_kv_heads,
            params_.head_dim,
            params_.causal,
            params_.window_size,
            &scores_workspace,
            mask_tensor.get(),
            params_.mpi_ctx_shared.get(),
            params_.device_id.toKernelDeviceIndex(),
            params_.head_start,
            params_.local_n_heads,
            params_.local_n_kv_heads);

        if (!success)
        {
            LOG_ERROR("[AttentionWithKVCacheStage::executePrefill] Attention kernel compute_tensor failed");
            return false;
        }

        LOG_DEBUG("[AttentionWithKVCacheStage::executePrefill] Complete");
        return true;
    }

    bool AttentionWithKVCacheStage::executeDecode(IDeviceContext *ctx)
    {
        (void)ctx;

        LOG_DEBUG("[AttentionWithKVCacheStage::executeDecode] layer=" << params_.layer_idx
                                                                      << " position=" << params_.position_offset);

        if (!params_.kv_cache)
        {
            LOG_ERROR("[AttentionWithKVCacheStage::executeDecode] Decode mode requires KV cache");
            return false;
        }

        // Cast ITensor* to TensorBase* for CPU operations
        auto *Q_base = requireTensorBasePtr(params_.Q, "Q");
        auto *K_base = requireTensorBasePtr(params_.K, "K");
        auto *V_base = requireTensorBasePtr(params_.V, "V");
        auto *output_base = requireTensorBasePtr(params_.output, "output");
        if (!Q_base || !K_base || !V_base || !output_base)
        {
            LOG_ERROR("[AttentionWithKVCacheStage::executeDecode] GPU tensors not yet supported");
            return false;
        }

        // Step 1: Append single token K/V to cache
        bool append_ok = params_.kv_cache->append(
            params_.layer_idx, 0, K_base, V_base, 1);
        if (!append_ok)
        {
            LOG_ERROR("[AttentionWithKVCacheStage] Failed to append decode token to cache");
            return false;
        }

        // Step 2: Get full cached K/V
        TensorBase *K_cached = dynamic_cast<TensorBase *>(params_.kv_cache->get_k(params_.layer_idx, 0));
        TensorBase *V_cached = dynamic_cast<TensorBase *>(params_.kv_cache->get_v(params_.layer_idx, 0));
        int kv_len = params_.kv_cache->get_cached_tokens(params_.layer_idx, 0);

        LOG_DEBUG("[AttentionWithKVCacheStage::executeDecode] Attending to " << kv_len << " cached tokens");

        // Step 3: Create tensor views with actual dimensions
        // Q is single token (decode), K/V is full cache length
        int q_dim = params_.n_heads * params_.head_dim;
        int kv_dim = params_.n_kv_heads * params_.head_dim;
        const int q_seq_len = 1; // Decode mode: single query token

        auto Q_view = Q_base->create_view({static_cast<size_t>(q_seq_len), static_cast<size_t>(q_dim)});
        auto K_view = K_cached->create_view({static_cast<size_t>(kv_len), static_cast<size_t>(kv_dim)});
        auto V_view = V_cached->create_view({static_cast<size_t>(kv_len), static_cast<size_t>(kv_dim)});
        auto out_view = output_base->create_view({static_cast<size_t>(q_seq_len), static_cast<size_t>(q_dim)});

        if (!Q_view || !K_view || !V_view || !out_view)
        {
            LOG_ERROR("[AttentionWithKVCacheStage::executeDecode] Failed to create tensor views");
            return false;
        }

        // Step 4: Get attention kernel via device-scoped KernelFactory cache.
        // This uses the typed kernel path which properly handles decode mode where Q.seq_len != K.seq_len.
        auto *attention_kernel = getOrCreateKernel(Q_view.get());

        if (!attention_kernel)
        {
            LOG_ERROR("[AttentionWithKVCacheStage::executeDecode] Failed to create attention kernel");
            return false;
        }
        bindStageStream(attention_kernel);

        // Step 5: Allocate workspace for attention scores [n_heads, q_seq_len, kv_len]
        // Use simple allocation - in production, use pre-allocated workspace
        FP32Tensor scores_workspace({static_cast<size_t>(params_.n_heads * q_seq_len * kv_len)});

        // Step 6: Build causal mask for decode
        // For decode: Q is at position kv_len-1 (end of sequence), attends to [0, kv_len-1]
        FP32Tensor mask_tensor({static_cast<size_t>(q_seq_len * kv_len)});
        float *mask_data = mask_tensor.mutable_data();

        for (int q = 0; q < q_seq_len; ++q)
        {
            int q_pos = (kv_len - q_seq_len) + q; // Q position in full sequence
            for (int k = 0; k < kv_len; ++k)
            {
                // Causal: Q at position q_pos can attend to K at positions [0, q_pos]
                mask_data[q * kv_len + k] = (k <= q_pos) ? 0.0f : -std::numeric_limits<float>::infinity();
            }
        }

        // Step 7: Dispatch via compute_tensor which handles decode mode (q_seq_len != kv_len)
        bool success = attention_kernel->compute_tensor(
            Q_view.get(), K_view.get(), V_view.get(), out_view.get(),
            params_.batch_size,
            q_seq_len, // Query sequence length (1 for decode)
            kv_len,    // Key/value sequence length (full cache)
            params_.n_heads,
            params_.n_kv_heads,
            params_.head_dim,
            true, // causal
            -1,   // window_size (disabled)
            &scores_workspace,
            &mask_tensor,
            params_.mpi_ctx_shared.get(),
            params_.device_id.toKernelDeviceIndex());

        if (!success)
        {
            LOG_ERROR("[AttentionWithKVCacheStage::executeDecode] Attention kernel compute_tensor failed");
            return false;
        }

        LOG_DEBUG("[AttentionWithKVCacheStage::executeDecode] Complete");
        return true;
    }

    bool AttentionWithKVCacheStage::executeBatched(IDeviceContext *ctx)
    {
        (void)ctx;

        LOG_DEBUG("[AttentionWithKVCacheStage::executeBatched] layer=" << params_.layer_idx
                                                                       << " batch_size=" << params_.batch_size << " seq_len=" << params_.seq_len
                                                                       << " sequence_lengths=" << (params_.sequence_lengths ? "valid" : "nullptr")
                                                                       << (params_.sequence_lengths ? " [0]=" + std::to_string((*params_.sequence_lengths)[0]) : ""));

        // Cast ITensor* to TensorBase* for CPU operations
        auto *Q_base = requireTensorBasePtr(params_.Q, "Q");
        auto *K_base = requireTensorBasePtr(params_.K, "K");
        auto *V_base = requireTensorBasePtr(params_.V, "V");
        auto *output_base = requireTensorBasePtr(params_.output, "output");
        if (!Q_base || !K_base || !V_base || !output_base)
        {
            LOG_ERROR("[AttentionWithKVCacheStage::executeBatched] GPU tensors not yet supported");
            return false;
        }

        // For batched mode with different sequence lengths, we need to:
        // 1. Append K/V per sequence to cache
        // 2. Build combined K/V tensors with padding
        // 3. Apply padding-aware attention mask

        if (params_.kv_cache)
        {
            // Append each sequence's K/V to cache
            const int d_kv = params_.n_kv_heads * params_.head_dim;
            (void)d_kv; // TODO: use for proper batch slicing
            for (int b = 0; b < params_.batch_size; ++b)
            {
                int actual_len = params_.sequence_lengths ? (*params_.sequence_lengths)[b] : params_.seq_len;

                // Create view of K/V for this batch
                // Note: This assumes K/V are [batch * seq_len, n_kv_heads * head_dim]
                // TODO: Implement proper batch slicing

                bool append_ok = params_.kv_cache->append(
                    params_.layer_idx, b, K_base, V_base, actual_len);
                if (!append_ok)
                {
                    LOG_ERROR("[AttentionWithKVCacheStage::executeBatched] Failed to append batch " << b);
                    return false;
                }
            }
        }

        // Get attention kernel via device-scoped KernelFactory cache
        auto *attention_kernel = getOrCreateKernel(Q_base);

        if (!attention_kernel)
        {
            LOG_ERROR("[AttentionWithKVCacheStage::executeBatched] Failed to create attention kernel");
            return false;
        }
        bindStageStream(attention_kernel);

        // Allocate workspace for attention scores
        // For batched: [batch_size * n_heads, seq_len, seq_len] but we compute per-batch
        FP32Tensor scores_workspace({static_cast<size_t>(params_.n_heads * params_.seq_len * params_.seq_len)});

        // Build padding-aware causal mask if needed
        std::unique_ptr<FP32Tensor> mask_tensor;
        if (params_.causal || params_.sequence_lengths)
        {
            // For batched: need full [batch*seq, batch*seq] mask for cross-batch isolation
            const int total_tokens = params_.batch_size * params_.seq_len;
            mask_tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(total_tokens * total_tokens)});
            float *mask_data = mask_tensor->mutable_data();

            // Initialize with -inf (block all)
            std::fill(mask_data, mask_data + total_tokens * total_tokens, -std::numeric_limits<float>::infinity());

            // Allow within-batch attention with causal masking
            for (int b = 0; b < params_.batch_size; ++b)
            {
                int actual_len = params_.sequence_lengths ? (*params_.sequence_lengths)[b] : params_.seq_len;
                int batch_offset = b * params_.seq_len;

                for (int q = 0; q < actual_len; ++q)
                {
                    int q_pos = batch_offset + q;
                    for (int k = 0; k < actual_len; ++k)
                    {
                        int k_pos = batch_offset + k;
                        // Causal: Q at position q can attend to K at positions [0, q] within batch
                        if (!params_.causal || k <= q)
                        {
                            mask_data[q_pos * total_tokens + k_pos] = 0.0f;
                        }
                    }
                }
            }
        }

        // Dispatch via compute_tensor
        bool success = attention_kernel->compute_tensor(
            Q_base, K_base, V_base, output_base,
            params_.batch_size,
            params_.seq_len, // Query sequence length
            params_.seq_len, // KV length (same for batched prefill)
            params_.n_heads,
            params_.n_kv_heads,
            params_.head_dim,
            params_.causal,
            params_.window_size,
            &scores_workspace,
            mask_tensor.get(),
            params_.mpi_ctx_shared.get(),
            params_.device_id.toKernelDeviceIndex(),
            params_.head_start,
            params_.local_n_heads,
            params_.local_n_kv_heads);

        if (!success)
        {
            LOG_ERROR("[AttentionWithKVCacheStage::executeBatched] Attention kernel compute_tensor failed");
            return false;
        }

        LOG_DEBUG("[AttentionWithKVCacheStage::executeBatched] Complete");
        return true;
    }

    size_t AttentionWithKVCacheStage::estimatedFlops() const
    {
        // QK: 2 * seq_len * kv_len * head_dim (per head)
        // For prefill: kv_len ≈ seq_len
        // For decode: kv_len = cached_tokens, seq_len = 1
        int estimated_kv_len = params_.seq_len; // Conservative estimate
        if (params_.kv_cache && params_.seq_len == 1)
        {
            estimated_kv_len = params_.kv_cache->get_cached_tokens(params_.layer_idx, 0);
        }

        size_t qk_flops = 2ULL * params_.seq_len * estimated_kv_len * params_.head_dim;
        size_t softmax_flops = 5ULL * params_.seq_len * estimated_kv_len;
        size_t v_flops = 2ULL * params_.seq_len * estimated_kv_len * params_.head_dim;
        return (qk_flops + softmax_flops + v_flops) * params_.n_heads * params_.batch_size;
    }

    size_t AttentionWithKVCacheStage::estimatedMemoryBytes() const
    {
        // Q + K + V + output (all at FP32)
        size_t qkv_bytes = static_cast<size_t>(params_.batch_size) * params_.seq_len *
                           (params_.n_heads + 2 * params_.n_kv_heads) * params_.head_dim * sizeof(float);
        return qkv_bytes;
    }

    bool AttentionWithKVCacheStage::supportsBackend(ComputeBackendType backend) const
    {
        // MpiAttentionOrchestrator currently only supports CPU
        switch (backend)
        {
        case ComputeBackendType::CPU:

            return true;
        default:
            return false;
        }
    }

    StageDumpInfo AttentionWithKVCacheStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;

        // Q/K/V inputs - use safe accessor for Q8_1 compatibility
        if (params_.Q)
        {
            const float *q_data = getSafeFp32Data(params_.Q);
            if (q_data)
            {
                info.addInput("Q", q_data,
                              params_.batch_size * params_.seq_len,
                              params_.n_heads * params_.head_dim);
            }
        }

        // Output - use safe accessor for Q8_1 compatibility
        if (params_.output)
        {
            const float *out_data = getSafeFp32Data(params_.output);
            if (out_data)
            {
                info.addOutput("output", out_data,
                               params_.batch_size * params_.seq_len,
                               params_.n_heads * params_.head_dim);
            }
        }

        // Scalar params
        info.addScalarInt("batch_size", params_.batch_size);
        info.addScalarInt("seq_len", params_.seq_len);
        info.addScalarInt("n_heads", params_.n_heads);
        info.addScalarInt("n_kv_heads", params_.n_kv_heads);
        info.addScalarInt("head_dim", params_.head_dim);
        info.addScalarBool("causal", params_.causal);

        return info;
    }

    StageBufferRequirements AttentionWithKVCacheStage::getBufferRequirements() const
    {
        StageBufferRequirements reqs;

        // Input: Q (query)
        if (params_.Q)
        {
            const size_t seq_dim = static_cast<size_t>(params_.batch_size * params_.seq_len);
            const size_t q_dim = static_cast<size_t>(params_.n_heads * params_.head_dim);
            BufferTensorType buf_type = toBufferTensorType(params_.Q->native_type());
            reqs.addInput("Q", {seq_dim, q_dim}, buf_type);
        }

        // Input: K (key)
        if (params_.K)
        {
            const size_t seq_dim = static_cast<size_t>(params_.batch_size * params_.seq_len);
            const size_t k_dim = static_cast<size_t>(params_.n_kv_heads * params_.head_dim);
            BufferTensorType buf_type = toBufferTensorType(params_.K->native_type());
            reqs.addInput("K", {seq_dim, k_dim}, buf_type);
        }

        // Input: V (value)
        if (params_.V)
        {
            const size_t seq_dim = static_cast<size_t>(params_.batch_size * params_.seq_len);
            const size_t v_dim = static_cast<size_t>(params_.n_kv_heads * params_.head_dim);
            BufferTensorType buf_type = toBufferTensorType(params_.V->native_type());
            reqs.addInput("V", {seq_dim, v_dim}, buf_type);
        }

        // Output: attention output
        if (params_.output)
        {
            const size_t seq_dim = static_cast<size_t>(params_.batch_size * params_.seq_len);
            const size_t out_dim = static_cast<size_t>(params_.n_heads * params_.head_dim);
            BufferTensorType buf_type = toBufferTensorType(params_.output->native_type());
            reqs.addOutput("output", {seq_dim, out_dim}, buf_type);
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
