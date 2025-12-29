/**
 * @file FusedAttentionWoStage.cpp
 * @brief Implementation of FusedAttentionWoStage
 */

#include "FusedAttentionWoStage.h"
#include "../ComputeStageUtils.h"
#include "../../../utils/Assertions.h"
#include "../../../utils/DebugEnv.h"
#include "../../../tensors/Tensors.h"
#include "../../../tensors/UnifiedKVCache.h"
#include "../../../utils/Logger.h"
#include "../../../kernels/KernelFactory.h"
#include "../../../kernels/cpu/attention/q8_1/FusedAttentionWoKernel.h"
#include "../../../kernels/cpu/attention/q16_1/Q16FusedAttentionKernel.h"
#include <cmath>

namespace llaminar2
{

    // =============================================================================
    // FusedAttentionWoStage Implementation
    // =============================================================================

    FusedAttentionWoStage::FusedAttentionWoStage(Params params)
        : params_(std::move(params))
    {
        if (params_.backend == FusedAttentionBackend::Q16_INTEGER)
        {
            // Create Q16_1 kernel via ITensorFusedAttentionWo interface
            q16_kernel_ = std::make_unique<kernels::q16_1::Q16FusedAttentionKernel>(/*use_jit=*/false);
            LOG_DEBUG("[FusedAttentionWoStage] Created Q16FusedAttentionKernel (Q16_INTEGER backend)");
        }
        else
        {
            // Create Q8_1 kernel (JIT/TILED/REFERENCE backends)
            FusedAttentionWoKernel::Config kernel_config;
            kernel_config.num_heads = params_.n_heads;
            kernel_config.num_kv_heads = params_.n_kv_heads;
            kernel_config.head_dim = params_.head_dim;
            kernel_config.d_model = params_.d_model;
            kernel_config.backend = params_.backend;
            kernel_config.use_hybrid_wo = params_.use_hybrid_wo;
            kernel_config.fuse_residual_add = params_.fuse_residual_add;

            kernel_ = std::make_unique<FusedAttentionWoKernel>(kernel_config);
        }
    }

    bool FusedAttentionWoStage::execute(IDeviceContext *ctx)
    {
        // Dynamic kv_len: query from KV cache at execution time if available
        int effective_kv_len = params_.kv_len;
        if (params_.kv_cache && params_.layer_idx >= 0)
        {
            effective_kv_len = params_.kv_cache->get_cached_tokens(params_.layer_idx, 0);
            if (effective_kv_len == 0)
            {
                effective_kv_len = params_.seq_len; // Prefill case
            }
            LOG_TRACE("[FusedAttentionWoStage] Dynamic kv_len from cache: " << effective_kv_len
                                                                            << " (static was: " << params_.kv_len << ")");
        }

        LOG_DEBUG("[FusedAttentionWoStage] Execute: batch=" << params_.batch_size
                                                            << " seq_len=" << params_.seq_len
                                                            << " kv_len=" << effective_kv_len
                                                            << " n_heads=" << params_.n_heads
                                                            << " n_kv_heads=" << params_.n_kv_heads
                                                            << " head_dim=" << params_.head_dim
                                                            << " d_model=" << params_.d_model
                                                            << " position_offset=" << params_.position_offset
                                                            << " backend=" << fusedAttentionBackendToString(params_.backend));

        // Validate inputs
        if (!params_.Q || !params_.K || !params_.V || !params_.Wo || !params_.output)
        {
            LOG_ERROR("[FusedAttentionWoStage] Null tensor pointers");
            return false;
        }

        if (params_.seq_len <= 0 || effective_kv_len <= 0 ||
            params_.n_heads <= 0 || params_.n_kv_heads <= 0 ||
            params_.head_dim <= 0 || params_.d_model <= 0)
        {
            LOG_ERROR("[FusedAttentionWoStage] Invalid dimensions");
            return false;
        }

        if (params_.n_heads % params_.n_kv_heads != 0)
        {
            LOG_ERROR("[FusedAttentionWoStage] n_heads (" << params_.n_heads
                                                          << ") must be divisible by n_kv_heads (" << params_.n_kv_heads << ")");
            return false;
        }

        // Compute position offset for decode mode
        int position_offset = params_.position_offset;
        if (position_offset == 0 && params_.seq_len < effective_kv_len)
        {
            // Auto-compute for decode mode: query is at end of cached context
            position_offset = effective_kv_len - params_.seq_len;
        }

        bool success = false;

        // Dispatch to appropriate kernel based on backend
        if (params_.backend == FusedAttentionBackend::Q16_INTEGER)
        {
            // Q16_1 kernel via ITensorFusedAttentionWo interface
            // NOTE: Q16_INTEGER backend requires fuse_residual_add=true (HybridQ16 mode)
            if (!params_.fuse_residual_add)
            {
                LOG_ERROR("[FusedAttentionWoStage] Q16_INTEGER backend requires fuse_residual_add=true "
                          "(use JIT or REFERENCE backend for non-fused mode)");
                return false;
            }

            if (!q16_kernel_)
            {
                LOG_ERROR("[FusedAttentionWoStage] Q16 kernel not initialized");
                return false;
            }

            // Build params for Q16 kernel
            FusedAttentionWoParams q16_params;

            // Extract typed data from tensors - Q16 kernel expects Q16_1Block* for Q/K/V
            // Use dynamic_cast + typed_data() for type-safe extraction
            auto *q_q16 = dynamic_cast<Q16_1Tensor *>(params_.Q);
            auto *k_q16 = dynamic_cast<Q16_1Tensor *>(params_.K);
            auto *v_q16 = dynamic_cast<Q16_1Tensor *>(params_.V);

            if (!q_q16 || !k_q16 || !v_q16)
            {
                LOG_ERROR("[FusedAttentionWoStage] Q16_INTEGER backend requires Q16_1 tensors for Q/K/V, got: "
                          << "Q=" << (params_.Q ? params_.Q->dtype_name() : "null") << ", "
                          << "K=" << (params_.K ? params_.K->dtype_name() : "null") << ", "
                          << "V=" << (params_.V ? params_.V->dtype_name() : "null"));
                return false;
            }

            q16_params.Q = q_q16->typed_data();
            q16_params.K = k_q16->typed_data();
            q16_params.V = v_q16->typed_data();

            // Wo weights - get VNNI packed weights via KernelFactory
            // Check if Wo implements IINT8Unpackable (quantized formats)
            if (dynamic_cast<IINT8Unpackable *>(params_.Wo) != nullptr)
            {
                // Get VNNI-packed weights from cache (or pack on first call)
                auto *packed = llaminar::v2::kernels::KernelFactory::ensurePackedWeightsInTensorCache(params_.Wo);
                q16_params.Wo_packed = packed;
                q16_params.Wo_fp32 = nullptr;
                LOG_DEBUG("[FusedAttentionWoStage] Q16_INTEGER: Using VNNI-packed Wo weights from "
                          << params_.Wo->dtype_name() << " tensor");
            }
            else if (auto *wo_fp32 = dynamic_cast<FP32Tensor *>(params_.Wo))
            {
                // FP32 Wo weights (no packing needed)
                q16_params.Wo_packed = nullptr;
                q16_params.Wo_fp32 = wo_fp32->data();
                LOG_DEBUG("[FusedAttentionWoStage] Q16_INTEGER: Using FP32 Wo weights");
            }
            else
            {
                LOG_ERROR("[FusedAttentionWoStage] Q16_INTEGER: Unsupported Wo weight type: "
                          << (params_.Wo ? params_.Wo->dtype_name() : "null"));
                return false;
            }

            // Residual fusion: output tensor IS the residual (in-place read-modify-write)
            // Q16_INTEGER always uses fused residual path (validated above)
            if (auto *out_q16 = dynamic_cast<Q16_1Tensor *>(params_.output))
            {
                // residual_in = residual_out = output tensor (in-place accumulation)
                q16_params.residual_in = out_q16->typed_data();
                q16_params.residual_out = out_q16->mutable_typed_data();
                LOG_DEBUG("[FusedAttentionWoStage] Q16_INTEGER: Residual fusion enabled (in-place Q16_1)");
            }
            else
            {
                LOG_ERROR("[FusedAttentionWoStage] Q16_INTEGER with fuse_residual_add requires Q16_1 output, got "
                          << (params_.output ? params_.output->dtype_name() : "null"));
                return false;
            }

            // Dimensions
            q16_params.seq_len_q = params_.seq_len;
            q16_params.kv_len = effective_kv_len;
            q16_params.n_heads = params_.n_heads;
            q16_params.n_kv_heads = params_.n_kv_heads;
            q16_params.head_dim = params_.head_dim;
            q16_params.d_model = params_.d_model;

            // Attention config
            q16_params.scale = 1.0f / std::sqrt(static_cast<float>(params_.head_dim));
            q16_params.causal = params_.causal;
            q16_params.position_offset = position_offset;

            // Snapshot buffers - FAIL FAST if snapshot capture is enabled but buffer is wrong type
            q16_params.scores_snapshot = nullptr;
            q16_params.context_snapshot = nullptr;
#ifdef ENABLE_PIPELINE_SNAPSHOTS
            if (params_.context_snapshot)
            {
                auto *ctx_fp32 = dynamic_cast<FP32Tensor *>(params_.context_snapshot);
                LLAMINAR_ASSERT_CAST(ctx_fp32, "FP32Tensor",
                                     "context_snapshot (got " +
                                         std::string(params_.context_snapshot->dtype_name()) + ")");
                q16_params.context_snapshot = ctx_fp32->mutable_data();
                LOG_TRACE("[FusedAttentionWoStage] Q16_INTEGER: context_snapshot buffer set, ptr="
                          << q16_params.context_snapshot);
            }
#endif

            success = q16_kernel_->compute(q16_params);
        }
        else
        {
            // Q8_1 kernel (JIT/TILED/REFERENCE)
            success = kernel_->compute(
                params_.Q,
                params_.K,
                params_.V,
                params_.Wo,
                params_.output,
                params_.seq_len,
                effective_kv_len,
                params_.causal,
                position_offset,
                params_.context_snapshot);
        }

        if (!success)
        {
            LOG_ERROR("[FusedAttentionWoStage] Kernel compute() failed");
            return false;
        }

        LOG_DEBUG("[FusedAttentionWoStage] Execute complete");
        return true;
    }

    size_t FusedAttentionWoStage::estimatedFlops() const
    {
        // Attention FLOPs + Wo projection FLOPs
        // Attention: 2 * batch * n_heads * seq_len * kv_len * head_dim (QK^T)
        //          + 4 * batch * n_heads * seq_len * kv_len (softmax)
        //          + 2 * batch * n_heads * seq_len * kv_len * head_dim (scores @ V)
        // Wo: 2 * batch * seq_len * d_model * (n_heads * head_dim)
        const size_t qk_flops = 2ULL * params_.batch_size * params_.n_heads *
                                params_.seq_len * params_.kv_len * params_.head_dim;
        const size_t softmax_flops = 4ULL * params_.batch_size * params_.n_heads *
                                     params_.seq_len * params_.kv_len;
        const size_t sv_flops = qk_flops;
        const size_t wo_flops = 2ULL * params_.batch_size * params_.seq_len *
                                params_.d_model * (params_.n_heads * params_.head_dim);
        return qk_flops + softmax_flops + sv_flops + wo_flops;
    }

    size_t FusedAttentionWoStage::estimatedMemoryBytes() const
    {
        // Fused kernel eliminates attention context intermediate
        // Main memory: Q, K, V, Wo weights, output
        // Scratch: internal scores/state (managed by kernel)
        return static_cast<size_t>(params_.n_heads) * params_.seq_len * params_.kv_len * sizeof(float);
    }

    bool FusedAttentionWoStage::supportsBackend(ComputeBackendType backend) const
    {
        // Currently CPU-only (JIT backend uses AVX-512 VNNI)
        return backend == ComputeBackendType::CPU;
    }

    StageDumpInfo FusedAttentionWoStage::getDumpInfo() const
    {
        StageDumpInfo info;

        // Context snapshot: pre-Wo attention output (for debugging/parity testing)
        if (params_.context_snapshot)
        {
            const float *ctx_data = getSafeFp32Data(params_.context_snapshot);
            LOG_DEBUG("[FusedAttentionWoStage::getDumpInfo] context_snapshot tensor type: "
                      << static_cast<int>(params_.context_snapshot->native_type())
                      << " ctx_data=" << (ctx_data ? "valid" : "NULL")
                      << " seq_len=" << params_.seq_len
                      << " n_heads*head_dim=" << params_.n_heads * params_.head_dim);
            if (ctx_data)
            {
                info.addOutput("context",
                               ctx_data,
                               params_.seq_len,
                               params_.n_heads * params_.head_dim);
            }
        }
        else
        {
            LOG_DEBUG("[FusedAttentionWoStage::getDumpInfo] context_snapshot is NULL");
        }

        // Output: fused attention+Wo projection result
        if (params_.output)
        {
            const float *out_data = getSafeFp32Data(params_.output);
            if (out_data)
            {
                info.addOutput("output",
                               out_data,
                               params_.seq_len,
                               params_.d_model);
            }
        }

        info.addScalarInt("batch_size", params_.batch_size);
        info.addScalarInt("seq_len", params_.seq_len);
        info.addScalarInt("kv_len", params_.kv_len);
        info.addScalarInt("n_heads", params_.n_heads);
        info.addScalarInt("n_kv_heads", params_.n_kv_heads);
        info.addScalarInt("head_dim", params_.head_dim);
        info.addScalarInt("d_model", params_.d_model);
        info.addScalarBool("causal", params_.causal);
        info.addScalarInt("backend", static_cast<int>(params_.backend));
        info.addScalarInt("device_idx", params_.device_idx);

        return info;
    }

    StageBufferRequirements FusedAttentionWoStage::getBufferRequirements() const
    {
        StageBufferRequirements reqs;

        // Input: Q, K, V (Q8_1)
        if (params_.Q)
        {
            const size_t q_rows = static_cast<size_t>(params_.batch_size * params_.seq_len);
            const size_t q_cols = static_cast<size_t>(params_.n_heads * params_.head_dim);
            BufferTensorType buf_type = toBufferTensorType(params_.Q->native_type());
            reqs.addInput("Q", {q_rows, q_cols}, buf_type);
        }

        if (params_.K)
        {
            const size_t k_rows = static_cast<size_t>(params_.batch_size * params_.kv_len);
            const size_t k_cols = static_cast<size_t>(params_.n_kv_heads * params_.head_dim);
            BufferTensorType buf_type = toBufferTensorType(params_.K->native_type());
            reqs.addInput("K", {k_rows, k_cols}, buf_type);
        }

        if (params_.V)
        {
            const size_t v_rows = static_cast<size_t>(params_.batch_size * params_.kv_len);
            const size_t v_cols = static_cast<size_t>(params_.n_kv_heads * params_.head_dim);
            BufferTensorType buf_type = toBufferTensorType(params_.V->native_type());
            reqs.addInput("V", {v_rows, v_cols}, buf_type);
        }

        // Input: Wo weight
        if (params_.Wo)
        {
            BufferTensorType buf_type = toBufferTensorType(params_.Wo->native_type());
            reqs.addInput("Wo", params_.Wo->shape(), buf_type);
        }

        // Output: projected attention (FP32)
        if (params_.output)
        {
            const size_t out_rows = static_cast<size_t>(params_.batch_size * params_.seq_len);
            const size_t out_cols = static_cast<size_t>(params_.d_model);
            BufferTensorType buf_type = toBufferTensorType(params_.output->native_type());
            reqs.addOutput("output", {out_rows, out_cols}, buf_type);
        }

        return reqs;
    }

} // namespace llaminar2
