/**
 * @file KVCacheAppendStage.cpp
 * @brief Implementation of KVCacheAppendStage
 */

#include "KVCacheAppendStage.h"
#include "../ComputeStageUtils.h"
#include "../../../utils/DebugEnv.h"
#include "../../../tensors/Tensors.h"
#include "../../../tensors/SIMDHelpers.h"
#include "../../../utils/Logger.h"
#include "../../../kernels/cpu/CPUKVCache.h"
#include "../../../utils/OpenMPUtils.h"
#include "../../../kernels/cpu/attention/q16_1/VNNISafetyConstants.h"

#include <immintrin.h>

namespace llaminar2
{

    // =============================================================================
    // KVCacheAppendStage Implementation
    // =============================================================================

    KVCacheAppendStage::KVCacheAppendStage(Params params)
        : IComputeStage(params.device_id)
        , params_(std::move(params))
    {
    }

    bool KVCacheAppendStage::execute(IDeviceContext *ctx)
    {
        (void)ctx;

        if (!params_.kv_cache)
        {
            LOG_ERROR("[KVCacheAppendStage] No KV cache provided");
            return false;
        }

        if (!params_.K || !params_.V)
        {
            LOG_ERROR("[KVCacheAppendStage] Invalid K/V tensors");
            return false;
        }

        // Determine total tokens to append
        int total_tokens = params_.num_tokens;
        if (total_tokens <= 0)
        {
            total_tokens = static_cast<int>(params_.K->shape()[0]);
        }

        // Determine batch handling mode
        const int batch_size = params_.batch_size;
        const int seq_len = params_.seq_len;

        // If batch_size > 1 and seq_len > 0, do per-sequence append
        // K/V layout: [batch_size * seq_len, kv_dim] - contiguous per-sequence
        if (batch_size > 1 && seq_len > 0)
        {
            const size_t kv_dim = params_.K->shape().size() > 1 ? params_.K->shape()[1] : 0;

            LOG_DEBUG("[KVCacheAppendStage] Batched append: batch_size=" << batch_size
                                                                         << " seq_len=" << seq_len
                                                                         << " kv_dim=" << kv_dim
                                                                         << " layer=" << params_.layer_idx);

            // Get raw data pointers for slicing
            const float *k_data = params_.K->fp32_data();
            const float *v_data = params_.V->fp32_data();

            if (!k_data || !v_data)
            {
                LOG_ERROR("[KVCacheAppendStage] Cannot get FP32 data for K/V tensors");
                return false;
            }

            // Create temporary tensors for per-sequence slices
            // Note: We create views/copies that the cache will copy from
            for (int b = 0; b < batch_size; ++b)
            {
                const int seq_idx = params_.seq_idx + b;
                const size_t offset = b * seq_len * kv_dim;

                // Create temporary FP32 tensors wrapping the slice data
                // These are views into the contiguous K/V buffer
                auto k_slice = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(seq_len), kv_dim});
                auto v_slice = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(seq_len), kv_dim});

                // Copy slice data (could be optimized with non-owning views)
                std::memcpy(k_slice->mutable_data(), k_data + offset, seq_len * kv_dim * sizeof(float));
                std::memcpy(v_slice->mutable_data(), v_data + offset, seq_len * kv_dim * sizeof(float));

                LOG_TRACE("[KVCacheAppendStage] Appending " << seq_len << " tokens to layer "
                                                            << params_.layer_idx << " seq_idx=" << seq_idx);

                bool success = params_.kv_cache->append(
                    params_.layer_idx, seq_idx,
                    k_slice.get(), v_slice.get(), seq_len);

                if (!success)
                {
                    LOG_ERROR("[KVCacheAppendStage] append failed for batch " << b);
                    return false;
                }
            }

            return true;
        }

        // Single-sequence path (original behavior)
        LOG_DEBUG("[KVCacheAppendStage] Single-sequence append: " << total_tokens
                                                                  << " tokens to layer " << params_.layer_idx << " seq " << params_.seq_idx);

        // Check if tensors match cache precision - if not, need to convert
        // This handles Hybrid mode where K_rope is FP32 and V is Q8_1 but cache is FP32
        bool cache_is_fp32 = (params_.kv_cache->precision() == ActivationPrecision::FP32);
        bool k_is_fp32 = (params_.K->native_type() == TensorType::FP32);
        bool v_is_fp32 = (params_.V->native_type() == TensorType::FP32);

        // If cache is FP32 but inputs are not, convert to FP32 for cache append
        if (cache_is_fp32 && (!k_is_fp32 || !v_is_fp32))
        {
            LOG_DEBUG("[KVCacheAppendStage] Converting K/V to FP32 for cache append"
                      << " (K=" << params_.K->dtype_name()
                      << ", V=" << params_.V->dtype_name()
                      << ", tokens=" << total_tokens << ")");

            const size_t kv_dim = params_.K->shape().size() > 1 ? params_.K->shape()[1] : 0;

            // Create FP32 wrapper tensors for cache append
            auto k_slice = std::make_unique<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(total_tokens), kv_dim});
            auto v_slice = std::make_unique<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(total_tokens), kv_dim});

            // Optimized path for small token counts (decode phase):
            // Use row-by-row dequantization to avoid dequantizing entire tensor.
            // For prefill (large token counts), use fp32_data() which is more efficient
            // due to better cache locality and parallelization.
            constexpr int SMALL_TOKEN_THRESHOLD = 32;

            if (total_tokens <= SMALL_TOKEN_THRESHOLD)
            {
                // Small token count - use row-by-row dequantization
                // This avoids dequantizing the entire [max_seq_len, kv_dim] tensor
                // when we only need [total_tokens, kv_dim] elements.

                // Handle K (may be FP32 already in Hybrid mode after RoPE)
                if (k_is_fp32)
                {
                    const float *k_fp32 = params_.K->fp32_data();
                    std::memcpy(k_slice->mutable_data(), k_fp32, total_tokens * kv_dim * sizeof(float));
                }
                else
                {
                    // K is Q8_1 - dequant row by row
                    const auto *k_q8 = dynamic_cast<const Q8_1Tensor *>(params_.K);
                    if (k_q8)
                    {
                        for (int t = 0; t < total_tokens; ++t)
                        {
                            k_q8->to_fp32_row(t, k_slice->mutable_data() + t * kv_dim);
                        }
                    }
                    else
                    {
                        // Fallback: use fp32_data()
                        const float *k_fp32 = params_.K->fp32_data();
                        if (!k_fp32)
                        {
                            LOG_ERROR("[KVCacheAppendStage] Cannot get FP32 data for K tensor");
                            return false;
                        }
                        std::memcpy(k_slice->mutable_data(), k_fp32, total_tokens * kv_dim * sizeof(float));
                    }
                }

                // Handle V (usually Q8_1 in Hybrid mode)
                if (v_is_fp32)
                {
                    const float *v_fp32 = params_.V->fp32_data();
                    std::memcpy(v_slice->mutable_data(), v_fp32, total_tokens * kv_dim * sizeof(float));
                }
                else
                {
                    // V is Q8_1 - dequant row by row
                    const auto *v_q8 = dynamic_cast<const Q8_1Tensor *>(params_.V);
                    if (v_q8)
                    {
                        for (int t = 0; t < total_tokens; ++t)
                        {
                            v_q8->to_fp32_row(t, v_slice->mutable_data() + t * kv_dim);
                        }
                    }
                    else
                    {
                        // Fallback: use fp32_data()
                        const float *v_fp32 = params_.V->fp32_data();
                        if (!v_fp32)
                        {
                            LOG_ERROR("[KVCacheAppendStage] Cannot get FP32 data for V tensor");
                            return false;
                        }
                        std::memcpy(v_slice->mutable_data(), v_fp32, total_tokens * kv_dim * sizeof(float));
                    }
                }

                LOG_TRACE("[KVCacheAppendStage] Used row-by-row dequant for " << total_tokens << " tokens");
            }
            else
            {
                // Large token count (prefill) - use fp32_data() for better performance
                const float *k_fp32 = params_.K->fp32_data();
                const float *v_fp32 = params_.V->fp32_data();

                if (!k_fp32 || !v_fp32)
                {
                    LOG_ERROR("[KVCacheAppendStage] Cannot get FP32 data for K/V tensors");
                    return false;
                }

                std::memcpy(k_slice->mutable_data(), k_fp32, total_tokens * kv_dim * sizeof(float));
                std::memcpy(v_slice->mutable_data(), v_fp32, total_tokens * kv_dim * sizeof(float));
            }

            // Hybrid mode: also populate V_dequant_out buffer for downstream attention
            if (params_.V_dequant_out && !v_is_fp32)
            {
                auto *v_dequant_fp32 = dynamic_cast<FP32Tensor *>(params_.V_dequant_out);
                if (v_dequant_fp32 && v_dequant_fp32->mutable_data())
                {
                    // Copy from the already-dequantized v_slice
                    std::memcpy(v_dequant_fp32->mutable_data(), v_slice->data(),
                                total_tokens * kv_dim * sizeof(float));
                    LOG_DEBUG("[KVCacheAppendStage] Populated V_dequant_out with "
                              << total_tokens * kv_dim << " FP32 values");
                }
            }

            bool success = params_.kv_cache->append(
                params_.layer_idx, params_.seq_idx,
                k_slice.get(), v_slice.get(), total_tokens);

            if (!success)
            {
                LOG_ERROR("[KVCacheAppendStage] append failed (after conversion)");
                return false;
            }

            return true;
        }

        // =================================================================
        // Q16_1 cache path with VNNI-safe fixed-scale quantization
        // =================================================================
        // For Q16_1 cache, we MUST use fixed-scale quantization with VNNI-safe
        // clipping to prevent INT32 overflow during VNNI dot-product accumulation.
        //
        // See: VNNISafetyConstants.h for MAX_SAFE_INT16 limits per head_dim
        // See: PROJECT_Q16_INTEGER_ATTENTION_V2.md "VNNI OVERFLOW PREVENTION CONTRACT"
        // =================================================================

        bool cache_is_q16_1 = (params_.kv_cache->precision() == ActivationPrecision::Q16_1);

        if (cache_is_q16_1)
        {
            const size_t kv_dim = params_.K->shape().size() > 1 ? params_.K->shape()[1] : 0;
            const float kv_cache_scale = params_.kv_cache_scale;
            const int head_dim = params_.head_dim;

            // Get VNNI-safe clipping limit
            const int16_t max_safe_int16 = vnni_safety::get_max_safe_int16(head_dim);

            LOG_DEBUG("[KVCacheAppendStage] Q16_1 cache with VNNI-safe fixed-scale quantization"
                      << " (scale=" << kv_cache_scale
                      << ", head_dim=" << head_dim
                      << ", max_safe_int16=" << max_safe_int16
                      << ", tokens=" << total_tokens << ")");

            // Determine input types
            bool k_is_q16_1 = (params_.K->native_type() == TensorType::Q16_1);
            bool k_is_fp32 = (params_.K->native_type() == TensorType::FP32);
            bool v_is_q8_1 = (params_.V->native_type() == TensorType::Q8_1);
            bool v_is_fp32 = (params_.V->native_type() == TensorType::FP32);

            // -----------------------------------------------------------------
            // Handle K tensor conversion/passthrough
            // -----------------------------------------------------------------
            std::unique_ptr<Q16_1Tensor> k_q16_owned;
            const TensorBase *k_for_cache = nullptr;

            if (k_is_q16_1)
            {
                // K is already Q16_1 - pass through directly
                // Assumption: K was quantized with the same fixed scale (e.g., from RoPE stage)
                // Cast ITensor* to TensorBase* (const)
                k_for_cache = dynamic_cast<const TensorBase *>(params_.K);
                if (!k_for_cache)
                {
                    LOG_ERROR("[KVCacheAppendStage] K tensor is Q16_1 but not a TensorBase (GPU?)");
                    return false;
                }
                LOG_TRACE("[KVCacheAppendStage] K is already Q16_1, passing through");
            }
            else if (k_is_fp32)
            {
                // K is FP32 - quantize to Q16_1 with fixed scale and VNNI clipping
                // NOTE: Must use same block_size as KV cache (optimal_q16_block_size(head_dim))
                k_q16_owned = std::make_unique<Q16_1Tensor>(
                    std::vector<size_t>{static_cast<size_t>(total_tokens), kv_dim},
                    optimal_q16_block_size(head_dim));

                const float *k_fp32 = params_.K->fp32_data();
                if (!k_fp32)
                {
                    LOG_ERROR("[KVCacheAppendStage] Cannot get FP32 data for K tensor");
                    return false;
                    ;
                }

                // Use fixed-scale quantization with VNNI-safe clipping
                if (!k_q16_owned->copyFrom_fp32_fixed_scale(k_fp32, kv_cache_scale, head_dim))
                {
                    LOG_ERROR("[KVCacheAppendStage] Fixed-scale K quantization failed");
                    return false;
                }

                k_for_cache = k_q16_owned.get();
                LOG_TRACE("[KVCacheAppendStage] Quantized K from FP32 to Q16_1 with fixed scale");
            }
            else
            {
                // K is some other format (Q8_1, etc.) - dequant to FP32 first
                // NOTE: Must use same block_size as KV cache (optimal_q16_block_size(head_dim))
                k_q16_owned = std::make_unique<Q16_1Tensor>(
                    std::vector<size_t>{static_cast<size_t>(total_tokens), kv_dim},
                    optimal_q16_block_size(head_dim));

                const float *k_fp32 = params_.K->fp32_data();
                if (!k_fp32)
                {
                    LOG_ERROR("[KVCacheAppendStage] Cannot dequantize K to FP32");
                    return false;
                }

                if (!k_q16_owned->copyFrom_fp32_fixed_scale(k_fp32, kv_cache_scale, head_dim))
                {
                    LOG_ERROR("[KVCacheAppendStage] Fixed-scale K quantization failed");
                    return false;
                }

                k_for_cache = k_q16_owned.get();
                LOG_TRACE("[KVCacheAppendStage] Converted K from " << params_.K->dtype_name()
                                                                   << " to Q16_1 via FP32 with fixed scale");
            }

            // -----------------------------------------------------------------
            // Handle V tensor conversion
            // -----------------------------------------------------------------
            std::unique_ptr<Q16_1Tensor> v_q16_owned;
            const TensorBase *v_for_cache = nullptr;

            if (v_is_fp32)
            {
                // V is FP32 - quantize to Q16_1 with fixed scale and VNNI clipping
                // NOTE: Must use same block_size as KV cache (optimal_q16_block_size(head_dim))
                v_q16_owned = std::make_unique<Q16_1Tensor>(
                    std::vector<size_t>{static_cast<size_t>(total_tokens), kv_dim},
                    optimal_q16_block_size(head_dim));

                const float *v_fp32 = params_.V->fp32_data();
                if (!v_fp32)
                {
                    LOG_ERROR("[KVCacheAppendStage] Cannot get FP32 data for V tensor");
                    return false;
                }

                if (!v_q16_owned->copyFrom_fp32_fixed_scale(v_fp32, kv_cache_scale, head_dim))
                {
                    LOG_ERROR("[KVCacheAppendStage] Fixed-scale V quantization failed");
                    return false;
                }

                v_for_cache = v_q16_owned.get();
                LOG_TRACE("[KVCacheAppendStage] Quantized V from FP32 to Q16_1 with fixed scale");
            }
            else if (v_is_q8_1)
            {
                // V is Q8_1 - dequant to FP32, then requant to Q16_1 with fixed scale
                // NOTE: This is different from the old path which did direct int8→int16 scaling!
                // The old path (×256) was NOT VNNI-safe because it produced values close to ±32767.
                // NOTE: Must use same block_size as KV cache (optimal_q16_block_size(head_dim))
                v_q16_owned = std::make_unique<Q16_1Tensor>(
                    std::vector<size_t>{static_cast<size_t>(total_tokens), kv_dim},
                    optimal_q16_block_size(head_dim));

                const auto *v_q8 = dynamic_cast<const Q8_1Tensor *>(params_.V);
                if (!v_q8)
                {
                    LOG_ERROR("[KVCacheAppendStage] Failed to cast V to Q8_1Tensor");
                    return false;
                }

                // Allocate temporary FP32 buffer for dequantization
                std::vector<float> v_fp32_temp(total_tokens * kv_dim);

                // Dequantize Q8_1 → FP32 (row by row for small token counts, batch for large)
                constexpr int SMALL_TOKEN_THRESHOLD = 32;
                if (total_tokens <= SMALL_TOKEN_THRESHOLD)
                {
                    for (int t = 0; t < total_tokens; ++t)
                    {
                        v_q8->to_fp32_row(t, v_fp32_temp.data() + t * kv_dim);
                    }
                }
                else
                {
                    // Use fp32_data() for larger token counts
                    const float *v_fp32 = v_q8->fp32_data();
                    if (!v_fp32)
                    {
                        LOG_ERROR("[KVCacheAppendStage] Cannot dequantize V from Q8_1");
                        return false;
                    }
                    std::memcpy(v_fp32_temp.data(), v_fp32, total_tokens * kv_dim * sizeof(float));
                }

                // Requantize with fixed scale and VNNI-safe clipping
                if (!v_q16_owned->copyFrom_fp32_fixed_scale(v_fp32_temp.data(), kv_cache_scale, head_dim))
                {
                    LOG_ERROR("[KVCacheAppendStage] Fixed-scale V quantization failed");
                    return false;
                }

                v_for_cache = v_q16_owned.get();
                LOG_TRACE("[KVCacheAppendStage] Converted V from Q8_1 to Q16_1 via FP32 with fixed scale"
                          << " (VNNI-safe, max_int16=" << max_safe_int16 << ")");
            }
            else
            {
                // V is some other format - try to dequant via fp32_data()
                // NOTE: Must use same block_size as KV cache (optimal_q16_block_size(head_dim))
                v_q16_owned = std::make_unique<Q16_1Tensor>(
                    std::vector<size_t>{static_cast<size_t>(total_tokens), kv_dim},
                    optimal_q16_block_size(head_dim));

                const float *v_fp32 = params_.V->fp32_data();
                if (!v_fp32)
                {
                    LOG_ERROR("[KVCacheAppendStage] Cannot convert V (" << params_.V->dtype_name()
                                                                        << ") to Q16_1");
                    return false;
                }

                if (!v_q16_owned->copyFrom_fp32_fixed_scale(v_fp32, kv_cache_scale, head_dim))
                {
                    LOG_ERROR("[KVCacheAppendStage] Fixed-scale V quantization failed");
                    return false;
                }

                v_for_cache = v_q16_owned.get();
                LOG_TRACE("[KVCacheAppendStage] Converted V from " << params_.V->dtype_name()
                                                                   << " to Q16_1 via FP32 with fixed scale");
            }

            // Also populate V_dequant_out if requested (for decomposed attention path)
            if (params_.V_dequant_out && v_q16_owned)
            {
                auto *v_dequant_q16 = dynamic_cast<Q16_1Tensor *>(params_.V_dequant_out);
                if (v_dequant_q16 && v_dequant_q16->mutable_typed_data())
                {
                    constexpr size_t block_size = Q16_1Block::BLOCK_SIZE;
                    const size_t blocks_per_row = (kv_dim + block_size - 1) / block_size;
                    std::memcpy(v_dequant_q16->mutable_typed_data(),
                                v_q16_owned->typed_data(),
                                total_tokens * blocks_per_row * sizeof(Q16_1Block));
                    LOG_DEBUG("[KVCacheAppendStage] Populated V_dequant_out with VNNI-safe Q16_1 values");
                }
            }

            bool success = params_.kv_cache->append(
                params_.layer_idx, params_.seq_idx,
                k_for_cache, v_for_cache, total_tokens);

            if (!success)
            {
                LOG_ERROR("[KVCacheAppendStage] append failed (Q16_1 cache)");
                return false;
            }

            return true;
        }

        // Direct append path - tensors already match cache precision
        // Cast ITensor* to TensorBase* for append_kv
        auto *K_base = dynamic_cast<const TensorBase *>(params_.K);
        auto *V_base = dynamic_cast<const TensorBase *>(params_.V);
        if (!K_base || !V_base)
        {
            LOG_ERROR("[KVCacheAppendStage] K/V tensors must be TensorBase");
            return false;
        }

        // Use stream-aware append when a GPU stream is set (for graph capture compatibility)
        bool success;
        void *stream = gpuStream();
        if (stream || params_.device_id.is_gpu())
        {
            success = params_.kv_cache->appendWithStream(
                params_.layer_idx, params_.seq_idx,
                K_base, V_base, total_tokens, stream);
        }
        else
        {
            success = params_.kv_cache->append(
                params_.layer_idx, params_.seq_idx,
                K_base, V_base, total_tokens);
        }

        if (!success)
        {
            LOG_ERROR("[KVCacheAppendStage] append failed");
            return false;
        }

        return true;
    }

    StageBufferRequirements KVCacheAppendStage::getBufferRequirements() const
    {
        StageBufferRequirements reqs;

        // Input: K (to be appended to cache)
        if (params_.K)
        {
            BufferTensorType buf_type = toBufferTensorType(params_.K->native_type());
            reqs.addInput("K", params_.K->shape(), buf_type);
        }

        // Input: V (to be appended to cache)
        if (params_.V)
        {
            BufferTensorType buf_type = toBufferTensorType(params_.V->native_type());
            reqs.addInput("V", params_.V->shape(), buf_type);
        }

        // Output: V_dequant (optional, for Hybrid mode)
        if (params_.V_dequant_out)
        {
            reqs.addOutput("V_dequant", params_.V_dequant_out->shape(), BufferTensorType::FP32);
        }

        // Note: KV cache itself is external state, not a buffer managed by this stage

        return reqs;
    }

    std::vector<BufferDescriptor> KVCacheAppendStage::getDeclaredOutputs() const
    {
        std::vector<BufferDescriptor> outputs;

        // V_dequant: Produced when in Hybrid mode (Q8_1 activations, FP32 attention)
        // This buffer MUST be populated by this stage when configured
        if (params_.V_dequant_out)
        {
            auto desc = BufferDescriptor::output(
                "V_dequant",
                params_.V_dequant_out->shape(),
                BufferTensorType::FP32);
            desc.withProducer("kv_append").validatePopulated();
            outputs.push_back(std::move(desc));
        }

        return outputs;
    }

    StageDumpInfo KVCacheAppendStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;

        // K input tensor
        if (params_.K)
        {
            info.addInput("K", params_.K, params_.K->rows(), params_.K->cols());
        }

        // V input tensor
        if (params_.V)
        {
            info.addInput("V", params_.V, params_.V->rows(), params_.V->cols());
        }

        // V_dequant output (optional, Hybrid mode)
        if (params_.V_dequant_out)
        {
            info.addOutput("V_dequant", params_.V_dequant_out, params_.V_dequant_out->rows(), params_.V_dequant_out->cols());
        }

        info.addScalarInt("layer_idx", params_.layer_idx);
        info.addScalarInt("seq_len", params_.seq_len);
        info.addScalarInt("num_tokens", params_.num_tokens);

        return info;
    }

} // namespace llaminar2
