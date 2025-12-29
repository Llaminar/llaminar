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
#include "../../../tensors/UnifiedKVCache.h"
#include "../../../utils/OpenMPUtils.h"

#include <immintrin.h>

namespace llaminar2
{

    // =============================================================================
    // KVCacheAppendStage Implementation
    // =============================================================================

    KVCacheAppendStage::KVCacheAppendStage(Params params)
        : params_(std::move(params)) {}

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

                bool success = params_.kv_cache->append_kv(
                    params_.layer_idx, seq_idx,
                    k_slice.get(), v_slice.get(), seq_len);

                if (!success)
                {
                    LOG_ERROR("[KVCacheAppendStage] append_kv failed for batch " << b);
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

            bool success = params_.kv_cache->append_kv(
                params_.layer_idx, params_.seq_idx,
                k_slice.get(), v_slice.get(), total_tokens);

            if (!success)
            {
                LOG_ERROR("[KVCacheAppendStage] append_kv failed (after conversion)");
                return false;
            }

            return true;
        }

        // Q16_1 cache conversion path - handles HybridQ16 mode where:
        // - K_rope is Q16_1 (from RoPE Q8_1→Q16_1 conversion)
        // - V is Q8_1 (from GEMM output)
        // - KV cache is Q16_1
        bool cache_is_q16_1 = (params_.kv_cache->precision() == ActivationPrecision::Q16_1);
        bool k_is_q16_1 = (params_.K->native_type() == TensorType::Q16_1);
        bool v_is_q8_1 = (params_.V->native_type() == TensorType::Q8_1);

        if (cache_is_q16_1 && k_is_q16_1 && v_is_q8_1)
        {
            LOG_DEBUG("[KVCacheAppendStage] HybridQ16 mode: K is Q16_1, converting V from Q8_1 to Q16_1"
                      << " (tokens=" << total_tokens << ")");

            const size_t kv_dim = params_.K->shape().size() > 1 ? params_.K->shape()[1] : 0;

            // K is already Q16_1 - can pass directly
            // V needs Q8_1 → Q16_1 conversion
            auto v_q16_slice = std::make_unique<Q16_1Tensor>(
                std::vector<size_t>{static_cast<size_t>(total_tokens), kv_dim});

            const auto *v_q8 = dynamic_cast<const Q8_1Tensor *>(params_.V);
            if (!v_q8)
            {
                LOG_ERROR("[KVCacheAppendStage] Expected Q8_1 V tensor but got " << params_.V->dtype_name());
                return false;
            }

            // Convert Q8_1 V to Q16_1 with SIMD acceleration
            // Q16_1 has 256× finer precision (±32767 vs ±127)
            constexpr size_t block_size = 32; // Q8_1 and Q16_1 both use 32-element blocks
            const size_t blocks_per_row = (kv_dim + block_size - 1) / block_size;
            const Q8_1Block *v_q8_data = v_q8->typed_data();
            Q16_1Block *v_q16_data = v_q16_slice->mutable_typed_data();

            // Parallel conversion using OMP_WORKSHARE_REGION
            auto do_conversion = [&]()
            {
#pragma omp for schedule(static)
                for (int t = 0; t < total_tokens; ++t)
                {
                    const Q8_1Block *src_blocks = v_q8_data + t * blocks_per_row;
                    Q16_1Block *dst_blocks = v_q16_data + t * blocks_per_row;

                    for (size_t b = 0; b < blocks_per_row; ++b)
                    {
                        const Q8_1Block &src = src_blocks[b];
                        Q16_1Block &dst = dst_blocks[b];

                        // Q8_1 scale is FP16, Q16_1 scale is FP32
                        // Scale up int8→int16 by 256, divide scale by 256 to preserve value
                        dst.d = simd::fp16_to_fp32(src.d) / 256.0f;

#if defined(__AVX512F__)
                        // AVX512: Process all 32 int8 values → 32 int16 values
                        // Load 32 int8 values (256 bits)
                        __m256i i8_vec = _mm256_loadu_si256(
                            reinterpret_cast<const __m256i *>(src.qs));
                        // Sign-extend lower 16 int8 → int16
                        __m256i lo_16 = _mm256_cvtepi8_epi16(
                            _mm256_castsi256_si128(i8_vec));
                        // Sign-extend upper 16 int8 → int16
                        __m256i hi_16 = _mm256_cvtepi8_epi16(
                            _mm256_extracti128_si256(i8_vec, 1));
                        // Multiply by 256 (left shift by 8)
                        __m256i lo_scaled = _mm256_slli_epi16(lo_16, 8);
                        __m256i hi_scaled = _mm256_slli_epi16(hi_16, 8);
                        // Store 32 int16 values
                        _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst.qs), lo_scaled);
                        _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst.qs + 16), hi_scaled);

                        // Compute sum_qs (sum of all 32 int16 values)
                        // Horizontal add within each lane
                        __m256i sum_lo = _mm256_madd_epi16(lo_scaled, _mm256_set1_epi16(1));
                        __m256i sum_hi = _mm256_madd_epi16(hi_scaled, _mm256_set1_epi16(1));
                        __m256i sum_all = _mm256_add_epi32(sum_lo, sum_hi);
                        // Reduce to single int32
                        __m128i sum128 = _mm_add_epi32(
                            _mm256_castsi256_si128(sum_all),
                            _mm256_extracti128_si256(sum_all, 1));
                        sum128 = _mm_hadd_epi32(sum128, sum128);
                        sum128 = _mm_hadd_epi32(sum128, sum128);
                        dst.sum_qs = _mm_cvtsi128_si32(sum128);

#elif defined(__AVX2__)
                        // AVX2: Process 32 int8 values in two halves
                        __m128i i8_lo = _mm_loadu_si128(
                            reinterpret_cast<const __m128i *>(src.qs));
                        __m128i i8_hi = _mm_loadu_si128(
                            reinterpret_cast<const __m128i *>(src.qs + 16));
                        // Sign-extend int8 → int16
                        __m256i i16_lo = _mm256_cvtepi8_epi16(i8_lo);
                        __m256i i16_hi = _mm256_cvtepi8_epi16(i8_hi);
                        // Multiply by 256 (left shift by 8)
                        __m256i scaled_lo = _mm256_slli_epi16(i16_lo, 8);
                        __m256i scaled_hi = _mm256_slli_epi16(i16_hi, 8);
                        // Store
                        _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst.qs), scaled_lo);
                        _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst.qs + 16), scaled_hi);

                        // Compute sum_qs
                        __m256i sum_lo = _mm256_madd_epi16(scaled_lo, _mm256_set1_epi16(1));
                        __m256i sum_hi = _mm256_madd_epi16(scaled_hi, _mm256_set1_epi16(1));
                        __m256i sum_all = _mm256_add_epi32(sum_lo, sum_hi);
                        __m128i sum128 = _mm_add_epi32(
                            _mm256_castsi256_si128(sum_all),
                            _mm256_extracti128_si256(sum_all, 1));
                        sum128 = _mm_hadd_epi32(sum128, sum128);
                        sum128 = _mm_hadd_epi32(sum128, sum128);
                        dst.sum_qs = _mm_cvtsi128_si32(sum128);
#else
                        // Scalar fallback
                        int32_t sum = 0;
                        for (size_t i = 0; i < block_size; ++i)
                        {
                            int16_t scaled = static_cast<int16_t>(src.qs[i]) * 256;
                            dst.qs[i] = scaled;
                            sum += scaled;
                        }
                        dst.sum_qs = sum;
#endif
                    }
                }
            };
            OMP_WORKSHARE_REGION(do_conversion);

            // Also populate V_dequant_out if requested (for decomposed attention path)
            if (params_.V_dequant_out)
            {
                auto *v_dequant_q16 = dynamic_cast<Q16_1Tensor *>(params_.V_dequant_out);
                if (v_dequant_q16 && v_dequant_q16->mutable_typed_data())
                {
                    // Copy Q16_1 data to V_dequant_out
                    std::memcpy(v_dequant_q16->mutable_typed_data(),
                                v_q16_slice->typed_data(),
                                total_tokens * blocks_per_row * sizeof(Q16_1Block));
                    LOG_DEBUG("[KVCacheAppendStage] Populated V_dequant_out with Q16_1 values");
                }
            }

            bool success = params_.kv_cache->append_kv(
                params_.layer_idx, params_.seq_idx,
                params_.K, v_q16_slice.get(), total_tokens);

            if (!success)
            {
                LOG_ERROR("[KVCacheAppendStage] append_kv failed (HybridQ16 conversion)");
                return false;
            }

            return true;
        }

        // Direct append path - tensors already match cache precision
        bool success = params_.kv_cache->append_kv(
            params_.layer_idx, params_.seq_idx,
            params_.K, params_.V, total_tokens);

        if (!success)
        {
            LOG_ERROR("[KVCacheAppendStage] append_kv failed");
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

    StageDumpInfo KVCacheAppendStage::getDumpInfo() const
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
