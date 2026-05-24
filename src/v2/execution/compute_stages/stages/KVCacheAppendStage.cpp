/**
 * @file KVCacheAppendStage.cpp
 * @brief Implementation of KVCacheAppendStage
 */

#include "KVCacheAppendStage.h"
#include "../ComputeStageUtils.h"
#include "../../local_execution/graph/GraphCaptureGuard.h"
#include "../../../utils/DebugEnv.h"
#include "../../../tensors/Tensors.h"
#include "../../../tensors/SIMDHelpers.h"
#include "../../../utils/Logger.h"
#include "../../../kernels/cpu/CPUKVCache.h"
#include "../../../kernels/cpu/turboquant/TurboQuantContext.h"
#include "../../../kernels/cpu/turboquant/TurboQuantQuantizeTQ8.h"
#include "../../../kernels/cpu/turboquant/TurboQuantQuantizeTQ4.h"
#include "../../../kernels/cpu/rotation/ActivationRotation.h"
#include "../../../utils/OpenMPUtils.h"

#include "../../../utils/KVCacheProfiler.h"
#include "../../../tensors/GpuTensorView.h"

#include <immintrin.h>
#include <cstring>
#include <chrono>

namespace
{
    static size_t estimateTensorAppendBytes(const llaminar2::ITensor *tensor, int num_tokens)
    {
        if (!tensor || num_tokens <= 0)
        {
            return 0;
        }

        const auto &shape = tensor->shape();
        if (shape.empty())
        {
            return 0;
        }

        const size_t rows = shape[0];
        if (rows == 0)
        {
            return 0;
        }

        const size_t bytes_per_row = tensor->size_bytes() / rows;
        return static_cast<size_t>(num_tokens) * bytes_per_row;
    }

    static size_t elementSizeForTensorType(llaminar2::TensorType t)
    {
        using llaminar2::TensorType;
        switch (t)
        {
        case TensorType::FP32:
            return sizeof(float);
        case TensorType::FP16:
        case TensorType::BF16:
            return sizeof(uint16_t);
        case TensorType::Q8_1:
            return sizeof(llaminar2::Q8_1Block);
        default:
            return 0;
        }
    }
}

namespace llaminar2
{

    // =============================================================================
    // KVCacheAppendStage Implementation
    // =============================================================================

    KVCacheAppendStage::KVCacheAppendStage(Params params)
        : IComputeStage(params.device_id), params_(std::move(params))
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

        // Determine the graph-shaped token count first, then narrow to the real
        // prefix for non-captured padded execution. Captured prefill records the
        // fixed bucket kernel shape and relies on onGraphReplayed() to advance
        // host metadata by the real prefix after the graph launch.
        int total_tokens = params_.num_tokens;
        if (total_tokens <= 0)
        {
            total_tokens = static_cast<int>(params_.K->shape()[0]);
        }
        const int bucket_tokens = total_tokens;
        if (!isGraphCaptureActive() && params_.batch_size <= 1 && replay_advance_tokens_ > 0)
        {
            if (replay_advance_tokens_ > bucket_tokens)
            {
                LOG_ERROR("[KVCacheAppendStage] Replay token count exceeds bucket token count: real="
                          << replay_advance_tokens_ << " bucket=" << bucket_tokens);
                return false;
            }
            total_tokens = replay_advance_tokens_;
        }

        auto append_to_cache = [&](int seq_idx,
                                   const ITensor *k_tensor,
                                   const ITensor *v_tensor,
                                   int num_tokens) -> bool
        {
            if (!k_tensor || !v_tensor)
            {
                LOG_ERROR("[KVCacheAppendStage] append_to_cache received null tensor");
                return false;
            }

            const auto start = std::chrono::high_resolution_clock::now();

            bool success = false;
            void *stream = gpuStream();
            if (stream || params_.device_id.is_gpu())
            {
                // GPU path: fine-grained profiling handled inside appendWithStream
                success = params_.kv_cache->appendWithStream(
                    params_.layer_idx, seq_idx,
                    k_tensor, v_tensor, num_tokens, stream);
            }
            else
            {
                success = params_.kv_cache->append(
                    params_.layer_idx, seq_idx,
                    k_tensor, v_tensor, num_tokens);
            }

            const auto end = std::chrono::high_resolution_clock::now();
            const uint64_t duration_ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());

            if (success && !(stream || params_.device_id.is_gpu()))
            {
                // CPU path only: record APPEND here (GPU path records internally)
                const uint64_t bytes = static_cast<uint64_t>(
                    estimateTensorAppendBytes(k_tensor, num_tokens) +
                    estimateTensorAppendBytes(v_tensor, num_tokens));
                const uint64_t tokens = static_cast<uint64_t>(num_tokens);
                KVCacheProfiler::record(KVCacheOpType::APPEND, duration_ns, tokens, bytes);
            }

            if (success &&
                isGraphCaptureActive() &&
                isGraphCaptureHostBookkeepingActive() &&
                params_.kv_cache->isGraphCaptureReady())
            {
                // Segmented decode capture records GPU append kernels, but the
                // immediate launch-after-capture deliberately skips
                // onGraphReplayed(). Advance host metadata during recording so
                // subsequent captured stages see the appended token in
                // get_cached_tokens()/dynamic dequant params, matching normal
                // execution. Prefill capture and collective Phase-2 capture keep
                // this guard disabled and continue to use replay callbacks or
                // real post-capture execution instead.
                params_.kv_cache->advanceHead(params_.layer_idx, seq_idx, num_tokens);
            }

            return success;
        };

        // Determine batch handling mode
        const int batch_size = params_.batch_size;
        const int seq_len = params_.seq_len;

        // If batch_size > 1 and seq_len > 0, do per-sequence append
        // K/V layout: [batch_size * seq_len, kv_dim] - contiguous per-sequence
        if (batch_size > 1 && seq_len > 0)
        {
            const size_t kv_dim = params_.K->shape().size() > 1 ? params_.K->shape()[1] : 0;

            if (params_.device_id.is_gpu())
            {
                const auto target = params_.device_id;
                const auto k_type = params_.K->native_type();
                const auto v_type = params_.V->native_type();

                const size_t k_elem_bytes = elementSizeForTensorType(k_type);
                const size_t v_elem_bytes = elementSizeForTensorType(v_type);
                if (k_elem_bytes == 0 || v_elem_bytes == 0)
                {
                    LOG_ERROR("[KVCacheAppendStage] Unsupported tensor type for GPU batched append: K="
                              << params_.K->dtype_name() << " V=" << params_.V->dtype_name());
                    return false;
                }

                const size_t k_cols = (k_type == TensorType::Q8_1)
                                          ? ((kv_dim + Q8_1Block::BLOCK_SIZE - 1) / Q8_1Block::BLOCK_SIZE)
                                          : kv_dim;
                const size_t v_cols = (v_type == TensorType::Q8_1)
                                          ? ((kv_dim + Q8_1Block::BLOCK_SIZE - 1) / Q8_1Block::BLOCK_SIZE)
                                          : kv_dim;

                const size_t k_seq_bytes = static_cast<size_t>(seq_len) * k_cols * k_elem_bytes;
                const size_t v_seq_bytes = static_cast<size_t>(seq_len) * v_cols * v_elem_bytes;

                auto *k_mut = const_cast<ITensor *>(params_.K);
                auto *v_mut = const_cast<ITensor *>(params_.V);
                if (!params_.K->gpu_data_ptr() && !k_mut->ensureOnDevice(target))
                {
                    LOG_ERROR("[KVCacheAppendStage] Failed to ensure K on GPU for batched append");
                    return false;
                }
                if (!params_.V->gpu_data_ptr() && !v_mut->ensureOnDevice(target))
                {
                    LOG_ERROR("[KVCacheAppendStage] Failed to ensure V on GPU for batched append");
                    return false;
                }

                const auto *k_base_ptr = static_cast<const uint8_t *>(params_.K->gpu_data_ptr());
                const auto *v_base_ptr = static_cast<const uint8_t *>(params_.V->gpu_data_ptr());
                if (!k_base_ptr || !v_base_ptr)
                {
                    LOG_ERROR("[KVCacheAppendStage] Missing GPU pointers for batched append");
                    return false;
                }

                const int gpu_ordinal = target.gpu_ordinal();
                for (int b = 0; b < batch_size; ++b)
                {
                    const int seq_idx = params_.seq_idx + b;
                    void *k_seq_ptr = const_cast<uint8_t *>(k_base_ptr + static_cast<size_t>(b) * k_seq_bytes);
                    void *v_seq_ptr = const_cast<uint8_t *>(v_base_ptr + static_cast<size_t>(b) * v_seq_bytes);

                    GpuTensorView k_view(k_seq_ptr, static_cast<size_t>(seq_len), k_cols, k_type, gpu_ordinal);
                    GpuTensorView v_view(v_seq_ptr, static_cast<size_t>(seq_len), v_cols, v_type, gpu_ordinal);

                    if (!append_to_cache(seq_idx, &k_view, &v_view, seq_len))
                    {
                        LOG_ERROR("[KVCacheAppendStage] append failed for GPU batch index " << b);
                        return false;
                    }
                }

                return true;
            }

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

                bool success = false;
                const ActivationPrecision cache_precision = params_.kv_cache->precision();
                if (cache_precision == ActivationPrecision::FP16)
                {
                    const auto conv_start = std::chrono::high_resolution_clock::now();

                    auto k_fp16 = std::make_unique<FP16Tensor>(
                        std::vector<size_t>{static_cast<size_t>(seq_len), kv_dim});
                    auto v_fp16 = std::make_unique<FP16Tensor>(
                        std::vector<size_t>{static_cast<size_t>(seq_len), kv_dim});

                    k_fp16->from_fp32(k_slice->data(), static_cast<size_t>(seq_len) * kv_dim);
                    v_fp16->from_fp32(v_slice->data(), static_cast<size_t>(seq_len) * kv_dim);

                    if (params_.device_id.is_gpu())
                    {
                        if (!k_fp16->ensureOnDevice(params_.device_id) ||
                            !v_fp16->ensureOnDevice(params_.device_id))
                        {
                            LOG_ERROR("[KVCacheAppendStage] Failed to upload FP16 converted K/V slices to GPU");
                            return false;
                        }
                    }

                    const auto conv_end = std::chrono::high_resolution_clock::now();
                    const uint64_t conv_ns = static_cast<uint64_t>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(conv_end - conv_start).count());
                    const uint64_t conv_bytes = static_cast<uint64_t>(
                        estimateTensorAppendBytes(k_fp16.get(), seq_len) +
                        estimateTensorAppendBytes(v_fp16.get(), seq_len));
                    KVCacheProfiler::record(KVCacheOpType::CONVERT_TO_FP16, conv_ns, static_cast<uint64_t>(seq_len), conv_bytes);

                    success = append_to_cache(seq_idx, k_fp16.get(), v_fp16.get(), seq_len);
                }
                else if (cache_precision == ActivationPrecision::Q8_1)
                {
                    const auto conv_start = std::chrono::high_resolution_clock::now();

                    auto k_q8 = Q8_1Tensor::quantize_from_fp32(
                        k_slice->data(),
                        {static_cast<size_t>(seq_len), kv_dim});
                    auto v_q8 = Q8_1Tensor::quantize_from_fp32(
                        v_slice->data(),
                        {static_cast<size_t>(seq_len), kv_dim});

                    if (!k_q8 || !v_q8)
                    {
                        LOG_ERROR("[KVCacheAppendStage] Failed to quantize batched K/V slices to Q8_1");
                        return false;
                    }

                    if (params_.device_id.is_gpu())
                    {
                        if (!k_q8->ensureOnDevice(params_.device_id) ||
                            !v_q8->ensureOnDevice(params_.device_id))
                        {
                            LOG_ERROR("[KVCacheAppendStage] Failed to upload Q8_1 converted K/V slices to GPU");
                            return false;
                        }
                    }

                    const auto conv_end = std::chrono::high_resolution_clock::now();
                    const uint64_t conv_ns = static_cast<uint64_t>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(conv_end - conv_start).count());
                    const uint64_t conv_bytes = static_cast<uint64_t>(
                        estimateTensorAppendBytes(k_q8.get(), seq_len) +
                        estimateTensorAppendBytes(v_q8.get(), seq_len));
                    KVCacheProfiler::record(KVCacheOpType::CONVERT_TO_Q8_1, conv_ns, static_cast<uint64_t>(seq_len), conv_bytes);

                    success = append_to_cache(seq_idx, k_q8.get(), v_q8.get(), seq_len);
                }
                else if (cache_precision == ActivationPrecision::TQ4)
                {
                    if (!params_.turboquant_ctx)
                    {
                        LOG_ERROR("[KVCacheAppendStage] TurboQuant cache requires turboquant_ctx in params");
                        return false;
                    }
                    const auto &turboquant_ctx = params_.turboquant_ctx->for_layer(params_.layer_idx);
                    const std::vector<size_t> tq_shape{static_cast<size_t>(seq_len), kv_dim};

                    auto k_tq4 = TQ4Tensor::quantize_from_fp32(k_slice->data(), tq_shape, params_.head_dim, turboquant_ctx);
                    auto v_tq4 = TQ4Tensor::quantize_from_fp32(v_slice->data(), tq_shape, params_.head_dim, turboquant_ctx);
                    if (!k_tq4 || !v_tq4)
                    {
                        LOG_ERROR("[KVCacheAppendStage] Failed to quantize batched K/V slices to TQ4");
                        return false;
                    }

                    success = append_to_cache(seq_idx, k_tq4.get(), v_tq4.get(), seq_len);
                }
                else if (cache_precision == ActivationPrecision::TQ8)
                {
                    // Split TQ: TQ8 for K, TQ4 for V
                    if (!params_.turboquant_ctx)
                    {
                        LOG_ERROR("[KVCacheAppendStage] Split TQ cache requires turboquant_ctx in params");
                        return false;
                    }
                    const auto &turboquant_ctx = params_.turboquant_ctx->for_layer(params_.layer_idx);
                    const std::vector<size_t> tq_shape{static_cast<size_t>(seq_len), kv_dim};

                    auto k_tq8 = TQ8Tensor::quantize_from_fp32(k_slice->data(), tq_shape, params_.head_dim, turboquant_ctx);
                    auto v_tq4 = TQ4Tensor::quantize_from_fp32(v_slice->data(), tq_shape, params_.head_dim, turboquant_ctx);
                    if (!k_tq8 || !v_tq4)
                    {
                        LOG_ERROR("[KVCacheAppendStage] Failed to quantize batched K/V slices to split TQ");
                        return false;
                    }

                    success = append_to_cache(seq_idx, k_tq8.get(), v_tq4.get(), seq_len);
                }
                else
                {
                    success = append_to_cache(seq_idx, k_slice.get(), v_slice.get(), seq_len);
                }

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
        bool cache_is_fp16 = (params_.kv_cache->precision() == ActivationPrecision::FP16);
        bool cache_is_q8_1 = (params_.kv_cache->precision() == ActivationPrecision::Q8_1);
        bool k_is_fp32 = (params_.K->native_type() == TensorType::FP32);
        bool v_is_fp32 = (params_.V->native_type() == TensorType::FP32);
        const bool has_gpu_inputs = (params_.K->gpu_data_ptr() != nullptr && params_.V->gpu_data_ptr() != nullptr);

        // If cache is FP32 but inputs are not, convert to FP32 for cache append
        if (cache_is_fp32 && (!k_is_fp32 || !v_is_fp32))
        {
            const auto conv_start = std::chrono::high_resolution_clock::now();

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

            const auto conv_end = std::chrono::high_resolution_clock::now();
            const uint64_t conv_ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(conv_end - conv_start).count());
            const uint64_t conv_bytes = static_cast<uint64_t>(
                estimateTensorAppendBytes(k_slice.get(), total_tokens) +
                estimateTensorAppendBytes(v_slice.get(), total_tokens));
            KVCacheProfiler::record(KVCacheOpType::CONVERT_TO_FP32, conv_ns, static_cast<uint64_t>(total_tokens), conv_bytes);

            bool success = append_to_cache(
                params_.seq_idx,
                k_slice.get(), v_slice.get(), total_tokens);

            if (!success)
            {
                LOG_ERROR("[KVCacheAppendStage] append failed (after conversion)");
                return false;
            }

            return true;
        }

        // If cache is FP16 but inputs are not, convert K/V to FP16 for cache append
        if (!has_gpu_inputs && cache_is_fp16 && (params_.K->native_type() != TensorType::FP16 || params_.V->native_type() != TensorType::FP16))
        {
            const auto conv_start = std::chrono::high_resolution_clock::now();

            const size_t kv_dim = params_.K->shape().size() > 1 ? params_.K->shape()[1] : 0;

            const std::vector<size_t> fp16_shape{static_cast<size_t>(total_tokens), kv_dim};
            if (!fp16_k_scratch_ || fp16_k_scratch_->shape() != fp16_shape)
            {
                fp16_k_scratch_ = std::make_unique<FP16Tensor>(fp16_shape);
            }
            if (!fp16_v_scratch_ || fp16_v_scratch_->shape() != fp16_shape)
            {
                fp16_v_scratch_ = std::make_unique<FP16Tensor>(fp16_shape);
            }

            const float *k_fp32 = params_.K->fp32_data();
            const float *v_fp32 = params_.V->fp32_data();
            if (!k_fp32 || !v_fp32)
            {
                LOG_ERROR("[KVCacheAppendStage] Cannot get FP32 data for FP16 cache conversion");
                return false;
            }

            fp16_k_scratch_->from_fp32(k_fp32, static_cast<size_t>(total_tokens) * kv_dim);
            fp16_v_scratch_->from_fp32(v_fp32, static_cast<size_t>(total_tokens) * kv_dim);

            if (params_.device_id.is_gpu())
            {
                if (!fp16_k_scratch_->ensureOnDevice(params_.device_id) ||
                    !fp16_v_scratch_->ensureOnDevice(params_.device_id))
                {
                    LOG_ERROR("[KVCacheAppendStage] Failed to upload FP16 converted K/V tensors to GPU");
                    return false;
                }
            }

            const auto conv_end = std::chrono::high_resolution_clock::now();
            const uint64_t conv_ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(conv_end - conv_start).count());
            const uint64_t conv_bytes = static_cast<uint64_t>(
                estimateTensorAppendBytes(fp16_k_scratch_.get(), total_tokens) +
                estimateTensorAppendBytes(fp16_v_scratch_.get(), total_tokens));
            KVCacheProfiler::record(KVCacheOpType::CONVERT_TO_FP16, conv_ns, static_cast<uint64_t>(total_tokens), conv_bytes);

            bool success = append_to_cache(params_.seq_idx, fp16_k_scratch_.get(), fp16_v_scratch_.get(), total_tokens);
            if (!success)
            {
                LOG_ERROR("[KVCacheAppendStage] append failed (FP16 cache conversion path)");
                return false;
            }

            return true;
        }

        // If cache is Q8_1 but inputs are not, convert K/V to Q8_1 for cache append
        if (!has_gpu_inputs && cache_is_q8_1 && (params_.K->native_type() != TensorType::Q8_1 || params_.V->native_type() != TensorType::Q8_1))
        {
            const auto conv_start = std::chrono::high_resolution_clock::now();

            const size_t kv_dim = params_.K->shape().size() > 1 ? params_.K->shape()[1] : 0;

            const std::vector<size_t> q8_shape{static_cast<size_t>(total_tokens), kv_dim};
            if (!q8_k_scratch_ || q8_k_scratch_->shape() != q8_shape)
            {
                q8_k_scratch_ = std::make_unique<Q8_1Tensor>(q8_shape);
            }
            if (!q8_v_scratch_ || q8_v_scratch_->shape() != q8_shape)
            {
                q8_v_scratch_ = std::make_unique<Q8_1Tensor>(q8_shape);
            }

            const float *k_fp32 = params_.K->fp32_data();
            const float *v_fp32 = params_.V->fp32_data();
            if (!k_fp32 || !v_fp32)
            {
                LOG_ERROR("[KVCacheAppendStage] Cannot get FP32 data for Q8_1 cache conversion");
                return false;
            }

            const size_t total_elements = static_cast<size_t>(total_tokens) * kv_dim;
            const size_t total_blocks = (total_elements + Q8_1Block::BLOCK_SIZE - 1) / Q8_1Block::BLOCK_SIZE;

            // Decode path hot loop: tiny workloads benefit from direct quantization
            // over both K and V in one loop to reduce helper/dispatch overhead.
            if (total_blocks <= 32)
            {
                Q8_1Block *k_blocks = q8_k_scratch_->mutable_typed_data();
                Q8_1Block *v_blocks = q8_v_scratch_->mutable_typed_data();
                if (!k_blocks || !v_blocks)
                {
                    LOG_ERROR("[KVCacheAppendStage] Failed to access mutable Q8_1 scratch blocks");
                    return false;
                }

                for (size_t block_idx = 0; block_idx < total_blocks; ++block_idx)
                {
                    const size_t offset = block_idx * Q8_1Block::BLOCK_SIZE;
                    const int count = static_cast<int>(std::min<size_t>(Q8_1Block::BLOCK_SIZE, total_elements - offset));
                    simd::quantize_single_block(k_fp32 + offset, k_blocks[block_idx], count);
                    simd::quantize_single_block(v_fp32 + offset, v_blocks[block_idx], count);
                }
            }
            else
            {
                if (!q8_k_scratch_->copyFrom_fp32_rows(k_fp32, static_cast<size_t>(total_tokens)) ||
                    !q8_v_scratch_->copyFrom_fp32_rows(v_fp32, static_cast<size_t>(total_tokens)))
                {
                    LOG_ERROR("[KVCacheAppendStage] Failed to quantize K/V for Q8_1 cache append (in-place)");
                    return false;
                }
            }

            if (params_.device_id.is_gpu())
            {
                if (!q8_k_scratch_->ensureOnDevice(params_.device_id) ||
                    !q8_v_scratch_->ensureOnDevice(params_.device_id))
                {
                    LOG_ERROR("[KVCacheAppendStage] Failed to upload Q8_1 converted K/V tensors to GPU");
                    return false;
                }
            }

            const auto conv_end = std::chrono::high_resolution_clock::now();
            const uint64_t conv_ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(conv_end - conv_start).count());
            const uint64_t conv_bytes = static_cast<uint64_t>(
                estimateTensorAppendBytes(q8_k_scratch_.get(), total_tokens) +
                estimateTensorAppendBytes(q8_v_scratch_.get(), total_tokens));
            KVCacheProfiler::record(KVCacheOpType::CONVERT_TO_Q8_1, conv_ns, static_cast<uint64_t>(total_tokens), conv_bytes);

            bool success = append_to_cache(params_.seq_idx, q8_k_scratch_.get(), q8_v_scratch_.get(), total_tokens);
            if (!success)
            {
                LOG_ERROR("[KVCacheAppendStage] append failed (Q8_1 cache conversion path)");
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
            const auto conv_start = std::chrono::high_resolution_clock::now();

            const size_t kv_dim = params_.K->shape().size() > 1 ? params_.K->shape()[1] : 0;
            const float kv_cache_scale_k = params_.kv_cache_scale_k;
            const float kv_cache_scale_v = params_.kv_cache_scale_v;
            const int head_dim = params_.head_dim;

            // -----------------------------------------------------------------
            // KV rotation helper: reduces activation kurtosis before Q16_1
            // quantization. Copies const FP32 data → scratch, rotates in-place,
            // returns pointer to rotated data. No-op when kv_rotation is null.
            // For mutable buffers (e.g. v_fp32_temp), use rotate_mutable instead.
            // -----------------------------------------------------------------
            const auto *kv_rot = params_.kv_rotation;

            // Copy const FP32 → scratch → rotate → return scratch pointer
            auto rotate_const = [&](const float *src, size_t n_elements) -> const float *
            {
                if (!kv_rot)
                    return src;
                const size_t needed = n_elements;
                if (kv_rotation_scratch_.size() < needed)
                    kv_rotation_scratch_.resize(needed);
                std::memcpy(kv_rotation_scratch_.data(), src, n_elements * sizeof(float));
                kv_rot->rotate_rows_inplace(kv_rotation_scratch_.data(),
                                            static_cast<int>(n_elements / kv_dim),
                                            static_cast<int>(kv_dim));
                return kv_rotation_scratch_.data();
            };

            // Rotate mutable FP32 buffer in-place
            auto rotate_mutable = [&](float *data, int n_rows)
            {
                if (!kv_rot)
                    return;
                kv_rot->rotate_rows_inplace(data, n_rows, static_cast<int>(kv_dim));
            };

            // VNNI-safe clipping limit: floor(sqrt(INT32_MAX / (head_dim/16)))
            // Prevents INT32 overflow during VPDPWSSD accumulation.
            const int16_t max_safe_int16 = (head_dim <= 64) ? 23170 : (head_dim <= 96) ? 18918
                                                                  : (head_dim <= 128)  ? 16383
                                                                  : (head_dim <= 192)  ? 13377
                                                                                       : 11585;

            LOG_DEBUG("[KVCacheAppendStage] Q16_1 cache with VNNI-safe fixed-scale quantization"
                      << " (scale_k=" << kv_cache_scale_k
                      << ", scale_v=" << kv_cache_scale_v
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

                // Apply block-diagonal rotation to reduce kurtosis before Q16_1
                k_fp32 = rotate_const(k_fp32, static_cast<size_t>(total_tokens) * kv_dim);

                // Use fixed-scale quantization with VNNI-safe clipping
                if (!k_q16_owned->copyFrom_fp32_fixed_scale(k_fp32, kv_cache_scale_k, head_dim))
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

                // Apply block-diagonal rotation to reduce kurtosis before Q16_1
                k_fp32 = rotate_const(k_fp32, static_cast<size_t>(total_tokens) * kv_dim);

                if (!k_q16_owned->copyFrom_fp32_fixed_scale(k_fp32, kv_cache_scale_k, head_dim))
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

                // Apply block-diagonal rotation to reduce kurtosis before Q16_1
                v_fp32 = rotate_const(v_fp32, static_cast<size_t>(total_tokens) * kv_dim);

                if (!v_q16_owned->copyFrom_fp32_fixed_scale(v_fp32, kv_cache_scale_v, head_dim))
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

                // Apply block-diagonal rotation to reduce kurtosis before Q16_1
                rotate_mutable(v_fp32_temp.data(), total_tokens);

                // Requantize with fixed scale and VNNI-safe clipping
                if (!v_q16_owned->copyFrom_fp32_fixed_scale(v_fp32_temp.data(), kv_cache_scale_v, head_dim))
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

                // Apply block-diagonal rotation to reduce kurtosis before Q16_1
                v_fp32 = rotate_const(v_fp32, static_cast<size_t>(total_tokens) * kv_dim);

                if (!v_q16_owned->copyFrom_fp32_fixed_scale(v_fp32, kv_cache_scale_v, head_dim))
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

            const auto conv_end = std::chrono::high_resolution_clock::now();
            const uint64_t conv_ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(conv_end - conv_start).count());
            const uint64_t conv_bytes = static_cast<uint64_t>(
                estimateTensorAppendBytes(k_for_cache, total_tokens) +
                estimateTensorAppendBytes(v_for_cache, total_tokens));
            KVCacheProfiler::record(KVCacheOpType::CONVERT_TO_Q16_1, conv_ns, static_cast<uint64_t>(total_tokens), conv_bytes);

            bool success = append_to_cache(
                params_.seq_idx,
                k_for_cache, v_for_cache, total_tokens);

            if (!success)
            {
                LOG_ERROR("[KVCacheAppendStage] append failed (Q16_1 cache)");
                return false;
            }

            return true;
        }

        // =================================================================
        // TQ4 cache path with TurboQuant rotation-based quantization
        // =================================================================
        bool cache_is_tq4 = (params_.kv_cache->precision() == ActivationPrecision::TQ4);

        if (cache_is_tq4)
        {
            if (!params_.turboquant_ctx)
            {
                LOG_ERROR("[KVCacheAppendStage] TurboQuant cache requires turboquant_ctx in params");
                return false;
            }

            // GPU fast path: pass FP32 directly for GPU-side quantization
            if (params_.device_id.is_gpu())
            {
                const auto conv_start = std::chrono::high_resolution_clock::now();
                bool success = append_to_cache(params_.seq_idx, params_.K, params_.V, total_tokens);
                const auto conv_end = std::chrono::high_resolution_clock::now();
                const uint64_t conv_ns = static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(conv_end - conv_start).count());
                KVCacheProfiler::record(KVCacheOpType::CONVERT_TO_TQ, conv_ns, static_cast<uint64_t>(total_tokens), 0);
                if (!success)
                {
                    LOG_ERROR("[KVCacheAppendStage] append failed (TQ4 GPU quantize path)");
                    return false;
                }
                return true;
            }

            // CPU path: quantize on CPU, then upload blocks to cache
            const auto conv_start = std::chrono::high_resolution_clock::now();
            const size_t kv_dim = params_.K->shape().size() > 1 ? params_.K->shape()[1] : 0;
            const int head_dim = params_.head_dim;
            const auto &turboquant_ctx = params_.turboquant_ctx->for_layer(params_.layer_idx);

            const float *k_fp32 = params_.K->fp32_data();
            const float *v_fp32 = params_.V->fp32_data();
            if (!k_fp32 || !v_fp32)
            {
                LOG_ERROR("[KVCacheAppendStage] Cannot get FP32 data for TurboQuant cache conversion");
                return false;
            }

            const std::vector<size_t> tq_shape{static_cast<size_t>(total_tokens), kv_dim};

            // --- Pre-allocate scratch tensors (reuse across calls) ---
            if (!tq4_k_scratch_ || tq4_k_scratch_->shape() != tq_shape)
            {
                tq4_k_scratch_ = std::make_shared<TQ4Tensor>(tq_shape, head_dim);
                tq4_k_scratch_->set_turboquant_context(&turboquant_ctx);
            }
            if (!tq4_v_scratch_ || tq4_v_scratch_->shape() != tq_shape)
            {
                tq4_v_scratch_ = std::make_shared<TQ4Tensor>(tq_shape, head_dim);
                tq4_v_scratch_->set_turboquant_context(&turboquant_ctx);
            }

            const size_t bpr = tq4_k_scratch_->blocks_per_row();

            // --- Decode fast path: fused K+V quantization per head ---
            if (total_tokens <= 2)
            {
                const TurboQuantContext *head_ctx_ptrs[16];
                for (size_t h = 0; h < bpr && h < 16; ++h)
                    head_ctx_ptrs[h] = &turboquant_ctx.for_layer(static_cast<int>(h));

                const size_t bb = tq4_k_scratch_->block_bytes();
                uint8_t *k_raw = static_cast<uint8_t *>(tq4_k_scratch_->raw_mutable_data());
                uint8_t *v_raw = static_cast<uint8_t *>(tq4_v_scratch_->raw_mutable_data());

                for (size_t r = 0; r < static_cast<size_t>(total_tokens); ++r)
                {
                    const float *k_row = k_fp32 + r * kv_dim;
                    const float *v_row = v_fp32 + r * kv_dim;
                    uint8_t *k_row_dst = k_raw + r * bpr * bb;
                    uint8_t *v_row_dst = v_raw + r * bpr * bb;
                    alignas(64) float scratch0[128];
                    alignas(64) float scratch1[128];

                    for (size_t h = 0; h < bpr; ++h)
                    {
                        const float *k_head = k_row + h * static_cast<size_t>(head_dim);
                        const float *v_head = v_row + h * static_cast<size_t>(head_dim);
                        const auto &hctx = *head_ctx_ptrs[h];

                        if (head_dim == 128)
                        {
                            auto *k_block = reinterpret_cast<TQ4Block_128 *>(k_row_dst + h * bb);
                            turboquant_quantize_tq4<128>(k_head, hctx, *k_block, scratch0, scratch1);
                            auto *v_block = reinterpret_cast<TQ4Block_128 *>(v_row_dst + h * bb);
                            turboquant_quantize_tq4<128>(v_head, hctx, *v_block, scratch0, scratch1);
                        }
                        else
                        {
                            auto *k_block = reinterpret_cast<TQ4Block_64 *>(k_row_dst + h * bb);
                            turboquant_quantize_tq4<64>(k_head, hctx, *k_block, scratch0, scratch1);
                            auto *v_block = reinterpret_cast<TQ4Block_64 *>(v_row_dst + h * bb);
                            turboquant_quantize_tq4<64>(v_head, hctx, *v_block, scratch0, scratch1);
                        }
                    }
                }
            }
            else
            {
                tq4_k_scratch_->copyFrom_fp32_rows(k_fp32, static_cast<size_t>(total_tokens), turboquant_ctx);
                tq4_v_scratch_->copyFrom_fp32_rows(v_fp32, static_cast<size_t>(total_tokens), turboquant_ctx);
            }

            const auto conv_end = std::chrono::high_resolution_clock::now();
            const uint64_t conv_ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(conv_end - conv_start).count());
            KVCacheProfiler::record(KVCacheOpType::CONVERT_TO_TQ, conv_ns, static_cast<uint64_t>(total_tokens), 0);

            bool success = append_to_cache(params_.seq_idx, tq4_k_scratch_.get(), tq4_v_scratch_.get(), total_tokens);
            if (!success)
            {
                LOG_ERROR("[KVCacheAppendStage] append failed (TQ4 cache)");
                return false;
            }

            return true;
        }

        // =================================================================
        // Split TQ cache path: TQ8 for K, TQ4 for V
        // =================================================================
        bool cache_is_tq8 = (params_.kv_cache->k_precision() == ActivationPrecision::TQ8);

        if (cache_is_tq8)
        {
            if (!params_.turboquant_ctx)
            {
                LOG_ERROR("[KVCacheAppendStage] Split TQ cache requires turboquant_ctx in params");
                return false;
            }

            // =============================================================
            // GPU fast path: pass FP32 K/V directly to the TQ cache.
            // The GPU cache (CUDARingKVCacheTQ / ROCmRingKVCacheTQ) has
            // built-in GPU quantize kernels that avoid the catastrophic
            // D2H → CPU quant → H2D round-trip.
            // =============================================================
            if (params_.device_id.is_gpu())
            {
                const auto conv_start = std::chrono::high_resolution_clock::now();

                // K/V are already on GPU from upstream stages (QKV proj, RoPE).
                // Pass them directly — appendWithStream() will detect FP32 and
                // use GPU quantize kernels (tq8_quantize_kernel + tq4_quantize_kernel).
                bool success = append_to_cache(params_.seq_idx, params_.K, params_.V, total_tokens);

                const auto conv_end = std::chrono::high_resolution_clock::now();
                const uint64_t conv_ns = static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(conv_end - conv_start).count());
                KVCacheProfiler::record(KVCacheOpType::CONVERT_TO_TQ, conv_ns, static_cast<uint64_t>(total_tokens), 0);

                if (!success)
                {
                    LOG_ERROR("[KVCacheAppendStage] append failed (split TQ GPU quantize path)");
                    return false;
                }
                return true;
            }

            // =============================================================
            // CPU path: quantize on CPU, then upload blocks to cache
            // =============================================================
            const auto conv_start = std::chrono::high_resolution_clock::now();
            const size_t kv_dim = params_.K->shape().size() > 1 ? params_.K->shape()[1] : 0;
            const int head_dim = params_.head_dim;
            const auto &turboquant_ctx = params_.turboquant_ctx->for_layer(params_.layer_idx);

            const float *k_fp32 = params_.K->fp32_data();
            const float *v_fp32 = params_.V->fp32_data();
            if (!k_fp32 || !v_fp32)
            {
                LOG_ERROR("[KVCacheAppendStage] Cannot get FP32 data for split TQ cache conversion");
                return false;
            }

            const std::vector<size_t> tq_shape{static_cast<size_t>(total_tokens), kv_dim};

            // --- Pre-allocate scratch tensors (reuse across calls) ---
            if (!tq8_k_scratch_ || tq8_k_scratch_->shape() != tq_shape)
            {
                tq8_k_scratch_ = std::make_shared<TQ8Tensor>(tq_shape, head_dim);
                tq8_k_scratch_->set_turboquant_context(&turboquant_ctx);
            }
            if (!tq4_v_scratch_ || tq4_v_scratch_->shape() != tq_shape)
            {
                tq4_v_scratch_ = std::make_shared<TQ4Tensor>(tq_shape, head_dim);
                tq4_v_scratch_->set_turboquant_context(&turboquant_ctx);
            }

            const size_t bpr = tq8_k_scratch_->blocks_per_row();

            // --- Decode fast path: fused K+V quantization per head ---
            // Interleaves TQ8(K) and TQ4(V) for the same head so the 64KB
            // rotation matrix stays hot in L1/L2 for both operations.
            // Also pre-resolves per-head contexts to avoid mutex+hashmap per call.
            if (total_tokens <= 2)
            {
                // Pre-resolve all per-head contexts once (avoids mutex per head)
                const TurboQuantContext *head_ctx_ptrs[16]; // max 16 KV heads
                for (size_t h = 0; h < bpr && h < 16; ++h)
                    head_ctx_ptrs[h] = &turboquant_ctx.for_layer(static_cast<int>(h));

                const size_t k_bb = tq8_k_scratch_->block_bytes();
                const size_t v_bb = tq4_v_scratch_->block_bytes();
                uint8_t *k_raw = static_cast<uint8_t *>(tq8_k_scratch_->raw_mutable_data());
                uint8_t *v_raw = static_cast<uint8_t *>(tq4_v_scratch_->raw_mutable_data());

                for (size_t r = 0; r < static_cast<size_t>(total_tokens); ++r)
                {
                    const float *k_row = k_fp32 + r * kv_dim;
                    const float *v_row = v_fp32 + r * kv_dim;
                    uint8_t *k_row_dst = k_raw + r * bpr * k_bb;
                    uint8_t *v_row_dst = v_raw + r * bpr * v_bb;
                    alignas(64) float scratch0[128];
                    alignas(64) float scratch1[128];

                    for (size_t h = 0; h < bpr; ++h)
                    {
                        const float *k_head = k_row + h * static_cast<size_t>(head_dim);
                        const float *v_head = v_row + h * static_cast<size_t>(head_dim);
                        const auto &hctx = *head_ctx_ptrs[h];

                        // K (TQ8) — rotation matrix loaded into cache
                        if (head_dim == 128)
                        {
                            auto *k_block = reinterpret_cast<TQ8Block_128 *>(k_row_dst + h * k_bb);
                            turboquant_quantize_tq8<128>(k_head, hctx, *k_block, scratch0, scratch1);
                            // V (TQ4) — same rotation matrix still hot in L1/L2
                            auto *v_block = reinterpret_cast<TQ4Block_128 *>(v_row_dst + h * v_bb);
                            turboquant_quantize_tq4<128>(v_head, hctx, *v_block, scratch0, scratch1);
                        }
                        else
                        {
                            auto *k_block = reinterpret_cast<TQ8Block_64 *>(k_row_dst + h * k_bb);
                            turboquant_quantize_tq8<64>(k_head, hctx, *k_block, scratch0, scratch1);
                            auto *v_block = reinterpret_cast<TQ4Block_64 *>(v_row_dst + h * v_bb);
                            turboquant_quantize_tq4<64>(v_head, hctx, *v_block, scratch0, scratch1);
                        }
                    }
                }
            }
            else
            {
                // Prefill path: use existing parallel quantization
                tq8_k_scratch_->copyFrom_fp32_rows(k_fp32, static_cast<size_t>(total_tokens), turboquant_ctx);
                tq4_v_scratch_->copyFrom_fp32_rows(v_fp32, static_cast<size_t>(total_tokens), turboquant_ctx);
            }

            const auto conv_end = std::chrono::high_resolution_clock::now();
            const uint64_t conv_ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(conv_end - conv_start).count());
            KVCacheProfiler::record(KVCacheOpType::CONVERT_TO_TQ, conv_ns, static_cast<uint64_t>(total_tokens), 0);

            bool success = append_to_cache(params_.seq_idx, tq8_k_scratch_.get(), tq4_v_scratch_.get(), total_tokens);
            if (!success)
            {
                LOG_ERROR("[KVCacheAppendStage] append failed (split TQ cache: TQ8 K + TQ4 V)");
                return false;
            }

            return true;
        }

        // Direct append path - tensors already match cache precision
        // Cast ITensor* to TensorBase* for append_kv
        bool success = append_to_cache(
            params_.seq_idx,
            params_.K, params_.V, total_tokens);

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

    StageBufferContract KVCacheAppendStage::bufferContract() const
    {
        if (!params_.k_buffer_id || !params_.v_buffer_id)
            return {};

        return StageBufferContract::build()
            .addInput(*params_.k_buffer_id)
            .addInput(*params_.v_buffer_id);
    }

} // namespace llaminar2
