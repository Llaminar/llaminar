/**
 * @file KVCacheAppendStage.cpp
 * @brief Publish projection rows through the cache's native conversion authority.
 *
 * Anchored compressed caches own basis initialization and fused K/V encoding on
 * every backend. The stage passes original projections and live row geometry;
 * it must not quantize keys before the request-relative basis is applied.
 */

#include "KVCacheAppendStage.h"
#include "../ComputeStageUtils.h"
#include "../../local_execution/graph/GraphCaptureGuard.h"
#include "../../../utils/DebugEnv.h"
#include "../../../utils/Assertions.h"
#include "../../../tensors/Tensors.h"
#include "../../../tensors/SIMDHelpers.h"
#include "../../../utils/Logger.h"
#include "../../../kernels/cpu/CPUKVCache.h"
#include "../../../kernels/cpu/turboquant/TurboQuantContext.h"
#include "../../../kernels/cpu/turboquant/TurboQuantQuantizeKV.h"
#include "../../../kernels/cpu/turboquant/TurboQuantQuantizeTQ8.h"
#include "../../../kernels/cpu/turboquant/TurboQuantQuantizeTQ4.h"
#include "../../../kernels/cpu/rotation/ActivationRotation.h"
#include "../../../transfer/TransferEngine.h"
#include "../../../utils/OpenMPUtils.h"

#include "../../../utils/KVCacheProfiler.h"
#include "../../../tensors/GpuTensorView.h"
#include "../../../backends/BackendManager.h"

#include <immintrin.h>
#include <cstring>
#include <chrono>
#include <algorithm>
#include <stdexcept>

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

    bool KVCacheAppendStage::bindCanonicalGraphAppendSources(
        void *stream,
        int runtime_seq_len)
    {
        if (!params_.kv_cache || !params_.kv_cache->isGraphCaptureReady() ||
            !stream)
        {
            return false;
        }

        const int request_count = std::max(1, params_.batch_size);
        const int physical_tokens =
            runtime_seq_len > 0
                ? runtime_seq_len
                : (params_.seq_len > 0
                       ? params_.seq_len
                       : params_.num_tokens / request_count);
        if (physical_tokens <= 0 || params_.num_tokens <= 0 ||
            physical_tokens * request_count > params_.num_tokens)
        {
            LOG_ERROR("[KVCacheAppendStage] Invalid graph append capture geometry"
                      << " physical_tokens=" << physical_tokens
                      << " request_count=" << request_count
                      << " total_tokens=" << params_.num_tokens
                      << " layer=" << params_.layer_idx
                      << " seq_idx=" << params_.seq_idx);
            return false;
        }

        bool ready = true;
        for (int request = 0; request < request_count; ++request)
        {
            const int32_t *const source =
                params_.request_sequence_lengths_device
                    ? params_.request_sequence_lengths_device + request
                    : nullptr;
            ready = params_.kv_cache->bindGraphAppendCountSource(
                        params_.layer_idx,
                        params_.seq_idx + request,
                        source,
                        physical_tokens,
                        stream) &&
                    ready;
        }
        return ready;
    }

    bool KVCacheAppendStage::prepareGraphLaunch(
        IDeviceContext *ctx,
        void *stream)
    {
        (void)ctx;
        if (!params_.device_id.is_gpu() || !stream)
        {
            LOG_ERROR("[KVCacheAppendStage] Graph capture preparation requires an explicit GPU stream");
            return false;
        }
        setGPUStream(stream);
        if (!bindCanonicalGraphAppendSources(stream, /*runtime_seq_len=*/0))
        {
            LOG_ERROR("[KVCacheAppendStage] Could not bind canonical append-count sources before graph capture"
                      << " layer=" << params_.layer_idx
                      << " first_seq_idx=" << params_.seq_idx);
            return false;
        }
        return true;
    }

    bool KVCacheAppendStage::shouldUseDecodeEquivalentVerifierAppend(
        int request_rows) const
    {
        if (params_.append_semantics !=
            KVCacheAppendSemantics::DecodeEquivalentVerifier)
            return false;

        /*
         * A one-row verifier transaction is the zero-draft boundary of the
         * same grouped contract, not a different serial execution mode.  All
         * cache backends accept one or more verifier rows, which keeps M=1
         * total and lets the graph exercise the identical publication path at
         * every supported verifier depth.
         */
        if (!params_.kv_cache || params_.layer_idx < 0 || params_.seq_idx < 0 ||
            request_rows < 1)
        {
            LOG_ERROR("[KVCacheAppendStage] Invalid grouped verifier cache-publication contract"
                      << " rows=" << request_rows
                      << " layer=" << params_.layer_idx
                      << " seq=" << params_.seq_idx
                      << " cache=" << params_.kv_cache);
            throw std::runtime_error(
                "invalid grouped verifier KV cache-publication contract");
        }
        return true;
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

        void *stage_stream = nullptr;
        if (params_.device_id.is_gpu())
        {
            stage_stream = requireGPUStream();
            const StageGPUExecution execution = gpuExecution();
            execution.requirePreparedInput(
                const_cast<ITensor *>(params_.K));
            execution.requirePreparedInput(
                const_cast<ITensor *>(params_.V));
        }

        // Determine the graph-shaped token count first, then narrow to the real
        // prefix for non-captured padded execution. Captured GPU execution keeps
        // head/count and real request lengths entirely in canonical device state.
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
            void *stream = stage_stream;

            if (debugEnv().attention.debug_kv_append_source_snapshot &&
                debugEnv().attention.debugKVAppendSourceLayerSelected(params_.layer_idx))
            {
                debug_append_source_k_rows_ = static_cast<size_t>(num_tokens);
                debug_append_source_k_cols_ = k_tensor->shape().size() > 1 ? k_tensor->shape()[1] : k_tensor->cols();
                debug_append_source_v_rows_ = static_cast<size_t>(num_tokens);
                debug_append_source_v_cols_ = v_tensor->shape().size() > 1 ? v_tensor->shape()[1] : v_tensor->cols();
            }
            else
            {
                debug_append_source_k_rows_ = 0;
                debug_append_source_k_cols_ = 0;
                debug_append_source_v_rows_ = 0;
                debug_append_source_v_cols_ = 0;
            }

            size_t requested_cache_snapshot_rows = 0;
            if (debugEnv().attention.debug_kv_cache_snapshot &&
                debugEnv().attention.debugKVCacheSnapshotLayerSelected(params_.layer_idx))
            {
                const int cached_tokens =
                    params_.kv_cache->get_cached_tokens(params_.layer_idx, seq_idx);
                requested_cache_snapshot_rows =
                    static_cast<size_t>(std::max(0, cached_tokens + num_tokens));
            }

            if (shouldUseDecodeEquivalentVerifierAppend(num_tokens))
            {
                success = params_.kv_cache->appendVerifierRowsDecodeEquivalent(
                    params_.layer_idx,
                    seq_idx,
                    k_tensor,
                    v_tensor,
                    num_tokens,
                    stream);
                if (!success)
                {
                    LOG_ERROR("[KVCacheAppendStage] KV cache does not support grouped decode-equivalent verifier append"
                              << " for layer=" << params_.layer_idx
                              << " rows=" << num_tokens
                              << " K=" << k_tensor->dtype_name()
                              << " V=" << v_tensor->dtype_name());
                }
            }
            else if (stream || params_.device_id.is_gpu())
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

            if (success)
            {
                if (requested_cache_snapshot_rows > 0)
                {
                    debug_cache_snapshot_rows_ = requested_cache_snapshot_rows;
                }
                // Snapshot callbacks may request KV cache contents from
                // buildDumpInfoImpl().  The executor often builds dump info
                // before execute() for coherence, so force a post-append
                // rebuild when debug cache/source snapshots are enabled.
                invalidateDumpInfoCache();
            }

            return success;
        };

        /*
         * Cache storage is an ordered K/V precision pair. Reading the legacy
         * single `precision()` value loses V's physical format and previously
         * made asymmetric caches depend on branch accidents. Resolve the pair
         * once and use it for every batched and scalar publication decision.
         */
        const ActivationPrecision cache_k_precision =
            params_.kv_cache->k_precision();
        const ActivationPrecision cache_v_precision =
            params_.kv_cache->v_precision();

        // Determine batch handling mode
        const int batch_size = params_.batch_size;
        const int seq_len = params_.seq_len;

        /*
         * Q16 publication is one operation for serial, grouped and request-
         * batched rows. Each physical block owns its range; no model-wide scale
         * or value clipping may silently discard the precision of this cache.
         * Attention bounds its transient query codes to make integer reductions
         * safe for full-range native keys on every CPU ISA.
         */
        if (cache_k_precision == ActivationPrecision::Q16_1 &&
            cache_v_precision == ActivationPrecision::Q16_1)
        {
            if (!params_.device_id.is_cpu())
                LLAMINAR_UNREACHABLE("Native Q16 append requires the CPU cache implementation");
            const size_t kv_dim = params_.K->cols();
            const auto block_size = optimal_q16_block_size(params_.head_dim);
            const size_t block_elements = q16_block_size_elements(block_size);
            const int requests = batch_size > 1 && seq_len > 0 ? batch_size : 1;
            const int rows = requests > 1 ? seq_len : total_tokens;
            if (!(params_.head_dim > 0 && kv_dim > 0 &&
                           kv_dim % block_elements == 0 &&
                           params_.V->cols() == kv_dim &&
                           rows > 0 && rows * requests == total_tokens &&
                           params_.K->rows() >= size_t(total_tokens) &&
                           params_.V->rows() >= size_t(total_tokens)))
                LLAMINAR_UNREACHABLE("Invalid Q16 cache publication geometry");

            /**
             * Prepare one request operand; owned storage survives until append
             * completes. Native inputs are already in the cache basis, so they
             * must not be rotated or quantized a second time. FP32 scratch is
             * reused between K and V only after encoding the previous operand.
             */
            const auto prepare = [&](const ITensor &input, int first_row,
                                     std::unique_ptr<Q16_1Tensor> &owned)
                -> const Q16_1Tensor *
            {
                if (const auto *native = dynamic_cast<const Q16_1Tensor *>(&input))
                {
                    if (native->q16_block_size() != block_size)
                        LLAMINAR_UNREACHABLE("Q16 source and cache physical block sizes differ");
                    if (first_row == 0)
                        return native;
                    // A request slice preserves native scale/code bytes. This
                    // avoids dequantize/requantize rounding and rotating twice.
                    owned = std::make_unique<Q16_1Tensor>(
                        std::vector<size_t>{size_t(rows), kv_dim}, block_size);
                    const size_t row_bytes = native->blocks_per_row() *
                                             q16_block_size_bytes(block_size);
                    std::memcpy(owned->raw_mutable_data(),
                                static_cast<const uint8_t *>(native->raw_data()) +
                                    size_t(first_row) * row_bytes,
                                size_t(rows) * row_bytes);
                    return owned.get();
                }

                const float *data = input.fp32_data();
                if (!data)
                    LLAMINAR_UNREACHABLE("Q16 source has no CPU FP32 projection data");
                data += size_t(first_row) * kv_dim;
                if (params_.kv_rotation)
                {
                    const size_t elements = size_t(rows) * kv_dim;
                    kv_rotation_scratch_.resize(elements);
                    std::memcpy(kv_rotation_scratch_.data(), data, elements * sizeof(float));
                    params_.kv_rotation->rotate_rows_inplace(
                        kv_rotation_scratch_.data(), rows, static_cast<int>(kv_dim));
                    data = kv_rotation_scratch_.data();
                }
                owned = std::make_unique<Q16_1Tensor>(
                    std::vector<size_t>{size_t(rows), kv_dim}, block_size);
                if (!owned->copyFrom_fp32(data))
                    LLAMINAR_UNREACHABLE("Q16 native block encoding failed");
                return owned.get();
            };

            for (int request = 0; request < requests; ++request)
            {
                const auto start = std::chrono::high_resolution_clock::now();
                const int first_row = request * rows;
                std::unique_ptr<Q16_1Tensor> key_owned, value_owned;
                const auto *key = prepare(*params_.K, first_row, key_owned);
                const auto *value = prepare(*params_.V, first_row, value_owned);

                // Optional decomposed-stage output describes these exact native
                // values, not the original projection before cache encoding.
                if (params_.V_dequant_out)
                {
                    auto *output = dynamic_cast<FP32Tensor *>(params_.V_dequant_out);
                    if (!(output && output->cols() == kv_dim &&
                          output->rows() >= size_t(total_tokens)))
                        LLAMINAR_UNREACHABLE("Q16 V_dequant_out requires the declared FP32 row geometry");
                    for (int row = 0; row < rows; ++row)
                        value->to_fp32_row(
                            row, output->mutable_data() + size_t(first_row + row) * kv_dim);
                }

                const auto elapsed = std::chrono::high_resolution_clock::now() - start;
                KVCacheProfiler::record(
                    KVCacheOpType::CONVERT_TO_Q16_1,
                    std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count(),
                    rows, estimateTensorAppendBytes(key, rows) +
                              estimateTensorAppendBytes(value, rows));
                if (!append_to_cache(params_.seq_idx + request, key, value, rows))
                    return false;
            }
            return true;
        }

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

                const auto *k_base_ptr = static_cast<const uint8_t *>(params_.K->gpu_data_ptr());
                const auto *v_base_ptr = static_cast<const uint8_t *>(params_.V->gpu_data_ptr());
                if (!k_base_ptr || !v_base_ptr)
                {
                    LOG_ERROR("[KVCacheAppendStage] Missing GPU pointers for batched append");
                    return false;
                }

                for (int b = 0; b < batch_size; ++b)
                {
                    const int seq_idx = params_.seq_idx + b;
                    void *k_seq_ptr = const_cast<uint8_t *>(k_base_ptr + static_cast<size_t>(b) * k_seq_bytes);
                    void *v_seq_ptr = const_cast<uint8_t *>(v_base_ptr + static_cast<size_t>(b) * v_seq_bytes);

                    /*
                     * The parent tensors were joined to stage_stream above.
                     * These wrappers describe pointer-offset slices of that
                     * already-ordered storage; they are not independent
                     * coherence owners and therefore carry the exact device
                     * and producer stream into the cache boundary.
                     */
                    PreparedGpuTensorView k_view(
                        k_seq_ptr,
                        static_cast<size_t>(seq_len),
                        k_cols,
                        k_type,
                        target,
                        stage_stream);
                    PreparedGpuTensorView v_view(
                        v_seq_ptr,
                        static_cast<size_t>(seq_len),
                        v_cols,
                        v_type,
                        target,
                        stage_stream);

                    if (!append_to_cache(seq_idx, &k_view, &v_view, seq_len))
                    {
                        LOG_ERROR("[KVCacheAppendStage] append failed for GPU batch index " << b);
                        return false;
                    }
                }

                /*
                 * append_to_cache() receives one pointer-offset view at a time,
                 * but SnapshotCapture observes this stage through the original
                 * request-major parent tensors.  Restore the parent geometry
                 * after every request has been enqueued so an integration
                 * diagnostic can compare request `b` with its serial row.
                 * Leaving the final one-row slice geometry here silently made
                 * every grouped snapshot describe request zero only.
                 */
                if (debugEnv().attention.debug_kv_append_source_snapshot &&
                    debugEnv().attention.debugKVAppendSourceLayerSelected(
                        params_.layer_idx))
                {
                    const size_t source_rows =
                        static_cast<size_t>(batch_size) *
                        static_cast<size_t>(seq_len);
                    debug_append_source_k_rows_ = source_rows;
                    debug_append_source_v_rows_ = source_rows;
                    debug_append_source_k_cols_ =
                        params_.K->shape().size() > 1
                            ? params_.K->shape()[1]
                            : params_.K->cols();
                    debug_append_source_v_cols_ =
                        params_.V->shape().size() > 1
                            ? params_.V->shape()[1]
                            : params_.V->cols();
                    invalidateDumpInfoCache();
                }

                return true;
            }

            LOG_TRACE("[KVCacheAppendStage] Batched append: batch_size=" << batch_size
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
                if (cache_k_precision == ActivationPrecision::FP16 &&
                    cache_v_precision == ActivationPrecision::FP16)
                {
                    const auto conv_start = std::chrono::high_resolution_clock::now();

                    auto k_fp16 = std::make_unique<FP16Tensor>(
                        std::vector<size_t>{static_cast<size_t>(seq_len), kv_dim});
                    auto v_fp16 = std::make_unique<FP16Tensor>(
                        std::vector<size_t>{static_cast<size_t>(seq_len), kv_dim});

                    k_fp16->from_fp32(k_slice->data(), static_cast<size_t>(seq_len) * kv_dim);
                    v_fp16->from_fp32(v_slice->data(), static_cast<size_t>(seq_len) * kv_dim);

                    const auto conv_end = std::chrono::high_resolution_clock::now();
                    const uint64_t conv_ns = static_cast<uint64_t>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(conv_end - conv_start).count());
                    const uint64_t conv_bytes = static_cast<uint64_t>(
                        estimateTensorAppendBytes(k_fp16.get(), seq_len) +
                        estimateTensorAppendBytes(v_fp16.get(), seq_len));
                    KVCacheProfiler::record(KVCacheOpType::CONVERT_TO_FP16, conv_ns, static_cast<uint64_t>(seq_len), conv_bytes);

                    success = append_to_cache(seq_idx, k_fp16.get(), v_fp16.get(), seq_len);
                }
                else if (cache_k_precision == ActivationPrecision::Q8_1 &&
                         cache_v_precision == ActivationPrecision::Q8_1)
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

                    const auto conv_end = std::chrono::high_resolution_clock::now();
                    const uint64_t conv_ns = static_cast<uint64_t>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(conv_end - conv_start).count());
                    const uint64_t conv_bytes = static_cast<uint64_t>(
                        estimateTensorAppendBytes(k_q8.get(), seq_len) +
                        estimateTensorAppendBytes(v_q8.get(), seq_len));
                    KVCacheProfiler::record(KVCacheOpType::CONVERT_TO_Q8_1, conv_ns, static_cast<uint64_t>(seq_len), conv_bytes);

                    success = append_to_cache(seq_idx, k_q8.get(), v_q8.get(), seq_len);
                }
                else if (cache_k_precision == ActivationPrecision::TQ4 &&
                         cache_v_precision == ActivationPrecision::TQ4)
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
                else if (cache_k_precision == ActivationPrecision::TQ8 &&
                         (cache_v_precision == ActivationPrecision::TQ4 ||
                          cache_v_precision == ActivationPrecision::TQ8))
                {
                    if (!params_.turboquant_ctx)
                    {
                        LOG_ERROR("[KVCacheAppendStage] Split TQ cache requires turboquant_ctx in params");
                        return false;
                    }
                    const auto &turboquant_ctx = params_.turboquant_ctx->for_layer(params_.layer_idx);
                    const std::vector<size_t> tq_shape{static_cast<size_t>(seq_len), kv_dim};

                    auto k_tq8 = TQ8Tensor::quantize_from_fp32(
                        k_slice->data(), tq_shape, params_.head_dim,
                        turboquant_ctx);
                    std::shared_ptr<TensorBase> v_tq;
                    if (cache_v_precision == ActivationPrecision::TQ8)
                        v_tq = TQ8Tensor::quantize_from_fp32(
                            v_slice->data(), tq_shape, params_.head_dim,
                            turboquant_ctx);
                    else
                        v_tq = TQ4Tensor::quantize_from_fp32(
                            v_slice->data(), tq_shape, params_.head_dim,
                            turboquant_ctx);
                    if (!k_tq8 || !v_tq)
                    {
                        LOG_ERROR("[KVCacheAppendStage] Failed to quantize batched K/V slices to the declared TQ pair");
                        return false;
                    }

                    success = append_to_cache(
                        seq_idx, k_tq8.get(), v_tq.get(), seq_len);
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
        LOG_TRACE("[KVCacheAppendStage] Single-sequence append: " << total_tokens
                                                                  << " tokens to layer " << params_.layer_idx << " seq " << params_.seq_idx);

        // Check if tensors match cache precision - if not, need to convert
        // This handles Hybrid mode where K_rope is FP32 and V is Q8_1 but cache is FP32
        const bool cache_is_fp32 =
            cache_k_precision == ActivationPrecision::FP32 &&
            cache_v_precision == ActivationPrecision::FP32;
        const bool cache_is_fp16 =
            cache_k_precision == ActivationPrecision::FP16 &&
            cache_v_precision == ActivationPrecision::FP16;
        const bool cache_is_q8_1 =
            cache_k_precision == ActivationPrecision::Q8_1 &&
            cache_v_precision == ActivationPrecision::Q8_1;
        bool k_is_fp32 = (params_.K->native_type() == TensorType::FP32);
        bool v_is_fp32 = (params_.V->native_type() == TensorType::FP32);
        const bool has_gpu_inputs = (params_.K->gpu_data_ptr() != nullptr && params_.V->gpu_data_ptr() != nullptr);

        /*
         * CPU, CUDA and ROCm compressed caches own one fused FP32-to-physical
         * append launch: AQ8 for K and Q8_1/TQ4/TQ8 for V. Keeping that
         * conversion behind the cache boundary preserves capture, avoids host
         * scratch, and lets Q8_1 operate without an unrelated TQ context.
         */
        const bool native_asymmetric_compressed_cache =
            cache_k_precision == ActivationPrecision::AQ8 &&
            (cache_v_precision == ActivationPrecision::Q8_1 ||
             cache_v_precision == ActivationPrecision::TQ4 ||
             cache_v_precision == ActivationPrecision::TQ8);
        if (native_asymmetric_compressed_cache)
        {
            const bool value_uses_turboquant =
                cache_v_precision == ActivationPrecision::TQ4 ||
                cache_v_precision == ActivationPrecision::TQ8;
            if (value_uses_turboquant && !params_.turboquant_ctx)
            {
                LOG_ERROR("[KVCacheAppendStage] AQ8/TQ cache requires turboquant_ctx for value rotation");
                return false;
            }
            if (params_.device_id.is_gpu() && !has_gpu_inputs)
            {
                LOG_ERROR("[KVCacheAppendStage] AQ8/compressed GPU cache requires device-resident K/V inputs");
                return false;
            }

            const auto conv_start = std::chrono::high_resolution_clock::now();
            const bool success = append_to_cache(
                params_.seq_idx, params_.K, params_.V, total_tokens);
            const auto conv_end = std::chrono::high_resolution_clock::now();
            const uint64_t conv_ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    conv_end - conv_start)
                    .count());
            KVCacheProfiler::record(
                value_uses_turboquant
                    ? KVCacheOpType::CONVERT_TO_TQ
                    : KVCacheOpType::CONVERT_TO_Q8_1,
                conv_ns, static_cast<uint64_t>(total_tokens), 0);
            if (!success)
            {
                LOG_ERROR("[KVCacheAppendStage] fused AQ8/compressed append failed for K="
                          << activationPrecisionToString(cache_k_precision)
                          << " V="
                          << activationPrecisionToString(cache_v_precision));
                return false;
            }
            return true;
        }

        /*
         * CPU caches convert mismatched producer tensors here because their
         * storage is host-owned. GPU caches own a fused convert-and-append
         * launch for every native floating format; routing a GPU append through
         * this branch would allocate host temporaries and invalidate capture.
         */
        if (!has_gpu_inputs && cache_is_fp32 && (!k_is_fp32 || !v_is_fp32))
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
        // TQ4 cache path with TurboQuant rotation-based quantization
        // =================================================================
        const bool cache_is_tq4 =
            cache_k_precision == ActivationPrecision::TQ4 &&
            cache_v_precision == ActivationPrecision::TQ4;

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
        // CPU TurboQuant cache path: TQ8 K with selectable TQ4 or TQ8 V
        // =================================================================
        const bool cache_is_tq8 =
            cache_k_precision == ActivationPrecision::TQ8 &&
            (cache_v_precision == ActivationPrecision::TQ4 ||
             cache_v_precision == ActivationPrecision::TQ8);

        if (cache_is_tq8)
        {
            if (!params_.turboquant_ctx)
            {
                LOG_ERROR("[KVCacheAppendStage] Split TQ cache requires turboquant_ctx in params");
                return false;
            }

            // CPU path: quantize into cache-native blocks before publication.
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
            const bool value_is_tq8 =
                cache_v_precision == ActivationPrecision::TQ8;
            if (value_is_tq8 &&
                (!tq8_v_scratch_ || tq8_v_scratch_->shape() != tq_shape))
            {
                tq8_v_scratch_ = std::make_shared<TQ8Tensor>(tq_shape, head_dim);
                tq8_v_scratch_->set_turboquant_context(&turboquant_ctx);
            }
            if (!value_is_tq8 &&
                (!tq4_v_scratch_ || tq4_v_scratch_->shape() != tq_shape))
            {
                tq4_v_scratch_ = std::make_shared<TQ4Tensor>(tq_shape, head_dim);
                tq4_v_scratch_->set_turboquant_context(&turboquant_ctx);
            }

            const size_t k_block_bytes = tq8_k_scratch_->block_bytes();
            const size_t v_block_bytes = value_is_tq8
                                             ? tq8_v_scratch_->block_bytes()
                                             : tq4_v_scratch_->block_bytes();
            const size_t blocks_per_row = tq8_k_scratch_->blocks_per_row();
            auto *k_blocks = static_cast<uint8_t *>(
                tq8_k_scratch_->raw_mutable_data());
            auto *v_blocks = static_cast<uint8_t *>(
                value_is_tq8
                    ? tq8_v_scratch_->raw_mutable_data()
                    : tq4_v_scratch_->raw_mutable_data());

            /*
             * Decode, grouped verification, and prefill share one implementation.
             * Each `(row, head)` invokes the same ISA-dispatched vector primitive
             * as serial decode, while K and V share one OpenMP workshare and a
             * cache-hot rotation context.
             */
            turboquant_quantize_kv_rows(
                k_fp32,
                v_fp32,
                k_blocks,
                v_blocks,
                total_tokens,
                head_dim,
                static_cast<int>(blocks_per_row),
                blocks_per_row * k_block_bytes,
                blocks_per_row * v_block_bytes,
                k_block_bytes,
                v_block_bytes,
                turboquant_ctx,
                value_is_tq8);

            const auto conv_end = std::chrono::high_resolution_clock::now();
            const uint64_t conv_ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(conv_end - conv_start).count());
            KVCacheProfiler::record(KVCacheOpType::CONVERT_TO_TQ, conv_ns, static_cast<uint64_t>(total_tokens), 0);

            const TensorBase *value_scratch = value_is_tq8
                                                  ? static_cast<const TensorBase *>(tq8_v_scratch_.get())
                                                  : static_cast<const TensorBase *>(tq4_v_scratch_.get());
            bool success = append_to_cache(
                params_.seq_idx, tq8_k_scratch_.get(), value_scratch, total_tokens);
            if (!success)
            {
                LOG_ERROR("[KVCacheAppendStage] append failed (TurboQuant cache: TQ8 K + "
                          << (value_is_tq8 ? "TQ8" : "TQ4") << " V)");
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

        if (debugEnv().attention.debug_kv_append_source_snapshot &&
            debugEnv().attention.debugKVAppendSourceLayerSelected(params_.layer_idx))
        {
            const size_t rows = debug_append_source_k_rows_ > 0
                                    ? debug_append_source_k_rows_
                                    : static_cast<size_t>(params_.num_tokens > 0 ? params_.num_tokens : 0);
            const size_t k_cols = debug_append_source_k_cols_ > 0
                                      ? debug_append_source_k_cols_
                                      : (params_.K->shape().size() > 1 ? params_.K->shape()[1] : params_.K->cols());
            const size_t v_cols = debug_append_source_v_cols_ > 0
                                      ? debug_append_source_v_cols_
                                      : (params_.V->shape().size() > 1 ? params_.V->shape()[1] : params_.V->cols());
            if (rows > 0 && k_cols > 0)
            {
                info.addOutput("source_k", params_.K, rows, k_cols);
            }
            if (rows > 0 && v_cols > 0)
            {
                info.addOutput("source_v", params_.V, rows, v_cols);
            }
        }

        if (debugEnv().attention.debug_kv_cache_snapshot &&
            debugEnv().attention.debugKVCacheSnapshotLayerSelected(params_.layer_idx) &&
            params_.kv_cache)
        {
            ITensor *cache_k = nullptr;
            ITensor *cache_v = nullptr;
            int kv_len = 0;
            const int cached_tokens =
                params_.kv_cache->get_cached_tokens(params_.layer_idx, params_.seq_idx);
            const int append_tokens =
                replay_advance_tokens_ > 0
                    ? replay_advance_tokens_
                    : (params_.num_tokens > 0 ? params_.num_tokens : static_cast<int>(params_.K->rows()));
            const int expected_tokens =
                debug_cache_snapshot_rows_ > 0
                    ? static_cast<int>(debug_cache_snapshot_rows_)
                    : std::max(0, cached_tokens + append_tokens);
            if (params_.kv_cache->get_kv_snapshot_view(
                    params_.layer_idx, params_.seq_idx, expected_tokens, &cache_k, &cache_v, &kv_len) &&
                kv_len == expected_tokens && kv_len > 0 && cache_k && cache_v)
            {
                const size_t rows = static_cast<size_t>(kv_len);
                const size_t k_cols = cache_k->cols();
                const size_t v_cols = cache_v->cols();
                if (rows > 0 && k_cols > 0)
                {
                    info.addOutput("cache_k", cache_k, rows, k_cols);
                }
                if (rows > 0 && v_cols > 0)
                {
                    info.addOutput("cache_v", cache_v, rows, v_cols);
                }
            }
            else
            {
                throw std::runtime_error(
                    "KV cache debug snapshot requested but cache could not expose a direct snapshot view for layer=" +
                    std::to_string(params_.layer_idx) +
                    " seq=" + std::to_string(params_.seq_idx) +
                    " expected_tokens=" + std::to_string(expected_tokens));
            }
        }
        else if (debugEnv().attention.debug_kv_append_source_snapshot &&
                 debugEnv().attention.debugKVAppendSourceLayerSelected(params_.layer_idx) &&
                 !debugEnv().attention.debug_kv_cache_snapshot)
        {
            LOG_TRACE("[KVCacheAppendStage] Capturing KV append source without "
                      "the optional post-append cache snapshot for layer="
                      << params_.layer_idx);
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
