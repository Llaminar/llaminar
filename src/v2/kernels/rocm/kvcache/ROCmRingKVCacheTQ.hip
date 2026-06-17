/**
 * @file ROCmRingKVCacheTQ.hip
 * @brief HIP implementation of ROCm TQ KV cache
 * @author David Sanftenberg
 *
 * V2: Per-layer FP16 scratch, fused incremental kernel, graph capture support.
 */

#include "ROCmRingKVCacheTQ.h"
#include "ROCmTurboQuantKernels.h"
#include "../../../execution/local_execution/graph/GraphCaptureGuard.h"
#include "../../../tensors/GpuTensorView.h"
#include "../../../kernels/cpu/turboquant/TurboQuantContext.h"
#include "../../../backends/GPUDeviceContextPool.h"
#include "../../../utils/Logger.h"
#include <hip/hip_runtime.h>
#include <cstring>

namespace llaminar2
{

    ROCmRingKVCacheTQ::ROCmRingKVCacheTQ(
        int n_layers, int batch_size, int max_seq_len,
        int n_kv_heads, int head_dim,
        const TurboQuantContext *tq_ctx, int device_id)
        : ROCmRingKVCacheBase(n_layers, batch_size, max_seq_len,
                               n_kv_heads, head_dim, n_kv_heads * head_dim, device_id)
    {
        hipSetDevice(device_id_);

        if (head_dim_ == 64)
        {
            tq8_block_size_ = sizeof(TQ8Block<64>);
            tq4_block_size_ = sizeof(TQ4Block<64>);
        }
        else if (head_dim_ == 128)
        {
            tq8_block_size_ = sizeof(TQ8Block<128>);
            tq4_block_size_ = sizeof(TQ4Block<128>);
        }
        else
        {
            LOG_ERROR("ROCmRingKVCacheTQ: unsupported head_dim=" << head_dim_);
            return;
        }

        const uint64_t rotation_seed = tq_ctx ? tq_ctx->rotation().seed : 42;

        hip_tq_upload_codebooks(0);
        rotations_ = hip_tq_create_rotations(n_layers_, n_kv_heads_, head_dim_,
                                              rotation_seed, device_id_, 0);

        const size_t k_entry_bytes = static_cast<size_t>(max_seq_len_) * n_kv_heads_ * tq8_block_size_;
        const size_t v_entry_bytes = static_cast<size_t>(max_seq_len_) * n_kv_heads_ * tq4_block_size_;

        entries_.resize(n_layers_);

        for (int l = 0; l < n_layers_; ++l)
        {
            entries_[l].resize(batch_size_);

            for (int s = 0; s < batch_size_; ++s)
            {
                hipMalloc(&entries_[l][s].d_K, k_entry_bytes);
                hipMalloc(&entries_[l][s].d_V, v_entry_bytes);
                entries_[l][s].head = 0;
                entries_[l][s].count = 0;
            }
        }

        // Allocate per-layer FP16 scratch buffers
        {
            const size_t scratch_bytes = static_cast<size_t>(max_seq_len_) * kv_dim_ * sizeof(_Float16);
            layer_scratch_.resize(n_layers_);
            for (int l = 0; l < n_layers_; ++l)
            {
                hipMalloc(&layer_scratch_[l].d_K, scratch_bytes);
                hipMalloc(&layer_scratch_[l].d_V, scratch_bytes);
                layer_scratch_[l].invalidate();
            }
            LOG_DEBUG("ROCmRingKVCacheTQ: per-layer FP16 scratch: "
                     << n_layers_ << " x " << (2 * scratch_bytes / 1024) << "KB");
        }

        // Pre-allocate temp buffers for GPU-side quantization (avoids per-call hipMalloc)
        {
            const size_t k_temp_bytes = static_cast<size_t>(max_seq_len) * n_kv_heads * tq8_block_size_;
            const size_t v_temp_bytes = static_cast<size_t>(max_seq_len) * n_kv_heads * tq4_block_size_;
            hipMalloc(&d_quantize_k_temp_, k_temp_bytes);
            hipMalloc(&d_quantize_v_temp_, v_temp_bytes);
            LOG_DEBUG("ROCmRingKVCacheTQ: pre-allocated quantize temp buffers: K="
                     << (k_temp_bytes / 1024) << "KB, V=" << (v_temp_bytes / 1024) << "KB");
        }

        // Allocate graph capture dynamic params (per-layer)
        {
            hipMalloc(&d_dequant_params_, n_layers_ * sizeof(HIPTQDequantDynamicParams));
            hipHostMalloc(&h_dequant_params_, n_layers_ * sizeof(HIPTQDequantDynamicParams),
                          hipHostMallocDefault);
            memset(h_dequant_params_, 0, n_layers_ * sizeof(HIPTQDequantDynamicParams));
            dequant_params_device_valid_.assign(static_cast<size_t>(n_layers_), 0);
            LOG_DEBUG("ROCmRingKVCacheTQ: graph capture params allocated for " << n_layers_ << " layers");
        }

        LOG_DEBUG("ROCmRingKVCacheTQ: created on device " << device_id_
                  << " (" << n_layers_ << " layers, " << batch_size_ << " batch, "
                  << max_seq_len_ << " maxseq, head_dim=" << head_dim_
                  << ", TQ8+TQ4, FP16 scratch)");
    }

    ROCmRingKVCacheTQ::~ROCmRingKVCacheTQ()
    {
        hipSetDevice(device_id_);

        for (auto &layer : entries_)
            for (auto &entry : layer)
            {
                if (entry.d_K) hipFree(entry.d_K);
                if (entry.d_V) hipFree(entry.d_V);
            }

        // Free per-layer scratch
        for (auto &scratch : layer_scratch_)
        {
            scratch.k_view.reset();
            scratch.v_view.reset();
            if (scratch.d_K) hipFree(scratch.d_K);
            if (scratch.d_V) hipFree(scratch.d_V);
        }

        // Free pre-allocated quantize temp buffers
        if (d_quantize_k_temp_) hipFree(d_quantize_k_temp_);
        if (d_quantize_v_temp_) hipFree(d_quantize_v_temp_);

        // Free graph capture params
        if (d_dequant_params_) hipFree(d_dequant_params_);
        if (h_dequant_params_) hipHostFree(h_dequant_params_);

        hip_tq_free_rotations(rotations_);
    }

    // =========================================================================
    // Query — get_cached_tokens(), get_head_position(), is_wrapped() are now
    // provided by ROCmRingKVCacheBase via entry accessor overrides.
    // =========================================================================

    // =========================================================================
    // Append
    // =========================================================================

    bool ROCmRingKVCacheTQ::append(int layer, int seq_idx,
                                    const ITensor *k, const ITensor *v,
                                    int num_tokens)
    {
        LOG_ERROR("[ROCmRingKVCacheTQ::append] Explicit stream required; use appendWithStream()");
        return false;
    }

    bool ROCmRingKVCacheTQ::appendWithStream(
        int layer, int seq_idx, const ITensor *k, const ITensor *v,
        int num_tokens, void *stream)
    {
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
            return false;
        if (!k || !v)
            return false;
        if (!stream)
        {
            LOG_ERROR("[ROCmRingKVCacheTQ::appendWithStream] Null stream is not allowed");
            return false;
        }

        hipSetDevice(device_id_);
        const auto hip_stream = static_cast<hipStream_t>(stream);
        cached_stream_ = hip_stream;
        auto &entry = entries_[layer][seq_idx];
        const bool capture_active = isGraphCaptureActive();

        const bool k_is_tq = (k->native_type() == TensorType::TQ8 || k->native_type() == TensorType::TQ4);
        const bool v_is_tq = (v->native_type() == TensorType::TQ4);

        LOG_DEBUG("[ROCmRingKVCacheTQ::append] layer=" << layer
                 << " seq=" << seq_idx << " tokens=" << num_tokens
                 << " K.dtype=" << k->dtype_name()
                 << " V.dtype=" << v->dtype_name()
                 << " pre_quantized=" << (k_is_tq && v_is_tq));

        // ================================================================
        // Fast path: K and V are already TQ-quantized blocks
        // ================================================================
        if (k_is_tq && v_is_tq)
        {
            const uint8_t *h_K_blocks = static_cast<const uint8_t *>(k->raw_data());
            const uint8_t *h_V_blocks = static_cast<const uint8_t *>(v->raw_data());
            if (!h_K_blocks || !h_V_blocks)
                return false;

            const size_t k_row_bytes = static_cast<size_t>(n_kv_heads_) * tq8_block_size_;
            const size_t v_row_bytes = static_cast<size_t>(n_kv_heads_) * tq4_block_size_;

            for (int t = 0; t < num_tokens; ++t)
            {
                const int pos = (entry.head + t) % max_seq_len_;
                const size_t k_dst_offset = static_cast<size_t>(pos) * k_row_bytes;
                const size_t v_dst_offset = static_cast<size_t>(pos) * v_row_bytes;
                const size_t k_src_offset = static_cast<size_t>(t) * k_row_bytes;
                const size_t v_src_offset = static_cast<size_t>(t) * v_row_bytes;

                hipMemcpyAsync(
                    static_cast<uint8_t *>(entry.d_K) + k_dst_offset,
                    h_K_blocks + k_src_offset,
                    k_row_bytes, hipMemcpyHostToDevice, hip_stream);
                hipMemcpyAsync(
                    static_cast<uint8_t *>(entry.d_V) + v_dst_offset,
                    h_V_blocks + v_src_offset,
                    v_row_bytes, hipMemcpyHostToDevice, hip_stream);
            }

            if (!capture_active)
            {
                entry.head = (entry.head + num_tokens) % max_seq_len_;
                entry.count = std::min(entry.count + num_tokens, max_seq_len_);
            }

            // DON'T invalidate per-layer scratch here — dequant_to_scratch's
            // can_incremental() will detect count == cached_count + 1 and do
            // a single-position fused dequant instead of a full re-linearize.

            return true;
        }

        // ================================================================
        // GPU quantize path: K and V are FP32 — quantize on GPU
        // Uses pre-allocated temp buffers (no per-call hipMalloc)
        // ================================================================
        const float *d_k_input = nullptr;
        const float *d_v_input = nullptr;
        void *d_k_uploaded = nullptr;
        void *d_v_uploaded = nullptr;

        if (k->is_on_gpu())
        {
            d_k_input = static_cast<const float *>(k->gpu_data_ptr());
        }
        else
        {
            const size_t bytes = static_cast<size_t>(num_tokens) * kv_dim_ * sizeof(float);
            hipMalloc(&d_k_uploaded, bytes);
            hipMemcpyAsync(d_k_uploaded, k->data(), bytes, hipMemcpyHostToDevice, hip_stream);
            d_k_input = static_cast<const float *>(d_k_uploaded);
        }

        if (v->is_on_gpu())
        {
            d_v_input = static_cast<const float *>(v->gpu_data_ptr());
        }
        else
        {
            const size_t bytes = static_cast<size_t>(num_tokens) * kv_dim_ * sizeof(float);
            hipMalloc(&d_v_uploaded, bytes);
            hipMemcpyAsync(d_v_uploaded, v->data(), bytes, hipMemcpyHostToDevice, hip_stream);
            d_v_input = static_cast<const float *>(d_v_uploaded);
        }

        const float *k_rotations = rotations_.rotation(layer, 0);

        // Quantize into pre-allocated temp buffers
        bool ok = hip_tq8_quantize(d_k_input, k_rotations, d_quantize_k_temp_,
                                    num_tokens, n_kv_heads_, head_dim_, hip_stream);
        if (ok)
        {
            ok = hip_tq4_quantize(d_v_input, k_rotations, d_quantize_v_temp_,
                                   num_tokens, n_kv_heads_, head_dim_, hip_stream);
        }

        if (ok)
        {
            // Copy quantized blocks from temp to ring buffer positions
            const size_t k_pos_bytes = static_cast<size_t>(n_kv_heads_) * tq8_block_size_;
            const size_t v_pos_bytes = static_cast<size_t>(n_kv_heads_) * tq4_block_size_;

            for (int t = 0; t < num_tokens; ++t)
            {
                int dst_pos = (entry.head + t) % max_seq_len_;
                hipMemcpyAsync(
                    static_cast<uint8_t *>(entry.d_K) + static_cast<size_t>(dst_pos) * k_pos_bytes,
                    static_cast<uint8_t *>(d_quantize_k_temp_) + static_cast<size_t>(t) * k_pos_bytes,
                    k_pos_bytes, hipMemcpyDeviceToDevice, hip_stream);
                hipMemcpyAsync(
                    static_cast<uint8_t *>(entry.d_V) + static_cast<size_t>(dst_pos) * v_pos_bytes,
                    static_cast<uint8_t *>(d_quantize_v_temp_) + static_cast<size_t>(t) * v_pos_bytes,
                    v_pos_bytes, hipMemcpyDeviceToDevice, hip_stream);
            }
        }

        if (ok && !capture_active)
        {
            entry.head = (entry.head + num_tokens) % max_seq_len_;
            entry.count = std::min(entry.count + num_tokens, max_seq_len_);

            // DON'T invalidate per-layer scratch — let dequant_to_scratch's
            // can_incremental() handle it for single-token decode.
            // For multi-token append (prefill), is_current_for will fail which
            // triggers a full linearize dequant — correct behavior.
        }

        if (d_k_uploaded) hipFree(d_k_uploaded);
        if (d_v_uploaded) hipFree(d_v_uploaded);

        return ok;
    }

    // =========================================================================
    // Per-Layer FP16 Scratch with Fused Incremental Dequant
    // =========================================================================

    hipStream_t ROCmRingKVCacheTQ::clearStream() const
    {
        if (cached_stream_)
            return cached_stream_;
        return static_cast<hipStream_t>(
            GPUDeviceContextPool::instance().getAMDContext(device_id_).defaultStream());
    }

    void ROCmRingKVCacheTQ::clearEntryStorage(int layer, int seq_idx, hipStream_t stream)
    {
        auto &entry = entries_[layer][seq_idx];
        const size_t k_bytes = static_cast<size_t>(max_seq_len_) * n_kv_heads_ * tq8_block_size_;
        const size_t v_bytes = static_cast<size_t>(max_seq_len_) * n_kv_heads_ * tq4_block_size_;

        if (entry.d_K)
        {
            hipError_t err = hipMemsetAsync(entry.d_K, 0, k_bytes, stream);
            if (err != hipSuccess)
                LOG_WARN("[ROCmRingKVCacheTQ::clear] K memset failed: " << hipGetErrorString(err));
        }
        if (entry.d_V)
        {
            hipError_t err = hipMemsetAsync(entry.d_V, 0, v_bytes, stream);
            if (err != hipSuccess)
                LOG_WARN("[ROCmRingKVCacheTQ::clear] V memset failed: " << hipGetErrorString(err));
        }
    }

    void ROCmRingKVCacheTQ::clearScratchStorage(int layer, hipStream_t stream)
    {
        auto &scratch = layer_scratch_[layer];
        const size_t scratch_bytes = static_cast<size_t>(max_seq_len_) * kv_dim_ * sizeof(_Float16);

        if (scratch.d_K)
        {
            hipError_t err = hipMemsetAsync(scratch.d_K, 0, scratch_bytes, stream);
            if (err != hipSuccess)
                LOG_WARN("[ROCmRingKVCacheTQ::clear] scratch K memset failed: " << hipGetErrorString(err));
        }
        if (scratch.d_V)
        {
            hipError_t err = hipMemsetAsync(scratch.d_V, 0, scratch_bytes, stream);
            if (err != hipSuccess)
                LOG_WARN("[ROCmRingKVCacheTQ::clear] scratch V memset failed: " << hipGetErrorString(err));
        }

        scratch.k_view.reset();
        scratch.v_view.reset();
        scratch.invalidate();
    }

    void ROCmRingKVCacheTQ::clearDynamicParams(int layer, int seq_idx, hipStream_t stream)
    {
        const int entry_idx = layer * batch_size_ + seq_idx;
        if (h_head_params_)
            h_head_params_[entry_idx] = 0;
        if (h_count_params_)
            h_count_params_[entry_idx] = 0;
        if (d_head_params_)
        {
            hipError_t err = hipMemsetAsync(&d_head_params_[entry_idx], 0, sizeof(int), stream);
            if (err != hipSuccess)
                LOG_WARN("[ROCmRingKVCacheTQ::clear] head param memset failed: " << hipGetErrorString(err));
        }
        if (d_count_params_)
        {
            hipError_t err = hipMemsetAsync(&d_count_params_[entry_idx], 0, sizeof(int), stream);
            if (err != hipSuccess)
                LOG_WARN("[ROCmRingKVCacheTQ::clear] count param memset failed: " << hipGetErrorString(err));
        }

        if (h_dequant_params_)
            std::memset(&h_dequant_params_[layer], 0, sizeof(HIPTQDequantDynamicParams));
        if (layer >= 0 && layer < static_cast<int>(dequant_params_device_valid_.size()))
            dequant_params_device_valid_[static_cast<size_t>(layer)] = 0;
        if (d_dequant_params_)
        {
            hipError_t err = hipMemsetAsync(&d_dequant_params_[layer], 0,
                                            sizeof(HIPTQDequantDynamicParams), stream);
            if (err != hipSuccess)
                LOG_WARN("[ROCmRingKVCacheTQ::clear] dequant param memset failed: " << hipGetErrorString(err));
        }
    }

    void ROCmRingKVCacheTQ::clear_sequence(int layer, int seq_idx)
    {
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
            return;

        hipSetDevice(device_id_);
        hipStream_t stream = clearStream();
        clearEntryStorage(layer, seq_idx, stream);
        clearScratchStorage(layer, stream);
        clearDynamicParams(layer, seq_idx, stream);

        hipError_t sync_err = hipStreamSynchronize(stream);
        if (sync_err != hipSuccess)
            LOG_WARN("[ROCmRingKVCacheTQ::clear_sequence] sync failed: " << hipGetErrorString(sync_err));

        cached_stream_ = nullptr;
        ROCmRingKVCacheBase::clear_sequence(layer, seq_idx);
    }

    void ROCmRingKVCacheTQ::clear_layer(int layer)
    {
        if (layer < 0 || layer >= n_layers_)
            return;

        hipSetDevice(device_id_);
        hipStream_t stream = clearStream();
        for (int seq = 0; seq < batch_size_; ++seq)
        {
            clearEntryStorage(layer, seq, stream);
            clearDynamicParams(layer, seq, stream);
        }
        clearScratchStorage(layer, stream);

        hipError_t sync_err = hipStreamSynchronize(stream);
        if (sync_err != hipSuccess)
            LOG_WARN("[ROCmRingKVCacheTQ::clear_layer] sync failed: " << hipGetErrorString(sync_err));

        cached_stream_ = nullptr;
        for (int seq = 0; seq < batch_size_; ++seq)
            ROCmRingKVCacheBase::clear_sequence(layer, seq);
    }

    void ROCmRingKVCacheTQ::clear_sequence(int seq_idx)
    {
        if (seq_idx < 0 || seq_idx >= batch_size_)
            return;

        hipSetDevice(device_id_);
        hipStream_t stream = clearStream();
        for (int layer = 0; layer < n_layers_; ++layer)
        {
            clearEntryStorage(layer, seq_idx, stream);
            clearScratchStorage(layer, stream);
            clearDynamicParams(layer, seq_idx, stream);
        }

        hipError_t sync_err = hipStreamSynchronize(stream);
        if (sync_err != hipSuccess)
            LOG_WARN("[ROCmRingKVCacheTQ::clear_sequence] sync failed: " << hipGetErrorString(sync_err));

        cached_stream_ = nullptr;
        for (int layer = 0; layer < n_layers_; ++layer)
            ROCmRingKVCacheBase::clear_sequence(layer, seq_idx);
    }

    void ROCmRingKVCacheTQ::clear()
    {
        hipSetDevice(device_id_);
        hipStream_t stream = clearStream();

        for (int layer = 0; layer < n_layers_; ++layer)
        {
            for (int seq = 0; seq < batch_size_; ++seq)
            {
                clearEntryStorage(layer, seq, stream);
                clearDynamicParams(layer, seq, stream);
            }
            clearScratchStorage(layer, stream);
        }

        hipError_t sync_err = hipStreamSynchronize(stream);
        if (sync_err != hipSuccess)
            LOG_WARN("[ROCmRingKVCacheTQ::clear] sync failed: " << hipGetErrorString(sync_err));

        cached_stream_ = nullptr;
        for (int layer = 0; layer < n_layers_; ++layer)
        {
            for (int seq = 0; seq < batch_size_; ++seq)
                ROCmRingKVCacheBase::clear_sequence(layer, seq);
        }
    }

    bool ROCmRingKVCacheTQ::dequant_to_scratch(int layer, int seq_idx, float rope_theta, int position_start, hipStream_t stream) const
    {
        auto &scratch = layer_scratch_[layer];
        const auto &entry = entries_[layer][seq_idx];

        if (entry.count == 0)
        {
            scratch.cached_count = 0;
            scratch.cached_layer = layer;
            scratch.cached_seq = seq_idx;
            return true;
        }

        // Check if scratch already holds the right data
        if (scratch.is_current_for(layer, seq_idx, entry.count, entry.head,
                                    rope_theta, position_start))
        {
            return true; // Already up-to-date
        }

        const int tail = entry.tail(max_seq_len_);
        const int D = head_dim_;
        const size_t layer_rot_offset = static_cast<size_t>(layer) * n_kv_heads_ * D * D;
        const float *d_K_rot_t = rotations_.d_rotations_t + layer_rot_offset;
        const float *d_V_rot_t = d_K_rot_t;
        const float *d_K_rot = rotations_.d_rotations + layer_rot_offset;
        const float *d_V_rot = d_K_rot;

        // Incremental dequant: only process the newly appended position.
        // Valid during decode when count increased by exactly 1 and no eviction.
        // Per-layer scratch makes this reliable — no cross-layer invalidation.
        //
        // Uses fused K+V kernel: 1 launch per layer instead of 2.
        if (scratch.can_incremental(layer, seq_idx, entry.count, tail,
                                     rope_theta, position_start))
        {
            bool ok;
            if (d_dequant_params_ && h_dequant_params_ && stream)
            {
                // Graph-capturable path: the kernel reads dynamic params from
                // device memory. Captured execution must not record H2D; params
                // are pre-uploaded by setDynamicDequantParams().
                if (isGraphCaptureActive())
                {
                    if (layer < 0 ||
                        layer >= static_cast<int>(dequant_params_device_valid_.size()) ||
                        !dequant_params_device_valid_[static_cast<size_t>(layer)])
                    {
                        LOG_ERROR("[ROCmRingKVCacheTQ::dequant_to_scratch] TQ dequant params were not ready before HIP graph capture");
                        return false;
                    }
                }
                else
                {
                    h_dequant_params_[layer].ring_pos = (tail + entry.count - 1) % max_seq_len_;
                    h_dequant_params_[layer].out_offset_elems = static_cast<int>((entry.count - 1) * kv_dim_);
                    h_dequant_params_[layer].rope_position =
                        (rope_theta > 0.0f) ? position_start + entry.count - 1 : 0;

                    hipError_t copy_err =
                        hipMemcpyAsync(&d_dequant_params_[layer], &h_dequant_params_[layer],
                                        sizeof(HIPTQDequantDynamicParams), hipMemcpyHostToDevice, stream);
                    if (copy_err != hipSuccess)
                    {
                        LOG_ERROR("[ROCmRingKVCacheTQ::dequant_to_scratch] hipMemcpyAsync failed: "
                                  << hipGetErrorString(copy_err));
                        return false;
                    }
                    dequant_params_device_valid_[static_cast<size_t>(layer)] = 1;
                }

                ok = hip_tq_incremental_single_fp16_dynamic(
                    scratch.d_K, scratch.d_V,
                    entry.d_K, entry.d_V,
                    d_K_rot, d_V_rot,
                    &d_dequant_params_[layer],
                    n_kv_heads_, head_dim_,
                    rope_theta, stream);
            }
            else
            {
                const int new_pos = entry.count - 1;
                const int new_ring_pos = (tail + new_pos) % max_seq_len_;
                const size_t out_offset = static_cast<size_t>(new_pos) * kv_dim_;

                ok = hip_tq_incremental_single_fp16(
                    scratch.d_K + out_offset,
                    scratch.d_V + out_offset,
                    entry.d_K, entry.d_V,
                    d_K_rot, d_V_rot,
                    new_ring_pos, 0,
                    n_kv_heads_, head_dim_,
                    rope_theta, (rope_theta > 0.0f) ? position_start + new_pos : 0,
                    stream);
            }

            if (ok)
            {
                scratch.cached_count = entry.count;
                scratch.cached_head = entry.head;
            }
            return ok;
        }

        // Full linearize + dequant + optional RoPE into per-layer FP16 scratch
        bool ok = hip_tq_ring_linearize_dequant_fp16(
            scratch.d_K, scratch.d_V,
            entry.d_K, entry.d_V,
            d_K_rot_t, d_V_rot_t,
            d_K_rot, d_V_rot,
            tail, entry.count, max_seq_len_,
            n_kv_heads_, head_dim_,
            rope_theta, position_start,
            stream);

        if (ok)
        {
            scratch.cached_layer = layer;
            scratch.cached_seq = seq_idx;
            scratch.cached_count = entry.count;
            scratch.cached_head = entry.head;
            scratch.cached_tail = tail;
            scratch.cached_rope_theta = rope_theta;
            scratch.cached_position_start = position_start;
        }

        return ok;
    }

    // =========================================================================
    // get_k / get_v / get_kv
    // =========================================================================

    ITensor *ROCmRingKVCacheTQ::get_k(int layer, int seq_idx)
    {
        return const_cast<ITensor *>(std::as_const(*this).get_k(layer, seq_idx));
    }

    ITensor *ROCmRingKVCacheTQ::get_v(int layer, int seq_idx)
    {
        return const_cast<ITensor *>(std::as_const(*this).get_v(layer, seq_idx));
    }

    const ITensor *ROCmRingKVCacheTQ::get_k(int layer, int seq_idx) const
    {
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
            return nullptr;

        hipSetDevice(device_id_);
        dequant_to_scratch(layer, seq_idx, 0.0f, 0, clearStream());
        auto &scratch = layer_scratch_[layer];

        if (!scratch.k_view)
        {
            scratch.k_view = std::make_unique<GpuTensorView>(
                scratch.d_K, scratch.cached_count, kv_dim_,
                TensorType::FP16, device_id_);
        }
        else
        {
            static_cast<GpuTensorView*>(scratch.k_view.get())->update_view(
                scratch.d_K, scratch.cached_count);
        }
        return scratch.k_view.get();
    }

    const ITensor *ROCmRingKVCacheTQ::get_v(int layer, int seq_idx) const
    {
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
            return nullptr;

        hipSetDevice(device_id_);
        dequant_to_scratch(layer, seq_idx, 0.0f, 0, clearStream());
        auto &scratch = layer_scratch_[layer];

        if (!scratch.v_view)
        {
            scratch.v_view = std::make_unique<GpuTensorView>(
                scratch.d_V, scratch.cached_count, kv_dim_,
                TensorType::FP16, device_id_);
        }
        else
        {
            static_cast<GpuTensorView*>(scratch.v_view.get())->update_view(
                scratch.d_V, scratch.cached_count);
        }
        return scratch.v_view.get();
    }

    bool ROCmRingKVCacheTQ::get_kv(int layer, int seq_idx,
                                    ITensor **out_k, ITensor **out_v,
                                    int *out_kv_len)
    {
        auto *k = get_k(layer, seq_idx);
        auto *v = get_v(layer, seq_idx);
        if (!k || !v)
            return false;
        if (out_k) *out_k = k;
        if (out_v) *out_v = v;
        if (out_kv_len) *out_kv_len = get_cached_tokens(layer, seq_idx);
        return true;
    }

    bool ROCmRingKVCacheTQ::get_kv(int layer, int seq_idx,
                                    const ITensor **out_k, const ITensor **out_v,
                                    int *out_kv_len) const
    {
        auto *k = get_k(layer, seq_idx);
        auto *v = get_v(layer, seq_idx);
        if (!k || !v)
            return false;
        if (out_k) *out_k = k;
        if (out_v) *out_v = v;
        if (out_kv_len) *out_kv_len = get_cached_tokens(layer, seq_idx);
        return true;
    }

    // =========================================================================
    // get_kv_converted (dequant + optional RoPE → FP16 scratch)
    // =========================================================================

    bool ROCmRingKVCacheTQ::get_kv_converted(
        int layer, int seq_idx,
        ActivationPrecision target,
        ITensor **out_k, ITensor **out_v,
        int *out_kv_len,
        const KVReadParams *rope)
    {
        (void)target;

        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
            return false;

        hipSetDevice(device_id_);
        const auto &entry = entries_[layer][seq_idx];

        if (entry.count == 0)
        {
            if (out_k)
                *out_k = nullptr;
            if (out_v)
                *out_v = nullptr;
            if (out_kv_len)
                *out_kv_len = 0;
            return true;
        }

        const float rope_theta = (rope && rope->rope_theta > 0.0f) ? rope->rope_theta : 0.0f;
        const int position_start = (rope && rope_theta > 0.0f) ? rope->position_start : 0;
        hipStream_t stream = (rope && rope->gpu_stream)
                                 ? static_cast<hipStream_t>(rope->gpu_stream)
                                 : clearStream();
        bool ok = dequant_to_scratch(layer, seq_idx, rope_theta, position_start, stream);
        if (!ok)
            return false;

        auto &scratch = layer_scratch_[layer];

        if (!scratch.k_view)
        {
            scratch.k_view = std::make_unique<GpuTensorView>(
                scratch.d_K, entry.count, kv_dim_,
                TensorType::FP16, device_id_);
        }
        else
        {
            static_cast<GpuTensorView*>(scratch.k_view.get())->update_view(
                scratch.d_K, entry.count);
        }
        if (!scratch.v_view)
        {
            scratch.v_view = std::make_unique<GpuTensorView>(
                scratch.d_V, entry.count, kv_dim_,
                TensorType::FP16, device_id_);
        }
        else
        {
            static_cast<GpuTensorView*>(scratch.v_view.get())->update_view(
                scratch.d_V, entry.count);
        }

        if (out_k) *out_k = scratch.k_view.get();
        if (out_v) *out_v = scratch.v_view.get();
        if (out_kv_len) *out_kv_len = entry.count;

        return true;
    }

    // =========================================================================
    // Graph Capture Support
    // =========================================================================

    void ROCmRingKVCacheTQ::setDynamicHead(int layer, int seq_idx, void *gpu_stream)
    {
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
            return;
        if (!d_dequant_params_ || !h_dequant_params_)
            return;

        (void)gpu_stream;
        // Append head params are handled by ROCmRingKVCacheBase. TQ dequant
        // metadata is prepared through setDynamicDequantParams().
    }

    void ROCmRingKVCacheTQ::setDynamicDequantParams(
        int layer, int seq_idx,
        float rope_theta, int position_start,
        void *gpu_stream)
    {
        if (!d_dequant_params_ || !h_dequant_params_)
            return;
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
            return;
        dequant_params_device_valid_[static_cast<size_t>(layer)] = 0;

        const auto &entry = entries_[layer][seq_idx];
        h_dequant_params_[layer].ring_pos = entry.head;
        h_dequant_params_[layer].out_offset_elems = entry.count * kv_dim_;
        h_dequant_params_[layer].rope_position =
            (rope_theta > 0.0f) ? position_start + entry.count : 0;

        auto *stream = static_cast<hipStream_t>(gpu_stream);
        if (!stream)
        {
            LOG_ERROR("[ROCmRingKVCacheTQ::setDynamicDequantParams] Explicit HIP stream required");
            return;
        }

        hipStreamCaptureStatus status = hipStreamCaptureStatusNone;
        const hipError_t cap_err = hipStreamIsCapturing(stream, &status);
        if (cap_err != hipSuccess)
        {
            LOG_ERROR("[ROCmRingKVCacheTQ::setDynamicDequantParams] hipStreamIsCapturing failed: "
                      << hipGetErrorString(cap_err));
            return;
        }
        if (isGraphCaptureActive() || status == hipStreamCaptureStatusActive)
        {
            LOG_ERROR("[ROCmRingKVCacheTQ::setDynamicDequantParams] Refusing to record TQ dequant-param H2D inside HIP graph capture");
            return;
        }

        const hipError_t copy_err =
            hipMemcpyAsync(&d_dequant_params_[layer], &h_dequant_params_[layer],
                           sizeof(HIPTQDequantDynamicParams), hipMemcpyHostToDevice, stream);
        if (copy_err != hipSuccess)
        {
            LOG_ERROR("[ROCmRingKVCacheTQ::setDynamicDequantParams] hipMemcpyAsync failed: "
                      << hipGetErrorString(copy_err));
            return;
        }
        dequant_params_device_valid_[static_cast<size_t>(layer)] = 1;
    }

    // advanceHead() is now provided by ROCmRingKVCacheBase.
    // clear(), clear_sequence(), clear_layer() are now provided by
    // ROCmRingKVCacheBase via entry accessor overrides (resetEntry, onClearSequence).

    // =========================================================================
    // Eviction
    // =========================================================================

    void ROCmRingKVCacheTQ::evict_oldest(int layer, int seq_idx, int num_tokens)
    {
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
            return;

        auto &entry = entries_[layer][seq_idx];
        if (num_tokens >= entry.count)
        {
            clear_sequence(layer, seq_idx);
            return;
        }

        entry.count -= num_tokens;

        layer_scratch_[layer].invalidate();
    }

    // =========================================================================
    // Factory function (callable from non-HIP code via Factory header)
    // =========================================================================

    std::unique_ptr<IKVCache> createROCmRingKVCacheTQ(
        int n_layers, int batch_size, int max_seq_len,
        int n_kv_heads, int head_dim,
        const TurboQuantContext *tq_ctx,
        int device_id)
    {
        return std::make_unique<ROCmRingKVCacheTQ>(
            n_layers, batch_size, max_seq_len,
            n_kv_heads, head_dim, tq_ctx, device_id);
    }

} // namespace llaminar2
