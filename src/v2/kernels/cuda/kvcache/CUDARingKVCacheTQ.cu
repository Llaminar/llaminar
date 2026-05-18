/**
 * @file CUDARingKVCacheTQ.cu
 * @brief Implementation of CUDARingKVCacheTQ - TurboQuant KV cache on CUDA
 * @author David Sanftenberg
 */

#include "CUDARingKVCacheTQ.h"
#include "CUDATurboQuantKernels.h"
#include "../../../tensors/GpuTensorView.h"
#include "../../../kernels/cpu/turboquant/TurboQuantContext.h"
#include "../../../backends/GPUDeviceContextPool.h"
#include "../../../utils/Logger.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cassert>
#include <cstring>

namespace llaminar2
{

    // =========================================================================
    // Construction / Destruction
    // =========================================================================

    CUDARingKVCacheTQ::CUDARingKVCacheTQ(
        int n_layers, int batch_size, int max_seq_len,
        int n_kv_heads, int head_dim,
        const TurboQuantContext *tq_ctx,
        int device_id)
        : CUDARingKVCacheBase(n_layers, batch_size, max_seq_len,
                              n_kv_heads, head_dim, n_kv_heads * head_dim, device_id),
          tq_ctx_(tq_ctx)
    {
        cudaSetDevice(device_id);

        // Determine block sizes
        if (head_dim == 64)
        {
            k_block_size_ = sizeof(TQ8Block<64>);
            v_block_size_ = sizeof(TQ4Block<64>);
        }
        else if (head_dim == 128)
        {
            k_block_size_ = sizeof(TQ8Block<128>);
            v_block_size_ = sizeof(TQ4Block<128>);
        }
        else
        {
            LOG_ERROR("CUDARingKVCacheTQ: unsupported head_dim=" << head_dim);
            throw std::runtime_error("CUDARingKVCacheTQ: head_dim must be 64 or 128");
        }

        k_pos_bytes_ = static_cast<size_t>(n_kv_heads) * k_block_size_;
        v_pos_bytes_ = static_cast<size_t>(n_kv_heads) * v_block_size_;

        LOG_DEBUG("CUDARingKVCacheTQ: K block=" << k_block_size_ << "B, V block=" << v_block_size_
                                               << "B, per-position: K=" << k_pos_bytes_ << "B V=" << v_pos_bytes_ << "B"
                                               << " (vs FP16: " << (n_kv_heads * head_dim * 2 * 2) << "B)");

        // Upload codebooks to constant memory
        cuda_tq_upload_codebooks(0);

        // Create GPU rotation matrices from TurboQuantContext
        if (tq_ctx)
        {
            rotations_ = cuda_tq_create_rotations(
                n_layers, n_kv_heads, head_dim,
                tq_ctx->rotation().seed, device_id, 0);

            // Diagnostic: verify GPU rotation matches CPU rotation
            {
                const auto &layer0_ctx = tq_ctx->for_layer(0);
                const auto &head0_ctx = layer0_ctx.for_layer(0);
                const auto &cpu_rot = head0_ctx.rotation();
                float gpu_rot_val = 0.0f;
                cudaStream_t init_stream = static_cast<cudaStream_t>(
                    GPUDeviceContextPool::instance().getNvidiaContext(device_id).defaultStream());
                cudaMemcpyAsync(&gpu_rot_val, rotations_.d_rotations,
                                sizeof(float), cudaMemcpyDeviceToHost, init_stream);
                cudaStreamSynchronize(init_stream);
                LOG_DEBUG("[CUDARingKVCacheTQ] Rotation check: CPU[0,0]=" << cpu_rot.matrix[0]
                                                                         << " GPU[0,0]=" << gpu_rot_val
                                                                         << " match=" << (std::abs(cpu_rot.matrix[0] - gpu_rot_val) < 1e-6f)
                                                                         << " seed=" << tq_ctx->rotation().seed);
            }
        }
        else
        {
            LOG_ERROR("CUDARingKVCacheTQ: null TurboQuantContext");
            throw std::runtime_error("CUDARingKVCacheTQ requires TurboQuantContext");
        }

        // Allocate per-layer TQ ring buffers
        entries_.resize(n_layers);
        for (int l = 0; l < n_layers; ++l)
        {
            entries_[l].resize(batch_size);
            for (int b = 0; b < batch_size; ++b)
            {
                allocate_entry(entries_[l][b]);
            }
        }

        // Allocate per-layer FP16 scratch buffers (enables incremental dequant + FP16 flash attention)
        {
            const size_t scratch_bytes = static_cast<size_t>(max_seq_len) * kv_dim_ * sizeof(__half);
            layer_scratch_.resize(n_layers);
            for (int l = 0; l < n_layers; ++l)
            {
                cudaMalloc(&layer_scratch_[l].d_K, scratch_bytes);
                cudaMalloc(&layer_scratch_[l].d_V, scratch_bytes);
                layer_scratch_[l].invalidate();
            }

            const size_t total_tq_bytes = static_cast<size_t>(n_layers) * batch_size *
                                          max_seq_len * (k_pos_bytes_ + v_pos_bytes_);
            const size_t total_scratch_bytes = static_cast<size_t>(n_layers) * 2 * scratch_bytes;
            LOG_DEBUG("CUDARingKVCacheTQ VRAM: TQ caches="
                     << (total_tq_bytes / 1024) << "KB, per-layer FP16 scratch="
                     << (total_scratch_bytes / (1024 * 1024)) << "MB (" << n_layers << " layers)");
        }

        // Pre-allocate temp buffers for GPU-side quantization (avoids per-call cudaMalloc)
        {
            const size_t k_temp_bytes = static_cast<size_t>(max_seq_len) * n_kv_heads * k_block_size_;
            const size_t v_temp_bytes = static_cast<size_t>(max_seq_len) * n_kv_heads * v_block_size_;
            cudaMalloc(&d_quantize_k_temp_, k_temp_bytes);
            cudaMalloc(&d_quantize_v_temp_, v_temp_bytes);
            LOG_DEBUG("CUDARingKVCacheTQ: pre-allocated quantize temp buffers: K="
                      << (k_temp_bytes / 1024) << "KB, V=" << (v_temp_bytes / 1024) << "KB");
        }

        // Allocate device-side dynamic params for CUDA graph capture
        // d_head_params_ and h_head_params_ are allocated by CUDARingKVCacheBase
        allocateDeviceParams();
        {
            // TQ-specific dequant params
            cudaMalloc(&d_dequant_params_, n_layers * sizeof(TQDequantDynamicParams));
            cudaMallocHost(&h_dequant_params_, n_layers * sizeof(TQDequantDynamicParams));
            memset(h_dequant_params_, 0, n_layers * sizeof(TQDequantDynamicParams));
        }

        LOG_DEBUG("CUDARingKVCacheTQ created: " << n_layers << " layers, "
                                               << max_seq_len << " max_seq_len, " << n_kv_heads << " KV heads, "
                                               << head_dim << " head_dim on cuda:" << device_id);
    }

    CUDARingKVCacheTQ::~CUDARingKVCacheTQ()
    {
        cudaSetDevice(device_id_);

        for (auto &layer : entries_)
        {
            for (auto &entry : layer)
                free_entry(entry);
        }

        // Free per-layer scratch buffers
        for (auto &scratch : layer_scratch_)
        {
            scratch.k_view.reset();
            scratch.v_view.reset();
            if (scratch.d_K)
                cudaFree(scratch.d_K);
            if (scratch.d_V)
                cudaFree(scratch.d_V);
        }
        layer_scratch_.clear();

        // Free pre-allocated quantize temp buffers
        if (d_quantize_k_temp_)
            cudaFree(d_quantize_k_temp_);
        if (d_quantize_v_temp_)
            cudaFree(d_quantize_v_temp_);

        // Free graph capture dynamic params (TQ-specific only)
        // Note: d_head_params_/h_head_params_ are freed by ~CUDARingKVCacheBase()
        if (d_dequant_params_)
            cudaFree(d_dequant_params_);
        if (h_dequant_params_)
            cudaFreeHost(h_dequant_params_);

        cuda_tq_free_rotations(rotations_);
    }

    // =========================================================================
    // Graph Capture Support (TQ-specific: setDynamicDequantParams)
    // Note: setDynamicHead and advanceHead are now in CUDARingKVCacheBase
    // =========================================================================

    void CUDARingKVCacheTQ::setDynamicDequantParams(
        int layer, int seq_idx,
        float rope_theta, int position_start,
        void * /*gpu_stream*/)
    {
        if (!d_dequant_params_ || !h_dequant_params_)
            return;
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
            return;

        const auto &entry = entries_[layer][seq_idx];

        // Compute dequant params for the NEXT incremental step.
        // At this point, entry.count and entry.head reflect the state
        // BEFORE this step's append. After append:
        //   - The new token is written at entry.head (current value)
        //   - The new token occupies scratch slot entry.count (0-indexed)
        h_dequant_params_[layer].ring_pos = entry.head;
        h_dequant_params_[layer].out_offset_elems = entry.count * kv_dim_;
        h_dequant_params_[layer].rope_position =
            (rope_theta > 0.0f) ? position_start + entry.count : 0;
    }

    // =========================================================================
    // Memory Management
    // =========================================================================

    void CUDARingKVCacheTQ::allocate_entry(TQEntry &entry)
    {
        const size_t k_bytes = static_cast<size_t>(max_seq_len_) * k_pos_bytes_;
        const size_t v_bytes = static_cast<size_t>(max_seq_len_) * v_pos_bytes_;

        cudaMalloc(&entry.d_K, k_bytes);
        cudaMalloc(&entry.d_V, v_bytes);
        cudaStream_t alloc_stream = static_cast<cudaStream_t>(
            GPUDeviceContextPool::instance().getNvidiaContext(device_id_).defaultStream());
        cudaMemsetAsync(entry.d_K, 0, k_bytes, alloc_stream);
        cudaMemsetAsync(entry.d_V, 0, v_bytes, alloc_stream);
        cudaStreamSynchronize(alloc_stream);

        entry.head = 0;
        entry.count = 0;
    }

    void CUDARingKVCacheTQ::free_entry(TQEntry &entry)
    {
        if (entry.d_K)
        {
            cudaFree(entry.d_K);
            entry.d_K = nullptr;
        }
        if (entry.d_V)
        {
            cudaFree(entry.d_V);
            entry.d_V = nullptr;
        }
    }

    // =========================================================================
    // IKVCache Basic Operations (now in CUDARingKVCacheBase)
    // get_cached_tokens, clear, clear_sequence, clear_layer,
    // get_head_position, is_wrapped are all inherited.
    // =========================================================================

    // =========================================================================
    // Append (FP32 → TQ8 K / TQ4 V on GPU)
    // =========================================================================

    bool CUDARingKVCacheTQ::append(int layer, int seq_idx,
                                   const ITensor *K, const ITensor *V,
                                   int num_tokens)
    {
        return appendWithStream(layer, seq_idx, K, V, num_tokens, nullptr);
    }

    bool CUDARingKVCacheTQ::appendWithStream(int layer, int seq_idx,
                                             const ITensor *K, const ITensor *V,
                                             int num_tokens, void *gpu_stream)
    {
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
            return false;
        if (num_tokens <= 0)
            return true;

        cudaSetDevice(device_id_);
        cudaStream_t stream = static_cast<cudaStream_t>(gpu_stream);
        cached_stream_ = stream; // Cache for get_kv_converted()

        TQEntry &entry = entries_[layer][seq_idx];

        // One-time warning when ring buffer starts wrapping
        if (entry.count + num_tokens > max_seq_len_ && !wrap_warned_)
        {
            LOG_WARN("Context window full (" << max_seq_len_
                                             << " tokens). Sliding window is now overwriting oldest tokens. "
                                             << "Use -c <size> to increase context length.");
            wrap_warned_ = true;
        }

        const bool k_is_tq = (K->native_type() == TensorType::TQ8 || K->native_type() == TensorType::TQ4);
        const bool v_is_tq = (V->native_type() == TensorType::TQ4);

        LOG_DEBUG("[CUDARingKVCacheTQ::append] layer=" << layer
                                                       << " seq=" << seq_idx << " tokens=" << num_tokens
                                                       << " K.is_gpu=" << K->is_on_gpu()
                                                       << " K.dtype=" << K->dtype_name()
                                                       << " V.dtype=" << V->dtype_name()
                                                       << " pre_quantized=" << (k_is_tq && v_is_tq));

        // ================================================================
        // Fast path: K and V are already TQ-quantized blocks
        // (from KVCacheAppendStage which pre-quantizes on CPU)
        // Just upload blocks directly to the GPU ring buffer.
        // ================================================================
        if (k_is_tq && v_is_tq)
        {
            const uint8_t *h_K_blocks = static_cast<const uint8_t *>(K->raw_data());
            const uint8_t *h_V_blocks = static_cast<const uint8_t *>(V->raw_data());
            if (!h_K_blocks || !h_V_blocks)
                return false;

            const size_t k_row_bytes = static_cast<size_t>(n_kv_heads_) * k_block_size_;
            const size_t v_row_bytes = static_cast<size_t>(n_kv_heads_) * v_block_size_;

            // Copy each token's blocks to the correct ring position
            for (int t = 0; t < num_tokens; ++t)
            {
                const int pos = (entry.head + t) % max_seq_len_;
                const size_t k_dst_offset = static_cast<size_t>(pos) * k_row_bytes;
                const size_t v_dst_offset = static_cast<size_t>(pos) * v_row_bytes;
                const size_t k_src_offset = static_cast<size_t>(t) * k_row_bytes;
                const size_t v_src_offset = static_cast<size_t>(t) * v_row_bytes;

                cudaMemcpyAsync(
                    static_cast<uint8_t *>(entry.d_K) + k_dst_offset,
                    h_K_blocks + k_src_offset,
                    k_row_bytes, cudaMemcpyHostToDevice, stream);
                cudaMemcpyAsync(
                    static_cast<uint8_t *>(entry.d_V) + v_dst_offset,
                    h_V_blocks + v_src_offset,
                    v_row_bytes, cudaMemcpyHostToDevice, stream);
            }

            // Update ring buffer state
            entry.head = (entry.head + num_tokens) % max_seq_len_;
            entry.count = std::min(entry.count + num_tokens, max_seq_len_);

            // Invalidate scratch if it holds data for this (layer, seq)
            if (layer_scratch_[layer].cached_layer == layer && layer_scratch_[layer].cached_seq == seq_idx)
                layer_scratch_[layer].invalidate();

            return true;
        }

        // ================================================================
        // GPU quantize path: K and V are FP32 — quantize on GPU
        // ================================================================

        // Fast path: single token + data already on GPU (decode hot path)
        // Fused kernel writes directly to ring buffer — no temp, no memcpy.
        if (num_tokens == 1 && K->is_on_gpu() && V->is_on_gpu())
        {
            const float *d_K_new = static_cast<const float *>(K->gpu_data_ptr());
            const float *d_V_new = static_cast<const float *>(V->gpu_data_ptr());
            const int D = head_dim_;
            const size_t layer_rot_offset = static_cast<size_t>(layer) * n_kv_heads_ * D * D;
            const float *d_rot = rotations_.d_rotations + layer_rot_offset;

            bool ok;
            if (d_head_params_ && h_head_params_ && stream)
            {
                // Graph-capturable path: H2D of ring_pos + dynamic kernel.
                // During capture, both the H2D and kernel are recorded.
                // On replay, setDynamicHead() updates h_head_params_ before
                // the captured H2D re-reads it.
                const int idx = layer * batch_size_ + seq_idx;
                h_head_params_[idx] = entry.head;
                cudaMemcpyAsync(&d_head_params_[idx], &h_head_params_[idx],
                                sizeof(int), cudaMemcpyHostToDevice, stream);
                ok = cuda_tq_quantize_fused_ring_dynamic(
                    d_K_new, d_V_new, d_rot,
                    entry.d_K, entry.d_V,
                    &d_head_params_[idx], n_kv_heads_, head_dim_, stream);
            }
            else
            {
                const int ring_pos = entry.head % max_seq_len_;
                ok = cuda_tq_quantize_fused_ring(
                    d_K_new, d_V_new, d_rot,
                    entry.d_K, entry.d_V,
                    ring_pos, n_kv_heads_, head_dim_, stream);
            }

            entry.head = (entry.head + 1) % max_seq_len_;
            entry.count = std::min(entry.count + 1, max_seq_len_);
            return ok;
        }

        // General path: multi-token or host data (prefill)
        // Uses temp buffers + D2D memcpy for ring-wrapping.
        const float *d_K_new = nullptr;
        const float *d_V_new = nullptr;
        void *d_K_uploaded = nullptr;
        void *d_V_uploaded = nullptr;

        if (K->is_on_gpu())
        {
            d_K_new = static_cast<const float *>(K->gpu_data_ptr());
        }
        else
        {
            // Need to upload K from host — use pre-allocated scratch as staging
            const float *h_K = K->data();
            if (!h_K)
                return false;
            size_t k_bytes = static_cast<size_t>(num_tokens) * kv_dim_ * sizeof(float);
            cudaMalloc(&d_K_uploaded, k_bytes);
            cudaMemcpyAsync(d_K_uploaded, h_K, k_bytes, cudaMemcpyHostToDevice, stream);
            d_K_new = static_cast<const float *>(d_K_uploaded);
        }

        if (V->is_on_gpu())
        {
            d_V_new = static_cast<const float *>(V->gpu_data_ptr());
        }
        else
        {
            const float *h_V = V->data();
            if (!h_V)
            {
                if (d_K_uploaded)
                    cudaFree(d_K_uploaded);
                return false;
            }
            size_t v_bytes = static_cast<size_t>(num_tokens) * kv_dim_ * sizeof(float);
            cudaMalloc(&d_V_uploaded, v_bytes);
            cudaMemcpyAsync(d_V_uploaded, h_V, v_bytes, cudaMemcpyHostToDevice, stream);
            d_V_new = static_cast<const float *>(d_V_uploaded);
        }

        // Get rotation matrices for this layer
        const int D = head_dim_;
        const size_t layer_rot_offset = static_cast<size_t>(layer) * n_kv_heads_ * D * D;
        const float *d_K_rot = rotations_.d_rotations + layer_rot_offset;

        // Quantize into pre-allocated temp buffers (no per-call cudaMalloc!)
        bool ok = cuda_tq8_quantize(d_K_new, d_K_rot, d_quantize_k_temp_,
                                    num_tokens, n_kv_heads_, head_dim_, stream);
        if (ok)
        {
            ok = cuda_tq4_quantize(d_V_new, d_K_rot, d_quantize_v_temp_,
                                   num_tokens, n_kv_heads_, head_dim_, stream);
        }

        if (ok)
        {
            // Copy quantized blocks from temp to ring buffer positions
            const size_t k_pos_bytes = static_cast<size_t>(n_kv_heads_) * k_block_size_;
            const size_t v_pos_bytes = static_cast<size_t>(n_kv_heads_) * v_block_size_;

            for (int t = 0; t < num_tokens; ++t)
            {
                int dst_pos = (entry.head + t) % max_seq_len_;
                cudaMemcpyAsync(
                    static_cast<uint8_t *>(entry.d_K) + static_cast<size_t>(dst_pos) * k_pos_bytes,
                    static_cast<uint8_t *>(d_quantize_k_temp_) + static_cast<size_t>(t) * k_pos_bytes,
                    k_pos_bytes, cudaMemcpyDeviceToDevice, stream);
                cudaMemcpyAsync(
                    static_cast<uint8_t *>(entry.d_V) + static_cast<size_t>(dst_pos) * v_pos_bytes,
                    static_cast<uint8_t *>(d_quantize_v_temp_) + static_cast<size_t>(t) * v_pos_bytes,
                    v_pos_bytes, cudaMemcpyDeviceToDevice, stream);
            }
        }

        // Update ring buffer state
        entry.head = (entry.head + num_tokens) % max_seq_len_;
        entry.count = std::min(entry.count + num_tokens, max_seq_len_);

        // Invalidate scratch — we wrote new data that the scratch doesn't know about
        // (scratch still holds the OLD dequanted content; new position needs dequant)
        // DON'T invalidate — let incremental dequant handle it!
        // The per-layer scratch has count-1 positions dequanted; the new position
        // will be dequanted incrementally on next get_kv_converted().

        // Free temp buffers if we uploaded from host
        if (d_K_uploaded)
            cudaFree(d_K_uploaded);
        if (d_V_uploaded)
            cudaFree(d_V_uploaded);

        return ok;
    }

    // =========================================================================
    // Per-Layer Scratch Buffer Management
    // =========================================================================

    bool CUDARingKVCacheTQ::dequant_to_scratch(
        int layer, int seq_idx, float rope_theta, int position_start, cudaStream_t stream) const
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
                // Graph-capturable path: H2D of dequant params + dynamic kernel.
                // Passes base scratch pointers — kernel reads offset from device params.
                // During capture, H2D + kernel are recorded. On replay,
                // setDynamicDequantParams() updates h_dequant_params_ before
                // the captured H2D re-reads it.
                h_dequant_params_[layer].ring_pos = (tail + entry.count - 1) % max_seq_len_;
                h_dequant_params_[layer].out_offset_elems = static_cast<int>((entry.count - 1) * kv_dim_);
                h_dequant_params_[layer].rope_position =
                    (rope_theta > 0.0f) ? position_start + entry.count - 1 : 0;

                cudaMemcpyAsync(&d_dequant_params_[layer], &h_dequant_params_[layer],
                                sizeof(TQDequantDynamicParams), cudaMemcpyHostToDevice, stream);

                ok = cuda_tq_incremental_single_fp16_dynamic(
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

                ok = cuda_tq_incremental_single_fp16(
                    scratch.d_K + out_offset,
                    scratch.d_V + out_offset,
                    entry.d_K, entry.d_V,
                    d_K_rot, d_V_rot,
                    new_ring_pos, 0, // out_offset_elems=0 since we offset the pointers
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
        bool ok = cuda_tq_ring_linearize_dequant_fp16(
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
    // get_k / get_v (dequant to shared FP32 scratch, return view)
    // =========================================================================
    //
    // NOTE: The returned ITensor* points into the per-layer scratch buffer.
    // Each layer has its own scratch, so there are no cross-layer conflicts.
    //

    ITensor *CUDARingKVCacheTQ::get_k(int layer, int seq_idx)
    {
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
            return nullptr;

        cudaSetDevice(device_id_);

        // Dequant into per-layer scratch (no RoPE for raw get_k)
        dequant_to_scratch(layer, seq_idx, 0.0f, 0, cached_stream_);

        auto &scratch = layer_scratch_[layer];
        const auto &entry = entries_[layer][seq_idx];

        if (!scratch.k_view)
        {
            scratch.k_view = std::make_unique<GpuTensorView>(
                scratch.d_K, entry.count, kv_dim_,
                TensorType::FP16, device_id_);
        }
        else
        {
            static_cast<GpuTensorView *>(scratch.k_view.get())->update_view(scratch.d_K, entry.count);
        }

        return scratch.k_view.get();
    }

    const ITensor *CUDARingKVCacheTQ::get_k(int layer, int seq_idx) const
    {
        return const_cast<CUDARingKVCacheTQ *>(this)->get_k(layer, seq_idx);
    }

    ITensor *CUDARingKVCacheTQ::get_v(int layer, int seq_idx)
    {
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
            return nullptr;

        cudaSetDevice(device_id_);

        dequant_to_scratch(layer, seq_idx, 0.0f, 0, cached_stream_);

        auto &scratch = layer_scratch_[layer];
        const auto &entry = entries_[layer][seq_idx];

        if (!scratch.v_view)
        {
            scratch.v_view = std::make_unique<GpuTensorView>(
                scratch.d_V, entry.count, kv_dim_,
                TensorType::FP16, device_id_);
        }
        else
        {
            static_cast<GpuTensorView *>(scratch.v_view.get())->update_view(scratch.d_V, entry.count);
        }

        return scratch.v_view.get();
    }

    const ITensor *CUDARingKVCacheTQ::get_v(int layer, int seq_idx) const
    {
        return const_cast<CUDARingKVCacheTQ *>(this)->get_v(layer, seq_idx);
    }

    // =========================================================================
    // get_kv (unified K+V access)
    // =========================================================================

    bool CUDARingKVCacheTQ::get_kv(int layer, int seq_idx,
                                   ITensor **out_k, ITensor **out_v,
                                   int *out_kv_len)
    {
        auto *k = get_k(layer, seq_idx);
        auto *v = get_v(layer, seq_idx);
        if (!k || !v)
            return false;
        if (out_k)
            *out_k = k;
        if (out_v)
            *out_v = v;
        if (out_kv_len)
            *out_kv_len = get_cached_tokens(layer, seq_idx);
        return true;
    }

    bool CUDARingKVCacheTQ::get_kv(int layer, int seq_idx,
                                   const ITensor **out_k, const ITensor **out_v,
                                   int *out_kv_len) const
    {
        auto *k = get_k(layer, seq_idx);
        auto *v = get_v(layer, seq_idx);
        if (!k || !v)
            return false;
        if (out_k)
            *out_k = k;
        if (out_v)
            *out_v = v;
        if (out_kv_len)
            *out_kv_len = get_cached_tokens(layer, seq_idx);
        return true;
    }

    // =========================================================================
    // get_kv_converted (dequant + optional RoPE → FP32 scratch)
    // =========================================================================

    bool CUDARingKVCacheTQ::get_kv_converted(
        int layer, int seq_idx,
        ActivationPrecision target,
        ITensor **out_k, ITensor **out_v,
        int *out_kv_len,
        const KVReadParams *rope)
    {
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
            return false;

        const auto &entry = entries_[layer][seq_idx];
        if (entry.count == 0)
        {
            if (out_kv_len)
                *out_kv_len = 0;
            return true;
        }

        cudaSetDevice(device_id_);

        // Determine RoPE parameters
        float rope_theta = 0.0f;
        int position_start = 0;
        if (rope && rope->rope_theta > 0.0f)
        {
            rope_theta = rope->rope_theta;
            position_start = rope->position_start;
        }

        // Dequant to per-layer FP32 scratch with optional RoPE
        // IMPORTANT: Must use cached_stream_ (set by append()), NOT nullptr.
        bool ok = dequant_to_scratch(layer, seq_idx, rope_theta, position_start, cached_stream_);
        if (!ok)
            return false;

        auto &scratch = layer_scratch_[layer];

        // Create/update FP16 scratch views (enables FP16 flash attention path)
        if (!scratch.k_view)
        {
            scratch.k_view = std::make_unique<GpuTensorView>(
                scratch.d_K, entry.count, kv_dim_,
                TensorType::FP16, device_id_);
        }
        else
        {
            static_cast<GpuTensorView *>(scratch.k_view.get())->update_view(scratch.d_K, entry.count);
        }
        if (!scratch.v_view)
        {
            scratch.v_view = std::make_unique<GpuTensorView>(
                scratch.d_V, entry.count, kv_dim_,
                TensorType::FP16, device_id_);
        }
        else
        {
            static_cast<GpuTensorView *>(scratch.v_view.get())->update_view(scratch.d_V, entry.count);
        }

        if (out_k)
            *out_k = scratch.k_view.get();
        if (out_v)
            *out_v = scratch.v_view.get();
        if (out_kv_len)
            *out_kv_len = entry.count;

        return true;
    }

    // =========================================================================
    // Eviction
    // =========================================================================

    void CUDARingKVCacheTQ::evict_oldest(int layer, int seq_idx, int num_tokens)
    {
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
            return;

        auto &entry = entries_[layer][seq_idx];
        int evict = std::min(num_tokens, entry.count);
        entry.count -= evict;

        // Invalidate per-layer scratch (eviction changes the tail position)
        layer_scratch_[layer].invalidate();
    }

} // namespace llaminar2
