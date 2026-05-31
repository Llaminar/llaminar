/**
 * @file CPURingKVCache.cpp
 * @brief Implementation of the ring-buffer KV cache for CPU-side transformer attention.
 *
 * This file contains all template method implementations for CPURingKVCache,
 * as well as the non-template factory functions. The template is explicitly
 * instantiated at the bottom of this file for all supported precisions.
 *
 * ## Key Concepts
 *
 * - **Ring buffer**: A fixed-size circular buffer where `head` points to the
 *   oldest token and `size` tracks the count of valid tokens. When full,
 *   new tokens overwrite the oldest.
 *
 * - **POSITION_MAJOR layout**: Each row = one token position, columns = all
 *   heads concatenated (`n_kv_heads * head_dim`). Simple row-level memcpy
 *   for append.
 *
 * - **HEAD_MAJOR layout**: Memory organized as `[head][position][head_dim]`.
 *   Requires per-head scatter/gather on append, but attention kernels can
 *   access each head's data contiguously.
 *
 * - **Quantized formats**: Q8_1 and Q16_1 store data as blocks rather than
 *   individual elements, so copy sizes are computed in terms of
 *   `blocks_per_row * sizeof(Block)` rather than `cols * sizeof(float)`.
 */

#include "CPURingKVCache.h"

#include "turboquant/TurboQuantDequantizeTQ4.h"
#include "turboquant/TurboQuantDequantizeSplitTQ.h"
#include "../../tensors/SIMDHelpers.h"

#include <algorithm>
#include <cstring>

namespace llaminar2
{

    namespace
    {
        /**
         * @brief Convert a legacy integer device index to a typed DeviceId.
         *
         * Legacy convention used by older code paths:
         *   - 0 or negative → CPU
         *   - 1 → cuda:0, 2 → cuda:1, etc.
         *
         * @param legacy_idx  Legacy device index.
         * @return Corresponding DeviceId.
         */
        DeviceId deviceIdFromLegacyIndex(int legacy_idx)
        {
            if (legacy_idx <= 0)
            {
                return DeviceId::cpu();
            }
            return DeviceId::cuda(legacy_idx - 1);
        }
    }

    // =========================================================================
    // Constructors
    // =========================================================================
    // All constructors ultimately delegate to the "primary" constructor that
    // takes (n_kv_heads, local_n_kv_heads, kv_head_start, head_dim, device).
    // The simpler constructors set local_n_kv_heads = n_kv_heads (no sharding)
    // and then optionally apply per-layer device overrides.
    // =========================================================================

    /**
     * Non-sharded, uniform device constructor.
     * Delegates to the sharded constructor with local_n_kv_heads == n_kv_heads.
     */
    template <ActivationPrecision KPrecision, ActivationPrecision VPrecision>
    CPURingKVCache<KPrecision, VPrecision>::CPURingKVCache(
        const IMPIContext &mpi_ctx, int n_layers, int batch_size, int max_seq_len,
        int n_kv_heads, int head_dim, DeviceId device,
        KVCacheLayoutMode layout_mode)
        : CPURingKVCache(mpi_ctx, n_layers, batch_size, max_seq_len,
                         n_kv_heads, n_kv_heads, 0, head_dim,
                         device, layout_mode)
    {
    }

    /**
     * Non-sharded, per-layer device constructor.
     * First delegates to the uniform-device constructor (with CPU as default),
     * then overrides each layer's device from the attention_devices vector
     * and re-initializes layers on their target devices.
     */
    template <ActivationPrecision KPrecision, ActivationPrecision VPrecision>
    CPURingKVCache<KPrecision, VPrecision>::CPURingKVCache(
        const IMPIContext &mpi_ctx, int n_layers, int batch_size, int max_seq_len,
        int n_kv_heads, int head_dim, const std::vector<int> &attention_devices,
        KVCacheLayoutMode layout_mode)
        : CPURingKVCache(mpi_ctx, n_layers, batch_size, max_seq_len,
                         n_kv_heads, n_kv_heads, 0, head_dim,
                         DeviceId::cpu(), layout_mode)
    {
        // Only apply per-layer device overrides if enough entries are provided.
        if (attention_devices.size() >= static_cast<size_t>(n_layers_))
        {
            // Convert legacy integer indices to typed DeviceId values.
            for (int i = 0; i < n_layers_; ++i)
            {
                layer_devices_[i] = deviceIdFromLegacyIndex(attention_devices[i]);
            }
            // Re-initialize each layer's tensors on its assigned device.
            // This replaces the CPU tensors allocated by the delegated constructor.
            for (int layer = 0; layer < n_layers_; ++layer)
            {
                initialize_layer(layer, layer_devices_[layer]);
            }
        }
    }

    /**
     * Primary constructor — all other constructors eventually delegate here.
     *
     * Initializes all member variables and pre-allocates K/V tensors for every
     * (layer, sequence) pair. This is where the bulk of the memory allocation
     * happens. For a model with 24 layers, batch_size=1, max_seq_len=2048,
     * and kv_dim=128, this allocates 24 * 2 * 2 = 96 tensors.
     */
    template <ActivationPrecision KPrecision, ActivationPrecision VPrecision>
    CPURingKVCache<KPrecision, VPrecision>::CPURingKVCache(
        const IMPIContext &mpi_ctx, int n_layers, int batch_size, int max_seq_len,
        int n_kv_heads, int local_n_kv_heads, int kv_head_start,
        int head_dim, DeviceId device,
        KVCacheLayoutMode layout_mode)
        : n_layers_(n_layers), batch_size_(batch_size), max_seq_len_(max_seq_len), n_kv_heads_(n_kv_heads), local_n_kv_heads_(local_n_kv_heads), kv_head_start_(kv_head_start), head_dim_(head_dim), kv_dim_(local_n_kv_heads * head_dim), is_sharded_(local_n_kv_heads != n_kv_heads), layout_mode_(layout_mode)
    {
        // TensorFactory handles device-aware allocation (CPU pinned, CUDA, etc.).
        tensor_factory_ = std::make_unique<TensorFactory>(mpi_ctx);

        // entries_ is a 2D vector: entries_[layer][seq_idx]
        entries_.resize(n_layers_);
        // All layers initially use the same device.
        layer_devices_.resize(n_layers_, device);

        // Pre-allocate K and V tensors for each (layer, sequence) pair.
        for (int layer = 0; layer < n_layers_; ++layer)
        {
            initialize_layer(layer, layer_devices_[layer]);
        }
    }

    /**
     * Sharded, per-layer device constructor.
     * Same as primary but with per-layer device overrides from attention_devices.
     */
    template <ActivationPrecision KPrecision, ActivationPrecision VPrecision>
    CPURingKVCache<KPrecision, VPrecision>::CPURingKVCache(
        const IMPIContext &mpi_ctx, int n_layers, int batch_size, int max_seq_len,
        int n_kv_heads, int local_n_kv_heads, int kv_head_start,
        int head_dim, const std::vector<int> &attention_devices,
        KVCacheLayoutMode layout_mode)
        : CPURingKVCache(mpi_ctx, n_layers, batch_size, max_seq_len,
                         n_kv_heads, local_n_kv_heads, kv_head_start,
                         head_dim, DeviceId::cpu(), layout_mode)
    {
        // Apply per-layer device overrides if the vector is large enough.
        if (attention_devices.size() >= static_cast<size_t>(n_layers_))
        {
            for (int i = 0; i < n_layers_; ++i)
            {
                layer_devices_[i] = deviceIdFromLegacyIndex(attention_devices[i]);
            }
            // Re-initialize to allocate tensors on the correct per-layer devices.
            for (int layer = 0; layer < n_layers_; ++layer)
            {
                initialize_layer(layer, layer_devices_[layer]);
            }
        }
    }

    // =========================================================================
    // Private Helpers
    // =========================================================================

    /**
     * @brief Allocate a typed K tensor using compile-time dispatch.
     *
     * Delegates to CPUKVCacheTensor<KPrecision>::allocate().
     */
    template <ActivationPrecision KPrecision, ActivationPrecision VPrecision>
    std::shared_ptr<typename CPURingKVCache<KPrecision, VPrecision>::KTensorT> CPURingKVCache<KPrecision, VPrecision>::allocate_k_tensor(
        size_t rows, size_t cols, DeviceId device)
    {
        return detail::CPUKVCacheTensor<KPrecision>::allocate(*tensor_factory_, rows, cols, head_dim_, device);
    }

    /**
     * @brief Allocate a typed V tensor using compile-time dispatch.
     *
     * Delegates to CPUKVCacheTensor<VPrecision>::allocate().
     */
    template <ActivationPrecision KPrecision, ActivationPrecision VPrecision>
    std::shared_ptr<typename CPURingKVCache<KPrecision, VPrecision>::VTensorT> CPURingKVCache<KPrecision, VPrecision>::allocate_v_tensor(
        size_t rows, size_t cols, DeviceId device)
    {
        return detail::CPUKVCacheTensor<VPrecision>::allocate(*tensor_factory_, rows, cols, head_dim_, device);
    }

    /**
     * @brief Initialize K/V tensors for all sequences in a given layer.
     *
     * Tensor shape depends on the layout mode:
     *
     *   POSITION_MAJOR: K, V are [max_seq_len, kv_dim]
     *     where kv_dim = local_n_kv_heads * head_dim.
     *     One row per token position, all heads concatenated.
     *
     *   HEAD_MAJOR: K, V are [local_n_kv_heads * max_seq_len, head_dim]
     *     Logically [n_heads][max_seq_len][head_dim] packed into 2D.
     *     Each head's positions are contiguous for efficient per-head access.
     */
    template <ActivationPrecision KPrecision, ActivationPrecision VPrecision>
    void CPURingKVCache<KPrecision, VPrecision>::initialize_layer(int layer, DeviceId device)
    {
        entries_[layer].resize(batch_size_);
        for (int seq_idx = 0; seq_idx < batch_size_; ++seq_idx)
        {
            auto &entry = entries_[layer][seq_idx];
            if (layout_mode_ == KVCacheLayoutMode::HEAD_MAJOR)
            {
                // HEAD_MAJOR: rows = n_heads * max_seq_len, cols = head_dim
                entry.K = allocate_k_tensor(local_n_kv_heads_ * max_seq_len_, head_dim_, device);
                entry.V = allocate_v_tensor(local_n_kv_heads_ * max_seq_len_, head_dim_, device);
            }
            else
            {
                // POSITION_MAJOR: rows = max_seq_len, cols = kv_dim
                entry.K = allocate_k_tensor(max_seq_len_, kv_dim_, device);
                entry.V = allocate_v_tensor(max_seq_len_, kv_dim_, device);
            }
            // Ring buffer starts empty.
            entry.head = 0;
            entry.size = 0;
        }
    }

    // =========================================================================
    // Token Counting and KV Retrieval
    // =========================================================================

    template <ActivationPrecision KPrecision, ActivationPrecision VPrecision>
    int CPURingKVCache<KPrecision, VPrecision>::get_cached_tokens(int layer, int seq_idx) const
    {
        // Bounds check — return 0 for invalid indices rather than crashing.
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
        {
            return 0;
        }
        return entries_[layer][seq_idx].size;
    }

    /**
     * @brief Mutable KV retrieval.
     *
     * Note: The returned tensors are the full pre-allocated ring buffers.
     * The caller must use `out_kv_len` (or `ring_head()`) to determine
     * which rows contain valid data and where the ring starts.
     */
    template <ActivationPrecision KPrecision, ActivationPrecision VPrecision>
    bool CPURingKVCache<KPrecision, VPrecision>::get_kv(int layer, int seq_idx,
                                                        ITensor **out_k, ITensor **out_v,
                                                        int *out_kv_len)
    {
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
        {
            // On failure, null out all output pointers for safety.
            if (out_k)
                *out_k = nullptr;
            if (out_v)
                *out_v = nullptr;
            if (out_kv_len)
                *out_kv_len = 0;
            return false;
        }

        auto &entry = entries_[layer][seq_idx];
        if (out_k)
            *out_k = entry.K.get();
        if (out_v)
            *out_v = entry.V.get();
        if (out_kv_len)
            *out_kv_len = entry.size;
        return true;
    }

    /** @brief Const overload — same logic, const pointers. */
    template <ActivationPrecision KPrecision, ActivationPrecision VPrecision>
    bool CPURingKVCache<KPrecision, VPrecision>::get_kv(int layer, int seq_idx,
                                                        const ITensor **out_k, const ITensor **out_v,
                                                        int *out_kv_len) const
    {
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
        {
            if (out_k)
                *out_k = nullptr;
            if (out_v)
                *out_v = nullptr;
            if (out_kv_len)
                *out_kv_len = 0;
            return false;
        }

        const auto &entry = entries_[layer][seq_idx];
        if (out_k)
            *out_k = entry.K.get();
        if (out_v)
            *out_v = entry.V.get();
        if (out_kv_len)
            *out_kv_len = entry.size;
        return true;
    }

    // =========================================================================
    // Converted KV Access (get_kv_converted)
    // =========================================================================

    template <ActivationPrecision KPrecision, ActivationPrecision VPrecision>
    bool CPURingKVCache<KPrecision, VPrecision>::get_kv_converted(
        int layer, int seq_idx,
        ActivationPrecision target,
        ITensor **out_k, ITensor **out_v,
        int *out_kv_len,
        const KVReadParams *rope)
    {
        // Only FP32 target is supported currently
        if (target != ActivationPrecision::FP32)
        {
            LOG_ERROR("[CPURingKVCache] get_kv_converted: only FP32 target supported, got "
                      << static_cast<int>(target));
            return false;
        }

        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
        {
            if (out_k)
                *out_k = nullptr;
            if (out_v)
                *out_v = nullptr;
            if (out_kv_len)
                *out_kv_len = 0;
            return false;
        }

        // FP32 cache: passthrough when head==0, linearize into shadow otherwise
        if constexpr (KPrecision == ActivationPrecision::FP32 && VPrecision == ActivationPrecision::FP32)
        {
            auto &entry = entries_[layer][seq_idx];

            if (entry.head == 0)
            {
                // No wrap: passthrough raw buffer (fast path)
                if (out_k)
                    *out_k = entry.K.get();
                if (out_v)
                    *out_v = entry.V.get();
                if (out_kv_len)
                    *out_kv_len = entry.size;

                if (rope && rope->rope_theta > 0.0f && rope->n_kv_heads > 0 && rope->head_dim > 0)
                {
                    auto &shadow = ensureFP32Shadow(layer, seq_idx);
                    if (shadow.converted_rows < entry.size)
                    {
                        const int new_rows = entry.size - shadow.converted_rows;
                        apply_rope_to_k_fp32(
                            entry.K->mutable_data() + static_cast<size_t>(shadow.converted_rows) * kv_dim_,
                            new_rows, rope->head_dim, rope->n_kv_heads,
                            rope->rope_theta, rope->position_start + shadow.converted_rows,
                            rope->rope_dim);
                        shadow.converted_rows = entry.size;
                    }
                }
                return true;
            }

            // Ring has wrapped (head != 0): linearize into shadow buffer
            auto &shadow = ensureFP32Shadow(layer, seq_idx);

            // Detect head movement → full re-linearization needed
            if (entry.head != shadow.last_head)
            {
                shadow.converted_rows = 0;
                shadow.last_head = entry.head;
            }

            if (shadow.converted_rows < entry.size)
            {
                float *k_fp32 = shadow.K->mutable_data();
                float *v_fp32 = shadow.V->mutable_data();
                const float *src_k = entry.K->data();
                const float *src_v = entry.V->data();

                for (int r = shadow.converted_rows; r < entry.size; ++r)
                {
                    const int phys = (entry.head + r) % max_seq_len_;
                    const size_t src_off = static_cast<size_t>(phys) * kv_dim_;
                    const size_t dst_off = static_cast<size_t>(r) * kv_dim_;
                    std::memcpy(k_fp32 + dst_off, src_k + src_off, kv_dim_ * sizeof(float));
                    std::memcpy(v_fp32 + dst_off, src_v + src_off, kv_dim_ * sizeof(float));
                }

                if (rope && rope->rope_theta > 0.0f && rope->n_kv_heads > 0 && rope->head_dim > 0)
                {
                    const size_t offset = static_cast<size_t>(shadow.converted_rows) * kv_dim_;
                    apply_rope_to_k_fp32(
                        k_fp32 + offset,
                        entry.size - shadow.converted_rows, rope->head_dim, rope->n_kv_heads,
                        rope->rope_theta, rope->position_start + shadow.converted_rows,
                        rope->rope_dim);
                }

                shadow.converted_rows = entry.size;
            }

            if (out_k)
                *out_k = shadow.K.get();
            if (out_v)
                *out_v = shadow.V.get();
            if (out_kv_len)
                *out_kv_len = entry.size;
            return true;
        }
        else
        {
            auto &entry = entries_[layer][seq_idx];
            const int kv_len = entry.size;

            auto &shadow = ensureFP32Shadow(layer, seq_idx);

            // Detect cache clear (kv_len decreased since last call)
            if (kv_len < shadow.converted_rows)
                shadow.converted_rows = 0;

            // Detect ring head movement (wrap) → full reconversion needed
            if (entry.head != shadow.last_head)
            {
                shadow.converted_rows = 0;
                shadow.last_head = entry.head;
            }

            // Convert new rows incrementally
            if (shadow.converted_rows < kv_len)
            {
                convertNewRows(layer, seq_idx, shadow, entry, rope);
            }

            if (out_k)
                *out_k = shadow.K.get();
            if (out_v)
                *out_v = shadow.V.get();
            if (out_kv_len)
                *out_kv_len = kv_len;
            return true;
        }
    }

    // -------------------------------------------------------------------------
    // Shadow buffer lazy allocation
    // -------------------------------------------------------------------------

    template <ActivationPrecision KPrecision, ActivationPrecision VPrecision>
    auto CPURingKVCache<KPrecision, VPrecision>::ensureFP32Shadow(int layer, int seq_idx) const
        -> FP32Shadow &
    {
        // Lazy init the 2D vector if empty
        if (fp32_shadows_.empty())
        {
            fp32_shadows_.resize(n_layers_);
            for (auto &layer_vec : fp32_shadows_)
                layer_vec.resize(batch_size_);
        }

        auto &shadow = fp32_shadows_[layer][seq_idx];
        if (!shadow.K)
        {
            const size_t total = static_cast<size_t>(max_seq_len_) * static_cast<size_t>(kv_dim_);
            shadow.K = std::make_unique<FP32Tensor>(std::vector<size_t>{total});
            shadow.V = std::make_unique<FP32Tensor>(std::vector<size_t>{total});
            shadow.converted_rows = 0;
        }
        return shadow;
    }

    template <ActivationPrecision KPrecision, ActivationPrecision VPrecision>
    void CPURingKVCache<KPrecision, VPrecision>::invalidateFP32Shadow(int layer, int seq_idx) const
    {
        if (fp32_shadows_.empty() || layer < 0 || layer >= static_cast<int>(fp32_shadows_.size()) ||
            seq_idx < 0 || seq_idx >= static_cast<int>(fp32_shadows_[layer].size()))
        {
            return;
        }

        auto &shadow = fp32_shadows_[layer][seq_idx];
        shadow.converted_rows = 0;
        shadow.last_head = -1;
    }

    // -------------------------------------------------------------------------
    // Per-precision incremental conversion dispatch
    // -------------------------------------------------------------------------

    template <ActivationPrecision KPrecision, ActivationPrecision VPrecision>
    void CPURingKVCache<KPrecision, VPrecision>::convertNewRows(
        int layer, int seq_idx,
        FP32Shadow &shadow, const EntryT &entry,
        const KVReadParams *rope) const
    {
        const int from = shadow.converted_rows;
        const int to = entry.size;
        float *k_fp32 = shadow.K->mutable_data();
        float *v_fp32 = shadow.V->mutable_data();

        // FP16 cache
        if constexpr (KPrecision == ActivationPrecision::FP16 && VPrecision == ActivationPrecision::FP16)
        {
            for (int r = from; r < to; ++r)
            {
                const int phys = (entry.head + r) % max_seq_len_;
                const size_t src_off = static_cast<size_t>(phys) * kv_dim_;
                const size_t dst_off = static_cast<size_t>(r) * kv_dim_;
                simd::convert_fp16_to_fp32(entry.K->typed_data() + src_off, k_fp32 + dst_off, kv_dim_);
                simd::convert_fp16_to_fp32(entry.V->typed_data() + src_off, v_fp32 + dst_off, kv_dim_);
            }

            // Apply RoPE to newly converted K rows if requested
            if (rope && rope->rope_theta > 0.0f && rope->n_kv_heads > 0)
            {
                const size_t offset = static_cast<size_t>(from) * kv_dim_;
                apply_rope_to_k_fp32(
                    k_fp32 + offset,
                    to - from, rope->head_dim, rope->n_kv_heads,
                    rope->rope_theta, rope->position_start + from,
                    rope->rope_dim);
            }
        }
        // BF16 cache
        else if constexpr (KPrecision == ActivationPrecision::BF16 && VPrecision == ActivationPrecision::BF16)
        {
            for (int r = from; r < to; ++r)
            {
                const int phys = (entry.head + r) % max_seq_len_;
                const size_t src_off = static_cast<size_t>(phys) * kv_dim_;
                const size_t dst_off = static_cast<size_t>(r) * kv_dim_;
                simd::convert_bf16_to_fp32(entry.K->typed_data() + src_off, k_fp32 + dst_off, kv_dim_);
                simd::convert_bf16_to_fp32(entry.V->typed_data() + src_off, v_fp32 + dst_off, kv_dim_);
            }

            if (rope && rope->rope_theta > 0.0f && rope->n_kv_heads > 0)
            {
                const size_t offset = static_cast<size_t>(from) * kv_dim_;
                apply_rope_to_k_fp32(
                    k_fp32 + offset,
                    to - from, rope->head_dim, rope->n_kv_heads,
                    rope->rope_theta, rope->position_start + from,
                    rope->rope_dim);
            }
        }
        // Q8_1 cache
        else if constexpr (KPrecision == ActivationPrecision::Q8_1 && VPrecision == ActivationPrecision::Q8_1)
        {
            const size_t blocks_per_row = entry.K->blocks_per_row();
            for (int r = from; r < to; ++r)
            {
                const int phys = (entry.head + r) % max_seq_len_;
                const Q8_1Block *k_row = entry.K->typed_data() + phys * blocks_per_row;
                const Q8_1Block *v_row = entry.V->typed_data() + phys * blocks_per_row;
                const size_t dst_off = static_cast<size_t>(r) * kv_dim_;
                simd::dequantize_q8_1_to_fp32(k_row, k_fp32 + dst_off, kv_dim_);
                simd::dequantize_q8_1_to_fp32(v_row, v_fp32 + dst_off, kv_dim_);
            }

            if (rope && rope->rope_theta > 0.0f && rope->n_kv_heads > 0)
            {
                const size_t offset = static_cast<size_t>(from) * kv_dim_;
                apply_rope_to_k_fp32(
                    k_fp32 + offset,
                    to - from, rope->head_dim, rope->n_kv_heads,
                    rope->rope_theta, rope->position_start + from,
                    rope->rope_dim);
            }
        }
        // TQ4 symmetric cache (both K and V are TQ4)
        // NOTE: TQ paths pass from/to as physical row indices to turboquant helpers.
        // After ring wrap, these need physical→logical mapping too (same as FP16/Q8_1 above).
        // Currently safe because TQ caches don't use sequences exceeding max_seq_len in practice,
        // but this should be fixed when TQ + ring wrap is needed.
        else if constexpr (KPrecision == ActivationPrecision::TQ4 && VPrecision == ActivationPrecision::TQ4)
        {
            if (!rope || !rope->turboquant_ctx)
            {
                LOG_ERROR("[CPURingKVCache] TQ4 get_kv_converted requires turboquant_ctx in KVReadParams");
                shadow.converted_rows = to;
                return;
            }
            const auto &layer_ctx = rope->turboquant_ctx->for_layer(layer);

            auto *K_tq4 = entry.K.get();
            auto *V_tq4 = entry.V.get();
            K_tq4->set_turboquant_context(&layer_ctx);
            V_tq4->set_turboquant_context(&layer_ctx);

            if (rope->rope_theta > 0.0f && rope->n_kv_heads > 0)
            {
                turboquant_dequantize_kv_rows_with_rope(
                    K_tq4->typed_data(), V_tq4->typed_data(),
                    layer_ctx, k_fp32, v_fp32,
                    from, to,
                    rope->head_dim, rope->n_kv_heads,
                    K_tq4->blocks_per_row() * K_tq4->block_bytes(),
                    V_tq4->blocks_per_row() * V_tq4->block_bytes(),
                    K_tq4->block_bytes(), V_tq4->block_bytes(),
                    rope->rope_theta, rope->position_start + from);
            }
            else
            {
                turboquant_dequantize_kv_rows(
                    K_tq4->typed_data(), V_tq4->typed_data(),
                    layer_ctx, k_fp32, v_fp32,
                    from, to,
                    head_dim_, n_kv_heads_,
                    K_tq4->blocks_per_row() * K_tq4->block_bytes(),
                    V_tq4->blocks_per_row() * V_tq4->block_bytes(),
                    K_tq4->block_bytes(), V_tq4->block_bytes());
            }
        }
        // Split TQ cache: K is TQ8, V is TQ4
        // NOTE: Same ring wrap limitation as TQ4 above.
        else if constexpr (KPrecision == ActivationPrecision::TQ8 && VPrecision == ActivationPrecision::TQ4)
        {
            if (!rope || !rope->turboquant_ctx)
            {
                LOG_ERROR("[CPURingKVCache] Split TQ get_kv_converted requires turboquant_ctx in KVReadParams");
                shadow.converted_rows = to;
                return;
            }
            const auto &layer_ctx = rope->turboquant_ctx->for_layer(layer);

            auto *K_tq8 = entry.K.get();
            auto *V_tq4 = entry.V.get();
            K_tq8->set_turboquant_context(&layer_ctx);
            V_tq4->set_turboquant_context(&layer_ctx);

            if (rope->rope_theta > 0.0f && rope->n_kv_heads > 0)
            {
                turboquant_dequantize_split_kv_rows_with_rope(
                    K_tq8->typed_data(), V_tq4->typed_data(),
                    layer_ctx, k_fp32, v_fp32,
                    from, to,
                    rope->head_dim, rope->n_kv_heads,
                    K_tq8->blocks_per_row() * K_tq8->block_bytes(),
                    V_tq4->blocks_per_row() * V_tq4->block_bytes(),
                    K_tq8->block_bytes(), V_tq4->block_bytes(),
                    rope->rope_theta, rope->position_start + from);
            }
            else
            {
                turboquant_dequantize_split_kv_rows(
                    K_tq8->typed_data(), V_tq4->typed_data(),
                    layer_ctx, k_fp32, v_fp32,
                    from, to,
                    head_dim_, n_kv_heads_,
                    K_tq8->blocks_per_row() * K_tq8->block_bytes(),
                    V_tq4->blocks_per_row() * V_tq4->block_bytes(),
                    K_tq8->block_bytes(), V_tq4->block_bytes());
            }
        }
        // Q16_1 cache (both K and V are Q16_1)
        // Q16_1 always uses HEAD_MAJOR layout: [n_kv_heads][position][head_dim]
        // The Q16_1Tensor has shape (local_n_kv_heads * max_seq_len, head_dim).
        // Each "raw row" is one head for one position (head_dim elements).
        // The FP32 shadow is POSITION_MAJOR: row r = [head0..head1..] kv_dim wide.
        else if constexpr (KPrecision == ActivationPrecision::Q16_1 && VPrecision == ActivationPrecision::Q16_1)
        {
            // Block geometry for K
            const size_t k_block_elems = q16_block_size_elements(entry.K->q16_block_size());
            const size_t k_block_bytes = q16_block_size_bytes(entry.K->q16_block_size());
            const size_t k_bpr = entry.K->blocks_per_row(); // blocks per head_dim
            const size_t k_head_row_bytes = k_bpr * k_block_bytes;

            // Block geometry for V
            const size_t v_block_elems = q16_block_size_elements(entry.V->q16_block_size());
            const size_t v_block_bytes = q16_block_size_bytes(entry.V->q16_block_size());
            const size_t v_bpr = entry.V->blocks_per_row();
            const size_t v_head_row_bytes = v_bpr * v_block_bytes;

            const uint8_t *k_raw = reinterpret_cast<const uint8_t *>(entry.K->raw_data());
            const uint8_t *v_raw = reinterpret_cast<const uint8_t *>(entry.V->raw_data());

            constexpr size_t QS_OFFSET = sizeof(float) + sizeof(int32_t); // int16_t qs[]

            for (int r = from; r < to; ++r)
            {
                const int phys = (entry.head + r) % max_seq_len_;
                const size_t dst_row_off = static_cast<size_t>(r) * kv_dim_;

                for (int h = 0; h < local_n_kv_heads_; ++h)
                {
                    // HEAD_MAJOR raw row = h * max_seq_len + phys
                    const size_t raw_row = static_cast<size_t>(h) * max_seq_len_ + phys;
                    const size_t dst_head_off = dst_row_off + static_cast<size_t>(h) * head_dim_;

                    // Dequantize K: head h, position phys
                    const uint8_t *k_head = k_raw + raw_row * k_head_row_bytes;
                    for (size_t b = 0; b < k_bpr; ++b)
                    {
                        const uint8_t *blk = k_head + b * k_block_bytes;
                        float d;
                        std::memcpy(&d, blk, sizeof(float));
                        const int16_t *qs = reinterpret_cast<const int16_t *>(blk + QS_OFFSET);
                        const size_t base = b * k_block_elems;
                        const size_t count = std::min(k_block_elems, static_cast<size_t>(head_dim_) - base);
                        for (size_t i = 0; i < count; ++i)
                            k_fp32[dst_head_off + base + i] = d * static_cast<float>(qs[i]);
                    }

                    // Dequantize V: head h, position phys
                    const uint8_t *v_head = v_raw + raw_row * v_head_row_bytes;
                    for (size_t b = 0; b < v_bpr; ++b)
                    {
                        const uint8_t *blk = v_head + b * v_block_bytes;
                        float d;
                        std::memcpy(&d, blk, sizeof(float));
                        const int16_t *qs = reinterpret_cast<const int16_t *>(blk + QS_OFFSET);
                        const size_t base = b * v_block_elems;
                        const size_t count = std::min(v_block_elems, static_cast<size_t>(head_dim_) - base);
                        for (size_t i = 0; i < count; ++i)
                            v_fp32[dst_head_off + base + i] = d * static_cast<float>(qs[i]);
                    }
                }
            }

            if (rope && rope->rope_theta > 0.0f && rope->n_kv_heads > 0)
            {
                const size_t offset = static_cast<size_t>(from) * kv_dim_;
                apply_rope_to_k_fp32(
                    k_fp32 + offset,
                    to - from, rope->head_dim, rope->n_kv_heads,
                    rope->rope_theta, rope->position_start + from,
                    rope->rope_dim);
            }
        }
        // Symmetric TQ8 cache (both K and V are TQ8) - not yet used, stub
        else if constexpr (KPrecision == ActivationPrecision::TQ8 && VPrecision == ActivationPrecision::TQ8)
        {
            LOG_ERROR("[CPURingKVCache] Symmetric TQ8 get_kv_converted not yet implemented");
            shadow.converted_rows = to;
            return;
        }
        else
        {
            LOG_ERROR("[CPURingKVCache] get_kv_converted not implemented for precision K="
                      << static_cast<int>(KPrecision) << " V=" << static_cast<int>(VPrecision));
            shadow.converted_rows = to;
            return;
        }

        shadow.converted_rows = to;
    }

    // =========================================================================
    // Individual K / V Accessors
    // =========================================================================
    // These are thin wrappers that return the raw K or V tensor pointer.
    // Prefer get_kv() which returns both in one call.

    template <ActivationPrecision KPrecision, ActivationPrecision VPrecision>
    ITensor *CPURingKVCache<KPrecision, VPrecision>::get_k(int layer, int seq_idx)
    {
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
        {
            return nullptr;
        }
        return entries_[layer][seq_idx].K.get();
    }

    template <ActivationPrecision KPrecision, ActivationPrecision VPrecision>
    const ITensor *CPURingKVCache<KPrecision, VPrecision>::get_k(int layer, int seq_idx) const
    {
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
        {
            return nullptr;
        }
        return entries_[layer][seq_idx].K.get();
    }

    template <ActivationPrecision KPrecision, ActivationPrecision VPrecision>
    ITensor *CPURingKVCache<KPrecision, VPrecision>::get_v(int layer, int seq_idx)
    {
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
        {
            return nullptr;
        }
        return entries_[layer][seq_idx].V.get();
    }

    template <ActivationPrecision KPrecision, ActivationPrecision VPrecision>
    const ITensor *CPURingKVCache<KPrecision, VPrecision>::get_v(int layer, int seq_idx) const
    {
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
        {
            return nullptr;
        }
        return entries_[layer][seq_idx].V.get();
    }

    // =========================================================================
    // Cache Management (clear operations)
    // =========================================================================
    // These reset the ring buffer bookkeeping (head and size) WITHOUT freeing
    // the underlying tensor memory. This allows the cache to be reused across
    // inference sessions without reallocation.

    template <ActivationPrecision KPrecision, ActivationPrecision VPrecision>
    void CPURingKVCache<KPrecision, VPrecision>::clear()
    {
        for (int layer = 0; layer < n_layers_; ++layer)
        {
            for (int seq_idx = 0; seq_idx < batch_size_; ++seq_idx)
            {
                entries_[layer][seq_idx].head = 0;
                entries_[layer][seq_idx].size = 0;
            }
        }
        for (auto &layer_shadows : fp32_shadows_)
        {
            for (auto &shadow : layer_shadows)
            {
                shadow.converted_rows = 0;
                shadow.last_head = -1;
            }
        }
        wrap_warned_ = false;
    }

    template <ActivationPrecision KPrecision, ActivationPrecision VPrecision>
    void CPURingKVCache<KPrecision, VPrecision>::clear_sequence(int layer, int seq_idx)
    {
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
        {
            return;
        }
        entries_[layer][seq_idx].head = 0;
        entries_[layer][seq_idx].size = 0;
        invalidateFP32Shadow(layer, seq_idx);
    }

    template <ActivationPrecision KPrecision, ActivationPrecision VPrecision>
    void CPURingKVCache<KPrecision, VPrecision>::clear_layer(int layer)
    {
        if (layer < 0 || layer >= n_layers_)
        {
            return;
        }
        for (int seq_idx = 0; seq_idx < batch_size_; ++seq_idx)
        {
            entries_[layer][seq_idx].head = 0;
            entries_[layer][seq_idx].size = 0;
            invalidateFP32Shadow(layer, seq_idx);
        }
    }

    // =========================================================================
    // Append Operations
    // =========================================================================

    /**
     * Convenience overload: infers num_tokens from the K tensor's row count.
     */
    template <ActivationPrecision KPrecision, ActivationPrecision VPrecision>
    bool CPURingKVCache<KPrecision, VPrecision>::append_kv(int layer, int seq_idx, const TensorBase *new_k, const TensorBase *new_v)
    {
        if (!new_k || !new_v)
        {
            return false;
        }
        return append_kv(layer, seq_idx, new_k, new_v, static_cast<int>(new_k->shape()[0]));
    }

    /**
     * Main append entry point. Validates bounds and types, then delegates
     * to append_kv_impl() for the actual data copy.
     *
     * The dynamic_cast is needed because the public API takes TensorBase*,
     * but the internal append logic needs the concrete typed tensor to access
     * typed_data() for precision-specific memcpy operations.
     */
    template <ActivationPrecision KPrecision, ActivationPrecision VPrecision>
    bool CPURingKVCache<KPrecision, VPrecision>::append_kv(int layer, int seq_idx, const TensorBase *new_k, const TensorBase *new_v, int num_tokens)
    {
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_ || !new_k || !new_v || num_tokens <= 0)
        {
            return false;
        }

        // Downcast to the concrete tensor types for this cache's K/V precisions.
        const KTensorT *typed_k = dynamic_cast<const KTensorT *>(new_k);
        const VTensorT *typed_v = dynamic_cast<const VTensorT *>(new_v);
        if (!typed_k || !typed_v)
        {
            return false;
        }

        return append_kv_impl(layer, seq_idx, typed_k, typed_v, num_tokens);
    }

    /**
     * @brief Core append implementation — handles both layout modes and all precisions.
     *
     * This is the heart of the ring buffer append logic. The algorithm:
     *
     * 1. Determine how many tokens to actually write (bounded by source size
     *    and ring capacity).
     * 2. If the source has more tokens than max_seq_len, skip the oldest ones
     *    from the source (only the most recent max_seq_len tokens matter).
     * 3. For each token, compute the destination position in the ring:
     *    - If the ring isn't full yet: write at `(head + size) % max_seq_len`
     *      and increment size.
     *    - If full: overwrite at `head` and advance head forward.
     * 4. Copy the token data using a precision-specific and layout-specific
     *    memcpy lambda.
     *
     * The two layout modes handle data differently:
     *
     *   POSITION_MAJOR: Each token is one contiguous row → single memcpy
     *   per token for K and V.
     *
     *   HEAD_MAJOR: Token data must be scattered across per-head blocks →
     *   one memcpy per head per token.
     */
    template <ActivationPrecision KPrecision, ActivationPrecision VPrecision>
    bool CPURingKVCache<KPrecision, VPrecision>::append_kv_impl(int layer, int seq_idx, const KTensorT *new_k, const VTensorT *new_v, int num_tokens)
    {
        auto &entry = entries_[layer][seq_idx];

        KTensorT *dst_k = entry.K.get();
        VTensorT *dst_v = entry.V.get();
        if (!dst_k || !dst_v)
        {
            return false;
        }

        const size_t src_rows_k = new_k->rows();
        const size_t src_rows_v = new_v->rows();

        // Take the minimum of requested tokens and what's actually available.
        const int rows_to_take = std::min({num_tokens, static_cast<int>(src_rows_k), static_cast<int>(src_rows_v)});
        if (rows_to_take <= 0)
        {
            return true; // Nothing to append — not an error.
        }

        // If the source has more tokens than our ring capacity, skip the oldest.
        // For example, if max_seq_len=128 and we're asked to append 200 tokens,
        // we skip the first 72 and only write the last 128.
        const int src_start = std::max(0, rows_to_take - max_seq_len_);
        const int tokens_to_write = rows_to_take - src_start;

        // -----------------------------------------------------------------
        // Lambda: copy one token in POSITION_MAJOR layout.
        //
        // POSITION_MAJOR stores each token as a single contiguous row of
        // `kv_dim` elements (all heads concatenated). So copying a token
        // is a single memcpy of row_bytes (computed via the CPUKVCacheTensor
        // trait, which handles element, block, and turbo-quant strides).
        // K and V may have different byte sizes when using asymmetric precision.
        // -----------------------------------------------------------------
        using KTrait = detail::CPUKVCacheTensor<KPrecision>;
        using VTrait = detail::CPUKVCacheTensor<VPrecision>;
        const size_t k_rb = KTrait::row_bytes(dst_k, kv_dim_, head_dim_);
        const size_t v_rb = VTrait::row_bytes(dst_v, kv_dim_, head_dim_);

        auto copy_position_major_token = [&](int src_row, int dst_row)
        {
            const auto *sk = reinterpret_cast<const uint8_t *>(new_k->raw_data());
            const auto *sv = reinterpret_cast<const uint8_t *>(new_v->raw_data());
            auto *dk = reinterpret_cast<uint8_t *>(dst_k->raw_mutable_data());
            auto *dv = reinterpret_cast<uint8_t *>(dst_v->raw_mutable_data());
            if (!sk || !sv || !dk || !dv)
            {
                return false;
            }

            std::memcpy(dk + static_cast<size_t>(dst_row) * k_rb, sk + static_cast<size_t>(src_row) * k_rb, k_rb);
            std::memcpy(dv + static_cast<size_t>(dst_row) * v_rb, sv + static_cast<size_t>(src_row) * v_rb, v_rb);
            return true;
        };

        // -----------------------------------------------------------------
        // Lambda: copy one token in HEAD_MAJOR layout.
        //
        // HEAD_MAJOR stores data as [head][position][head_dim]. The source
        // tensor (from the projection stage) is in POSITION_MAJOR format:
        // [position][n_heads * head_dim]. So for each source token, we need
        // to scatter the per-head slices into their respective head blocks.
        //
        // Source addressing (bytes):  src + src_row * rb + h * hb
        // Dest addressing (bytes):    dst + (h * max_seq_len + dst_pos) * hb
        // -----------------------------------------------------------------
        const size_t k_hb = KTrait::head_bytes(dst_k, kv_dim_, head_dim_);
        const size_t v_hb = VTrait::head_bytes(dst_v, kv_dim_, head_dim_);

        auto copy_head_major_token = [&](int src_row, int dst_pos)
        {
            if (local_n_kv_heads_ <= 0 || head_dim_ <= 0)
            {
                return false;
            }

            const auto *sk = reinterpret_cast<const uint8_t *>(new_k->raw_data());
            const auto *sv = reinterpret_cast<const uint8_t *>(new_v->raw_data());
            auto *dk = reinterpret_cast<uint8_t *>(dst_k->raw_mutable_data());
            auto *dv = reinterpret_cast<uint8_t *>(dst_v->raw_mutable_data());
            if (!sk || !sv || !dk || !dv)
            {
                return false;
            }

            const auto *src_k_row = sk + static_cast<size_t>(src_row) * k_rb;
            const auto *src_v_row = sv + static_cast<size_t>(src_row) * v_rb;
            for (int h = 0; h < local_n_kv_heads_; ++h)
            {
                const auto *src_k_head = src_k_row + static_cast<size_t>(h) * k_hb;
                const auto *src_v_head = src_v_row + static_cast<size_t>(h) * v_hb;
                auto *dst_k_head = dk + (static_cast<size_t>(h) * max_seq_len_ + static_cast<size_t>(dst_pos)) * k_hb;
                auto *dst_v_head = dv + (static_cast<size_t>(h) * max_seq_len_ + static_cast<size_t>(dst_pos)) * v_hb;
                std::memcpy(dst_k_head, src_k_head, k_hb);
                std::memcpy(dst_v_head, src_v_head, v_hb);
            }
            return true;
        };

        // -----------------------------------------------------------------
        // Main token-write loop: iterate over each token to append.
        // -----------------------------------------------------------------
        for (int i = 0; i < tokens_to_write; ++i)
        {
            const int src_row = src_start + i;
            int dst_pos = 0;

            if (entry.size < max_seq_len_)
            {
                // Ring buffer is not yet full: append at the next free slot.
                // The next free slot is (head + size) wrapped around.
                dst_pos = (entry.head + entry.size) % max_seq_len_;
                ++entry.size;
            }
            else
            {
                // Ring buffer is full: overwrite the oldest token at `head`,
                // then advance head to the next oldest.
                if (!wrap_warned_)
                {
                    LOG_WARN("Context window full (" << max_seq_len_
                                                     << " tokens). Sliding window is now overwriting oldest tokens. "
                                                     << "Use -c <size> to increase context length.");
                    wrap_warned_ = true;
                }
                dst_pos = entry.head;
                entry.head = (entry.head + 1) % max_seq_len_;
            }

            const bool ok = (layout_mode_ == KVCacheLayoutMode::HEAD_MAJOR)
                                ? copy_head_major_token(src_row, dst_pos)
                                : copy_position_major_token(src_row, dst_pos);
            if (!ok)
            {
                return false;
            }
        }

        return true;
    }

    /**
     * @brief Simplified single-tensor append using raw byte copies.
     *
     * This is an alternative append path that works purely at the byte level,
     * computing row byte sizes from the tensor's total size_bytes / rows.
     * It handles only POSITION_MAJOR-style row-wise copies.
     *
     * Used as a fallback for formats where typed access isn't needed or
     * for simpler single-tensor operations.
     */
    template <ActivationPrecision KPrecision, ActivationPrecision VPrecision>
    bool CPURingKVCache<KPrecision, VPrecision>::append_one_tensor(TensorBase *dst, const TensorBase *src, EntryT &entry, int num_tokens)
    {
        if (!dst || !src || max_seq_len_ <= 0)
        {
            return false;
        }

        const size_t src_rows = src->rows();
        const size_t dst_rows = dst->rows();
        if (dst_rows < static_cast<size_t>(max_seq_len_) || src_rows == 0)
        {
            return false;
        }

        const int rows_to_take = std::min(num_tokens, static_cast<int>(src_rows));
        if (rows_to_take <= 0)
        {
            return true;
        }

        const int src_start = std::max(0, rows_to_take - max_seq_len_);
        const int tokens_to_write = rows_to_take - src_start;

        const size_t src_row_bytes = src->size_bytes() / std::max<size_t>(1, src->rows());
        const size_t dst_row_bytes = dst->size_bytes() / std::max<size_t>(1, dst->rows());
        const size_t row_bytes = std::min(src_row_bytes, dst_row_bytes);

        const uint8_t *src_bytes = reinterpret_cast<const uint8_t *>(src->raw_data());
        uint8_t *dst_bytes = reinterpret_cast<uint8_t *>(dst->raw_mutable_data());
        if (!src_bytes || !dst_bytes || row_bytes == 0)
        {
            return false;
        }

        for (int i = 0; i < tokens_to_write; ++i)
        {
            const int src_row = src_start + i;
            int dst_row = 0;
            if (entry.size < max_seq_len_)
            {
                dst_row = (entry.head + entry.size) % max_seq_len_;
                ++entry.size;
            }
            else
            {
                dst_row = entry.head;
                entry.head = (entry.head + 1) % max_seq_len_;
            }

            std::memcpy(dst_bytes + static_cast<size_t>(dst_row) * dst_row_bytes,
                        src_bytes + static_cast<size_t>(src_row) * src_row_bytes,
                        row_bytes);
        }

        return true;
    }

    // =========================================================================
    // Batched Gather
    // =========================================================================

    /**
     * @brief Gather and linearize ring buffer contents into contiguous output tensors.
     *
     * This is the most complex operation in the KV cache. During batched inference,
     * the attention kernel needs all sequences' K/V data packed into a flat tensor.
     * This method:
     *
     * 1. Determines the maximum kv_len across all sequences.
     * 2. Validates that the output tensors are large enough.
     * 3. For each sequence, "unrolls" the ring buffer so tokens appear in
     *    logical order (oldest → newest), copying into the output tensor at
     *    row offset `seq_idx * max_kv_len + logical_position`.
     *
     * For HEAD_MAJOR source caches, the gather also transposes the data back
     * to POSITION_MAJOR format in the output (the attention kernel expects it).
     *
     * @return max_kv_len (the number of padded rows per sequence), or -1 on error.
     */
    template <ActivationPrecision KPrecision, ActivationPrecision VPrecision>
    int CPURingKVCache<KPrecision, VPrecision>::gather_kv_batched(
        int layer,
        int num_sequences,
        TensorBase *out_k,
        TensorBase *out_v,
        std::vector<int> &out_kv_lens)
    {
        if (layer < 0 || layer >= n_layers_)
        {
            return -1;
        }
        if (num_sequences <= 0 || num_sequences > batch_size_)
        {
            return -1;
        }

        KTensorT *typed_k = dynamic_cast<KTensorT *>(out_k);
        VTensorT *typed_v = dynamic_cast<VTensorT *>(out_v);
        if (!typed_k || !typed_v)
        {
            return -1;
        }

        out_kv_lens.resize(num_sequences);
        int max_kv_len = 0;

        // First pass: collect per-sequence token counts and find the max.
        // The max_kv_len determines the stride in the output tensor.
        for (int seq_idx = 0; seq_idx < num_sequences; ++seq_idx)
        {
            const int kv_len = entries_[layer][seq_idx].size;
            out_kv_lens[seq_idx] = kv_len;
            max_kv_len = std::max(max_kv_len, kv_len);
        }

        if (max_kv_len == 0)
        {
            return 0;
        }

        // Validate output tensor dimensions.
        // Expected shape: [num_sequences * max_kv_len, kv_dim]
        const size_t expected_rows = static_cast<size_t>(num_sequences) * static_cast<size_t>(max_kv_len);
        const size_t expected_cols = static_cast<size_t>(kv_dim_);
        if (typed_k->shape().size() < 2 || typed_v->shape().size() < 2)
        {
            return -1;
        }
        if (typed_k->shape()[0] < expected_rows || typed_k->shape()[1] != expected_cols)
        {
            return -1;
        }
        if (typed_v->shape()[0] < expected_rows || typed_v->shape()[1] != expected_cols)
        {
            return -1;
        }

        // -----------------------------------------------------------------
        // Lambda: gather one tensor (K or V) for one sequence.
        //
        // Parameterized by row/head byte sizes so it works for both K and V
        // even when they have different precisions (asymmetric cache).
        // -----------------------------------------------------------------
        auto gather_tensor = [&](const TensorBase *src_tensor, TensorBase *dst_tensor, size_t g_rb, size_t g_hb, const EntryT &entry, int seq_idx, int kv_len) -> bool
        {
            if (!src_tensor || !dst_tensor || kv_len <= 0)
            {
                return true; // Empty sequence — nothing to gather.
            }

            const int head = entry.head;

            const auto *src = reinterpret_cast<const uint8_t *>(src_tensor->raw_data());
            auto *dst = reinterpret_cast<uint8_t *>(dst_tensor->raw_mutable_data());
            if (!src || !dst)
            {
                return false;
            }

            if (layout_mode_ == KVCacheLayoutMode::POSITION_MAJOR)
            {
                // POSITION_MAJOR: each ring slot is one contiguous row of g_rb bytes.
                for (int logical = 0; logical < kv_len; ++logical)
                {
                    const int phys = (head + logical) % max_seq_len_;
                    const size_t dst_row = static_cast<size_t>(seq_idx) * static_cast<size_t>(max_kv_len) + static_cast<size_t>(logical);
                    std::memcpy(dst + dst_row * g_rb, src + static_cast<size_t>(phys) * g_rb, g_rb);
                }
                return true;
            }

            // =============================================================
            // HEAD_MAJOR gather: transpose [head][pos][head_dim] →
            //   [pos][n_heads * head_dim] in the output.
            // =============================================================
            if (g_rb < static_cast<size_t>(local_n_kv_heads_) * g_hb)
            {
                return false; // Sanity: row must fit all heads.
            }

            for (int logical = 0; logical < kv_len; ++logical)
            {
                const int phys = (head + logical) % max_seq_len_;
                const size_t dst_row = static_cast<size_t>(seq_idx) * static_cast<size_t>(max_kv_len) + static_cast<size_t>(logical);
                auto *dst_row_ptr = dst + dst_row * g_rb;
                for (int h = 0; h < local_n_kv_heads_; ++h)
                {
                    const auto *src_head = src + (static_cast<size_t>(h) * max_seq_len_ + static_cast<size_t>(phys)) * g_hb;
                    std::memcpy(dst_row_ptr + static_cast<size_t>(h) * g_hb, src_head, g_hb);
                }
            }
            return true;
        };

        // Compute per-precision byte strides for K and V.
        using KGatherTrait = detail::CPUKVCacheTensor<KPrecision>;
        using VGatherTrait = detail::CPUKVCacheTensor<VPrecision>;

        // Gather K and V for each sequence in the batch.
        for (int seq_idx = 0; seq_idx < num_sequences; ++seq_idx)
        {
            const auto &entry = entries_[layer][seq_idx];
            const int kv_len = out_kv_lens[seq_idx];
            if (kv_len <= 0)
            {
                continue; // Skip empty sequences.
            }

            // K gather — use K precision trait for byte strides.
            const size_t k_g_rb = KGatherTrait::row_bytes(entry.K.get(), kv_dim_, head_dim_);
            const size_t k_g_hb = KGatherTrait::head_bytes(entry.K.get(), kv_dim_, head_dim_);
            if (!gather_tensor(entry.K.get(), typed_k, k_g_rb, k_g_hb, entry, seq_idx, kv_len))
            {
                return -1;
            }

            // V gather — use V precision trait for byte strides.
            const size_t v_g_rb = VGatherTrait::row_bytes(entry.V.get(), kv_dim_, head_dim_);
            const size_t v_g_hb = VGatherTrait::head_bytes(entry.V.get(), kv_dim_, head_dim_);
            if (!gather_tensor(entry.V.get(), typed_v, v_g_rb, v_g_hb, entry, seq_idx, kv_len))
            {
                return -1;
            }
        }

        return max_kv_len;
    }

    // =========================================================================
    // Eviction
    // =========================================================================

    /**
     * @brief Evict oldest tokens from ALL sequences across ALL layers.
     *
     * This is a bulk eviction operation. Each sequence in each layer has
     * its oldest tokens discarded by advancing the ring head.
     */
    template <ActivationPrecision KPrecision, ActivationPrecision VPrecision>
    void CPURingKVCache<KPrecision, VPrecision>::evict_oldest(int tokens_to_evict)
    {
        for (int seq_idx = 0; seq_idx < batch_size_; ++seq_idx)
        {
            evict_oldest_from_sequence(seq_idx, tokens_to_evict);
        }
    }

    /**
     * @brief Evict oldest tokens from one sequence across all layers.
     *
     * Advances ring head and decreases size uniformly across layers.
     * The underlying tensor data is NOT zeroed — it simply becomes
     * logically invalid and will be overwritten on the next append.
     *
     * The total_evicted_ counter tracks cumulative evictions for diagnostics.
     */
    template <ActivationPrecision KPrecision, ActivationPrecision VPrecision>
    void CPURingKVCache<KPrecision, VPrecision>::evict_oldest_from_sequence(int seq_idx, int tokens_to_evict)
    {
        if (seq_idx < 0 || seq_idx >= batch_size_ || tokens_to_evict <= 0)
        {
            return;
        }

        for (int layer = 0; layer < n_layers_; ++layer)
        {
            auto &entry = entries_[layer][seq_idx];
            // Don't evict more tokens than are actually cached.
            const int evict = std::min(tokens_to_evict, entry.size);
            if (evict <= 0)
            {
                continue;
            }

            // Advance head past the evicted tokens and shrink the valid range.
            entry.head = (entry.head + evict) % max_seq_len_;
            entry.size -= evict;
            total_evicted_ += evict;
            invalidateFP32Shadow(layer, seq_idx);
        }
    }

    // =========================================================================
    // Device and Ring Diagnostics
    // =========================================================================

    /** @brief Returns the device where a given layer's KV tensors are stored. */
    template <ActivationPrecision KPrecision, ActivationPrecision VPrecision>
    DeviceId CPURingKVCache<KPrecision, VPrecision>::get_layer_device(int layer) const
    {
        if (layer < 0 || layer >= n_layers_)
        {
            return DeviceId::cpu(); // Safe fallback for OOB.
        }
        return layer_devices_[layer];
    }

    /** @brief Returns the ring head index (oldest valid token) for diagnostics/testing. */
    template <ActivationPrecision KPrecision, ActivationPrecision VPrecision>
    int CPURingKVCache<KPrecision, VPrecision>::ring_head(int layer, int seq_idx) const
    {
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
        {
            return 0;
        }
        return entries_[layer][seq_idx].head;
    }

    /** @brief Returns the ring size (number of valid tokens) for diagnostics/testing. */
    template <ActivationPrecision KPrecision, ActivationPrecision VPrecision>
    int CPURingKVCache<KPrecision, VPrecision>::ring_size(int layer, int seq_idx) const
    {
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
        {
            return 0;
        }
        return entries_[layer][seq_idx].size;
    }

    // =========================================================================
    // Logical KV Block IO
    // =========================================================================

    template <ActivationPrecision KPrecision, ActivationPrecision VPrecision>
    typename IKVCache::KVCacheLogicalBlockLayout
    CPURingKVCache<KPrecision, VPrecision>::logicalBlockLayout(int global_layer, int token_count) const
    {
        KVCacheLogicalBlockLayout layout;
        layout.k_precision = KPrecision;
        layout.v_precision = VPrecision;
        layout.layout = kv_layout();
        layout.local_kv_heads = local_n_kv_heads_;
        layout.kv_head_start = kv_head_start_;
        layout.head_dim = head_dim_;

        const int local_layer = remapLayerIndex(global_layer);
        if (local_layer < 0 || local_layer >= n_layers_ || batch_size_ <= 0 || token_count <= 0)
        {
            if (local_layer >= 0 && local_layer < n_layers_)
            {
                layout.device_resident = !layer_devices_[local_layer].is_cpu();
            }
            return layout;
        }

        layout.device_resident = !layer_devices_[local_layer].is_cpu();

        const auto &entry = entries_[local_layer][0];
        using KTrait = detail::CPUKVCacheTensor<KPrecision>;
        using VTrait = detail::CPUKVCacheTensor<VPrecision>;

        if (layout_mode_ == KVCacheLayoutMode::HEAD_MAJOR)
        {
            layout.k_bytes = static_cast<size_t>(local_n_kv_heads_) *
                             static_cast<size_t>(token_count) *
                             KTrait::head_bytes(entry.K.get(), kv_dim_, head_dim_);
            layout.v_bytes = static_cast<size_t>(local_n_kv_heads_) *
                             static_cast<size_t>(token_count) *
                             VTrait::head_bytes(entry.V.get(), kv_dim_, head_dim_);
        }
        else
        {
            layout.k_bytes = static_cast<size_t>(token_count) *
                             KTrait::row_bytes(entry.K.get(), kv_dim_, head_dim_);
            layout.v_bytes = static_cast<size_t>(token_count) *
                             VTrait::row_bytes(entry.V.get(), kv_dim_, head_dim_);
        }

        return layout;
    }

    template <ActivationPrecision KPrecision, ActivationPrecision VPrecision>
    typename IKVCache::KVCacheSequenceState
    CPURingKVCache<KPrecision, VPrecision>::sequenceState(int global_layer, int seq_idx) const
    {
        const int local_layer = remapLayerIndex(global_layer);
        if (local_layer < 0 || local_layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
        {
            return {};
        }

        const auto &entry = entries_[local_layer][seq_idx];
        KVCacheSequenceState state;
        state.cached_tokens = entry.size;
        state.implementation_head = entry.head;
        state.wrapped = entry.head != 0;
        return state;
    }

    template <ActivationPrecision KPrecision, ActivationPrecision VPrecision>
    bool CPURingKVCache<KPrecision, VPrecision>::exportLogicalBlock(
        const KVCacheLogicalBlockDescriptor &desc, void *dst_k, void *dst_v) const
    {
        const int local_layer = remapLayerIndex(desc.layer);
        if (local_layer < 0 || local_layer >= n_layers_ ||
            desc.seq_idx < 0 || desc.seq_idx >= batch_size_ ||
            desc.logical_token_start < 0 || desc.token_count < 0)
        {
            return false;
        }
        if (!layer_devices_[local_layer].is_cpu())
        {
            return false;
        }

        const auto &entry = entries_[local_layer][desc.seq_idx];
        if (desc.logical_token_start > entry.size ||
            desc.token_count > entry.size - desc.logical_token_start)
        {
            return false;
        }
        if (desc.token_count == 0)
        {
            return true;
        }
        if (!dst_k || !dst_v || !entry.K || !entry.V)
        {
            return false;
        }

        const auto *src_k = reinterpret_cast<const uint8_t *>(entry.K->raw_data());
        const auto *src_v = reinterpret_cast<const uint8_t *>(entry.V->raw_data());
        auto *out_k = reinterpret_cast<uint8_t *>(dst_k);
        auto *out_v = reinterpret_cast<uint8_t *>(dst_v);
        if (!src_k || !src_v)
        {
            return false;
        }

        using KTrait = detail::CPUKVCacheTensor<KPrecision>;
        using VTrait = detail::CPUKVCacheTensor<VPrecision>;
        const size_t k_rb = KTrait::row_bytes(entry.K.get(), kv_dim_, head_dim_);
        const size_t v_rb = VTrait::row_bytes(entry.V.get(), kv_dim_, head_dim_);
        const size_t k_hb = KTrait::head_bytes(entry.K.get(), kv_dim_, head_dim_);
        const size_t v_hb = VTrait::head_bytes(entry.V.get(), kv_dim_, head_dim_);

        if (layout_mode_ == KVCacheLayoutMode::HEAD_MAJOR)
        {
            for (int h = 0; h < local_n_kv_heads_; ++h)
            {
                for (int i = 0; i < desc.token_count; ++i)
                {
                    const int logical = desc.logical_token_start + i;
                    const int phys = (entry.head + logical) % max_seq_len_;
                    const size_t src_k_row = static_cast<size_t>(h) * static_cast<size_t>(max_seq_len_) + static_cast<size_t>(phys);
                    const size_t src_v_row = static_cast<size_t>(h) * static_cast<size_t>(max_seq_len_) + static_cast<size_t>(phys);
                    const size_t dst_row = static_cast<size_t>(h) * static_cast<size_t>(desc.token_count) + static_cast<size_t>(i);
                    std::memcpy(out_k + dst_row * k_hb, src_k + src_k_row * k_hb, k_hb);
                    std::memcpy(out_v + dst_row * v_hb, src_v + src_v_row * v_hb, v_hb);
                }
            }
            return true;
        }

        for (int i = 0; i < desc.token_count; ++i)
        {
            const int logical = desc.logical_token_start + i;
            const int phys = (entry.head + logical) % max_seq_len_;
            std::memcpy(out_k + static_cast<size_t>(i) * k_rb,
                        src_k + static_cast<size_t>(phys) * k_rb,
                        k_rb);
            std::memcpy(out_v + static_cast<size_t>(i) * v_rb,
                        src_v + static_cast<size_t>(phys) * v_rb,
                        v_rb);
        }

        return true;
    }

    template <ActivationPrecision KPrecision, ActivationPrecision VPrecision>
    bool CPURingKVCache<KPrecision, VPrecision>::importLogicalBlock(
        const KVCacheLogicalBlockDescriptor &desc, const void *src_k, const void *src_v)
    {
        const int local_layer = remapLayerIndex(desc.layer);
        if (local_layer < 0 || local_layer >= n_layers_ ||
            desc.seq_idx < 0 || desc.seq_idx >= batch_size_ ||
            desc.logical_token_start < 0 || desc.token_count < 0 ||
            desc.logical_token_start > max_seq_len_ ||
            desc.token_count > max_seq_len_ - desc.logical_token_start)
        {
            return false;
        }
        if (!layer_devices_[local_layer].is_cpu())
        {
            return false;
        }

        auto &entry = entries_[local_layer][desc.seq_idx];
        if (desc.token_count == 0)
        {
            if (desc.logical_token_start == 0)
            {
                entry.head = 0;
                entry.size = 0;
                invalidateFP32Shadow(local_layer, desc.seq_idx);
            }
            return true;
        }
        if (!src_k || !src_v || !entry.K || !entry.V)
        {
            return false;
        }
        if (desc.logical_token_start > entry.size || entry.head != 0)
        {
            return false;
        }

        const auto *in_k = reinterpret_cast<const uint8_t *>(src_k);
        const auto *in_v = reinterpret_cast<const uint8_t *>(src_v);
        auto *dst_k = reinterpret_cast<uint8_t *>(entry.K->raw_mutable_data());
        auto *dst_v = reinterpret_cast<uint8_t *>(entry.V->raw_mutable_data());
        if (!dst_k || !dst_v)
        {
            return false;
        }

        using KTrait = detail::CPUKVCacheTensor<KPrecision>;
        using VTrait = detail::CPUKVCacheTensor<VPrecision>;
        const size_t k_rb = KTrait::row_bytes(entry.K.get(), kv_dim_, head_dim_);
        const size_t v_rb = VTrait::row_bytes(entry.V.get(), kv_dim_, head_dim_);
        const size_t k_hb = KTrait::head_bytes(entry.K.get(), kv_dim_, head_dim_);
        const size_t v_hb = VTrait::head_bytes(entry.V.get(), kv_dim_, head_dim_);

        if (layout_mode_ == KVCacheLayoutMode::HEAD_MAJOR)
        {
            for (int h = 0; h < local_n_kv_heads_; ++h)
            {
                for (int i = 0; i < desc.token_count; ++i)
                {
                    const size_t src_row = static_cast<size_t>(h) * static_cast<size_t>(desc.token_count) + static_cast<size_t>(i);
                    const size_t dst_row = static_cast<size_t>(h) * static_cast<size_t>(max_seq_len_) +
                                           static_cast<size_t>(desc.logical_token_start + i);
                    std::memcpy(dst_k + dst_row * k_hb, in_k + src_row * k_hb, k_hb);
                    std::memcpy(dst_v + dst_row * v_hb, in_v + src_row * v_hb, v_hb);
                }
            }
        }
        else
        {
            for (int i = 0; i < desc.token_count; ++i)
            {
                const size_t dst_row = static_cast<size_t>(desc.logical_token_start + i);
                std::memcpy(dst_k + dst_row * k_rb,
                            in_k + static_cast<size_t>(i) * k_rb,
                            k_rb);
                std::memcpy(dst_v + dst_row * v_rb,
                            in_v + static_cast<size_t>(i) * v_rb,
                            v_rb);
            }
        }

        entry.head = 0;
        entry.size = std::max(entry.size, desc.logical_token_start + desc.token_count);
        invalidateFP32Shadow(local_layer, desc.seq_idx);
        return true;
    }

    template <ActivationPrecision KPrecision, ActivationPrecision VPrecision>
    bool CPURingKVCache<KPrecision, VPrecision>::truncateSequence(int seq_idx, int cached_tokens, void *stream)
    {
        (void)stream;
        if (seq_idx < 0 || seq_idx >= batch_size_ || cached_tokens < 0 || cached_tokens > max_seq_len_)
        {
            return false;
        }

        for (int layer = 0; layer < n_layers_; ++layer)
        {
            if (cached_tokens > entries_[layer][seq_idx].size)
            {
                return false;
            }
        }

        for (int layer = 0; layer < n_layers_; ++layer)
        {
            auto &entry = entries_[layer][seq_idx];
            if (entry.size == cached_tokens)
            {
                continue;
            }
            if (cached_tokens == 0)
            {
                entry.head = 0;
            }
            entry.size = cached_tokens;
            invalidateFP32Shadow(layer, seq_idx);
        }
        return true;
    }

    template <ActivationPrecision KPrecision, ActivationPrecision VPrecision>
    void CPURingKVCache<KPrecision, VPrecision>::advanceHead(int layer, int seq_idx, int num_tokens)
    {
        if (layer < 0 || layer >= n_layers_ ||
            seq_idx < 0 || seq_idx >= batch_size_ ||
            num_tokens <= 0)
        {
            return;
        }

        auto &entry = entries_[layer][seq_idx];
        if (entry.size + num_tokens <= max_seq_len_)
        {
            entry.size += num_tokens;
        }
        else
        {
            const int to_evict = entry.size + num_tokens - max_seq_len_;
            entry.head = (entry.head + to_evict) % max_seq_len_;
            entry.size = max_seq_len_;
        }
        invalidateFP32Shadow(layer, seq_idx);
    }

    // =========================================================================
    // Factory Functions
    // =========================================================================
    // These type-erased factories allow runtime precision selection.
    // They map the ActivationPrecision enum to the correct template
    // instantiation. This is the pattern for creating a CPURingKVCache
    // when the precision is not known at compile time (e.g., parsed from
    // a model config file).
    // =========================================================================

    /**
     * @brief Factory: non-sharded cache, uniform device.
     */
    std::unique_ptr<ICPUKVCache> createCPURingKVCache(
        ActivationPrecision precision,
        const IMPIContext &mpi_ctx,
        int n_layers, int batch_size, int max_seq_len,
        int n_kv_heads, int head_dim,
        DeviceId device,
        KVCacheLayoutMode layout_mode)
    {
        switch (precision)
        {
        case ActivationPrecision::FP32:
            return std::make_unique<CPURingKVCacheFP32>(mpi_ctx, n_layers, batch_size, max_seq_len, n_kv_heads, head_dim, device, layout_mode);
        case ActivationPrecision::BF16:
            return std::make_unique<CPURingKVCacheBF16>(mpi_ctx, n_layers, batch_size, max_seq_len, n_kv_heads, head_dim, device, layout_mode);
        case ActivationPrecision::FP16:
            return std::make_unique<CPURingKVCacheFP16>(mpi_ctx, n_layers, batch_size, max_seq_len, n_kv_heads, head_dim, device, layout_mode);
        case ActivationPrecision::Q8_1:
            return std::make_unique<CPURingKVCacheQ8_1>(mpi_ctx, n_layers, batch_size, max_seq_len, n_kv_heads, head_dim, device, layout_mode);
        case ActivationPrecision::Q16_1:
            return std::make_unique<CPURingKVCacheQ16_1>(mpi_ctx, n_layers, batch_size, max_seq_len, n_kv_heads, head_dim, device, layout_mode);
        case ActivationPrecision::TQ4:
            return std::make_unique<CPURingKVCacheTQ4>(mpi_ctx, n_layers, batch_size, max_seq_len, n_kv_heads, head_dim, device, layout_mode);
        case ActivationPrecision::TQ8:
            return std::make_unique<CPURingKVCacheTQ>(mpi_ctx, n_layers, batch_size, max_seq_len, n_kv_heads, head_dim, device, layout_mode);
        default:
            LOG_ERROR("createCPURingKVCache: unsupported precision " << static_cast<int>(precision));
            return nullptr;
        }
    }

    /**
     * @brief Factory: non-sharded cache, per-layer device placement.
     */
    std::unique_ptr<ICPUKVCache> createCPURingKVCache(
        ActivationPrecision precision,
        const IMPIContext &mpi_ctx,
        int n_layers, int batch_size, int max_seq_len,
        int n_kv_heads, int head_dim,
        const std::vector<int> &attention_devices,
        KVCacheLayoutMode layout_mode)
    {
        switch (precision)
        {
        case ActivationPrecision::FP32:
            return std::make_unique<CPURingKVCacheFP32>(mpi_ctx, n_layers, batch_size, max_seq_len, n_kv_heads, head_dim, attention_devices, layout_mode);
        case ActivationPrecision::BF16:
            return std::make_unique<CPURingKVCacheBF16>(mpi_ctx, n_layers, batch_size, max_seq_len, n_kv_heads, head_dim, attention_devices, layout_mode);
        case ActivationPrecision::FP16:
            return std::make_unique<CPURingKVCacheFP16>(mpi_ctx, n_layers, batch_size, max_seq_len, n_kv_heads, head_dim, attention_devices, layout_mode);
        case ActivationPrecision::Q8_1:
            return std::make_unique<CPURingKVCacheQ8_1>(mpi_ctx, n_layers, batch_size, max_seq_len, n_kv_heads, head_dim, attention_devices, layout_mode);
        case ActivationPrecision::Q16_1:
            return std::make_unique<CPURingKVCacheQ16_1>(mpi_ctx, n_layers, batch_size, max_seq_len, n_kv_heads, head_dim, attention_devices, layout_mode);
        case ActivationPrecision::TQ4:
            return std::make_unique<CPURingKVCacheTQ4>(mpi_ctx, n_layers, batch_size, max_seq_len, n_kv_heads, head_dim, attention_devices, layout_mode);
        case ActivationPrecision::TQ8:
            return std::make_unique<CPURingKVCacheTQ>(mpi_ctx, n_layers, batch_size, max_seq_len, n_kv_heads, head_dim, attention_devices, layout_mode);
        default:
            LOG_ERROR("createCPURingKVCache(attention_devices): unsupported precision " << static_cast<int>(precision));
            return nullptr;
        }
    }

    /**
     * @brief Factory: sharded cache for tensor parallelism, uniform device.
     */
    std::unique_ptr<ICPUKVCache> createShardedCPURingKVCache(
        ActivationPrecision precision,
        const IMPIContext &mpi_ctx,
        int n_layers, int batch_size, int max_seq_len,
        int n_kv_heads, int local_n_kv_heads, int kv_head_start,
        int head_dim, DeviceId device,
        KVCacheLayoutMode layout_mode)
    {
        switch (precision)
        {
        case ActivationPrecision::FP32:
            return std::make_unique<CPURingKVCacheFP32>(mpi_ctx, n_layers, batch_size, max_seq_len,
                                                        n_kv_heads, local_n_kv_heads, kv_head_start, head_dim, device, layout_mode);
        case ActivationPrecision::BF16:
            return std::make_unique<CPURingKVCacheBF16>(mpi_ctx, n_layers, batch_size, max_seq_len,
                                                        n_kv_heads, local_n_kv_heads, kv_head_start, head_dim, device, layout_mode);
        case ActivationPrecision::FP16:
            return std::make_unique<CPURingKVCacheFP16>(mpi_ctx, n_layers, batch_size, max_seq_len,
                                                        n_kv_heads, local_n_kv_heads, kv_head_start, head_dim, device, layout_mode);
        case ActivationPrecision::Q8_1:
            return std::make_unique<CPURingKVCacheQ8_1>(mpi_ctx, n_layers, batch_size, max_seq_len,
                                                        n_kv_heads, local_n_kv_heads, kv_head_start, head_dim, device, layout_mode);
        case ActivationPrecision::Q16_1:
            return std::make_unique<CPURingKVCacheQ16_1>(mpi_ctx, n_layers, batch_size, max_seq_len,
                                                         n_kv_heads, local_n_kv_heads, kv_head_start, head_dim, device, layout_mode);
        case ActivationPrecision::TQ4:
            return std::make_unique<CPURingKVCacheTQ4>(mpi_ctx, n_layers, batch_size, max_seq_len,
                                                       n_kv_heads, local_n_kv_heads, kv_head_start, head_dim, device, layout_mode);
        case ActivationPrecision::TQ8:
            return std::make_unique<CPURingKVCacheTQ>(mpi_ctx, n_layers, batch_size, max_seq_len,
                                                      n_kv_heads, local_n_kv_heads, kv_head_start, head_dim, device, layout_mode);
        default:
            LOG_ERROR("createShardedCPURingKVCache: unsupported precision " << static_cast<int>(precision));
            return nullptr;
        }
    }

    /**
     * @brief Factory: sharded cache for tensor parallelism, per-layer devices.
     */
    std::unique_ptr<ICPUKVCache> createShardedCPURingKVCache(
        ActivationPrecision precision,
        const IMPIContext &mpi_ctx,
        int n_layers, int batch_size, int max_seq_len,
        int n_kv_heads, int local_n_kv_heads, int kv_head_start,
        int head_dim,
        const std::vector<int> &attention_devices,
        KVCacheLayoutMode layout_mode)
    {
        switch (precision)
        {
        case ActivationPrecision::FP32:
            return std::make_unique<CPURingKVCacheFP32>(mpi_ctx, n_layers, batch_size, max_seq_len,
                                                        n_kv_heads, local_n_kv_heads, kv_head_start, head_dim, attention_devices, layout_mode);
        case ActivationPrecision::BF16:
            return std::make_unique<CPURingKVCacheBF16>(mpi_ctx, n_layers, batch_size, max_seq_len,
                                                        n_kv_heads, local_n_kv_heads, kv_head_start, head_dim, attention_devices, layout_mode);
        case ActivationPrecision::FP16:
            return std::make_unique<CPURingKVCacheFP16>(mpi_ctx, n_layers, batch_size, max_seq_len,
                                                        n_kv_heads, local_n_kv_heads, kv_head_start, head_dim, attention_devices, layout_mode);
        case ActivationPrecision::Q8_1:
            return std::make_unique<CPURingKVCacheQ8_1>(mpi_ctx, n_layers, batch_size, max_seq_len,
                                                        n_kv_heads, local_n_kv_heads, kv_head_start, head_dim, attention_devices, layout_mode);
        case ActivationPrecision::Q16_1:
            return std::make_unique<CPURingKVCacheQ16_1>(mpi_ctx, n_layers, batch_size, max_seq_len,
                                                         n_kv_heads, local_n_kv_heads, kv_head_start, head_dim, attention_devices, layout_mode);
        case ActivationPrecision::TQ4:
            return std::make_unique<CPURingKVCacheTQ4>(mpi_ctx, n_layers, batch_size, max_seq_len,
                                                       n_kv_heads, local_n_kv_heads, kv_head_start, head_dim, attention_devices, layout_mode);
        case ActivationPrecision::TQ8:
            return std::make_unique<CPURingKVCacheTQ>(mpi_ctx, n_layers, batch_size, max_seq_len,
                                                      n_kv_heads, local_n_kv_heads, kv_head_start, head_dim, attention_devices, layout_mode);
        default:
            LOG_ERROR("createShardedCPURingKVCache(attention_devices): unsupported precision " << static_cast<int>(precision));
            return nullptr;
        }
    }

    // =========================================================================
    // Explicit Template Instantiations
    // =========================================================================
    // Because the template methods are defined in this .cpp file (not in the
    // header), we must explicitly instantiate all precision variants here.
    // Without these lines, the linker would fail with "undefined reference"
    // errors when other translation units try to use CPURingKVCache<FP32>, etc.
    // =========================================================================

    template class CPURingKVCache<ActivationPrecision::FP32>;
    template class CPURingKVCache<ActivationPrecision::BF16>;
    template class CPURingKVCache<ActivationPrecision::FP16>;
    template class CPURingKVCache<ActivationPrecision::Q8_1>;
    template class CPURingKVCache<ActivationPrecision::Q16_1>;
    template class CPURingKVCache<ActivationPrecision::TQ4>;
    template class CPURingKVCache<ActivationPrecision::TQ8>;
    // Asymmetric TQ: TQ8 for K, TQ4 for V
    template class CPURingKVCache<ActivationPrecision::TQ8, ActivationPrecision::TQ4>;

} // namespace llaminar2
