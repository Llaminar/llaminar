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
    template <ActivationPrecision Precision>
    CPURingKVCache<Precision>::CPURingKVCache(
        const MPIContext &mpi_ctx, int n_layers, int batch_size, int max_seq_len,
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
    template <ActivationPrecision Precision>
    CPURingKVCache<Precision>::CPURingKVCache(
        const MPIContext &mpi_ctx, int n_layers, int batch_size, int max_seq_len,
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
    template <ActivationPrecision Precision>
    CPURingKVCache<Precision>::CPURingKVCache(
        const MPIContext &mpi_ctx, int n_layers, int batch_size, int max_seq_len,
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
    template <ActivationPrecision Precision>
    CPURingKVCache<Precision>::CPURingKVCache(
        const MPIContext &mpi_ctx, int n_layers, int batch_size, int max_seq_len,
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
     * @brief Allocate a typed tensor using compile-time dispatch.
     *
     * Uses `if constexpr` to select the correct TensorFactory creation method
     * based on the Precision template parameter. This is resolved entirely at
     * compile time — no runtime branching.
     */
    template <ActivationPrecision Precision>
    std::shared_ptr<typename CPURingKVCache<Precision>::TensorT> CPURingKVCache<Precision>::allocate_tensor(
        size_t rows, size_t cols, DeviceId device)
    {
        if constexpr (Precision == ActivationPrecision::FP32)
        {
            return tensor_factory_->createFP32({rows, cols}, device);
        }
        else if constexpr (Precision == ActivationPrecision::BF16)
        {
            return tensor_factory_->createBF16({rows, cols});
        }
        else if constexpr (Precision == ActivationPrecision::FP16)
        {
            return tensor_factory_->createFP16({rows, cols});
        }
        else if constexpr (Precision == ActivationPrecision::Q8_1)
        {
            return tensor_factory_->createQ8_1({rows, cols}, device);
        }
        else
        {
            // Q16_1 precision with head-dim-aligned block size.
            // Uses BLOCK_128 for head_dim=128 (Llama-3, Qwen3), BLOCK_64 for
            // head_dim=64 (Qwen2.5-0.5B), giving 1 block per head with a
            // single scale factor — optimal for VNNI decode attention.
            return tensor_factory_->createQ16_1(
                {rows, cols}, optimal_q16_block_size(head_dim_), device);
        }
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
    template <ActivationPrecision Precision>
    void CPURingKVCache<Precision>::initialize_layer(int layer, DeviceId device)
    {
        entries_[layer].resize(batch_size_);
        for (int seq_idx = 0; seq_idx < batch_size_; ++seq_idx)
        {
            auto &entry = entries_[layer][seq_idx];
            if (layout_mode_ == KVCacheLayoutMode::HEAD_MAJOR)
            {
                // HEAD_MAJOR: rows = n_heads * max_seq_len, cols = head_dim
                entry.K = allocate_tensor(local_n_kv_heads_ * max_seq_len_, head_dim_, device);
                entry.V = allocate_tensor(local_n_kv_heads_ * max_seq_len_, head_dim_, device);
            }
            else
            {
                // POSITION_MAJOR: rows = max_seq_len, cols = kv_dim
                entry.K = allocate_tensor(max_seq_len_, kv_dim_, device);
                entry.V = allocate_tensor(max_seq_len_, kv_dim_, device);
            }
            // Ring buffer starts empty.
            entry.head = 0;
            entry.size = 0;
        }
    }

    // =========================================================================
    // Token Counting and KV Retrieval
    // =========================================================================

    template <ActivationPrecision Precision>
    int CPURingKVCache<Precision>::get_cached_tokens(int layer, int seq_idx) const
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
    template <ActivationPrecision Precision>
    bool CPURingKVCache<Precision>::get_kv(int layer, int seq_idx,
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
    template <ActivationPrecision Precision>
    bool CPURingKVCache<Precision>::get_kv(int layer, int seq_idx,
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
    // Individual K / V Accessors
    // =========================================================================
    // These are thin wrappers that return the raw K or V tensor pointer.
    // Prefer get_kv() which returns both in one call.

    template <ActivationPrecision Precision>
    ITensor *CPURingKVCache<Precision>::get_k(int layer, int seq_idx)
    {
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
        {
            return nullptr;
        }
        return entries_[layer][seq_idx].K.get();
    }

    template <ActivationPrecision Precision>
    const ITensor *CPURingKVCache<Precision>::get_k(int layer, int seq_idx) const
    {
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
        {
            return nullptr;
        }
        return entries_[layer][seq_idx].K.get();
    }

    template <ActivationPrecision Precision>
    ITensor *CPURingKVCache<Precision>::get_v(int layer, int seq_idx)
    {
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
        {
            return nullptr;
        }
        return entries_[layer][seq_idx].V.get();
    }

    template <ActivationPrecision Precision>
    const ITensor *CPURingKVCache<Precision>::get_v(int layer, int seq_idx) const
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

    template <ActivationPrecision Precision>
    void CPURingKVCache<Precision>::clear()
    {
        for (int layer = 0; layer < n_layers_; ++layer)
        {
            for (int seq_idx = 0; seq_idx < batch_size_; ++seq_idx)
            {
                entries_[layer][seq_idx].head = 0;
                entries_[layer][seq_idx].size = 0;
            }
        }
    }

    template <ActivationPrecision Precision>
    void CPURingKVCache<Precision>::clear_sequence(int layer, int seq_idx)
    {
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
        {
            return;
        }
        entries_[layer][seq_idx].head = 0;
        entries_[layer][seq_idx].size = 0;
    }

    template <ActivationPrecision Precision>
    void CPURingKVCache<Precision>::clear_layer(int layer)
    {
        if (layer < 0 || layer >= n_layers_)
        {
            return;
        }
        for (int seq_idx = 0; seq_idx < batch_size_; ++seq_idx)
        {
            entries_[layer][seq_idx].head = 0;
            entries_[layer][seq_idx].size = 0;
        }
    }

    // =========================================================================
    // Append Operations
    // =========================================================================

    /**
     * Convenience overload: infers num_tokens from the K tensor's row count.
     */
    template <ActivationPrecision Precision>
    bool CPURingKVCache<Precision>::append_kv(int layer, int seq_idx, const TensorBase *new_k, const TensorBase *new_v)
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
    template <ActivationPrecision Precision>
    bool CPURingKVCache<Precision>::append_kv(int layer, int seq_idx, const TensorBase *new_k, const TensorBase *new_v, int num_tokens)
    {
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_ || !new_k || !new_v || num_tokens <= 0)
        {
            return false;
        }

        // Downcast to the concrete tensor type for this cache's precision.
        const TensorT *typed_k = dynamic_cast<const TensorT *>(new_k);
        const TensorT *typed_v = dynamic_cast<const TensorT *>(new_v);
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
    template <ActivationPrecision Precision>
    bool CPURingKVCache<Precision>::append_kv_impl(int layer, int seq_idx, const TensorT *new_k, const TensorT *new_v, int num_tokens)
    {
        auto &entry = entries_[layer][seq_idx];

        TensorT *dst_k = entry.K.get();
        TensorT *dst_v = entry.V.get();
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
        // is a single memcpy of `kv_dim * sizeof(element)` bytes.
        //
        // For quantized formats (Q8_1, Q16_1), the copy size is
        // `blocks_per_row * sizeof(Block)` instead.
        // -----------------------------------------------------------------
        auto copy_position_major_token = [&](int src_row, int dst_row)
        {
            if constexpr (Precision == ActivationPrecision::FP32)
            {
                const float *src_k = new_k->typed_data();
                const float *src_v = new_v->typed_data();
                float *dst_k_data = dst_k->mutable_typed_data();
                float *dst_v_data = dst_v->mutable_typed_data();
                if (!src_k || !src_v || !dst_k_data || !dst_v_data)
                {
                    return false;
                }

                const float *src_k_row = src_k + static_cast<size_t>(src_row) * kv_dim_;
                const float *src_v_row = src_v + static_cast<size_t>(src_row) * kv_dim_;
                float *dst_k_row = dst_k_data + static_cast<size_t>(dst_row) * kv_dim_;
                float *dst_v_row = dst_v_data + static_cast<size_t>(dst_row) * kv_dim_;
                std::memcpy(dst_k_row, src_k_row, static_cast<size_t>(kv_dim_) * sizeof(float));
                std::memcpy(dst_v_row, src_v_row, static_cast<size_t>(kv_dim_) * sizeof(float));
                return true;
            }
            else if constexpr (Precision == ActivationPrecision::BF16 || Precision == ActivationPrecision::FP16)
            {
                const uint16_t *src_k = new_k->typed_data();
                const uint16_t *src_v = new_v->typed_data();
                uint16_t *dst_k_data = dst_k->mutable_typed_data();
                uint16_t *dst_v_data = dst_v->mutable_typed_data();
                if (!src_k || !src_v || !dst_k_data || !dst_v_data)
                {
                    return false;
                }

                const uint16_t *src_k_row = src_k + static_cast<size_t>(src_row) * kv_dim_;
                const uint16_t *src_v_row = src_v + static_cast<size_t>(src_row) * kv_dim_;
                uint16_t *dst_k_row = dst_k_data + static_cast<size_t>(dst_row) * kv_dim_;
                uint16_t *dst_v_row = dst_v_data + static_cast<size_t>(dst_row) * kv_dim_;
                std::memcpy(dst_k_row, src_k_row, static_cast<size_t>(kv_dim_) * sizeof(uint16_t));
                std::memcpy(dst_v_row, src_v_row, static_cast<size_t>(kv_dim_) * sizeof(uint16_t));
                return true;
            }
            else if constexpr (Precision == ActivationPrecision::Q8_1)
            {
                const Q8_1Block *src_k = new_k->typed_data();
                const Q8_1Block *src_v = new_v->typed_data();
                Q8_1Block *dst_k_data = dst_k->mutable_typed_data();
                Q8_1Block *dst_v_data = dst_v->mutable_typed_data();
                if (!src_k || !src_v || !dst_k_data || !dst_v_data)
                {
                    return false;
                }

                const size_t blocks_per_row = (static_cast<size_t>(kv_dim_) + Q8_1Block::BLOCK_SIZE - 1) / Q8_1Block::BLOCK_SIZE;
                const Q8_1Block *src_k_row = src_k + static_cast<size_t>(src_row) * blocks_per_row;
                const Q8_1Block *src_v_row = src_v + static_cast<size_t>(src_row) * blocks_per_row;
                Q8_1Block *dst_k_row = dst_k_data + static_cast<size_t>(dst_row) * blocks_per_row;
                Q8_1Block *dst_v_row = dst_v_data + static_cast<size_t>(dst_row) * blocks_per_row;
                std::memcpy(dst_k_row, src_k_row, blocks_per_row * sizeof(Q8_1Block));
                std::memcpy(dst_v_row, src_v_row, blocks_per_row * sizeof(Q8_1Block));
                return true;
            }
            else
            {
                const Q16BlockSize bs = dst_k->q16_block_size();
                const size_t block_bytes = q16_block_size_bytes(bs);
                const size_t block_elements = q16_block_size_elements(bs);
                const size_t blocks_per_row = (static_cast<size_t>(kv_dim_) + block_elements - 1) / block_elements;

                const uint8_t *src_k_bytes = reinterpret_cast<const uint8_t *>(new_k->raw_data());
                const uint8_t *src_v_bytes = reinterpret_cast<const uint8_t *>(new_v->raw_data());
                uint8_t *dst_k_bytes = reinterpret_cast<uint8_t *>(dst_k->raw_mutable_data());
                uint8_t *dst_v_bytes = reinterpret_cast<uint8_t *>(dst_v->raw_mutable_data());
                if (!src_k_bytes || !src_v_bytes || !dst_k_bytes || !dst_v_bytes)
                {
                    return false;
                }

                const uint8_t *src_k_row = src_k_bytes + static_cast<size_t>(src_row) * blocks_per_row * block_bytes;
                const uint8_t *src_v_row = src_v_bytes + static_cast<size_t>(src_row) * blocks_per_row * block_bytes;
                uint8_t *dst_k_row = dst_k_bytes + static_cast<size_t>(dst_row) * blocks_per_row * block_bytes;
                uint8_t *dst_v_row = dst_v_bytes + static_cast<size_t>(dst_row) * blocks_per_row * block_bytes;
                std::memcpy(dst_k_row, src_k_row, blocks_per_row * block_bytes);
                std::memcpy(dst_v_row, src_v_row, blocks_per_row * block_bytes);
                return true;
            }
        };

        // -----------------------------------------------------------------
        // Lambda: copy one token in HEAD_MAJOR layout.
        //
        // HEAD_MAJOR stores data as [head][position][head_dim]. The source
        // tensor (from the projection stage) is in POSITION_MAJOR format:
        // [position][n_heads * head_dim]. So for each source token, we need
        // to scatter the per-head slices into their respective head blocks.
        //
        // Source addressing:  src + src_row * kv_dim + h * head_dim
        // Dest addressing:    dst + (h * max_seq_len + dst_pos) * head_dim
        //
        // This is more expensive than POSITION_MAJOR append (n_heads memcpys
        // per token instead of 1) but pays off during attention where each
        // head's data is contiguous.
        // -----------------------------------------------------------------
        auto copy_head_major_token = [&](int src_row, int dst_pos)
        {
            if (local_n_kv_heads_ <= 0 || head_dim_ <= 0)
            {
                return false;
            }

            if constexpr (Precision == ActivationPrecision::FP32)
            {
                const float *src_k_all = new_k->typed_data();
                const float *src_v_all = new_v->typed_data();
                float *dst_k_ptr = dst_k->mutable_typed_data();
                float *dst_v_ptr = dst_v->mutable_typed_data();
                if (!src_k_all || !src_v_all || !dst_k_ptr || !dst_v_ptr)
                {
                    return false;
                }
                const float *src_k = src_k_all + static_cast<size_t>(src_row) * kv_dim_;
                const float *src_v = src_v_all + static_cast<size_t>(src_row) * kv_dim_;
                for (int h = 0; h < local_n_kv_heads_; ++h)
                {
                    const float *src_k_head = src_k + static_cast<size_t>(h) * head_dim_;
                    const float *src_v_head = src_v + static_cast<size_t>(h) * head_dim_;
                    float *dst_k_head = dst_k_ptr + (static_cast<size_t>(h) * max_seq_len_ + static_cast<size_t>(dst_pos)) * head_dim_;
                    float *dst_v_head = dst_v_ptr + (static_cast<size_t>(h) * max_seq_len_ + static_cast<size_t>(dst_pos)) * head_dim_;
                    std::memcpy(dst_k_head, src_k_head, static_cast<size_t>(head_dim_) * sizeof(float));
                    std::memcpy(dst_v_head, src_v_head, static_cast<size_t>(head_dim_) * sizeof(float));
                }
                return true;
            }
            else if constexpr (Precision == ActivationPrecision::BF16 || Precision == ActivationPrecision::FP16)
            {
                const uint16_t *src_k_all = new_k->typed_data();
                const uint16_t *src_v_all = new_v->typed_data();
                uint16_t *dst_k_ptr = dst_k->mutable_typed_data();
                uint16_t *dst_v_ptr = dst_v->mutable_typed_data();
                if (!src_k_all || !src_v_all || !dst_k_ptr || !dst_v_ptr)
                {
                    return false;
                }
                const uint16_t *src_k = src_k_all + static_cast<size_t>(src_row) * kv_dim_;
                const uint16_t *src_v = src_v_all + static_cast<size_t>(src_row) * kv_dim_;
                for (int h = 0; h < local_n_kv_heads_; ++h)
                {
                    const uint16_t *src_k_head = src_k + static_cast<size_t>(h) * head_dim_;
                    const uint16_t *src_v_head = src_v + static_cast<size_t>(h) * head_dim_;
                    uint16_t *dst_k_head = dst_k_ptr + (static_cast<size_t>(h) * max_seq_len_ + static_cast<size_t>(dst_pos)) * head_dim_;
                    uint16_t *dst_v_head = dst_v_ptr + (static_cast<size_t>(h) * max_seq_len_ + static_cast<size_t>(dst_pos)) * head_dim_;
                    std::memcpy(dst_k_head, src_k_head, static_cast<size_t>(head_dim_) * sizeof(uint16_t));
                    std::memcpy(dst_v_head, src_v_head, static_cast<size_t>(head_dim_) * sizeof(uint16_t));
                }
                return true;
            }
            else if constexpr (Precision == ActivationPrecision::Q8_1)
            {
                const auto *src_k_all = new_k->typed_data();
                const auto *src_v_all = new_v->typed_data();
                auto *dst_k_blocks = dst_k->mutable_typed_data();
                auto *dst_v_blocks = dst_v->mutable_typed_data();
                if (!src_k_all || !src_v_all || !dst_k_blocks || !dst_v_blocks)
                {
                    return false;
                }

                const size_t blocks_per_head = (static_cast<size_t>(head_dim_) + Q8_1Block::BLOCK_SIZE - 1) / Q8_1Block::BLOCK_SIZE;
                const size_t src_blocks_per_row = (static_cast<size_t>(kv_dim_) + Q8_1Block::BLOCK_SIZE - 1) / Q8_1Block::BLOCK_SIZE;
                const Q8_1Block *src_k_blocks = src_k_all + static_cast<size_t>(src_row) * src_blocks_per_row;
                const Q8_1Block *src_v_blocks = src_v_all + static_cast<size_t>(src_row) * src_blocks_per_row;

                for (int h = 0; h < local_n_kv_heads_; ++h)
                {
                    const Q8_1Block *src_k_head = src_k_blocks + static_cast<size_t>(h) * blocks_per_head;
                    const Q8_1Block *src_v_head = src_v_blocks + static_cast<size_t>(h) * blocks_per_head;
                    (void)src_blocks_per_row;
                    Q8_1Block *dst_k_head = dst_k_blocks + (static_cast<size_t>(h) * max_seq_len_ + static_cast<size_t>(dst_pos)) * blocks_per_head;
                    Q8_1Block *dst_v_head = dst_v_blocks + (static_cast<size_t>(h) * max_seq_len_ + static_cast<size_t>(dst_pos)) * blocks_per_head;
                    std::memcpy(dst_k_head, src_k_head, blocks_per_head * sizeof(Q8_1Block));
                    std::memcpy(dst_v_head, src_v_head, blocks_per_head * sizeof(Q8_1Block));
                }
                return true;
            }
            else
            {
                const Q16BlockSize bs = dst_k->q16_block_size();
                const size_t block_bytes = q16_block_size_bytes(bs);
                const size_t block_elements = q16_block_size_elements(bs);
                const uint8_t *src_k_bytes = reinterpret_cast<const uint8_t *>(new_k->raw_data());
                const uint8_t *src_v_bytes = reinterpret_cast<const uint8_t *>(new_v->raw_data());
                uint8_t *dst_k_bytes = reinterpret_cast<uint8_t *>(dst_k->raw_mutable_data());
                uint8_t *dst_v_bytes = reinterpret_cast<uint8_t *>(dst_v->raw_mutable_data());
                if (!src_k_bytes || !src_v_bytes || !dst_k_bytes || !dst_v_bytes)
                {
                    return false;
                }

                const size_t blocks_per_head = (static_cast<size_t>(head_dim_) + block_elements - 1) / block_elements;
                const size_t src_blocks_per_row = (static_cast<size_t>(kv_dim_) + block_elements - 1) / block_elements;
                const uint8_t *src_k_row = src_k_bytes + static_cast<size_t>(src_row) * src_blocks_per_row * block_bytes;
                const uint8_t *src_v_row = src_v_bytes + static_cast<size_t>(src_row) * src_blocks_per_row * block_bytes;
                for (int h = 0; h < local_n_kv_heads_; ++h)
                {
                    const uint8_t *src_k_head = src_k_row + static_cast<size_t>(h) * blocks_per_head * block_bytes;
                    const uint8_t *src_v_head = src_v_row + static_cast<size_t>(h) * blocks_per_head * block_bytes;
                    uint8_t *dst_k_head = dst_k_bytes + (static_cast<size_t>(h) * max_seq_len_ + static_cast<size_t>(dst_pos)) * blocks_per_head * block_bytes;
                    uint8_t *dst_v_head = dst_v_bytes + (static_cast<size_t>(h) * max_seq_len_ + static_cast<size_t>(dst_pos)) * blocks_per_head * block_bytes;
                    std::memcpy(dst_k_head, src_k_head, blocks_per_head * block_bytes);
                    std::memcpy(dst_v_head, src_v_head, blocks_per_head * block_bytes);
                }
                return true;
            }
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
    template <ActivationPrecision Precision>
    bool CPURingKVCache<Precision>::append_one_tensor(TensorT *dst, const TensorT *src, EntryT &entry, int num_tokens)
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
    template <ActivationPrecision Precision>
    int CPURingKVCache<Precision>::gather_kv_batched(
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

        TensorT *typed_k = dynamic_cast<TensorT *>(out_k);
        TensorT *typed_v = dynamic_cast<TensorT *>(out_v);
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
        // Reads tokens from the ring buffer in logical order (oldest first)
        // by mapping logical index → physical position via:
        //   physical_pos = (ring_head + logical_index) % max_seq_len
        //
        // Handles both POSITION_MAJOR and HEAD_MAJOR source layouts.
        // In HEAD_MAJOR mode, the gather also transposes the data back
        // to POSITION_MAJOR in the output.
        // -----------------------------------------------------------------
        auto gather_tensor = [&](const TensorT *src_tensor, TensorT *dst_tensor, const EntryT &entry, int seq_idx, int kv_len) -> bool
        {
            if (!src_tensor || !dst_tensor || kv_len <= 0)
            {
                return true; // Empty sequence — nothing to gather.
            }

            const int head = entry.head; // Oldest valid position in the ring.

            if (layout_mode_ == KVCacheLayoutMode::POSITION_MAJOR)
            {
                if constexpr (Precision == ActivationPrecision::FP32)
                {
                    const float *src = src_tensor->typed_data();
                    float *dst = dst_tensor->mutable_typed_data();
                    if (!src || !dst)
                    {
                        return false;
                    }

                    for (int logical = 0; logical < kv_len; ++logical)
                    {
                        const int phys = (head + logical) % max_seq_len_;
                        const size_t dst_row = static_cast<size_t>(seq_idx) * static_cast<size_t>(max_kv_len) + static_cast<size_t>(logical);
                        const float *src_row = src + static_cast<size_t>(phys) * kv_dim_;
                        float *dst_row_ptr = dst + dst_row * kv_dim_;
                        std::memcpy(dst_row_ptr, src_row, static_cast<size_t>(kv_dim_) * sizeof(float));
                    }
                    return true;
                }
                else if constexpr (Precision == ActivationPrecision::BF16 || Precision == ActivationPrecision::FP16)
                {
                    const uint16_t *src = src_tensor->typed_data();
                    uint16_t *dst = dst_tensor->mutable_typed_data();
                    if (!src || !dst)
                    {
                        return false;
                    }

                    for (int logical = 0; logical < kv_len; ++logical)
                    {
                        const int phys = (head + logical) % max_seq_len_;
                        const size_t dst_row = static_cast<size_t>(seq_idx) * static_cast<size_t>(max_kv_len) + static_cast<size_t>(logical);
                        const uint16_t *src_row = src + static_cast<size_t>(phys) * kv_dim_;
                        uint16_t *dst_row_ptr = dst + dst_row * kv_dim_;
                        std::memcpy(dst_row_ptr, src_row, static_cast<size_t>(kv_dim_) * sizeof(uint16_t));
                    }
                    return true;
                }
                else if constexpr (Precision == ActivationPrecision::Q8_1)
                {
                    const Q8_1Block *src = src_tensor->typed_data();
                    Q8_1Block *dst = dst_tensor->mutable_typed_data();
                    if (!src || !dst)
                    {
                        return false;
                    }

                    const size_t blocks_per_row = (static_cast<size_t>(kv_dim_) + Q8_1Block::BLOCK_SIZE - 1) / Q8_1Block::BLOCK_SIZE;
                    for (int logical = 0; logical < kv_len; ++logical)
                    {
                        const int phys = (head + logical) % max_seq_len_;
                        const size_t dst_row = static_cast<size_t>(seq_idx) * static_cast<size_t>(max_kv_len) + static_cast<size_t>(logical);
                        const Q8_1Block *src_row = src + static_cast<size_t>(phys) * blocks_per_row;
                        Q8_1Block *dst_row_ptr = dst + dst_row * blocks_per_row;
                        std::memcpy(dst_row_ptr, src_row, blocks_per_row * sizeof(Q8_1Block));
                    }
                    return true;
                }
                else
                {
                    const Q16BlockSize bs = src_tensor->q16_block_size();
                    const size_t block_bytes = q16_block_size_bytes(bs);
                    const size_t block_elements = q16_block_size_elements(bs);
                    const size_t blocks_per_row = (static_cast<size_t>(kv_dim_) + block_elements - 1) / block_elements;

                    const uint8_t *src_bytes = reinterpret_cast<const uint8_t *>(src_tensor->raw_data());
                    uint8_t *dst_bytes = reinterpret_cast<uint8_t *>(dst_tensor->raw_mutable_data());
                    if (!src_bytes || !dst_bytes)
                    {
                        return false;
                    }

                    for (int logical = 0; logical < kv_len; ++logical)
                    {
                        const int phys = (head + logical) % max_seq_len_;
                        const size_t dst_row = static_cast<size_t>(seq_idx) * static_cast<size_t>(max_kv_len) + static_cast<size_t>(logical);
                        const uint8_t *src_row = src_bytes + static_cast<size_t>(phys) * blocks_per_row * block_bytes;
                        uint8_t *dst_row_ptr = dst_bytes + dst_row * blocks_per_row * block_bytes;
                        std::memcpy(dst_row_ptr, src_row, blocks_per_row * block_bytes);
                    }
                    return true;
                }
            }

            // =============================================================
            // HEAD_MAJOR gather: transpose [head][pos][head_dim] →
            //   [pos][n_heads * head_dim] in the output.
            //
            // For each logical token position, iterate over all heads and
            // copy each head's slice into the correct column offset in the
            // output row.
            // =============================================================

            if constexpr (Precision == ActivationPrecision::FP32)
            {
                const float *src = reinterpret_cast<const float *>(src_tensor->raw_data());
                float *dst = reinterpret_cast<float *>(dst_tensor->raw_mutable_data());
                if (!src || !dst)
                {
                    return false;
                }
                for (int logical = 0; logical < kv_len; ++logical)
                {
                    const int phys = (head + logical) % max_seq_len_;
                    const size_t dst_row = static_cast<size_t>(seq_idx) * static_cast<size_t>(max_kv_len) + static_cast<size_t>(logical);
                    float *dst_row_ptr = dst + dst_row * static_cast<size_t>(kv_dim_);
                    for (int h = 0; h < local_n_kv_heads_; ++h)
                    {
                        const float *src_head = src + (static_cast<size_t>(h) * max_seq_len_ + static_cast<size_t>(phys)) * head_dim_;
                        std::memcpy(dst_row_ptr + static_cast<size_t>(h) * head_dim_, src_head, static_cast<size_t>(head_dim_) * sizeof(float));
                    }
                }
                return true;
            }
            else if constexpr (Precision == ActivationPrecision::BF16 || Precision == ActivationPrecision::FP16)
            {
                const uint16_t *src = reinterpret_cast<const uint16_t *>(src_tensor->raw_data());
                uint16_t *dst = reinterpret_cast<uint16_t *>(dst_tensor->raw_mutable_data());
                if (!src || !dst)
                {
                    return false;
                }
                for (int logical = 0; logical < kv_len; ++logical)
                {
                    const int phys = (head + logical) % max_seq_len_;
                    const size_t dst_row = static_cast<size_t>(seq_idx) * static_cast<size_t>(max_kv_len) + static_cast<size_t>(logical);
                    uint16_t *dst_row_ptr = dst + dst_row * static_cast<size_t>(kv_dim_);
                    for (int h = 0; h < local_n_kv_heads_; ++h)
                    {
                        const uint16_t *src_head = src + (static_cast<size_t>(h) * max_seq_len_ + static_cast<size_t>(phys)) * head_dim_;
                        std::memcpy(dst_row_ptr + static_cast<size_t>(h) * head_dim_, src_head, static_cast<size_t>(head_dim_) * sizeof(uint16_t));
                    }
                }
                return true;
            }
            else if constexpr (Precision == ActivationPrecision::Q8_1)
            {
                const Q8_1Block *src = reinterpret_cast<const Q8_1Block *>(src_tensor->raw_data());
                Q8_1Block *dst = reinterpret_cast<Q8_1Block *>(dst_tensor->raw_mutable_data());
                if (!src || !dst)
                {
                    return false;
                }
                const size_t blocks_per_head = (static_cast<size_t>(head_dim_) + Q8_1Block::BLOCK_SIZE - 1) / Q8_1Block::BLOCK_SIZE;
                const size_t dst_blocks_per_row = (static_cast<size_t>(kv_dim_) + Q8_1Block::BLOCK_SIZE - 1) / Q8_1Block::BLOCK_SIZE;
                if (dst_blocks_per_row < static_cast<size_t>(local_n_kv_heads_) * blocks_per_head)
                {
                    return false;
                }

                for (int logical = 0; logical < kv_len; ++logical)
                {
                    const int phys = (head + logical) % max_seq_len_;
                    const size_t dst_row = static_cast<size_t>(seq_idx) * static_cast<size_t>(max_kv_len) + static_cast<size_t>(logical);
                    Q8_1Block *dst_row_ptr = dst + dst_row * dst_blocks_per_row;
                    for (int h = 0; h < local_n_kv_heads_; ++h)
                    {
                        const Q8_1Block *src_head = src + (static_cast<size_t>(h) * max_seq_len_ + static_cast<size_t>(phys)) * blocks_per_head;
                        std::memcpy(dst_row_ptr + static_cast<size_t>(h) * blocks_per_head,
                                    src_head,
                                    blocks_per_head * sizeof(Q8_1Block));
                    }
                }
                return true;
            }
            else
            {
                const Q16BlockSize bs = src_tensor->q16_block_size();
                const size_t block_bytes = q16_block_size_bytes(bs);
                const size_t block_elements = q16_block_size_elements(bs);
                const uint8_t *src = reinterpret_cast<const uint8_t *>(src_tensor->raw_data());
                uint8_t *dst = reinterpret_cast<uint8_t *>(dst_tensor->raw_mutable_data());
                if (!src || !dst)
                {
                    return false;
                }

                const size_t blocks_per_head = (static_cast<size_t>(head_dim_) + block_elements - 1) / block_elements;
                const size_t dst_blocks_per_row = (static_cast<size_t>(kv_dim_) + block_elements - 1) / block_elements;
                if (dst_blocks_per_row < static_cast<size_t>(local_n_kv_heads_) * blocks_per_head)
                {
                    return false;
                }

                for (int logical = 0; logical < kv_len; ++logical)
                {
                    const int phys = (head + logical) % max_seq_len_;
                    const size_t dst_row = static_cast<size_t>(seq_idx) * static_cast<size_t>(max_kv_len) + static_cast<size_t>(logical);
                    uint8_t *dst_row_ptr = dst + dst_row * dst_blocks_per_row * block_bytes;
                    for (int h = 0; h < local_n_kv_heads_; ++h)
                    {
                        const uint8_t *src_head = src + (static_cast<size_t>(h) * max_seq_len_ + static_cast<size_t>(phys)) * blocks_per_head * block_bytes;
                        std::memcpy(dst_row_ptr + static_cast<size_t>(h) * blocks_per_head * block_bytes,
                                    src_head,
                                    blocks_per_head * block_bytes);
                    }
                }
                return true;
            }
        };

        // Gather K and V for each sequence in the batch.
        for (int seq_idx = 0; seq_idx < num_sequences; ++seq_idx)
        {
            const auto &entry = entries_[layer][seq_idx];
            const int kv_len = out_kv_lens[seq_idx];
            if (kv_len <= 0)
            {
                continue; // Skip empty sequences.
            }

            if (!gather_tensor(entry.K.get(), typed_k, entry, seq_idx, kv_len))
            {
                return -1;
            }
            if (!gather_tensor(entry.V.get(), typed_v, entry, seq_idx, kv_len))
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
    template <ActivationPrecision Precision>
    void CPURingKVCache<Precision>::evict_oldest(int tokens_to_evict)
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
    template <ActivationPrecision Precision>
    void CPURingKVCache<Precision>::evict_oldest_from_sequence(int seq_idx, int tokens_to_evict)
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
        }
    }

    // =========================================================================
    // Device and Ring Diagnostics
    // =========================================================================

    /** @brief Returns the device where a given layer's KV tensors are stored. */
    template <ActivationPrecision Precision>
    DeviceId CPURingKVCache<Precision>::get_layer_device(int layer) const
    {
        if (layer < 0 || layer >= n_layers_)
        {
            return DeviceId::cpu(); // Safe fallback for OOB.
        }
        return layer_devices_[layer];
    }

    /** @brief Returns the ring head index (oldest valid token) for diagnostics/testing. */
    template <ActivationPrecision Precision>
    int CPURingKVCache<Precision>::ring_head(int layer, int seq_idx) const
    {
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
        {
            return 0;
        }
        return entries_[layer][seq_idx].head;
    }

    /** @brief Returns the ring size (number of valid tokens) for diagnostics/testing. */
    template <ActivationPrecision Precision>
    int CPURingKVCache<Precision>::ring_size(int layer, int seq_idx) const
    {
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
        {
            return 0;
        }
        return entries_[layer][seq_idx].size;
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
        const MPIContext &mpi_ctx,
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
        const MPIContext &mpi_ctx,
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
        const MPIContext &mpi_ctx,
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
        const MPIContext &mpi_ctx,
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

} // namespace llaminar2
