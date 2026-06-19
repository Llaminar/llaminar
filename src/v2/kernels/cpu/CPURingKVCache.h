/**
 * @file CPURingKVCache.h
 * @brief Ring-buffer KV cache for transformer attention on CPU (and optionally GPU).
 *
 * ## What is a KV Cache?
 *
 * In transformer-based LLM inference, each layer computes Key (K) and Value (V)
 * projections for every token. During autoregressive decoding, previously computed
 * K/V values are reused so we don't recompute them. The KV cache stores these
 * tensors and grows by one token per decode step.
 *
 * ## Why a Ring Buffer?
 *
 * A naive KV cache would grow unboundedly with sequence length. A ring buffer
 * imposes a fixed maximum sequence length (`max_seq_len`). Once full, the oldest
 * tokens are overwritten in a circular fashion — this is how sliding-window
 * attention works. The ring buffer uses two integers per entry:
 *
 *   - `head`: Index of the oldest valid token in the circular buffer.
 *   - `size`: Number of valid tokens currently stored.
 *
 * When `size < max_seq_len`, new tokens are appended at `(head + size) % max_seq_len`.
 * When the buffer is full (`size == max_seq_len`), the oldest token at `head` is
 * overwritten and `head` advances forward.
 *
 * ## Memory Layouts
 *
 * Two memory layouts are supported (see KVCacheLayoutMode):
 *
 *   - **POSITION_MAJOR**: `[position][n_kv_heads * head_dim]`
 *     Each row is one token position containing all KV heads concatenated.
 *     Cache-append friendly (one memcpy per token).
 *
 *   - **HEAD_MAJOR**: `[n_kv_heads][position][head_dim]`
 *     Each head's positions are contiguous in memory.
 *     Attention-compute friendly (no scatter for per-head access).
 *
 * ## Tensor Parallelism (Sharding)
 *
 * In multi-device tensor-parallel inference, KV heads are split across ranks.
 * A "sharded" cache only stores `local_n_kv_heads` out of the total `n_kv_heads`,
 * starting at index `kv_head_start`. This reduces per-rank memory usage.
 *
 * ## Template Parameter
 *
 * The `Precision` template parameter controls the storage format of K/V tensors:
 * FP32, BF16, FP16, Q8_1 (8-bit quantized), or Q16_1 (16-bit quantized).
 * This is resolved at compile time via `detail::CPUKVCacheTensor<Precision>::Type`.
 *
 * @see ICPUKVCache  Base interface this class implements.
 * @see KVCacheLayoutMode  Enum controlling memory layout.
 * @see CPUKVCache.h  Contains the ICPUKVCache interface and precision→tensor mappings.
 */

#pragma once

#include "CPUKVCache.h"
#include "../../utils/Logger.h"

namespace llaminar2
{

    /**
     * @brief Per-sequence, per-layer ring buffer entry for the KV cache.
     *
     * Each entry holds one K tensor and one V tensor (both pre-allocated to
     * `max_seq_len` rows), plus ring buffer bookkeeping:
     *
     *   - `head`: Index of the oldest valid token (front of the ring).
     *   - `size`: Number of valid tokens currently in the ring.
     *
     * Together, these define the valid range of data within the pre-allocated
     * tensor. Tokens are logically ordered from `head` to `(head + size - 1) % max_seq_len`.
     *
     * @tparam KPrecision  Activation storage precision for K cache.
     * @tparam VPrecision  Activation storage precision for V cache (defaults to KPrecision).
     */
    template <ActivationPrecision KPrecision, ActivationPrecision VPrecision = KPrecision>
    struct CPURingKVCacheEntry
    {
        /// @brief The concrete tensor type for K storage at this precision.
        using KTensorT = typename detail::CPUKVCacheTensor<KPrecision>::Type;
        /// @brief The concrete tensor type for V storage at this precision.
        using VTensorT = typename detail::CPUKVCacheTensor<VPrecision>::Type;

        std::shared_ptr<KTensorT> K; ///< Key tensor, pre-allocated to max_seq_len rows.
        std::shared_ptr<VTensorT> V; ///< Value tensor, pre-allocated to max_seq_len rows.
        int head = 0;                ///< Ring buffer head: index of the oldest valid token.
        int size = 0;                ///< Number of valid tokens currently stored in the ring.
    };

    /**
     * @brief Ring-buffer KV cache implementation for CPU-side transformer attention.
     *
     * This is the primary KV cache used during inference. It pre-allocates fixed-size
     * K and V tensors for each (layer, sequence) pair and manages them as circular
     * buffers. The ring buffer design provides O(1) append and bounded memory usage.
     *
     * ### Lifecycle
     *
     * 1. **Construction**: Pre-allocates K/V tensors for all layers × batch sequences.
     * 2. **Prefill**: `append_kv()` is called with all prompt tokens at once.
     * 3. **Decode**: `append_kv()` is called with one new token per step.
     * 4. **Attention**: `get_kv()` or `gather_kv_batched()` provides K/V for the
     *    attention kernel to read.
     * 5. **Eviction**: When the ring is full, oldest tokens are automatically overwritten.
     *    Explicit eviction is also available via `evict_oldest()`.
     *
     * ### Device Placement
     *
     * Each layer's KV tensors can live on a different device (CPU or GPU). This is
     * controlled either by a single `DeviceId` (same device for all layers) or by
     * a per-layer `attention_devices` vector passed to the constructor.
     *
     * @tparam KPrecision  Activation storage format for K cache. Defaults to FP32.
     * @tparam VPrecision  Activation storage format for V cache. Defaults to KPrecision.
     *                     When KPrecision != VPrecision, enables asymmetric K/V storage
     *                     (e.g., TQ8 for K, TQ4 for V).
     *
     * @see ICPUKVCache  The polymorphic interface this class implements.
     * @see CPURingKVCacheEntry  The per-sequence ring buffer bookkeeping struct.
     */
    template <ActivationPrecision KPrecision = ActivationPrecision::FP32, ActivationPrecision VPrecision = KPrecision>
    class CPURingKVCache : public ICPUKVCache
    {
    public:
        /// @brief Concrete tensor type for K storage at this precision.
        using KTensorT = typename detail::CPUKVCacheTensor<KPrecision>::Type;

        /// @brief Concrete tensor type for V storage at this precision.
        using VTensorT = typename detail::CPUKVCacheTensor<VPrecision>::Type;

        /// @brief Ring buffer entry type (K tensor + V tensor + head/size bookkeeping).
        using EntryT = CPURingKVCacheEntry<KPrecision, VPrecision>;

        // =====================================================================
        // Constructors
        // =====================================================================

        /**
         * @brief Construct a non-sharded KV cache with uniform device placement.
         *
         * All KV heads are stored locally (no tensor parallelism).
         * All layers use the same device.
         *
         * @param mpi_ctx     MPI context (used by TensorFactory for allocation).
         * @param n_layers    Number of transformer layers.
         * @param batch_size  Maximum batch size (number of concurrent sequences).
         * @param max_seq_len Maximum sequence length (ring buffer capacity per sequence).
         * @param n_kv_heads  Total number of KV attention heads.
         * @param head_dim    Dimension of each attention head.
         * @param device      Device to allocate K/V tensors on (default: CPU).
         * @param layout_mode Memory layout for K/V tensors (default: POSITION_MAJOR).
         */
        CPURingKVCache(const IMPIContext &mpi_ctx, int n_layers, int batch_size, int max_seq_len,
                       int n_kv_heads, int head_dim, DeviceId device = DeviceId::cpu(),
                       KVCacheLayoutMode layout_mode = KVCacheLayoutMode::POSITION_MAJOR);

        /**
         * @brief Construct a non-sharded KV cache with per-layer device placement.
         *
         * Each layer's K/V tensors can be placed on a different device. The
         * `attention_devices` vector maps layer index → legacy device index
         * (0 = CPU, 1+ = CUDA device N-1).
         *
         * @param mpi_ctx           MPI context for tensor allocation.
         * @param n_layers          Number of transformer layers.
         * @param batch_size        Maximum batch size.
         * @param max_seq_len       Ring buffer capacity per sequence.
         * @param n_kv_heads        Total number of KV attention heads.
         * @param head_dim          Dimension of each attention head.
         * @param attention_devices Per-layer device assignment (size >= n_layers).
         * @param layout_mode       Memory layout (default: POSITION_MAJOR).
         */
        CPURingKVCache(const IMPIContext &mpi_ctx, int n_layers, int batch_size, int max_seq_len,
                       int n_kv_heads, int head_dim, const std::vector<int> &attention_devices,
                       KVCacheLayoutMode layout_mode = KVCacheLayoutMode::POSITION_MAJOR);

        /**
         * @brief Construct a sharded KV cache with uniform device placement.
         *
         * Used in tensor-parallel inference where each rank handles a subset of
         * KV heads. Only `local_n_kv_heads` heads are allocated, starting at
         * `kv_head_start`.
         *
         * @param mpi_ctx          MPI context for tensor allocation.
         * @param n_layers         Number of transformer layers.
         * @param batch_size       Maximum batch size.
         * @param max_seq_len      Ring buffer capacity per sequence.
         * @param n_kv_heads       Total KV heads across all ranks.
         * @param local_n_kv_heads Number of KV heads owned by this rank.
         * @param kv_head_start    Starting head index for this rank's shard.
         * @param head_dim         Dimension of each attention head.
         * @param device           Device for all layers (default: CPU).
         * @param layout_mode      Memory layout (default: POSITION_MAJOR).
         */
        CPURingKVCache(const IMPIContext &mpi_ctx, int n_layers, int batch_size, int max_seq_len,
                       int n_kv_heads, int local_n_kv_heads, int kv_head_start,
                       int head_dim, DeviceId device = DeviceId::cpu(),
                       KVCacheLayoutMode layout_mode = KVCacheLayoutMode::POSITION_MAJOR);

        /**
         * @brief Construct a sharded KV cache with per-layer device placement.
         *
         * Combines tensor parallelism (head sharding) with heterogeneous device
         * placement across layers.
         *
         * @param mpi_ctx           MPI context for tensor allocation.
         * @param n_layers          Number of transformer layers.
         * @param batch_size        Maximum batch size.
         * @param max_seq_len       Ring buffer capacity per sequence.
         * @param n_kv_heads        Total KV heads across all ranks.
         * @param local_n_kv_heads  Number of KV heads owned by this rank.
         * @param kv_head_start     Starting head index for this rank's shard.
         * @param head_dim          Dimension of each attention head.
         * @param attention_devices Per-layer device assignment (size >= n_layers).
         * @param layout_mode       Memory layout (default: POSITION_MAJOR).
         */
        CPURingKVCache(const IMPIContext &mpi_ctx, int n_layers, int batch_size, int max_seq_len,
                       int n_kv_heads, int local_n_kv_heads, int kv_head_start,
                       int head_dim, const std::vector<int> &attention_devices,
                       KVCacheLayoutMode layout_mode = KVCacheLayoutMode::POSITION_MAJOR);

        // =====================================================================
        // Metadata Accessors
        // =====================================================================

        /** @brief Returns the K-cache activation precision. */
        ActivationPrecision k_precision() const override { return KPrecision; }

        /** @brief Returns the V-cache activation precision. */
        ActivationPrecision v_precision() const override { return VPrecision; }

        /** @brief Returns the maximum sequence length (ring buffer capacity). */
        int max_seq_len() const override { return max_seq_len_; }

        /** @brief Returns the number of transformer layers. */
        int n_layers() const override { return n_layers_; }

        /**
         * @brief Returns the tensor layout enum matching the configured layout mode.
         *
         * HEAD_MAJOR maps to KV_HEAD_POS_DIM, POSITION_MAJOR maps to KV_POS_HEAD_DIM.
         * Attention kernels use this to know how to index into the cache tensors.
         */
        TensorLayout kv_layout() const override
        {
            return (layout_mode_ == KVCacheLayoutMode::HEAD_MAJOR)
                       ? TensorLayout::KV_HEAD_POS_DIM
                       : TensorLayout::KV_POS_HEAD_DIM;
        }

        // =====================================================================
        // Token Counting and KV Retrieval
        // =====================================================================

        /**
         * @brief Returns the number of cached tokens for a given layer and sequence.
         *
         * This is the ring buffer's `size` field — the count of valid tokens,
         * not the physical capacity.
         *
         * @param layer   Layer index (0-based).
         * @param seq_idx Sequence index within the batch (default: 0).
         * @return Number of cached tokens, or 0 if indices are out of bounds.
         */
        int get_cached_tokens(int layer, int seq_idx = 0) const override;

        /**
         * @brief Retrieve mutable K and V tensor pointers for a (layer, sequence) pair.
         *
         * The returned tensors are the full pre-allocated ring buffers. Use
         * `get_cached_tokens()` or the `out_kv_len` parameter to know how many
         * rows contain valid data. The valid token range starts at `ring_head()`
         * and wraps around circularly.
         *
         * @param layer     Layer index.
         * @param seq_idx   Sequence index.
         * @param[out] out_k      Receives pointer to the K cache tensor (may be nullptr on failure).
         * @param[out] out_v      Receives pointer to the V cache tensor (may be nullptr on failure).
         * @param[out] out_kv_len Receives the number of cached tokens (optional, may be nullptr).
         * @return true on success, false if indices are out of bounds.
         */
        bool get_kv(int layer, int seq_idx,
                    ITensor **out_k, ITensor **out_v,
                    int *out_kv_len = nullptr) override;

        /** @brief Const overload of get_kv(). */
        bool get_kv(int layer, int seq_idx,
                    const ITensor **out_k, const ITensor **out_v,
                    int *out_kv_len = nullptr) const override;

        /// Bring in ICPUKVCache convenience overloads (seq_idx=0 defaults).
        using ICPUKVCache::get_kv;

        // =====================================================================
        // Individual K / V Accessors (prefer get_kv() for new code)
        // =====================================================================

        /** @brief Get mutable K tensor for a (layer, sequence). Returns nullptr on OOB. */
        ITensor *get_k(int layer, int seq_idx = 0) override;
        /** @brief Get const K tensor for a (layer, sequence). Returns nullptr on OOB. */
        const ITensor *get_k(int layer, int seq_idx = 0) const override;
        /** @brief Get mutable V tensor for a (layer, sequence). Returns nullptr on OOB. */
        ITensor *get_v(int layer, int seq_idx = 0) override;
        /** @brief Get const V tensor for a (layer, sequence). Returns nullptr on OOB. */
        const ITensor *get_v(int layer, int seq_idx = 0) const override;

        // =====================================================================
        // Cache Management
        // =====================================================================

        /** @brief Reset all entries across all layers and sequences (head=0, size=0). */
        void clear() override;

        /**
         * @brief Reset a single (layer, sequence) entry.
         *
         * The underlying tensor memory is not freed — only the ring pointers are reset.
         */
        void clear_sequence(int layer, int seq_idx) override;

        /** @brief Reset all sequences for a given layer. */
        void clear_layer(int layer) override;

        /** @brief Advance ring metadata after an externally managed append/replay. */
        void advanceHead(int layer, int seq_idx, int num_tokens) override;

        /// Bring in ICPUKVCache::clear_sequence(seq_idx) which clears across all layers.
        using ICPUKVCache::clear_sequence;

        // =====================================================================
        // Sharding (Tensor Parallelism) Accessors
        // =====================================================================

        /** @brief True if this cache stores a subset of total KV heads (TP sharding). */
        bool is_sharded() const override { return is_sharded_; }

        /** @brief Number of KV heads stored locally on this rank. */
        int local_n_kv_heads() const override { return local_n_kv_heads_; }

        /** @brief Starting KV head index for this rank's shard (0 if not sharded). */
        int kv_head_start() const override { return kv_head_start_; }

        /** @brief Local KV dimension: `local_n_kv_heads * head_dim`. */
        int local_kv_dim() const override { return kv_dim_; }

        // =====================================================================
        // Additional Metadata
        // =====================================================================

        /** @brief Number of transformer layers (alias for n_layers()). */
        int num_layers() const override { return n_layers_; }

        /** @brief Maximum batch size (number of concurrent sequences). */
        int batch_size() const override { return batch_size_; }

        /** @brief The configured memory layout mode (POSITION_MAJOR or HEAD_MAJOR). */
        KVCacheLayoutMode layout_mode() const override { return layout_mode_; }

        /** @brief Total number of KV heads across all ranks. */
        int n_kv_heads() const override { return n_kv_heads_; }

        // =====================================================================
        // Append Operations
        // =====================================================================

        /**
         * @brief Append new K/V token(s) to the cache for a given layer and sequence.
         *
         * Infers the number of tokens to append from `new_k->shape()[0]`.
         * If the ring buffer is full, the oldest tokens are overwritten.
         *
         * @param layer   Layer index.
         * @param seq_idx Sequence index.
         * @param new_k   Key tensor to append (rows = number of new tokens).
         * @param new_v   Value tensor to append (must match new_k in row count).
         * @return true on success, false on invalid input or type mismatch.
         */
        bool append_kv(int layer, int seq_idx, const TensorBase *new_k, const TensorBase *new_v) override;

        /**
         * @brief Append a specific number of tokens from K/V tensors.
         *
         * Useful when the source tensors have more rows than you want to write
         * (e.g., only appending the first `num_tokens` rows).
         *
         * @param layer      Layer index.
         * @param seq_idx    Sequence index.
         * @param new_k      Key tensor source.
         * @param new_v      Value tensor source.
         * @param num_tokens Number of tokens (rows) to append from the source tensors.
         * @return true on success.
         */
        bool append_kv(int layer, int seq_idx, const TensorBase *new_k, const TensorBase *new_v, int num_tokens) override;

        /// Bring in ICPUKVCache convenience overloads (seq_idx=0 defaults).
        using ICPUKVCache::append_kv;

        /**
         * @brief Append grouped verifier rows using serial-decode ring semantics.
         *
         * This is the CPU implementation of the Phase 9.8 verifier contract.
         * The source may be position-major or head-major, but the destination is
         * always the cache's configured layout.  The method writes all rows in
         * one cache operation and updates ring metadata once, so the stage layer
         * no longer needs to recursively execute one-token append calls.
         */
        bool appendVerifierRowsDecodeEquivalent(int layer,
                                                int seq_idx,
                                                const ITensor *K,
                                                const ITensor *V,
                                                int verifier_rows,
                                                void *gpu_stream = nullptr) override;

        // =====================================================================
        // Batched Gather
        // =====================================================================

        /**
         * @brief Gather K/V data from multiple sequences into contiguous output tensors.
         *
         * This is used for batched attention where the kernel expects all sequences'
         * K/V data packed into a single tensor with shape
         * `[num_sequences * max_kv_len, kv_dim]`.
         *
         * The ring buffer is "unrolled" so that tokens appear in logical order
         * (oldest first) regardless of their physical position in the ring.
         *
         * @param layer          Layer index.
         * @param num_sequences  Number of sequences to gather.
         * @param[out] out_k     Output K tensor (must be pre-allocated, ≥ expected rows).
         * @param[out] out_v     Output V tensor (must be pre-allocated, ≥ expected rows).
         * @param[out] out_kv_lens Per-sequence token counts (resized to num_sequences).
         * @return Maximum kv_len across all sequences, or -1 on error.
         */
        int gather_kv_batched(int layer, int num_sequences, TensorBase *out_k, TensorBase *out_v,
                              std::vector<int> &out_kv_lens) override;

        // =====================================================================
        // Eviction
        // =====================================================================

        /**
         * @brief Evict the oldest tokens from all sequences across all layers.
         *
         * Advances the ring head and decreases size for every (layer, sequence).
         *
         * @param tokens_to_evict Number of oldest tokens to discard per sequence.
         */
        void evict_oldest(int tokens_to_evict) override;

        /**
         * @brief Evict the oldest tokens from a specific sequence across all layers.
         *
         * @param seq_idx         Sequence index to evict from.
         * @param tokens_to_evict Number of oldest tokens to discard.
         */
        void evict_oldest_from_sequence(int seq_idx, int tokens_to_evict) override;

        // =====================================================================
        // Device and Diagnostics
        // =====================================================================

        /** @brief Returns the device where a given layer's KV tensors are stored. */
        DeviceId get_layer_device(int layer) const override;

        /** @brief Total number of tokens evicted since last `reset_eviction_counter()`. */
        int get_total_evicted() const override { return total_evicted_; }

        /** @brief Reset the eviction counter to zero. */
        void reset_eviction_counter() override { total_evicted_ = 0; }

        // =====================================================================
        // Converted KV Access (dequant-on-read with optional fused RoPE)
        // =====================================================================

        /**
         * @brief Get K/V converted to target precision with optional fused RoPE.
         *
         * Manages internal FP32 shadow buffers per (layer, seq_idx). Only newly
         * appended rows are dequantized each call (incremental conversion).
         * When rope->rope_theta > 0, RoPE is fused into K dequantization.
         *
         * Supports all cache precisions: FP32 (passthrough), FP16, BF16, Q8_1,
         * Q16_1, TQ4, TQ8, and split TQ (TQ8 K + TQ4 V).
         */
        bool get_kv_converted(int layer, int seq_idx,
                              ActivationPrecision target,
                              ITensor **out_k, ITensor **out_v,
                              int *out_kv_len = nullptr,
                              const KVReadParams *rope = nullptr) override;

        // =====================================================================
        // Ring Buffer Inspection
        // =====================================================================

        /**
         * @brief Get the physical ring head index for a (layer, sequence) entry.
         *
         * This is the index of the oldest valid token in the pre-allocated tensor.
         * Useful for debugging and testing ring buffer behavior.
         */
        int ring_head(int layer, int seq_idx = 0) const override;

        /**
         * @brief Get the ring buffer occupancy for a (layer, sequence) entry.
         *
         * Equivalent to `get_cached_tokens()` but named to emphasize ring semantics.
         */
        int ring_size(int layer, int seq_idx = 0) const;

        // =====================================================================
        // Logical KV Block IO
        // =====================================================================

        KVCacheLogicalBlockLayout logicalBlockLayout(int global_layer, int token_count) const override;
        KVCacheSequenceState sequenceState(int global_layer, int seq_idx) const override;
        bool exportLogicalBlock(const KVCacheLogicalBlockDescriptor &desc, void *dst_k, void *dst_v) const override;
        bool importLogicalBlock(const KVCacheLogicalBlockDescriptor &desc, const void *src_k, const void *src_v) override;
        bool truncateSequence(int seq_idx, int cached_tokens, void *stream = nullptr) override;

    private:
        // =====================================================================
        // Member Variables
        // =====================================================================

        int n_layers_;                  ///< Number of transformer layers.
        int batch_size_;                ///< Max concurrent sequences.
        int max_seq_len_;               ///< Ring buffer capacity (max tokens per sequence).
        int n_kv_heads_;                ///< Total KV heads across all TP ranks.
        int local_n_kv_heads_;          ///< KV heads stored locally (== n_kv_heads_ if not sharded).
        int kv_head_start_;             ///< Starting head index for this rank's shard.
        int head_dim_;                  ///< Dimension per attention head.
        int kv_dim_;                    ///< local_n_kv_heads_ * head_dim_ (columns per row in POSITION_MAJOR).
        bool is_sharded_;               ///< True if local_n_kv_heads_ != n_kv_heads_.
        int total_evicted_ = 0;         ///< Running count of evicted tokens (for diagnostics).
        bool wrap_warned_ = false;      ///< One-time warning when ring buffer wraps (context full).
        KVCacheLayoutMode layout_mode_; ///< POSITION_MAJOR or HEAD_MAJOR.

        /// @brief 2D array of ring entries: entries_[layer][seq_idx].
        std::vector<std::vector<EntryT>> entries_;

        /// @brief Per-layer device placement (CPU, CUDA:0, etc.).
        std::vector<DeviceId> layer_devices_;

        // =====================================================================
        // FP32 Shadow Buffers (for get_kv_converted)
        // =====================================================================

        /**
         * @brief Per-(layer, seq_idx) shadow state for incremental FP32 conversion.
         *
         * Lazily allocated on first get_kv_converted() call. Tracks how many rows
         * have been converted so far, enabling O(1) incremental update per decode step.
         */
        struct FP32Shadow
        {
            std::unique_ptr<FP32Tensor> K; ///< FP32 K shadow [max_seq_len × kv_dim]
            std::unique_ptr<FP32Tensor> V; ///< FP32 V shadow [max_seq_len × kv_dim]
            int converted_rows = 0;        ///< How many rows have been converted so far
            int last_head = -1;            ///< Last seen ring head (-1 = uninitialised)
        };

        /// @brief 2D array of FP32 shadow entries: fp32_shadows_[layer][seq_idx].
        /// Lazily allocated (empty until first get_kv_converted call).
        mutable std::vector<std::vector<FP32Shadow>> fp32_shadows_;

        /// @brief Lazily allocate and return the FP32Shadow for (layer, seq_idx).
        FP32Shadow &ensureFP32Shadow(int layer, int seq_idx) const;

        /// @brief Invalidate any converted/RoPE shadow state for a cache entry.
        void invalidateFP32Shadow(int layer, int seq_idx) const;

        /// @brief Convert newly appended rows from native precision to FP32 (+ optional RoPE).
        void convertNewRows(int layer, int seq_idx,
                            FP32Shadow &shadow, const EntryT &entry,
                            const KVReadParams *rope) const;

        /// @brief Factory for creating typed tensors with correct device placement.
        std::unique_ptr<TensorFactory> tensor_factory_;

        // =====================================================================
        // Private Helpers
        // =====================================================================

        /**
         * @brief Allocate a single K tensor with the given shape and device.
         *
         * Dispatches to the appropriate TensorFactory method based on the
         * compile-time KPrecision template parameter.
         */
        std::shared_ptr<KTensorT> allocate_k_tensor(size_t rows, size_t cols, DeviceId device);

        /**
         * @brief Allocate a single V tensor with the given shape and device.
         *
         * Dispatches to the appropriate TensorFactory method based on the
         * compile-time VPrecision template parameter.
         */
        std::shared_ptr<VTensorT> allocate_v_tensor(size_t rows, size_t cols, DeviceId device);

        /**
         * @brief Initialize all sequence entries for a given layer.
         *
         * Allocates K and V tensors for each sequence in the batch, with shape
         * determined by the layout mode. Resets head and size to 0.
         *
         * @param layer  Layer index to initialize.
         * @param device Device to allocate the tensors on.
         */
        void initialize_layer(int layer, DeviceId device);

        /**
         * @brief Core append implementation after type-checking the input tensors.
         *
         * Handles both POSITION_MAJOR and HEAD_MAJOR layouts. Writes tokens into
         * the ring buffer and advances head/size as needed.
         *
         * @param layer      Layer index.
         * @param seq_idx    Sequence index.
         * @param new_k      Type-checked K tensor to append.
         * @param new_v      Type-checked V tensor to append.
         * @param num_tokens Number of token rows to write.
         * @return true on success.
         */
        bool append_kv_impl(int layer, int seq_idx, const KTensorT *new_k, const VTensorT *new_v, int num_tokens);

        /**
         * @brief Append tokens from a single source tensor to a single destination
         *        tensor using raw byte copies.
         *
         * This is an alternative simpler append path that works on raw byte rows.
         * Used internally as a fallback. Handles ring wrap-around.
         *
         * @param dst        Destination ring buffer tensor.
         * @param src        Source tensor with new token data.
         * @param entry      Ring buffer entry to update (head/size).
         * @param num_tokens Number of tokens to copy.
         * @return true on success.
         */
        bool append_one_tensor(TensorBase *dst, const TensorBase *src, EntryT &entry, int num_tokens);
    };

    // =========================================================================
    // Convenience Type Aliases
    // =========================================================================

    using CPURingKVCacheFP32 = CPURingKVCache<ActivationPrecision::FP32>;   ///< 32-bit float KV cache.
    using CPURingKVCacheBF16 = CPURingKVCache<ActivationPrecision::BF16>;   ///< BFloat16 KV cache.
    using CPURingKVCacheFP16 = CPURingKVCache<ActivationPrecision::FP16>;   ///< Float16 KV cache.
    using CPURingKVCacheQ8_1 = CPURingKVCache<ActivationPrecision::Q8_1>;   ///< 8-bit quantized KV cache.
    using CPURingKVCacheQ16_1 = CPURingKVCache<ActivationPrecision::Q16_1>; ///< 16-bit quantized KV cache.
    using CPURingKVCacheTQ4 = CPURingKVCache<ActivationPrecision::TQ4>;     ///< TurboQuant 4-bit KV cache.
    using CPURingKVCacheTQ8 = CPURingKVCache<ActivationPrecision::TQ8>;     ///< TurboQuant 8-bit KV cache.
    /// Asymmetric TQ: TQ8 for K (high-fidelity scores), TQ4 for V (graceful degradation).
    using CPURingKVCacheTQ = CPURingKVCache<ActivationPrecision::TQ8, ActivationPrecision::TQ4>;

    // =========================================================================
    // Factory Functions
    // =========================================================================

    /**
     * @brief Create a non-sharded ring KV cache with runtime precision selection.
     *
     * This factory selects the correct template instantiation at runtime based
     * on the `precision` parameter. All layers use the same device.
     *
     * @param precision   Activation precision (FP32, BF16, FP16, Q8_1, Q16_1).
     * @param mpi_ctx     MPI context for tensor allocation.
     * @param n_layers    Number of transformer layers.
     * @param batch_size  Maximum batch size.
     * @param max_seq_len Ring buffer capacity.
     * @param n_kv_heads  Total number of KV heads.
     * @param head_dim    Dimension per head.
     * @param device      Device placement (default: CPU).
     * @param layout_mode Memory layout (default: POSITION_MAJOR).
     * @return Unique pointer to the created cache, or nullptr on unsupported precision.
     */
    std::unique_ptr<ICPUKVCache> createCPURingKVCache(
        ActivationPrecision precision,
        const IMPIContext &mpi_ctx,
        int n_layers, int batch_size, int max_seq_len,
        int n_kv_heads, int head_dim,
        DeviceId device = DeviceId::cpu(),
        KVCacheLayoutMode layout_mode = KVCacheLayoutMode::POSITION_MAJOR);

    /**
     * @brief Create a non-sharded ring KV cache with per-layer device placement.
     *
     * @param precision         Activation precision.
     * @param mpi_ctx           MPI context.
     * @param n_layers          Number of transformer layers.
     * @param batch_size        Maximum batch size.
     * @param max_seq_len       Ring buffer capacity.
     * @param n_kv_heads        Total KV heads.
     * @param head_dim          Dimension per head.
     * @param attention_devices Per-layer device indices (0=CPU, 1+=CUDA N-1).
     * @param layout_mode       Memory layout (default: POSITION_MAJOR).
     * @return Unique pointer to the created cache, or nullptr on unsupported precision.
     */
    std::unique_ptr<ICPUKVCache> createCPURingKVCache(
        ActivationPrecision precision,
        const IMPIContext &mpi_ctx,
        int n_layers, int batch_size, int max_seq_len,
        int n_kv_heads, int head_dim,
        const std::vector<int> &attention_devices,
        KVCacheLayoutMode layout_mode = KVCacheLayoutMode::POSITION_MAJOR);

    /**
     * @brief Create a sharded ring KV cache for tensor-parallel inference.
     *
     * Only allocates storage for `local_n_kv_heads` heads starting at `kv_head_start`.
     *
     * @param precision        Activation precision.
     * @param mpi_ctx          MPI context.
     * @param n_layers         Number of transformer layers.
     * @param batch_size       Maximum batch size.
     * @param max_seq_len      Ring buffer capacity.
     * @param n_kv_heads       Total KV heads across all ranks.
     * @param local_n_kv_heads KV heads for this rank.
     * @param kv_head_start    Starting KV head index for this rank.
     * @param head_dim         Dimension per head.
     * @param device           Device placement (default: CPU).
     * @param layout_mode      Memory layout (default: POSITION_MAJOR).
     * @return Unique pointer to the created cache, or nullptr on unsupported precision.
     */
    std::unique_ptr<ICPUKVCache> createShardedCPURingKVCache(
        ActivationPrecision precision,
        const IMPIContext &mpi_ctx,
        int n_layers, int batch_size, int max_seq_len,
        int n_kv_heads, int local_n_kv_heads, int kv_head_start,
        int head_dim, DeviceId device = DeviceId::cpu(),
        KVCacheLayoutMode layout_mode = KVCacheLayoutMode::POSITION_MAJOR);

    /**
     * @brief Create a sharded ring KV cache with per-layer device placement.
     *
     * Combines tensor parallelism with heterogeneous device assignment.
     *
     * @param precision         Activation precision.
     * @param mpi_ctx           MPI context.
     * @param n_layers          Number of transformer layers.
     * @param batch_size        Maximum batch size.
     * @param max_seq_len       Ring buffer capacity.
     * @param n_kv_heads        Total KV heads across all ranks.
     * @param local_n_kv_heads  KV heads for this rank.
     * @param kv_head_start     Starting KV head index for this rank.
     * @param head_dim          Dimension per head.
     * @param attention_devices Per-layer device indices.
     * @param layout_mode       Memory layout (default: POSITION_MAJOR).
     * @return Unique pointer to the created cache, or nullptr on unsupported precision.
     */
    std::unique_ptr<ICPUKVCache> createShardedCPURingKVCache(
        ActivationPrecision precision,
        const IMPIContext &mpi_ctx,
        int n_layers, int batch_size, int max_seq_len,
        int n_kv_heads, int local_n_kv_heads, int kv_head_start,
        int head_dim,
        const std::vector<int> &attention_devices,
        KVCacheLayoutMode layout_mode = KVCacheLayoutMode::POSITION_MAJOR);

} // namespace llaminar2
