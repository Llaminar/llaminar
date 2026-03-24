/**
 * @file CPUFlashAttentionKernelT.h
 * @brief Flash Attention kernel for CPU, using tiled online softmax.
 *
 * ## What is Flash Attention?
 *
 * Standard attention computes `softmax(Q·K^T / √d) · V` by first materialising
 * the full `seq_len × kv_len` score matrix, running softmax over every row, and
 * then multiplying by V.  For long sequences this score matrix can be enormous
 * (e.g. 4096² × sizeof(float) ≈ 64 MB per head) and thrashes the CPU cache.
 *
 * Flash Attention avoids ever materialising the full score matrix.  Instead it
 * processes K/V in *tiles* of `kv_tile` positions at a time, computing a partial
 * softmax that is corrected "on the fly" as new tiles arrive.  This is called
 * **online softmax** (see Milakov & Gimelshein 2018).  Only a small tile of
 * scores lives in L1/L2 at a time, giving a dramatic cache-locality win.
 *
 * ## Template parameter
 *
 * The kernel is templated on `ActivationPrecision` so the same code serves
 * FP32, BF16, and FP16 builds.  When `Precision ≠ FP32`, the kernel delegates
 * to a `CPUAttentionKernelT<Precision>` fallback that handles non-float paths.
 *
 * ## Integer quantised prefill path (I16/I12)
 *
 * For long prefill sequences the QK dot-products can optionally be computed in
 * integer arithmetic using 16-bit quantisation with a 12-bit value range
 * (qmax ≈ 2047).  When AVX-512 VNNI is available this uses the `VPDPWSSD`
 * instruction which does 32 × int16 multiply-add per cycle.  Two consecutive K
 * rows are packed into a single "pair" buffer so a single VNNI loop produces
 * two dot-products at once.  This is a significant speedup for prefill where
 * `seq_len × kv_len` can be in the millions.
 *
 * ## How this file is organised
 *
 * 1. `detail::CacheAwareFlashKVTilePolicy` – chooses the KV tile size based on
 *    the CPU cache hierarchy and the actual working-set size.
 * 2. `detail::FlashAttentionPrecisionToTensor` – maps `ActivationPrecision`
 *    enum values to concrete tensor classes (FP32Tensor, BF16Tensor, …).
 * 3. `CPUFlashAttentionKernelT<Precision>` – the main kernel class.
 *    - Public API: `compute()`, `compute_batch()`, `compute_decode()`,
 *      `compute_tensor()` – front-end methods matching the `ITensorAttention`
 *      interface.
 *    - Private helpers: SIMD dot-products, quantisation routines, and the
 *      core `compute_flash_fp32()` tiled-softmax implementation.
 *
 * @see CPUAttentionKernelT  The non-flash fallback kernel used for non-FP32
 *                           precisions and for sharded (tensor-parallel) heads.
 * @see ITensorAttention     The polymorphic interface this class implements.
 * @see AttentionCacheConfig  Cache-hierarchy model used by the tile policy.
 */

#pragma once

#include "../../../tensors/TensorKernels.h"
#include "../../../tensors/Tensors.h"
#include "../../../utils/CPUFeatures.h"
#include "../../../utils/DebugEnv.h"
#include "../../../utils/KernelProfiler.h"
#include "../../../utils/OpenMPUtils.h"
#include "../primitives/ActivationTraits.h"
#include "CPUAttentionKernelT.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <type_traits>
#include <vector>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

namespace llaminar2
{
    namespace detail
    {
        /**
         * @brief Cache-aware policy that decides how many KV positions to process per tile.
         *
         * ## Why tile size matters
         *
         * Flash Attention iterates over KV positions in tiles.  Each tile loads
         * `kv_tile` rows of K *and* V into registers / L1 cache, computes the
         * partial QK scores, runs partial softmax, and accumulates into the
         * output.  A tile that is too large spills to L2/L3 and loses cache
         * locality; a tile that is too small pays excessive loop overhead.
         *
         * This policy inspects the CPU cache hierarchy (L1/L2/L3 sizes fetched
         * once from `AttentionCacheConfig`) and picks the largest power-of-two
         * tile that keeps the per-tile working set within a target fraction of
         * the relevant cache level.
         *
         * ## Template parameters
         *
         * @tparam DecodeTargetPct    Target percentage of the cache to use during
         *                            decode (single-token generation).  Default 50%.
         * @tparam PrefillTargetPct   Target percentage during prefill (multi-token).
         *                            Default 37% – lower because prefill has more
         *                            concurrent data (Q rows × K tile).
         * @tparam DecodeMinTile      Minimum tile size for decode.  Must be ≥ 1.
         * @tparam DecodeMaxTile      Maximum tile size for decode.
         * @tparam PrefillMinTile     Minimum tile size for prefill.
         * @tparam PrefillMaxTile     Maximum tile size for prefill.
         *
         * ## Environment overrides
         *
         * The tile size can be overridden at runtime with the debug environment
         * variables `LLAMINAR_ATTN_FLASH_KV_TILE_DECODE` and
         * `LLAMINAR_ATTN_FLASH_KV_TILE_PREFILL` (see `debugEnv()`).
         */
        template <int DecodeTargetPct,
                  int PrefillTargetPct,
                  int DecodeMinTile,
                  int DecodeMaxTile,
                  int PrefillMinTile,
                  int PrefillMaxTile>
        struct CacheAwareFlashKVTilePolicy
        {
            static_assert(DecodeTargetPct > 0 && DecodeTargetPct <= 100, "DecodeTargetPct must be in (0, 100]");
            static_assert(PrefillTargetPct > 0 && PrefillTargetPct <= 100, "PrefillTargetPct must be in (0, 100]");

            /**
             * @brief Select the optimal KV tile size for the given attention parameters.
             *
             * The algorithm works as follows:
             * 1. Check for a user-supplied override via the debug environment.
             * 2. Build an `AttentionCacheConfig` to query the CPU cache hierarchy.
             * 3. Compute the bytes needed per KV position in a tile
             *    (one K row + one V row + one score float).
             * 4. Pick the cache level that best matches the work size (SMALL → L1,
             *    LARGE → L2, XL → L3/8).
             * 5. Divide the usable cache budget by the per-position cost to get
             *    the raw tile size, then clamp to the nearest power-of-two within
             *    the [min, max] range.
             *
             * @param head_dim    Dimension per attention head (e.g. 64 or 128).
             * @param n_kv_heads  Number of key/value heads (for GQA/MQA).
             * @param kv_len      Current key/value sequence length.
             * @param is_decode   True when generating one token at a time (decode),
             *                    false during prompt prefill.
             * @return Tile size as a power of two, in range [MinTile, MaxTile].
             */
            static int choose(int head_dim, int n_kv_heads, int kv_len, bool is_decode)
            {
                const int override_tile = tile_override(is_decode);
                if (override_tile > 0)
                {
                    return clamp_to_power2(override_tile,
                                           is_decode ? DecodeMinTile : PrefillMinTile,
                                           is_decode ? DecodeMaxTile : PrefillMaxTile);
                }

                AttentionCacheConfig cfg(head_dim, n_kv_heads, kv_len);
                const size_t kv_row_bytes = static_cast<size_t>(head_dim) * sizeof(float) * 2ULL;
                const size_t score_bytes = sizeof(float);
                const size_t bytes_per_kv_position = kv_row_bytes + score_bytes;

                size_t target_cache_bytes = 0;
                switch (cfg.work_size())
                {
                case AttentionWorkSize::SMALL:
                    target_cache_bytes = static_cast<size_t>(cfg.l1_size);
                    break;
                case AttentionWorkSize::LARGE:
                    target_cache_bytes = static_cast<size_t>(cfg.l2_size);
                    break;
                case AttentionWorkSize::XL:
                default:
                    target_cache_bytes = std::max<size_t>(cfg.l3_size / 8, cfg.l2_size);
                    break;
                }

                const int target_pct = is_decode ? DecodeTargetPct : PrefillTargetPct;
                target_cache_bytes = (target_cache_bytes * static_cast<size_t>(target_pct)) / 100ULL;

                size_t raw_tile = target_cache_bytes > 0
                                      ? (target_cache_bytes / std::max<size_t>(1, bytes_per_kv_position))
                                      : static_cast<size_t>(is_decode ? DecodeMinTile : PrefillMinTile);

                if (!is_decode && cfg.prefer_kv8_tile())
                {
                    raw_tile = std::max<size_t>(raw_tile, 8);
                }

                const int min_tile = is_decode ? DecodeMinTile : PrefillMinTile;
                const int max_tile = is_decode ? DecodeMaxTile : PrefillMaxTile;
                return clamp_to_power2(static_cast<int>(raw_tile), min_tile, max_tile);
            }

        private:
            /**
             * @brief Round a value down to the nearest power-of-two within [min_tile, max_tile].
             *
             * The flash-attention loop benefits from power-of-two tile sizes
             * because the inner loops can be fully unrolled by the compiler and
             * SIMD widths (16 floats for AVX-512) divide evenly.
             *
             * @param value     Raw (non-power-of-two) tile size estimate.
             * @param min_tile  Floor – never return less than this.
             * @param max_tile  Ceiling – never return more than this.
             * @return The clamped power-of-two tile size.
             */
            static int clamp_to_power2(int value, int min_tile, int max_tile)
            {
                if (value <= 0)
                {
                    value = min_tile;
                }

                int p2 = 1;
                while ((p2 << 1) > 0 && (p2 << 1) <= value)
                {
                    p2 <<= 1;
                }

                if (p2 < min_tile)
                {
                    p2 = min_tile;
                }
                if (p2 > max_tile)
                {
                    p2 = max_tile;
                }
                return std::max(1, p2);
            }

            /**
             * @brief Check for a user-supplied tile-size override from the debug environment.
             *
             * Returns 0 (meaning "no override") when the relevant env var is unset.
             *
             * @param is_decode  Whether we are in decode (true) or prefill (false).
             * @return The override value, or 0 if none was set.
             */
            static int tile_override(bool is_decode)
            {
                const auto &env = debugEnv();
                const int decode_override = env.attention.flash_kv_tile_decode;
                const int prefill_override = env.attention.flash_kv_tile_prefill;
                return is_decode ? decode_override : prefill_override;
            }
        };

        /**
         * @brief The default tile-size policy used by CPUFlashAttentionKernelT.
         *
         * Parameters:
         * - Decode: target 50% of cache, tiles in [8, 32].
         * - Prefill: target 37% of cache, tiles in [4, 32].
         */
        using DefaultFlashKVTilePolicy = CacheAwareFlashKVTilePolicy<50, 37, 8, 32, 4, 32>;

        /**
         * @brief Compile-time map from ActivationPrecision → concrete tensor type.
         *
         * The flash attention kernel needs to know which tensor class
         * corresponds to each precision enum value so it can declare typed
         * aliases.  Each specialisation below simply defines a `Type` member.
         *
         * Example:
         * @code
         * using TensorT = FlashAttentionPrecisionToTensor<ActivationPrecision::FP32>::Type;
         * // TensorT == FP32Tensor
         * @endcode
         */
        template <ActivationPrecision P>
        struct FlashAttentionPrecisionToTensor;

        template <>
        struct FlashAttentionPrecisionToTensor<ActivationPrecision::FP32>
        {
            using Type = FP32Tensor;
        };

        template <>
        struct FlashAttentionPrecisionToTensor<ActivationPrecision::BF16>
        {
            using Type = BF16Tensor;
        };

        template <>
        struct FlashAttentionPrecisionToTensor<ActivationPrecision::FP16>
        {
            using Type = FP16Tensor;
        };
    }

    /**
     * @brief CPU Flash Attention kernel with tiled online softmax.
     *
     * This class implements the `ITensorAttention` interface using an
     * *online-softmax flash-attention* algorithm on the CPU.  It is the
     * primary attention kernel used during CPU inference in Llaminar V2.
     *
     * ## How online softmax works (simplified)
     *
     * Traditional softmax requires two passes over the score vector:
     *   1. Find the maximum (for numerical stability).
     *   2. Compute `exp(s - max)` and sum, then normalise.
     *
     * Online softmax fuses both passes into a single streaming pass by
     * maintaining a *running maximum* (`running_m`) and a *running
     * denominator* (`running_l`).  When a new tile of KV positions arrives
     * with a new local maximum, the previously accumulated output and
     * denominator are *rescaled* by `exp(old_max − new_max)` before the new
     * tile's contributions are added.  At the end, the accumulated output is
     * divided by `running_l` to produce the final normalised result.
     *
     * ## Precision handling
     *
     * When `Precision == FP32`, all work is done directly in this kernel.
     * For BF16 and FP16, the kernel delegates to the `CPUAttentionKernelT`
     * fallback which handles mixed-precision conversions.  This delegation
     * is resolved entirely at compile time via `if constexpr`.
     *
     * ## Thread safety
     *
     * All parallel work is wrapped in `OMP_WORKSHARE_REGION()` so the kernel
     * can participate in an outer parallel region without redundant thread
     * fork/join overhead (see copilot-instructions.md § OpenMP).
     *
     * @tparam Precision  The activation storage format.  Currently FP32 is
     *                    fully optimised; BF16/FP16 fall back to the non-flash
     *                    implementation.
     *
     * @see CPUAttentionKernelT  The fallback kernel for non-FP32 precisions.
     * @see CacheAwareFlashKVTilePolicy  Tile-size selection logic.
     */
    template <ActivationPrecision Precision>
    class CPUFlashAttentionKernelT : public ITensorAttention
    {
    public:
        /** @brief The concrete tensor class for this precision (e.g. FP32Tensor). */
        using TensorT = typename detail::FlashAttentionPrecisionToTensor<Precision>::Type;
        /** @brief The scalar element type for this precision (e.g. float). */
        using ElementType = typename primitives::ActivationTraits<TensorT>::ElementType;

        CPUFlashAttentionKernelT() = default;
        ~CPUFlashAttentionKernelT() override = default;

        /**
         * @brief Reports whether this kernel can run on a given device.
         *
         * Flash attention is a CPU-only kernel, so it only supports device -1
         * (the host / CPU device sentinel).
         *
         * @param device_idx  Device index to check (-1 = CPU).
         * @return true if `device_idx == -1`, false otherwise.
         */
        bool supports_device(int device_idx) const override
        {
            return device_idx == -1;
        }

        /**
         * @brief Compute attention for a single-batch, equal-length Q/KV input.
         *
         * This is the simplest entry point: `seq_len == kv_len`, batch size 1.
         * Raw float pointers are expected in row-major layout:
         *   - Q, K, V: `[seq_len, n_heads/n_kv_heads * head_dim]`
         *   - output:   `[seq_len, n_heads * head_dim]`
         *
         * If the precision is not FP32, the call is forwarded to the
         * `CPUAttentionKernelT` fallback.
         *
         * @param Q               Query matrix, shape [seq_len, n_heads * head_dim].
         * @param K               Key matrix,   shape [seq_len, n_kv_heads * head_dim].
         * @param V               Value matrix, shape [seq_len, n_kv_heads * head_dim].
         * @param output          Output buffer, same shape as Q.
         * @param seq_len         Number of query (and key/value) positions.
         * @param n_heads         Number of query attention heads.
         * @param n_kv_heads      Number of key/value heads (≤ n_heads; GQA when < n_heads).
         * @param head_dim        Dimension per head (e.g. 64 or 128).
         * @param causal          If true, positions can only attend to earlier positions.
         * @param window_size     Sliding-window size (-1 = unlimited).
         * @param workspace_scores  (unused) Pre-allocated score workspace.
         * @param workspace_buffer  (unused) Pre-allocated scratch buffer.
         * @param workspace_context (unused) Pre-allocated context buffer.
         * @param workspace_mask  Optional additive mask [seq_len, kv_len].
         * @param use_bf16        (unused) BF16 hint – precision is selected at compile time.
         * @param mpi_ctx         (unused) MPI context.
         * @param device_idx      (unused) Device index.
         * @return true on success, false on failure.
         */
        bool compute(
            const float *Q, const float *K, const float *V, float *output,
            int seq_len, int n_heads, int n_kv_heads, int head_dim,
            bool causal = false,
            int window_size = -1,
            TensorBase *workspace_scores = nullptr,
            TensorBase *workspace_buffer = nullptr,
            TensorBase *workspace_context = nullptr,
            TensorBase *workspace_mask = nullptr,
            bool use_bf16 = false,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1)
        {
            (void)workspace_scores;
            (void)workspace_buffer;
            (void)workspace_context;
            (void)use_bf16;
            (void)mpi_ctx;
            (void)device_idx;

            if constexpr (!std::is_same_v<ElementType, float>)
            {
                return fallback_.compute(Q, K, V, output,
                                         seq_len, n_heads, n_kv_heads, head_dim,
                                         causal, window_size,
                                         workspace_scores, workspace_buffer,
                                         workspace_context, workspace_mask,
                                         use_bf16, mpi_ctx, device_idx);
            }

            const float *mask = workspace_mask ? workspace_mask->data() : nullptr;
            return compute_flash_fp32(Q, K, V, output,
                                      seq_len, seq_len,
                                      n_heads, n_kv_heads, head_dim,
                                      causal, window_size, 0,
                                      mask);
        }

        /**
         * @brief Compute attention for multiple independent sequences (batch > 1).
         *
         * Each batch element is processed sequentially by calling
         * `compute_flash_fp32()` with the appropriate pointer offsets.
         * Parallelism happens *within* each batch element (over heads).
         *
         * @param Q           Query tensor, shape   [batch, seq_len, n_heads * head_dim].
         * @param K           Key tensor,   shape   [batch, seq_len, n_kv_heads * head_dim].
         * @param V           Value tensor, shape   [batch, seq_len, n_kv_heads * head_dim].
         * @param output      Output tensor, shape  [batch, seq_len, n_heads * head_dim].
         * @param batch_size  Number of independent sequences.
         * @param seq_len     Positions per sequence (same for Q and KV here).
         * @param n_heads     Number of query heads.
         * @param n_kv_heads  Number of key/value heads.
         * @param head_dim    Dimension per head.
         * @param causal      Apply causal masking.
         * @param window_size Sliding-window size (-1 = unlimited).
         * @return true on success, false if any batch element fails.
         */
        bool compute_batch(
            const float *Q, const float *K, const float *V, float *output,
            int batch_size, int seq_len, int n_heads, int n_kv_heads, int head_dim,
            bool causal = false,
            int window_size = -1,
            TensorBase *workspace_scores = nullptr,
            TensorBase *workspace_buffer = nullptr,
            TensorBase *workspace_context = nullptr,
            TensorBase *workspace_mask = nullptr,
            bool use_bf16 = false,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1)
        {
            (void)workspace_buffer;
            (void)workspace_context;

            if constexpr (!std::is_same_v<ElementType, float>)
            {
                return fallback_.compute_batch(Q, K, V, output,
                                               batch_size, seq_len, n_heads, n_kv_heads, head_dim,
                                               causal, window_size,
                                               workspace_scores, workspace_buffer,
                                               workspace_context, workspace_mask,
                                               use_bf16, mpi_ctx, device_idx);
            }

            const size_t q_stride = static_cast<size_t>(seq_len) * static_cast<size_t>(n_heads) * static_cast<size_t>(head_dim);
            const size_t kv_stride = static_cast<size_t>(seq_len) * static_cast<size_t>(n_kv_heads) * static_cast<size_t>(head_dim);
            const float *mask = workspace_mask ? workspace_mask->data() : nullptr;

            for (int b = 0; b < batch_size; ++b)
            {
                const float *Q_b = Q + static_cast<size_t>(b) * q_stride;
                const float *K_b = K + static_cast<size_t>(b) * kv_stride;
                const float *V_b = V + static_cast<size_t>(b) * kv_stride;
                float *O_b = output + static_cast<size_t>(b) * q_stride;
                if (!compute_flash_fp32(Q_b, K_b, V_b, O_b,
                                        seq_len, seq_len,
                                        n_heads, n_kv_heads, head_dim,
                                        causal, window_size, 0,
                                        mask))
                {
                    return false;
                }
            }
            return true;
        }

        /**
         * @brief Compute attention for autoregressive decode (seq_len ≠ kv_len).
         *
         * During decode, typically `seq_len == 1` (the single new token) while
         * `kv_len` is the accumulated context length from the KV cache.  The
         * `position_offset` parameter tells the causal-mask logic what absolute
         * position the first query token sits at (so the mask is
         * `q_abs = position_offset + q_pos`).
         *
         * @param Q               Query, shape [seq_len, n_heads * head_dim].
         * @param K               Keys from KV cache, shape [kv_len, n_kv_heads * head_dim].
         * @param V               Values from KV cache, shape [kv_len, n_kv_heads * head_dim].
         * @param output          Output, shape [seq_len, n_heads * head_dim].
         * @param seq_len         Number of query positions (usually 1 in decode).
         * @param kv_len          Number of cached key/value positions.
         * @param n_heads         Number of query heads.
         * @param n_kv_heads      Number of KV heads.
         * @param head_dim        Dimension per head.
         * @param causal          Apply causal masking (default true for decode).
         * @param position_offset Absolute position of the first query token.
         * @return true on success.
         */
        bool compute_decode(
            const float *Q, const float *K, const float *V, float *output,
            int seq_len, int kv_len, int n_heads, int n_kv_heads, int head_dim,
            bool causal = true,
            int position_offset = 0)
        {
            if constexpr (!std::is_same_v<ElementType, float>)
            {
                return fallback_.compute_decode(Q, K, V, output,
                                                seq_len, kv_len, n_heads, n_kv_heads, head_dim,
                                                causal, position_offset);
            }

            return compute_flash_fp32(Q, K, V, output,
                                      seq_len, kv_len,
                                      n_heads, n_kv_heads, head_dim,
                                      causal, -1, position_offset,
                                      nullptr);
        }

        /**
         * @brief High-level tensor-based entry point used by compute stages.
         *
         * This method accepts polymorphic `ITensor` pointers (the common
         * interface used by `DeviceGraphExecutor` and the compute-stage
         * framework) and routes to the appropriate raw-pointer method.
         *
         * ### Fallback conditions
         *
         * The flash path is used only when *all* of the following hold:
         * - Precision is FP32.
         * - No head sharding (`head_start == 0`, `local_n_heads == -1`).
         * - Q and output are FP32 tensors.
         * - Raw `fp32_data()` pointers are obtainable.
         *
         * Otherwise, the call is delegated to the `CPUAttentionKernelT`
         * fallback which handles mixed precision, sharded heads, and
         * quantised KV caches.
         *
         * @param Q                 Query tensor.
         * @param K                 Key tensor.
         * @param V                 Value tensor.
         * @param output            Output tensor (written in-place).
         * @param batch_size        Batch dimension.
         * @param seq_len           Query sequence length.
         * @param kv_len            Key/Value sequence length.
         * @param n_heads           Total number of query heads.
         * @param n_kv_heads        Total number of KV heads.
         * @param head_dim          Dimension per head.
         * @param causal            Apply causal masking.
         * @param window_size       Sliding-window size (-1 = unlimited).
         * @param workspace_scores  Optional pre-allocated score workspace.
         * @param workspace_mask    Optional additive mask tensor.
         * @param mpi_ctx           MPI context (unused on this path).
         * @param device_idx        Device index (unused, CPU only).
         * @param head_start        First head index when sharded (0 = unsharded).
         * @param local_n_heads     Local query heads (-1 = all).
         * @param local_n_kv_heads  Local KV heads (-1 = all).
         * @return true on success.
         */
        bool compute_tensor(
            const ITensor *Q,
            const ITensor *K,
            const ITensor *V,
            ITensor *output,
            int batch_size,
            int seq_len,
            int kv_len,
            int n_heads,
            int n_kv_heads,
            int head_dim,
            bool causal = false,
            int window_size = -1,
            ITensor *workspace_scores = nullptr,
            ITensor *workspace_mask = nullptr,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1,
            int head_start = 0,
            int local_n_heads = -1,
            int local_n_kv_heads = -1) override
        {
            if constexpr (!std::is_same_v<ElementType, float>)
            {
                return fallback_.compute_tensor(Q, K, V, output,
                                                batch_size, seq_len, kv_len,
                                                n_heads, n_kv_heads, head_dim,
                                                causal, window_size,
                                                workspace_scores, workspace_mask,
                                                mpi_ctx, device_idx,
                                                head_start, local_n_heads, local_n_kv_heads);
            }

            if (head_start != 0 || local_n_heads != -1 || local_n_kv_heads != -1)
            {
                return fallback_.compute_tensor(Q, K, V, output,
                                                batch_size, seq_len, kv_len,
                                                n_heads, n_kv_heads, head_dim,
                                                causal, window_size,
                                                workspace_scores, workspace_mask,
                                                mpi_ctx, device_idx,
                                                head_start, local_n_heads, local_n_kv_heads);
            }

            if (Q->native_type() != TensorType::FP32 ||
                output->native_type() != TensorType::FP32)
            {
                return fallback_.compute_tensor(Q, K, V, output,
                                                batch_size, seq_len, kv_len,
                                                n_heads, n_kv_heads, head_dim,
                                                causal, window_size,
                                                workspace_scores, workspace_mask,
                                                mpi_ctx, device_idx,
                                                head_start, local_n_heads, local_n_kv_heads);
            }

            const auto *Q_base = dynamic_cast<const TensorBase *>(Q);
            const auto *K_base = dynamic_cast<const TensorBase *>(K);
            const auto *V_base = dynamic_cast<const TensorBase *>(V);
            auto *O_base = dynamic_cast<TensorBase *>(output);
            auto *scores_base = dynamic_cast<TensorBase *>(workspace_scores);
            auto *mask_base = dynamic_cast<TensorBase *>(workspace_mask);

            if (!Q_base || !K_base || !V_base || !O_base)
            {
                return false;
            }

            // ---------------------------------------------------------------
            // FP16 KV decode fast-path: read FP16 directly, no FP32 buffer.
            // This halves KV memory bandwidth and eliminates DRAM bank
            // disturbance that degrades subsequent GEMM at long context.
            // ---------------------------------------------------------------
#if defined(__AVX512F__) && defined(__F16C__)
            if (kv_len != seq_len && batch_size == 1 &&
                K->native_type() == TensorType::FP16 &&
                V->native_type() == TensorType::FP16)
            {
                const auto *K_fp16 = dynamic_cast<const FP16Tensor *>(K);
                const auto *V_fp16 = dynamic_cast<const FP16Tensor *>(V);
                if (K_fp16 && V_fp16)
                {
                    const float *Q_ptr = Q_base->fp32_data();
                    float *O_ptr = O_base->mutable_data();
                    if (Q_ptr && O_ptr)
                    {
                        const int position_offset = (kv_len > seq_len)
                                                        ? (kv_len - seq_len)
                                                        : 0;
                        return compute_decode_fp16kv(
                            Q_ptr,
                            K_fp16->typed_data(),
                            V_fp16->typed_data(),
                            O_ptr,
                            kv_len, n_heads, n_kv_heads, head_dim,
                            causal, position_offset);
                    }
                }
            }
#endif

            // ---------------------------------------------------------------
            // Q16_1 KV decode fast-path: VNNI int16 QK dot product + int16 V.
            // Halves KV bandwidth (int16 values are 2 bytes vs 4 bytes FP32)
            // and leverages VPDPWSSD for compute-free QK scoring.
            // ---------------------------------------------------------------
#if defined(__AVX512F__) && defined(__AVX512VNNI__)
            if (K->native_type() == TensorType::Q16_1 &&
                V->native_type() == TensorType::Q16_1 &&
                batch_size == 1)
            {
                const auto *K_q16 = dynamic_cast<const Q16_1Tensor *>(K);
                const auto *V_q16 = dynamic_cast<const Q16_1Tensor *>(V);
                if (K_q16 && V_q16)
                {
                    const float *Q_ptr = Q_base->fp32_data();
                    float *O_ptr = O_base->mutable_data();
                    if (Q_ptr && O_ptr)
                    {
                        const int position_offset = (kv_len > seq_len)
                                                        ? (kv_len - seq_len)
                                                        : 0;
                        if (kv_len != seq_len)
                        {
                            return compute_decode_q16kv(
                                Q_ptr, K_q16, V_q16, O_ptr,
                                kv_len, n_heads, n_kv_heads, head_dim,
                                causal, position_offset);
                        }
                        else
                        {
                            return compute_prefill_q16kv(
                                Q_ptr, K_q16, V_q16, O_ptr,
                                seq_len, kv_len, n_heads, n_kv_heads, head_dim,
                                causal, window_size, position_offset);
                        }
                    }
                }
            }
#endif

            const float *Q_ptr = Q_base->fp32_data();
            const float *K_ptr = K_base->fp32_data();
            const float *V_ptr = V_base->fp32_data();
            float *O_ptr = O_base->mutable_data();

            if (!Q_ptr || !K_ptr || !V_ptr || !O_ptr)
            {
                return fallback_.compute_tensor(Q, K, V, output,
                                                batch_size, seq_len, kv_len,
                                                n_heads, n_kv_heads, head_dim,
                                                causal, window_size,
                                                workspace_scores, workspace_mask,
                                                mpi_ctx, device_idx,
                                                head_start, local_n_heads, local_n_kv_heads);
            }

            if (batch_size > 1)
            {
                return compute_batch(Q_ptr, K_ptr, V_ptr, O_ptr,
                                     batch_size, seq_len, n_heads, n_kv_heads, head_dim,
                                     causal, window_size,
                                     scores_base, nullptr, nullptr, mask_base,
                                     false, mpi_ctx, device_idx);
            }

            if (kv_len != seq_len)
            {
                const int position_offset = (kv_len > seq_len) ? (kv_len - seq_len) : 0;
                return compute_decode(Q_ptr, K_ptr, V_ptr, O_ptr,
                                      seq_len, kv_len, n_heads, n_kv_heads, head_dim,
                                      causal, position_offset);
            }

            return compute(Q_ptr, K_ptr, V_ptr, O_ptr,
                           seq_len, n_heads, n_kv_heads, head_dim,
                           causal, window_size,
                           scores_base, nullptr, nullptr, mask_base,
                           false, mpi_ctx, device_idx);
        }

        /**
         * @brief Return metadata describing this kernel's I/O for the snapshot framework.
         *
         * The snapshot/dump system uses this to know what tensors to capture
         * (Q, K, V, output) and what scalar parameters exist (seq_len, etc.)
         * when `LLAMINAR_STAGE_DUMP_ENABLED=1`.
         */
        KernelSnapshotInfo getKernelSnapshotInfo() const override
        {
            return KernelSnapshotInfo::attention()
                .withInput("Q", "query tensor [seq_len, n_heads * head_dim]", KernelBufferDtype::FP32)
                .withInput("K", "key tensor [kv_len, n_kv_heads * head_dim]", KernelBufferDtype::FP32)
                .withInput("V", "value tensor [kv_len, n_kv_heads * head_dim]", KernelBufferDtype::FP32)
                .withOutput("output", "attention output [seq_len, n_heads * head_dim]", KernelBufferDtype::FP32)
                .withScalar("seq_len", "query sequence length", KernelBufferDtype::INT32)
                .withScalar("kv_len", "key/value sequence length", KernelBufferDtype::INT32)
                .withScalar("n_heads", "number of query heads", KernelBufferDtype::INT32)
                .withScalar("n_kv_heads", "number of key/value heads", KernelBufferDtype::INT32)
                .withScalar("head_dim", "dimension per head", KernelBufferDtype::INT32)
                .withScalar("causal", "apply causal masking", KernelBufferDtype::INT32);
        }

    private:
        /**
         * @brief Non-flash fallback kernel.
         *
         * Used for non-FP32 precisions and for sharded-head (tensor parallel)
         * configurations that the flash path does not yet handle.
         */
        CPUAttentionKernelT<Precision> fallback_;

        // -----------------------------------------------------------------
        // Dot-product helpers
        // -----------------------------------------------------------------

        /**
         * @brief Scalar (no-SIMD) dot product of two FP32 vectors.
         *
         * Used as a fallback when AVX-512 is not available at compile time.
         *
         * @param a  First input vector.
         * @param b  Second input vector.
         * @param n  Number of elements.
         * @return   The dot product `Σ a[i]*b[i]`.
         */
        static float dot_fp32_scalar(const float *a, const float *b, int n)
        {
            float sum = 0.0f;
            for (int i = 0; i < n; ++i)
            {
                sum += a[i] * b[i];
            }
            return sum;
        }

        /**
         * @brief AVX-512 vectorised dot product of two FP32 vectors.
         *
         * Processes 16 floats per iteration using `_mm512_fmadd_ps`, with a
         * scalar tail loop for the remaining elements.  Falls back to the
         * scalar version if AVX-512 is not available at compile time.
         *
         * @param a  First input vector.
         * @param b  Second input vector.
         * @param n  Number of elements.
         * @return   The dot product `Σ a[i]*b[i]`.
         */
        static float dot_fp32_avx512(const float *a, const float *b, int n)
        {
#if defined(__AVX512F__)
            __m512 acc = _mm512_setzero_ps();
            int i = 0;
            for (; i + 15 < n; i += 16)
            {
                __m512 va = _mm512_loadu_ps(a + i);
                __m512 vb = _mm512_loadu_ps(b + i);
                acc = _mm512_fmadd_ps(va, vb, acc);
            }
            float sum = _mm512_reduce_add_ps(acc);
            for (; i < n; ++i)
            {
                sum += a[i] * b[i];
            }
            return sum;
#else
            return dot_fp32_scalar(a, b, n);
#endif
        }

#if defined(__AVX512F__)
        /**
         * @brief Fast vectorised exp() for 16 FP32 values using AVX-512.
         *
         * Uses range reduction (x = n*ln2 + f) and a 5th-order polynomial
         * approximation of exp(f) on [-ln2/2, ln2/2].  Inputs below -88.7
         * are clamped to produce 0 (avoids NaN from -inf).
         * Accuracy: <1 ULP for |x| < 87.
         */
        static inline __m512 fast_exp_avx512(__m512 x)
        {
            // Clamp to avoid NaN from -inf inputs (exp(-88.7) ≈ 0)
            x = _mm512_max_ps(x, _mm512_set1_ps(-88.722839f));

            const __m512 log2e = _mm512_set1_ps(1.4426950408889634f);
            const __m512 ln2_hi = _mm512_set1_ps(0.693145751953125f);
            const __m512 ln2_lo = _mm512_set1_ps(1.42860682030941723212e-06f);

            // Range reduction: n = round(x / ln2), f = x - n*ln2
            __m512 t = _mm512_mul_ps(x, log2e);
            __m512 n = _mm512_roundscale_ps(t, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            __m512 f = _mm512_fnmadd_ps(n, ln2_hi, x);
            f = _mm512_fnmadd_ps(n, ln2_lo, f);

            // Horner evaluation of exp(f) ≈ 1 + f + f²/2 + f³/6 + f⁴/24 + f⁵/120
            __m512 p = _mm512_set1_ps(8.36564774e-03f);                 // ≈ 1/120
            p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(4.16689515e-02f)); // ≈ 1/24
            p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(1.66666716e-01f)); // ≈ 1/6
            p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(4.99999851e-01f)); // ≈ 1/2
            p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(1.00000000e+00f)); // 1
            p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(1.00000000e+00f)); // 1

            // Construct 2^n by setting the IEEE754 exponent field
            // Clamp ni >= 0 so that extreme negative inputs (masked -inf
            // positions clamped to -88.7) produce 0 rather than NaN.
            __m512i ni = _mm512_cvtps_epi32(n);
            ni = _mm512_add_epi32(ni, _mm512_set1_epi32(127));
            ni = _mm512_max_epi32(ni, _mm512_setzero_si512());
            ni = _mm512_slli_epi32(ni, 23);
            return _mm512_mul_ps(p, _mm512_castsi512_ps(ni));
        }

        /**
         * @brief Compute 4 dot products Q·K[0..3] simultaneously for ILP.
         *
         * By running 4 independent FMA accumulator chains, we keep both
         * FMA ports busy on Cascade Lake (2 FMA/cycle, 4-cycle latency).
         * Throughput: ~13 cycles per K-row vs ~32 for the single-accumulator
         * dot_fp32_avx512 path (limited by FMA latency).
         */
        static void dot_fp32_avx512_4row(
            const float *q,
            const float *k0, const float *k1, const float *k2, const float *k3,
            int head_dim,
            float &s0, float &s1, float &s2, float &s3)
        {
            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();
            __m512 acc2 = _mm512_setzero_ps();
            __m512 acc3 = _mm512_setzero_ps();

            int d = 0;
            for (; d + 15 < head_dim; d += 16)
            {
                __m512 vq = _mm512_loadu_ps(q + d);
                acc0 = _mm512_fmadd_ps(vq, _mm512_loadu_ps(k0 + d), acc0);
                acc1 = _mm512_fmadd_ps(vq, _mm512_loadu_ps(k1 + d), acc1);
                acc2 = _mm512_fmadd_ps(vq, _mm512_loadu_ps(k2 + d), acc2);
                acc3 = _mm512_fmadd_ps(vq, _mm512_loadu_ps(k3 + d), acc3);
            }

            s0 = _mm512_reduce_add_ps(acc0);
            s1 = _mm512_reduce_add_ps(acc1);
            s2 = _mm512_reduce_add_ps(acc2);
            s3 = _mm512_reduce_add_ps(acc3);

            for (; d < head_dim; ++d)
            {
                float qd = q[d];
                s0 += qd * k0[d];
                s1 += qd * k1[d];
                s2 += qd * k2[d];
                s3 += qd * k3[d];
            }
        }

        // =================================================================
        // FP16 KV helpers — load FP16, convert to FP32, compute
        // =================================================================

        /**
         * @brief Dot product of FP32 query × FP16 key with on-the-fly conversion.
         *
         * Reads 16 FP16 values at a time (32 bytes), converts to FP32 via
         * vcvtph2ps, then FMA-accumulates against the FP32 query vector.
         * Half the KV memory bandwidth vs the FP32 path.
         */
        static float dot_fp16_avx512(const float *q, const uint16_t *k, int head_dim)
        {
            __m512 acc = _mm512_setzero_ps();
            int d = 0;
            for (; d + 15 < head_dim; d += 16)
            {
                __m512 vq = _mm512_loadu_ps(q + d);
                __m256i k16 = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i *>(k + d));
                __m512 vk = _mm512_cvtph_ps(k16);
                acc = _mm512_fmadd_ps(vq, vk, acc);
            }
            float sum = _mm512_reduce_add_ps(acc);
            for (; d < head_dim; ++d)
            {
                // Scalar FP16→FP32 conversion for tail elements
                __m128i h = _mm_set1_epi16(static_cast<short>(k[d]));
                float kf = _mm_cvtss_f32(_mm_cvtph_ps(h));
                sum += q[d] * kf;
            }
            return sum;
        }

        /**
         * @brief 4-row batched dot product: FP32 Q × FP16 K for ILP.
         */
        static void dot_fp16_avx512_4row(
            const float *q,
            const uint16_t *k0, const uint16_t *k1,
            const uint16_t *k2, const uint16_t *k3,
            int head_dim,
            float &s0, float &s1, float &s2, float &s3)
        {
            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();
            __m512 acc2 = _mm512_setzero_ps();
            __m512 acc3 = _mm512_setzero_ps();

            int d = 0;
            for (; d + 15 < head_dim; d += 16)
            {
                __m512 vq = _mm512_loadu_ps(q + d);
                acc0 = _mm512_fmadd_ps(vq, _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(k0 + d))), acc0);
                acc1 = _mm512_fmadd_ps(vq, _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(k1 + d))), acc1);
                acc2 = _mm512_fmadd_ps(vq, _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(k2 + d))), acc2);
                acc3 = _mm512_fmadd_ps(vq, _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(k3 + d))), acc3);
            }

            s0 = _mm512_reduce_add_ps(acc0);
            s1 = _mm512_reduce_add_ps(acc1);
            s2 = _mm512_reduce_add_ps(acc2);
            s3 = _mm512_reduce_add_ps(acc3);

            for (; d < head_dim; ++d)
            {
                float qd = q[d];
                auto cvt = [](uint16_t h)
                {
                    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_set1_epi16(static_cast<short>(h))));
                };
                s0 += qd * cvt(k0[d]);
                s1 += qd * cvt(k1[d]);
                s2 += qd * cvt(k2[d]);
                s3 += qd * cvt(k3[d]);
            }
        }

        /**
         * @brief Weighted V accumulation: out[d] += weight * FP16_to_FP32(v[d]).
         */
        static void accum_weighted_v_fp16(
            float *out, const uint16_t *v, float weight, int head_dim)
        {
            __m512 w = _mm512_set1_ps(weight);
            int d = 0;
            for (; d + 15 < head_dim; d += 16)
            {
                __m512 o = _mm512_loadu_ps(out + d);
                __m256i v16 = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i *>(v + d));
                __m512 vv = _mm512_cvtph_ps(v16);
                o = _mm512_fmadd_ps(vv, w, o);
                _mm512_storeu_ps(out + d, o);
            }
            for (; d < head_dim; ++d)
            {
                __m128i h = _mm_set1_epi16(static_cast<short>(v[d]));
                out[d] += weight * _mm_cvtss_f32(_mm_cvtph_ps(h));
            }
        }

        /**
         * @brief 4-row batched FP16 V accumulation with 2× unrolled inner loop.
         *
         * Like `accum_weighted_v_fp16` but processes 4 KV rows in one pass:
         *   out[d] += w0*cvt(v0[d]) + w1*cvt(v1[d]) + w2*cvt(v2[d]) + w3*cvt(v3[d])
         * for each dimension d.  Loads/stores `out` once per 32-element chunk
         * instead of 4× (once per row), saving ~60% L1 traffic.
         */
        static void accum_weighted_v_fp16_4row(
            float *__restrict out,
            const uint16_t *v0, float w0,
            const uint16_t *v1, float w1,
            const uint16_t *v2, float w2,
            const uint16_t *v3, float w3,
            int head_dim)
        {
            const __m512 vw0 = _mm512_set1_ps(w0);
            const __m512 vw1 = _mm512_set1_ps(w1);
            const __m512 vw2 = _mm512_set1_ps(w2);
            const __m512 vw3 = _mm512_set1_ps(w3);
            int d = 0;
            // 2x unrolled: 32 elements per iteration
            for (; d + 31 < head_dim; d += 32)
            {
                __m512 oA = _mm512_loadu_ps(out + d);
                __m512 oB = _mm512_loadu_ps(out + d + 16);
                oA = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(v0 + d))), vw0, oA);
                oB = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(v0 + d + 16))), vw0, oB);
                oA = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(v1 + d))), vw1, oA);
                oB = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(v1 + d + 16))), vw1, oB);
                oA = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(v2 + d))), vw2, oA);
                oB = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(v2 + d + 16))), vw2, oB);
                oA = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(v3 + d))), vw3, oA);
                oB = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(v3 + d + 16))), vw3, oB);
                _mm512_storeu_ps(out + d, oA);
                _mm512_storeu_ps(out + d + 16, oB);
            }
            for (; d + 15 < head_dim; d += 16)
            {
                __m512 o = _mm512_loadu_ps(out + d);
                o = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(v0 + d))), vw0, o);
                o = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(v1 + d))), vw1, o);
                o = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(v2 + d))), vw2, o);
                o = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(v3 + d))), vw3, o);
                _mm512_storeu_ps(out + d, o);
            }
            for (; d < head_dim; ++d)
            {
                auto cvt = [](uint16_t fp16_val) -> float
                {
                    __m128i h = _mm_set1_epi16(static_cast<short>(fp16_val));
                    return _mm_cvtss_f32(_mm_cvtph_ps(h));
                };
                out[d] += w0 * cvt(v0[d]) + w1 * cvt(v1[d]) + w2 * cvt(v2[d]) + w3 * cvt(v3[d]);
            }
        }

        /**
         * @brief Scale a vector by a scalar: out[d] *= alpha (FP32, AVX-512).
         * Duplicated from scale_vec to avoid dependency on use_avx512 bool.
         */
        static void scale_vec_fp16path(float *out, float alpha, int head_dim)
        {
            __m512 a = _mm512_set1_ps(alpha);
            int d = 0;
            for (; d + 15 < head_dim; d += 16)
            {
                _mm512_storeu_ps(out + d, _mm512_mul_ps(a, _mm512_loadu_ps(out + d)));
            }
            for (; d < head_dim; ++d)
            {
                out[d] *= alpha;
            }
        }
#endif

        /**
         * @brief Compute optimal thread count for attention based on work size.
         *
         * Balances parallelism against OpenMP fork/join overhead (~30-50µs)
         * by estimating total FLOPs and ensuring each thread has enough work
         * to amortise the cost of entering a parallel region.
         *
         * At ~25 GFLOP/s effective FP32 throughput per core, 50µs = 1.25M FLOPs.
         * We use 2M as the threshold to ensure clear amortisation.
         */
        static int computeOptimalAttentionThreads(
            int n_heads, int seq_len, int kv_len, int head_dim, bool causal)
        {
            const int max_threads = omp_get_max_threads();

            // Work per head: QK dots + V accumulation + exp overhead
            // Causal masking halves the average KV positions per query.
            const int64_t avg_kv = causal ? (static_cast<int64_t>(kv_len) + 1) / 2
                                          : static_cast<int64_t>(kv_len);
            const int64_t flops_per_head =
                static_cast<int64_t>(seq_len) * avg_kv * static_cast<int64_t>(head_dim) * 4 + static_cast<int64_t>(seq_len) * avg_kv * 10; // exp + softmax overhead
            const int64_t total_flops = flops_per_head * n_heads;

            // Minimum FLOPs per thread to justify OMP overhead
            constexpr int64_t MIN_FLOPS_PER_THREAD = 2000000; // ~80µs at 25 GFLOP/s

            if (total_flops <= MIN_FLOPS_PER_THREAD)
            {
                return 1;
            }

            int desired = static_cast<int>(total_flops / MIN_FLOPS_PER_THREAD);
            return std::max(1, std::min({desired, n_heads, max_threads}));
        }

        // -----------------------------------------------------------------
        // Vector accumulation and scaling helpers
        // -----------------------------------------------------------------

        /**
         * @brief Accumulate `weight * v[d]` into `out[d]` for each dimension.
         *
         * This is the inner loop of the V accumulation step:
         * `out[d] += weight * V[k, d]` for one KV position k.
         *
         * Uses AVX-512 FMA when available (16 floats/cycle).
         *
         * @param out       Running output accumulator, length `head_dim`.
         * @param v         Value row for this KV position, length `head_dim`.
         * @param weight    Softmax probability for this KV position.
         * @param head_dim  Number of dimensions per head.
         * @param use_avx512 True if AVX-512 is available at runtime.
         */
        static void accum_weighted_v(float *out, const float *v, float weight, int head_dim, bool use_avx512)
        {
#if defined(__AVX512F__)
            if (use_avx512)
            {
                __m512 w = _mm512_set1_ps(weight);
                int d = 0;
                for (; d + 15 < head_dim; d += 16)
                {
                    __m512 o = _mm512_loadu_ps(out + d);
                    __m512 vv = _mm512_loadu_ps(v + d);
                    o = _mm512_fmadd_ps(vv, w, o);
                    _mm512_storeu_ps(out + d, o);
                }
                for (; d < head_dim; ++d)
                {
                    out[d] += weight * v[d];
                }
                return;
            }
#endif
            for (int d = 0; d < head_dim; ++d)
            {
                out[d] += weight * v[d];
            }
        }

        /**
         * @brief Batched 4-row FP32 V accumulation with single output load/store.
         *
         * Accumulates 4 weighted V vectors into out in one pass, saving ~60%
         * of L1 traffic vs calling accum_weighted_v four times.
         */
        static void accum_weighted_v_4row(
            float *__restrict out,
            const float *v0, float w0,
            const float *v1, float w1,
            const float *v2, float w2,
            const float *v3, float w3,
            int head_dim)
        {
#if defined(__AVX512F__)
            const __m512 vw0 = _mm512_set1_ps(w0);
            const __m512 vw1 = _mm512_set1_ps(w1);
            const __m512 vw2 = _mm512_set1_ps(w2);
            const __m512 vw3 = _mm512_set1_ps(w3);
            int d = 0;
            for (; d + 31 < head_dim; d += 32)
            {
                __m512 oA = _mm512_loadu_ps(out + d);
                __m512 oB = _mm512_loadu_ps(out + d + 16);
                oA = _mm512_fmadd_ps(_mm512_loadu_ps(v0 + d), vw0, oA);
                oB = _mm512_fmadd_ps(_mm512_loadu_ps(v0 + d + 16), vw0, oB);
                oA = _mm512_fmadd_ps(_mm512_loadu_ps(v1 + d), vw1, oA);
                oB = _mm512_fmadd_ps(_mm512_loadu_ps(v1 + d + 16), vw1, oB);
                oA = _mm512_fmadd_ps(_mm512_loadu_ps(v2 + d), vw2, oA);
                oB = _mm512_fmadd_ps(_mm512_loadu_ps(v2 + d + 16), vw2, oB);
                oA = _mm512_fmadd_ps(_mm512_loadu_ps(v3 + d), vw3, oA);
                oB = _mm512_fmadd_ps(_mm512_loadu_ps(v3 + d + 16), vw3, oB);
                _mm512_storeu_ps(out + d, oA);
                _mm512_storeu_ps(out + d + 16, oB);
            }
            for (; d + 15 < head_dim; d += 16)
            {
                __m512 o = _mm512_loadu_ps(out + d);
                o = _mm512_fmadd_ps(_mm512_loadu_ps(v0 + d), vw0, o);
                o = _mm512_fmadd_ps(_mm512_loadu_ps(v1 + d), vw1, o);
                o = _mm512_fmadd_ps(_mm512_loadu_ps(v2 + d), vw2, o);
                o = _mm512_fmadd_ps(_mm512_loadu_ps(v3 + d), vw3, o);
                _mm512_storeu_ps(out + d, o);
            }
            for (; d < head_dim; ++d)
            {
                out[d] += w0 * v0[d] + w1 * v1[d] + w2 * v2[d] + w3 * v3[d];
            }
#else
            for (int d = 0; d < head_dim; ++d)
            {
                out[d] += w0 * v0[d] + w1 * v1[d] + w2 * v2[d] + w3 * v3[d];
            }
#endif
        }

        /**
         * @brief Multiply every element of `out` by a scalar `alpha`.
         *
         * Used during online softmax to rescale the running accumulator when a
         * new tile's maximum exceeds the previous running maximum:
         *   `out[d] *= exp(old_max − new_max)`
         *
         * @param out        Vector to scale in-place, length `head_dim`.
         * @param alpha      Scalar multiplier.
         * @param head_dim   Vector length.
         * @param use_avx512 True if AVX-512 is available at runtime.
         */
        static void scale_vec(float *out, float alpha, int head_dim, bool use_avx512)
        {
#if defined(__AVX512F__)
            if (use_avx512)
            {
                __m512 a = _mm512_set1_ps(alpha);
                int d = 0;
                for (; d + 15 < head_dim; d += 16)
                {
                    __m512 o = _mm512_loadu_ps(out + d);
                    o = _mm512_mul_ps(o, a);
                    _mm512_storeu_ps(out + d, o);
                }
                for (; d < head_dim; ++d)
                {
                    out[d] *= alpha;
                }
                return;
            }
#endif
            for (int d = 0; d < head_dim; ++d)
            {
                out[d] *= alpha;
            }
        }

        /**
         * @brief Divide every element of `out` by `denom`.  No-op if denom ≤ 0.
         *
         * Called once at the end of the online softmax for each (head, query)
         * pair to normalise the accumulated output by the softmax denominator.
         *
         * @param out        Vector to normalise in-place.
         * @param denom      Softmax denominator (sum of exp(scores − max)).
         * @param head_dim   Vector length.
         * @param use_avx512 True if AVX-512 is available at runtime.
         */
        static void div_vec(float *out, float denom, int head_dim, bool use_avx512)
        {
            // Guard against division by zero or negative denominators which
            // can occur when all scores were -infinity (completely masked).
            if (denom <= 0.0f)
            {
                return;
            }
            scale_vec(out, 1.0f / denom, head_dim, use_avx512);
        }

        // -----------------------------------------------------------------
        // I16/I12 quantisation helpers for the integer prefill path
        // -----------------------------------------------------------------
        // These routines quantise FP32 rows into 16-bit integers using an
        // absmax scheme with a configurable `qmax` (typically 2047, giving
        // ~12 effective bits).  The resulting integers can be fed into the
        // VNNI dot-product routines below for high-throughput QK scoring.
        //
        // The "packed pair" variants interleave two K rows into a single
        // buffer so that a single VNNI loop can compute two dot-products
        // simultaneously (doubling throughput for the QK phase).
        // -----------------------------------------------------------------

        /**
         * @brief Quantise a single FP32 row to int16 using absmax scaling.
         *
         * 1. Find the absolute maximum of the row.
         * 2. Compute `scale = max_abs / qmax`.
         * 3. Quantise each element: `dst[i] = round(src[i] / scale)`.
         *
         * @param src   Source FP32 vector, length `n`.
         * @param dst   Destination int16 vector, length `n`.
         * @param n     Number of elements.
         * @param qmax  Maximum quantised value (e.g. 2047 for "I12").
         * @return The absmax scale factor.  Multiply `(int_dot * scale_q * scale_k)`
         *         to recover the approximate FP32 dot product.
         */
        static float quantize_row_i16_i12(const float *src, int16_t *dst, int n, int qmax)
        {
#if defined(__AVX512F__) && defined(__AVX512BW__)
            // Pass 1: vectorized absmax
            const __m512i abs_mask = _mm512_set1_epi32(0x7FFFFFFF);
            __m512 vmax = _mm512_setzero_ps();
            int i = 0;
            for (; i + 15 < n; i += 16)
            {
                const __m512 v = _mm512_loadu_ps(src + i);
                const __m512 va = _mm512_castsi512_ps(
                    _mm512_and_epi32(_mm512_castps_si512(v), abs_mask));
                vmax = _mm512_max_ps(vmax, va);
            }
            float max_abs = _mm512_reduce_max_ps(vmax);
            for (; i < n; ++i)
                max_abs = std::max(max_abs, std::abs(src[i]));

            if (max_abs <= 1e-12f)
            {
                std::memset(dst, 0, static_cast<size_t>(n) * sizeof(int16_t));
                return 0.0f;
            }

            // Pass 2: vectorized quantize (cvtps_epi32 uses round-to-nearest-even)
            const float scale = max_abs / static_cast<float>(qmax);
            const float inv_scale = 1.0f / scale;
            const __m512 v_inv = _mm512_set1_ps(inv_scale);
            const __m512i v_hi = _mm512_set1_epi32(qmax);
            const __m512i v_lo = _mm512_set1_epi32(-qmax);
            i = 0;
            for (; i + 15 < n; i += 16)
            {
                const __m512 v = _mm512_loadu_ps(src + i);
                __m512i i32 = _mm512_cvtps_epi32(_mm512_mul_ps(v, v_inv));
                i32 = _mm512_max_epi32(v_lo, _mm512_min_epi32(v_hi, i32));
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst + i),
                                    _mm512_cvtsepi32_epi16(i32));
            }
            for (; i < n; ++i)
            {
                const int q = static_cast<int>(std::lrint(src[i] * inv_scale));
                dst[i] = static_cast<int16_t>(std::max(-qmax, std::min(q, qmax)));
            }
            return scale;
#else
            float max_abs = 0.0f;
            for (int i = 0; i < n; ++i)
                max_abs = std::max(max_abs, std::abs(src[i]));

            if (max_abs <= 1e-12f)
            {
                std::memset(dst, 0, static_cast<size_t>(n) * sizeof(int16_t));
                return 0.0f;
            }

            const float scale = max_abs / static_cast<float>(qmax);
            const float inv_scale = 1.0f / scale;
            for (int i = 0; i < n; ++i)
            {
                const int q = static_cast<int>(std::lrint(src[i] * inv_scale));
                dst[i] = static_cast<int16_t>(std::max(-qmax, std::min(q, qmax)));
            }
            return scale;
#endif
        }

        /**
         * @brief Quantise with zero-padding to a multiple of the VNNI width.
         *
         * Same as `quantize_row_i16_i12()` but pads the output with zeros
         * from index `n` to `padded_n - 1`.  The VNNI dot-product routines
         * operate in 32-element blocks, so the row length must be rounded up.
         *
         * @param src       Source FP32 vector.
         * @param dst       Destination int16 vector, length `padded_n`.
         * @param n         Actual number of elements.
         * @param padded_n  Padded length (multiple of 32).
         * @param qmax      Maximum quantised value.
         * @return The absmax scale factor.
         */
        static float quantize_row_i16_i12_padded(const float *src, int16_t *dst, int n, int padded_n, int qmax)
        {
            const float scale = quantize_row_i16_i12(src, dst, n, qmax);
            if (padded_n > n)
            {
                std::memset(dst + n, 0, static_cast<size_t>(padded_n - n) * sizeof(int16_t));
            }
            return scale;
        }

        /**
         * @brief Quantise into a "packed pair" buffer for dual-row VNNI.
         *
         * Two K rows share one contiguous buffer laid out as interleaved
         * 32-element blocks:
         *
         *   [block0_row0 (32 int16)] [block0_row1 (32 int16)]
         *   [block1_row0 (32 int16)] [block1_row1 (32 int16)]
         *   …
         *
         * `row_sel` (0 or 1) selects which half of each 64-element block
         * this call writes into.  The other half is expected to have been
         * written by a prior call with the other `row_sel`.
         *
         * @param src       Source FP32 vector.
         * @param pair_dst  Packed-pair destination buffer.
         * @param n         Actual number of elements per row.
         * @param padded_n  Padded length per row (multiple of 32).
         * @param qmax      Maximum quantised value.
         * @param row_sel   Which row slot (0 or 1) to write into.
         * @return The absmax scale factor for this row.
         */
        static float quantize_row_i16_i12_to_packedpair(
            const float *src,
            int16_t *pair_dst,
            int n,
            int padded_n,
            int qmax,
            int row_sel)
        {
            float max_abs = 0.0f;
            for (int i = 0; i < n; ++i)
            {
                max_abs = std::max(max_abs, std::abs(src[i]));
            }

            if (max_abs <= 1e-12f)
            {
                for (int i = 0; i < padded_n; ++i)
                {
                    const size_t block = static_cast<size_t>(i / 32) * 64ULL;
                    const size_t lane = static_cast<size_t>(i % 32);
                    pair_dst[block + static_cast<size_t>(row_sel) * 32ULL + lane] = 0;
                }
                return 0.0f;
            }

            const float scale = max_abs / static_cast<float>(qmax);
            const float inv_scale = 1.0f / scale;
            for (int i = 0; i < n; ++i)
            {
                const int q = static_cast<int>(std::lrint(src[i] * inv_scale));
                const int16_t v = static_cast<int16_t>(std::max(-qmax, std::min(q, qmax)));
                const size_t block = static_cast<size_t>(i / 32) * 64ULL;
                const size_t lane = static_cast<size_t>(i % 32);
                pair_dst[block + static_cast<size_t>(row_sel) * 32ULL + lane] = v;
            }
            for (int i = n; i < padded_n; ++i)
            {
                const size_t block = static_cast<size_t>(i / 32) * 64ULL;
                const size_t lane = static_cast<size_t>(i % 32);
                pair_dst[block + static_cast<size_t>(row_sel) * 32ULL + lane] = 0;
            }
            return scale;
        }

        // -----------------------------------------------------------------
        // Integer dot-product routines (I16 × I16 → I32)
        // -----------------------------------------------------------------
        // These compute `Σ a[i]*b[i]` entirely in integer arithmetic.
        // The VNNI variants use the AVX-512 VPDPWSSD instruction which
        // performs 32 × (int16 × int16) → int32 accumulate per cycle.
        // -----------------------------------------------------------------

        /**
         * @brief Scalar int16 dot product (fallback, no SIMD).
         */
        static int32_t dot_i16_i16_i32_scalar(const int16_t *a, const int16_t *b, int n)
        {
            int32_t sum = 0;
            for (int i = 0; i < n; ++i)
            {
                sum += static_cast<int32_t>(a[i]) * static_cast<int32_t>(b[i]);
            }
            return sum;
        }

        /**
         * @brief VNNI-accelerated int16 dot product.
         *
         * Uses `_mm512_dpwssd_epi32` which computes 32 int16 multiply-adds
         * per instruction.  Tail elements are handled by scalar code.
         */
        static int32_t dot_i16_i16_i32_vnni(const int16_t *a, const int16_t *b, int n)
        {
#if defined(__AVX512F__) && defined(__AVX512VNNI__)
            __m512i acc = _mm512_setzero_si512();
            int i = 0;
            for (; i + 31 < n; i += 32)
            {
                const __m512i va = _mm512_loadu_si512(reinterpret_cast<const void *>(a + i));
                const __m512i vb = _mm512_loadu_si512(reinterpret_cast<const void *>(b + i));
                acc = _mm512_dpwssd_epi32(acc, va, vb);
            }

            alignas(64) int32_t lanes[16];
            _mm512_store_si512(reinterpret_cast<void *>(lanes), acc);
            int32_t sum = 0;
            for (int lane = 0; lane < 16; ++lane)
            {
                sum += lanes[lane];
            }

            for (; i < n; ++i)
            {
                sum += static_cast<int32_t>(a[i]) * static_cast<int32_t>(b[i]);
            }
            return sum;
#else
            return dot_i16_i16_i32_scalar(a, b, n);
#endif
        }

        /**
         * @brief Dual-row VNNI dot product:  `out0 = q·k0` and `out1 = q·k1`.
         *
         * Computes two dot-products against two separate K rows in a single
         * pass over the Q vector, exploiting instruction-level parallelism.
         * The inner loop is unrolled 2× (64 elements / iteration) for ILP.
         */
        static void dot_i16_i16_i32_vnni_2row(
            const int16_t *q,
            const int16_t *k0,
            const int16_t *k1,
            int n,
            int32_t &out0,
            int32_t &out1)
        {
#if defined(__AVX512F__) && defined(__AVX512VNNI__)
            __m512i acc0 = _mm512_setzero_si512();
            __m512i acc1 = _mm512_setzero_si512();

            int i = 0;
            for (; i + 63 < n; i += 64)
            {
                const __m512i q0 = _mm512_loadu_si512(reinterpret_cast<const void *>(q + i));
                const __m512i q1 = _mm512_loadu_si512(reinterpret_cast<const void *>(q + i + 32));

                const __m512i k00 = _mm512_loadu_si512(reinterpret_cast<const void *>(k0 + i));
                const __m512i k01 = _mm512_loadu_si512(reinterpret_cast<const void *>(k0 + i + 32));
                const __m512i k10 = _mm512_loadu_si512(reinterpret_cast<const void *>(k1 + i));
                const __m512i k11 = _mm512_loadu_si512(reinterpret_cast<const void *>(k1 + i + 32));

                acc0 = _mm512_dpwssd_epi32(acc0, q0, k00);
                acc0 = _mm512_dpwssd_epi32(acc0, q1, k01);
                acc1 = _mm512_dpwssd_epi32(acc1, q0, k10);
                acc1 = _mm512_dpwssd_epi32(acc1, q1, k11);
            }

            for (; i + 31 < n; i += 32)
            {
                const __m512i qv = _mm512_loadu_si512(reinterpret_cast<const void *>(q + i));
                const __m512i k0v = _mm512_loadu_si512(reinterpret_cast<const void *>(k0 + i));
                const __m512i k1v = _mm512_loadu_si512(reinterpret_cast<const void *>(k1 + i));
                acc0 = _mm512_dpwssd_epi32(acc0, qv, k0v);
                acc1 = _mm512_dpwssd_epi32(acc1, qv, k1v);
            }

            alignas(64) int32_t lanes0[16];
            alignas(64) int32_t lanes1[16];
            _mm512_store_si512(reinterpret_cast<void *>(lanes0), acc0);
            _mm512_store_si512(reinterpret_cast<void *>(lanes1), acc1);

            int32_t sum0 = 0;
            int32_t sum1 = 0;
            for (int lane = 0; lane < 16; ++lane)
            {
                sum0 += lanes0[lane];
                sum1 += lanes1[lane];
            }

            for (; i < n; ++i)
            {
                const int32_t qv = static_cast<int32_t>(q[i]);
                sum0 += qv * static_cast<int32_t>(k0[i]);
                sum1 += qv * static_cast<int32_t>(k1[i]);
            }

            out0 = sum0;
            out1 = sum1;
#else
            out0 = dot_i16_i16_i32_scalar(q, k0, n);
            out1 = dot_i16_i16_i32_scalar(q, k1, n);
#endif
        }

        /**
         * @brief Dual-row VNNI dot product from a *packed pair* K buffer.
         *
         * Like `dot_i16_i16_i32_vnni_2row()` but reads both K rows from a
         * single interleaved buffer (see `quantize_row_i16_i12_to_packedpair`).
         * Each 64-element chunk contains [row0_block, row1_block].
         *
         * @param q       Quantised query vector, length `n` (padded to 32).
         * @param k_pair  Packed pair buffer holding two K rows interleaved.
         * @param n       Padded row length.
         * @param out0    (out) Dot product with the first K row.
         * @param out1    (out) Dot product with the second K row.
         */
        static void dot_i16_i16_i32_vnni_2row_packedpair(
            const int16_t *q,
            const int16_t *k_pair,
            int n,
            int32_t &out0,
            int32_t &out1)
        {
#if defined(__AVX512F__) && defined(__AVX512VNNI__)
            __m512i acc0 = _mm512_setzero_si512();
            __m512i acc1 = _mm512_setzero_si512();

            int i = 0;
            const int16_t *pair_ptr = k_pair;
            for (; i + 31 < n; i += 32)
            {
                const __m512i qv = _mm512_loadu_si512(reinterpret_cast<const void *>(q + i));
                const __m512i k0v = _mm512_loadu_si512(reinterpret_cast<const void *>(pair_ptr));
                const __m512i k1v = _mm512_loadu_si512(reinterpret_cast<const void *>(pair_ptr + 32));
                acc0 = _mm512_dpwssd_epi32(acc0, qv, k0v);
                acc1 = _mm512_dpwssd_epi32(acc1, qv, k1v);
                pair_ptr += 64;
            }

            alignas(64) int32_t lanes0[16];
            alignas(64) int32_t lanes1[16];
            _mm512_store_si512(reinterpret_cast<void *>(lanes0), acc0);
            _mm512_store_si512(reinterpret_cast<void *>(lanes1), acc1);

            int32_t sum0 = 0;
            int32_t sum1 = 0;
            for (int lane = 0; lane < 16; ++lane)
            {
                sum0 += lanes0[lane];
                sum1 += lanes1[lane];
            }

            for (; i < n; ++i)
            {
                sum0 += static_cast<int32_t>(q[i]) * static_cast<int32_t>(k_pair[(static_cast<size_t>(i) / 32) * 64 + (i % 32)]);
                sum1 += static_cast<int32_t>(q[i]) * static_cast<int32_t>(k_pair[(static_cast<size_t>(i) / 32) * 64 + 32 + (i % 32)]);
            }

            out0 = sum0;
            out1 = sum1;
#else
            out0 = 0;
            out1 = 0;
            for (int i = 0; i < n; ++i)
            {
                const int pair_block = i / 32;
                const int lane = i % 32;
                const int32_t qv = static_cast<int32_t>(q[i]);
                out0 += qv * static_cast<int32_t>(k_pair[static_cast<size_t>(pair_block) * 64 + lane]);
                out1 += qv * static_cast<int32_t>(k_pair[static_cast<size_t>(pair_block) * 64 + 32 + lane]);
            }
#endif
        }

        /**
         * @brief Quad-row VNNI dot product from two packed-pair K buffers.
         *
         * Computes *four* dot-products in a single pass over Q:
         *   - out0 = Q · K_pair0[row0]
         *   - out1 = Q · K_pair0[row1]
         *   - out2 = Q · K_pair1[row0]
         *   - out3 = Q · K_pair1[row1]
         *
         * This is the widest variant and is used when ≥4 consecutive KV
         * positions remain in the tile, achieving maximum ILP by keeping
         * four independent accumulator chains active.
         *
         * @param q        Quantised query vector, length `n` (padded to 32).
         * @param k_pair0  First packed-pair buffer (rows 0 and 1).
         * @param k_pair1  Second packed-pair buffer (rows 2 and 3).
         * @param n        Padded row length.
         * @param out0     (out) Dot product with k_pair0, row 0.
         * @param out1     (out) Dot product with k_pair0, row 1.
         * @param out2     (out) Dot product with k_pair1, row 0.
         * @param out3     (out) Dot product with k_pair1, row 1.
         */
        static void dot_i16_i16_i32_vnni_4row_packedpair(
            const int16_t *q,
            const int16_t *k_pair0,
            const int16_t *k_pair1,
            int n,
            int32_t &out0,
            int32_t &out1,
            int32_t &out2,
            int32_t &out3)
        {
#if defined(__AVX512F__) && defined(__AVX512VNNI__)
            __m512i acc0 = _mm512_setzero_si512();
            __m512i acc1 = _mm512_setzero_si512();
            __m512i acc2 = _mm512_setzero_si512();
            __m512i acc3 = _mm512_setzero_si512();

            int i = 0;
            const int16_t *p0 = k_pair0;
            const int16_t *p1 = k_pair1;
            for (; i + 31 < n; i += 32)
            {
                const __m512i qv = _mm512_loadu_si512(reinterpret_cast<const void *>(q + i));
                const __m512i k00 = _mm512_loadu_si512(reinterpret_cast<const void *>(p0));
                const __m512i k01 = _mm512_loadu_si512(reinterpret_cast<const void *>(p0 + 32));
                const __m512i k10 = _mm512_loadu_si512(reinterpret_cast<const void *>(p1));
                const __m512i k11 = _mm512_loadu_si512(reinterpret_cast<const void *>(p1 + 32));
                acc0 = _mm512_dpwssd_epi32(acc0, qv, k00);
                acc1 = _mm512_dpwssd_epi32(acc1, qv, k01);
                acc2 = _mm512_dpwssd_epi32(acc2, qv, k10);
                acc3 = _mm512_dpwssd_epi32(acc3, qv, k11);
                p0 += 64;
                p1 += 64;
            }

            alignas(64) int32_t lanes0[16];
            alignas(64) int32_t lanes1[16];
            alignas(64) int32_t lanes2[16];
            alignas(64) int32_t lanes3[16];
            _mm512_store_si512(reinterpret_cast<void *>(lanes0), acc0);
            _mm512_store_si512(reinterpret_cast<void *>(lanes1), acc1);
            _mm512_store_si512(reinterpret_cast<void *>(lanes2), acc2);
            _mm512_store_si512(reinterpret_cast<void *>(lanes3), acc3);

            int32_t sum0 = 0;
            int32_t sum1 = 0;
            int32_t sum2 = 0;
            int32_t sum3 = 0;
            for (int lane = 0; lane < 16; ++lane)
            {
                sum0 += lanes0[lane];
                sum1 += lanes1[lane];
                sum2 += lanes2[lane];
                sum3 += lanes3[lane];
            }

            for (; i < n; ++i)
            {
                const size_t block = static_cast<size_t>(i / 32) * 64ULL;
                const size_t lane = static_cast<size_t>(i % 32);
                const int32_t qv = static_cast<int32_t>(q[i]);
                sum0 += qv * static_cast<int32_t>(k_pair0[block + lane]);
                sum1 += qv * static_cast<int32_t>(k_pair0[block + 32ULL + lane]);
                sum2 += qv * static_cast<int32_t>(k_pair1[block + lane]);
                sum3 += qv * static_cast<int32_t>(k_pair1[block + 32ULL + lane]);
            }

            out0 = sum0;
            out1 = sum1;
            out2 = sum2;
            out3 = sum3;
#else
            out0 = 0;
            out1 = 0;
            out2 = 0;
            out3 = 0;
            for (int i = 0; i < n; ++i)
            {
                const size_t block = static_cast<size_t>(i / 32) * 64ULL;
                const size_t lane = static_cast<size_t>(i % 32);
                const int32_t qv = static_cast<int32_t>(q[i]);
                out0 += qv * static_cast<int32_t>(k_pair0[block + lane]);
                out1 += qv * static_cast<int32_t>(k_pair0[block + 32ULL + lane]);
                out2 += qv * static_cast<int32_t>(k_pair1[block + lane]);
                out3 += qv * static_cast<int32_t>(k_pair1[block + 32ULL + lane]);
            }
#endif
        }

        /**
         * @brief Single-row VNNI dot product from one slot of a packed pair.
         *
         * Extracts and dot-products against only one of the two rows stored
         * in a packed-pair buffer.  Used for the "tail" case when an odd
         * number of KV positions remain in the tile (the last position has
         * no partner to pair with).
         *
         * @param q        Quantised query vector, length `n`.
         * @param k_pair   Packed-pair buffer containing two interleaved K rows.
         * @param n        Padded row length.
         * @param row_sel  Which row to read: 0 = first row, 1 = second row.
         * @return The int32 dot product `Σ q[i] * k_pair[row_sel][i]`.
         */
        static int32_t dot_i16_i16_i32_vnni_single_from_packedpair(
            const int16_t *q,
            const int16_t *k_pair,
            int n,
            int row_sel)
        {
#if defined(__AVX512F__) && defined(__AVX512VNNI__)
            __m512i acc = _mm512_setzero_si512();
            int i = 0;
            const int16_t *pair_ptr = k_pair;
            const int row_off = (row_sel != 0) ? 32 : 0;
            for (; i + 31 < n; i += 32)
            {
                const __m512i qv = _mm512_loadu_si512(reinterpret_cast<const void *>(q + i));
                const __m512i kv = _mm512_loadu_si512(reinterpret_cast<const void *>(pair_ptr + row_off));
                acc = _mm512_dpwssd_epi32(acc, qv, kv);
                pair_ptr += 64;
            }

            alignas(64) int32_t lanes[16];
            _mm512_store_si512(reinterpret_cast<void *>(lanes), acc);
            int32_t sum = 0;
            for (int lane = 0; lane < 16; ++lane)
            {
                sum += lanes[lane];
            }

            for (; i < n; ++i)
            {
                sum += static_cast<int32_t>(q[i]) * static_cast<int32_t>(k_pair[(static_cast<size_t>(i) / 32) * 64 + row_off + (i % 32)]);
            }
            return sum;
#else
            int32_t sum = 0;
            const int row_off = (row_sel != 0) ? 32 : 0;
            for (int i = 0; i < n; ++i)
            {
                sum += static_cast<int32_t>(q[i]) * static_cast<int32_t>(k_pair[(static_cast<size_t>(i) / 32) * 64 + row_off + (i % 32)]);
            }
            return sum;
#endif
        }

        /**
         * @brief 4-row VNNI dot product against separate (non-packed) K rows.
         *
         * Computes four independent int16 dot products in a single pass over
         * the Q vector, exploiting ILP across four accumulator chains.
         * Unlike the packed-pair variant, each K row is at a separate address.
         * Used by the Q16_1 KV cache decode path where K rows are stored as
         * contiguous Q16_1Block qs[] arrays.
         *
         * @param q   Quantised query vector, length `n`.
         * @param k0  First K row (int16), length `n`.
         * @param k1  Second K row (int16), length `n`.
         * @param k2  Third K row (int16), length `n`.
         * @param k3  Fourth K row (int16), length `n`.
         * @param n   Number of elements.
         * @param out0-out3  (out) Int32 dot products.
         */
        static void dot_i16_i16_i32_vnni_4row(
            const int16_t *q,
            const int16_t *k0,
            const int16_t *k1,
            const int16_t *k2,
            const int16_t *k3,
            int n,
            int32_t &out0,
            int32_t &out1,
            int32_t &out2,
            int32_t &out3)
        {
#if defined(__AVX512F__) && defined(__AVX512VNNI__)
            __m512i acc0 = _mm512_setzero_si512();
            __m512i acc1 = _mm512_setzero_si512();
            __m512i acc2 = _mm512_setzero_si512();
            __m512i acc3 = _mm512_setzero_si512();

            int i = 0;
            // Unrolled 2× (64 elements / iteration) for ILP
            for (; i + 63 < n; i += 64)
            {
                const __m512i q0 = _mm512_loadu_si512(reinterpret_cast<const void *>(q + i));
                const __m512i q1 = _mm512_loadu_si512(reinterpret_cast<const void *>(q + i + 32));

                acc0 = _mm512_dpwssd_epi32(acc0, q0, _mm512_loadu_si512(reinterpret_cast<const void *>(k0 + i)));
                acc0 = _mm512_dpwssd_epi32(acc0, q1, _mm512_loadu_si512(reinterpret_cast<const void *>(k0 + i + 32)));
                acc1 = _mm512_dpwssd_epi32(acc1, q0, _mm512_loadu_si512(reinterpret_cast<const void *>(k1 + i)));
                acc1 = _mm512_dpwssd_epi32(acc1, q1, _mm512_loadu_si512(reinterpret_cast<const void *>(k1 + i + 32)));
                acc2 = _mm512_dpwssd_epi32(acc2, q0, _mm512_loadu_si512(reinterpret_cast<const void *>(k2 + i)));
                acc2 = _mm512_dpwssd_epi32(acc2, q1, _mm512_loadu_si512(reinterpret_cast<const void *>(k2 + i + 32)));
                acc3 = _mm512_dpwssd_epi32(acc3, q0, _mm512_loadu_si512(reinterpret_cast<const void *>(k3 + i)));
                acc3 = _mm512_dpwssd_epi32(acc3, q1, _mm512_loadu_si512(reinterpret_cast<const void *>(k3 + i + 32)));
            }

            for (; i + 31 < n; i += 32)
            {
                const __m512i qv = _mm512_loadu_si512(reinterpret_cast<const void *>(q + i));
                acc0 = _mm512_dpwssd_epi32(acc0, qv, _mm512_loadu_si512(reinterpret_cast<const void *>(k0 + i)));
                acc1 = _mm512_dpwssd_epi32(acc1, qv, _mm512_loadu_si512(reinterpret_cast<const void *>(k1 + i)));
                acc2 = _mm512_dpwssd_epi32(acc2, qv, _mm512_loadu_si512(reinterpret_cast<const void *>(k2 + i)));
                acc3 = _mm512_dpwssd_epi32(acc3, qv, _mm512_loadu_si512(reinterpret_cast<const void *>(k3 + i)));
            }

            out0 = _mm512_reduce_add_epi32(acc0);
            out1 = _mm512_reduce_add_epi32(acc1);
            out2 = _mm512_reduce_add_epi32(acc2);
            out3 = _mm512_reduce_add_epi32(acc3);

            for (; i < n; ++i)
            {
                const int32_t qv = static_cast<int32_t>(q[i]);
                out0 += qv * static_cast<int32_t>(k0[i]);
                out1 += qv * static_cast<int32_t>(k1[i]);
                out2 += qv * static_cast<int32_t>(k2[i]);
                out3 += qv * static_cast<int32_t>(k3[i]);
            }
#else
            out0 = dot_i16_i16_i32_scalar(q, k0, n);
            out1 = dot_i16_i16_i32_scalar(q, k1, n);
            out2 = dot_i16_i16_i32_scalar(q, k2, n);
            out3 = dot_i16_i16_i32_scalar(q, k3, n);
#endif
        }

        /**
         * @brief Weighted V accumulation from Q16_1 block: out[d] += weight * (scale * qs[d]).
         *
         * Loads int16 values from a Q16_1 block's qs[] array, sign-extends to
         * int32 via _mm512_cvtepi16_epi32, converts to FP32, multiplies by the
         * combined weight (softmax_weight * block_scale), and FMA-accumulates
         * into the FP32 output vector.
         *
         * @param out       FP32 output vector (accumulated in-place).
         * @param v_qs      Pointer to int16_t qs[] data from Q16_1 block.
         * @param combined  Pre-computed (softmax_weight * v_block.d).
         * @param head_dim  Number of elements.
         */
        static void accum_weighted_v_q16(
            float *out, const int16_t *v_qs, float combined, int head_dim)
        {
#if defined(__AVX512F__)
            const __m512 w = _mm512_set1_ps(combined);
            int d = 0;
            // 2× unrolled for ILP (32 elements per iteration)
            for (; d + 31 < head_dim; d += 32)
            {
                __m512 o0 = _mm512_loadu_ps(out + d);
                __m512 o1 = _mm512_loadu_ps(out + d + 16);
                const __m512 vf0 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(
                    _mm256_loadu_si256(reinterpret_cast<const __m256i *>(v_qs + d))));
                const __m512 vf1 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(
                    _mm256_loadu_si256(reinterpret_cast<const __m256i *>(v_qs + d + 16))));
                o0 = _mm512_fmadd_ps(vf0, w, o0);
                o1 = _mm512_fmadd_ps(vf1, w, o1);
                _mm512_storeu_ps(out + d, o0);
                _mm512_storeu_ps(out + d + 16, o1);
            }
            for (; d + 15 < head_dim; d += 16)
            {
                __m512 o = _mm512_loadu_ps(out + d);
                const __m512 vf = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(
                    _mm256_loadu_si256(reinterpret_cast<const __m256i *>(v_qs + d))));
                o = _mm512_fmadd_ps(vf, w, o);
                _mm512_storeu_ps(out + d, o);
            }
            for (; d < head_dim; ++d)
            {
                out[d] += combined * static_cast<float>(v_qs[d]);
            }
#else
            for (int d = 0; d < head_dim; ++d)
            {
                out[d] += combined * static_cast<float>(v_qs[d]);
            }
#endif
        }

        /**
         * @brief Batched 4-row V accumulation: out[d] += Σ_i w_i * v_i_qs[d].
         *
         * Loads the output vector once, accumulates 4 dequantised V rows with
         * their softmax weights, then stores once.  Saves ~60% of L1 traffic
         * vs calling accum_weighted_v_q16 four separate times.
         */
        static void accum_weighted_v_q16_4row(
            float *__restrict out,
            const int16_t *v0, float w0,
            const int16_t *v1, float w1,
            const int16_t *v2, float w2,
            const int16_t *v3, float w3,
            int head_dim)
        {
#if defined(__AVX512F__)
            const __m512 vw0 = _mm512_set1_ps(w0);
            const __m512 vw1 = _mm512_set1_ps(w1);
            const __m512 vw2 = _mm512_set1_ps(w2);
            const __m512 vw3 = _mm512_set1_ps(w3);
            int d = 0;
            // 2× unrolled: 32 elements/iteration for head_dim=128 (4 iterations)
            // Keeps both FMA ports fed by interleaving independent chains.
            for (; d + 31 < head_dim; d += 32)
            {
                __m512 oA = _mm512_loadu_ps(out + d);
                __m512 oB = _mm512_loadu_ps(out + d + 16);
                oA = _mm512_fmadd_ps(
                    _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(
                        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(v0 + d)))),
                    vw0, oA);
                oB = _mm512_fmadd_ps(
                    _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(
                        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(v0 + d + 16)))),
                    vw0, oB);
                oA = _mm512_fmadd_ps(
                    _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(
                        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(v1 + d)))),
                    vw1, oA);
                oB = _mm512_fmadd_ps(
                    _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(
                        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(v1 + d + 16)))),
                    vw1, oB);
                oA = _mm512_fmadd_ps(
                    _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(
                        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(v2 + d)))),
                    vw2, oA);
                oB = _mm512_fmadd_ps(
                    _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(
                        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(v2 + d + 16)))),
                    vw2, oB);
                oA = _mm512_fmadd_ps(
                    _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(
                        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(v3 + d)))),
                    vw3, oA);
                oB = _mm512_fmadd_ps(
                    _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(
                        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(v3 + d + 16)))),
                    vw3, oB);
                _mm512_storeu_ps(out + d, oA);
                _mm512_storeu_ps(out + d + 16, oB);
            }
            for (; d + 15 < head_dim; d += 16)
            {
                __m512 o = _mm512_loadu_ps(out + d);
                o = _mm512_fmadd_ps(
                    _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(
                        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(v0 + d)))),
                    vw0, o);
                o = _mm512_fmadd_ps(
                    _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(
                        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(v1 + d)))),
                    vw1, o);
                o = _mm512_fmadd_ps(
                    _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(
                        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(v2 + d)))),
                    vw2, o);
                o = _mm512_fmadd_ps(
                    _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(
                        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(v3 + d)))),
                    vw3, o);
                _mm512_storeu_ps(out + d, o);
            }
            for (; d < head_dim; ++d)
            {
                out[d] += w0 * static_cast<float>(v0[d]) + w1 * static_cast<float>(v1[d]) + w2 * static_cast<float>(v2[d]) + w3 * static_cast<float>(v3[d]);
            }
#else
            accum_weighted_v_q16(out, v0, w0, head_dim);
            accum_weighted_v_q16(out, v1, w1, head_dim);
            accum_weighted_v_q16(out, v2, w2, head_dim);
            accum_weighted_v_q16(out, v3, w3, head_dim);
#endif
        }

        // =================================================================
        // Core flash-attention implementation
        // =================================================================

        /**
         * @brief Core tiled flash-attention with online softmax (FP32 path).
         *
         * This is the heart of the flash attention kernel.  It computes:
         *
         *     output[q][h] = softmax( Q[q,h] · K[:,kv_h]^T / √d ) · V[:,kv_h]
         *
         * …without ever materialising the full `[seq_len × kv_len]` score matrix.
         *
         * ### Algorithm outline
         *
         * For each query head `h` and query position `q`:
         *   1. Initialise `running_m = -∞` (running max) and `running_l = 0`
         *      (running softmax denominator).
         *   2. Iterate over KV-position tiles of size `kv_tile`.
         *      a. Compute `score[k] = Q[q,h] · K[k, kv_h] / √d` for each
         *         position k in the tile.  Optionally use I16/VNNI arithmetic
         *         for the dot-product during prefill.
         *      b. Find `block_max` = max score in this tile.
         *      c. Compute `new_m = max(running_m, block_max)`.
         *      d. **Rescale** the running output and denominator:
         *         `out *= exp(running_m − new_m)` and
         *         `running_l *= exp(running_m − new_m)`.
         *      e. For each k in tile: `p = exp(score[k] − new_m)`,
         *         `running_l += p`, `out += p * V[k, kv_h]`.
         *      f. Update `running_m = new_m`.
         *   3. Normalise: `out /= running_l`.
         *
         * ### Parallelism
         *
         * The outer loop over heads (`h`) is parallelised with OpenMP via
         * `OMP_WORKSHARE_REGION`.  Each head is fully independent, so this
         * achieves near-linear scaling across CPU cores.
         *
         * @param Q               Query data, row-major [seq_len, n_heads * head_dim].
         * @param K               Key data,   row-major [kv_len, n_kv_heads * head_dim].
         * @param V               Value data, row-major [kv_len, n_kv_heads * head_dim].
         * @param output          Output buffer,       [seq_len, n_heads * head_dim].
         * @param seq_len         Number of query positions.
         * @param kv_len          Number of key/value positions.
         * @param n_heads         Total query heads.
         * @param n_kv_heads      Total KV heads (GQA: n_heads / n_kv_heads > 1).
         * @param head_dim        Elements per head.
         * @param causal          If true, position k is masked when k > q_abs.
         * @param window_size     Sliding-window limit (-1 = unlimited).
         * @param position_offset Absolute position of the first query token
         *                        (non-zero during decode where q_pos 0 is not
         *                        the beginning of the sequence).
         * @param mask            Optional additive mask [seq_len, kv_len] or nullptr.
         * @return true on success, false if pointers are null or dimensions invalid.
         */
        static bool compute_flash_fp32(
            const float *Q, const float *K, const float *V, float *output,
            int seq_len, int kv_len,
            int n_heads, int n_kv_heads, int head_dim,
            bool causal, int window_size, int position_offset,
            const float *mask)
        {
            KERNEL_PROFILE_SCOPE(KernelType::ATTENTION);

            // --- Input validation ---
            if (!Q || !K || !V || !output)
            {
                return false;
            }
            if (seq_len <= 0 || kv_len <= 0 || n_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0)
            {
                return false;
            }
            // GQA requires n_heads to be a multiple of n_kv_heads (e.g. 32 Q heads / 8 KV heads = 4 Q heads per KV head).
            if (n_heads % n_kv_heads != 0)
            {
                return false;
            }

            // --- Pre-compute constants ---
            const bool use_avx512 = cpu_supports_avx512();
            const bool is_decode = (seq_len == 1 && kv_len >= 1);

            // Choose the KV tile size based on cache hierarchy.
            const int kv_tile = detail::DefaultFlashKVTilePolicy::choose(head_dim, n_kv_heads, kv_len, is_decode);

            // For Grouped Query Attention: how many Q heads share one KV head.
            const int heads_per_kv = n_heads / n_kv_heads;

            // Attention score scaling factor:  1/√d
            const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

            // Row strides in the flat [seq_len, heads*head_dim] layout.
            const int q_stride = n_heads * head_dim;     // Q row = all Q heads concatenated
            const int kv_stride = n_kv_heads * head_dim; // K/V row = all KV heads concatenated

            // --- I16/VNNI quantisation layout parameters ---
            // Pad head_dim up to a multiple of 32 (the VNNI block width).
            const int i16_row_stride = ((head_dim + 31) / 32) * 32;
            const int i16_chunks = i16_row_stride / 32; // Number of 32-element VNNI blocks per row

            // Packed pairs: two K rows are interleaved into one buffer.
            // This halves the number of outer loop iterations in the QK phase.
            const int kv_pair_count = (kv_len + 1) / 2;
            // Each packed pair = i16_chunks blocks × 64 int16 elements (32 per row × 2 rows).
            const int k_pair_stride = i16_chunks * 64;

            // Profiling: track QK dot-product and V accumulation times separately.
            const bool profiling_enabled = KernelProfiler::isEnabled();
            uint64_t qk_duration_ns = 0;
            uint64_t v_duration_ns = 0;

            // --- Decide whether to use the I16/I12 integer dot-product path ---
            // The integer path is faster for long prefill sequences on CPUs with
            // VNNI support, because VPDPWSSD does 32 × int16 multiply-adds per
            // cycle vs. 16 × FP32 FMAs.  We only enable it when the problem is
            // large enough to amortise the quantisation overhead.
            const auto &env = debugEnv();
            const int64_t prefill_work = static_cast<int64_t>(seq_len) * static_cast<int64_t>(kv_len);
            const bool use_i16_i12_prefill =
                !is_decode &&
                env.attention.flash_prefill_i16_i12 &&                          // Feature flag from debugEnv
                cpu_supports_avx512_vnni() &&                                   // Hardware must have VNNI
                seq_len >= env.attention.flash_prefill_i16_i12_min_seq &&       // Minimum seq_len threshold
                kv_len >= env.attention.flash_prefill_i16_i12_min_kv &&         // Minimum kv_len threshold
                prefill_work >= env.attention.flash_prefill_i16_i12_min_work && // Minimum total work
                head_dim <= env.attention.flash_prefill_i16_i12_max_head_dim;   // head_dim cap

            // qmax controls the I12 quantisation range.  Typical value: 2047, giving
            // ~12 effective bits of precision within 16-bit integer storage.
            const int qmax = std::max(1, std::min(env.attention.flash_prefill_i16_i12_qmax, 32767));

            // Pre-quantised packed-pair K buffers and their per-row absmax scales.
            // These are populated once (before the main attention loop) and reused
            // for every Q row, amortising the quantisation cost.
            std::vector<int16_t> packed_k_pairs_i16;
            std::vector<float> packed_k_pair_scales;

            // ---------------------------------------------------------------
            // Phase 1 (optional): Pre-quantise all K rows into packed pairs
            // ---------------------------------------------------------------
            // This runs only during prefill when the I16 path is active.
            // Each pair of consecutive K rows (k0, k1) is quantised into a
            // single interleaved buffer so the VNNI dot-product loop can
            // compute two scores per iteration.
            if (use_i16_i12_prefill)
            {
                // Allocate: [n_kv_heads × kv_pair_count] packed-pair buffers.
                packed_k_pairs_i16.resize(static_cast<size_t>(n_kv_heads) * static_cast<size_t>(kv_pair_count) * static_cast<size_t>(k_pair_stride));
                // Two scale factors per pair (one per K row).
                packed_k_pair_scales.resize(static_cast<size_t>(n_kv_heads) * static_cast<size_t>(kv_pair_count) * 2ULL, 0.0f);

                // Parallel quantisation across KV heads and pair indices.
                auto do_pack = [&]()
                {
#pragma omp for collapse(2) schedule(static)
                    for (int kv_h = 0; kv_h < n_kv_heads; ++kv_h)
                    {
                        for (int pair_idx = 0; pair_idx < kv_pair_count; ++pair_idx)
                        {
                            // Each pair contains two consecutive KV positions.
                            const int k0 = pair_idx * 2;
                            const int k1 = k0 + 1;

                            // Global index into the flat packed_k_pairs arrays.
                            const size_t pair_global = static_cast<size_t>(kv_h) * static_cast<size_t>(kv_pair_count) + static_cast<size_t>(pair_idx);
                            int16_t *pair_dst = packed_k_pairs_i16.data() + pair_global * static_cast<size_t>(k_pair_stride);

                            // Quantise row 0 of the pair (slot 0 in the interleaved buffer).
                            float s0 = 0.0f;
                            float s1 = 0.0f;
                            if (k0 < kv_len)
                            {
                                const float *k_src0 = K + static_cast<size_t>(k0) * kv_stride + static_cast<size_t>(kv_h) * head_dim;
                                s0 = quantize_row_i16_i12_to_packedpair(k_src0, pair_dst, head_dim, i16_row_stride, qmax, 0);
                            }
                            else
                            {
                                // Beyond the end of the KV sequence — zero-fill slot 0.
                                for (int i = 0; i < i16_row_stride; ++i)
                                {
                                    const size_t block = static_cast<size_t>(i / 32) * 64ULL;
                                    const size_t lane = static_cast<size_t>(i % 32);
                                    pair_dst[block + lane] = 0;
                                }
                            }

                            // Quantise row 1 of the pair (slot 1 in the interleaved buffer).
                            if (k1 < kv_len)
                            {
                                const float *k_src1 = K + static_cast<size_t>(k1) * kv_stride + static_cast<size_t>(kv_h) * head_dim;
                                s1 = quantize_row_i16_i12_to_packedpair(k_src1, pair_dst, head_dim, i16_row_stride, qmax, 1);
                            }
                            else
                            {
                                // Beyond the end — zero-fill slot 1.
                                for (int i = 0; i < i16_row_stride; ++i)
                                {
                                    const size_t block = static_cast<size_t>(i / 32) * 64ULL;
                                    const size_t lane = static_cast<size_t>(i % 32);
                                    pair_dst[block + 32ULL + lane] = 0;
                                }
                            }

                            // Store the absmax scale for each row so we can recover
                            // the approximate FP32 dot product later:
                            //   fp32_dot ≈ int32_dot × q_scale × k_scale
                            packed_k_pair_scales[pair_global * 2ULL + 0ULL] = s0;
                            packed_k_pair_scales[pair_global * 2ULL + 1ULL] = s1;
                        }
                    }
                };
                OMP_WORKSHARE_REGION(do_pack);
            }

            // ---------------------------------------------------------------
            // Phase 2: Main flash-attention loop (parallel over heads)
            // ---------------------------------------------------------------
            // Compute optimal thread count.  For small problems (short
            // sequences, decode), the per-head work may be too small to
            // amortise OMP fork/join overhead (~30-50µs).  The heuristic
            // estimates total FLOPs and scales threads accordingly.
            const int attn_threads = computeOptimalAttentionThreads(
                n_heads, seq_len, kv_len, head_dim, causal);

            auto work = [&]()
            {
#pragma omp for schedule(static) reduction(+ : qk_duration_ns, v_duration_ns)
                for (int h = 0; h < n_heads; ++h)
                {
                    // GQA mapping: which KV head does this Q head read from?
                    const int kv_h = h / heads_per_kv;

                    // Pointer to the pre-quantised packed-pair K data for this KV head
                    // (nullptr when using the FP32 dot-product path).
                    const int16_t *k_head_pairs_i16 = use_i16_i12_prefill
                                                          ? packed_k_pairs_i16.data() + static_cast<size_t>(kv_h) * static_cast<size_t>(kv_pair_count) * static_cast<size_t>(k_pair_stride)
                                                          : nullptr;
                    const float *k_head_pair_scales = use_i16_i12_prefill
                                                          ? packed_k_pair_scales.data() + static_cast<size_t>(kv_h) * static_cast<size_t>(kv_pair_count) * 2ULL
                                                          : nullptr;

                    // --- Loop over query positions for this head ---
                    for (int q_pos = 0; q_pos < seq_len; ++q_pos)
                    {
                        // Pointer to this (q_pos, head) slice of the output buffer.
                        float *out = output + static_cast<size_t>(q_pos) * q_stride + static_cast<size_t>(h) * head_dim;
                        std::fill(out, out + head_dim, 0.0f); // Zero-init the accumulator

                        // Online softmax running state:
                        //   running_m = max score seen so far (for numerical stability)
                        //   running_l = sum of exp(score - running_m) seen so far (denominator)
                        float running_m = -std::numeric_limits<float>::infinity();
                        float running_l = 0.0f;

                        // Raw Q pointer for this (q_pos, head) pair.
                        const float *q_ptr = Q + static_cast<size_t>(q_pos) * q_stride + static_cast<size_t>(h) * head_dim;

                        // Absolute position of this query token in the full sequence
                        // (needed for causal masking during decode).
                        const int q_abs = position_offset + q_pos;

                        // Optional additive mask row for this query position.
                        const float *mask_row = mask ? (mask + static_cast<size_t>(q_pos) * kv_len) : nullptr;

                        // Thread-local I16 quantised Q buffer, reused across q_pos iterations.
                        thread_local std::vector<int16_t> q_i16;
                        float q_scale_i16 = 0.0f;
                        if (use_i16_i12_prefill)
                        {
                            if (static_cast<int>(q_i16.size()) < i16_row_stride)
                            {
                                q_i16.resize(static_cast<size_t>(i16_row_stride));
                            }
                            // Quantise the Q row once; it will be dotted against every K pair.
                            q_scale_i16 = quantize_row_i16_i12_padded(q_ptr, q_i16.data(), head_dim, i16_row_stride, qmax);
                        }

                        // ===================================================
                        // KV tile loop — the heart of flash attention
                        // ===================================================
                        // We iterate over KV positions in tiles of `kv_tile`.
                        // For each tile we:
                        //   (a) compute QK scores (the "QK phase")
                        //   (b) run the online softmax correction
                        //   (c) accumulate weighted V into the output (the "V phase")
                        for (int k0 = 0; k0 < kv_len; k0 += kv_tile)
                        {
                            const int k1 = std::min(k0 + kv_tile, kv_len); // End of this tile
                            float block_max = -std::numeric_limits<float>::infinity();

                            // Thread-local score buffer, reused across tiles.
                            thread_local std::vector<float> block_scores;
                            const int blk = k1 - k0; // Number of KV positions in this tile
                            if (static_cast<int>(block_scores.size()) < blk)
                            {
                                block_scores.resize(blk);
                            }

                            const auto qk_start = profiling_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();

                            // --- Compute the valid attention window within this tile ---
                            // Positions outside [valid_start, valid_end) are masked to -inf
                            // (they must not contribute to the softmax).
                            int valid_start = k0;
                            int valid_end = k1;
                            if (window_size > 0)
                            {
                                // Sliding window: only attend to the most recent `window_size` positions.
                                valid_start = std::max(valid_start, q_abs - window_size + 1);
                            }
                            if (causal)
                            {
                                // Causal mask: cannot attend to future positions.
                                valid_end = std::min(valid_end, q_abs + 1);
                            }
                            // Clamp to tile boundaries.
                            valid_start = std::max(k0, std::min(valid_start, k1));
                            valid_end = std::max(k0, std::min(valid_end, k1));

                            // Fill masked positions with -inf so exp(-inf) = 0.
                            for (int k = k0; k < valid_start; ++k)
                            {
                                block_scores[static_cast<size_t>(k - k0)] = -std::numeric_limits<float>::infinity();
                            }
                            for (int k = valid_end; k < k1; ++k)
                            {
                                block_scores[static_cast<size_t>(k - k0)] = -std::numeric_limits<float>::infinity();
                            }

                            // --- QK Phase: compute score[k] = Q · K[k] / √d ---
                            // The code below dispatches to I16 VNNI or FP32 AVX-512
                            // dot-products depending on settings.  The I16 path
                            // processes 4, 2, or 1 KV positions at a time using the
                            // packed-pair layout for maximum throughput.
                            for (int k = valid_start; k < valid_end;)
                            {
                                if (use_i16_i12_prefill)
                                {
                                    // --- I16 fast path: try 4-at-a-time (two packed pairs) ---
                                    if (k + 3 < valid_end)
                                    {
                                        const int pair_idx0 = k / 2;
                                        const int pair_idx1 = pair_idx0 + 1;
                                        const int16_t *k_pair0 = k_head_pairs_i16 + static_cast<size_t>(pair_idx0) * static_cast<size_t>(k_pair_stride);
                                        const int16_t *k_pair1 = k_head_pairs_i16 + static_cast<size_t>(pair_idx1) * static_cast<size_t>(k_pair_stride);
                                        const float k_scale0 = k_head_pair_scales[static_cast<size_t>(pair_idx0) * 2ULL + 0ULL];
                                        const float k_scale1 = k_head_pair_scales[static_cast<size_t>(pair_idx0) * 2ULL + 1ULL];
                                        const float k_scale2 = k_head_pair_scales[static_cast<size_t>(pair_idx1) * 2ULL + 0ULL];
                                        const float k_scale3 = k_head_pair_scales[static_cast<size_t>(pair_idx1) * 2ULL + 1ULL];

                                        int32_t dot0 = 0, dot1 = 0, dot2 = 0, dot3 = 0;
                                        dot_i16_i16_i32_vnni_4row_packedpair(q_i16.data(), k_pair0, k_pair1, i16_row_stride, dot0, dot1, dot2, dot3);

                                        float s0 = static_cast<float>(dot0) * (q_scale_i16 * k_scale0) * scale;
                                        float s1 = static_cast<float>(dot1) * (q_scale_i16 * k_scale1) * scale;
                                        float s2 = static_cast<float>(dot2) * (q_scale_i16 * k_scale2) * scale;
                                        float s3 = static_cast<float>(dot3) * (q_scale_i16 * k_scale3) * scale;

                                        if (mask_row)
                                        {
                                            s0 += mask_row[k + 0];
                                            s1 += mask_row[k + 1];
                                            s2 += mask_row[k + 2];
                                            s3 += mask_row[k + 3];
                                        }

                                        block_scores[static_cast<size_t>(k - k0)] = s0;
                                        block_scores[static_cast<size_t>(k + 1 - k0)] = s1;
                                        block_scores[static_cast<size_t>(k + 2 - k0)] = s2;
                                        block_scores[static_cast<size_t>(k + 3 - k0)] = s3;
                                        block_max = std::max(block_max, s0);
                                        block_max = std::max(block_max, s1);
                                        block_max = std::max(block_max, s2);
                                        block_max = std::max(block_max, s3);

                                        k += 4;
                                        continue;
                                    }

                                    // --- I16 fast path: try 2-at-a-time (one packed pair) ---
                                    if (k + 1 < valid_end)
                                    {
                                        const int pair_idx = k / 2;
                                        const int16_t *k_pair = k_head_pairs_i16 + static_cast<size_t>(pair_idx) * static_cast<size_t>(k_pair_stride);
                                        const float k_scale0 = k_head_pair_scales[static_cast<size_t>(pair_idx) * 2ULL + 0ULL];
                                        const float k_scale1 = k_head_pair_scales[static_cast<size_t>(pair_idx) * 2ULL + 1ULL];
                                        int32_t dot_i32_0 = 0;
                                        int32_t dot_i32_1 = 0;
                                        dot_i16_i16_i32_vnni_2row_packedpair(q_i16.data(), k_pair, i16_row_stride, dot_i32_0, dot_i32_1);

                                        float s0 = static_cast<float>(dot_i32_0) * (q_scale_i16 * k_scale0);
                                        float s1 = static_cast<float>(dot_i32_1) * (q_scale_i16 * k_scale1);
                                        s0 *= scale;
                                        s1 *= scale;

                                        if (mask_row)
                                        {
                                            s0 += mask_row[k];
                                            s1 += mask_row[k + 1];
                                        }

                                        block_scores[static_cast<size_t>(k - k0)] = s0;
                                        block_scores[static_cast<size_t>(k + 1 - k0)] = s1;
                                        block_max = std::max(block_max, s0);
                                        block_max = std::max(block_max, s1);
                                        k += 2;
                                        continue;
                                    }

                                    // --- I16 tail: single remaining position ---
                                    const int pair_idx = k / 2;
                                    const int row_sel = (k & 1); // 0 = first row in pair, 1 = second
                                    const int16_t *k_pair = k_head_pairs_i16 + static_cast<size_t>(pair_idx) * static_cast<size_t>(k_pair_stride);
                                    const float k_scale = k_head_pair_scales[static_cast<size_t>(pair_idx) * 2ULL + static_cast<size_t>(row_sel)];
                                    const int32_t dot_i32 = dot_i16_i16_i32_vnni_single_from_packedpair(q_i16.data(), k_pair, i16_row_stride, row_sel);
                                    // Recover approximate FP32 dot: int_dot × q_scale × k_scale
                                    float s = static_cast<float>(dot_i32) * (q_scale_i16 * k_scale);
                                    s *= scale;

                                    if (mask_row)
                                    {
                                        s += mask_row[k];
                                    }

                                    block_scores[static_cast<size_t>(k - k0)] = s;
                                    block_max = std::max(block_max, s);
                                    ++k;
                                    continue;
                                }

                                // --- FP32 path: batched 4-row dot products for ILP ---
#if defined(__AVX512F__)
                                if (k + 3 < valid_end)
                                {
                                    const float *kbase = K + static_cast<size_t>(kv_h) * head_dim;
                                    float s0, s1, s2, s3;
                                    dot_fp32_avx512_4row(
                                        q_ptr,
                                        kbase + static_cast<size_t>(k + 0) * kv_stride,
                                        kbase + static_cast<size_t>(k + 1) * kv_stride,
                                        kbase + static_cast<size_t>(k + 2) * kv_stride,
                                        kbase + static_cast<size_t>(k + 3) * kv_stride,
                                        head_dim, s0, s1, s2, s3);
                                    s0 *= scale;
                                    s1 *= scale;
                                    s2 *= scale;
                                    s3 *= scale;
                                    if (mask_row)
                                    {
                                        s0 += mask_row[k + 0];
                                        s1 += mask_row[k + 1];
                                        s2 += mask_row[k + 2];
                                        s3 += mask_row[k + 3];
                                    }
                                    block_scores[static_cast<size_t>(k - k0 + 0)] = s0;
                                    block_scores[static_cast<size_t>(k - k0 + 1)] = s1;
                                    block_scores[static_cast<size_t>(k - k0 + 2)] = s2;
                                    block_scores[static_cast<size_t>(k - k0 + 3)] = s3;
                                    block_max = std::max(block_max, std::max(std::max(s0, s1), std::max(s2, s3)));
                                    k += 4;
                                    continue;
                                }
#endif
                                // Scalar / tail path
                                float s = dot_fp32_avx512(
                                    q_ptr,
                                    K + static_cast<size_t>(k) * kv_stride + static_cast<size_t>(kv_h) * head_dim,
                                    head_dim);
                                s *= scale;

                                if (mask_row)
                                {
                                    s += mask_row[k];
                                }

                                block_scores[static_cast<size_t>(k - k0)] = s;
                                block_max = std::max(block_max, s);
                                ++k;
                            }

                            if (profiling_enabled)
                            {
                                qk_duration_ns += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - qk_start).count());
                            }

                            // --- Online softmax correction ---
                            // The key insight: if a new tile has a higher max than what
                            // we've seen, all previous contributions must be rescaled.
                            //
                            //   new_m = max(running_m, block_max)
                            //   alpha = exp(running_m - new_m)      ← correction factor
                            //   out   *= alpha                       ← rescale accumulated output
                            //   running_l *= alpha                   ← rescale denominator
                            //
                            // If running_m was -inf (first tile), alpha = 0, which correctly
                            // zeroes out any stale data in the accumulator.
                            const float new_m = std::max(running_m, block_max);
                            const float alpha = std::isfinite(running_m) ? std::exp(running_m - new_m) : 0.0f;
                            scale_vec(out, alpha, head_dim, use_avx512);
                            float new_l = running_l * alpha;

                            // --- V Phase: accumulate weighted values ---
                            // For each KV position in this tile, compute the unnormalised
                            // softmax probability p = exp(score - new_m) and add p*V to out.
                            // Positions in [valid_start, valid_end) are guaranteed finite.
                            // Positions outside that range are -inf (masked) — skip them.
                            const auto v_start = profiling_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();

                            {
                                const float *v_base = V + static_cast<size_t>(kv_h) * head_dim;
                                int k = valid_start;
#if defined(__AVX512F__)
                                // 4-wide batched: vectorized exp + batched V accumulation
                                for (; k + 3 < valid_end; k += 4)
                                {
                                    const __m128 scores4 = _mm_set_ps(
                                        block_scores[static_cast<size_t>(k - k0 + 3)],
                                        block_scores[static_cast<size_t>(k - k0 + 2)],
                                        block_scores[static_cast<size_t>(k - k0 + 1)],
                                        block_scores[static_cast<size_t>(k - k0 + 0)]);
                                    const __m128 nm4 = _mm_set1_ps(new_m);
                                    const __m512 exp_in = _mm512_castps128_ps512(_mm_sub_ps(scores4, nm4));
                                    const __m512 exp_out = fast_exp_avx512(exp_in);
                                    const __m128 probs = _mm512_castps512_ps128(exp_out);
                                    alignas(16) float pp[4];
                                    _mm_store_ps(pp, probs);
                                    new_l += pp[0] + pp[1] + pp[2] + pp[3];

                                    accum_weighted_v_4row(
                                        out,
                                        v_base + static_cast<size_t>(k + 0) * kv_stride, pp[0],
                                        v_base + static_cast<size_t>(k + 1) * kv_stride, pp[1],
                                        v_base + static_cast<size_t>(k + 2) * kv_stride, pp[2],
                                        v_base + static_cast<size_t>(k + 3) * kv_stride, pp[3],
                                        head_dim);
                                }
#endif
                                // Scalar tail: remaining valid positions
                                for (; k < valid_end; ++k)
                                {
                                    const float p = std::exp(block_scores[static_cast<size_t>(k - k0)] - new_m);
                                    new_l += p;
                                    accum_weighted_v(out, v_base + static_cast<size_t>(k) * kv_stride, p, head_dim, use_avx512);
                                }
                            }

                            if (profiling_enabled)
                            {
                                v_duration_ns += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - v_start).count());
                            }

                            // Advance running state for this query position.
                            running_m = new_m;
                            running_l = new_l;
                        } // end KV tile loop

                        // --- Final normalisation ---
                        // Divide the accumulated output by the softmax denominator
                        // to get the true weighted average:  out = out / Σ exp(s - m).
                        div_vec(out, running_l, head_dim, use_avx512);
                    } // end q_pos loop
                } // end head loop
            };

            // DECODE THREADING STRATEGY: prevent libgomp thread pool sleep.
            //
            // When kv_len is large enough that attention takes >~200μs
            // (roughly kv_len > 100 for this model), the idle gap between the
            // last OMP region (RoPE) and the next one (Wo GEMV) causes libgomp
            // to put worker threads to futex-sleep. Waking 27 threads costs
            // ~5ms via futex(FUTEX_WAKE), which was the root cause of decode
            // degradation from 10→4 tok/s at long context.
            //
            // Fix: for decode at kv_len > 100, use ALL threads in the
            // attention OMP region. This keeps the thread pool alive between
            // RoPE and Wo. Each thread handles ~1 head (28 heads / 28 threads).
            //
            // For short context (kv_len <= 100) or prefill, use the heuristic
            // thread count — threads stay within libgomp's spin window and
            // Wo GEMV fork is fast regardless.
            //
            // Note: OMP_WAIT_POLICY=active and GOMP_SPINCOUNT=infinite do NOT
            // prevent the inter-region sleep; libgomp's team-end idle path has
            // a fixed spin count independent of these settings.
            const bool is_decode_phase = (kv_len != seq_len);
            const bool force_full_pool = is_decode_phase && kv_len > 100;

            // Use OMP_WORKSHARE_REGION for decode so the kernel is
            // compatible with a persistent outer parallel region (the
            // macro detects omp_in_parallel() and skips creating a new
            // team, eliminating ~16 µs fork/join overhead per call).
            int actual_threads;
            if (force_full_pool)
            {
                OMP_WORKSHARE_REGION(work);
                actual_threads = omp_get_max_threads();
            }
            else
            {
#pragma omp parallel num_threads(attn_threads)
                {
                    work();
                }
                actual_threads = attn_threads;
            }

            // Report profiling breakdown (QK vs V phase) if enabled.
            // qk_duration_ns and v_duration_ns are thread-time sums from
            // the OMP reduction — divide by active thread count to get
            // wall-clock estimates that are consistent with other kernel
            // timings in the profiler output.
            if (profiling_enabled)
            {
                KernelProfiler::recordParallel(KernelType::ATTENTION_QK, qk_duration_ns, actual_threads);
                KernelProfiler::recordParallel(KernelType::ATTENTION_V, v_duration_ns, actual_threads);
            }
            return true;
        }

#if defined(__AVX512F__) && defined(__AVX512VNNI__)
        /**
         * @brief Prefill flash attention with Q16_1 KV cache (VNNI accelerated).
         *
         * Handles seq_len >= 1 (including prefill where seq_len == kv_len).
         * Uses the same VNNI QK dot products and Q16_1 V accumulation as
         * the decode path, but loops over multiple query positions.
         *
         * QK Phase: Q is quantized per-head to int16, then dotted against
         * K block qs[] data using VPDPWSSD. For single-block-per-head layouts
         * (block_size >= head_dim), a 4-row batched VNNI dot is used.
         *
         * V Phase: 4-wide vectorized exp via fast_exp_avx512 + batched
         * accum_weighted_v_q16_4row for maximum throughput.
         */
        static bool compute_prefill_q16kv(
            const float *Q,
            const Q16_1Tensor *K_q16, const Q16_1Tensor *V_q16,
            float *output,
            int seq_len, int kv_len,
            int n_heads, int n_kv_heads, int head_dim,
            bool causal, int window_size, int position_offset)
        {
            KERNEL_PROFILE_SCOPE(KernelType::ATTENTION);

            if (!Q || !K_q16 || !V_q16 || !output)
                return false;
            if (seq_len <= 0 || kv_len <= 0 || n_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0)
                return false;
            if (n_heads % n_kv_heads != 0)
                return false;

            const int heads_per_kv = n_heads / n_kv_heads;
            const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

            // Q16_1 block layout
            const Q16BlockSize blk_size = K_q16->q16_block_size();
            const size_t block_bytes = q16_block_size_bytes(blk_size);
            const size_t block_elems = q16_block_size_elements(blk_size);
            const size_t blocks_per_kv_row = K_q16->blocks_per_row();
            constexpr size_t QS_OFFSET = sizeof(float) + sizeof(int32_t); // = 8 bytes

            const uint8_t *k_raw = static_cast<const uint8_t *>(K_q16->raw_data());
            const uint8_t *v_raw = static_cast<const uint8_t *>(V_q16->raw_data());
            if (!k_raw || !v_raw)
                return false;

            const int q_stride = n_heads * head_dim;

            // KV tile size for prefill
            const bool is_decode = (seq_len == 1 && kv_len >= 1);
            const int kv_tile = detail::DefaultFlashKVTilePolicy::choose(
                head_dim, n_kv_heads, kv_len, is_decode);

            const bool profiling_enabled = KernelProfiler::isEnabled();
            uint64_t qk_duration_ns = 0;
            uint64_t v_duration_ns = 0;

            const int attn_threads = computeOptimalAttentionThreads(
                n_heads, seq_len, kv_len, head_dim, causal);

            // For VNNI QK: Q is quantized per-head to int16
            constexpr int QMAX = 2047;

            auto work = [&]()
            {
                thread_local std::vector<int16_t> q_i16_buf;
                if (static_cast<int>(q_i16_buf.size()) < head_dim)
                    q_i16_buf.resize(static_cast<size_t>(head_dim));

#pragma omp for schedule(static) reduction(+ : qk_duration_ns, v_duration_ns)
                for (int h = 0; h < n_heads; ++h)
                {
                    const int kv_h = h / heads_per_kv;

                    // Hoist per-head block layout invariants
                    const size_t blocks_per_head = (static_cast<size_t>(head_dim) + block_elems - 1) / block_elems;
                    const size_t head_block_start = static_cast<size_t>(kv_h) * blocks_per_head;
                    const size_t row_stride_bytes = blocks_per_kv_row * block_bytes;
                    const size_t blk_off_bytes = head_block_start * block_bytes;

                    thread_local std::vector<float> block_scores;

                    for (int q_pos = 0; q_pos < seq_len; ++q_pos)
                    {
                        float *out = output + static_cast<size_t>(q_pos) * q_stride + static_cast<size_t>(h) * head_dim;
                        std::fill(out, out + head_dim, 0.0f);

                        float running_m = -std::numeric_limits<float>::infinity();
                        float running_l = 0.0f;

                        const float *q_ptr = Q + static_cast<size_t>(q_pos) * q_stride + static_cast<size_t>(h) * head_dim;
                        const int q_abs = position_offset + q_pos;

                        // Quantize Q to int16 once per (head, q_pos)
                        const float q_scale = quantize_row_i16_i12(
                            q_ptr, q_i16_buf.data(), head_dim, QMAX);
                        const float qk_combined_scale = q_scale * scale;

                        // KV tile loop
                        for (int k0 = 0; k0 < kv_len; k0 += kv_tile)
                        {
                            const int k1 = std::min(k0 + kv_tile, kv_len);
                            float block_max = -std::numeric_limits<float>::infinity();

                            const int blk = k1 - k0;
                            if (static_cast<int>(block_scores.size()) < blk)
                                block_scores.resize(static_cast<size_t>(blk));

                            // Valid window for causal/sliding-window masking
                            int valid_start = k0;
                            int valid_end = k1;
                            if (window_size > 0)
                                valid_start = std::max(valid_start, q_abs - window_size + 1);
                            if (causal)
                                valid_end = std::min(valid_end, q_abs + 1);
                            valid_start = std::max(k0, std::min(valid_start, k1));
                            valid_end = std::max(k0, std::min(valid_end, k1));

                            // Fill masked positions
                            for (int k = k0; k < valid_start; ++k)
                                block_scores[static_cast<size_t>(k - k0)] = -std::numeric_limits<float>::infinity();
                            for (int k = valid_end; k < k1; ++k)
                                block_scores[static_cast<size_t>(k - k0)] = -std::numeric_limits<float>::infinity();

                            const auto qk_start = profiling_enabled
                                                      ? std::chrono::steady_clock::now()
                                                      : std::chrono::steady_clock::time_point();

                            // --- QK Phase: VNNI int16 dot products ---
                            for (int k = valid_start; k < valid_end;)
                            {
                                if (blocks_per_head == 1)
                                {
                                    // Fast path: 1 block per head
                                    if (k + 3 < valid_end)
                                    {
                                        const uint8_t *b0 = k_raw + static_cast<size_t>(k + 0) * row_stride_bytes + blk_off_bytes;
                                        const uint8_t *b1 = k_raw + static_cast<size_t>(k + 1) * row_stride_bytes + blk_off_bytes;
                                        const uint8_t *b2 = k_raw + static_cast<size_t>(k + 2) * row_stride_bytes + blk_off_bytes;
                                        const uint8_t *b3 = k_raw + static_cast<size_t>(k + 3) * row_stride_bytes + blk_off_bytes;

                                        if (k + 7 < valid_end)
                                        {
                                            _mm_prefetch(reinterpret_cast<const char *>(
                                                             k_raw + static_cast<size_t>(k + 4) * row_stride_bytes + blk_off_bytes),
                                                         _MM_HINT_T0);
                                            _mm_prefetch(reinterpret_cast<const char *>(
                                                             k_raw + static_cast<size_t>(k + 5) * row_stride_bytes + blk_off_bytes),
                                                         _MM_HINT_T0);
                                            _mm_prefetch(reinterpret_cast<const char *>(
                                                             k_raw + static_cast<size_t>(k + 6) * row_stride_bytes + blk_off_bytes),
                                                         _MM_HINT_T0);
                                            _mm_prefetch(reinterpret_cast<const char *>(
                                                             k_raw + static_cast<size_t>(k + 7) * row_stride_bytes + blk_off_bytes),
                                                         _MM_HINT_T0);
                                        }

                                        float kd0, kd1, kd2, kd3;
                                        std::memcpy(&kd0, b0, sizeof(float));
                                        std::memcpy(&kd1, b1, sizeof(float));
                                        std::memcpy(&kd2, b2, sizeof(float));
                                        std::memcpy(&kd3, b3, sizeof(float));

                                        const int16_t *k_qs0 = reinterpret_cast<const int16_t *>(b0 + QS_OFFSET);
                                        const int16_t *k_qs1 = reinterpret_cast<const int16_t *>(b1 + QS_OFFSET);
                                        const int16_t *k_qs2 = reinterpret_cast<const int16_t *>(b2 + QS_OFFSET);
                                        const int16_t *k_qs3 = reinterpret_cast<const int16_t *>(b3 + QS_OFFSET);

                                        int32_t dot0, dot1, dot2, dot3;
                                        dot_i16_i16_i32_vnni_4row(
                                            q_i16_buf.data(), k_qs0, k_qs1, k_qs2, k_qs3,
                                            head_dim, dot0, dot1, dot2, dot3);

                                        float s0 = static_cast<float>(dot0) * qk_combined_scale * kd0;
                                        float s1 = static_cast<float>(dot1) * qk_combined_scale * kd1;
                                        float s2 = static_cast<float>(dot2) * qk_combined_scale * kd2;
                                        float s3 = static_cast<float>(dot3) * qk_combined_scale * kd3;

                                        block_scores[static_cast<size_t>(k - k0 + 0)] = s0;
                                        block_scores[static_cast<size_t>(k - k0 + 1)] = s1;
                                        block_scores[static_cast<size_t>(k - k0 + 2)] = s2;
                                        block_scores[static_cast<size_t>(k - k0 + 3)] = s3;
                                        block_max = std::max(block_max, std::max(std::max(s0, s1), std::max(s2, s3)));
                                        k += 4;
                                        continue;
                                    }

                                    // Scalar tail
                                    const uint8_t *blk_s = k_raw + static_cast<size_t>(k) * row_stride_bytes + blk_off_bytes;
                                    float kd;
                                    std::memcpy(&kd, blk_s, sizeof(float));
                                    const int16_t *k_qs = reinterpret_cast<const int16_t *>(blk_s + QS_OFFSET);
                                    int32_t dot = dot_i16_i16_i32_vnni(q_i16_buf.data(), k_qs, head_dim);
                                    float s = static_cast<float>(dot) * qk_combined_scale * kd;
                                    block_scores[static_cast<size_t>(k - k0)] = s;
                                    block_max = std::max(block_max, s);
                                    ++k;
                                }
                                else
                                {
                                    // Multi-block-per-head fallback
                                    float s = 0.0f;
                                    for (size_t bi = 0; bi < blocks_per_head; ++bi)
                                    {
                                        const size_t blk_idx = static_cast<size_t>(k) * blocks_per_kv_row + head_block_start + bi;
                                        const uint8_t *blk_ptr = k_raw + blk_idx * block_bytes;
                                        float kd;
                                        std::memcpy(&kd, blk_ptr, sizeof(float));
                                        const int16_t *k_qs = reinterpret_cast<const int16_t *>(blk_ptr + QS_OFFSET);
                                        const int elem_count = static_cast<int>(
                                            std::min(block_elems, static_cast<size_t>(head_dim) - bi * block_elems));
                                        int32_t dot = dot_i16_i16_i32_vnni(
                                            q_i16_buf.data() + bi * block_elems, k_qs, elem_count);
                                        s += static_cast<float>(dot) * q_scale * kd;
                                    }
                                    s *= scale;
                                    block_scores[static_cast<size_t>(k - k0)] = s;
                                    block_max = std::max(block_max, s);
                                    ++k;
                                }
                            }

                            if (profiling_enabled)
                                qk_duration_ns += static_cast<uint64_t>(
                                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                                        std::chrono::steady_clock::now() - qk_start)
                                        .count());

                            // --- Online softmax correction ---
                            const float new_m = std::max(running_m, block_max);
                            const float alpha = std::isfinite(running_m)
                                                    ? std::exp(running_m - new_m)
                                                    : 0.0f;
                            {
                                const __m512 va = _mm512_set1_ps(alpha);
                                int sd = 0;
                                for (; sd + 15 < head_dim; sd += 16)
                                    _mm512_storeu_ps(out + sd, _mm512_mul_ps(va, _mm512_loadu_ps(out + sd)));
                                for (; sd < head_dim; ++sd)
                                    out[sd] *= alpha;
                            }
                            float new_l = running_l * alpha;

                            // --- V Phase: Q16_1 V weighted accumulation ---
                            const auto v_start = profiling_enabled
                                                     ? std::chrono::steady_clock::now()
                                                     : std::chrono::steady_clock::time_point();

                            if (blocks_per_head == 1)
                            {
                                int k = valid_start;
                                for (; k + 3 < valid_end; k += 4)
                                {
                                    if (k + 7 < valid_end)
                                    {
                                        _mm_prefetch(reinterpret_cast<const char *>(
                                                         v_raw + static_cast<size_t>(k + 4) * row_stride_bytes + blk_off_bytes),
                                                     _MM_HINT_T0);
                                        _mm_prefetch(reinterpret_cast<const char *>(
                                                         v_raw + static_cast<size_t>(k + 5) * row_stride_bytes + blk_off_bytes),
                                                     _MM_HINT_T0);
                                        _mm_prefetch(reinterpret_cast<const char *>(
                                                         v_raw + static_cast<size_t>(k + 6) * row_stride_bytes + blk_off_bytes),
                                                     _MM_HINT_T0);
                                        _mm_prefetch(reinterpret_cast<const char *>(
                                                         v_raw + static_cast<size_t>(k + 7) * row_stride_bytes + blk_off_bytes),
                                                     _MM_HINT_T0);
                                    }

                                    const __m128 scores4 = _mm_set_ps(
                                        block_scores[static_cast<size_t>(k - k0 + 3)],
                                        block_scores[static_cast<size_t>(k - k0 + 2)],
                                        block_scores[static_cast<size_t>(k - k0 + 1)],
                                        block_scores[static_cast<size_t>(k - k0 + 0)]);
                                    const __m128 nm4 = _mm_set1_ps(new_m);
                                    const __m512 exp_in = _mm512_castps128_ps512(_mm_sub_ps(scores4, nm4));
                                    const __m512 exp_out = fast_exp_avx512(exp_in);
                                    const __m128 probs = _mm512_castps512_ps128(exp_out);
                                    alignas(16) float pp[4];
                                    _mm_store_ps(pp, probs);
                                    new_l += pp[0] + pp[1] + pp[2] + pp[3];

                                    const uint8_t *vb0 = v_raw + static_cast<size_t>(k + 0) * row_stride_bytes + blk_off_bytes;
                                    const uint8_t *vb1 = v_raw + static_cast<size_t>(k + 1) * row_stride_bytes + blk_off_bytes;
                                    const uint8_t *vb2 = v_raw + static_cast<size_t>(k + 2) * row_stride_bytes + blk_off_bytes;
                                    const uint8_t *vb3 = v_raw + static_cast<size_t>(k + 3) * row_stride_bytes + blk_off_bytes;

                                    float vd0, vd1, vd2, vd3;
                                    std::memcpy(&vd0, vb0, sizeof(float));
                                    std::memcpy(&vd1, vb1, sizeof(float));
                                    std::memcpy(&vd2, vb2, sizeof(float));
                                    std::memcpy(&vd3, vb3, sizeof(float));

                                    accum_weighted_v_q16_4row(
                                        out,
                                        reinterpret_cast<const int16_t *>(vb0 + QS_OFFSET), pp[0] * vd0,
                                        reinterpret_cast<const int16_t *>(vb1 + QS_OFFSET), pp[1] * vd1,
                                        reinterpret_cast<const int16_t *>(vb2 + QS_OFFSET), pp[2] * vd2,
                                        reinterpret_cast<const int16_t *>(vb3 + QS_OFFSET), pp[3] * vd3,
                                        head_dim);
                                }

                                // Scalar tail
                                for (; k < valid_end; ++k)
                                {
                                    const float p = std::exp(block_scores[static_cast<size_t>(k - k0)] - new_m);
                                    new_l += p;
                                    const uint8_t *blk_ptr = v_raw + static_cast<size_t>(k) * row_stride_bytes + blk_off_bytes;
                                    float vd;
                                    std::memcpy(&vd, blk_ptr, sizeof(float));
                                    accum_weighted_v_q16(out,
                                                         reinterpret_cast<const int16_t *>(blk_ptr + QS_OFFSET),
                                                         p * vd, head_dim);
                                }
                            }
                            else
                            {
                                // Multi-block fallback
                                for (int k = valid_start; k < valid_end; ++k)
                                {
                                    const float p = std::exp(block_scores[static_cast<size_t>(k - k0)] - new_m);
                                    new_l += p;
                                    for (size_t bi = 0; bi < blocks_per_head; ++bi)
                                    {
                                        const size_t blk_idx = static_cast<size_t>(k) * blocks_per_kv_row + head_block_start + bi;
                                        const uint8_t *blk_ptr = v_raw + blk_idx * block_bytes;
                                        float vd;
                                        std::memcpy(&vd, blk_ptr, sizeof(float));
                                        const int16_t *v_qs = reinterpret_cast<const int16_t *>(blk_ptr + QS_OFFSET);
                                        const int elem_count = static_cast<int>(
                                            std::min(block_elems, static_cast<size_t>(head_dim) - bi * block_elems));
                                        accum_weighted_v_q16(
                                            out + bi * block_elems, v_qs, p * vd, elem_count);
                                    }
                                }
                            }

                            if (profiling_enabled)
                                v_duration_ns += static_cast<uint64_t>(
                                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                                        std::chrono::steady_clock::now() - v_start)
                                        .count());

                            running_m = new_m;
                            running_l = new_l;
                        } // end KV tile loop

                        // Final normalisation
                        if (running_l > 0.0f)
                        {
                            const float inv_l = 1.0f / running_l;
                            const __m512 vi = _mm512_set1_ps(inv_l);
                            int sd = 0;
                            for (; sd + 15 < head_dim; sd += 16)
                                _mm512_storeu_ps(out + sd, _mm512_mul_ps(vi, _mm512_loadu_ps(out + sd)));
                            for (; sd < head_dim; ++sd)
                                out[sd] *= inv_l;
                        }
                    } // end q_pos loop
                } // end head loop
            };

            // Threading: prefill has enough work for all threads
            OMP_WORKSHARE_REGION(work);
            const int actual_threads = omp_get_max_threads();

            if (profiling_enabled)
            {
                KernelProfiler::recordParallel(KernelType::ATTENTION_QK, qk_duration_ns, actual_threads);
                KernelProfiler::recordParallel(KernelType::ATTENTION_V, v_duration_ns, actual_threads);
            }
            return true;
        }
#endif // __AVX512F__ && __AVX512VNNI__

#if defined(__AVX512F__) && defined(__AVX512VNNI__)
        /**
         * @brief Decode-only flash attention with Q16_1 KV cache (VNNI accelerated).
         *
         * Reads K and V directly from Q16_1Tensor block storage. The QK dot
         * product uses VPDPWSSD (int16×int16→int32) against the block's qs[]
         * array, with the query quantized on-the-fly to int16. The V phase
         * loads int16 values, widens to FP32, and FMA-accumulates weighted
         * by the softmax probability × block scale.
         *
         * Bandwidth savings: 2 bytes/element (int16 qs[]) + 8 bytes/block
         * overhead vs 4 bytes/element for FP32. For head_dim=128 with BLOCK_128:
         * 264 bytes/head vs 512 bytes FP32 = 48% bandwidth reduction.
         *
         * Only supports the decode path (seq_len == 1, kv_len >= 1).
         */
        static bool compute_decode_q16kv(
            const float *Q,
            const Q16_1Tensor *K_q16, const Q16_1Tensor *V_q16,
            float *output,
            int kv_len,
            int n_heads, int n_kv_heads, int head_dim,
            bool causal, int position_offset)
        {
            KERNEL_PROFILE_SCOPE(KernelType::ATTENTION);

            if (!Q || !K_q16 || !V_q16 || !output)
                return false;
            if (kv_len <= 0 || n_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0)
                return false;
            if (n_heads % n_kv_heads != 0)
                return false;

            const int heads_per_kv = n_heads / n_kv_heads;
            const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

            // Q16_1 block layout: all blocks have {float d, int32_t sum_qs, int16_t qs[block_elems]}
            // The qs offset is always 8 bytes from block start (sizeof(float) + sizeof(int32_t)).
            const Q16BlockSize blk_size = K_q16->q16_block_size();
            const size_t block_bytes = q16_block_size_bytes(blk_size);
            const size_t block_elems = q16_block_size_elements(blk_size);
            const size_t blocks_per_kv_row = K_q16->blocks_per_row();
            constexpr size_t QS_OFFSET = sizeof(float) + sizeof(int32_t); // = 8 bytes

            // Detect HEAD_MAJOR layout: Q16_1 KV caches store data as [head][pos][head_dim],
            // where each row contains one head's data (blocks_per_row == blocks_per_head).
            // POSITION_MAJOR stores [pos][all_heads] with blocks_per_row == n_kv_heads * blocks_per_head.
            const size_t blocks_per_head_detect = (static_cast<size_t>(head_dim) + block_elems - 1) / block_elems;
            const bool is_head_major = (blocks_per_kv_row == blocks_per_head_detect && n_kv_heads > 1);
            // For HEAD_MAJOR: number of physical rows allocated per head (max_seq_len)
            const size_t rows_per_head = is_head_major
                                             ? (K_q16->rows() / static_cast<size_t>(n_kv_heads))
                                             : 0;

            // Raw byte data for direct block access (avoids block-size-templated dispatch)
            const uint8_t *k_raw = static_cast<const uint8_t *>(K_q16->raw_data());
            const uint8_t *v_raw = static_cast<const uint8_t *>(V_q16->raw_data());

            if (!k_raw || !v_raw)
                return false;

            // KV tile size for decode
            const int kv_tile = detail::DefaultFlashKVTilePolicy::choose(
                head_dim, n_kv_heads, kv_len, /*is_decode=*/true);

            const bool profiling_enabled = KernelProfiler::isEnabled();
            uint64_t qk_duration_ns = 0;
            uint64_t v_duration_ns = 0;

            const int attn_threads = computeOptimalAttentionThreads(
                n_heads, 1, kv_len, head_dim, causal);

            // For VNNI QK: Q is quantized once per head to int16 (head_dim elements).
            // qmax=2047 ≈ 12 effective bits, matching the existing I12 scheme.
            constexpr int QMAX = 2047;

            auto work = [&]()
            {
                // Per-thread Q quantisation buffer
                thread_local std::vector<int16_t> q_i16_buf;
                if (static_cast<int>(q_i16_buf.size()) < head_dim)
                    q_i16_buf.resize(static_cast<size_t>(head_dim));

#pragma omp for schedule(static) reduction(+ : qk_duration_ns, v_duration_ns)
                for (int h = 0; h < n_heads; ++h)
                {
                    const int kv_h = h / heads_per_kv;
                    float *out = output + static_cast<size_t>(h) * head_dim;
                    std::fill(out, out + head_dim, 0.0f);

                    float running_m = -std::numeric_limits<float>::infinity();
                    float running_l = 0.0f;

                    const float *q_ptr = Q + static_cast<size_t>(h) * head_dim;
                    const int q_abs = position_offset;

                    // Quantize Q to int16 once per head
                    const float q_scale = quantize_row_i16_i12(
                        q_ptr, q_i16_buf.data(), head_dim, QMAX);

                    // Precompute combined scale: q_scale * (1/√d)
                    // This saves one multiply per QK score reconstruction.
                    const float qk_combined_scale = q_scale * scale;

                    // Hoist per-head block layout invariants outside the tile loop
                    const size_t blocks_per_head = (static_cast<size_t>(head_dim) + block_elems - 1) / block_elems;
                    const size_t row_stride_bytes = blocks_per_kv_row * block_bytes;
                    // HEAD_MAJOR: head data at offset kv_h * rows_per_head * row_stride
                    // POSITION_MAJOR: head data at offset kv_h * blocks_per_head * block_bytes within each row
                    const size_t blk_off_bytes = is_head_major
                                                     ? (static_cast<size_t>(kv_h) * rows_per_head * blocks_per_head * block_bytes)
                                                     : (static_cast<size_t>(kv_h) * blocks_per_head * block_bytes);
                    const size_t head_block_start = is_head_major
                                                        ? (static_cast<size_t>(kv_h) * rows_per_head * blocks_per_head)
                                                        : (static_cast<size_t>(kv_h) * blocks_per_head);

                    thread_local std::vector<float> block_scores;

                    for (int k0 = 0; k0 < kv_len; k0 += kv_tile)
                    {
                        const int k1 = std::min(k0 + kv_tile, kv_len);
                        float block_max = -std::numeric_limits<float>::infinity();

                        const int blk = k1 - k0;
                        if (static_cast<int>(block_scores.size()) < blk)
                            block_scores.resize(static_cast<size_t>(blk));

                        // Valid window for causal masking
                        int valid_end = causal ? std::min(k1, q_abs + 1) : k1;
                        valid_end = std::max(k0, std::min(valid_end, k1));

                        // Fill masked positions
                        for (int k = valid_end; k < k1; ++k)
                            block_scores[static_cast<size_t>(k - k0)] = -std::numeric_limits<float>::infinity();

                        const auto qk_start = profiling_enabled
                                                  ? std::chrono::steady_clock::now()
                                                  : std::chrono::steady_clock::time_point();

                        // --- QK Phase: VNNI int16 dot products ---

                        for (int k = k0; k < valid_end;)
                        {
                            if (blocks_per_head == 1)
                            {
                                // Fast path: 1 block per head — qs[] is contiguous head_dim int16s
                                if (k + 3 < valid_end)
                                {
                                    const uint8_t *b0 = k_raw + static_cast<size_t>(k + 0) * row_stride_bytes + blk_off_bytes;
                                    const uint8_t *b1 = k_raw + static_cast<size_t>(k + 1) * row_stride_bytes + blk_off_bytes;
                                    const uint8_t *b2 = k_raw + static_cast<size_t>(k + 2) * row_stride_bytes + blk_off_bytes;
                                    const uint8_t *b3 = k_raw + static_cast<size_t>(k + 3) * row_stride_bytes + blk_off_bytes;

                                    // Prefetch next 4 K blocks (stride is too large for HW prefetcher)
                                    if (k + 7 < valid_end)
                                    {
                                        _mm_prefetch(reinterpret_cast<const char *>(
                                                         k_raw + static_cast<size_t>(k + 4) * row_stride_bytes + blk_off_bytes),
                                                     _MM_HINT_T0);
                                        _mm_prefetch(reinterpret_cast<const char *>(
                                                         k_raw + static_cast<size_t>(k + 5) * row_stride_bytes + blk_off_bytes),
                                                     _MM_HINT_T0);
                                        _mm_prefetch(reinterpret_cast<const char *>(
                                                         k_raw + static_cast<size_t>(k + 6) * row_stride_bytes + blk_off_bytes),
                                                     _MM_HINT_T0);
                                        _mm_prefetch(reinterpret_cast<const char *>(
                                                         k_raw + static_cast<size_t>(k + 7) * row_stride_bytes + blk_off_bytes),
                                                     _MM_HINT_T0);
                                    }

                                    // Get K block scales (float d at offset 0 of each block)
                                    float kd0, kd1, kd2, kd3;
                                    std::memcpy(&kd0, b0, sizeof(float));
                                    std::memcpy(&kd1, b1, sizeof(float));
                                    std::memcpy(&kd2, b2, sizeof(float));
                                    std::memcpy(&kd3, b3, sizeof(float));

                                    // Get qs[] pointers (at QS_OFFSET into each block)
                                    const int16_t *k_qs0 = reinterpret_cast<const int16_t *>(b0 + QS_OFFSET);
                                    const int16_t *k_qs1 = reinterpret_cast<const int16_t *>(b1 + QS_OFFSET);
                                    const int16_t *k_qs2 = reinterpret_cast<const int16_t *>(b2 + QS_OFFSET);
                                    const int16_t *k_qs3 = reinterpret_cast<const int16_t *>(b3 + QS_OFFSET);

                                    int32_t dot0, dot1, dot2, dot3;
                                    dot_i16_i16_i32_vnni_4row(
                                        q_i16_buf.data(), k_qs0, k_qs1, k_qs2, k_qs3,
                                        head_dim, dot0, dot1, dot2, dot3);

                                    // Score: int_dot * (q_scale / √d) * k_scale
                                    float s0 = static_cast<float>(dot0) * qk_combined_scale * kd0;
                                    float s1 = static_cast<float>(dot1) * qk_combined_scale * kd1;
                                    float s2 = static_cast<float>(dot2) * qk_combined_scale * kd2;
                                    float s3 = static_cast<float>(dot3) * qk_combined_scale * kd3;

                                    block_scores[static_cast<size_t>(k - k0 + 0)] = s0;
                                    block_scores[static_cast<size_t>(k - k0 + 1)] = s1;
                                    block_scores[static_cast<size_t>(k - k0 + 2)] = s2;
                                    block_scores[static_cast<size_t>(k - k0 + 3)] = s3;
                                    block_max = std::max(block_max, std::max(std::max(s0, s1), std::max(s2, s3)));
                                    k += 4;
                                    continue;
                                }

                                // Scalar tail: 1 row at a time
                                const uint8_t *blk_s = k_raw + static_cast<size_t>(k) * row_stride_bytes + blk_off_bytes;
                                float kd;
                                std::memcpy(&kd, blk_s, sizeof(float));
                                const int16_t *k_qs = reinterpret_cast<const int16_t *>(blk_s + QS_OFFSET);
                                int32_t dot = dot_i16_i16_i32_vnni(q_i16_buf.data(), k_qs, head_dim);
                                float s = static_cast<float>(dot) * qk_combined_scale * kd;
                                block_scores[static_cast<size_t>(k - k0)] = s;
                                block_max = std::max(block_max, s);
                                ++k;
                            }
                            else
                            {
                                // Multi-block-per-head fallback: accumulate across blocks
                                float s = 0.0f;
                                for (size_t bi = 0; bi < blocks_per_head; ++bi)
                                {
                                    const size_t blk_idx = static_cast<size_t>(k) * blocks_per_kv_row + head_block_start + bi;
                                    const uint8_t *blk_ptr = k_raw + blk_idx * block_bytes;
                                    float kd;
                                    std::memcpy(&kd, blk_ptr, sizeof(float));
                                    const int16_t *k_qs = reinterpret_cast<const int16_t *>(blk_ptr + QS_OFFSET);
                                    const int elem_count = static_cast<int>(
                                        std::min(block_elems, static_cast<size_t>(head_dim) - bi * block_elems));
                                    int32_t dot = dot_i16_i16_i32_vnni(
                                        q_i16_buf.data() + bi * block_elems, k_qs, elem_count);
                                    s += static_cast<float>(dot) * q_scale * kd;
                                }
                                s *= scale;
                                block_scores[static_cast<size_t>(k - k0)] = s;
                                block_max = std::max(block_max, s);
                                ++k;
                            }
                        }

                        if (profiling_enabled)
                            qk_duration_ns += static_cast<uint64_t>(
                                std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    std::chrono::steady_clock::now() - qk_start)
                                    .count());

                        // --- Online softmax correction ---
                        const float new_m = std::max(running_m, block_max);
                        const float alpha = std::isfinite(running_m)
                                                ? std::exp(running_m - new_m)
                                                : 0.0f;
                        {
                            // Scale output: out[d] *= alpha (inline to avoid F16C guard dep)
                            const __m512 va = _mm512_set1_ps(alpha);
                            int sd = 0;
                            for (; sd + 15 < head_dim; sd += 16)
                                _mm512_storeu_ps(out + sd, _mm512_mul_ps(va, _mm512_loadu_ps(out + sd)));
                            for (; sd < head_dim; ++sd)
                                out[sd] *= alpha;
                        }
                        float new_l = running_l * alpha;

                        // --- V Phase: Q16_1 V weighted accumulation ---
                        const auto v_start = profiling_enabled
                                                 ? std::chrono::steady_clock::now()
                                                 : std::chrono::steady_clock::time_point();

                        if (blocks_per_head == 1)
                        {
                            // Fast path: 4-wide batched V accumulation
                            // Positions in [k0, valid_end) are guaranteed finite (computed in QK).
                            // Positions in [valid_end, k1) are -inf (masked) — skip them.
                            int k = k0;
                            for (; k + 3 < valid_end; k += 4)
                            {
                                // Prefetch next 4 V blocks
                                if (k + 7 < valid_end)
                                {
                                    _mm_prefetch(reinterpret_cast<const char *>(
                                                     v_raw + static_cast<size_t>(k + 4) * row_stride_bytes + blk_off_bytes),
                                                 _MM_HINT_T0);
                                    _mm_prefetch(reinterpret_cast<const char *>(
                                                     v_raw + static_cast<size_t>(k + 5) * row_stride_bytes + blk_off_bytes),
                                                 _MM_HINT_T0);
                                    _mm_prefetch(reinterpret_cast<const char *>(
                                                     v_raw + static_cast<size_t>(k + 6) * row_stride_bytes + blk_off_bytes),
                                                 _MM_HINT_T0);
                                    _mm_prefetch(reinterpret_cast<const char *>(
                                                     v_raw + static_cast<size_t>(k + 7) * row_stride_bytes + blk_off_bytes),
                                                 _MM_HINT_T0);
                                }

                                // Vectorized exp: compute 4 softmax weights in one shot
                                // using fast_exp_avx512 (< 1 ULP accuracy, ~5x faster than
                                // 4× scalar std::exp on Cascade Lake).
                                const __m128 scores4 = _mm_set_ps(
                                    block_scores[static_cast<size_t>(k - k0 + 3)],
                                    block_scores[static_cast<size_t>(k - k0 + 2)],
                                    block_scores[static_cast<size_t>(k - k0 + 1)],
                                    block_scores[static_cast<size_t>(k - k0 + 0)]);
                                const __m128 nm4 = _mm_set1_ps(new_m);
                                // Broadcast 4 scores to __m512, compute exp, extract lower 4
                                const __m512 exp_in = _mm512_castps128_ps512(_mm_sub_ps(scores4, nm4));
                                const __m512 exp_out = fast_exp_avx512(exp_in);
                                const __m128 probs = _mm512_castps512_ps128(exp_out);
                                alignas(16) float pp[4];
                                _mm_store_ps(pp, probs);
                                const float p0 = pp[0], p1 = pp[1], p2 = pp[2], p3 = pp[3];
                                new_l += p0 + p1 + p2 + p3;

                                const uint8_t *vb0 = v_raw + static_cast<size_t>(k + 0) * row_stride_bytes + blk_off_bytes;
                                const uint8_t *vb1 = v_raw + static_cast<size_t>(k + 1) * row_stride_bytes + blk_off_bytes;
                                const uint8_t *vb2 = v_raw + static_cast<size_t>(k + 2) * row_stride_bytes + blk_off_bytes;
                                const uint8_t *vb3 = v_raw + static_cast<size_t>(k + 3) * row_stride_bytes + blk_off_bytes;

                                float vd0, vd1, vd2, vd3;
                                std::memcpy(&vd0, vb0, sizeof(float));
                                std::memcpy(&vd1, vb1, sizeof(float));
                                std::memcpy(&vd2, vb2, sizeof(float));
                                std::memcpy(&vd3, vb3, sizeof(float));

                                accum_weighted_v_q16_4row(
                                    out,
                                    reinterpret_cast<const int16_t *>(vb0 + QS_OFFSET), p0 * vd0,
                                    reinterpret_cast<const int16_t *>(vb1 + QS_OFFSET), p1 * vd1,
                                    reinterpret_cast<const int16_t *>(vb2 + QS_OFFSET), p2 * vd2,
                                    reinterpret_cast<const int16_t *>(vb3 + QS_OFFSET), p3 * vd3,
                                    head_dim);
                            }

                            // Scalar tail: remaining valid positions
                            for (; k < valid_end; ++k)
                            {
                                const float p = std::exp(block_scores[static_cast<size_t>(k - k0)] - new_m);
                                new_l += p;
                                const uint8_t *blk_ptr = v_raw + static_cast<size_t>(k) * row_stride_bytes + blk_off_bytes;
                                float vd;
                                std::memcpy(&vd, blk_ptr, sizeof(float));
                                accum_weighted_v_q16(out,
                                                     reinterpret_cast<const int16_t *>(blk_ptr + QS_OFFSET),
                                                     p * vd, head_dim);
                            }
                            // Positions [valid_end, k1) are masked — no V accumulation needed
                        }
                        else
                        {
                            // Multi-block fallback (unchanged)
                            for (int k = k0; k < k1; ++k)
                            {
                                const float s = block_scores[static_cast<size_t>(k - k0)];
                                if (!std::isfinite(s))
                                    continue;
                                const float p = std::exp(s - new_m);
                                new_l += p;

                                for (size_t bi = 0; bi < blocks_per_head; ++bi)
                                {
                                    const size_t blk_idx = static_cast<size_t>(k) * blocks_per_kv_row + head_block_start + bi;
                                    const uint8_t *blk_ptr = v_raw + blk_idx * block_bytes;
                                    float vd;
                                    std::memcpy(&vd, blk_ptr, sizeof(float));
                                    const int16_t *v_qs = reinterpret_cast<const int16_t *>(blk_ptr + QS_OFFSET);
                                    const int elem_count = static_cast<int>(
                                        std::min(block_elems, static_cast<size_t>(head_dim) - bi * block_elems));
                                    accum_weighted_v_q16(
                                        out + bi * block_elems, v_qs, p * vd, elem_count);
                                }
                            }
                        }

                        if (profiling_enabled)
                            v_duration_ns += static_cast<uint64_t>(
                                std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    std::chrono::steady_clock::now() - v_start)
                                    .count());

                        running_m = new_m;
                        running_l = new_l;
                    } // end KV tile loop

                    // Final normalisation
                    if (running_l > 0.0f)
                    {
                        const float inv_l = 1.0f / running_l;
                        const __m512 vi = _mm512_set1_ps(inv_l);
                        int sd = 0;
                        for (; sd + 15 < head_dim; sd += 16)
                            _mm512_storeu_ps(out + sd, _mm512_mul_ps(vi, _mm512_loadu_ps(out + sd)));
                        for (; sd < head_dim; ++sd)
                            out[sd] *= inv_l;
                    }
                } // end head loop
            };

            // Same decode threading strategy — see detailed comment in compute_flash_fp32.
            const bool force_full_pool_q16 = kv_len > 100;

            int actual_threads_q16;
            if (force_full_pool_q16)
            {
                OMP_WORKSHARE_REGION(work);
                actual_threads_q16 = omp_get_max_threads();
            }
            else
            {
#pragma omp parallel num_threads(attn_threads)
                {
                    work();
                }
                actual_threads_q16 = attn_threads;
            }

            if (profiling_enabled)
            {
                KernelProfiler::recordParallel(KernelType::ATTENTION_QK, qk_duration_ns, actual_threads_q16);
                KernelProfiler::recordParallel(KernelType::ATTENTION_V, v_duration_ns, actual_threads_q16);
            }
            return true;
        }
#endif // __AVX512F__ && __AVX512VNNI__

#if defined(__AVX512F__) && defined(__F16C__)
        /**
         * @brief Decode-only flash attention with FP16 KV cache (no FP32 copy).
         *
         * Reads K and V directly as FP16, converting to FP32 on the fly in the
         * dot product and V accumulation inner loops. This halves KV memory
         * bandwidth and eliminates the need for persistent FP32 KV buffers,
         * reducing DRAM bank disturbance that causes GEMM throughput degradation
         * at long context (>300 tokens).
         *
         * Only supports the decode path (seq_len == 1, kv_len >= 1).
         */
        static bool compute_decode_fp16kv(
            const float *Q,
            const uint16_t *K_fp16, const uint16_t *V_fp16,
            float *output,
            int kv_len,
            int n_heads, int n_kv_heads, int head_dim,
            bool causal, int position_offset)
        {
            KERNEL_PROFILE_SCOPE(KernelType::ATTENTION);

            if (!Q || !K_fp16 || !V_fp16 || !output)
                return false;
            if (kv_len <= 0 || n_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0)
                return false;
            if (n_heads % n_kv_heads != 0)
                return false;

            const int heads_per_kv = n_heads / n_kv_heads;
            const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
            const int q_stride = n_heads * head_dim;
            const int kv_stride = n_kv_heads * head_dim;

            // KV tile size for decode
            const bool is_decode = true;
            const int kv_tile = detail::DefaultFlashKVTilePolicy::choose(
                head_dim, n_kv_heads, kv_len, is_decode);

            const bool profiling_enabled = KernelProfiler::isEnabled();
            uint64_t qk_duration_ns = 0;
            uint64_t v_duration_ns = 0;

            const int attn_threads = computeOptimalAttentionThreads(
                n_heads, 1, kv_len, head_dim, causal);

            auto work = [&]()
            {
#pragma omp for schedule(static) reduction(+ : qk_duration_ns, v_duration_ns)
                for (int h = 0; h < n_heads; ++h)
                {
                    const int kv_h = h / heads_per_kv;
                    float *out = output + static_cast<size_t>(h) * head_dim;
                    std::fill(out, out + head_dim, 0.0f);

                    float running_m = -std::numeric_limits<float>::infinity();
                    float running_l = 0.0f;

                    const float *q_ptr = Q + static_cast<size_t>(h) * head_dim;
                    const int q_abs = position_offset;

                    thread_local std::vector<float> block_scores;

                    for (int k0 = 0; k0 < kv_len; k0 += kv_tile)
                    {
                        const int k1 = std::min(k0 + kv_tile, kv_len);
                        float block_max = -std::numeric_limits<float>::infinity();

                        const int blk = k1 - k0;
                        if (static_cast<int>(block_scores.size()) < blk)
                            block_scores.resize(blk);

                        // Valid window for causal masking
                        int valid_end = causal ? std::min(k1, q_abs + 1) : k1;
                        valid_end = std::max(k0, std::min(valid_end, k1));

                        // Fill masked positions
                        for (int k = valid_end; k < k1; ++k)
                            block_scores[static_cast<size_t>(k - k0)] = -std::numeric_limits<float>::infinity();

                        const auto qk_start = profiling_enabled
                                                  ? std::chrono::steady_clock::now()
                                                  : std::chrono::steady_clock::time_point();

                        // --- QK Phase: FP16 K dot products ---
                        for (int k = k0; k < valid_end;)
                        {
                            const uint16_t *kbase = K_fp16 + static_cast<size_t>(kv_h) * head_dim;

                            if (k + 3 < valid_end)
                            {
                                float s0, s1, s2, s3;
                                dot_fp16_avx512_4row(
                                    q_ptr,
                                    kbase + static_cast<size_t>(k + 0) * kv_stride,
                                    kbase + static_cast<size_t>(k + 1) * kv_stride,
                                    kbase + static_cast<size_t>(k + 2) * kv_stride,
                                    kbase + static_cast<size_t>(k + 3) * kv_stride,
                                    head_dim, s0, s1, s2, s3);
                                s0 *= scale;
                                s1 *= scale;
                                s2 *= scale;
                                s3 *= scale;
                                block_scores[static_cast<size_t>(k - k0 + 0)] = s0;
                                block_scores[static_cast<size_t>(k - k0 + 1)] = s1;
                                block_scores[static_cast<size_t>(k - k0 + 2)] = s2;
                                block_scores[static_cast<size_t>(k - k0 + 3)] = s3;
                                block_max = std::max(block_max, std::max(std::max(s0, s1), std::max(s2, s3)));
                                k += 4;
                                continue;
                            }

                            float s = dot_fp16_avx512(
                                q_ptr,
                                K_fp16 + static_cast<size_t>(k) * kv_stride + static_cast<size_t>(kv_h) * head_dim,
                                head_dim);
                            s *= scale;
                            block_scores[static_cast<size_t>(k - k0)] = s;
                            block_max = std::max(block_max, s);
                            ++k;
                        }

                        if (profiling_enabled)
                            qk_duration_ns += static_cast<uint64_t>(
                                std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    std::chrono::steady_clock::now() - qk_start)
                                    .count());

                        // --- Online softmax correction ---
                        const float new_m = std::max(running_m, block_max);
                        const float alpha = std::isfinite(running_m)
                                                ? std::exp(running_m - new_m)
                                                : 0.0f;
                        scale_vec_fp16path(out, alpha, head_dim);
                        float new_l = running_l * alpha;

                        // --- V Phase: FP16 V weighted accumulation ---
                        // Iterate [k0, valid_end) — all positions are guaranteed finite.
                        // Positions [valid_end, k1) are masked to -inf — skip them.
                        const auto v_start = profiling_enabled
                                                 ? std::chrono::steady_clock::now()
                                                 : std::chrono::steady_clock::time_point();

                        {
                            const uint16_t *v_base = V_fp16 + static_cast<size_t>(kv_h) * head_dim;
                            int k = k0;
                            // 4-wide batched: vectorized exp + batched FP16 V accumulation
                            for (; k + 3 < valid_end; k += 4)
                            {
                                const __m128 scores4 = _mm_set_ps(
                                    block_scores[static_cast<size_t>(k - k0 + 3)],
                                    block_scores[static_cast<size_t>(k - k0 + 2)],
                                    block_scores[static_cast<size_t>(k - k0 + 1)],
                                    block_scores[static_cast<size_t>(k - k0 + 0)]);
                                const __m128 nm4 = _mm_set1_ps(new_m);
                                const __m512 exp_in = _mm512_castps128_ps512(_mm_sub_ps(scores4, nm4));
                                const __m512 exp_out = fast_exp_avx512(exp_in);
                                const __m128 probs = _mm512_castps512_ps128(exp_out);
                                alignas(16) float pp[4];
                                _mm_store_ps(pp, probs);
                                new_l += pp[0] + pp[1] + pp[2] + pp[3];

                                accum_weighted_v_fp16_4row(
                                    out,
                                    v_base + static_cast<size_t>(k + 0) * kv_stride, pp[0],
                                    v_base + static_cast<size_t>(k + 1) * kv_stride, pp[1],
                                    v_base + static_cast<size_t>(k + 2) * kv_stride, pp[2],
                                    v_base + static_cast<size_t>(k + 3) * kv_stride, pp[3],
                                    head_dim);
                            }
                            // Scalar tail: remaining valid positions
                            for (; k < valid_end; ++k)
                            {
                                const float p = std::exp(block_scores[static_cast<size_t>(k - k0)] - new_m);
                                new_l += p;
                                const uint16_t *v_ptr = v_base + static_cast<size_t>(k) * kv_stride;
                                accum_weighted_v_fp16(out, v_ptr, p, head_dim);
                            }
                        }

                        if (profiling_enabled)
                            v_duration_ns += static_cast<uint64_t>(
                                std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    std::chrono::steady_clock::now() - v_start)
                                    .count());

                        running_m = new_m;
                        running_l = new_l;
                    } // end KV tile loop

                    // Final normalisation
                    if (running_l > 0.0f)
                    {
                        float inv_l = 1.0f / running_l;
                        scale_vec_fp16path(out, inv_l, head_dim);
                    }
                } // end head loop
            };

            // Same decode threading fix as FP32 path — see detailed comment there.
            // This path is always decode (kv_len != seq_len by construction).
            const bool force_full_pool_fp16 = kv_len > 100;

            int actual_threads_fp16;
            if (force_full_pool_fp16)
            {
                OMP_WORKSHARE_REGION(work);
                actual_threads_fp16 = omp_get_max_threads();
            }
            else
            {
#pragma omp parallel num_threads(attn_threads)
                {
                    work();
                }
                actual_threads_fp16 = attn_threads;
            }

            if (profiling_enabled)
            {
                KernelProfiler::recordParallel(KernelType::ATTENTION_QK, qk_duration_ns, actual_threads_fp16);
                KernelProfiler::recordParallel(KernelType::ATTENTION_V, v_duration_ns, actual_threads_fp16);
            }
            return true;
        }
#endif // __AVX512F__ && __F16C__
    };

    extern template class CPUFlashAttentionKernelT<ActivationPrecision::FP32>;
    extern template class CPUFlashAttentionKernelT<ActivationPrecision::BF16>;
    extern template class CPUFlashAttentionKernelT<ActivationPrecision::FP16>;

} // namespace llaminar2
