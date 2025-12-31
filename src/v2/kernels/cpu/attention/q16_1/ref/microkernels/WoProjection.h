/**
 * @file WoProjection.h
 * @brief Integer Wo projection microkernel: INT32 context → Q16_1 output
 *
 * This microkernel projects the INT32 context (from P×V accumulation) through
 * the Wo weight matrix to produce Q16_1 output blocks. This is the final
 * computation before residual addition.
 *
 * ## Algorithm
 *
 * For each output row d in [0, d_model):
 *   proj[d] = sum_{h=0}^{n_heads-1} sum_{i=0}^{head_dim-1} context[h*head_dim+i] × Wo[d, h*head_dim+i]
 *
 * ## Integer Pipeline
 *
 * 1. INT32 context normalized (divide by softmax sum) → INT16 range
 * 2. Wo weights: Q16_1 format (INT16 quantized with scale)
 * 3. INT16 × INT16 → INT32 accumulation (VPDPWSSD in JIT)
 * 4. INT32 accumulators → Q16_1 output blocks (scale computed from max)
 *
 * ## Memory Patterns
 *
 * - **Flash Decode**: GEMV pattern, stream Wo rows, context in L2
 * - **FA2 Prefill**: Batched GEMM, amortize Wo loads across queries
 *
 * ## Cache-Aware Tiling
 *
 * The Wo projection uses cache-aware tiling based on detected CPU cache hierarchy:
 *   - **Output tile (M_TILE)**: Number of output rows to process together
 *     - Targets L2: accumulators [M_TILE × 4 bytes] + context [input_dim × 2 bytes]
 *   - **Input tile (K_TILE)**: Reduction dimension chunk size
 *     - Targets L1: partial Wo [M_TILE × K_TILE × 2 bytes] + context [K_TILE × 2 bytes]
 *
 * @see docs/v2/PROJECT_Q16_INTEGER_ATTENTION_V2.md
 */

#pragma once

#include "tensors/BlockStructures.h"
#include "utils/CPUFeatures.h"
#include <cstdint>
#include <algorithm>

namespace llaminar2::kernels::q16_1::microkernels
{

    // Import Q16 types from BlockStructures.h
    using llaminar2::Q16_1Block_128;
    using llaminar2::Q16_1Block_64;
    using llaminar2::Q16BlockSize;

    // ============================================================================
    // Wo Projection Cache-Aware Tile Configuration
    // ============================================================================

    /**
     * @brief Cache-aware tile configuration for Wo projection.
     *
     * Wo projection is a GEMM/GEMV operation:
     *   - Decode: output[d_model] = context[input_dim] × Wo^T[d_model, input_dim]
     *   - Prefill: output[batch, d_model] = context[batch, input_dim] × Wo^T
     *
     * Tiling Strategy:
     *   - **M_tile (output rows)**: Process M_tile outputs at once
     *     - Accumulators [M_tile × 4 bytes] should fit in registers/L1
     *   - **K_tile (reduction dim)**: Stream input in K_tile chunks
     *     - Wo block [M_tile × K_tile × 2 bytes] + context [K_tile × 2 bytes] fit in L1
     *   - **N_tile (batch dim, prefill only)**: Process queries together
     *     - Amortizes Wo loads across multiple queries
     *
     * Memory Layout per tile:
     *   - Context tile: K_tile × 2 bytes (INT16)
     *   - Wo tile: M_tile × K_tile × 2 bytes (INT16)
     *   - Accumulator tile: M_tile × N_tile × 4 bytes (INT32)
     */
    struct WoTileConfig
    {
        int M_tile;    ///< Output tile size (d_model dimension)
        int K_tile;    ///< Reduction tile size (input_dim dimension)
        int N_tile;    ///< Batch tile size (prefill only, =1 for decode)
        int d_model;   ///< Total output dimension
        int input_dim; ///< Total input dimension (n_heads × head_dim)

        /// Memory footprint of Wo tile [M_tile × K_tile × 2 bytes]
        size_t wo_tile_bytes() const
        {
            return static_cast<size_t>(M_tile) * K_tile * sizeof(int16_t);
        }

        /// Memory footprint of context tile [K_tile × 2 bytes] (per query)
        size_t context_tile_bytes() const
        {
            return static_cast<size_t>(K_tile) * sizeof(int16_t);
        }

        /// Memory footprint of accumulator tile [M_tile × N_tile × 4 bytes]
        size_t accumulator_tile_bytes() const
        {
            return static_cast<size_t>(M_tile) * N_tile * sizeof(int32_t);
        }

        /// Total L1 working set per query (Wo tile + context tile)
        size_t l1_working_set() const
        {
            return wo_tile_bytes() + context_tile_bytes();
        }
    };

    /**
     * @brief Compute optimal Wo projection tile sizes based on cache hierarchy.
     *
     * Algorithm:
     * 1. Start with M_tile from L2 constraint
     * 2. Compute K_tile to fit [M_tile × K_tile × 2 + K_tile × 2] in L1
     * 3. If K_tile is too small, reduce M_tile and retry
     * 4. N_tile: Batch chunk to amortize Wo loads (prefill)
     *
     * @param d_model Output dimension (typically 896, 2048, etc.)
     * @param input_dim Input dimension (n_heads × head_dim)
     * @param batch_size Batch size (1 for decode, >1 for prefill)
     * @return WoTileConfig with optimal tile sizes
     */
    inline WoTileConfig compute_wo_tile_config(int d_model, int input_dim, int batch_size = 1)
    {
        const auto &cache = cache_info();

        WoTileConfig cfg;
        cfg.d_model = d_model;
        cfg.input_dim = input_dim;

        const size_t l1_target = cache.l1_size / 2; // 50% of L1
        const size_t l2_target = cache.l2_size / 4; // 25% of L2

        // ===== Step 1: Initial M_tile from L2 constraint =====
        // Accumulators [M_tile × N_tile × 4 bytes] should fit in L2
        int M_tile = static_cast<int>(l2_target / (batch_size * sizeof(int32_t)));
        M_tile = (M_tile / 16) * 16; // Round DOWN to multiple of 16
        M_tile = std::clamp(M_tile, 16, 256);

        // ===== Step 2: Compute K_tile to fit in L1 =====
        // L1 working set: Wo tile [M_tile × K_tile × 2] + context tile [K_tile × 2]
        // K_tile × (M_tile + 1) × 2 ≤ l1_target
        // K_tile ≤ l1_target / ((M_tile + 1) × 2)
        auto compute_k_tile = [&](int m_tile) -> int
        {
            int k = static_cast<int>(l1_target / ((m_tile + 1) * sizeof(int16_t)));
            k = (k / 32) * 32; // Round DOWN to multiple of 32
            return std::clamp(k, 32, 512);
        };

        int K_tile = compute_k_tile(M_tile);

        // ===== Step 3: Reduce M_tile if K_tile is too small =====
        // We want K_tile >= 64 for good SIMD utilization
        while (K_tile < 64 && M_tile > 16)
        {
            M_tile -= 16;
            K_tile = compute_k_tile(M_tile);
        }

        // Ensure K_tile doesn't exceed input_dim
        if (K_tile > input_dim)
        {
            K_tile = (input_dim / 32) * 32;
            if (K_tile == 0)
                K_tile = input_dim;
        }

        cfg.M_tile = M_tile;
        cfg.K_tile = K_tile;

        // ===== Step 4: Determine N_tile (batch dimension) =====
        if (batch_size <= 1)
        {
            cfg.N_tile = 1;
        }
        else
        {
            int N_tile = static_cast<int>(l2_target / (M_tile * sizeof(int32_t)));
            cfg.N_tile = std::clamp(N_tile, 1, std::min(batch_size, 16));
        }

        return cfg;
    }

    /**
     * @brief Get default Wo projection tile config (compile-time constants).
     *
     * Conservative defaults that work across most CPUs:
     *   - M_tile=64: Good AVX-512 utilization
     *   - K_tile=128: Fits in L1 with Wo tile
     *   - N_tile=4: Moderate batch for prefill
     */
    inline constexpr WoTileConfig default_wo_tile_config(int d_model, int input_dim, int batch_size = 1)
    {
        return WoTileConfig{
            64,                       // M_tile
            128,                      // K_tile
            (batch_size > 1) ? 4 : 1, // N_tile
            d_model,
            input_dim};
    }

    /// Legacy constants for backwards compatibility
    /// @deprecated Use compute_wo_tile_config() instead
    constexpr int WO_M_TILE = 64;
    constexpr int WO_K_TILE = 128;

    // ============================================================================
    // Normalization: INT32 → INT16 for Wo multiply
    // ============================================================================

    /**
     * @brief Normalize INT32 context to INT16 range for Wo multiplication.
     *
     * The P×V accumulation produces large INT32 values. Before Wo projection,
     * we need to normalize these to INT16 range [-32768, 32767] to maintain
     * precision in the VPDPWSSD multiply.
     *
     * @param context_int32 Input INT32 context [n_heads × head_dim]
     * @param context_int16 Output INT16 normalized context [n_heads × head_dim]
     * @param context_scale Output scale factor (context_int32 = context_int16 × scale)
     * @param num_elements Total elements (n_heads × head_dim)
     */
    void q16_context_normalize_to_int16(
        const int32_t *context_int32,
        int16_t *context_int16,
        float &context_scale,
        int num_elements);

    // ============================================================================
    // Flash Decode: Single Row Wo Projection (GEMV)
    // ============================================================================

    /**
     * @brief Single-row Wo projection for decode.
     *
     * Computes: output[d] = context_int16 · Wo[d, :] for one output dimension.
     *
     * @tparam BlockType Q16_1 block type (Q16_1Block_64 or Q16_1Block_128)
     *
     * @param context_int16 Normalized INT16 context [n_heads × head_dim]
     * @param Wo Single row of Wo weights in Q16_1 format
     * @param output_int32 Output INT32 accumulator (single value)
     * @param input_dim Total input dimension (n_heads × head_dim)
     * @param blocks_per_input Number of Q16_1 blocks per input dim
     */
    template <typename BlockType>
    void q16_wo_row_gemv(
        const int16_t *context_int16,
        const BlockType *Wo,
        int32_t &output_int32,
        int input_dim,
        int blocks_per_input);

    // ============================================================================
    // Full Wo Projection (GEMV for decode)
    // ============================================================================

    /**
     * @brief Full Wo projection for decode: context × Wo^T → output.
     *
     * Projects the concatenated context from all heads through Wo to produce
     * the final projection. Output is written as Q16_1 blocks.
     *
     * @tparam BlockType Q16_1 block type
     *
     * @param context_int32 INT32 context from P×V [n_heads × head_dim]
     * @param Wo Wo weight matrix in Q16_1 [d_model × n_heads × head_dim]
     * @param output Q16_1 output blocks [d_model / block_size blocks]
     * @param d_model Output dimension
     * @param input_dim Input dimension (n_heads × head_dim)
     * @param blocks_per_input Blocks per input row
     * @param blocks_per_output Blocks per output
     */
    template <typename BlockType>
    void q16_wo_projection(
        const int32_t *context_int32,
        const BlockType *Wo,
        BlockType *output,
        int d_model,
        int input_dim,
        int blocks_per_input,
        int blocks_per_output);

    // ============================================================================
    // FA2 Prefill: Batched Wo Projection (GEMM)
    // ============================================================================

    /**
     * @brief Batched Wo projection for prefill.
     *
     * Projects multiple queries through Wo simultaneously to amortize
     * weight matrix loads. Each query's context is normalized independently.
     *
     * @tparam BlockType Q16_1 block type
     *
     * @param context_int32 INT32 contexts [batch × n_heads × head_dim]
     * @param Wo Wo weight matrix [d_model × n_heads × head_dim]
     * @param output Q16_1 output [batch × d_model / block_size blocks]
     * @param batch_size Number of queries in batch
     * @param d_model Output dimension
     * @param input_dim Input dimension per query
     * @param context_stride Stride between query contexts
     * @param output_stride Stride between query outputs (in blocks)
     * @param blocks_per_input Blocks per input dimension
     * @param blocks_per_output Blocks per output dimension
     */
    template <typename BlockType>
    void q16_wo_projection_batched(
        const int32_t *context_int32,
        const BlockType *Wo,
        BlockType *output,
        int batch_size,
        int d_model,
        int input_dim,
        int context_stride,
        int output_stride,
        int blocks_per_input,
        int blocks_per_output);

    // ============================================================================
    // Output Quantization: INT32 → Q16_1
    // ============================================================================

    /**
     * @brief Quantize INT32 accumulators to Q16_1 blocks.
     *
     * Takes a contiguous buffer of INT32 values and packs them into Q16_1 blocks.
     * Computes the optimal scale per block based on max absolute value.
     *
     * @tparam BlockType Q16_1 block type
     *
     * @param accumulators INT32 input values
     * @param output Q16_1 output blocks
     * @param num_values Total number of values
     * @param input_scale Scale factor from context normalization
     * @param blocks_per_output Number of output blocks
     */
    template <typename BlockType>
    void q16_quantize_to_q16_1(
        const int32_t *accumulators,
        BlockType *output,
        int num_values,
        float input_scale,
        int blocks_per_output);

    // ============================================================================
    // Dispatch Functions (Runtime Block Size Selection)
    // ============================================================================

    /**
     * @brief Runtime dispatch for Wo projection based on block size.
     */
    void q16_wo_projection_dispatch(
        const int32_t *context_int32,
        const void *Wo,
        void *output,
        int d_model,
        int input_dim,
        Q16BlockSize block_size);

    /**
     * @brief Runtime dispatch for batched Wo projection.
     */
    void q16_wo_projection_batched_dispatch(
        const int32_t *context_int32,
        const void *Wo,
        void *output,
        int batch_size,
        int d_model,
        int input_dim,
        int context_stride,
        int output_stride,
        Q16BlockSize block_size);

} // namespace llaminar2::kernels::q16_1::microkernels
