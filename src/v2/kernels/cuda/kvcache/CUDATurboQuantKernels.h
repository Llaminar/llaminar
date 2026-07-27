/**
 * @file CUDATurboQuantKernels.h
 * @brief CUDA kernel declarations for TurboQuant KV cache operations
 * @author David Sanftenberg
 *
 * Provides CUDA kernels for:
 * - TQ8/TQ4 quantization (FP32 → TQ block, for KV cache append)
 * - TQ8/TQ4 dequantization (TQ block → FP32, for KV cache read)
 * - Fused dequant + RoPE (dequant to FP32 with rotary position encoding)
 * - Ring buffer linearize with dequant + RoPE fusion
 *
 * Codebooks (TQ4_CENTROIDS, TQ8_CENTROIDS) are stored in __constant__ memory
 * for fast cache-resident access.
 *
 * Rotation matrices are stored in global memory (one per layer × head),
 * uploaded once at model load time.
 */

#pragma once

#include "../../../tensors/BlockStructures.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace llaminar2
{

    // =========================================================================
    // Constant Memory Codebook Upload
    // =========================================================================

    /**
     * @brief Upload TQ4 and TQ8 codebooks to CUDA constant memory.
     * Must be called once before any TQ kernel launch.
     * Thread-safe (uses internal flag to skip redundant uploads).
     */
    void cuda_tq_upload_codebooks(cudaStream_t stream);

    // =========================================================================
    // Rotation Matrix Management
    // =========================================================================

    /**
     * @brief GPU-resident rotation matrices for TurboQuant.
     *
     * Stores Π (rotation) and Πᵀ (transpose) for each (layer, head) pair.
     * Allocated once at model load time.
     */
    struct CUDATurboQuantRotations
    {
        float *d_rotations = nullptr;   ///< [n_layers * n_kv_heads * D * D] rotation matrices
        float *d_rotations_t = nullptr; ///< [n_layers * n_kv_heads * D * D] transposed rotations
        int n_layers = 0;
        int n_kv_heads = 0;
        int head_dim = 0;
        int device_id = -1; ///< Backend-local CUDA ordinal owning both allocations.

        /// Get rotation matrix Π for (layer, head)
        const float *rotation(int layer, int head) const
        {
            return d_rotations + static_cast<size_t>((layer * n_kv_heads + head) * head_dim * head_dim);
        }

        /// Get transposed rotation Πᵀ for (layer, head)
        const float *rotation_t(int layer, int head) const
        {
            return d_rotations_t + static_cast<size_t>((layer * n_kv_heads + head) * head_dim * head_dim);
        }

        /// Total bytes for all rotation matrices (both forward and transpose)
        size_t total_bytes() const
        {
            return 2ULL * n_layers * n_kv_heads * head_dim * head_dim * sizeof(float);
        }
    };

    /**
     * @brief Allocate and upload rotation matrices to GPU.
     *
     * For each (layer, head) pair, derives the TurboQuantContext::for_layer()
     * rotation matrix and uploads both Π and Πᵀ.
     *
     * @param n_layers     Number of transformer layers
     * @param n_kv_heads   Number of KV heads per layer
     * @param head_dim     Head dimension (64 or 128)
     * @param rotation_seed Seed for rotation matrix generation (from TurboQuantContext)
     * @param device_id    CUDA device
     * @param stream       CUDA stream for async upload
     * @return CUDATurboQuantRotations with allocated device memory
     */
    CUDATurboQuantRotations cuda_tq_create_rotations(
        int n_layers, int n_kv_heads, int head_dim,
        uint64_t rotation_seed, int device_id,
        cudaStream_t stream,
        int kv_head_start = 0);

    /**
     * @brief Free GPU rotation matrices.
     */
    void cuda_tq_free_rotations(CUDATurboQuantRotations &rotations);

    // =========================================================================
    // TQ8 Quantize: FP32 → TQ8Block (for K cache append)
    // =========================================================================

    /**
     * @brief Quantize FP32 K projections to TQ8 blocks on GPU.
     *
     * Each block handles one head of one token:
     *   1. Compute norm
     *   2. Normalize and scale by √D
     *   3. Apply rotation Π
     *   4. Find nearest TQ8 centroid (binary search in 256-level codebook)
     *   5. Store norm + uint8 indices
     *
     * @param d_input      FP32 input: [num_tokens, n_kv_heads * head_dim]
     * @param d_rotations  Rotation matrices for this layer: [n_kv_heads * head_dim * head_dim]
     * @param d_output     Output TQ8 blocks: [num_tokens * n_kv_heads] TQ8Blocks
     * @param num_tokens   Number of tokens to quantize
     * @param n_kv_heads   Number of KV heads
     * @param head_dim     Head dimension (64 or 128)
     * @param stream       CUDA stream
     */
    extern "C" bool cuda_tq8_quantize(
        const float *d_input,
        const float *d_rotations,
        void *d_output,
        int num_tokens, int n_kv_heads, int head_dim,
        cudaStream_t stream);

    // =========================================================================
    // TQ4 Quantize: FP32 → TQ4Block (for V cache append)
    // =========================================================================

    /**
     * @brief Quantize FP32 V projections to TQ4 blocks on GPU.
     *
     * Similar to TQ8 but uses 4-bit codebook (16 centroids) with
     * 3-bit MSE + 1 high-bit packing.
     *
     * @param d_input      FP32 input: [num_tokens, n_kv_heads * head_dim]
     * @param d_rotations  Rotation matrices for this layer: [n_kv_heads * head_dim * head_dim]
     * @param d_output     Output TQ4 blocks: [num_tokens * n_kv_heads] TQ4Blocks
     * @param num_tokens   Number of tokens to quantize
     * @param n_kv_heads   Number of KV heads
     * @param head_dim     Head dimension (64 or 128)
     * @param stream       CUDA stream
     */
    extern "C" bool cuda_tq4_quantize(
        const float *d_input,
        const float *d_rotations,
        void *d_output,
        int num_tokens, int n_kv_heads, int head_dim,
        cudaStream_t stream);

    /**
     * @brief Quantize all MTP verifier rows directly into their TQ ring slots.
     *
     * One grid covers K/TQ8 and V/TQ4 for every `(verifier row, KV head)`
     * pair.  Each block uses the exact same reduction and rotation order as
     * the serial fused decode kernel, preserving byte identity while avoiding
     * temporary quantized buffers and per-row device copies.
     *
     * @param d_K_input Device-resident FP32 K rows.
     * @param d_V_input Device-resident FP32 V rows.
     * @param d_rotations Device-resident local-head rotation matrices.
     * @param d_K_ring Destination TQ8 K ring.
     * @param d_V_ring Destination TQ4 V ring.
     * @param ring_head Static first destination ring row.
     * @param max_seq_len Ring capacity used for wraparound.
     * @param verifier_rows Number of grouped rows, in `[1,4]`.
     * @param n_kv_heads Number of local KV heads.
     * @param head_dim Per-head width, either 64 or 128.
     * @param k_head_major Whether K uses `[head][row][dim]` layout.
     * @param v_head_major Whether V uses `[head][row][dim]` layout.
     * @param stream Mandatory explicit CUDA stream.
     */
    extern "C" bool cuda_tq_quantize_grouped_ring(
        const float *d_K_input, const float *d_V_input,
        const float *d_rotations,
        void *d_K_ring, void *d_V_ring,
        int ring_head, int max_seq_len,
        int verifier_rows, int n_kv_heads, int head_dim,
        bool k_head_major, bool v_head_major,
        cudaStream_t stream);

    /**
     * @brief Graph-capturable grouped TQ verifier publication.
     *
     * This is the device-owned counterpart of
     * `cuda_tq_quantize_grouped_ring()`: the first destination row is read
     * from a persistent device scalar on every graph replay. When
     * @p d_row_count is non-null, rows at or beyond that resident count no-op;
     * this preserves fixed bucket launch geometry for unequal request batches.
     */
    extern "C" bool cuda_tq_quantize_grouped_ring_dynamic(
        const float *d_K_input, const float *d_V_input,
        const float *d_rotations,
        void *d_K_ring, void *d_V_ring,
        const int *d_ring_head, const int *d_row_count, int max_seq_len,
        int verifier_rows, int n_kv_heads, int head_dim,
        bool k_head_major, bool v_head_major,
        cudaStream_t stream);

    /**
     * @brief Publish prepared device TQ8-K/TQ4-V rows in one D2D kernel.
     */
    extern "C" bool cuda_tq_copy_prepared_rows_ring(
        const void *source_k, const void *source_v,
        void *ring_k, void *ring_v,
        int ring_head, int max_seq_len, int rows,
        size_t k_row_bytes, size_t v_row_bytes,
        bool k_head_major, bool v_head_major,
        int n_kv_heads,
        cudaStream_t stream);

    /**
     * @brief Device-head graph-capture variant of prepared TQ row publication.
     *
     * @p d_row_count optionally limits a fixed bucket launch to the request's
     * real resident row count.
     */
    extern "C" bool cuda_tq_copy_prepared_rows_ring_dynamic(
        const void *source_k, const void *source_v,
        void *ring_k, void *ring_v,
        const int *d_ring_head, const int *d_row_count,
        int max_seq_len, int rows,
        size_t k_row_bytes, size_t v_row_bytes,
        bool k_head_major, bool v_head_major,
        int n_kv_heads,
        cudaStream_t stream);

    // =========================================================================
    // TQ8 Dequantize: TQ8Block → FP32 (for K cache read)
    // =========================================================================

    /**
     * @brief Dequantize TQ8 blocks to FP32 with optional RoPE.
     *
     * For each block:
     *   1. Centroid lookup (uint8 → float via codebook)
     *   2. Divide by √D
     *   3. Inverse rotation Πᵀ
     *   4. Scale by norm
     *   5. (Optional) Apply RoPE: cos/sin rotation at position `pos`
     *
     * @param d_tq8_blocks Input TQ8 blocks: [count * n_kv_heads] blocks
     * @param d_rotations_t Transposed rotation matrices: [n_kv_heads * head_dim * head_dim]
     * @param d_output     FP32 output: [count, n_kv_heads * head_dim]
     * @param count        Number of positions to dequantize
     * @param n_kv_heads   Number of KV heads
     * @param head_dim     Head dimension
     * @param rope_theta   RoPE base frequency (0 = no RoPE)
     * @param position_start Starting position for RoPE (ring buffer position offset)
     * @param max_seq_len  Ring buffer capacity (for position wrapping)
     * @param tail         Ring buffer tail (oldest token position)
     * @param stream       CUDA stream
     */
    extern "C" bool cuda_tq8_dequantize_fp32(
        const void *d_tq8_blocks,
        const float *d_rotations_t,
        float *d_output,
        int count, int n_kv_heads, int head_dim,
        float rope_theta, int position_start,
        int max_seq_len, int tail,
        cudaStream_t stream);

    // =========================================================================
    // TQ4 Dequantize: TQ4Block → FP32 (for V cache read)
    // =========================================================================

    /**
     * @brief Dequantize TQ4 blocks to FP32.
     *
     * Same path as TQ8 dequant but with 4-bit index unpacking.
     * No RoPE fusion (V doesn't need RoPE).
     *
     * @param d_tq4_blocks Input TQ4 blocks: [count * n_kv_heads] blocks
     * @param d_rotations_t Transposed rotation matrices: [n_kv_heads * head_dim * head_dim]
     * @param d_output     FP32 output: [count, n_kv_heads * head_dim]
     * @param count        Number of positions to dequantize
     * @param n_kv_heads   Number of KV heads
     * @param head_dim     Head dimension
     * @param stream       CUDA stream
     */
    extern "C" bool cuda_tq4_dequantize_fp32(
        const void *d_tq4_blocks,
        const float *d_rotations_t,
        float *d_output,
        int count, int n_kv_heads, int head_dim,
        cudaStream_t stream);

    // =========================================================================
    // Ring Buffer Linearize + Dequant + RoPE (for attention read)
    // =========================================================================

    /**
     * @brief Linearize TQ ring buffer to contiguous FP32 with fused dequant + RoPE.
     *
     * Reads wrapped ring buffer, dequantizes TQ8 K / TQ4 V blocks to FP32,
     * applies RoPE to K if rope_theta > 0.
     *
     * @param d_K_out      FP32 K output: [count, n_kv_heads * head_dim]
     * @param d_V_out      FP32 V output: [count, n_kv_heads * head_dim]
     * @param d_K_cache    TQ8 K ring buffer: [max_seq_len * n_kv_heads] TQ8Blocks
     * @param d_V_cache    TQ4 V ring buffer: [max_seq_len * n_kv_heads] TQ4Blocks
     * @param d_K_rotations_t Transposed K rotation matrices
     * @param d_V_rotations_t Transposed V rotation matrices
     * @param d_K_rotations   Non-transposed K rotation matrices (row-major R, for coalesced GPU access)
     * @param d_V_rotations   Non-transposed V rotation matrices (row-major R, for coalesced GPU access)
     * @param tail         Ring buffer tail position
     * @param count        Number of valid tokens
     * @param max_seq_len  Ring buffer capacity
     * @param n_kv_heads   Number of KV heads
     * @param head_dim     Head dimension
     * @param rope_theta   RoPE base frequency (0 = no RoPE)
     * @param stream       CUDA stream
     */
    extern "C" bool cuda_tq_ring_linearize_dequant(
        float *d_K_out, float *d_V_out,
        const void *d_K_cache, const void *d_V_cache,
        const float *d_K_rotations_t, const float *d_V_rotations_t,
        const float *d_K_rotations, const float *d_V_rotations,
        int tail, int count, int max_seq_len,
        int n_kv_heads, int head_dim,
        float rope_theta, int position_start,
        cudaStream_t stream);

    /**
     * @brief FP16 output variant of ring buffer linearize + dequant + RoPE.
     *
     * Same as cuda_tq_ring_linearize_dequant but outputs __half instead of float.
     * Halves scratch memory and enables FP16 flash attention path (2× less bandwidth).
     * Internal computation is still FP32; only the final write converts to FP16.
     */
    extern "C" bool cuda_tq_ring_linearize_dequant_fp16(
        __half *d_K_out, __half *d_V_out,
        const void *d_K_cache, const void *d_V_cache,
        const float *d_K_rotations_t, const float *d_V_rotations_t,
        const float *d_K_rotations, const float *d_V_rotations,
        int tail, int count, int max_seq_len,
        int n_kv_heads, int head_dim,
        float rope_theta, int position_start, int rope_dim,
        cudaStream_t stream);

    /**
     * @brief Group-dequantize request-local TQ rings using resident ring state.
     *
     * Separate grouped TQ8 and TQ4 grids preserve the established serial
     * dequant arithmetic while amortizing launch work across all requests.
     * Padded rows beyond each device count are explicitly zeroed.
     */
    extern "C" bool cuda_tq_batched_ring_dequant_fp16_device_state(
        __half *d_K_out,
        __half *d_V_out,
        const void *const *d_K_entry_table,
        const void *const *d_V_entry_table,
        const int *d_heads,
        const int *d_counts,
        const float *d_rotations,
        int entry_offset,
        int request_count,
        int max_kv_len,
        int max_seq_len,
        int n_kv_heads,
        int head_dim,
        float rope_theta,
        int position_start,
        int rope_dim,
        cudaStream_t stream);

    // =========================================================================
    // Batched Incremental Dequant (all layers in 2 kernel launches)
    // =========================================================================

    /**
     * @brief Per-layer parameters for batched incremental dequant.
     *
     * Each element describes one layer's incremental dequant work:
     * a single new position that needs dequantization.
     */
    struct IncrementalDequantParam
    {
        const uint8_t *cache;  ///< TQ cache (K or V) for this layer
        __half *output;        ///< FP16 scratch output for this layer
        const float *rotation; ///< Rotation Π for this layer [n_kv_heads * D * D]
        int ring_pos;          ///< Ring buffer position of the new entry
        int out_offset;        ///< Output offset: (count-1) * kv_dim
        int rope_position;     ///< Position for RoPE (0 if no RoPE)
    };

    /**
     * @brief Batched incremental dequant: process 1 new position for ALL layers.
     *
     * During decode, each layer has exactly 1 new position. Instead of
     * 2×n_layers separate kernel launches, this function launches 2 kernels
     * (TQ8 for K + TQ4 for V) with grid.y = n_layers.
     *
     * @param d_K_params   Device array of n_layers IncrementalDequantParam for K (TQ8)
     * @param d_V_params   Device array of n_layers IncrementalDequantParam for V (TQ4)
     * @param n_layers     Number of layers
     * @param n_kv_heads   Number of KV heads
     * @param head_dim     Head dimension (64 or 128)
     * @param rope_theta   RoPE base (0 = no RoPE, applied to K only)
     * @param stream       CUDA stream
     */
    extern "C" bool cuda_tq_batch_incremental_dequant_fp16(
        const IncrementalDequantParam *d_K_params,
        const IncrementalDequantParam *d_V_params,
        int n_layers, int n_kv_heads, int head_dim,
        float rope_theta,
        cudaStream_t stream);

    /**
     * @brief Fused single-position incremental dequant: K (TQ8) + V (TQ4)
     *        in one kernel launch per layer. Halves launch overhead vs separate calls.
     *
     * @param d_K_out          FP16 K output (scratch buffer)
     * @param d_V_out          FP16 V output (scratch buffer)
     * @param d_K_cache        TQ8 K cache for this layer/seq
     * @param d_V_cache        TQ4 V cache for this layer/seq
     * @param d_K_rotation     K rotation matrix Π (D×D per head, row-major)
     * @param d_V_rotation     V rotation matrix Π (D×D per head, row-major)
     * @param ring_pos         Ring buffer position of the new token
     * @param out_offset_elems Element offset into scratch (position × kv_dim)
     * @param n_kv_heads       Number of KV heads
     * @param head_dim         Head dimension (64 or 128)
     * @param rope_theta       RoPE base (0 = no RoPE, applied to K only)
     * @param rope_position    Absolute position for RoPE
     * @param stream           CUDA stream
     */
    extern "C" bool cuda_tq_incremental_single_fp16(
        __half *d_K_out, __half *d_V_out,
        const void *d_K_cache, const void *d_V_cache,
        const float *d_K_rotation, const float *d_V_rotation,
        int ring_pos, int out_offset_elems,
        int n_kv_heads, int head_dim,
        float rope_theta, int rope_position,
        cudaStream_t stream);

    /**
     * @brief Graph-capturable incremental dequant driven by device ring state.
     *
     * Outputs are base pointers. The fused kernel reads the cache's canonical
     * post-append head/count and derives its compressed source row, scratch
     * destination row, and optional RoPE position in-register. No host-owned
     * dequant parameter mailbox is uploaded between graph replays.
     */
    extern "C" bool cuda_tq_incremental_single_fp16_dynamic(
        __half *d_K_base, __half *d_V_base,
        const void *d_K_cache, const void *d_V_cache,
        const float *d_K_rotation, const float *d_V_rotation,
        const int *d_ring_head, const int *d_cached_count,
        int max_seq_len, int kv_dim, int position_start,
        int n_kv_heads, int head_dim,
        float rope_theta,
        cudaStream_t stream);

    // =========================================================================
    // Generic RoPE kernel (for non-TQ caches)
    // =========================================================================

    /**
     * @brief Apply RoPE to pre-linearized FP16 K tensor on GPU.
     *
     * Used by non-TQ caches (FP16, Q8_1) that use get_kv_converted() with RoPE.
     * Applies cos/sin rotation in-place.
     *
     * @param d_K          FP16 K tensor: [count, n_kv_heads * head_dim]
     * @param count        Number of positions
     * @param n_kv_heads   Number of KV heads
     * @param head_dim     Head dimension (must be even)
     * @param rope_theta   RoPE base frequency
     * @param position_start Starting position for RoPE
     * @param stream       CUDA stream
     * @param rope_dim     Number of dimensions to rotate per head (0 = full head_dim)
     */
    extern "C" bool cuda_rope_apply_fp16(
        __half *d_K, int count,
        int n_kv_heads, int head_dim,
        float rope_theta, int position_start,
        cudaStream_t stream, int rope_dim = 0);

    /**
     * @brief Apply RoPE to FP32 K tensor on GPU (for FP32 caches).
     */
    extern "C" bool cuda_rope_apply_fp32(
        float *d_K, int count,
        int n_kv_heads, int head_dim,
        float rope_theta, int position_start,
        cudaStream_t stream, int rope_dim = 0);

    /**
     * @brief Apply canonical serial RoPE arithmetic to a fixed-stride request batch.
     *
     * The request-major buffer has shape `[request_count, max_kv_len, kv_dim]`.
     * Canonical device counts mask padded rows, so they retain exact positive-zero
     * bits and never enter trigonometric arithmetic.
     */
    extern "C" bool cuda_rope_apply_batched_fp16_device_state(
        __half *d_K,
        const int *d_counts,
        int entry_offset,
        int request_count,
        int max_kv_len,
        int max_seq_len,
        int n_kv_heads,
        int head_dim,
        float rope_theta,
        int position_start,
        int rope_dim,
        cudaStream_t stream);

    /**
     * @brief Rotate FP32 ring rows and publish FP16 request-major output.
     *
     * This fused read keeps the same FP32 RoPE result and FP16 rounding as the
     * serial `cuda_rope_apply_fp32()` plus conversion sequence while avoiding an
     * intermediate request-batched FP32 allocation.
     */
    extern "C" bool cuda_rope_apply_batched_fp32_ring_to_fp16_device_state(
        __half *d_K_out,
        const float *const *d_K_entry_table,
        const int *d_heads,
        const int *d_counts,
        int entry_offset,
        int request_count,
        int max_kv_len,
        int max_seq_len,
        int n_kv_heads,
        int head_dim,
        float rope_theta,
        int position_start,
        int rope_dim,
        cudaStream_t stream);

} // namespace llaminar2
