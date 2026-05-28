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
    // Dynamic Params for Graph-Capturable Incremental Dequant
    // =========================================================================

    /**
     * @brief Device-side dynamic parameters for TQ incremental dequant.
     *
     * During CUDA graph capture, kernel arguments are baked into the graph.
     * This struct lives in device memory; the kernel reads from it at
     * runtime. Between graph replays, the host writes new values to a
     * pinned shadow buffer and a captured H2D memcpy re-copies them.
     */
    struct TQDequantDynamicParams
    {
        int ring_pos;         ///< Ring buffer position of the new token
        int out_offset_elems; ///< Element offset into scratch (position × kv_dim)
        int rope_position;    ///< Absolute position for RoPE (0 if no RoPE)
    };

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
        cudaStream_t stream);

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
     * @brief Fused quantize K(TQ8) + V(TQ4) directly into ring buffer.
     *        Single kernel launch for decode (1 token). Eliminates temp buffer
     *        and D2D memcpy overhead.
     *
     * @param d_K_input    FP32 K projections: [n_kv_heads * head_dim]
     * @param d_V_input    FP32 V projections: [n_kv_heads * head_dim]
     * @param d_rotations  Rotation matrix Π: [n_kv_heads * head_dim * head_dim]
     * @param d_K_ring     K ring buffer (TQ8 blocks)
     * @param d_V_ring     V ring buffer (TQ4 blocks)
     * @param ring_pos     Ring buffer write position
     * @param n_kv_heads   Number of KV heads
     * @param head_dim     Head dimension (64 or 128)
     * @param stream       CUDA stream
     */
    extern "C" bool cuda_tq_quantize_fused_ring(
        const float *d_K_input, const float *d_V_input,
        const float *d_rotations,
        void *d_K_ring, void *d_V_ring,
        int ring_pos, int n_kv_heads, int head_dim,
        cudaStream_t stream);

    /**
     * @brief Graph-capturable variant: reads ring_pos from device memory.
     *
     * Same as cuda_tq_quantize_fused_ring but ring_pos is read from a
     * device pointer, enabling CUDA graph capture. Between replays,
     * the host updates the pinned shadow and the captured H2D re-copies.
     */
    extern "C" bool cuda_tq_quantize_fused_ring_dynamic(
        const float *d_K_input, const float *d_V_input,
        const float *d_rotations,
        void *d_K_ring, void *d_V_ring,
        const int *d_ring_pos, int n_kv_heads, int head_dim,
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
    // Ring Buffer TQ Append (fused quantize + ring write)
    // =========================================================================

    /**
     * @brief Append FP32 K/V to TQ ring buffer (fused quantize + ring write).
     *
     * K is quantized to TQ8, V to TQ4, both written at ring head position.
     *
     * @param d_K_cache    TQ8 K ring buffer: [max_seq_len * n_kv_heads] TQ8Blocks
     * @param d_V_cache    TQ4 V ring buffer: [max_seq_len * n_kv_heads] TQ4Blocks
     * @param d_K_new      FP32 K input: [num_tokens, n_kv_heads * head_dim]
     * @param d_V_new      FP32 V input: [num_tokens, n_kv_heads * head_dim]
     * @param d_K_rotations Rotation matrices for K: [n_kv_heads * head_dim * head_dim]
     * @param d_V_rotations Rotation matrices for V: [n_kv_heads * head_dim * head_dim]
     * @param head         Current ring buffer head position
     * @param max_seq_len  Ring buffer capacity
     * @param n_kv_heads   Number of KV heads
     * @param head_dim     Head dimension
     * @param num_tokens   Number of tokens to append
     * @param stream       CUDA stream
     */
    extern "C" bool cuda_tq_ring_append(
        void *d_K_cache, void *d_V_cache,
        const float *d_K_new, const float *d_V_new,
        const float *d_K_rotations, const float *d_V_rotations,
        int head, int max_seq_len,
        int n_kv_heads, int head_dim, int num_tokens,
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
        float rope_theta, int position_start,
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
     * @brief Graph-capturable variant: reads dynamic params from device memory.
     *
     * Outputs are BASE pointers (no pre-applied offset). The kernel reads
     * ring_pos, out_offset_elems, and rope_position from d_params, enabling
     * CUDA graph capture. Between replays, the host updates the pinned
     * shadow and the captured H2D re-copies.
     */
    extern "C" bool cuda_tq_incremental_single_fp16_dynamic(
        __half *d_K_base, __half *d_V_base,
        const void *d_K_cache, const void *d_V_cache,
        const float *d_K_rotation, const float *d_V_rotation,
        const TQDequantDynamicParams *d_params,
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

} // namespace llaminar2
