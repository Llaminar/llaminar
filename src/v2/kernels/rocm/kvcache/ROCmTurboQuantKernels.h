/**
 * @file ROCmTurboQuantKernels.h
 * @brief ROCm/HIP kernel declarations for TurboQuant KV cache operations
 * @author David Sanftenberg
 *
 * HIP mirror of CUDATurboQuantKernels.h for AMD GPUs.
 * Same functionality: TQ8/TQ4 quantize, dequant, fused RoPE.
 * Codebooks in __constant__ memory, rotation matrices in global memory.
 */

#pragma once

#include "../../../tensors/BlockStructures.h"
#include "../../kvcache/TurboQuantKVMode.h"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>

namespace llaminar2
{

    // =========================================================================
    // Constant Memory Codebook Upload
    // =========================================================================

    /**
     * @brief Enqueue model-lifetime TQ codebooks on one HIP initialization stream.
     *
     * The owning cache performs the sole lifecycle fence after all persistent
     * allocations and uploads. Production kernel launchers consume the already
     * initialized constants and may not perform upload or synchronization.
     *
     * @param stream Non-null cache-initialization stream on the active device.
     * @return true when every constant-memory copy was accepted by HIP.
     */
    [[nodiscard]] bool hip_tq_upload_codebooks(hipStream_t stream);

    // =========================================================================
    // Rotation Matrix Management
    // =========================================================================

    struct ROCmTurboQuantRotations
    {
        float *d_rotations = nullptr;
        float *d_rotations_t = nullptr;
        int n_layers = 0;
        int n_kv_heads = 0;
        int head_dim = 0;
        int device_id = -1; ///< Backend-local ROCm ordinal owning both allocations.

        const float *rotation(int layer, int head) const
        {
            return d_rotations + static_cast<size_t>((layer * n_kv_heads + head) * head_dim * head_dim);
        }

        const float *rotation_t(int layer, int head) const
        {
            return d_rotations_t + static_cast<size_t>((layer * n_kv_heads + head) * head_dim * head_dim);
        }

        size_t total_bytes() const
        {
            return 2ULL * n_layers * n_kv_heads * head_dim * head_dim * sizeof(float);
        }
    };

    ROCmTurboQuantRotations hip_tq_create_rotations(
        int n_layers, int n_kv_heads, int head_dim,
        uint64_t rotation_seed, int device_id,
        hipStream_t stream,
        int kv_head_start = 0);

    void hip_tq_free_rotations(ROCmTurboQuantRotations &rotations);

    // =========================================================================
    // TQ8/TQ4 Quantize
    // =========================================================================

    extern "C" bool hip_tq8_quantize(
        const float *d_input, const float *d_rotations, void *d_output,
        int num_tokens, int n_kv_heads, int head_dim, hipStream_t stream);

    extern "C" bool hip_tq4_quantize(
        const float *d_input, const float *d_rotations, void *d_output,
        int num_tokens, int n_kv_heads, int head_dim, hipStream_t stream);

    /**
     * @brief Quantize grouped FP32 verifier rows directly into TQ8/TQ4 ring storage.
     *
     * A single HIP grid covers every verifier row, local KV head, and K/V
     * phase.  Within each block the arithmetic order matches serial decode.
     */
    extern "C" bool hip_tq_quantize_grouped_ring(
        const float *d_k_input, const float *d_v_input,
        const float *d_rotations,
        void *d_k_ring, void *d_v_ring,
        int ring_head, int max_seq_len,
        int verifier_rows, int n_kv_heads, int head_dim,
        bool k_head_major, bool v_head_major,
        TurboQuantKVMode mode,
        hipStream_t stream);

    /**
     * @brief Device-head variant for fully graph-captured grouped publication.
     *
     * @p d_row_count optionally limits fixed bucket geometry to the request's
     * real resident row count.
     * An empty cache copies its first input key as its immutable basis, even
     * if an oversized append evicts it. Non-empty/restored caches preserve the
     * existing basis; prefill and verifier callers cannot select another rule.
     */
    extern "C" bool hip_tq_quantize_grouped_ring_dynamic(
        const float *d_k_input, const float *d_v_input,
        const float *d_rotations,
        void *d_k_ring, void *d_v_ring, float *d_k_anchor,
        const int *d_ring_head, const int *d_cached_count,
        const int *d_row_count, int max_seq_len,
        int verifier_rows, int n_kv_heads, int head_dim,
        bool k_head_major, bool v_head_major,
        TurboQuantKVMode mode,
        hipStream_t stream);

    // =========================================================================
    // TQ8/TQ4 Dequantize to FP32
    // =========================================================================

    extern "C" bool hip_tq8_dequantize_fp32(
        const void *d_tq8_blocks, const float *d_rotations_t,
        float *d_output,
        int count, int n_kv_heads, int head_dim,
        float rope_theta, int position_start,
        int max_seq_len, int tail, hipStream_t stream);

    extern "C" bool hip_tq4_dequantize_fp32(
        const void *d_tq4_blocks, const float *d_rotations_t,
        float *d_output,
        int count, int n_kv_heads, int head_dim, hipStream_t stream);

    // =========================================================================
    // Ring Buffer TQ Operations
    // =========================================================================

    extern "C" bool hip_tq_ring_linearize_dequant(
        float *d_K_out, float *d_V_out,
        const void *d_K_cache, const void *d_V_cache,
        const float *d_K_rotations_t, const float *d_V_rotations_t,
        const float *d_K_rotations, const float *d_V_rotations,
        int tail, int count, int max_seq_len,
        int n_kv_heads, int head_dim,
        float rope_theta, int position_start, hipStream_t stream);

    // =========================================================================
    // Generic RoPE kernels
    // =========================================================================

    extern "C" bool hip_rope_apply_fp16(
        _Float16 *d_K, int count,
        int n_kv_heads, int head_dim,
        float rope_theta, int position_start,
        hipStream_t stream, int rope_dim = 0);

    extern "C" bool hip_rope_apply_fp32(
        float *d_K, int count,
        int n_kv_heads, int head_dim,
        float rope_theta, int position_start,
        hipStream_t stream, int rope_dim = 0);

    extern "C" bool hip_rope_apply_batched_fp16_device_state(
        _Float16 *d_K,
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
        hipStream_t stream);

    extern "C" bool hip_rope_apply_batched_fp32_ring_to_fp16_device_state(
        _Float16 *d_K_out,
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
        hipStream_t stream);

    // =========================================================================
    // RoPE Frequency Precomputation
    // =========================================================================

    /**
     * @brief Upload precomputed RoPE frequencies to constant memory.
     *
     * freq[i] = 1.0 / (theta ^ (2i / head_dim)) for i in [0, head_dim/2).
     * Eliminates per-thread powf() in dequant kernels.
     * Thread-safe; skips if already uploaded for the same head_dim.
     */
    void hip_tq_upload_rope_freqs(float rope_theta, int head_dim, hipStream_t stream);

    // =========================================================================
    // FP16 Linearize + Dequant (full sequence, for prefill)
    // =========================================================================

    extern "C" bool hip_tq_ring_linearize_dequant_fp16(
        _Float16 *d_K_out, _Float16 *d_V_out,
        const void *d_K_cache, const void *d_V_cache,
        const float *d_K_anchor,
        const float *d_K_rotations_t, const float *d_V_rotations_t,
        const float *d_K_rotations, const float *d_V_rotations,
        int tail, int count, int max_seq_len,
        int n_kv_heads, int head_dim,
        float rope_theta, int position_start, int rope_dim,
        TurboQuantKVMode mode,
        hipStream_t stream);

    /**
     * @brief Group-dequantize request-local TQ rings from resident ring state.
     *
     * The grouped TQ8 and TQ4 grids retain the serial ROCm FP16 arithmetic and
     * write zero padding after each request's live device count.
     */
    extern "C" bool hip_tq_batched_ring_dequant_fp16_device_state(
        _Float16 *d_K_out,
        _Float16 *d_V_out,
        const void *const *d_K_entry_table,
        const void *const *d_V_entry_table,
        const void *const *d_K_anchor_table,
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
        TurboQuantKVMode mode,
        hipStream_t stream);

    // =========================================================================
    // FP16 Fused Incremental Dequant (single position, for decode)
    // =========================================================================

    /**
     * @brief Fused single-position incremental dequant: K (TQ8) + V (TQ4)
     *        in one kernel launch per layer. Outputs FP16.
     */
    extern "C" bool hip_tq_incremental_single_fp16(
        _Float16 *d_K_out, _Float16 *d_V_out,
        const void *d_K_cache, const void *d_V_cache,
        const float *d_K_rotation, const float *d_V_rotation,
        int ring_pos, int out_offset_elems,
        int n_kv_heads, int head_dim,
        float rope_theta, int rope_position,
        hipStream_t stream);

    /**
     * @brief Graph-capturable incremental dequant driven by device ring state.
     *
     * The fused kernel reads the canonical post-append head/count and derives
     * its source row, scratch destination, and optional RoPE position without
     * a pinned-host parameter upload between HIP graph replays.
     */
    extern "C" bool hip_tq_incremental_single_fp16_dynamic(
        _Float16 *d_K_base, _Float16 *d_V_base,
        const void *d_K_cache, const void *d_V_cache,
        const float *d_K_rotation, const float *d_V_rotation,
        const int *d_ring_head, const int *d_cached_count,
        int max_seq_len, int kv_dim, int position_start,
        int n_kv_heads, int head_dim,
        float rope_theta,
        hipStream_t stream);

} // namespace llaminar2
