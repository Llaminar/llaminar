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
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>

namespace llaminar2
{

    // =========================================================================
    // Constant Memory Codebook Upload
    // =========================================================================

    void hip_tq_upload_codebooks(hipStream_t stream);

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
        hipStream_t stream);

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

    extern "C" bool hip_tq_ring_append(
        void *d_K_cache, void *d_V_cache,
        const float *d_K_new, const float *d_V_new,
        const float *d_K_rotations, const float *d_V_rotations,
        int head, int max_seq_len,
        int n_kv_heads, int head_dim, int num_tokens,
        hipStream_t stream);

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
    // Dynamic Params for Graph-Capturable Incremental Dequant
    // =========================================================================

    /**
     * @brief Device-side dynamic parameters for TQ incremental dequant.
     *
     * During HIP graph capture, kernel arguments are baked into the graph.
     * This struct lives in device memory; the kernel reads from it at runtime.
     * Between graph replays, host code uploads new values to that device buffer
     * before graph launch on the explicit stage stream.
     */
    struct HIPTQDequantDynamicParams
    {
        int ring_pos;         ///< Ring buffer position of the new token
        int out_offset_elems; ///< Element offset into scratch (position × kv_dim)
        int rope_position;    ///< Absolute position for RoPE (0 if no RoPE)
    };

    // =========================================================================
    // FP16 Linearize + Dequant (full sequence, for prefill)
    // =========================================================================

    extern "C" bool hip_tq_ring_linearize_dequant_fp16(
        _Float16 *d_K_out, _Float16 *d_V_out,
        const void *d_K_cache, const void *d_V_cache,
        const float *d_K_rotations_t, const float *d_V_rotations_t,
        const float *d_K_rotations, const float *d_V_rotations,
        int tail, int count, int max_seq_len,
        int n_kv_heads, int head_dim,
        float rope_theta, int position_start, hipStream_t stream);

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
     * @brief Graph-capturable variant: reads dynamic params from device memory.
     */
    extern "C" bool hip_tq_incremental_single_fp16_dynamic(
        _Float16 *d_K_base, _Float16 *d_V_base,
        const void *d_K_cache, const void *d_V_cache,
        const float *d_K_rotation, const float *d_V_rotation,
        const HIPTQDequantDynamicParams *d_params,
        int n_kv_heads, int head_dim,
        float rope_theta,
        hipStream_t stream);

} // namespace llaminar2
