/**
 * @file WoProjection.h
 * @brief Microkernel μK4: Wo output projection from FP32 context
 * @author David Sanftenberg
 *
 * Projects FP32 attention context through Wo weight matrix.
 * This is the key fusion point - instead of quantizing context to Q8_1
 * and then doing a separate Wo GEMM, we project directly from FP32.
 *
 * Wo layout: [d_model, n_heads * head_dim]
 * For head h: Wo[:, h*head_dim : (h+1)*head_dim]
 */

#pragma once

#include "Q8DotProduct.h" // For Q8_1Block definition
#include <cstdint>

namespace llaminar::v2::kernels::microkernels
{

    /**
     * @brief Weight type for Wo matrix
     */
    enum class WoWeightType
    {
        FP32,             ///< Full precision float (32-bit)
        FP16,             ///< Half precision float (16-bit IEEE 754)
        BF16,             ///< Brain float (16-bit, same exponent range as FP32)
        Q8_1,             ///< Block quantized Q8_1
        Q8_1_VNNI_PACKED, ///< gemm_v4::QuantisedPackedWeights (AVX-512 VNNI), JIT-only
        Q4_0,             ///< Block quantized Q4_0
        // Hybrid mode: VNNI-packed weights with FP32 streaming dequantization
        // This gives highest precision by avoiding activation quantization
        FP32_STREAMING_DEQUANT,
    };

    /**
     * @brief Parameters for Wo projection
     */
    struct WoProjectionParams
    {
        const float *context = nullptr;            ///< FP32 normalized attention context [head_dim]
        const void *wo_weights = nullptr;          ///< Weight data (layout depends on wo_type)
        WoWeightType wo_type = WoWeightType::FP32; ///< Type of Wo weights
        int head_dim = 0;                          ///< Input dimension (64 or 128)
        int d_model = 0;                           ///< Output dimension (896 for Qwen2.5-0.5B)
        int head_idx = 0;                          ///< Which attention head (for striding)
        int n_heads = 0;                           ///< Total number of heads
        float *output = nullptr;                   ///< Output buffer [d_model]
        bool accumulate = false;                   ///< If true, add to output; if false, overwrite
    };

    /**
     * @brief Reference implementation of Wo projection (FP32 weights)
     *
     * Algorithm:
     *   weight_offset = head_idx * head_dim
     *   for out_col in [0, d_model):
     *     dot = 0
     *     for d in [0, head_dim):
     *       dot += context[d] * Wo[out_col * (n_heads * head_dim) + weight_offset + d]
     *     if accumulate:
     *       output[out_col] += dot
     *     else:
     *       output[out_col] = dot
     *
     * @param params Projection parameters
     */
    void wo_projection_fp32_ref(const WoProjectionParams &params);

    /**
     * @brief Reference implementation with FP16 weights
     *
     * Dequantizes FP16 weights on-the-fly and projects.
     *
     * @param params Projection parameters (wo_type must be FP16)
     */
    void wo_projection_fp16_ref(const WoProjectionParams &params);

    /**
     * @brief Reference implementation with BF16 weights
     *
     * Dequantizes BF16 weights on-the-fly and projects.
     * BF16 has same exponent range as FP32, making it ideal for weights.
     *
     * @param params Projection parameters (wo_type must be BF16)
     */
    void wo_projection_bf16_ref(const WoProjectionParams &params);

    /**
     * @brief Reference implementation with Q8_1 weights
     *
     * Dequantizes Wo weights on-the-fly and projects.
     *
     * @param params Projection parameters (wo_type must be Q8_1)
     */
    void wo_projection_q8_ref(const WoProjectionParams &params);

    /**
     * @brief AVX-512 implementation (FP32 weights)
     */
    void wo_projection_fp32_avx512(const WoProjectionParams &params);

    /**
     * @brief AVX-512 implementation (FP16 weights)
     */
    void wo_projection_fp16_avx512(const WoProjectionParams &params);

    /**
     * @brief AVX-512 implementation (BF16 weights)
     */
    void wo_projection_bf16_avx512(const WoProjectionParams &params);

    /**
     * @brief AVX-512 implementation (Q8_1 weights)
     */
    void wo_projection_q8_avx512(const WoProjectionParams &params);

    /**
     * @brief Dispatch to appropriate implementation based on weight type
     */
    void wo_projection_ref(const WoProjectionParams &params);

    /**
     * @brief Dispatch to best available SIMD implementation
     */
    void wo_projection(const WoProjectionParams &params);

    /**
     * @brief Get pointer to Wo slice for a specific head
     *
     * Helper to compute correct offset into Wo weight matrix.
     *
     * @param wo_weights Base pointer to Wo weights
     * @param wo_type Weight type
     * @param head_idx Head index
     * @param head_dim Dimension per head
     * @param n_heads Total heads
     * @param d_model Output dimension
     * @return Pointer to start of this head's Wo slice
     */
    const void *get_wo_head_slice(
        const void *wo_weights,
        WoWeightType wo_type,
        int head_idx,
        int head_dim,
        int n_heads,
        int d_model);

} // namespace llaminar::v2::kernels::microkernels
