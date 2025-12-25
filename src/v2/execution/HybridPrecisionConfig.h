/**
 * @file HybridPrecisionConfig.h
 * @brief Per-buffer precision configuration for Hybrid activation precision mode
 *
 * This module defines the precision mapping for each buffer type when using
 * ActivationPrecision::Hybrid. The Hybrid mode optimizes precision per-operation
 * to eliminate unnecessary quantization/requantization while maintaining accuracy.
 *
 * Hybrid Precision Map:
 * - Residual/Normalized: FP32 (numerical stability)
 * - QKV GEMM output: Q8_1 (acceptable quantization error)
 * - Q/K after RoPE: FP32 (avoid requantization)
 * - KV Cache: BF16 (better than Q8_1, 2x compression)
 * - Attention context: FP32 (softmax × V output)
 * - FFN gate/up: Q8_1 (acceptable for FFN intermediates)
 * - All outputs to residual: FP32
 */

#pragma once

#include "RuntimeConfig.h"
#include <string>

namespace llaminar2
{

    /**
     * @brief Buffer types that have precision requirements in Hybrid mode
     */
    enum class HybridBufferType
    {
        // Core buffers (always FP32)
        Residual,
        Normalized,
        Hidden,
        Logits,

        // QKV path
        QKV_GEMM_Output, // Output of QKV projection GEMM
        Q_After_RoPE,    // Q after rotary position embedding
        K_After_RoPE,    // K after rotary position embedding
        KV_Cache,        // KV cache storage

        // Attention
        Attention_Context, // Softmax × V output (pre-Wo)
        Attention_Output,  // After Wo projection (to residual)

        // FFN
        FFN_Gate, // Gate projection output
        FFN_Up,   // Up projection output
        FFN_Down, // Down projection output (to residual)
    };

    /**
     * @brief Per-buffer precision configuration for Hybrid mode
     *
     * Design rationale:
     * - QKV GEMM: Q8_1 output is acceptable (single quantization)
     * - RoPE: Output FP32 to avoid dequant→rotate→requant cycle
     * - KV Cache: BF16 provides better accuracy than Q8_1 with 2x compression
     * - Attention context: Already FP32 from softmax × V
     * - FFN: Q8_1 intermediate is acceptable, but output to residual is FP32
     *
     * NOTE (TODO): KV cache is currently set to FP32 because:
     * - K_rope is FP32 (post-RoPE output)
     * - V is still Q8_1 (no RoPE)
     * - BF16 KV cache would require conversion support in KVCacheAppendStage
     * Future work: Add FP32→BF16 and Q8_1→BF16 conversion in KVCacheAppendStage
     */
    struct HybridPrecisionConfig
    {
        // QKV path precision
        ActivationPrecision qkv_gemm_output = ActivationPrecision::Q8_1; // QKV projection output
        ActivationPrecision q_after_rope = ActivationPrecision::FP32;    // Q after RoPE (no requant!)
        ActivationPrecision k_after_rope = ActivationPrecision::FP32;    // K after RoPE (no requant!)
        // TODO: Change to BF16 once KVCacheAppendStage supports precision conversion
        ActivationPrecision kv_cache = ActivationPrecision::FP32; // KV cache storage (FP32 for now)

        // Attention precision
        ActivationPrecision attention_context = ActivationPrecision::FP32; // softmax × V
        ActivationPrecision attention_output = ActivationPrecision::FP32;  // After Wo (to residual)

        // FFN precision
        ActivationPrecision ffn_gate = ActivationPrecision::Q8_1; // Gate projection
        ActivationPrecision ffn_up = ActivationPrecision::Q8_1;   // Up projection
        ActivationPrecision ffn_down = ActivationPrecision::FP32; // Down projection (to residual)

        /**
         * @brief Get precision for a specific buffer type
         */
        ActivationPrecision getPrecision(HybridBufferType buffer_type) const;

        /**
         * @brief Factory for default hybrid configuration
         */
        static HybridPrecisionConfig defaultConfig();

        /**
         * @brief Check if a buffer type requires separate allocation in Hybrid mode
         *
         * Some buffers (like Q_After_RoPE) need separate FP32 storage when
         * the QKV GEMM output is Q8_1.
         */
        static bool requiresSeparateBuffer(HybridBufferType buffer_type);
    };

    /**
     * @brief Resolve effective precision for a buffer given global activation precision
     *
     * @param global_precision The global activation precision setting
     * @param buffer_type The specific buffer type
     * @param hybrid_config Optional hybrid config (only used when global_precision == Hybrid)
     * @return Effective precision for this buffer
     *
     * Behavior:
     * - FP32/BF16/FP16/Q8_1: Returns global_precision for all buffers
     * - Hybrid: Returns per-buffer precision from hybrid_config
     */
    ActivationPrecision resolveBufferPrecision(
        ActivationPrecision global_precision,
        HybridBufferType buffer_type,
        const HybridPrecisionConfig *hybrid_config = nullptr);

    /**
     * @brief Get human-readable name for buffer type
     */
    const char *hybridBufferTypeToString(HybridBufferType buffer_type);

} // namespace llaminar2
