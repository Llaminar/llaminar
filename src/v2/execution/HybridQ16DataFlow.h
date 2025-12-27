/**
 * @file HybridQ16DataFlow.h
 * @brief Centralized contract for HybridQ16 mode data flow
 *
 * This file defines the SINGLE SOURCE OF TRUTH for how data flows through
 * the HybridQ16 pipeline. Instead of having format decisions scattered across
 * Qwen2Graph.cpp, ComputeStage.cpp, FusedAttentionWoKernel.h, and JitFusedAttentionWo.h,
 * all components should reference this contract.
 *
 * HybridQ16 Mode Data Flow:
 * ════════════════════════════════════════════════════════════════════════════
 *
 *   ┌─────────────┐     FP32 embedding      ┌─────────────────┐
 *   │  Embedding  │ ─────────────────────▶  │ Q16_1 Residual  │ (quantized)
 *   │   Lookup    │                         │    Buffer       │
 *   └─────────────┘                         └────────┬────────┘
 *                                                    │
 *   ┌─────────────────────────────────────────────────────────────────────┐
 *   │                     TRANSFORMER LAYER N                             │
 *   │                                                                     │
 *   │  ┌──────────────┐                                                   │
 *   │  │ Q16_1→FP32   │◀─────────── Q16_1 Residual Buffer                │
 *   │  │  Dequant     │                                                   │
 *   │  └──────┬───────┘                                                   │
 *   │         │ FP32                                                      │
 *   │         ▼                                                           │
 *   │  ┌──────────────┐                                                   │
 *   │  │   RMSNorm    │                                                   │
 *   │  └──────┬───────┘                                                   │
 *   │         │ FP32                                                      │
 *   │         ▼                                                           │
 *   │  ┌──────────────┐     ┌──────────────┐                              │
 *   │  │  Q/K/V Proj  │────▶│   RoPE       │                              │
 *   │  │   (Q4_0)     │     │  (in-place)  │                              │
 *   │  └──────────────┘     └──────┬───────┘                              │
 *   │                              │ FP32 Q, K, V                         │
 *   │                              ▼                                      │
 *   │  ┌─────────────────────────────────────────────────────────────┐    │
 *   │  │              FUSED ATTENTION + Wo + RESIDUAL                │    │
 *   │  │                                                             │    │
 *   │  │  1. Quantize FP32 Q → Q8_1 (on-the-fly)                     │    │
 *   │  │  2. Attention(Q8_1, K_cache, V_cache) → FP32 context        │    │
 *   │  │  3. Wo projection: FP32 context × dequant(Wo) → FP32        │    │
 *   │  │     (FP32_STREAMING_DEQUANT format for highest precision)   │    │
 *   │  │  4. Load Q16_1 residual, dequant to FP32                    │    │
 *   │  │  5. Add: FP32_wo_output + FP32_residual                     │    │
 *   │  │  6. Quantize result → Q16_1, store to residual buffer       │    │
 *   │  │                                                             │    │
 *   │  └─────────────────────────────────────────────────────────────┘    │
 *   │                              │                                      │
 *   │                              │ Q16_1 (in residual buffer)           │
 *   │                              ▼                                      │
 *   │  ┌──────────────┐                                                   │
 *   │  │ Q16_1→FP32   │                                                   │
 *   │  │  Dequant     │                                                   │
 *   │  └──────┬───────┘                                                   │
 *   │         │ FP32                                                      │
 *   │         ▼                                                           │
 *   │  ┌──────────────┐                                                   │
 *   │  │ FFN RMSNorm  │                                                   │
 *   │  └──────┬───────┘                                                   │
 *   │         │ FP32                                                      │
 *   │         ▼                                                           │
 *   │  ┌──────────────────────────────────────────────────────────────┐   │
 *   │  │                    FFN Block                                 │   │
 *   │  │  Gate/Up (Q4_0) → SwiGLU → Down (Q4_0) → Q8_1 delta          │   │
 *   │  └───────────────────────────────┬──────────────────────────────┘   │
 *   │                                  │ Q8_1 delta                       │
 *   │                                  ▼                                  │
 *   │  ┌──────────────────────────────────────────────────────────────┐   │
 *   │  │              Q8_1 + Q16_1 → Q16_1 RESIDUAL ADD               │   │
 *   │  │                                                              │   │
 *   │  │  1. Load Q16_1 residual, dequant to FP32                     │   │
 *   │  │  2. Load Q8_1 FFN output, dequant to FP32                    │   │
 *   │  │  3. Add: FP32_ffn + FP32_residual                            │   │
 *   │  │  4. Quantize result → Q16_1, store to residual buffer        │   │
 *   │  │                                                              │   │
 *   │  └──────────────────────────────────────────────────────────────┘   │
 *   │                              │                                      │
 *   │                              │ Q16_1 (in residual buffer)           │
 *   └──────────────────────────────┼──────────────────────────────────────┘
 *                                  │
 *                                  ▼
 *                        (Next layer or final norm)
 *
 * ════════════════════════════════════════════════════════════════════════════
 *
 * KEY INVARIANTS:
 * ───────────────
 * 1. The Q16_1 residual buffer is the ONLY place where the residual stream lives
 * 2. Attention fusion reads FROM and writes TO the same residual buffer
 * 3. FFN residual add reads FROM and writes TO the same residual buffer
 * 4. All intermediate activations (Q, K, V, context) are FP32
 * 5. Wo uses FP32_STREAMING_DEQUANT (highest precision, no requantization of context)
 *
 * COMPONENT RESPONSIBILITIES:
 * ───────────────────────────
 * - Qwen2Graph.cpp: Wire buffers according to this contract
 * - FusedAttentionWoStage: Set fuse_residual_add=true, output=residual
 * - FusedAttentionWoKernel: Set residual_type=Q16_1, use_hybrid_wo=true
 * - JitFusedAttentionWo: Handle WoFormat::FP32_STREAMING_DEQUANT + Q16_1 fusion
 * - ResidualAddStage: Use Q8_1 + Q16_1 → Q16_1 kernel for FFN
 */

#pragma once

#include "tensors/TensorTypes.h"
#include "execution/ActivationPrecision.h"

namespace llaminar2
{

    /**
     * @brief Validates that a buffer configuration matches HybridQ16 contract
     *
     * Call this at graph build time to catch misconfigurations early.
     */
    struct HybridQ16Contract
    {
        // Expected tensor types at each stage
        static constexpr TensorType RESIDUAL_BUFFER_TYPE = TensorType::Q16_1;
        static constexpr TensorType ATTENTION_INPUT_TYPE = TensorType::FP32;   // After dequant from Q16_1
        static constexpr TensorType QKV_PROJECTION_OUTPUT = TensorType::FP32;  // Q, K, V after projection
        static constexpr TensorType ATTENTION_CONTEXT_TYPE = TensorType::FP32; // Pre-Wo context
        static constexpr TensorType WO_OUTPUT_TYPE = TensorType::FP32;         // Wo output (in registers)
        static constexpr TensorType FFN_DELTA_TYPE = TensorType::Q8_1;         // FFN down_proj output

        /**
         * @brief Verify attention stage configuration
         *
         * For HybridQ16 fused attention:
         * - Input: FP32 Q (from Q_rope buffer), Q8_1 K/V (from cache)
         * - Output: Q16_1 (directly to residual buffer)
         * - use_hybrid_wo: true (FP32 streaming dequant)
         * - fuse_residual_add: true
         */
        static bool validateAttentionConfig(
            TensorType output_type,
            bool use_hybrid_wo,
            bool fuse_residual_add)
        {
            if (output_type != RESIDUAL_BUFFER_TYPE)
            {
                LOG_ERROR("[HybridQ16Contract] Attention output must be Q16_1, got: "
                          << static_cast<int>(output_type));
                return false;
            }
            if (!use_hybrid_wo)
            {
                LOG_ERROR("[HybridQ16Contract] use_hybrid_wo must be true for HybridQ16");
                return false;
            }
            if (!fuse_residual_add)
            {
                LOG_ERROR("[HybridQ16Contract] fuse_residual_add must be true for HybridQ16");
                return false;
            }
            return true;
        }

        /**
         * @brief Verify FFN residual add configuration
         *
         * For HybridQ16 FFN residual:
         * - Delta input: Q8_1 (from FFN down_proj)
         * - Residual input: Q16_1 (residual buffer)
         * - Output: Q16_1 (same residual buffer, in-place)
         */
        static bool validateFFNResidualConfig(
            TensorType delta_type,
            TensorType residual_type,
            TensorType output_type)
        {
            if (delta_type != FFN_DELTA_TYPE)
            {
                LOG_ERROR("[HybridQ16Contract] FFN delta must be Q8_1, got: "
                          << static_cast<int>(delta_type));
                return false;
            }
            if (residual_type != RESIDUAL_BUFFER_TYPE)
            {
                LOG_ERROR("[HybridQ16Contract] FFN residual must be Q16_1, got: "
                          << static_cast<int>(residual_type));
                return false;
            }
            if (output_type != RESIDUAL_BUFFER_TYPE)
            {
                LOG_ERROR("[HybridQ16Contract] FFN output must be Q16_1, got: "
                          << static_cast<int>(output_type));
                return false;
            }
            return true;
        }

        /**
         * @brief Check if activation precision enables HybridQ16 mode
         */
        static bool isHybridQ16Mode(ActivationPrecision precision)
        {
            return precision == ActivationPrecision::HybridQ16;
        }
    };

    /**
     * @brief Debug helper to trace HybridQ16 data flow
     *
     * Enable with LLAMINAR_HYBRIDQ16_TRACE=1 environment variable.
     * Logs the actual tensor types at each stage boundary.
     */
    class HybridQ16Tracer
    {
    public:
        static void traceStage(const char *stage_name,
                               const char *buffer_name,
                               TensorType actual_type,
                               TensorType expected_type)
        {
            static bool enabled = std::getenv("LLAMINAR_HYBRIDQ16_TRACE") != nullptr;
            if (!enabled)
                return;

            bool match = (actual_type == expected_type);
            if (match)
            {
                LOG_DEBUG("[HybridQ16Trace] " << stage_name << " " << buffer_name
                                              << ": " << tensor_type_name(actual_type) << " ✓");
            }
            else
            {
                LOG_WARN("[HybridQ16Trace] " << stage_name << " " << buffer_name
                                             << ": EXPECTED " << tensor_type_name(expected_type)
                                             << " GOT " << tensor_type_name(actual_type) << " ✗");
            }
        }

    private:
        static const char *tensor_type_name(TensorType t)
        {
            switch (t)
            {
            case TensorType::FP32:
                return "FP32";
            case TensorType::Q8_1:
                return "Q8_1";
            case TensorType::Q16_1:
                return "Q16_1";
            case TensorType::BF16:
                return "BF16";
            default:
                return "UNKNOWN";
            }
        }
    };

} // namespace llaminar2
