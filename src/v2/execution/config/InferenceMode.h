/**
 * @file InferenceMode.h
 * @brief Centralized inference mode context for precision and buffer management
 * @author GitHub Copilot
 * @date December 2025
 *
 * This module provides a centralized abstraction for inference mode configuration,
 * replacing scattered `use_hybrid_rope` checks throughout the codebase.
 *
 * ## Problem Solved
 *
 * Previously, Hybrid mode logic was scattered across 3+ files with repeated checks:
 * ```cpp
 * // Scattered in QwenStandardGraph.cpp, DeviceGraphOrchestrator.cpp, etc.
 * bool use_hybrid_rope = (config_.activation_precision == ActivationPrecision::Hybrid) &&
 *                        buffers.Q_rope && buffers.K_rope;
 * if (use_hybrid_rope) { ... }
 * ```
 *
 * Now, a single `InferenceMode` object encapsulates all mode-specific decisions:
 * ```cpp
 * InferenceMode mode(config.activation_precision);
 * if (mode.needsKRope()) { ... }
 * if (mode.needsVDequant()) { ... }
 * ```
 *
 * ## Design Decisions
 *
 * 1. **Single Source of Truth**: All mode-specific behavior derives from `InferenceMode`
 * 2. **Buffer Validation**: Can validate that required buffers are allocated
 * 3. **Self-Documenting**: Method names clearly indicate intent (vs checking enums)
 * 4. **Extensible**: Easy to add new modes without touching multiple files
 *
 * @see HybridPrecisionConfig for per-buffer precision configuration
 * @see ActivationBuffers for buffer struct
 */

#pragma once

#include "RuntimeConfig.h"
#include <string>
#include <vector>

namespace llaminar2
{

    // Forward declarations
    struct ActivationBuffers;

    /**
     * @brief Centralized inference mode context
     *
     * Encapsulates all mode-specific decisions for FP32, Q8_1, and Hybrid modes.
     * Replaces scattered `use_hybrid_rope` checks throughout the codebase.
     *
     * ## Usage
     *
     * ```cpp
     * InferenceMode mode(config.activation_precision);
     *
     * // Check mode requirements
     * if (mode.needsKRope()) {
     *     rope_params.K_out = buffers.K_rope;
     * }
     *
     * // Validate buffer availability
     * auto validation = mode.validateBuffers(buffers);
     * if (!validation.valid) {
     *     LOG_ERROR(validation.error_message);
     * }
     * ```
     */
    class InferenceMode
    {
    public:
        /**
         * @brief Construct from activation precision
         * @param precision The global activation precision setting
         */
        explicit InferenceMode(ActivationPrecision precision = ActivationPrecision::FP32);

        // =========================================================================
        // Mode Identification
        // =========================================================================

        /// Get the underlying precision enum
        ActivationPrecision precision() const { return precision_; }

        /// Check if this is FP32 mode
        bool isFP32() const { return precision_ == ActivationPrecision::FP32; }

        /// Check if this is Q8_1 mode (fully quantized)
        bool isQ8_1() const { return precision_ == ActivationPrecision::Q8_1; }

        /// Check if this is Hybrid mode (mixed precision)
        bool isHybrid() const { return precision_ == ActivationPrecision::Hybrid; }

        /// Check if this is HybridQ16 mode (Q16_1 residual stream)
        bool isHybridQ16() const { return precision_ == ActivationPrecision::HybridQ16; }

        /// Check if this is any Hybrid variant (Hybrid or HybridQ16)
        bool isAnyHybrid() const { return isHybrid() || isHybridQ16(); }

        /// Human-readable mode name
        std::string name() const;

        // =========================================================================
        // Buffer Requirements (What Extra Buffers Does This Mode Need?)
        // =========================================================================

        /**
         * @brief Does this mode need a separate Q_rope buffer?
         *
         * Hybrid/HybridQ16 modes output RoPE to FP32 to avoid Q8_1→rotate→Q8_1 requantization.
         * FP32 and Q8_1 modes apply RoPE in-place.
         */
        bool needsQRope() const { return isAnyHybrid(); }

        /**
         * @brief Does this mode need a separate K_rope buffer?
         *
         * Hybrid/HybridQ16 modes output RoPE to FP32 for KV cache storage.
         * FP32 and Q8_1 modes apply RoPE in-place.
         */
        bool needsKRope() const { return isAnyHybrid(); }

        /**
         * @brief Does this mode need a V_dequant buffer?
         *
         * Hybrid/HybridQ16 modes dequantize V (Q8_1) to FP32 for attention computation.
         * FP32 mode: V is already FP32.
         * Q8_1 mode: Uses fused attention that operates on Q8_1 directly.
         */
        bool needsVDequant() const { return isAnyHybrid(); }

        /**
         * @brief Get list of extra buffer names required by this mode
         *
         * Beyond the standard Q, K, V, attn_output buffers.
         */
        std::vector<std::string> extraRequiredBuffers() const;

        // =========================================================================
        // Attention Strategy (How Should Attention Be Computed?)
        // =========================================================================

        /**
         * @brief Should we use fused Q8_1 attention?
         *
         * Q8_1 mode uses fused attention (JIT microkernel) for efficiency.
         * FP32 and Hybrid modes use decomposed FP32 attention.
         */
        bool usesFusedQ8Attention() const { return isQ8_1(); }

        /**
         * @brief Should we use decomposed FP32 attention?
         *
         * FP32 and Hybrid modes use standard FP32 attention kernels.
         * Q8_1 mode uses fused JIT microkernel.
         */
        bool usesDecomposedFP32Attention() const { return !isQ8_1(); }

        /**
         * @brief What precision is used for attention context (softmax × V)?
         *
         * All modes currently produce FP32 attention context.
         */
        ActivationPrecision attentionContextPrecision() const
        {
            return ActivationPrecision::FP32;
        }

        // =========================================================================
        // Buffer Validation
        // =========================================================================

        /**
         * @brief Result of buffer validation
         */
        struct ValidationResult
        {
            bool valid = true;
            std::string error_message;
            std::vector<std::string> missing_buffers;

            /// Check if validation passed
            operator bool() const { return valid; }
        };

        /**
         * @brief Validate that required buffers are available
         *
         * @param buffers The activation buffers to validate
         * @return ValidationResult with error details if invalid
         */
        ValidationResult validateBuffers(const ActivationBuffers &buffers) const;

        // =========================================================================
        // Factory Methods
        // =========================================================================

        /// Create InferenceMode for FP32
        static InferenceMode FP32() { return InferenceMode(ActivationPrecision::FP32); }

        /// Create InferenceMode for Q8_1
        static InferenceMode Q8_1() { return InferenceMode(ActivationPrecision::Q8_1); }

        /// Create InferenceMode for Hybrid
        static InferenceMode Hybrid() { return InferenceMode(ActivationPrecision::Hybrid); }

    private:
        ActivationPrecision precision_;
    };

    /**
     * @brief Convenience function to check Hybrid mode with buffer availability
     *
     * This is the replacement for the scattered pattern:
     * ```cpp
     * bool use_hybrid_rope = (config.activation_precision == ActivationPrecision::Hybrid) &&
     *                        buffers.Q_rope && buffers.K_rope;
     * ```
     *
     * New pattern:
     * ```cpp
     * bool use_hybrid_rope = isHybridModeActive(mode, buffers);
     * ```
     *
     * @param mode The inference mode
     * @param buffers The activation buffers
     * @return true if Hybrid mode and required buffers are available
     */
    bool isHybridModeActive(const InferenceMode &mode, const ActivationBuffers &buffers);

} // namespace llaminar2
