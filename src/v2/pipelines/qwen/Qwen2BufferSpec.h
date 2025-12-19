/**
 * @file Qwen2BufferSpec.h
 * @brief Formal buffer specification for Qwen2 pipeline
 * @author David Sanftenberg
 * @date January 2025
 *
 * This file defines the formal buffer specification for Qwen2 pipeline,
 * enabling automatic buffer allocation and lifecycle management through
 * GraphBufferManager with aliasing optimization.
 *
 * ## Buffer Role Semantics
 *
 * - **INPUT**: Read-only buffers (e.g., embedding table lookup)
 * - **OUTPUT**: Write-only buffers, must be preserved across stages
 * - **INOUT**: Modified in-place (residual connections)
 * - **SCRATCH**: Temporary workspace, can be aliased
 *
 * ## Aliasing Opportunities
 *
 * SCRATCH buffers with non-overlapping lifetimes can share physical memory:
 * - Attention phase: Q, K, V, attn_output are consumed before FFN
 * - FFN phase: gate, up, ffn_output are consumed within FFN
 * - Therefore: Q ↔ gate, K ↔ up, V ↔ ffn_output can alias
 *
 * This saves ~35% activation memory per layer.
 */

#pragma once

#include "../../execution/BufferRole.h"
#include <vector>
#include <string>
#include <cstddef>

namespace llaminar2
{

    // =========================================================================
    // Buffer Name Constants
    // =========================================================================

    namespace BufferNames
    {
        // === INOUT Buffers ===
        constexpr const char *RESIDUAL = "residual";
        constexpr const char *NORMALIZED = "normalized";

        // === Attention SCRATCH Buffers ===
        constexpr const char *Q = "Q";
        constexpr const char *K = "K";
        constexpr const char *V = "V";
        constexpr const char *ATTN_OUTPUT = "attn_output";
        constexpr const char *ATTN_PROJ = "attn_proj";
        constexpr const char *WORKSPACE_SCORES = "workspace_scores";
        constexpr const char *WORKSPACE_CONTEXT = "workspace_context";
        constexpr const char *WORKSPACE_MASK = "workspace_mask";

        // === FFN SCRATCH Buffers ===
        constexpr const char *GATE = "gate";
        constexpr const char *UP = "up";
        constexpr const char *FFN_OUTPUT = "ffn_output";

        // === Model-Level Buffers ===
        constexpr const char *CURRENT_HIDDEN = "current_hidden";
        constexpr const char *LOGITS = "logits";

    } // namespace BufferNames

    // =========================================================================
    // Buffer Specification
    // =========================================================================

    /**
     * @brief Buffer specification entry
     *
     * Defines a single buffer's properties for allocation.
     */
    struct Qwen2BufferSpec
    {
        std::string name;             ///< Buffer identifier
        BufferRole role;              ///< Buffer role (INPUT, OUTPUT, SCRATCH, INOUT)
        BufferTensorType tensor_type; ///< Tensor element type
        std::vector<size_t> shape;    ///< Buffer dimensions
        int device_idx;               ///< Target device (-1 for CPU)

        /// Human-readable description
        std::string description;
    };

    // =========================================================================
    // Qwen2BufferSpecBuilder
    // =========================================================================

    /**
     * @brief Builder for Qwen2 buffer specifications
     *
     * Generates buffer specifications based on model configuration.
     *
     * Usage:
     * @code
     * Qwen2BufferSpecBuilder builder(config);
     * auto layer_specs = builder.buildLayerSpecs(seq_len);
     * auto model_specs = builder.buildModelSpecs(batch_size, seq_len);
     * @endcode
     */
    class Qwen2BufferSpecBuilder
    {
    public:
        /**
         * @brief Construct builder with model configuration
         *
         * @param d_model Hidden dimension
         * @param n_heads Number of attention heads
         * @param n_kv_heads Number of KV heads (GQA)
         * @param head_dim Dimension per head
         * @param d_ff FFN intermediate dimension
         * @param vocab_size Vocabulary size
         * @param activation_type Tensor type for activations (default FP32)
         * @param device_idx Default device index
         */
        Qwen2BufferSpecBuilder(
            int d_model,
            int n_heads,
            int n_kv_heads,
            int head_dim,
            int d_ff,
            int vocab_size,
            BufferTensorType activation_type = BufferTensorType::FP32,
            int device_idx = -1);

        /**
         * @brief Build specifications for per-layer activation buffers
         *
         * These buffers are reused across all layers.
         *
         * @param seq_len Sequence length (max tokens per forward pass)
         * @return Vector of buffer specifications
         */
        std::vector<Qwen2BufferSpec> buildLayerSpecs(int seq_len) const;

        /**
         * @brief Build specifications for model-level buffers
         *
         * Includes current_hidden and logits.
         *
         * @param batch_size Batch size
         * @param seq_len Sequence length
         * @return Vector of buffer specifications
         */
        std::vector<Qwen2BufferSpec> buildModelSpecs(int batch_size, int seq_len) const;

        /**
         * @brief Build attention-specific buffer specs
         *
         * Q, K, V, attn_output, attn_proj, workspace_*
         *
         * @param seq_len Sequence length
         * @return Vector of buffer specifications
         */
        std::vector<Qwen2BufferSpec> buildAttentionSpecs(int seq_len) const;

        /**
         * @brief Build FFN-specific buffer specs
         *
         * gate, up, ffn_output
         *
         * @param seq_len Sequence length
         * @return Vector of buffer specifications
         */
        std::vector<Qwen2BufferSpec> buildFFNSpecs(int seq_len) const;

        /**
         * @brief Get aliasing group suggestions
         *
         * Returns groups of buffer names that can safely share memory.
         * This is informational - actual aliasing is computed by LivenessAnalyzer.
         *
         * @return Vector of buffer name groups
         */
        std::vector<std::vector<std::string>> getAliasingGroups() const;

        /**
         * @brief Estimate memory savings from aliasing
         *
         * Theoretical savings assuming optimal aliasing.
         *
         * @param seq_len Sequence length
         * @return Pair of (original_bytes, optimized_bytes)
         */
        std::pair<size_t, size_t> estimateMemorySavings(int seq_len) const;

    private:
        int d_model_;
        int n_heads_;
        int n_kv_heads_;
        int head_dim_;
        int d_ff_;
        int vocab_size_;
        BufferTensorType activation_type_;
        int device_idx_;

        size_t elementSize() const;
    };

    // =========================================================================
    // Utility Functions
    // =========================================================================

    /**
     * @brief Convert buffer specs to StageBufferRequirements
     *
     * For integration with GraphBufferManager.
     */
    StageBufferRequirements toBufferRequirements(const std::vector<Qwen2BufferSpec> &specs);

} // namespace llaminar2
