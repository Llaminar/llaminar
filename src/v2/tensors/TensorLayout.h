/**
 * @file TensorLayout.h
 * @brief Tensor memory layout contracts for attention tensors
 *
 * Defines canonical layouts for Q/K/V tensors throughout the pipeline.
 * Layouts are orthogonal to quantization format - a Q16_1Tensor can have
 * any of these layouts depending on where it is in the pipeline.
 *
 * Phase 3 of Tensor Contracts and Validation Framework.
 */

#pragma once

#include <cstdint>
#include <string>

namespace llaminar2
{

    /**
     * @brief Tensor memory layout for multi-dimensional data
     *
     * All attention tensors use one of these canonical layouts.
     * Layouts are orthogonal to quantization format.
     *
     * Naming convention:
     * - First dimension listed is outermost (stride = product of remaining dims)
     * - Last dimension listed is innermost (stride = 1, contiguous)
     *
     * Example: KV_HEAD_POS_DIM means [n_kv_heads][position][head_dim]
     * - Iterating over positions for a single head is contiguous
     * - Iterating over heads requires jumping by (position * head_dim) elements
     */
    enum class TensorLayout : uint8_t
    {
        // =========================================================================
        // Query tensor layouts
        // =========================================================================

        /// Query: [seq_len][n_heads][head_dim] - sequence-major
        /// Natural for batched GEMM output, good for position-wise operations
        Q_SEQ_HEAD_DIM,

        /// Query: [n_heads][seq_len][head_dim] - head-major
        /// Good for per-head parallel attention computation
        Q_HEAD_SEQ_DIM,

        // =========================================================================
        // Key/Value tensor layouts
        // =========================================================================

        /// K/V: [position][n_kv_heads][head_dim] - position-major
        /// Natural for KV cache append (new positions go at end)
        /// Used by: UnifiedKVCache default storage
        KV_POS_HEAD_DIM,

        /// K/V: [n_kv_heads][position][head_dim] - head-major
        /// Optimal for attention computation (K^T @ Q per head)
        /// Used by: Q16IntegerAttention kernel
        KV_HEAD_POS_DIM,

        // =========================================================================
        // Generic layouts
        // =========================================================================

        /// Generic 2D: [rows][cols] - standard row-major
        /// Used for embeddings, hidden states, residuals
        ROW_MAJOR_2D,

        /// Generic 1D: [elements] - flat vector
        ROW_MAJOR_1D,

        // =========================================================================
        // Special layouts
        // =========================================================================

        /// Layout not specified or not applicable
        /// Default for tensors that haven't had layout set explicitly
        UNKNOWN
    };

    /**
     * @brief Get human-readable layout name
     * @param layout The layout enum value
     * @return String representation of the layout
     */
    inline const char *layoutName(TensorLayout layout)
    {
        switch (layout)
        {
        case TensorLayout::Q_SEQ_HEAD_DIM:
            return "Q_SEQ_HEAD_DIM [seq][n_heads][head_dim]";
        case TensorLayout::Q_HEAD_SEQ_DIM:
            return "Q_HEAD_SEQ_DIM [n_heads][seq][head_dim]";
        case TensorLayout::KV_POS_HEAD_DIM:
            return "KV_POS_HEAD_DIM [pos][n_kv_heads][head_dim]";
        case TensorLayout::KV_HEAD_POS_DIM:
            return "KV_HEAD_POS_DIM [n_kv_heads][pos][head_dim]";
        case TensorLayout::ROW_MAJOR_2D:
            return "ROW_MAJOR_2D [rows][cols]";
        case TensorLayout::ROW_MAJOR_1D:
            return "ROW_MAJOR_1D [elements]";
        case TensorLayout::UNKNOWN:
            return "UNKNOWN";
        default:
            return "INVALID";
        }
    }

    /**
     * @brief Get short layout name (for logging)
     * @param layout The layout enum value
     * @return Short string representation
     */
    inline const char *layoutNameShort(TensorLayout layout)
    {
        switch (layout)
        {
        case TensorLayout::Q_SEQ_HEAD_DIM:
            return "Q_SHD";
        case TensorLayout::Q_HEAD_SEQ_DIM:
            return "Q_HSD";
        case TensorLayout::KV_POS_HEAD_DIM:
            return "KV_PHD";
        case TensorLayout::KV_HEAD_POS_DIM:
            return "KV_HPD";
        case TensorLayout::ROW_MAJOR_2D:
            return "2D";
        case TensorLayout::ROW_MAJOR_1D:
            return "1D";
        case TensorLayout::UNKNOWN:
            return "UNK";
        default:
            return "???";
        }
    }

    /**
     * @brief Check if two layouts are compatible without transpose
     *
     * Compatible means data can be reinterpreted without copying:
     * - Same layout is always compatible
     * - UNKNOWN is compatible with anything (permissive for migration)
     * - ROW_MAJOR_2D is compatible with specific layouts when semantically equivalent
     *
     * @param a First layout
     * @param b Second layout
     * @return true if layouts are compatible without transpose
     */
    inline bool layoutsCompatible(TensorLayout a, TensorLayout b)
    {
        // Same layout is always compatible
        if (a == b)
            return true;

        // UNKNOWN is compatible with anything (permissive during migration)
        if (a == TensorLayout::UNKNOWN || b == TensorLayout::UNKNOWN)
            return true;

        // ROW_MAJOR_2D can be reinterpreted as Q_SEQ_HEAD_DIM or KV_POS_HEAD_DIM
        // when the tensor shape matches (this is the common case during pipeline setup)
        if (a == TensorLayout::ROW_MAJOR_2D || b == TensorLayout::ROW_MAJOR_2D)
            return true;

        // All other combinations require explicit transpose
        return false;
    }

    /**
     * @brief Check if a layout is for Query tensors
     */
    inline bool isQueryLayout(TensorLayout layout)
    {
        return layout == TensorLayout::Q_SEQ_HEAD_DIM ||
               layout == TensorLayout::Q_HEAD_SEQ_DIM;
    }

    /**
     * @brief Check if a layout is for Key/Value tensors
     */
    inline bool isKVLayout(TensorLayout layout)
    {
        return layout == TensorLayout::KV_POS_HEAD_DIM ||
               layout == TensorLayout::KV_HEAD_POS_DIM;
    }

    /**
     * @brief Check if a layout is generic (not attention-specific)
     */
    inline bool isGenericLayout(TensorLayout layout)
    {
        return layout == TensorLayout::ROW_MAJOR_2D ||
               layout == TensorLayout::ROW_MAJOR_1D ||
               layout == TensorLayout::UNKNOWN;
    }

    /**
     * @brief Get the transpose target for converting between position-major and head-major
     *
     * @param current Current layout
     * @return Target layout after transpose, or UNKNOWN if not a KV layout
     */
    inline TensorLayout kvTransposeTarget(TensorLayout current)
    {
        switch (current)
        {
        case TensorLayout::KV_POS_HEAD_DIM:
            return TensorLayout::KV_HEAD_POS_DIM;
        case TensorLayout::KV_HEAD_POS_DIM:
            return TensorLayout::KV_POS_HEAD_DIM;
        default:
            return TensorLayout::UNKNOWN;
        }
    }

    /**
     * @brief Get the transpose target for converting between sequence-major and head-major
     *
     * @param current Current layout
     * @return Target layout after transpose, or UNKNOWN if not a Q layout
     */
    inline TensorLayout queryTransposeTarget(TensorLayout current)
    {
        switch (current)
        {
        case TensorLayout::Q_SEQ_HEAD_DIM:
            return TensorLayout::Q_HEAD_SEQ_DIM;
        case TensorLayout::Q_HEAD_SEQ_DIM:
            return TensorLayout::Q_SEQ_HEAD_DIM;
        default:
            return TensorLayout::UNKNOWN;
        }
    }

} // namespace llaminar2
