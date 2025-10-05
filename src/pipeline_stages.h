/**
 * @file pipeline_stages.h
 * @brief Pipeline stage definitions for snapshot capture and instrumentation
 * @author David Sanftenberg
 *
 * This header provides a standardized enumeration of transformer pipeline stages
 * used for:
 * - Parity testing (capturing snapshots for comparison with reference implementations)
 * - Performance profiling (timing individual stages)
 * - Debugging and validation (inspecting intermediate states)
 *
 * Design Philosophy:
 * - Stage definitions are architecture-agnostic where possible
 * - Granular enough for detailed validation
 * - Shared between production and test code
 * - Zero overhead when features are disabled
 */

#pragma once

#include <string>

namespace llaminar
{
    /**
     * @brief Stages of the transformer pipeline where snapshots/instrumentation can occur
     *
     * Stages are ordered roughly in execution sequence within a transformer layer.
     * Non-layer stages (embedding, final norm, LM head) use layer_index = -1.
     */
    enum class PipelineStage
    {
        // === Input Processing ===
        EMBEDDING, ///< Token embedding lookup (before layer loop)

        // === Attention Block ===
        ATTENTION_NORM,     ///< RMSNorm/LayerNorm before attention
        QKV_PROJECTION,     ///< Combined Q, K, V linear projections
        Q_PROJECTION,       ///< Query projection only (if separate)
        K_PROJECTION,       ///< Key projection only (if separate)
        V_PROJECTION,       ///< Value projection only (if separate)
        ROPE_APPLICATION,   ///< Rotary position embeddings applied to Q and K
        ATTENTION_SCORES,   ///< Q @ K^T attention scores (before softmax)
        ATTENTION_SOFTMAX,  ///< Softmax over attention scores
        ATTENTION_CONTEXT,  ///< Attention weights @ V (context vectors)
        ATTENTION_OUTPUT,   ///< Output projection W_o (after context)
        ATTENTION_RESIDUAL, ///< After attention residual connection

        // === Feed-Forward Block ===
        FFN_NORM,     ///< RMSNorm/LayerNorm before FFN/MLP
        FFN_GATE,     ///< Gate projection (SwiGLU gate or first linear)
        FFN_UP,       ///< Up projection (SwiGLU up or hidden expansion)
        FFN_SWIGLU,   ///< SwiGLU activation (gate * silu(up))
        FFN_DOWN,     ///< Down projection (back to hidden dimension)
        FFN_RESIDUAL, ///< After FFN residual connection

        // === Output Processing ===
        FINAL_NORM, ///< Final RMSNorm/LayerNorm (after all layers)
        LM_HEAD,    ///< Language model head output (logits)

        // === Extensibility ===
        CUSTOM ///< Custom stage for architecture-specific extensions
    };

    /**
     * @brief Convert PipelineStage enum to string for logging/debugging
     * @param stage Pipeline stage enum value
     * @return String representation of the stage
     */
    inline std::string stage_to_string(PipelineStage stage)
    {
        switch (stage)
        {
        case PipelineStage::EMBEDDING:
            return "EMBEDDING";
        case PipelineStage::ATTENTION_NORM:
            return "ATTENTION_NORM";
        case PipelineStage::QKV_PROJECTION:
            return "QKV_PROJECTION";
        case PipelineStage::Q_PROJECTION:
            return "Q_PROJECTION";
        case PipelineStage::K_PROJECTION:
            return "K_PROJECTION";
        case PipelineStage::V_PROJECTION:
            return "V_PROJECTION";
        case PipelineStage::ROPE_APPLICATION:
            return "ROPE_APPLICATION";
        case PipelineStage::ATTENTION_SCORES:
            return "ATTENTION_SCORES";
        case PipelineStage::ATTENTION_SOFTMAX:
            return "ATTENTION_SOFTMAX";
        case PipelineStage::ATTENTION_CONTEXT:
            return "ATTENTION_CONTEXT";
        case PipelineStage::ATTENTION_OUTPUT:
            return "ATTENTION_OUTPUT";
        case PipelineStage::ATTENTION_RESIDUAL:
            return "ATTENTION_RESIDUAL";
        case PipelineStage::FFN_NORM:
            return "FFN_NORM";
        case PipelineStage::FFN_GATE:
            return "FFN_GATE";
        case PipelineStage::FFN_UP:
            return "FFN_UP";
        case PipelineStage::FFN_SWIGLU:
            return "FFN_SWIGLU";
        case PipelineStage::FFN_DOWN:
            return "FFN_DOWN";
        case PipelineStage::FFN_RESIDUAL:
            return "FFN_RESIDUAL";
        case PipelineStage::FINAL_NORM:
            return "FINAL_NORM";
        case PipelineStage::LM_HEAD:
            return "LM_HEAD";
        case PipelineStage::CUSTOM:
            return "CUSTOM";
        default:
            return "UNKNOWN";
        }
    }

    /**
     * @brief Convert string to PipelineStage enum (for configuration parsing)
     * @param str String representation of stage
     * @return PipelineStage enum value, or CUSTOM if unrecognized
     */
    inline PipelineStage string_to_stage(const std::string &str)
    {
        if (str == "EMBEDDING")
            return PipelineStage::EMBEDDING;
        if (str == "ATTENTION_NORM")
            return PipelineStage::ATTENTION_NORM;
        if (str == "QKV_PROJECTION")
            return PipelineStage::QKV_PROJECTION;
        if (str == "Q_PROJECTION")
            return PipelineStage::Q_PROJECTION;
        if (str == "K_PROJECTION")
            return PipelineStage::K_PROJECTION;
        if (str == "V_PROJECTION")
            return PipelineStage::V_PROJECTION;
        if (str == "ROPE_APPLICATION")
            return PipelineStage::ROPE_APPLICATION;
        if (str == "ATTENTION_SCORES")
            return PipelineStage::ATTENTION_SCORES;
        if (str == "ATTENTION_SOFTMAX")
            return PipelineStage::ATTENTION_SOFTMAX;
        if (str == "ATTENTION_CONTEXT")
            return PipelineStage::ATTENTION_CONTEXT;
        if (str == "ATTENTION_OUTPUT")
            return PipelineStage::ATTENTION_OUTPUT;
        if (str == "ATTENTION_RESIDUAL")
            return PipelineStage::ATTENTION_RESIDUAL;
        if (str == "FFN_NORM")
            return PipelineStage::FFN_NORM;
        if (str == "FFN_GATE")
            return PipelineStage::FFN_GATE;
        if (str == "FFN_UP")
            return PipelineStage::FFN_UP;
        if (str == "FFN_SWIGLU")
            return PipelineStage::FFN_SWIGLU;
        if (str == "FFN_DOWN")
            return PipelineStage::FFN_DOWN;
        if (str == "FFN_RESIDUAL")
            return PipelineStage::FFN_RESIDUAL;
        if (str == "FINAL_NORM")
            return PipelineStage::FINAL_NORM;
        if (str == "LM_HEAD")
            return PipelineStage::LM_HEAD;
        return PipelineStage::CUSTOM;
    }

} // namespace llaminar
