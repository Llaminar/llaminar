/**
 * @file BufferId.h
 * @brief Typed buffer enumeration for the BufferArena system
 *
 * Replaces string-based buffer identification with a compile-time typed enum.
 * Every activation, workspace, and collective staging buffer has a unique id.
 * KV cache buffers use a separate KVBufferId that encodes layer + key/value.
 */

#pragma once

#include <cstdint>
#include <string>

namespace llaminar2
{

    /**
     * @brief Typed identifier for all activation, workspace, and staging buffers.
     *
     * These are the "well-known" buffers that the graph orchestrator registers
     * with BufferArena.  Weights are handled separately (registerExternalBuffer).
     */
    enum class BufferId : uint32_t
    {
        // ── Cross-layer persistent buffers ──────────────────────────────────
        HIDDEN_STATE, ///< Main hidden state [batch, d_model]
        LOGITS,       ///< Final logits [batch, vocab_size]
        LOGITS_LOCAL, ///< Column-parallel partial logits (TP)

        // ── Per-layer activation buffers (recycled across layers) ───────────
        NORMALIZED,  ///< RMSNorm output
        RESIDUAL,    ///< Residual stream
        Q_PROJ,      ///< Q projection output
        K_PROJ,      ///< K projection output
        V_PROJ,      ///< V projection output
        Q_ROPE,      ///< Post-RoPE Q
        K_ROPE,      ///< Post-RoPE K
        V_DEQUANT,   ///< Dequantized V
        ATTN_OUTPUT, ///< Attention context output
        ATTN_PROJ,   ///< Wo projection output
        GATE_PROJ,   ///< FFN gate projection
        UP_PROJ,     ///< FFN up projection
        FFN_OUTPUT,  ///< FFN down projection output

        // ── Workspace / scratch buffers ─────────────────────────────────────
        ATTN_SCORES_WORKSPACE,  ///< Attention score scratch
        ATTN_CONTEXT_WORKSPACE, ///< Attention context scratch
        GEMM_WORKSPACE,         ///< GEMM temporary storage

        // ── Collective staging buffers ──────────────────────────────────────
        ALLREDUCE_STAGING, ///< Staging buffer for allreduce
        ALLGATHER_STAGING, ///< Staging buffer for allgather

        // ── FFN norm (for Qwen3+ models) ───────────────────────────────────
        FFN_NORMALIZED, ///< FFN-side RMSNorm output

        // ── Quantization intermediaries ─────────────────────────────────────
        Q_QUANTIZED, ///< Q quantized to Q16_1
        K_QUANTIZED, ///< K quantized to Q16_1

        _COUNT ///< Sentinel – must be last
    };

    /**
     * @brief Convert BufferId to a human-readable string.
     */
    inline const char *bufferIdName(BufferId id)
    {
        switch (id)
        {
        case BufferId::HIDDEN_STATE:
            return "HIDDEN_STATE";
        case BufferId::LOGITS:
            return "LOGITS";
        case BufferId::LOGITS_LOCAL:
            return "LOGITS_LOCAL";
        case BufferId::NORMALIZED:
            return "NORMALIZED";
        case BufferId::RESIDUAL:
            return "RESIDUAL";
        case BufferId::Q_PROJ:
            return "Q_PROJ";
        case BufferId::K_PROJ:
            return "K_PROJ";
        case BufferId::V_PROJ:
            return "V_PROJ";
        case BufferId::Q_ROPE:
            return "Q_ROPE";
        case BufferId::K_ROPE:
            return "K_ROPE";
        case BufferId::V_DEQUANT:
            return "V_DEQUANT";
        case BufferId::ATTN_OUTPUT:
            return "ATTN_OUTPUT";
        case BufferId::ATTN_PROJ:
            return "ATTN_PROJ";
        case BufferId::GATE_PROJ:
            return "GATE_PROJ";
        case BufferId::UP_PROJ:
            return "UP_PROJ";
        case BufferId::FFN_OUTPUT:
            return "FFN_OUTPUT";
        case BufferId::ATTN_SCORES_WORKSPACE:
            return "ATTN_SCORES_WORKSPACE";
        case BufferId::ATTN_CONTEXT_WORKSPACE:
            return "ATTN_CONTEXT_WORKSPACE";
        case BufferId::GEMM_WORKSPACE:
            return "GEMM_WORKSPACE";
        case BufferId::ALLREDUCE_STAGING:
            return "ALLREDUCE_STAGING";
        case BufferId::ALLGATHER_STAGING:
            return "ALLGATHER_STAGING";
        case BufferId::FFN_NORMALIZED:
            return "FFN_NORMALIZED";
        case BufferId::Q_QUANTIZED:
            return "Q_QUANTIZED";
        case BufferId::K_QUANTIZED:
            return "K_QUANTIZED";
        case BufferId::_COUNT:
            return "_COUNT";
        }
        return "UNKNOWN";
    }

    /**
     * @brief Per-layer KV cache buffer identifier.
     *
     * KV caches are per-layer and don't fit into the fixed BufferId enum.
     * They are registered separately with BufferArena via layer + type.
     */
    struct KVBufferId
    {
        int layer;
        enum Type : uint8_t
        {
            KEY,
            VALUE
        } type;

        bool operator==(const KVBufferId &o) const { return layer == o.layer && type == o.type; }
        bool operator!=(const KVBufferId &o) const { return !(*this == o); }

        std::string to_string() const
        {
            return std::string("KV_") + (type == KEY ? "KEY" : "VALUE") + "_L" + std::to_string(layer);
        }
    };

} // namespace llaminar2
