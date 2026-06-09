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
        ALL_POSITION_LOGITS,       ///< Verifier logits for every row [seq_len, vocab_size]
        ALL_POSITION_LOGITS_LOCAL, ///< Column-parallel verifier logits [seq_len, local_vocab]

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
        LM_HEAD_INPUT_ROW,      ///< Stable [1, d_model] scratch row feeding LM head in bucketed prefill
        LM_HEAD_INPUT_ROWS,     ///< Stable [M<=4, d_model] compact verifier rows feeding LM head

        // ── Collective staging buffers ──────────────────────────────────────
        ALLREDUCE_STAGING, ///< Staging buffer for allreduce
        ALLGATHER_STAGING, ///< Staging buffer for allgather

        // ── FFN norm (for Qwen3+ models) ───────────────────────────────────
        FFN_NORMALIZED, ///< FFN-side RMSNorm output

        // ── Quantization intermediaries ─────────────────────────────────────
        Q_QUANTIZED, ///< Q quantized to Q16_1
        K_QUANTIZED, ///< K quantized to Q16_1

        // ── MoE buffers ─────────────────────────────────────────────────────
        MOE_ROUTER_LOGITS,        ///< Router gate logits [batch, num_experts]
        MOE_EXPERT_INDICES,       ///< Top-k expert indices per token
        MOE_EXPERT_WEIGHTS,       ///< Top-k routing weights per token
        MOE_EXPERT_OUTPUT,        ///< Scratch for per-expert FFN output
        MOE_COMBINED_OUTPUT,      ///< Final combined expert output
        MOE_SHARED_EXPERT_OUTPUT, ///< Shared expert FFN output
        MOE_SHARED_GATE_OUTPUT,   ///< Shared expert after sigmoid gating
        MOE_GATE_SCRATCH,         ///< Expert gate projection scratch [seq, intermediate]
        MOE_UP_SCRATCH,           ///< Expert up projection scratch [seq, intermediate]

        // ── GDN (Gated Delta Network) buffers ───────────────────────────
        ATTN_OUTPUT_GATE,   ///< Sigmoid gate weight for attention output [batch, d_model]
        GDN_CONV_STATE,     ///< Short convolution causal state [n_heads, conv_kernel-1, head_dim]
        GDN_RECURRENCE_IN,  ///< GDN recurrence input (after conv+silu)
        GDN_RECURRENCE_OUT, ///< GDN recurrence output
        GDN_QKV,            ///< GDN QKV projection output [seq_len, qkv_dim]
        GDN_Z,              ///< GDN gate Z projection [seq_len, n_heads * d_v]
        GDN_ALPHA,          ///< GDN alpha (dt) projection [seq_len, n_heads]
        GDN_BETA,           ///< GDN beta projection [seq_len, n_heads]
        FA_GATE,            ///< FA sigmoid gate (extracted from Q projection) [seq_len, n_heads * d_head]
        FA_Q_RAW,           ///< FA raw Q GEMM output (query+gate interleaved) [seq_len, n_heads * d_head * 2]

        // ── Sampling scratch buffers ────────────────────────────────────────
        ARGMAX_PARTIAL_VALS, ///< Per-block partial max values for two-pass GPU argmax [1, kArgmaxPartials]
        ARGMAX_PARTIAL_IDXS, ///< Per-block partial max indices for two-pass GPU argmax [1, kArgmaxPartials]
        STOCHASTIC_TARGET_TOKEN_IDS, ///< Compact verifier/main distribution token ids [4, 256]
        STOCHASTIC_TARGET_PROBS,     ///< Compact verifier/main distribution probabilities [4, 256]
        STOCHASTIC_DRAFT_TOKEN_IDS,  ///< Compact MTP draft distribution token ids [3, 256]
        STOCHASTIC_DRAFT_PROBS,      ///< Compact MTP draft distribution probabilities [3, 256]
        STOCHASTIC_TOPK_PARTIAL_VALS, ///< Per-block stochastic top-k partial values [blocks, 32]
        STOCHASTIC_TOPK_PARTIAL_IDXS, ///< Per-block stochastic top-k partial indices [blocks, 32]
        STOCHASTIC_VERIFY_TOKENS,    ///< Scalar stochastic verifier output tokens [1, 4]
        STOCHASTIC_VERIFY_ACCEPTED,  ///< Scalar stochastic verifier accept flags [1, 4]
        STOCHASTIC_VERIFY_ACCEPT_PROBS, ///< Scalar stochastic verifier accept probabilities [1, 4]
        STOCHASTIC_VERIFY_THRESHOLDS,   ///< Scalar stochastic verifier thresholds [1, 4]

        // ── Prefix cache restore/harvest staging ───────────────────────────
        PREFIX_K_STAGING,
        PREFIX_V_STAGING,
        PREFIX_HYBRID_STATE_STAGING,
        PREFIX_MTP_K_STAGING,
        PREFIX_MTP_V_STAGING,
        PREFIX_TERMINAL_HIDDEN,
        PREFIX_TERMINAL_LOGITS,

        // ── MTP sidecar graph buffers ───────────────────────────────────────
        MTP_EMBEDDING,
        MTP_NORM_HIDDEN,
        MTP_NORM_EMBEDDING,
        MTP_CONCAT,
        MTP_PROJECTED,
        MTP_HIDDEN,
        MTP_Q_PROJ,
        MTP_K_PROJ,
        MTP_V_PROJ,
        MTP_FA_Q_RAW,
        MTP_FA_GATE,
        MTP_Q_ROPE,
        MTP_K_ROPE,
        MTP_ATTN_OUTPUT,
        MTP_ATTN_PROJ,
        MTP_GATE_PROJ,
        MTP_UP_PROJ,
        MTP_FFN_OUTPUT,
        MTP_LOGITS,

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
        case BufferId::ALL_POSITION_LOGITS:
            return "ALL_POSITION_LOGITS";
        case BufferId::ALL_POSITION_LOGITS_LOCAL:
            return "ALL_POSITION_LOGITS_LOCAL";
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
        case BufferId::LM_HEAD_INPUT_ROW:
            return "LM_HEAD_INPUT_ROW";
        case BufferId::LM_HEAD_INPUT_ROWS:
            return "LM_HEAD_INPUT_ROWS";
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
        case BufferId::MOE_ROUTER_LOGITS:
            return "MOE_ROUTER_LOGITS";
        case BufferId::MOE_EXPERT_INDICES:
            return "MOE_EXPERT_INDICES";
        case BufferId::MOE_EXPERT_WEIGHTS:
            return "MOE_EXPERT_WEIGHTS";
        case BufferId::MOE_EXPERT_OUTPUT:
            return "MOE_EXPERT_OUTPUT";
        case BufferId::MOE_COMBINED_OUTPUT:
            return "MOE_COMBINED_OUTPUT";
        case BufferId::MOE_SHARED_EXPERT_OUTPUT:
            return "MOE_SHARED_EXPERT_OUTPUT";
        case BufferId::MOE_SHARED_GATE_OUTPUT:
            return "MOE_SHARED_GATE_OUTPUT";
        case BufferId::ARGMAX_PARTIAL_VALS:
            return "ARGMAX_PARTIAL_VALS";
        case BufferId::ARGMAX_PARTIAL_IDXS:
            return "ARGMAX_PARTIAL_IDXS";
        case BufferId::STOCHASTIC_TARGET_TOKEN_IDS:
            return "STOCHASTIC_TARGET_TOKEN_IDS";
        case BufferId::STOCHASTIC_TARGET_PROBS:
            return "STOCHASTIC_TARGET_PROBS";
        case BufferId::STOCHASTIC_DRAFT_TOKEN_IDS:
            return "STOCHASTIC_DRAFT_TOKEN_IDS";
        case BufferId::STOCHASTIC_DRAFT_PROBS:
            return "STOCHASTIC_DRAFT_PROBS";
        case BufferId::STOCHASTIC_TOPK_PARTIAL_VALS:
            return "STOCHASTIC_TOPK_PARTIAL_VALS";
        case BufferId::STOCHASTIC_TOPK_PARTIAL_IDXS:
            return "STOCHASTIC_TOPK_PARTIAL_IDXS";
        case BufferId::STOCHASTIC_VERIFY_TOKENS:
            return "STOCHASTIC_VERIFY_TOKENS";
        case BufferId::STOCHASTIC_VERIFY_ACCEPTED:
            return "STOCHASTIC_VERIFY_ACCEPTED";
        case BufferId::STOCHASTIC_VERIFY_ACCEPT_PROBS:
            return "STOCHASTIC_VERIFY_ACCEPT_PROBS";
        case BufferId::STOCHASTIC_VERIFY_THRESHOLDS:
            return "STOCHASTIC_VERIFY_THRESHOLDS";
        case BufferId::PREFIX_K_STAGING:
            return "PREFIX_K_STAGING";
        case BufferId::PREFIX_V_STAGING:
            return "PREFIX_V_STAGING";
        case BufferId::PREFIX_HYBRID_STATE_STAGING:
            return "PREFIX_HYBRID_STATE_STAGING";
        case BufferId::PREFIX_MTP_K_STAGING:
            return "PREFIX_MTP_K_STAGING";
        case BufferId::PREFIX_MTP_V_STAGING:
            return "PREFIX_MTP_V_STAGING";
        case BufferId::PREFIX_TERMINAL_HIDDEN:
            return "PREFIX_TERMINAL_HIDDEN";
        case BufferId::PREFIX_TERMINAL_LOGITS:
            return "PREFIX_TERMINAL_LOGITS";
        case BufferId::MTP_EMBEDDING:
            return "MTP_EMBEDDING";
        case BufferId::MTP_NORM_HIDDEN:
            return "MTP_NORM_HIDDEN";
        case BufferId::MTP_NORM_EMBEDDING:
            return "MTP_NORM_EMBEDDING";
        case BufferId::MTP_CONCAT:
            return "MTP_CONCAT";
        case BufferId::MTP_PROJECTED:
            return "MTP_PROJECTED";
        case BufferId::MTP_HIDDEN:
            return "MTP_HIDDEN";
        case BufferId::MTP_Q_PROJ:
            return "MTP_Q_PROJ";
        case BufferId::MTP_K_PROJ:
            return "MTP_K_PROJ";
        case BufferId::MTP_V_PROJ:
            return "MTP_V_PROJ";
        case BufferId::MTP_FA_Q_RAW:
            return "MTP_FA_Q_RAW";
        case BufferId::MTP_FA_GATE:
            return "MTP_FA_GATE";
        case BufferId::MTP_Q_ROPE:
            return "MTP_Q_ROPE";
        case BufferId::MTP_K_ROPE:
            return "MTP_K_ROPE";
        case BufferId::MTP_ATTN_OUTPUT:
            return "MTP_ATTN_OUTPUT";
        case BufferId::MTP_ATTN_PROJ:
            return "MTP_ATTN_PROJ";
        case BufferId::MTP_GATE_PROJ:
            return "MTP_GATE_PROJ";
        case BufferId::MTP_UP_PROJ:
            return "MTP_UP_PROJ";
        case BufferId::MTP_FFN_OUTPUT:
            return "MTP_FFN_OUTPUT";
        case BufferId::MTP_LOGITS:
            return "MTP_LOGITS";
        case BufferId::MOE_GATE_SCRATCH:
            return "MOE_GATE_SCRATCH";
        case BufferId::MOE_UP_SCRATCH:
            return "MOE_UP_SCRATCH";
        case BufferId::ATTN_OUTPUT_GATE:
            return "ATTN_OUTPUT_GATE";
        case BufferId::GDN_CONV_STATE:
            return "GDN_CONV_STATE";
        case BufferId::GDN_RECURRENCE_IN:
            return "GDN_RECURRENCE_IN";
        case BufferId::GDN_RECURRENCE_OUT:
            return "GDN_RECURRENCE_OUT";
        case BufferId::GDN_QKV:
            return "GDN_QKV";
        case BufferId::GDN_Z:
            return "GDN_Z";
        case BufferId::GDN_ALPHA:
            return "GDN_ALPHA";
        case BufferId::GDN_BETA:
            return "GDN_BETA";
        case BufferId::FA_GATE:
            return "FA_GATE";
        case BufferId::FA_Q_RAW:
            return "FA_Q_RAW";
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
