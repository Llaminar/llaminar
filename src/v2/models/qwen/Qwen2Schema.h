/**
 * @file Qwen2Schema.h
 * @brief Declarative schema definition for Qwen2 architecture
 * @author David Sanftenberg
 * @date December 2025
 *
 * This file defines the declarative schema for Qwen2 models.
 * The schema describes the graph structure without any runtime
 * conditionals - those are handled by GraphResolver.
 *
 * Qwen2 Architecture:
 * - Pre-RMSNorm attention (with QKV biases)
 * - Grouped Query Attention (GQA)
 * - RoPE position encoding
 * - SwiGLU FFN activation
 * - Tied embeddings (optional)
 *
 * Key TP Annotations:
 * - QKV projection: column_parallel (shard by head)
 * - Wo projection: row_parallel (allreduce after)
 * - Gate/Up projection: column_parallel (shard by d_ff)
 * - Down projection: row_parallel (allreduce after)
 * - LM head: column_parallel (allgather logits)
 */

#pragma once

#include "../../execution/GraphSchema.h"
#include <string>

namespace llaminar2
{

    // =========================================================================
    // Buffer Name Constants
    // =========================================================================
    // Canonical buffer names used throughout the Qwen2 pipeline.
    // These are referenced by schema, buffer allocation, and binding code.

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
        constexpr const char *HIDDEN = "hidden";
        constexpr const char *LOGITS = "logits";

    } // namespace BufferNames

    /**
     * @brief Schema factory for Qwen2 architecture
     */
    class Qwen2SchemaFactory : public ISchemaFactory
    {
    public:
        Qwen2SchemaFactory() = default;

        std::string architectureName() const override { return "qwen2"; }

        GraphSchema createSchema() const override
        {
            GraphSchema schema;
            schema.name = "qwen2";
            schema.version = "1.0";

            // Required parameters
            schema.required_params = {
                "n_layers", "d_model", "n_heads", "n_kv_heads",
                "head_dim", "d_ff", "vocab_size",
                "rms_norm_eps", "rope_theta"};

            // ==============================================================
            // Embedding Stage
            // ==============================================================
            schema.embedding = StageSpec{
                .name = "embedding",
                .type = StageType::Embedding,
                .inputs = {
                    {"token_ids", BufferSemantic::Input},
                    {"weights.embedding_table", BufferSemantic::Input}},
                .outputs = {{"hidden", BufferSemantic::Output}},
                .is_optional = true,
                .exec_policy_key = "exec_embedding"};

            // ==============================================================
            // Layer Template - Attention Block
            // ==============================================================
            schema.layer_template.attention_stages = {
                // Pre-attention RMSNorm
                StageSpec{
                    .name = "attn_norm",
                    .type = StageType::RMSNorm,
                    .inputs = {
                        {"hidden", BufferSemantic::Input},
                        {"weights.attn_norm", BufferSemantic::Input}},
                    .outputs = {{"normalized", BufferSemantic::Output}},
                    .is_optional = true,
                    .exec_policy_key = "exec_rmsnorm"},

                // Fused Q/K/V projection
                StageSpec{.name = "qkv_proj", .type = StageType::FusedQKVGEMM, .inputs = {
                                                                                   {"normalized", BufferSemantic::Input}, {"weights.wq", BufferSemantic::Input}, {"weights.wk", BufferSemantic::Input}, {"weights.wv", BufferSemantic::Input}, {"weights.q_bias", BufferSemantic::Input}, // Optional
                                                                                   {"weights.k_bias", BufferSemantic::Input},                                                                                                                                                             // Optional
                                                                                   {"weights.v_bias", BufferSemantic::Input}                                                                                                                                                              // Optional
                                                                               },
                          .outputs = {{"Q", BufferSemantic::Output}, {"K", BufferSemantic::Output}, {"V", BufferSemantic::Output}},
                          .dependencies = {"attn_norm"},
                          .tp_mode = TPMode::ColumnParallel,
                          .is_optional = true,
                          .exec_policy_key = "exec_gemm"},

                // RoPE position encoding
                StageSpec{.name = "rope", .type = StageType::RoPE, .inputs = {{"Q", BufferSemantic::InOut}, {"K", BufferSemantic::InOut}}, .outputs = {{"Q", BufferSemantic::InOut}, {"K", BufferSemantic::InOut}}, .dependencies = {"qkv_proj"}, .is_optional = true, .exec_policy_key = "exec_rope"},

                // KV Cache append (only if KV cache present)
                StageSpec{.name = "kv_append", .type = StageType::KVCacheAppend, .inputs = {{"K", BufferSemantic::Input}, {"V", BufferSemantic::Input}, {"kv_cache", BufferSemantic::InOut}}, .outputs = {}, .dependencies = {"rope"}, .requires_kv_cache = true},

                // Attention computation
                StageSpec{.name = "attention", .type = StageType::AttentionCompute, .inputs = {
                                                                                        {"Q", BufferSemantic::Input}, {"K", BufferSemantic::Input}, // May be redirected to cache
                                                                                        {"V", BufferSemantic::Input}                                // May be redirected to cache
                                                                                    },
                          .outputs = {{"attn_output", BufferSemantic::Output}},
                          .dependencies = {"kv_append"}, // Or "rope" if no cache
                          .is_optional = true,
                          .exec_policy_key = "exec_attention",
                          .causal = true,
                          .window_size = -1},

                // Output projection (Wo)
                StageSpec{.name = "wo_proj", .type = StageType::GEMM, .inputs = {{"attn_output", BufferSemantic::Input}, {"weights.wo", BufferSemantic::Input}}, .outputs = {{"attn_proj", BufferSemantic::Output}}, .dependencies = {"attention"},
                          .tp_mode = TPMode::RowParallel, // Resolver adds allreduce
                          .is_optional = true,
                          .exec_policy_key = "exec_gemm"},

                // Attention residual
                StageSpec{.name = "attn_residual", .type = StageType::ResidualAdd, .inputs = {{"attn_proj", BufferSemantic::Input}, {"hidden", BufferSemantic::InOut}}, .outputs = {{"hidden", BufferSemantic::InOut}}, .dependencies = {"wo_proj"}, // Or "wo_allreduce" if TP
                          .is_optional = true,
                          .exec_policy_key = "exec_residual"}};

            // ==============================================================
            // Layer Template - FFN Block
            // ==============================================================
            schema.layer_template.ffn_stages = {
                // Pre-FFN RMSNorm
                StageSpec{
                    .name = "ffn_norm",
                    .type = StageType::RMSNorm,
                    .inputs = {
                        {"hidden", BufferSemantic::Input},
                        {"weights.ffn_norm", BufferSemantic::Input}},
                    .outputs = {{"normalized", BufferSemantic::Output}},
                    .is_optional = true,
                    .exec_policy_key = "exec_rmsnorm"},

                // Fused gate/up projection
                StageSpec{.name = "gate_up_proj", .type = StageType::FusedGateUpGEMM, .inputs = {{"normalized", BufferSemantic::Input}, {"weights.gate_proj", BufferSemantic::Input}, {"weights.up_proj", BufferSemantic::Input}}, .outputs = {{"gate", BufferSemantic::Output}, {"up", BufferSemantic::Output}}, .dependencies = {"ffn_norm"}, .tp_mode = TPMode::ColumnParallel, .is_optional = true, .exec_policy_key = "exec_gemm"},

                // SwiGLU activation
                StageSpec{.name = "swiglu", .type = StageType::SwiGLU, .inputs = {{"gate", BufferSemantic::Input}, {"up", BufferSemantic::InOut}}, .outputs = {
                                                                                                                                                       {"up", BufferSemantic::InOut} // In-place
                                                                                                                                                   },
                          .dependencies = {"gate_up_proj"},
                          .is_optional = true,
                          .exec_policy_key = "exec_swiglu"},

                // Down projection
                StageSpec{.name = "down_proj", .type = StageType::GEMM, .inputs = {{"up", BufferSemantic::Input}, {"weights.down_proj", BufferSemantic::Input}}, .outputs = {
                                                                                                                                                                     {"attn_proj", BufferSemantic::Output} // Reuse buffer
                                                                                                                                                                 },
                          .dependencies = {"swiglu"},
                          .tp_mode = TPMode::RowParallel, // Resolver adds allreduce
                          .is_optional = true,
                          .exec_policy_key = "exec_gemm"},

                // FFN residual
                StageSpec{.name = "ffn_residual", .type = StageType::ResidualAdd, .inputs = {{"attn_proj", BufferSemantic::Input}, {"hidden", BufferSemantic::InOut}}, .outputs = {{"hidden", BufferSemantic::InOut}}, .dependencies = {"down_proj"}, // Or "down_allreduce" if TP
                          .is_optional = true,
                          .exec_policy_key = "exec_residual"}};

            // ==============================================================
            // LM Head Stages
            // ==============================================================
            schema.lm_head_stages = {
                // Final RMSNorm
                StageSpec{
                    .name = "final_norm",
                    .type = StageType::RMSNorm,
                    .inputs = {
                        {"hidden", BufferSemantic::InOut},
                        {"weights.final_norm", BufferSemantic::Input}},
                    .outputs = {
                        {"hidden", BufferSemantic::InOut} // In-place
                    },
                    .is_optional = true,
                    .exec_policy_key = "exec_rmsnorm"},

                // LM head projection
                StageSpec{.name = "lm_head", .type = StageType::LMHead, .inputs = {{"hidden", BufferSemantic::Input}, {"weights.lm_head", BufferSemantic::Input}}, .outputs = {
                                                                                                                                                                       {"logits", BufferSemantic::Output} // Or "logits_local" if TP
                                                                                                                                                                   },
                          .dependencies = {"final_norm"},
                          .tp_mode = TPMode::ColumnParallel, // Resolver adds allgather
                          .is_optional = true,
                          .exec_policy_key = "exec_lm_head"}};

            // ==============================================================
            // Buffer Specifications (with aliasing for memory optimization)
            // ==============================================================
            //
            // Buffer aliasing saves ~35% activation memory per layer:
            // - Attention scratch (Q, K, V, attn_output) consumed before FFN
            // - FFN scratch (gate, up) consumed within FFN
            // - Therefore: Q ↔ gate, K ↔ up can share memory
            //
            // Alias groups:
            // - "attn_scratch": Q, K, V, attn_output (largest: Q or attn_output)
            // - "ffn_scratch": gate, up (same size, either can be primary)
            //
            schema.layer_buffers = {
                // Normalized output (reused each layer)
                {"normalized", {"seq_len", "d_model"}, "fp32", BufferSemantic::Scratch, "", 0, "RMSNorm output, consumed by projections"},

                // Attention scratch buffers (can alias with FFN scratch)
                {"Q", {"seq_len", "local_qkv_dim"}, "fp32", BufferSemantic::Scratch, "attn_scratch", 10, "Query projection output"},
                {"K", {"seq_len", "local_kv_dim"}, "fp32", BufferSemantic::Scratch, "attn_scratch", 5, "Key projection output (GQA: smaller than Q)"},
                {"V", {"seq_len", "local_kv_dim"}, "fp32", BufferSemantic::Scratch, "attn_scratch", 5, "Value projection output (GQA: smaller than Q)"},
                {"attn_output", {"seq_len", "local_qkv_dim"}, "fp32", BufferSemantic::Scratch, "attn_scratch", 10, "Attention context output"},

                // Attention projection (d_model sized, persists to residual)
                {"attn_proj", {"seq_len", "d_model"}, "fp32", BufferSemantic::Scratch, "", 0, "Wo projection output, fed to residual"},

                // FFN scratch buffers (can alias with attention scratch)
                {"gate", {"seq_len", "local_d_ff"}, "fp32", BufferSemantic::Scratch, "ffn_scratch", 10, "Gate projection for SwiGLU"},
                {"up", {"seq_len", "local_d_ff"}, "fp32", BufferSemantic::Scratch, "ffn_scratch", 10, "Up projection for SwiGLU"},
            };

            schema.model_buffers = {
                // Primary hidden state (persists across layers)
                {"hidden", {"seq_len", "d_model"}, "fp32", BufferSemantic::InOut, "", 0, "Main hidden state, accumulates residuals"},

                // Final output (full vocab for sampling)
                {"logits", {"seq_len", "vocab_size"}, "fp32", BufferSemantic::Output, "", 0, "Final logits for sampling"},

                // TP-local logits (before allgather)
                {"logits_local", {"seq_len", "local_vocab"}, "fp32", BufferSemantic::Scratch, "", 0, "Local logits shard (TP only)"},
            };

            // ==============================================================
            // Alias Group Definitions
            // ==============================================================
            schema.alias_groups = {
                AliasGroupSpec{
                    .name = "attn_scratch",
                    .buffer_names = {"Q", "K", "V", "attn_output"},
                    .description = "Attention scratch buffers - consumed before FFN phase",
                    .estimated_savings_percent = 20.0f},
                AliasGroupSpec{
                    .name = "ffn_scratch",
                    .buffer_names = {"gate", "up"},
                    .description = "FFN scratch buffers - consumed within FFN phase",
                    .estimated_savings_percent = 15.0f}};

            return schema;
        }
    };

} // namespace llaminar2
