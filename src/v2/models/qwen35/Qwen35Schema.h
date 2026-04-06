/**
 * @file Qwen35Schema.h
 * @brief Declarative schema definition for Qwen3.5 Dense architecture
 *
 * Qwen3.5 is a hybrid architecture with two layer types:
 * 1. GDN (Gated Delta Network) layers — 75% of layers
 * 2. Full Attention (FA) layers — 25% of layers (every Nth layer)
 *
 * GDN layers replace standard QKV attention with:
 * - Fused QKV projection (attn_qkv.weight)
 * - Short causal conv1d (kernel=4) + SiLU activation
 * - GDN delta-rule recurrence (replaces softmax attention)
 * - Gated RMSNorm output: RMSNorm(out) ⊙ SiLU(z)
 * - Output projection (ssm_out.weight)
 * - Sigmoid output gate (attn_gate.weight)
 *
 * Full Attention layers use standard GQA:
 * - Separate Q/K/V projections (with per-head QK norms)
 * - Partial RoPE (only first portion of head_dim)
 * - KV cache for autoregressive decode
 * - Output projection (Wo)
 * - Sigmoid output gate (attn_gate.weight)
 *
 * Both layer types share:
 * - Pre-attention RMSNorm
 * - Post-attention RMSNorm
 * - SwiGLU dense FFN
 * - Residual connections
 * - Attention output gating
 *
 * TP annotations for full attention layers are identical to Qwen3.
 * GDN layers have limited TP support (GDN projection = column parallel,
 * GDN output = row parallel).
 */

#pragma once

#include "../../execution/local_execution/graph/GraphSchema.h"
#include "../qwen/Qwen2Schema.h" // Reuse Qwen2 stage sharding config as base
#include <string>

namespace llaminar2
{

    /**
     * @brief Schema factory for Qwen3.5 Dense architecture
     *
     * Creates a heterogeneous schema with two named_templates:
     * - "gdn": GDN layers with delta-rule recurrence
     * - "full_attention": Standard GQA layers with QK norms
     *
     * Both templates share the same FFN block (dense SwiGLU).
     */
    class Qwen35SchemaFactory : public ISchemaFactory
    {
    public:
        Qwen35SchemaFactory() = default;

        std::string architectureName() const override { return "qwen35"; }

        WeightShardingConfig getWeightShardingConfig() const override
        {
            WeightShardingConfig config;

            // Exact match for LM head (no blk prefix)
            config.exact_matches["output.weight"] = WeightShardingMode::ColumnParallel;
            config.exact_dimension_matches["output.weight"] = WeightDimensionType::Vocab;

            config.patterns = {
                // ===== Full Attention Weights =====
                {"attn_output.weight", WeightShardingMode::InputParallel, WeightDimensionType::Heads,
                 "Wo projection - split input dim by heads, allreduce after"},
                {"attn_q.weight", WeightShardingMode::ColumnParallel, WeightDimensionType::Heads,
                 "Q projection - split by attention heads"},
                {"attn_k.weight", WeightShardingMode::ColumnParallel, WeightDimensionType::KVHeads,
                 "K projection - split by KV heads"},
                {"attn_v.weight", WeightShardingMode::ColumnParallel, WeightDimensionType::KVHeads,
                 "V projection - split by KV heads"},

                // QK norm weights: replicated (per-head, size [head_dim])
                {"attn_q_norm.weight", WeightShardingMode::Replicate, WeightDimensionType::Heads,
                 "Q norm gamma - replicated"},
                {"attn_k_norm.weight", WeightShardingMode::Replicate, WeightDimensionType::KVHeads,
                 "K norm gamma - replicated"},

                // ===== GDN Weights =====
                // GDN fused QKV and gate are column-parallel (split output dim)
                {"attn_qkv.weight", WeightShardingMode::ColumnParallel, WeightDimensionType::FusedQKVHeads,
                 "GDN fused QKV projection - 3 sub-blocks [Q|K|V] each split by heads"},
                {"attn_gate.weight", WeightShardingMode::ColumnParallel, WeightDimensionType::Heads,
                 "GDN output gate Z - split output dim"},
                // SSM output projection is row-parallel (split input dim)
                {"ssm_out.weight", WeightShardingMode::InputParallel, WeightDimensionType::Heads,
                 "GDN output projection - split input dim, allreduce after"},
                // SSM alpha/beta: column-parallel to match local head count in recurrence
                {"ssm_alpha.weight", WeightShardingMode::ColumnParallel, WeightDimensionType::Heads,
                 "GDN alpha (decay) - split by heads for TP"},
                {"ssm_beta.weight", WeightShardingMode::ColumnParallel, WeightDimensionType::Heads,
                 "GDN beta (input gate) - split by heads for TP"},
                // SSM conv1d: column-parallel (channels == QKV dim, must match sharded QKV)
                {"ssm_conv1d.weight", WeightShardingMode::ColumnParallel, WeightDimensionType::FusedQKVHeads,
                 "GDN conv1d kernel - channels match fused QKV [Q|K|V] structure"},
                // SSM per-head scalars: column-parallel for correct head assignment per rank
                {"ssm_dt.bias", WeightShardingMode::ColumnParallel, WeightDimensionType::Heads,
                 "GDN dt bias - split by heads for TP"},
                {"ssm_a", WeightShardingMode::ColumnParallel, WeightDimensionType::Heads,
                 "GDN decay A - split by heads for TP"},
                // SSM norm: replicated (gamma size = d_v = state_size, shared across all heads)
                {"ssm_norm.weight", WeightShardingMode::Replicate, WeightDimensionType::Heads,
                 "GDN output norm gamma (size=d_v) - replicated, shared across all heads"},

                // ===== FFN Weights =====
                {"ffn_gate.weight", WeightShardingMode::ColumnParallel, WeightDimensionType::FFNHidden,
                 "Gate projection - split by d_ff dimension"},
                {"ffn_up.weight", WeightShardingMode::ColumnParallel, WeightDimensionType::FFNHidden,
                 "Up projection - split by d_ff dimension"},
                {"ffn_down.weight", WeightShardingMode::InputParallel, WeightDimensionType::FFNHidden,
                 "Down projection - split input dim by d_ff, allreduce after"},
            };

            config.default_mode = WeightShardingMode::Replicate;
            return config;
        }

        GraphSchema createSchema() const override
        {
            GraphSchema schema;
            schema.name = "qwen35";
            schema.version = "1.0";

            schema.tp_allreduce_default_precision = "fp16";
            schema.tp_allreduce_fp32_layer_count = 6;

            schema.required_params = {
                "n_layers", "d_model", "n_heads", "n_kv_heads",
                "head_dim", "d_ff", "vocab_size",
                "rms_norm_eps", "rope_theta"};

            // ==============================================================
            // Embedding Stage (shared)
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
            // Named Template: GDN Layer
            // ==============================================================
            LayerTemplate gdn_template;
            gdn_template.attention_stages = {
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

                // GDN Projection: in_proj_qkv + in_proj_z (4 GEMMs)
                StageSpec{.name = "gdn_proj", .type = StageType::GDNProjection, .inputs = {{"normalized", BufferSemantic::Input}, {"weights.attn_qkv", BufferSemantic::Input}, {"weights.attn_gate", BufferSemantic::Input}, {"weights.ssm_alpha", BufferSemantic::Input}, {"weights.ssm_beta", BufferSemantic::Input}}, .outputs = {{"gdn_qkv", BufferSemantic::Output}, {"gdn_z", BufferSemantic::Output}, {"gdn_alpha", BufferSemantic::Output}, {"gdn_beta", BufferSemantic::Output}}, .dependencies = {"attn_norm"}, .tp_mode = TPMode::ColumnParallel, .is_optional = true, .exec_policy_key = "exec_gemm"},

                // Short conv1d + SiLU on QKV
                StageSpec{.name = "short_conv", .type = StageType::ShortConv1d, .inputs = {{"gdn_qkv", BufferSemantic::InOut}, {"weights.ssm_conv1d", BufferSemantic::Input}}, .outputs = {{"gdn_qkv", BufferSemantic::InOut}}, .dependencies = {"gdn_proj"}, .is_optional = true, .exec_policy_key = "exec_conv1d"},

                // GDN Recurrence (delta rule)
                StageSpec{.name = "gdn_recurrence", .type = StageType::GDNRecurrence, .inputs = {{"gdn_qkv", BufferSemantic::Input}, {"gdn_alpha", BufferSemantic::Input}, {"gdn_beta", BufferSemantic::Input}, {"weights.ssm_dt_bias", BufferSemantic::Input}, {"weights.ssm_a", BufferSemantic::Input}, {"weights.ssm_norm", BufferSemantic::Input}}, .outputs = {{"attn_output", BufferSemantic::Output}}, .dependencies = {"short_conv"}, .is_optional = true, .exec_policy_key = "exec_gdn_recurrence"},

                // Gated RMSNorm: RMSNorm(out) ⊙ SiLU(z)
                StageSpec{.name = "gated_norm", .type = StageType::GatedRMSNorm, .inputs = {{"attn_output", BufferSemantic::InOut}, {"gdn_z", BufferSemantic::Input}, {"weights.ssm_norm", BufferSemantic::Input}}, .outputs = {{"attn_output", BufferSemantic::InOut}}, .dependencies = {"gdn_recurrence"}, .is_optional = true, .exec_policy_key = "exec_gated_rmsnorm"},

                // GDN output projection (ssm_out)
                StageSpec{.name = "gdn_out_proj", .type = StageType::GEMM, .inputs = {{"attn_output", BufferSemantic::Input}, {"weights.ssm_out", BufferSemantic::Input}}, .outputs = {{"attn_proj", BufferSemantic::Output}}, .dependencies = {"gated_norm"}, .tp_mode = TPMode::RowParallel, .is_optional = true, .exec_policy_key = "exec_gemm"},

                // Attention output gate: sigmoid(gate) ⊙ output
                StageSpec{.name = "attn_output_gate", .type = StageType::AttentionOutputGate, .inputs = {{"attn_proj", BufferSemantic::InOut}, {"weights.attn_gate", BufferSemantic::Input}}, .outputs = {{"attn_proj", BufferSemantic::InOut}}, .dependencies = {"gdn_out_proj"}, .is_optional = true, .exec_policy_key = "exec_output_gate"},

                // Attention residual add
                StageSpec{.name = "attn_residual", .type = StageType::ResidualAdd, .inputs = {{"attn_proj", BufferSemantic::Input}, {"hidden", BufferSemantic::InOut}}, .outputs = {{"hidden", BufferSemantic::InOut}}, .dependencies = {"attn_output_gate"}, .is_optional = true, .exec_policy_key = "exec_residual"}};

            // GDN FFN block (same as standard SwiGLU)
            gdn_template.ffn_stages = buildFFNStages();

            // ==============================================================
            // Named Template: Full Attention Layer
            // ==============================================================
            LayerTemplate fa_template;
            fa_template.attention_stages = {
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

                // Fused Q/K/V projection (no biases in Qwen3.5)
                StageSpec{.name = "qkv_proj", .type = StageType::FusedQKVGEMM, .inputs = {{"normalized", BufferSemantic::Input}, {"weights.wq", BufferSemantic::Input}, {"weights.wk", BufferSemantic::Input}, {"weights.wv", BufferSemantic::Input}}, .outputs = {{"Q", BufferSemantic::Output}, {"K", BufferSemantic::Output}, {"V", BufferSemantic::Output}}, .dependencies = {"attn_norm"}, .tp_mode = TPMode::ColumnParallel, .is_optional = true, .exec_policy_key = "exec_gemm"},

                // Per-head Q normalization
                StageSpec{.name = "q_norm", .type = StageType::QKNorm, .inputs = {{"Q", BufferSemantic::InOut}, {"weights.q_norm", BufferSemantic::Input}}, .outputs = {{"Q", BufferSemantic::InOut}}, .dependencies = {"qkv_proj"}, .is_optional = true, .exec_policy_key = "exec_rmsnorm"},

                // Per-head K normalization
                StageSpec{.name = "k_norm", .type = StageType::QKNorm, .inputs = {{"K", BufferSemantic::InOut}, {"weights.k_norm", BufferSemantic::Input}}, .outputs = {{"K", BufferSemantic::InOut}}, .dependencies = {"qkv_proj"}, .is_optional = true, .exec_policy_key = "exec_rmsnorm"},

                // RoPE position encoding (partial — only first portion of head_dim)
                StageSpec{.name = "rope", .type = StageType::RoPE, .inputs = {{"Q", BufferSemantic::InOut}, {"K", BufferSemantic::InOut}}, .outputs = {{"Q", BufferSemantic::InOut}, {"K", BufferSemantic::InOut}}, .dependencies = {"q_norm", "k_norm"}, .is_optional = true, .exec_policy_key = "exec_rope"},

                // KV Cache append
                StageSpec{.name = "kv_append", .type = StageType::KVCacheAppend, .inputs = {{"K", BufferSemantic::Input}, {"V", BufferSemantic::Input}, {"kv_cache", BufferSemantic::InOut}}, .outputs = {}, .dependencies = {"rope"}, .requires_kv_cache = true},

                // Attention computation
                StageSpec{.name = "attention", .type = StageType::AttentionCompute, .inputs = {{"Q", BufferSemantic::Input}, {"K", BufferSemantic::Input}, {"V", BufferSemantic::Input}}, .outputs = {{"attn_output", BufferSemantic::Output}}, .dependencies = {"kv_append"}, .is_optional = true, .exec_policy_key = "exec_attention", .causal = true, .window_size = -1},

                // Output projection (Wo)
                StageSpec{.name = "wo_proj", .type = StageType::GEMM, .inputs = {{"attn_output", BufferSemantic::Input}, {"weights.wo", BufferSemantic::Input}}, .outputs = {{"attn_proj", BufferSemantic::Output}}, .dependencies = {"attention"}, .tp_mode = TPMode::RowParallel, .is_optional = true, .exec_policy_key = "exec_gemm"},

                // Attention output gate: sigmoid(gate) ⊙ output
                StageSpec{.name = "attn_output_gate", .type = StageType::AttentionOutputGate, .inputs = {{"attn_proj", BufferSemantic::InOut}, {"weights.attn_gate", BufferSemantic::Input}}, .outputs = {{"attn_proj", BufferSemantic::InOut}}, .dependencies = {"wo_proj"}, .is_optional = true, .exec_policy_key = "exec_output_gate"},

                // Attention residual
                StageSpec{.name = "attn_residual", .type = StageType::ResidualAdd, .inputs = {{"attn_proj", BufferSemantic::Input}, {"hidden", BufferSemantic::InOut}}, .outputs = {{"hidden", BufferSemantic::InOut}}, .dependencies = {"attn_output_gate"}, .is_optional = true, .exec_policy_key = "exec_residual"}};

            // FA FFN block (same as standard SwiGLU)
            fa_template.ffn_stages = buildFFNStages();

            // ==============================================================
            // Assign named templates
            // ==============================================================
            schema.named_templates["gdn"] = std::move(gdn_template);
            schema.named_templates["full_attention"] = std::move(fa_template);

            // NOTE: layer_template_names is populated by the graph builder
            // based on GraphConfig::full_attention_interval, NOT here.
            // The schema factory creates templates; the config builder
            // assigns layers to templates.

            // Default layer_template = FA (fallback for non-heterogeneous)
            schema.layer_template = schema.named_templates.at("full_attention");

            // ==============================================================
            // LM Head Stages
            // ==============================================================
            schema.lm_head_stages = {
                StageSpec{
                    .name = "final_norm",
                    .type = StageType::RMSNorm,
                    .inputs = {
                        {"hidden", BufferSemantic::InOut},
                        {"weights.final_norm", BufferSemantic::Input}},
                    .outputs = {{"hidden", BufferSemantic::InOut}},
                    .is_optional = true,
                    .exec_policy_key = "exec_rmsnorm"},

                StageSpec{.name = "lm_head", .type = StageType::LMHead, .inputs = {{"hidden", BufferSemantic::Input}, {"weights.lm_head", BufferSemantic::Input}}, .outputs = {{"logits", BufferSemantic::Output}}, .dependencies = {"final_norm"}, .tp_mode = TPMode::ColumnParallel, .is_optional = true, .exec_policy_key = "exec_lm_head"}};

            // ==============================================================
            // Buffer Specifications
            // ==============================================================
            schema.layer_buffers = {
                // Shared buffers (used by both GDN and FA)
                {"normalized", {"seq_len", "d_model"}, "fp32", BufferSemantic::Scratch, "", 0, "RMSNorm output"},
                {"residual", {"seq_len", "d_model"}, "fp32", BufferSemantic::InOut, "", 0, "Residual stream buffer"},
                {"attn_output", {"seq_len", "attn_output_dim"}, "fp32", BufferSemantic::Scratch, "attn_scratch", 10, "Attention/GDN output"},
                {"attn_proj", {"seq_len", "d_model"}, "fp32", BufferSemantic::Scratch, "", 0, "Projection output"},

                // FA-specific buffers
                // fa_q_raw: full Q GEMM output before deinterleaving (query + gate interleaved per head)
                //   Shape: [seq, n_heads * head_dim * 2] since Q proj outputs query + sigmoid gate
                {"fa_q_raw", {"seq_len", "fa_q_full_dim"}, "fp32", BufferSemantic::Scratch, "attn_scratch", 10, "FA Q GEMM output (query+gate interleaved)"},
                {"fa_gate", {"seq_len", "local_qkv_dim"}, "fp32", BufferSemantic::Scratch, "attn_scratch", 5, "FA sigmoid gate (extracted from Q)"},
                {"Q", {"seq_len", "local_qkv_dim"}, "fp32", BufferSemantic::Scratch, "attn_scratch", 10, "Query projection output"},
                {"K", {"seq_len", "local_kv_dim"}, "fp32", BufferSemantic::Scratch, "attn_scratch", 5, "Key projection output"},
                {"V", {"seq_len", "local_kv_dim"}, "fp32", BufferSemantic::Scratch, "attn_scratch", 5, "Value projection output"},

                // GDN-specific buffers
                {"gdn_qkv", {"seq_len", "gdn_qkv_dim"}, "fp32", BufferSemantic::Scratch, "gdn_scratch", 10, "GDN fused QKV"},
                {"gdn_z", {"seq_len", "gdn_inner_size"}, "fp32", BufferSemantic::Scratch, "gdn_scratch", 5, "GDN gate Z"},
                {"gdn_alpha", {"seq_len", "gdn_time_step_rank"}, "fp32", BufferSemantic::Scratch, "gdn_scratch", 1, "GDN alpha"},
                {"gdn_beta", {"seq_len", "gdn_time_step_rank"}, "fp32", BufferSemantic::Scratch, "gdn_scratch", 1, "GDN beta"},

                // FFN buffers (shared)
                {"gate", {"seq_len", "local_d_ff"}, "fp32", BufferSemantic::Scratch, "ffn_scratch", 10, "Gate projection for SwiGLU"},
                {"up", {"seq_len", "local_d_ff"}, "fp32", BufferSemantic::Scratch, "ffn_scratch", 10, "Up projection for SwiGLU"},
                {"ffn_output", {"seq_len", "d_model"}, "fp32", BufferSemantic::Scratch, "", 0, "FFN Down projection output"},

                // Attention workspace buffers (FA layers need these)
                {"workspace_scores", {"batch_size * local_n_heads * seq_len", "seq_len"}, "fp32", BufferSemantic::Scratch, "attn_workspace", 10, "Attention scores"},
                {"workspace_context", {"batch_size * local_n_heads * seq_len", "head_dim"}, "fp32", BufferSemantic::Scratch, "attn_workspace", 5, "Attention context"},
                {"workspace_mask", {"batch_size * seq_len", "seq_len"}, "fp32", BufferSemantic::Scratch, "attn_workspace", 5, "Attention mask"},
            };

            schema.model_buffers = {
                {"hidden", {"seq_len", "d_model"}, "fp32", BufferSemantic::InOut, "", 0, "Main hidden state"},
                {"logits", {"seq_len", "vocab_size"}, "fp32", BufferSemantic::Output, "", 0, "Final logits"},
                {"logits_local", {"seq_len", "local_vocab"}, "fp32", BufferSemantic::Scratch, "", 0, "Local logits shard (TP only)"},
            };

            schema.alias_groups = {
                AliasGroupSpec{
                    .name = "attn_scratch",
                    .buffer_names = {"Q", "K", "V", "attn_output"},
                    .description = "FA attention scratch buffers",
                    .estimated_savings_percent = 20.0f},
                AliasGroupSpec{
                    .name = "gdn_scratch",
                    .buffer_names = {"gdn_qkv", "gdn_z", "gdn_alpha", "gdn_beta"},
                    .description = "GDN scratch buffers",
                    .estimated_savings_percent = 15.0f},
                AliasGroupSpec{
                    .name = "ffn_scratch",
                    .buffer_names = {"gate", "up"},
                    .description = "FFN scratch buffers",
                    .estimated_savings_percent = 15.0f},
                AliasGroupSpec{
                    .name = "attn_workspace",
                    .buffer_names = {"workspace_scores", "workspace_context", "workspace_mask"},
                    .description = "FA attention workspace buffers",
                    .estimated_savings_percent = 10.0f}};

            return schema;
        }

        bool isWeightOptional(const std::string &gguf_weight_name) const override
        {
            // FA-only weights are optional on GDN layers and vice versa.
            // The weight validator checks per-layer; we mark layer-type-specific
            // weights as optional since not all layers have all weights.

            // FA-specific weights (only on full_attention layers)
            if (gguf_weight_name.find("attn_q.weight") != std::string::npos ||
                gguf_weight_name.find("attn_k.weight") != std::string::npos ||
                gguf_weight_name.find("attn_v.weight") != std::string::npos ||
                gguf_weight_name.find("attn_output.weight") != std::string::npos ||
                gguf_weight_name.find("attn_q_norm.weight") != std::string::npos ||
                gguf_weight_name.find("attn_k_norm.weight") != std::string::npos)
            {
                return true;
            }

            // GDN-specific weights (only on GDN layers)
            if (gguf_weight_name.find("attn_qkv.weight") != std::string::npos ||
                gguf_weight_name.find("ssm_alpha.weight") != std::string::npos ||
                gguf_weight_name.find("ssm_beta.weight") != std::string::npos ||
                gguf_weight_name.find("ssm_conv1d.weight") != std::string::npos ||
                gguf_weight_name.find("ssm_dt.bias") != std::string::npos ||
                gguf_weight_name.find("ssm_a") != std::string::npos ||
                gguf_weight_name.find("ssm_norm.weight") != std::string::npos ||
                gguf_weight_name.find("ssm_out.weight") != std::string::npos)
            {
                return true;
            }

            // attn_gate.weight is on ALL layers (both GDN and FA)
            // but mark optional for safety during early development
            if (gguf_weight_name.find("attn_gate.weight") != std::string::npos)
            {
                return true;
            }

            // post_attention_norm replaces ffn_norm for this architecture — required on all layers

            return false;
        }

        std::vector<std::string> layerWeightSuffixes() const override
        {
            return {
                // Shared weights (all layers)
                "attn_norm.weight",
                "post_attention_norm.weight",
                "attn_gate.weight",

                // Full attention weights (FA layers only)
                "attn_q.weight", "attn_k.weight", "attn_v.weight",
                "attn_output.weight",
                "attn_q_norm.weight", "attn_k_norm.weight",

                // GDN weights (GDN layers only)
                "attn_qkv.weight",
                "ssm_alpha.weight", "ssm_beta.weight",
                "ssm_conv1d.weight",
                "ssm_dt.bias", "ssm_a",
                "ssm_norm.weight", "ssm_out.weight",

                // FFN weights (all layers)
                "ffn_gate.weight", "ffn_up.weight",
                "ffn_down.weight"};
        }

        StageShardingConfig getStageShardingConfig() const override
        {
            // Start with Qwen2's sharding config (covers standard attention + FFN)
            Qwen2SchemaFactory qwen2;
            auto config = qwen2.getStageShardingConfig();

            // Add GDN-specific stage sharding annotations
            config["GDN_PROJECTION"] = SnapshotShardingMode::COLUMN_PARALLEL;
            config["GDN_CONV1D"] = SnapshotShardingMode::COLUMN_PARALLEL;
            config["GDN_RECURRENCE"] = SnapshotShardingMode::COLUMN_PARALLEL;
            config["GATED_RMSNORM"] = SnapshotShardingMode::COLUMN_PARALLEL;
            config["GDN_OUTPUT"] = SnapshotShardingMode::ROW_PARALLEL;
            config["ATTENTION_OUTPUT_GATE"] = SnapshotShardingMode::REPLICATED;

            return config;
        }

    private:
        /// Build shared FFN stages (used by both GDN and FA templates)
        static std::vector<StageSpec> buildFFNStages()
        {
            return {
                StageSpec{
                    .name = "ffn_norm",
                    .type = StageType::RMSNorm,
                    .inputs = {
                        {"hidden", BufferSemantic::Input},
                        {"weights.ffn_norm", BufferSemantic::Input}},
                    .outputs = {{"normalized", BufferSemantic::Output}},
                    .is_optional = true,
                    .exec_policy_key = "exec_rmsnorm"},

                StageSpec{.name = "gate_up_proj", .type = StageType::FusedGateUpGEMM, .inputs = {{"normalized", BufferSemantic::Input}, {"weights.gate_proj", BufferSemantic::Input}, {"weights.up_proj", BufferSemantic::Input}}, .outputs = {{"gate", BufferSemantic::Output}, {"up", BufferSemantic::Output}}, .dependencies = {"ffn_norm"}, .tp_mode = TPMode::ColumnParallel, .is_optional = true, .exec_policy_key = "exec_gemm"},

                StageSpec{.name = "swiglu", .type = StageType::SwiGLU, .inputs = {{"gate", BufferSemantic::Input}, {"up", BufferSemantic::InOut}}, .outputs = {{"up", BufferSemantic::InOut}}, .dependencies = {"gate_up_proj"}, .is_optional = true, .exec_policy_key = "exec_swiglu"},

                StageSpec{.name = "down_proj", .type = StageType::GEMM, .inputs = {{"up", BufferSemantic::Input}, {"weights.down_proj", BufferSemantic::Input}}, .outputs = {{"attn_proj", BufferSemantic::Output}}, .dependencies = {"swiglu"}, .tp_mode = TPMode::RowParallel, .is_optional = true, .exec_policy_key = "exec_gemm"},

                StageSpec{.name = "ffn_residual", .type = StageType::ResidualAdd, .inputs = {{"attn_proj", BufferSemantic::Input}, {"hidden", BufferSemantic::InOut}}, .outputs = {{"hidden", BufferSemantic::InOut}}, .dependencies = {"down_proj"}, .is_optional = true, .exec_policy_key = "exec_residual"}};
        }
    };

} // namespace llaminar2
