/**
 * @file Qwen3Schema.h
 * @brief Declarative schema definition for Qwen3 architecture
 *
 * Qwen3 is architecturally similar to Qwen2 with two key differences:
 * 1. Per-head QK RMSNorm (attn_q_norm, attn_k_norm) applied before RoPE
 * 2. No QKV biases
 *
 * Qwen3 Architecture:
 * - Pre-RMSNorm attention (no QKV biases)
 * - Per-head Q/K normalization before RoPE (NEW vs Qwen2)
 * - Grouped Query Attention (GQA)
 * - RoPE position encoding
 * - SwiGLU FFN activation
 * - Tied embeddings (optional)
 *
 * TP annotations are identical to Qwen2:
 * - QKV projection: column_parallel (shard by head)
 * - Wo projection: row_parallel (allreduce after)
 * - Gate/Up projection: column_parallel (shard by d_ff)
 * - Down projection: row_parallel (allreduce after)
 * - LM head: column_parallel (allgather logits)
 * - QK norm weights: REPLICATE (per-head, size [head_dim])
 */

#pragma once

#include "../../execution/local_execution/graph/GraphSchema.h"
#include "../qwen/Qwen2Schema.h" // Reuse Qwen2 buffer names
#include <string>

namespace llaminar2
{

    /**
     * @brief Schema factory for Qwen3 architecture
     *
     * Extends Qwen2 schema with:
     * - QKNorm stages between QKV projection and RoPE
     * - No QKV bias weights
     * - QK norm weight entries in sharding config (REPLICATE)
     */
    class Qwen3SchemaFactory : public ISchemaFactory
    {
    public:
        Qwen3SchemaFactory() = default;

        std::string architectureName() const override { return "qwen3"; }

        SamplingParams getRecommendedSamplingParams() const override
        {
            SamplingParams params;
            params.temperature = 0.6f;
            params.top_p = 0.95f;
            params.top_k = 20;
            params.presence_penalty = 1.5f;
            return params;
        }

        std::string getStopThinkingPrompt() const override
        {
            // Qwen3 uses the same thinking format as Qwen3.5
            return "Considering the limited time by the user, I have to give the "
                   "solution based on the thinking directly now.\n</think>\n\n";
        }

        /**
         * @brief Get weight sharding configuration for Qwen3 tensor parallelism
         *
         * Same as Qwen2 but:
         * - No QKV bias patterns (Qwen3 has no biases)
         * - QK norm weights are replicated (size [head_dim], too small to shard)
         */
        WeightShardingConfig getWeightShardingConfig() const override
        {
            WeightShardingConfig config;

            // Exact match for LM head (no blk prefix)
            config.exact_matches["output.weight"] = WeightShardingMode::ColumnParallel;
            config.exact_dimension_matches["output.weight"] = WeightDimensionType::Vocab;

            // Vocab-parallel embedding: each device holds vocab_size/tp_degree rows
            config.exact_matches["token_embd.weight"] = WeightShardingMode::ColumnParallel;
            config.exact_dimension_matches["token_embd.weight"] = WeightDimensionType::Vocab;

            // Pattern rules (evaluated in order, first match wins)
            config.patterns = {
                // ===== Attention Weights =====
                {"attn_output.weight", WeightShardingMode::InputParallel, WeightDimensionType::Heads,
                 "Wo projection - split input dim by heads, allreduce after"},
                {"attn_q.weight", WeightShardingMode::ColumnParallel, WeightDimensionType::Heads,
                 "Q projection - split by attention heads"},
                {"attn_k.weight", WeightShardingMode::ColumnParallel, WeightDimensionType::KVHeads,
                 "K projection - split by KV heads"},
                {"attn_v.weight", WeightShardingMode::ColumnParallel, WeightDimensionType::KVHeads,
                 "V projection - split by KV heads"},

                // QK norm weights: replicated (too small to shard, size [head_dim])
                {"attn_q_norm.weight", WeightShardingMode::Replicate, WeightDimensionType::Heads,
                 "Q norm gamma - replicated (per-head, size head_dim)"},
                {"attn_k_norm.weight", WeightShardingMode::Replicate, WeightDimensionType::KVHeads,
                 "K norm gamma - replicated (per-head, size head_dim)"},

                // NOTE: No QKV bias patterns (Qwen3 has no biases)

                // ===== FFN Weights =====
                {"ffn_gate.weight", WeightShardingMode::ColumnParallel, WeightDimensionType::FFNHidden,
                 "Gate projection - split by d_ff dimension"},
                {"ffn_up.weight", WeightShardingMode::ColumnParallel, WeightDimensionType::FFNHidden,
                 "Up projection - split by d_ff dimension"},
                {"ffn_down.weight", WeightShardingMode::InputParallel, WeightDimensionType::FFNHidden,
                 "Down projection - split input dim by d_ff, allreduce after"},
            };

            // Default: replicate (norms, embeddings, unmatched weights)
            config.default_mode = WeightShardingMode::Replicate;

            return config;
        }

        GraphSchema createSchema() const override
        {
            GraphSchema schema;
            schema.name = "qwen3";
            schema.version = "1.0";

            // TP Allreduce Precision Policy — first 6 layers FP32 for
            // numerical stability; remaining layers FP16
            schema.tp_allreduce_default_precision = "fp16";
            schema.tp_allreduce_fp32_layer_count = 6;

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
                .tp_mode = TPMode::RowParallel,
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

                // Fused Q/K/V projection (no biases in Qwen3)
                StageSpec{.name = "qkv_proj", .type = StageType::FusedQKVGEMM, .inputs = {{"normalized", BufferSemantic::Input}, {"weights.wq", BufferSemantic::Input}, {"weights.wk", BufferSemantic::Input}, {"weights.wv", BufferSemantic::Input}}, .outputs = {{"Q", BufferSemantic::Output}, {"K", BufferSemantic::Output}, {"V", BufferSemantic::Output}}, .dependencies = {"attn_norm"}, .tp_mode = TPMode::ColumnParallel, .is_optional = true, .exec_policy_key = "exec_gemm"},

                // Per-head Q normalization (Qwen3-specific)
                StageSpec{.name = "q_norm", .type = StageType::QKNorm, .inputs = {{"Q", BufferSemantic::InOut}, {"weights.q_norm", BufferSemantic::Input}}, .outputs = {{"Q", BufferSemantic::InOut}}, .dependencies = {"qkv_proj"}, .is_optional = true, .exec_policy_key = "exec_rmsnorm"},

                // Per-head K normalization (Qwen3-specific)
                StageSpec{.name = "k_norm", .type = StageType::QKNorm, .inputs = {{"K", BufferSemantic::InOut}, {"weights.k_norm", BufferSemantic::Input}}, .outputs = {{"K", BufferSemantic::InOut}}, .dependencies = {"qkv_proj"}, .is_optional = true, .exec_policy_key = "exec_rmsnorm"},

                // RoPE position encoding (depends on QK norms)
                StageSpec{.name = "rope", .type = StageType::RoPE, .inputs = {{"Q", BufferSemantic::InOut}, {"K", BufferSemantic::InOut}}, .outputs = {{"Q", BufferSemantic::InOut}, {"K", BufferSemantic::InOut}}, .dependencies = {"q_norm", "k_norm"}, .is_optional = true, .exec_policy_key = "exec_rope"},

                // KV Cache append
                StageSpec{.name = "kv_append", .type = StageType::KVCacheAppend, .inputs = {{"K", BufferSemantic::Input}, {"V", BufferSemantic::Input}, {"kv_cache", BufferSemantic::InOut}}, .outputs = {}, .dependencies = {"rope"}, .requires_kv_cache = true},

                // Attention computation
                StageSpec{.name = "attention", .type = StageType::AttentionCompute, .inputs = {{"Q", BufferSemantic::Input}, {"K", BufferSemantic::Input}, {"V", BufferSemantic::Input}}, .outputs = {{"attn_output", BufferSemantic::Output}}, .dependencies = {"kv_append"}, .is_optional = true, .exec_policy_key = "exec_attention", .causal = true, .window_size = -1},

                // Output projection (Wo)
                StageSpec{.name = "wo_proj", .type = StageType::GEMM, .inputs = {{"attn_output", BufferSemantic::Input}, {"weights.wo", BufferSemantic::Input}}, .outputs = {{"attn_proj", BufferSemantic::Output}}, .dependencies = {"attention"}, .tp_mode = TPMode::RowParallel, .is_optional = true, .exec_policy_key = "exec_gemm"},

                // Attention residual
                StageSpec{.name = "attn_residual", .type = StageType::ResidualAdd, .inputs = {{"attn_proj", BufferSemantic::Input}, {"hidden", BufferSemantic::InOut}}, .outputs = {{"hidden", BufferSemantic::InOut}}, .dependencies = {"wo_proj"}, .is_optional = true, .exec_policy_key = "exec_residual"}};

            // ==============================================================
            // Layer Template - FFN Block (identical to Qwen2)
            // ==============================================================
            schema.layer_template.ffn_stages = {
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

            // ==============================================================
            // LM Head Stages (identical to Qwen2)
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
            // Buffer Specifications (same as Qwen2)
            // ==============================================================
            schema.layer_buffers = {
                {"normalized", {"seq_len", "d_model"}, "fp32", BufferSemantic::Scratch, "", 0, "RMSNorm output"},
                {"Q", {"seq_len", "local_qkv_dim"}, "fp32", BufferSemantic::Scratch, "attn_scratch", 10, "Query projection output"},
                {"K", {"seq_len", "local_kv_dim"}, "fp32", BufferSemantic::Scratch, "attn_scratch", 5, "Key projection output"},
                {"V", {"seq_len", "local_kv_dim"}, "fp32", BufferSemantic::Scratch, "attn_scratch", 5, "Value projection output"},
                {"attn_output", {"seq_len", "local_qkv_dim"}, "fp32", BufferSemantic::Scratch, "attn_scratch", 10, "Attention context output"},
                {"attn_proj", {"seq_len", "d_model"}, "fp32", BufferSemantic::Scratch, "", 0, "Wo projection output"},
                {"gate", {"seq_len", "local_d_ff"}, "fp32", BufferSemantic::Scratch, "ffn_scratch", 10, "Gate projection for SwiGLU"},
                {"up", {"seq_len", "local_d_ff"}, "fp32", BufferSemantic::Scratch, "ffn_scratch", 10, "Up projection for SwiGLU"},
            };

            schema.model_buffers = {
                {"hidden", {"seq_len", "d_model"}, "fp32", BufferSemantic::InOut, "", 0, "Main hidden state"},
                // Only 1 row needed: LMHeadStage computes only last-token logits
                {"logits", {"1", "vocab_size"}, "fp32", BufferSemantic::Output, "", 0, "Final logits"},
                {"logits_local", {"1", "local_vocab"}, "fp32", BufferSemantic::Scratch, "", 0, "Local logits shard (TP only)"},
            };

            schema.alias_groups = {
                AliasGroupSpec{
                    .name = "attn_scratch",
                    .buffer_names = {"Q", "K", "V", "attn_output"},
                    .description = "Attention scratch buffers",
                    .estimated_savings_percent = 20.0f},
                AliasGroupSpec{
                    .name = "ffn_scratch",
                    .buffer_names = {"gate", "up"},
                    .description = "FFN scratch buffers",
                    .estimated_savings_percent = 15.0f}};

            return schema;
        }

        /**
         * @brief Check if a weight tensor is optional in Qwen3 architecture
         *
         * Optional weights in Qwen3:
         * - attn_q_norm.weight, attn_k_norm.weight: QK norm gammas
         *   (optional since the graph code checks for nullptr before inserting QKNorm stages)
         * - attn_q.bias, attn_k.bias, attn_v.bias: Qwen3 has NO QKV biases
         *   (the eager loader always includes these; mark optional so loading doesn't fail)
         */
        bool isWeightOptional(const std::string &gguf_weight_name) const override
        {
            // QK norm weights: present in Qwen3, treated as optional for graceful handling
            if (gguf_weight_name.find("attn_q_norm.weight") != std::string::npos ||
                gguf_weight_name.find("attn_k_norm.weight") != std::string::npos)
            {
                return true;
            }
            // QKV biases: Qwen3 has no biases at all (unlike Qwen2)
            if (gguf_weight_name.find("attn_q.bias") != std::string::npos ||
                gguf_weight_name.find("attn_k.bias") != std::string::npos ||
                gguf_weight_name.find("attn_v.bias") != std::string::npos)
            {
                return true;
            }
            return false;
        }

        std::vector<std::string> layerWeightSuffixes() const override
        {
            return {
                // Attention weights (required)
                "attn_q.weight", "attn_k.weight", "attn_v.weight",
                "attn_output.weight", "attn_norm.weight",
                // Attention biases (optional — Qwen3 has none)
                "attn_q.bias", "attn_k.bias", "attn_v.bias",
                // QK norm weights (optional per isWeightOptional)
                "attn_q_norm.weight", "attn_k_norm.weight",
                // FFN weights (required)
                "ffn_gate.weight", "ffn_up.weight",
                "ffn_down.weight", "ffn_norm.weight"};
        }

        /**
         * Identical to Qwen2 - TP annotations are the same (see header comment).
         */
        StageShardingConfig getStageShardingConfig() const override
        {
            // Reuse Qwen2's config since TP layout is identical
            Qwen2SchemaFactory qwen2;
            return qwen2.getStageShardingConfig();
        }
    };

} // namespace llaminar2
