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

#include "../../execution/local_execution/graph/GraphSchema.h"
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

        /**
         * @brief Get weight sharding configuration for Qwen2 tensor parallelism
         *
         * Qwen2 Tensor Parallelism Strategy:
         * ==================================
         *
         * Attention:
         * - Q/K/V projections: COLUMN_PARALLEL (split by heads)
         *   - Weight: [n_heads * head_dim, d_model] → [local_heads * head_dim, d_model]
         * - Wo projection: INPUT_PARALLEL (allreduce after)
         *   - Weight: [d_model, n_heads * head_dim] → [d_model, local_heads * head_dim]
         *
         * FFN (SwiGLU):
         * - Gate/Up projections: COLUMN_PARALLEL (split d_ff)
         *   - Weight: [d_ff, d_model] → [d_ff_local, d_model]
         * - Down projection: INPUT_PARALLEL (allreduce after)
         *   - Weight: [d_model, d_ff] → [d_model, d_ff_local]
         *
         * LM Head:
         * - output.weight: COLUMN_PARALLEL (split vocab, allgather logits)
         *
         * Replicated (full copy on each rank):
         * - Norms, biases, embeddings
         */
        WeightShardingConfig getWeightShardingConfig() const override
        {
            WeightShardingConfig config;

            // Exact match for LM head (no blk prefix)
            config.exact_matches["output.weight"] = WeightShardingMode::ColumnParallel;
            config.exact_dimension_matches["output.weight"] = WeightDimensionType::Vocab;

            // Pattern rules (evaluated in order, first match wins)
            // Format: {pattern, mode, dimension_type, description}
            config.patterns = {
                // ===== Attention Weights =====
                // Wo (attn_output) uses INPUT_PARALLEL - split input dim to match local heads
                {"attn_output.weight", WeightShardingMode::InputParallel, WeightDimensionType::Heads,
                 "Wo projection - split input dim by heads, allreduce after"},

                // Q uses COLUMN_PARALLEL - split output dim (heads)
                {"attn_q.weight", WeightShardingMode::ColumnParallel, WeightDimensionType::Heads,
                 "Q projection - split by attention heads"},

                // K/V use COLUMN_PARALLEL - split output dim (KV heads, may differ from Q heads)
                {"attn_k.weight", WeightShardingMode::ColumnParallel, WeightDimensionType::KVHeads,
                 "K projection - split by KV heads"},
                {"attn_v.weight", WeightShardingMode::ColumnParallel, WeightDimensionType::KVHeads,
                 "V projection - split by KV heads"},

                // QKV biases follow their weights
                {"attn_q.bias", WeightShardingMode::ColumnParallel, WeightDimensionType::Heads,
                 "Q bias - split by attention heads"},
                {"attn_k.bias", WeightShardingMode::ColumnParallel, WeightDimensionType::KVHeads,
                 "K bias - split by KV heads"},
                {"attn_v.bias", WeightShardingMode::ColumnParallel, WeightDimensionType::KVHeads,
                 "V bias - split by KV heads"},

                // ===== FFN Weights =====
                // Gate/Up use COLUMN_PARALLEL - split output dim (d_ff)
                {"ffn_gate.weight", WeightShardingMode::ColumnParallel, WeightDimensionType::FFNHidden,
                 "Gate projection - split by d_ff dimension"},
                {"ffn_up.weight", WeightShardingMode::ColumnParallel, WeightDimensionType::FFNHidden,
                 "Up projection - split by d_ff dimension"},

                // Down uses INPUT_PARALLEL - split input dim to match Gate/Up output
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
            schema.name = "qwen2";
            schema.version = "1.0";

            // ==============================================================
            // Quantization Configuration
            // ==============================================================
            // Use default kv_cache_scale of 8.0 (±8.0 range)
            // This conservative fixed scale covers typical Qwen2 K/V activations
            // with ~2× headroom. Post-RMSNorm activations typically fall in [-3, 3].
            //
            // For tighter precision (if profiling shows smaller activation range):
            //   schema.kv_cache_scale = 4.0f;  // ±4.0 range, better precision
            //
            // schema.kv_cache_scale = 8.0f;  // Default, explicit for documentation

            // Required parameters
            // ==============================================================
            // TP Allreduce Precision Policy
            // ==============================================================
            // First 6 layers use FP32 allreduce for numerical stability in
            // early layers; remaining layers use FP16 (halves transfer size,
            // negligible precision loss vs Q8_0 quantization noise).
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
                StageSpec{.name = "qkv_proj", .type = StageType::FusedQKVGEMM, .inputs = {{"normalized", BufferSemantic::Input}, {"weights.wq", BufferSemantic::Input}, {"weights.wk", BufferSemantic::Input}, {"weights.wv", BufferSemantic::Input}, TensorRef::optional("weights.q_bias", BufferSemantic::Input), TensorRef::optional("weights.k_bias", BufferSemantic::Input), TensorRef::optional("weights.v_bias", BufferSemantic::Input)}, .outputs = {{"Q", BufferSemantic::Output}, {"K", BufferSemantic::Output}, {"V", BufferSemantic::Output}}, .dependencies = {"attn_norm"}, .tp_mode = TPMode::ColumnParallel, .is_optional = true, .exec_policy_key = "exec_gemm"},

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

        /**
         * @brief Check if a weight tensor is optional in Qwen2 architecture
         *
         * Optional weights in Qwen2:
         * - attn_q.bias, attn_k.bias, attn_v.bias: QKV biases (some Qwen2 variants don't have them)
         *
         * Required weights (any blk.N. layer weight):
         * - attn_q.weight, attn_k.weight, attn_v.weight, attn_output.weight
         * - attn_norm.weight, ffn_norm.weight
         * - ffn_gate.weight, ffn_up.weight, ffn_down.weight
         *
         * Model-level required:
         * - token_embd.weight, output_norm.weight, output.weight
         *
         * @param gguf_weight_name The GGUF tensor name (e.g., "blk.0.attn_q.bias")
         * @return true if the weight is optional, false if it's required
         */
        bool isWeightOptional(const std::string &gguf_weight_name) const override
        {
            // QKV biases are optional in Qwen2 (present in some variants)
            // Pattern: blk.N.attn_q.bias, blk.N.attn_k.bias, blk.N.attn_v.bias
            if (gguf_weight_name.find("attn_q.bias") != std::string::npos ||
                gguf_weight_name.find("attn_k.bias") != std::string::npos ||
                gguf_weight_name.find("attn_v.bias") != std::string::npos)
            {
                return true; // Optional
            }

            // QK norm weights: Qwen3-specific, not present in Qwen2
            if (gguf_weight_name.find("attn_q_norm.weight") != std::string::npos ||
                gguf_weight_name.find("attn_k_norm.weight") != std::string::npos)
            {
                return true;
            }

            // All other weights are required
            return false;
        }

        std::vector<std::string> layerWeightSuffixes() const override
        {
            return {
                // Attention weights (required)
                "attn_q.weight", "attn_k.weight", "attn_v.weight",
                "attn_output.weight", "attn_norm.weight",
                // Attention biases (optional per isWeightOptional)
                "attn_q.bias", "attn_k.bias", "attn_v.bias",
                // QK norm weights (optional per isWeightOptional)
                "attn_q_norm.weight", "attn_k_norm.weight",
                // FFN weights (required)
                "ffn_gate.weight", "ffn_up.weight",
                "ffn_down.weight", "ffn_norm.weight"};
        }

        /**
         * @brief Stage output sharding config for Qwen2 tensor parallelism
         *
         * Maps each stage type to how its output is distributed across TP devices.
         * This replaces the hardcoded if-chain in TPSnapshot.h::getStageShardingMode().
         */
        StageShardingConfig getStageShardingConfig() const override
        {
            return {
                // Embedding - replicated (full table on each device)
                {"EMBEDDING", SnapshotShardingMode::REPLICATED},

                // Attention projections - column-parallel (split by heads)
                {"Q_PROJECTION", SnapshotShardingMode::COLUMN_PARALLEL},
                {"K_PROJECTION", SnapshotShardingMode::COLUMN_PARALLEL},
                {"V_PROJECTION", SnapshotShardingMode::COLUMN_PARALLEL},
                {"QKV_PROJECTION", SnapshotShardingMode::COLUMN_PARALLEL},

                // RoPE outputs - column-parallel (per-head)
                {"Q_ROPE", SnapshotShardingMode::COLUMN_PARALLEL},
                {"K_ROPE", SnapshotShardingMode::COLUMN_PARALLEL},

                // Attention context - column-parallel (split by heads)
                {"ATTENTION_CONTEXT", SnapshotShardingMode::COLUMN_PARALLEL},
                {"FUSED_ATTENTION_WO_CONTEXT", SnapshotShardingMode::COLUMN_PARALLEL},

                // Attention output (Wo) - row-parallel (AllReduce combines)
                {"ATTENTION_OUTPUT", SnapshotShardingMode::ROW_PARALLEL},
                {"FUSED_ATTENTION_WO", SnapshotShardingMode::ROW_PARALLEL},

                // Norms - replicated
                {"ATTENTION_NORM", SnapshotShardingMode::REPLICATED},
                {"FFN_NORM", SnapshotShardingMode::REPLICATED},
                {"FINAL_NORM", SnapshotShardingMode::REPLICATED},

                // FFN - column-parallel for gate/up, row-parallel for down
                {"FFN_GATE", SnapshotShardingMode::COLUMN_PARALLEL},
                {"FFN_UP", SnapshotShardingMode::COLUMN_PARALLEL},
                {"FFN_GATE_UP", SnapshotShardingMode::COLUMN_PARALLEL},
                {"FUSED_FFN_GATE_UP", SnapshotShardingMode::COLUMN_PARALLEL},
                {"FFN_SWIGLU", SnapshotShardingMode::COLUMN_PARALLEL},
                {"FFN_DOWN", SnapshotShardingMode::ROW_PARALLEL},

                // Residuals - replicated (after AllReduce)
                {"FFN_RESIDUAL", SnapshotShardingMode::REPLICATED},

                // LM head - column-parallel then AllGather
                {"LM_HEAD", SnapshotShardingMode::GATHERED},
            };
        }
    };

} // namespace llaminar2
