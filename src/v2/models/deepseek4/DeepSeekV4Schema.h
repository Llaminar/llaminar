/**
 * @file DeepSeekV4Schema.h
 * @brief Declarative DeepSeek-V4 placement and native-weight contracts.
 *
 * Routed/shared experts and the router reuse the shared Qwen MoE schema roles.
 * Hyper-connection, attention, compressor and indexer operands are replicated
 * per participant (the first implementation is single-participant). Exact arena
 * declarations come from DeepSeekV4BufferLayout after metadata admission.
 */
#pragma once
#include "../qwen35moe/Qwen35MoESchema.h"

namespace llaminar2
{
/** @brief Source-independent schema policy; tensor geometry belongs to DeepSeekV4Config. */
class DeepSeekV4SchemaFactory final : public Qwen35MoESchemaFactory
{
public:
    std::string architectureName() const override { return "deepseek4"; }

    /** @brief The Sinkhorn terminal mixer and the ordinary output norm feed the vocabulary head. */
    std::vector<const char *> terminalInputWeightNames() const override
    {
        return {"output_hc_fn.weight", "output_hc_base.weight", "output_hc_scale.weight", "output_norm.weight"};
    }

    /** @brief DeepSeek-V4 emits DSML tool-call blocks, not Qwen XML. */
    ToolCallFormat getToolCallFormat() const override { return ToolCallFormat::DEEPSEEK_V4_DSML; }

    /**
     * @brief Budget close for the Llaminar thinking-budget extension.
     * @details DeepSeek-V4 answers immediately after "</think>" (no blank line,
     * unlike Qwen), so the forced close ends with the bare delimiter.
     */
    std::string getStopThinkingPrompt() const override
    {
        return "\n\nConsidering the limited time by the user, I have to give the solution based on the "
               "thinking directly now.</think>";
    }

    GraphSchema createSchema() const override
    {
        auto schema = Qwen35MoESchemaFactory::createSchema();
        schema.name = architectureName();
        return schema;
    }

    WeightShardingConfig getWeightShardingConfig() const override
    {
        auto config = Qwen35MoESchemaFactory::getWeightShardingConfig();
        for (const char *prefix : {"hc_", "attn_q_a", "attn_q_b", "attn_kv", "attn_output_a", "attn_output_b",
                                   "attn_sinks", "attn_compressor", "indexer", "exp_probs_b", "ffn_gate_tid2eid"})
            config.patterns.insert(config.patterns.begin(), {prefix, WeightShardingMode::Replicate,
                WeightDimensionType::None, "DeepSeek-V4 operands are replicated per participant"});
        return config;
    }

    /** @brief Ratio- and hash-dependent roles are optional in the generic inventory; DeepSeekV4Config
     *         authenticates exact presence per layer. */
    bool isWeightOptional(const std::string &name) const override
    {
        for (const char *role : {"attn_compressor", "indexer", "exp_probs_b", "ffn_gate_tid2eid",
                                 "ffn_gate_inp_shexp", "attn_q.", "attn_k.", "attn_v.", "attn_output.weight",
                                 "post_attention_norm"})
            if (name.find(role) != std::string::npos) return true;
        return Qwen35MoESchemaFactory::isWeightOptional(name);
    }

    std::vector<std::string> layerWeightSuffixes() const override
    {
        auto names = Qwen35MoESchemaFactory::layerWeightSuffixes();
        for (const char *name : {
                 "ffn_norm.weight", "hc_attn_fn.weight", "hc_attn_base.weight", "hc_attn_scale.weight",
                 "hc_ffn_fn.weight", "hc_ffn_base.weight", "hc_ffn_scale.weight",
                 "attn_q_a.weight", "attn_q_a_norm.weight", "attn_q_b.weight", "attn_kv.weight",
                 "attn_kv_a_norm.weight", "attn_output_a.weight", "attn_output_b.weight", "attn_sinks.weight",
                 "attn_compressor_kv.weight", "attn_compressor_gate.weight", "attn_compressor_ape.weight",
                 "attn_compressor_norm.weight", "indexer_compressor_kv.weight", "indexer_compressor_gate.weight",
                 "indexer_compressor_ape.weight", "indexer_compressor_norm.weight", "indexer.attn_q_b.weight",
                 "indexer.proj.weight", "exp_probs_b.bias", "exp_probs_b_vl.bias", "ffn_gate_tid2eid.weight"})
            names.emplace_back(name);
        return names;
    }
};
} // namespace llaminar2
