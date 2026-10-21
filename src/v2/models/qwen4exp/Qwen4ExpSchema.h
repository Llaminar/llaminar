/**
 * @file Qwen4ExpSchema.h
 * @brief Declarative Flash Next placement and native-weight/snapshot contracts.
 *
 * Ordinary GDN, attention projections and expert collectives reuse the shared
 * Qwen MoE schema. Branch mixers, lexical projections and the sparse indexer
 * are explicitly replicated within each TP domain. Exact device/PP arena
 * declarations are supplied by Qwen4ExpBufferLayout after metadata admission.
 */
#pragma once
#include "../qwen35moe/Qwen35MoESchema.h"
#include <algorithm>

namespace llaminar2
{
/** @brief Source-independent schema policy; exact tensor geometry belongs to Qwen4ExpConfig. */
class Qwen4ExpSchemaFactory final : public Qwen35MoESchemaFactory
{
public:
    /** @brief The GGUF architecture is independent of ordinary Qwen3.5 MoE. */
    std::string architectureName() const override { return "qwen4exp"; }
    /** @brief The native terminal branch mixer replaces ordinary final normalization. */
    std::vector<const char *> terminalInputWeightNames() const override
    {
        return {"output_hc_norm.weight", "output_hc_down.weight", "output_hc_up.weight"};
    }
    /** @brief Reuse ordinary block storage; the graph applies its exact participant extension. */
    GraphSchema createSchema() const override
    {
        auto schema=Qwen35MoESchemaFactory::createSchema();
        schema.name=architectureName();
        return schema;
    }
    /** @brief Replicate branch/lexical/indexer computation while retaining native core TP/expert policies. */
    WeightShardingConfig getWeightShardingConfig() const override
    {
        auto config=Qwen35MoESchemaFactory::getWeightShardingConfig();
        for(const char *prefix:{"hc_","ple_","indexer."})
            config.patterns.insert(config.patterns.begin(),{prefix,WeightShardingMode::Replicate,
                WeightDimensionType::None,"Flash Next branch, lexical and indexer operands are replicated per participant"});
        return config;
    }
    /**
     * @brief Describe conditional source presence without weakening exact metadata validation.
     *
     * Qwen4ExpConfig proves every required tensor in its owning layer. Generic
     * suffix inventory visits all layer kinds, so lexical/indexer names are
     * optional outside that exact metadata validation boundary.
     */
    bool isWeightOptional(const std::string &name) const override
    {
        if(name.find(".hc_")!=std::string::npos || name.find("output_hc_")==0)
            return false;
        if(name=="output_norm.weight" || name.find("attn_norm.weight")!=std::string::npos ||
            name.find("post_attention_norm.weight")!=std::string::npos ||
            name.find("ffn_norm.weight")!=std::string::npos || name.find(".indexer.")!=std::string::npos ||
            name.find(".ple_")!=std::string::npos)
            return true;
        return Qwen35MoESchemaFactory::isWeightOptional(name);
    }
    /** @brief Enumerate all typed layer roles once; source validation selects mixer/PLE presence. */
    std::vector<std::string> layerWeightSuffixes() const override
    {
        auto names=Qwen35MoESchemaFactory::layerWeightSuffixes();
        for(const char *role:{"attn","ffn"})
            for(const char *part:{"norm","down","up","inject"})
                names.push_back(std::string("hc_")+role+"_"+part+".weight");
        for(const char *name:{"indexer.q_proj.weight","indexer.k_proj.weight","indexer.q_norm.weight",
            "indexer.k_norm.weight","ple_key.weight","ple_value.weight","ple_norm_key.weight",
            "ple_norm_query.weight","ple_norm_conv.weight","ple_conv1d.weight"})
            names.emplace_back(name);
        return names;
    }
    /** @brief Branch, lexical and indexer snapshots are replicated, independent of core head shards. */
    StageShardingConfig getStageShardingConfig() const override
    {
        auto policy=Qwen35MoESchemaFactory::getStageShardingConfig();
        for(const char *name:{"GR_NORM","GR_READ","GR_WRITE","QSA_INDEXER_QUERY","QSA_INDEXER_POOLED_KEYS",
            "QSA_SELECTED_INDICES","PLE_HASH_INDICES","PLE_GATHER","PLE_GATE","PLE_CONVOLUTION"})
            policy[name]=SnapshotShardingMode::REPLICATED;
        return policy;
    }
};
} // namespace llaminar2
