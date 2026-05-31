#pragma once

#include <string>
#include <vector>

namespace llaminar2
{
    class IModelLoader;

    struct MTPDepthWeightNames
    {
        int depth_index = 0;
        int source_layer_index = -1;
        bool nextn_block_layout = false;
        bool moe_ffn_layout = false;

        std::string fc;
        std::string pre_fc_norm_hidden;
        std::string pre_fc_norm_embedding;
        std::string final_norm;

        std::string attn_norm;
        std::string wq;
        std::string wk;
        std::string wv;
        std::string wo;
        std::string q_norm;
        std::string k_norm;
        std::string ffn_norm;
        std::string gate_proj;
        std::string up_proj;
        std::string down_proj;

        std::string moe_gate;
        std::string moe_gate_exps;
        std::string moe_up_exps;
        std::string moe_down_exps;
        std::string shared_expert_gate;
        std::string shared_expert_up;
        std::string shared_expert_down;
        std::string shared_expert_gate_inp;

        std::vector<std::string> requiredNames() const;
    };

    struct MTPWeightManifest
    {
        bool available = false;
        int depth = 0;
        bool use_dedicated_embeddings = false;
        std::vector<MTPDepthWeightNames> depths;
        std::vector<std::string> missing_required;
        std::string diagnostic;

        std::vector<std::string> requiredNames() const;
    };

    MTPWeightManifest discoverMTPWeightManifest(
        const IModelLoader &loader,
        const std::string &architecture,
        int base_layer_count,
        bool explicit_mtp);

} // namespace llaminar2
