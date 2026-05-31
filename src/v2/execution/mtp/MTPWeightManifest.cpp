#include "MTPWeightManifest.h"

#include "../../loaders/IModelLoader.h"

#include <algorithm>
#include <sstream>
#include <utility>

namespace llaminar2
{
    namespace
    {
        constexpr int kSupportedMTPDepth = 1;

        bool hasAll(const IModelLoader &loader, const std::vector<std::string> &names)
        {
            return std::all_of(names.begin(), names.end(),
                               [&](const std::string &name)
                               {
                                   return loader.hasTensor(name);
                               });
        }

        std::vector<std::string> missingFrom(const IModelLoader &loader, const std::vector<std::string> &names)
        {
            std::vector<std::string> missing;
            for (const auto &name : names)
            {
                if (!loader.hasTensor(name))
                    missing.push_back(name);
            }
            return missing;
        }

        int metadataDepth(const IModelLoader &loader, const std::string &architecture)
        {
            const std::vector<std::string> keys = {
                architecture + ".nextn_predict_layers",
                architecture + ".mtp_num_hidden_layers",
                architecture + ".mtp.num_hidden_layers",
                "mtp.num_hidden_layers",
                "mtp_num_hidden_layers",
            };

            for (const auto &key : keys)
            {
                const int value = loader.getInt(key, 0);
                if (value > 0)
                    return value;
            }

            return 0;
        }

        MTPDepthWeightNames makeNextNDepth(int depth_index, int source_layer_index)
        {
            const std::string prefix = "blk." + std::to_string(source_layer_index) + ".";
            MTPDepthWeightNames names;
            names.depth_index = depth_index;
            names.source_layer_index = source_layer_index;
            names.nextn_block_layout = true;

            names.fc = prefix + "nextn.eh_proj.weight";
            names.pre_fc_norm_hidden = prefix + "nextn.hnorm.weight";
            names.pre_fc_norm_embedding = prefix + "nextn.enorm.weight";
            names.final_norm = prefix + "nextn.shared_head_norm.weight";

            names.attn_norm = prefix + "attn_norm.weight";
            names.wq = prefix + "attn_q.weight";
            names.wk = prefix + "attn_k.weight";
            names.wv = prefix + "attn_v.weight";
            names.wo = prefix + "attn_output.weight";
            names.q_norm = prefix + "attn_q_norm.weight";
            names.k_norm = prefix + "attn_k_norm.weight";
            names.ffn_norm = prefix + "post_attention_norm.weight";
            names.gate_proj = prefix + "ffn_gate.weight";
            names.up_proj = prefix + "ffn_up.weight";
            names.down_proj = prefix + "ffn_down.weight";
            return names;
        }

        MTPDepthWeightNames makeNextNMoEDepth(int depth_index, int source_layer_index)
        {
            auto names = makeNextNDepth(depth_index, source_layer_index);
            const std::string prefix = "blk." + std::to_string(source_layer_index) + ".";
            names.moe_ffn_layout = true;

            names.gate_proj.clear();
            names.up_proj.clear();
            names.down_proj.clear();
            names.moe_gate = prefix + "ffn_gate_inp.weight";
            names.moe_gate_exps = prefix + "ffn_gate_exps.weight";
            names.moe_up_exps = prefix + "ffn_up_exps.weight";
            names.moe_down_exps = prefix + "ffn_down_exps.weight";
            names.shared_expert_gate = prefix + "ffn_gate_shexp.weight";
            names.shared_expert_up = prefix + "ffn_up_shexp.weight";
            names.shared_expert_down = prefix + "ffn_down_shexp.weight";
            names.shared_expert_gate_inp = prefix + "ffn_gate_inp_shexp.weight";
            return names;
        }

        MTPDepthWeightNames makeGenericMTPDepth(int depth_index)
        {
            const std::string prefix = "mtp.layers." + std::to_string(depth_index) + ".";
            MTPDepthWeightNames names;
            names.depth_index = depth_index;

            names.fc = "mtp.fc.weight";
            names.pre_fc_norm_hidden = "mtp.pre_fc_norm_hidden.weight";
            names.pre_fc_norm_embedding = "mtp.pre_fc_norm_embedding.weight";
            names.final_norm = "mtp.norm.weight";

            names.attn_norm = prefix + "input_layernorm.weight";
            names.wq = prefix + "self_attn.q_proj.weight";
            names.wk = prefix + "self_attn.k_proj.weight";
            names.wv = prefix + "self_attn.v_proj.weight";
            names.wo = prefix + "self_attn.o_proj.weight";
            names.q_norm = prefix + "self_attn.q_norm.weight";
            names.k_norm = prefix + "self_attn.k_norm.weight";
            names.ffn_norm = prefix + "post_attention_layernorm.weight";
            names.gate_proj = prefix + "mlp.gate_proj.weight";
            names.up_proj = prefix + "mlp.up_proj.weight";
            names.down_proj = prefix + "mlp.down_proj.weight";
            return names;
        }

        MTPWeightManifest makeNextNManifest(int depth, int source_layer_start, bool moe_ffn_layout)
        {
            MTPWeightManifest manifest;
            manifest.depth = depth;
            manifest.depths.reserve(static_cast<size_t>(depth));
            for (int i = 0; i < depth; ++i)
            {
                manifest.depths.push_back(moe_ffn_layout
                                             ? makeNextNMoEDepth(i, source_layer_start + i)
                                             : makeNextNDepth(i, source_layer_start + i));
            }
            return manifest;
        }

        std::vector<int> nextNSourceLayerStartCandidates(int block_count_or_base_layer_count, int depth)
        {
            std::vector<int> candidates;
            if (block_count_or_base_layer_count < 0 || depth <= 0)
                return candidates;

            // Qwen3.6 GGUFs report block_count including the trailing nextn sidecar
            // block(s), while some tests and future loaders may pass base layer count.
            if (block_count_or_base_layer_count >= depth)
                candidates.push_back(block_count_or_base_layer_count - depth);
            candidates.push_back(block_count_or_base_layer_count);

            std::sort(candidates.begin(), candidates.end());
            candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
            return candidates;
        }

        MTPWeightManifest unavailable(std::string diagnostic)
        {
            MTPWeightManifest manifest;
            manifest.diagnostic = std::move(diagnostic);
            return manifest;
        }
    } // namespace

    std::vector<std::string> MTPDepthWeightNames::requiredNames() const
    {
        std::vector<std::string> names = {
            fc,
            pre_fc_norm_hidden,
            pre_fc_norm_embedding,
            final_norm,
            attn_norm,
            wq,
            wk,
            wv,
            wo,
            q_norm,
            k_norm,
            ffn_norm,
            gate_proj,
            up_proj,
            down_proj,
            moe_gate,
            moe_gate_exps,
            moe_up_exps,
            moe_down_exps,
            shared_expert_gate,
            shared_expert_up,
            shared_expert_down,
            shared_expert_gate_inp,
        };

        names.erase(std::remove_if(names.begin(), names.end(),
                                   [](const std::string &name)
                                   { return name.empty(); }),
                    names.end());
        return names;
    }

    std::vector<std::string> MTPWeightManifest::requiredNames() const
    {
        std::vector<std::string> names;
        for (const auto &depth_names : depths)
        {
            auto required = depth_names.requiredNames();
            names.insert(names.end(), required.begin(), required.end());
        }
        std::sort(names.begin(), names.end());
        names.erase(std::unique(names.begin(), names.end()), names.end());
        return names;
    }

    MTPWeightManifest discoverMTPWeightManifest(
        const IModelLoader &loader,
        const std::string &architecture,
        int base_layer_count,
        bool explicit_mtp)
    {
        int depth = metadataDepth(loader, architecture);
        if (depth <= 0)
        {
            bool found_nextn_depth = false;
            for (int source_layer_start : nextNSourceLayerStartCandidates(base_layer_count, 1))
            {
                if (loader.hasTensor(makeNextNDepth(0, source_layer_start).fc))
                {
                    found_nextn_depth = true;
                    break;
                }
            }

            if (found_nextn_depth)
            {
                depth = 1;
            }
            else if (loader.hasTensor(makeGenericMTPDepth(0).fc))
            {
                depth = 1;
            }
        }

        if (depth <= 0)
        {
            return unavailable(explicit_mtp
                                   ? "MTP was requested, but no MTP/nextn metadata or tensors were found"
                                   : "MTP metadata/tensors not present");
        }

        if (depth != kSupportedMTPDepth)
        {
            std::ostringstream oss;
            oss << "MTP depth " << depth << " discovered, but only depth "
                << kSupportedMTPDepth << " is supported in this phase";
            auto manifest = unavailable(oss.str());
            manifest.depth = depth;
            return manifest;
        }

        std::vector<std::string> best_nextn_missing;
        for (int source_layer_start : nextNSourceLayerStartCandidates(base_layer_count, depth))
        {
            for (bool moe_ffn_layout : {false, true})
            {
                auto nextn_manifest = makeNextNManifest(depth, source_layer_start, moe_ffn_layout);
                auto nextn_required = nextn_manifest.requiredNames();
                auto nextn_missing = missingFrom(loader, nextn_required);
                if (nextn_missing.empty())
                {
                    nextn_manifest.available = true;
                    nextn_manifest.diagnostic = moe_ffn_layout
                                                   ? "using blk.<n>.nextn MoE MTP layout"
                                                   : "using blk.<n>.nextn MTP layout";
                    return nextn_manifest;
                }

                if (best_nextn_missing.empty() || nextn_missing.size() < best_nextn_missing.size())
                {
                    best_nextn_missing = std::move(nextn_missing);
                }
            }
        }

        MTPWeightManifest generic_manifest;
        generic_manifest.depth = depth;
        generic_manifest.depths.reserve(static_cast<size_t>(depth));
        for (int i = 0; i < depth; ++i)
            generic_manifest.depths.push_back(makeGenericMTPDepth(i));
        auto generic_required = generic_manifest.requiredNames();
        if (hasAll(loader, generic_required))
        {
            generic_manifest.available = true;
            generic_manifest.diagnostic = "using mtp.layers MTP layout";
            return generic_manifest;
        }

        auto manifest = unavailable("MTP tensors are incomplete");
        manifest.depth = depth;
        manifest.missing_required = std::move(best_nextn_missing);
        auto generic_missing = missingFrom(loader, generic_required);
        manifest.missing_required.insert(
            manifest.missing_required.end(),
            generic_missing.begin(),
            generic_missing.end());
        std::sort(manifest.missing_required.begin(), manifest.missing_required.end());
        manifest.missing_required.erase(
            std::unique(manifest.missing_required.begin(), manifest.missing_required.end()),
            manifest.missing_required.end());
        return manifest;
    }

} // namespace llaminar2
