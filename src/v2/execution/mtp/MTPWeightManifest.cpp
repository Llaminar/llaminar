/**
 * @file MTPWeightManifest.cpp
 * @brief Canonical learned-predictor discovery and fail-fast MTP admission.
 *
 * Discovery reads only metadata and tensor names. Auto planning and concrete
 * graph construction use the same mandatory-weight contract before estimating
 * or allocating predictor state; no synthetic predictor or hidden downgrade is
 * permitted when a plain GGUF has no learned NextN/MTP weights.
 */
#include "MTPWeightManifest.h"
#include "MTPLearnedBlockCount.h"

#include "../../loaders/IModelLoader.h"
#include "../../loaders/ModelLoader.h"
#include "planning/ModelMemoryProfile.h"

#include <algorithm>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace llaminar2
{
    namespace
    {
        constexpr int kSupportedMTPDepth = 1;

        /** @return Whether every required source role is present in this exact directory. */
        template<class HasTensor>
        bool hasAll(const HasTensor &has_tensor, const std::vector<std::string> &names)
        {
            return std::all_of(names.begin(), names.end(),
                               [&](const std::string &name)
                               {
                                   return has_tensor(name);
                               });
        }

        /** @return Missing source names, preserving role order for useful diagnostics. */
        template<class HasTensor>
        std::vector<std::string> missingFrom(const HasTensor &has_tensor, const std::vector<std::string> &names)
        {
            std::vector<std::string> missing;
            for (const auto &name : names)
            {
                if (!has_tensor(name))
                    missing.push_back(name);
            }
            return missing;
        }

        /** @return Dense NextN roles for one source block without consulting payloads. */
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

        /** @return MoE NextN roles, with routed and shared parents replacing the dense FFN. */
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

        /** @return Dedicated `mtp.layers` roles for one learned predictor block. */
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

        /** @return Unvalidated roles for exactly the declared learned-block interval. */
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

        /** @return Distinct starts for encodings whose raw block count includes or excludes NextN. */
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

        /**
         * @brief Apply one count algorithm to borrowed loader and GGUF views.
         * @param architecture Source metadata namespace.
         * @param block_count Raw inventory count or a known main-layer boundary.
         * @param integer_lookup Lookup of one unsigned source metadata scalar.
         * @param tensor_lookup Presence query over the same source directory.
         * @return Declared count, or the existing single-block directory inference.
         * @throws std::invalid_argument For unrepresentable declared geometry.
         *
         * The complete manifest validates availability afterward. In particular,
         * an FC name alone establishes accounting geometry, not permission to
         * synthesize missing attention/FFN weights or execute that predictor.
         */
        template <typename IntegerLookup, typename TensorLookup>
        int resolveLearnedBlockCount(
            const std::string &architecture, int block_count,
            const IntegerLookup &integer_lookup, const TensorLookup &tensor_lookup)
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
                const uint64_t value = integer_lookup(key);
                if (value > static_cast<uint64_t>(std::numeric_limits<int>::max()))
                    throw std::invalid_argument("MTP learned-block count exceeds runtime geometry: " + key);
                if (value > 0)
                    return static_cast<int>(value);
            }

            // Some exporters omit the optional count key. Use the exact FC
            // roles already owned by discovery, at both supported raw-count
            // conventions, instead of asking planners to infer another count.
            for (const int source_layer : nextNSourceLayerStartCandidates(block_count, 1))
                if (tensor_lookup(makeNextNDepth(0, source_layer).fc))
                    return 1;
            return tensor_lookup(makeGenericMTPDepth(0).fc) ? 1 : 0;
        }

        /** @return Empty, unavailable discovery with the precise admission diagnostic. */
        MTPWeightManifest unavailable(std::string diagnostic)
        {
            MTPWeightManifest manifest;
            manifest.diagnostic = std::move(diagnostic);
            return manifest;
        }
    } // namespace

    int mtpLearnedBlockCount(
        const IModelLoader &loader, const std::string &architecture, int block_count)
    {
        return resolveLearnedBlockCount(architecture, block_count,
            [&](const std::string &key) { return loader.getUInt64(key, 0); },
            [&](const std::string &name) { return loader.hasTensor(name); });
    }

    int mtpLearnedBlockCount(const GGUFModel &model)
    {
        if (model.block_count > static_cast<uint64_t>(std::numeric_limits<int>::max()))
            throw std::invalid_argument("MTP source block count exceeds runtime geometry");
        return resolveLearnedBlockCount(model.architecture, static_cast<int>(model.block_count),
            [&](const std::string &key) {
                const auto entry = model.metadata.find(key);
                return entry == model.metadata.end() ? uint64_t{0} : entry->second.asUInt64();
            },
            [&](const std::string &name) { return model.findTensor(name) != nullptr; });
    }

    MTPWeightManifest requireMTPWeightManifest(
        const IModelLoader &loader,
        const std::string &architecture,
        int base_layer_count)
    {
        auto manifest = discoverMTPWeightManifest(
            loader, architecture, base_layer_count, /*explicit_mtp=*/true);
        if (!manifest.available)
        {
            std::string diagnostic = manifest.diagnostic;
            if (!manifest.missing_required.empty())
                diagnostic += "; missing required tensor: " + manifest.missing_required.front();
            throw std::invalid_argument(
                diagnostic + "; supply a GGUF with complete learned MTP/NextN weights "
                             "or explicitly disable retained MTP capacity");
        }
        return manifest;
    }

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

    std::vector<std::string> MTPWeightManifest::participantReplicaNames(
        MTPRoutedExpertWeightAuthority routed_authority) const
    {
        std::vector<std::string> names = requiredNames();
        if (routed_authority !=
            MTPRoutedExpertWeightAuthority::ExpertOverlay)
        {
            return names;
        }

        /*
         * Compare against manifest fields rather than naming heuristics.  A
         * future architecture may spell its expert parents differently, but
         * discovery must already populate these typed roles before the model
         * can advertise an available MTP block.
         */
        std::vector<std::string> overlay_owned;
        overlay_owned.reserve(depths.size() * 3u);
        for (const auto &depth : depths)
        {
            for (const std::string *name : {
                     &depth.moe_gate_exps,
                     &depth.moe_up_exps,
                     &depth.moe_down_exps,
                 })
            {
                if (!name->empty())
                    overlay_owned.push_back(*name);
            }
        }

        names.erase(
            std::remove_if(
                names.begin(),
                names.end(),
                [&](const std::string &name)
                {
                    return std::find(
                               overlay_owned.begin(),
                               overlay_owned.end(),
                               name) != overlay_owned.end();
                }),
            names.end());
        return names;
    }

    int mainLayerCountExcludingMTP(
        const IModelLoader &loader,
        const std::string &architecture,
        int raw_layer_count)
    {
        if (raw_layer_count <= 0)
            return raw_layer_count;

        const int depth = mtpLearnedBlockCount(loader, architecture, raw_layer_count);
        if (depth <= 0 || raw_layer_count < depth)
            return raw_layer_count;

        const int sidecar_source_layer = raw_layer_count - depth;
        if (loader.hasTensor(makeNextNDepth(0, sidecar_source_layer).fc))
        {
            return sidecar_source_layer;
        }

        return raw_layer_count;
    }

    /** @brief One discovery algorithm for source loaders and published tensor directories. */
    template<class HasTensor>
    static MTPWeightManifest discoverMTPManifestForDirectory(
        int depth,
        int base_layer_count,
        bool explicit_mtp,
        const HasTensor &has_tensor)
    {
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
                auto nextn_missing = missingFrom(has_tensor, nextn_required);
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
        if (hasAll(has_tensor, generic_required))
        {
            generic_manifest.available = true;
            generic_manifest.diagnostic = "using mtp.layers MTP layout";
            return generic_manifest;
        }

        auto manifest = unavailable("MTP tensors are incomplete");
        manifest.depth = depth;
        manifest.missing_required = std::move(best_nextn_missing);
        auto generic_missing = missingFrom(has_tensor, generic_required);
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

    MTPWeightManifest discoverMTPWeightManifest(
        const IModelLoader &loader, const std::string &architecture,
        int base_layer_count, bool explicit_mtp)
    {
        return discoverMTPManifestForDirectory(
            mtpLearnedBlockCount(loader, architecture, base_layer_count),
            base_layer_count, explicit_mtp,
            [&](const std::string &name) { return loader.hasTensor(name); });
    }

    MTPWeightManifest discoverMTPWeightManifest(const ModelMemoryProfile &profile,
                                             bool explicit_mtp)
    {
        return discoverMTPManifestForDirectory(profile.mtp_layer_count,
            profile.n_layers, explicit_mtp, [&](const std::string &name) {
                return std::any_of(profile.tensors.begin(), profile.tensors.end(),
                    [&](const auto &tensor) { return tensor.name == name; });
            });
    }

} // namespace llaminar2
