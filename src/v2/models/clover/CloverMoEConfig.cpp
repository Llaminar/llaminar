/**
 * @file CloverMoEConfig.cpp
 * @brief Authenticate Clover's causal GGUF contract before graph or memory admission.
 *
 * Arithmetic/layout references are llama.cpp PR #31306, revision
 * ed26f2f7621306a8c393dd5193c18b3c3cea7940, and the official Preview GGUF
 * revision 32e22a02b885e173177b7ce7661ad13448b4f89e. GGUF already contains
 * final RMSNorm gains, negative exponentiated SSM A, and tiled GDN value heads;
 * these must not be transformed again. Block diffusion/coreset metadata does
 * not change the explicitly causal graph implemented here.
 */
#include "CloverMoEConfig.h"
#include "../GraphTypes.h"
#include "../../loaders/ModelLoader.h"
#include "../../planning/ModelMemoryProfile.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>

namespace llaminar2
{
    namespace
    {
        /** @brief Fail source admission with the exact offending key or tensor. */
        [[noreturn]] void reject(const std::string &name, const std::string &reason)
        {
            throw std::invalid_argument("clover_moe: " + name + ": " + reason);
        }

        /** @brief Read a required metadata value without a loader-provided default. */
        const GGUFValue &required(const GGUFModel &model, const std::string &suffix)
        {
            const std::string key = "clover_moe." + suffix;
            const auto found = model.metadata.find(key);
            if (found == model.metadata.end()) reject(key, "required metadata is missing");
            return found->second;
        }

        /** @brief Decode a checked nonnegative integer representable by graph geometry. */
        int integer(const GGUFModel &model, const std::string &suffix, bool positive = true)
        {
            uint64_t value;
            try { value = required(model, suffix).asUInt64Checked(); }
            catch (const std::invalid_argument &error) { reject(suffix, error.what()); }
            if (value > static_cast<uint64_t>(std::numeric_limits<int>::max()) || (positive && value == 0))
                reject(suffix, "requires a representable " + std::string(positive ? "positive" : "nonnegative") + " integer");
            return static_cast<int>(value);
        }

        /** @brief Require an exact finite positive floating parameter. */
        float positiveFloat(const GGUFModel &model, const std::string &suffix)
        {
            const auto &raw = required(model, suffix);
            if (raw.type != GGUFValueType::FLOAT32 || raw.data.size() != sizeof(float))
                reject(suffix, "requires a FLOAT32 scalar");
            const float value = raw.asFloat32();
            if (!std::isfinite(value) || value <= 0) reject(suffix, "requires a finite positive value");
            return value;
        }

        /** @brief Check products before converting physical dimensions to graph integers. */
        int product(int a, int b, const std::string &name)
        {
            const int64_t value = int64_t{a} * b;
            if (value <= 0 || value > std::numeric_limits<int>::max()) reject(name, "dimension product overflows");
            return static_cast<int>(value);
        }
    }

    bool CloverMoEConfig::isDiffusionOnlyTensor(std::string_view name) noexcept
    {
        if (!name.starts_with("blk.")) return false;
        const auto end = name.find('.', 4);
        if (end == std::string_view::npos || end == 4) return false;
        for (size_t i = 4; i < end; ++i)
            if (name[i] < '0' || name[i] > '9') return false;
        const auto suffix = name.substr(end + 1);
        return suffix == "ssm_conv1d.weight_right" || suffix == "ssm_bd_history.bias" ||
               suffix == "ssm_bd_history.weight" || suffix == "ssm_bd_logit_scale.weight";
    }

    CloverMoEConfig::CloverMoEConfig(const GGUFModel &model)
    {
        if (model.architecture != "clover_moe") reject("general.architecture", "requires clover_moe");
        auto &g = geometry_;
        g.layers = integer(model, "block_count");
        g.context_length = integer(model, "context_length");
        g.model_width = integer(model, "embedding_length");
        g.query_heads = integer(model, "attention.head_count");
        g.kv_heads = integer(model, "attention.head_count_kv");
        g.head_dimension = integer(model, "attention.key_length");
        if (integer(model, "attention.value_length") != g.head_dimension || g.query_heads % g.kv_heads != 0)
            reject("attention", "requires equal K/V head widths and complete query/KV groups");
        g.rotary_dimension = integer(model, "rope.dimension_count");
        if (g.rotary_dimension > g.head_dimension || g.rotary_dimension % 2 != 0)
            reject("rope.dimension_count", "requires an even width no greater than the attention head");
        g.rotary_base = positiveFloat(model, "rope.freq_base");
        g.norm_epsilon = positiveFloat(model, "attention.layer_norm_rms_epsilon");
        g.attention_interval = integer(model, "full_attention_interval");
        if (g.attention_interval < 2 || g.layers < g.attention_interval)
            reject("full_attention_interval", "requires both causal GDN and full-attention blocks");
        if (model.hasMetadata("clover_moe.nextn_predict_layers") && integer(model, "nextn_predict_layers", false) != 0)
            reject("nextn_predict_layers", "Clover AR does not implement learned nextn prediction or diffusion drafting");

        g.key_heads = integer(model, "ssm.group_count");
        g.value_heads = integer(model, "ssm.time_step_rank");
        g.state_dimension = integer(model, "ssm.state_size");
        g.value_width = integer(model, "ssm.inner_size");
        g.convolution_kernel = integer(model, "ssm.conv_kernel");
        if (g.value_heads % g.key_heads != 0 || g.value_width != product(g.state_dimension, g.value_heads, "ssm.inner_size"))
            reject("ssm", "requires complete tiled value-head groups and inner_size = state_size * time_step_rank");
        // The installed shared native GDN kernels qualify the Preview's width-four convolution.
        if (g.convolution_kernel != 4)
            reject("ssm.conv_kernel", "the implemented Clover family requires a width-four causal convolution");
        const int qk_width = product(product(g.state_dimension, g.key_heads, "ssm.group_count"), 2, "attn_qkv.weight");
        if (g.value_width > std::numeric_limits<int>::max() - qk_width)
            reject("attn_qkv.weight", "combined QKV width overflows");
        const int qkv_width = qk_width + g.value_width;
        const int attention_width = product(g.query_heads, g.head_dimension, "attention.head_count");
        const int attention_kv_width = product(g.kv_heads, g.head_dimension, "attention.head_count_kv");
        const int query_gate_width = product(attention_width, 2, "attn_q.weight");

        g.experts = integer(model, "expert_count");
        g.selected_experts = integer(model, "expert_used_count");
        g.expert_width = integer(model, "expert_feed_forward_length");
        if (g.selected_experts > g.experts) reject("expert_used_count", "top-k exceeds the routed expert count");
        // The original official GGUF omits expert_shared_count but explicitly
        // declares the single shared expert's width and complete tensor family.
        // This is that exporter contract, not an execution fallback on load failure.
        const int shared_count = model.hasMetadata("clover_moe.expert_shared_count")
            ? integer(model, "expert_shared_count", false) : 1;
        if (shared_count > 1) reject("expert_shared_count", "only the single gated shared expert is implemented");
        g.shared_width = shared_count ? integer(model, "expert_shared_feed_forward_length") : 0;
        if (!shared_count && model.hasMetadata("clover_moe.expert_shared_feed_forward_length") &&
            integer(model, "expert_shared_feed_forward_length", false) != 0)
            reject("expert_shared_feed_forward_length", "nonzero width contradicts zero shared experts");

        if (model.hasMetadata("clover_moe.attention.recurrent_layers"))
        {
            const auto &mask = required(model, "attention.recurrent_layers");
            if (mask.type != GGUFValueType::ARRAY || mask.array_element_type != GGUFValueType::BOOL ||
                mask.array_length != static_cast<uint64_t>(g.layers) || mask.data.size() != static_cast<size_t>(g.layers))
                reject("attention.recurrent_layers", "requires one BOOL per main block");
            for (int layer = 0; layer < g.layers; ++layer)
                if (mask.data[layer] != ((layer + 1) % g.attention_interval != 0 ? 1 : 0))
                    reject("attention.recurrent_layers", "nonperiodic or contradictory layer maps are not implemented");
        }

        const auto *embedding = model.findTensor("token_embd.weight");
        if (!embedding || embedding->dimensions.size() != 2 || embedding->dimensions[0] == 0 ||
            embedding->dimensions[0] > static_cast<uint64_t>(std::numeric_limits<int>::max()))
            reject("token_embd.weight", "requires a representable nonempty vocabulary matrix");
        g.vocabulary = static_cast<int>(embedding->dimensions[0]);
        if (model.hasMetadata("clover_moe.vocab_size") && integer(model, "vocab_size") != g.vocabulary)
            reject("vocab_size", "disagrees with embedding rows");

        std::set<std::string> consumed;
        const auto tensor = [&](const std::string &name, std::initializer_list<uint64_t> shape)
        {
            const auto *source = model.findTensor(name);
            if (!source || source->dimensions != std::vector<uint64_t>(shape))
                reject(name, "missing tensor or shape incompatible with declared causal geometry");
            consumed.insert(name);
        };
        const uint64_t h = g.model_width, v = g.vocabulary, e = g.experts, f = g.expert_width;
        const uint64_t shared = g.shared_width, state = g.state_dimension, heads = g.value_heads;
        tensor("token_embd.weight", {v, h});
        tensor("output_norm.weight", {h});
        if (model.findTensor("output.weight")) tensor("output.weight", {v, h});
        const bool has_diffusion = std::any_of(model.tensors.begin(), model.tensors.end(),
            [](const auto &source) { return isDiffusionOnlyTensor(source.name); });
        for (int layer = 0; layer < g.layers; ++layer)
        {
            const std::string prefix = "blk." + std::to_string(layer) + ".";
            tensor(prefix + "attn_norm.weight", {h});
            tensor(prefix + "post_attention_norm.weight", {h});
            tensor(prefix + "ffn_gate_inp.weight", {e, h});
            tensor(prefix + "ffn_gate_exps.weight", {h, f, e});
            tensor(prefix + "ffn_up_exps.weight", {h, f, e});
            tensor(prefix + "ffn_down_exps.weight", {f, h, e});
            if (shared_count)
            {
                tensor(prefix + "ffn_gate_shexp.weight", {shared, h});
                tensor(prefix + "ffn_up_shexp.weight", {shared, h});
                tensor(prefix + "ffn_down_shexp.weight", {h, shared});
                tensor(prefix + "ffn_gate_inp_shexp.weight", {h});
            }
            if ((layer + 1) % g.attention_interval == 0)
            {
                tensor(prefix + "attn_q.weight", {static_cast<uint64_t>(query_gate_width), h});
                tensor(prefix + "attn_k.weight", {static_cast<uint64_t>(attention_kv_width), h});
                tensor(prefix + "attn_v.weight", {static_cast<uint64_t>(attention_kv_width), h});
                tensor(prefix + "attn_output.weight", {h, static_cast<uint64_t>(attention_width)});
                tensor(prefix + "attn_q_norm.weight", {static_cast<uint64_t>(g.head_dimension)});
                tensor(prefix + "attn_k_norm.weight", {static_cast<uint64_t>(g.head_dimension)});
            }
            else
            {
                tensor(prefix + "attn_qkv.weight", {static_cast<uint64_t>(qkv_width), h});
                tensor(prefix + "attn_gate.weight", {static_cast<uint64_t>(g.value_width), h});
                tensor(prefix + "ssm_alpha.weight", {heads, h});
                tensor(prefix + "ssm_beta.weight", {heads, h});
                tensor(prefix + "ssm_a", {heads});
                tensor(prefix + "ssm_dt.bias", {heads});
                tensor(prefix + "ssm_conv1d.weight", {static_cast<uint64_t>(qkv_width), 4});
                tensor(prefix + "ssm_norm.weight", {state});
                tensor(prefix + "ssm_out.weight", {h, static_cast<uint64_t>(g.value_width)});
                if (has_diffusion)
                {
                    tensor(prefix + "ssm_conv1d.weight_right", {static_cast<uint64_t>(qkv_width), 3});
                    tensor(prefix + "ssm_bd_history.bias", {heads});
                    tensor(prefix + "ssm_bd_history.weight", {heads});
                    tensor(prefix + "ssm_bd_logit_scale.weight", {heads});
                }
            }
        }
        for (const auto &source : model.tensors)
            if (!consumed.contains(source.name)) reject(source.name, "tensor has no declared Clover AR or Preview diffusion role");
        if (consumed.size() != model.tensors.size()) reject("tensors", "duplicate source tensor names");
    }

    void CloverMoEConfig::populate(GraphConfig &config) const
    {
        const auto &g = geometry_;
        if (config.pp_layer_offset < 0 || config.pp_layer_offset >= g.layers || config.n_layers < 0)
            reject("graph", "invalid main-decoder PP interval");
        if (config.n_layers == 0) config.n_layers = g.layers - config.pp_layer_offset;
        if (config.n_layers > g.layers - config.pp_layer_offset) reject("graph", "PP interval exceeds main blocks");
        config.total_n_layers = g.layers;
        config.d_model = g.model_width;
        config.vocab_size = g.vocabulary;
        config.d_ff = std::max(g.expert_width, g.shared_width);
        config.n_heads = g.query_heads;
        config.n_kv_heads = g.kv_heads;
        config.head_dim = g.head_dimension;
        config.rms_norm_eps = g.norm_epsilon;
        config.rope_theta = g.rotary_base;
        config.partial_rotary_factor = static_cast<float>(g.rotary_dimension) / g.head_dimension;
        config.has_attention_output_gate = true;
        config.kv_cache_scale_k = 1024.0f;
        config.kv_cache_scale_v = 256.0f;
        config.gdn.conv_kernel_size = g.convolution_kernel;
        config.gdn.state_size = g.state_dimension;
        config.gdn.inner_size = g.value_width;
        config.gdn.group_count = g.key_heads;
        config.gdn.time_step_rank = g.value_heads;
        config.gdn.full_attention_interval = g.attention_interval;
        config.layer_types.resize(g.layers);
        config.layer_head_dim.resize(g.layers);
        for (int layer = 0; layer < g.layers; ++layer)
        {
            const bool full_attention = (layer + 1) % g.attention_interval == 0;
            config.layer_types[layer] = full_attention ? "full_attention" : "gdn";
            config.layer_head_dim[layer] = full_attention ? g.head_dimension : g.state_dimension;
        }
        config.moe.num_experts = g.experts;
        config.moe.top_k = g.selected_experts;
        config.moe.intermediate_size = g.expert_width;
        config.moe.shared_intermediate_size = g.shared_width;
        config.moe.has_shared_expert = g.shared_width > 0;
        config.moe.shared_expert_gate = g.shared_width > 0;
        config.moe.norm_topk_prob = true;
        config.moe.decode_histogram_token_boundary_layer = config.n_layers - 1;
    }

    void CloverMoEConfig::populate(ModelMemoryProfile &profile) const
    {
        const auto &g = geometry_;
        profile.n_layers = g.layers;
        profile.d_model = g.model_width;
        profile.d_ff = std::max(g.expert_width, g.shared_width);
        profile.vocab_size = g.vocabulary;
        profile.max_seq_len = g.context_length;
        profile.n_heads = g.query_heads;
        profile.n_kv_heads = g.kv_heads;
        profile.head_dim = g.head_dimension;
        profile.expert_count = g.experts;
        profile.expert_used_count = g.selected_experts;
        profile.expert_feed_forward_length = g.expert_width;
        profile.expert_shared_feed_forward_length = g.shared_width;
        profile.mtp_layer_count = 0;
        profile.full_attention_interval = g.attention_interval;
        profile.gdn_conv_kernel_size = g.convolution_kernel;
        profile.gdn_state_size = g.state_dimension;
        profile.gdn_inner_size = g.value_width;
        profile.gdn_group_count = g.key_heads;
        profile.gdn_time_step_rank = g.value_heads;
    }
}
